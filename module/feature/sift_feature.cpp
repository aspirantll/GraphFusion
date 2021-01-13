//
// Created by liulei on 2020/6/5.
//

#include <memory>
#include <utility>
#include<Eigen/Core>
#include <opencv2/core/eigen.hpp>

#include "feature2d.h"
#include "feature_matcher.h"
#include "../tool/math.h"
#include "../processor/image_cuda.cuh"
#include <FreeImage.h>
#include <GL/gl.h>


namespace rtf {
    // Mutexes that ensure that only one thread extracts/matches on the same GPU
    // at the same time, since SiftGPU internally uses static variables.
    static std::map<int, std::unique_ptr<std::mutex>> siftExtractionMutexes;
    static std::map<int, std::unique_ptr<std::mutex>> siftMatchingMutexes;

    SIFTFeatureExtractor::SIFTFeatureExtractor(SIFTExtractionConfig config) : config(std::move(config)) {
        this->siftGPU = make_shared<SiftGPU>(SiftGPU());
        initializeSiftGPU();
    }

    SIFTFeatureExtractor::SIFTFeatureExtractor() {
        this->config = SIFTExtractionConfig();
        this->siftGPU = make_shared<SiftGPU>(SiftGPU());
        initializeSiftGPU();
    }

    void SIFTFeatureExtractor::initializeSiftGPU() {
        // SiftGPU uses many global static state variables and the initialization must
        // be thread-safe in order to work correctly. This is enforced here.
        static std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);

        std::vector<int> gpuIndices = StringUtil::toIntVec(config.gpu_index);
        CHECK_EQ(gpuIndices.size(), 1) << "SiftGPU can only run on one GPU";

        std::vector<std::string> siftGpuArgs;

        siftGpuArgs.emplace_back("./sift_gpu");

#ifdef CUDA_ENABLED
        // Use CUDA version by default if darkness adaptivity is disabled.
        if (!config.darkness_adaptivity && gpuIndices[0] < 0) {
            gpuIndices[0] = 0;
        }

        if (gpuIndices[0] >= 0) {
            siftGpuArgs.emplace_back("-cuda");
            siftGpuArgs.push_back(std::to_string(gpuIndices[0]));
        }
#endif  // CUDA_ENABLED

        // Darkness adaptivity (hidden feature). Significantly improves
        // distribution of features. Only available in GLSL version.
        if (config.darkness_adaptivity) {
            if (gpuIndices[0] >= 0) {
                LOG(WARNING) << "darkness adaptivity is not available" << endl;
            }
            siftGpuArgs.emplace_back("-da");
        }

        // No verbose logging.
        siftGpuArgs.emplace_back("-v");
        siftGpuArgs.emplace_back("0");

        // Fixed maximum image dimension.
        siftGpuArgs.emplace_back("-maxd");
        siftGpuArgs.push_back(std::to_string(config.max_image_size));

        // Keep the highest level features.
        siftGpuArgs.emplace_back("-tc2");
        siftGpuArgs.push_back(std::to_string(config.max_num_features));

        // First octave level.
        siftGpuArgs.emplace_back("-fo");
        siftGpuArgs.push_back(std::to_string(config.first_octave));

        // Number of octave levels.
        siftGpuArgs.emplace_back("-d");
        siftGpuArgs.push_back(std::to_string(config.octave_resolution));

        // Peak threshold.
        siftGpuArgs.emplace_back("-t");
        siftGpuArgs.push_back(std::to_string(config.peak_threshold));

        // Edge threshold.
        siftGpuArgs.emplace_back("-e");
        siftGpuArgs.push_back(std::to_string(config.edge_threshold));

        if (config.upright) {
            // Fix the orientation to 0 for upright features.
            siftGpuArgs.emplace_back("-ofix");
            // Maximum number of orientations.
            siftGpuArgs.emplace_back("-mo");
            siftGpuArgs.emplace_back("1");
        } else {
            // Maximum number of orientations.
            siftGpuArgs.emplace_back("-mo");
            siftGpuArgs.push_back(std::to_string(config.max_num_orientations));
        }

        std::vector<const char*> siftGpuArgsCstr;
        siftGpuArgsCstr.reserve(siftGpuArgs.size());
        for (const auto& arg : siftGpuArgs) {
            siftGpuArgsCstr.push_back(arg.c_str());
        }

        siftGPU->ParseParam(siftGpuArgsCstr.size(), siftGpuArgsCstr.data());

        siftGPU->gpu_index = gpuIndices[0];
        if (siftExtractionMutexes.count(gpuIndices[0]) == 0) {
            siftExtractionMutexes.emplace(
                    gpuIndices[0], std::make_unique<std::mutex>());
        }

        auto supportedType = siftGPU->VerifyContextGL();

        LOG_ASSERT( supportedType == SiftGPU::SIFTGPU_FULL_SUPPORTED) << "ERROR: SiftGPU not fully supported." << std::endl;
    }

    // feature extractor begin
    void SIFTFeatureExtractor::extractFeatures(shared_ptr<FrameRGBD> frameRGBD, SIFTFeaturePoints& siftFeaturePoints) {
        std::unique_lock<std::mutex> lock(
                *siftExtractionMutexes[siftGPU->gpu_index]);

        shared_ptr<cv::Mat> rgbImage = frameRGBD->getRGBImage();
        LOG_ASSERT(rgbImage->channels()==4) << "the function named FrameConverter::convertImageType should be invoked before now";
        siftFeaturePoints.setCamera(frameRGBD->getCamera());
        siftFeaturePoints.setFIndex(frameRGBD->getFrameIndex());

        // Note, that this produces slightly different results than using SiftGPU
        // directly for RGB->GRAY conversion, since it uses different weights.
        std::vector<uint8_t> bitmap_raw_bits;
        convertToRawBits(rgbImage, bitmap_raw_bits);
        const int code =
                siftGPU->RunSIFT(rgbImage->cols, rgbImage->rows,
                                  bitmap_raw_bits.data(), GL_LUMINANCE, GL_UNSIGNED_BYTE);

        const int kSuccessCode = 1;
        if (code != kSuccessCode) {
            LOG(ERROR) << "ERROR: failed to extract the sift feature" << endl;
        }

        const size_t num_features = static_cast<size_t>(siftGPU->GetFeatureNum());
        if(num_features==0) return;

        std::vector<SiftKeypoint> keypoints(num_features);

        // Eigen's default is ColMajor, but SiftGPU stores result as RowMajor.
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
                descriptors_float(num_features, 128);

        // Download the extracted keypoints and descriptors.
        siftGPU->GetFeatureVector(keypoints.data(), descriptors_float.data());

        // remove distortion
        cv::Mat distCoef = frameRGBD->getCamera()->getDistCoef();
        cv::Mat mat(num_features,2,CV_32F);

        FeatureKeypoints& featurePoints = siftFeaturePoints.getKeyPoints();
        vector<int> selectedFeatures;
        bool needUndistorted = distCoef.at<float>(0)!=0;

        if(needUndistorted) {
            cv::Mat K;
            cv::eigen2cv(frameRGBD->getCamera()->getK(), K);
            // Fill matrix with points
            for(int i=0; i<num_features; i++)
            {
                mat.at<float>(i,0)=keypoints[i].x;
                mat.at<float>(i,1)=keypoints[i].y;
            }

            // Undistort points
            mat=mat.reshape(2);
            cv::undistortPoints(mat,mat,K,distCoef,cv::Mat(),K);
            mat=mat.reshape(1);
        }


        for (size_t i = 0; i < num_features; ++i) {
            Point2D pixel(keypoints[i].x, keypoints[i].y);
            float depth = frameRGBD->getDepth(pixel);
            bool validDepth = frameRGBD->inDepthMask(pixel);
            if(validDepth) {
                shared_ptr<SIFTFeatureKeypoint> kp;
                if(needUndistorted) {
                    kp = make_shared<SIFTFeatureKeypoint>(mat.at<float>(i, 0), mat.at<float>(i, 1), depth, keypoints[i].s, keypoints[i].o);
                }else {
                    kp = make_shared<SIFTFeatureKeypoint>(keypoints[i].x, keypoints[i].y, depth, keypoints[i].s, keypoints[i].o);
                }
                kp->setIndex(featurePoints.size());
                featurePoints.emplace_back(kp);
                selectedFeatures.emplace_back(i);
            }
        }

        // Save and normalize the descriptors.
        if (config.normalization == SIFTExtractionConfig::Normalization::L2) {
            descriptors_float = descriptors_float.rowwise().normalized();
        } else if (config.normalization ==
                   SIFTExtractionConfig::Normalization::L1_ROOT) {
            Eigen::MatrixXf descriptors_normalized(descriptors_float.rows(),
                                                   descriptors_float.cols());
            for (Eigen::MatrixXf::Index r = 0; r < descriptors_float.rows(); ++r) {
                const float norm = descriptors_float.row(r).lpNorm<1>();
                descriptors_normalized.row(r) = descriptors_float.row(r) / norm;
                descriptors_normalized.row(r) =
                        descriptors_normalized.row(r).array().sqrt();
            }
            descriptors_float = descriptors_normalized;
        } else {
            LOG(FATAL) << "Normalization type not supported";
        }


        SIFTFeatureDescriptors& descriptors = siftFeaturePoints.getDescriptors();
        descriptors.resize(selectedFeatures.size(),
                descriptors_float.cols());
        for (int ind = 0; ind < selectedFeatures.size(); ++ind) {
            int r = selectedFeatures[ind];
            for (Eigen::MatrixXf::Index c = 0; c < descriptors_float.cols(); ++c) {
                const float scaled_value = std::round(512.0f * descriptors_float(r, c));
                descriptors(ind, c) =
                        TruncateCast<float, uint8_t>(scaled_value);
            }
        }
        siftFeaturePoints.assignFeaturesToGrid();
    }
    // feature extractor end

    const int SIFTFeatureMatcher::HISTO_LENGTH = 30;

    SIFTFeatureMatcher::SIFTFeatureMatcher(float max_ratio) {
        config.max_ratio = max_ratio;
        initializeSiftMatchGPU();
    }


    void SIFTFeatureMatcher::initializeSiftMatchGPU() {
        if(siftMatchGPU) {
            return;
        }

        // SiftGPU uses many global static state variables and the initialization must
        // be thread-safe in order to work correctly. This is enforced here.
        static std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);

        std::vector<int> gpuIndices = StringUtil::toIntVec(config.gpu_index);
        CHECK_EQ(gpuIndices.size(), 1) << "SiftGPU can only run on one GPU";

        siftMatchGPU = make_shared<SiftMatchGPU>(SiftMatchGPU(config.max_num_matches));

#ifdef CUDA_ENABLED
        if (gpuIndices[0] >= 0) {
            siftMatchGPU->SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA_DEVICE0 +
                                        gpuIndices[0]);
        } else {
            siftMatchGPU->SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA);
        }
#else   // CUDA_ENABLED
        siftMatchGPU->SetLanguage(SiftMatchGPU::SIFTMATCH_GLSL);
#endif  // CUDA_ENABLED

        if (siftMatchGPU->VerifyContextGL() == 0) {
            return ;
        }

        if (!siftMatchGPU->Allocate(config.max_num_matches,
                                    config.cross_check)) {
            std::cout << "ERROR: Not enough GPU memory to match "<< config.max_num_matches <<" features. "
                                                                                             "Reduce the maximum number of matches."
                      << std::endl;
            return ;
        }

#ifndef CUDA_ENABLED
        if (siftMatchGPU->GetMaxSift() < config.max_num_matches) {
    std::cout << "WARNING: OpenGL version of SiftGPU only supports a "
                 "maximum of" << siftMatchGPU->GetMaxSift() <<  "matches - consider changing to CUDA-based "
                     "feature matching to avoid this limitation."
              << std::endl;
  }
#endif  // CUDA_ENABLED

        siftMatchGPU->gpu_index = gpuIndices[0];
        if (siftMatchingMutexes.count(gpuIndices[0]) == 0) {
            siftMatchingMutexes.emplace(
                    gpuIndices[0], std::make_unique<std::mutex>());
        }

    }


    void computeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
    {
        int max1=0;
        int max2=0;
        int max3=0;

        for(int i=0; i<L; i++)
        {
            const int s = histo[i].size();
            if(s>max1)
            {
                max3=max2;
                max2=max1;
                max1=s;
                ind3=ind2;
                ind2=ind1;
                ind1=i;
            }
            else if(s>max2)
            {
                max3=max2;
                max2=s;
                ind3=ind2;
                ind2=i;
            }
            else if(s>max3)
            {
                max3=s;
                ind3=i;
            }
        }

        if(max2<0.1f*(float)max1)
        {
            ind2=-1;
            ind3=-1;
        }
        else if(max3<0.1f*(float)max1)
        {
            ind3=-1;
        }
    }


    FeatureMatches SIFTFeatureMatcher::matchKeyPointsPair(SIFTFeaturePoints& k1, SIFTFeaturePoints& k2) {

        std::unique_lock<std::mutex> lock(
                *siftMatchingMutexes[siftMatchGPU->gpu_index]);

        FeatureDescriptors<uint8_t>& descriptors1 = k1.getDescriptors();

        CHECK_EQ(descriptors1.cols(), 128);
        if (siftMatchGPU->GetMaxSift() < descriptors1.rows()) {
            std::cout << "WARNING: Clamping features from " << descriptors1.rows() << " to " << descriptors1.rows()
                      <<" - consider increasing the maximum number of matches."<< std::endl;
        }
        siftMatchGPU->SetDescriptors(0, descriptors1.rows(),
                                     descriptors1.data());

        FeatureDescriptors<uint8_t> &descriptors2 = k2.getDescriptors();
        CHECK_EQ(descriptors2.cols(), 128);
        if (siftMatchGPU->GetMaxSift() < descriptors2.rows()) {
            std::cout << "WARNING: Clamping features from " << descriptors2.rows() << " to " << descriptors2.rows()
                      <<" - consider increasing the maximum number of matches."<< std::endl;
        }
        siftMatchGPU->SetDescriptors(1, descriptors2.rows(),
                                     descriptors2.data());


        vector<FeatureMatch> matches;
        uint32_t match_buffer[config.max_num_matches][2];

        const int num_matches = siftMatchGPU->GetSiftMatch(
                config.max_num_matches,
                match_buffer,
                static_cast<float>(config.max_distance),
                static_cast<float>(config.max_ratio), config.cross_check);

        if (num_matches < 0) {
            std::cerr << "ERROR: Feature matching failed. This is probably caused by "
                         "insufficient GPU memory. Consider reducing the maximum "
                         "number of features and/or matches."
                      << std::endl;
            matches.clear();
        } else {
            for(int i=0; i<num_matches; i++) {
                matches.emplace_back(match_buffer[i][0], match_buffer[i][1]);
            }
        }

        // check orientation
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;
        for(int i=0; i<matches.size(); i++) {
            int ind1 = matches[i].getPX();
            int ind2 = matches[i].getPY();

            auto kp1 = (SIFTFeatureKeypoint*)k1.getKeyPoints()[ind1].get();
            auto kp2 = (SIFTFeatureKeypoint*)k2.getKeyPoints()[ind2].get();

            float rot = kp1->ComputeOrientation() - kp2->ComputeOrientation();
            if (rot < 0.0)
                rot += 360.0f;
            int bin = round(rot * factor);
            if (bin == HISTO_LENGTH)
                bin = 0;
            assert(bin >= 0 && bin < HISTO_LENGTH);
            rotHist[bin].push_back(i);
        }

        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        computeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        FeatureMatches featureMatches(k1, k2, vector<FeatureMatch>());
        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3) {
                for(int j=0; j<rotHist[i].size(); j++) {
                    featureMatches.getMatches().emplace_back(matches[rotHist[i][j]]);
                }
            }
        }

        return featureMatches;
    }

    FeatureMatches SIFTFeatureMatcher::matchKeyPointsWithProjection(SIFTFeaturePoints& k1, SIFTFeaturePoints& k2, Transform T) {
        vector<FeatureMatch> matches;

        for(int i=0; i<k2.size(); i++) {
            shared_ptr<FeatureKeypoint> kp = k2.getKeypoint(i);
            Eigen::Matrix<int , 1, -1, Eigen::RowMajor> d2 = k2.getDescriptors().row(i).cast<int>();
            Point3D rePixel = PointUtil::transformPixel(*kp, T, k2.getCamera());
            if(rePixel.x<k1.getMinX()||rePixel.x>=k1.getMaxX()
               ||rePixel.y<k1.getMinY()||rePixel.y>=k1.getMaxY()) continue;

            vector<int> indices = k1.getFeaturesInArea(rePixel.x, rePixel.y, config.search_radius);
            float bestDist = 0;
            float nxtDist = 0;
            int bestIdx = -1;

            for(int ind: indices)  {
                Eigen::Matrix<int, 1, -1, Eigen::RowMajor> d1 = k1.getDescriptors().row(ind).cast<int>();

                const float dist = d1.dot(d2); // dot product for distance

                if(dist>bestDist) {
                    nxtDist = bestDist;
                    bestDist=dist;
                    bestIdx=ind;
                }else {
                    nxtDist = max(nxtDist, dist);
                }
            }
            float dist =  acos(min(bestDist * 0.000003814697265625f, 1.0f));
            float nDist =  acos(min(nxtDist * 0.000003814697265625f, 1.0f));
            if(dist<config.max_distance&&dist < nDist * config.max_ratio)  {
                matches.emplace_back(bestIdx, i);
            }
        }

        // check orientation
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;
        for(int i=0; i<matches.size(); i++) {
            int ind1 = matches[i].getPX();
            int ind2 = matches[i].getPY();

            auto kp1 = (SIFTFeatureKeypoint*)k1.getKeyPoints()[ind1].get();
            auto kp2 = (SIFTFeatureKeypoint*)k2.getKeyPoints()[ind2].get();

            float rot = kp1->ComputeOrientation() - kp2->ComputeOrientation();
            if (rot < 0.0)
                rot += 360.0f;
            int bin = round(rot * factor);
            if (bin == HISTO_LENGTH)
                bin = 0;
            assert(bin >= 0 && bin < HISTO_LENGTH);
            rotHist[bin].push_back(i);
        }

        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        computeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        FeatureMatches featureMatches(k1, k2, vector<FeatureMatch>());
        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3) {
                for(int j=0; j<rotHist[i].size(); j++) {
                    featureMatches.getMatches().emplace_back(matches[rotHist[i][j]]);
                }
            }
        }

        return featureMatches;
    }
}
