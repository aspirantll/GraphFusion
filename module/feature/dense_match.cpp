//
// Created by liulei on 2020/10/12.
//

#include "feature_matcher.h"
#include "../processor/dense_match.cuh"

namespace rtf {

    DenseMatchingConfig::DenseMatchingConfig() {}

    DenseMatchingConfig::DenseMatchingConfig(const GlobalConfig& config) {
        neigh = config.neigh; // search neigh range
        windowRadius = config.windowRadius;// NCC radius
        sigmaSpatial = config.sigmaSpatial;
        sigmaColor = config.sigmaColor;
        deltaNormalTh = config.deltaNormalTh;
        nccTh = config.nccTh;
        downSampleScale = config.downSampleScale;
    }


    DenseFeatureMatcher::DenseFeatureMatcher(const DenseMatchingConfig &config) : config(config) {}

    DenseFeatureMatcher::DenseFeatureMatcher(const GlobalConfig &config) : config(config) {}


//    FeatureMatches downSampleDenseMatches()

    DenseFeatureMatches DenseFeatureMatcher::matchKeyPointsPair(shared_ptr<FrameRGBD> f1, shared_ptr<FrameRGBD> f2, Transform T) {
        std::unique_lock<std::mutex> lock(m);
        DenseMatchParams params;
        params.neigh = config.neigh; // search neigh range
        params.windowRadius = config.windowRadius;// NCC radius
        params.sigmaSpatial = config.sigmaSpatial;
        params.sigmaColor = config.sigmaColor;
        params.deltaNormalTh = config.deltaNormalTh;
        params.nccTh = config.nccTh;
        params.downSampleScale = config.downSampleScale;

        // upload data
        int height = f1->getCamera()->getHeight(), width = f2->getCamera()->getWidth();

        cv::cuda::GpuMat grayImg1(height, width, CV_32F), grayImg2(height, width, CV_32F);
        cv::cuda::GpuMat depthImg1(height, width, CV_32F), depthImg2(height, width, CV_32F);
        cv::cuda::GpuMat normalImg1(height, width, CV_32FC4), normalImg2(height, width, CV_32FC4);

        {
            cv::cuda::GpuMat rgbImg1(height, width, CV_8UC4), rgbImg2(height, width, CV_8UC4);
            rgbImg1.upload(*f1->getRGBImage());
            cv::cuda::cvtColor(rgbImg1, grayImg1, cv::COLOR_RGBA2GRAY);
            rgbImg2.upload(*f2->getRGBImage());
            cv::cuda::cvtColor(rgbImg2, grayImg2, cv::COLOR_RGBA2GRAY);
        }

        depthImg1.upload(*f1->getDepthImage());
        normalImg1.upload(*f1->getNormalImage());


        depthImg2.upload(*f2->getDepthImage());
        normalImg2.upload(*f2->getNormalImage());

        bindTextureParams(grayImg1.ptr<uchar>(), depthImg1.ptr<float>(), normalImg1.ptr<float4>(),
                          grayImg2.ptr<uchar>(), depthImg2.ptr<float>(), normalImg2.ptr<float4>(), width, height);

        shared_ptr<Camera> camera1 = f1->getCamera(), camera2 = f2->getCamera();
        Transform invT = T.inverse();
        // left match
        params.width = width;
        params.height = height;
        params.trans = MatrixConversion::toCUDA(invT);
        params.invK = MatrixConversion::toCUDA(camera1->getReverseK());
        params.K = MatrixConversion::toCUDA(camera2->getK());
        updateDenseMatchParams(params);

        CUDAMatrixl leftMatches(width, height);
        CUDAMatrixs matchScores(width, height);
        leftDenseMatch(leftMatches, matchScores);

        // right match
        params.trans = MatrixConversion::toCUDA(T);
        params.invK = MatrixConversion::toCUDA(camera2->getReverseK());
        params.K = MatrixConversion::toCUDA(camera1->getK());
        updateDenseMatchParams(params);

        CUDAMatrixl rightMatches(width, height);
        rightDenseMatch(rightMatches);

        // release texture
        grayImg1.release();
        depthImg1.release();
        normalImg1.release();
        grayImg2.release();
        depthImg2.release();
        normalImg2.release();

        // cross check
        CUDAMatrixc mask(width, height);
        crossCheck(leftMatches, rightMatches, matchScores, mask);

        vector<long> matchMatrix;
        leftMatches.download(matchMatrix);
        vector<uchar> maskMatrix;
        mask.download(maskMatrix);

        // collect match points
        vector<FeatureKeypoint> kp1, kp2;
        for(int i=0; i<width; i++) {
            for(int j=0; j<height; j++) {
                long index = mask.convert1DIndex(i,j);
                if(maskMatrix[index]) {
                    long mi, mj;
                    rightMatches.convert2DIndex(matchMatrix[index], &mi, &mj);

                    Point2D p1 = Point2D(i, j);
                    Point2D p2 = Point2D(mi, mj);

                    if(f1->inDepthMask(p1)&&f2->inDepthMask(p2)) {
                        kp1.emplace_back(i, j, f1->getDepth(p1));
                        kp2.emplace_back(mi, mj, f2->getDepth(p2));
                    }
                }
            }
        }
        return DenseFeatureMatches(f1->getCamera(), f2->getCamera(), kp1, kp2);
    }
}