//
// Created by liulei on 2020/6/5.
//

#ifndef GraphFusion_FEATURE_MATCHER_H
#define GraphFusion_FEATURE_MATCHER_H

#include <DBoW2/TemplatedVocabulary.h>
#include <DBoW2/FSIFT.h>
#include <vector>

#include "base_feature.h"
#include "feature2d.h"
#include "../datastructure/frame_types.h"
#include "../datastructure/point_types.h"

using namespace std;

typedef DBoW2::TemplatedVocabulary<DBoW2::FSIFT::TDescriptor, DBoW2::FSIFT>
        SIFTVocabulary;

namespace rtf {

    class SIFTMatchingConfig {
    public:
        // Number of threads for feature matching and geometric verification.
        int num_threads = -1;

        // Whether to use the GPU for feature matching.
        bool use_gpu = true;

        // Index of the GPU used for feature matching. For multi-GPU matching,
        // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
        std::string gpu_index = "-1";

        // Maximum distance ratio between first and second best match.
        double max_ratio = 0.8;

        // Maximum distance to best match.
        double max_distance = 0.7;

        // Whether to enable cross checking in matching.
        bool cross_check = true;

        // Maximum number of matches.
        int max_num_matches = 8192;

        // Maximum epipolar error in pixels for geometric verification.
        double max_error = 4.0;

        // Confidence threshold for geometric verification.
        double confidence = 0.999;

        // Minimum/maximum number of RANSAC iterations. Note that this option
        // overrules the min_inlier_ratio option.
        int min_num_trials = 100;
        int max_num_trials = 10000;

        // A priori assumed minimum inlier ratio, which determines the maximum
        // number of iterations.
        double min_inlier_ratio = 0.25;

        // Minimum number of inliers for an image pair to be considered as
        // geometrically verified.
        int min_num_inliers = 15;

        // Whether to attempt to estimate multiple geometric models per image pair.
        bool multiple_models = false;

        // Whether to perform guided matching, if geometric verification succeeds.
        bool guided_matching = false;
    };

    class SIFTFeatureMatcher {
    protected:
        SIFTMatchingConfig config;
        shared_ptr<SiftMatchGPU> siftMatchGPU;
        static const int HISTO_LENGTH;
    public:
        SIFTFeatureMatcher();

        SIFTFeatureMatcher(SIFTMatchingConfig config);

        void initializeSiftMatchGPU();

        FeatureMatches matchKeyPointsPair(SIFTFeaturePoints& k1, SIFTFeaturePoints& k2);
    };

    void toDescriptorVector(SIFTFeatureDescriptors & desc, vector<vector<float>>&converted);

    class DenseMatchingConfig {
    public:
        int neigh = 2; // search neigh range
        int windowRadius = 5;// NCC radius
        float sigmaSpatial = -1;
        float sigmaColor = 0.2f;
        float deltaNormalTh = 0.2;
        float nccTh = 0.5;
        int downSampleScale = 8;

        DenseMatchingConfig();

        DenseMatchingConfig(const GlobalConfig& config);
    };
    class DenseFeatureMatcher {
    protected:
        DenseMatchingConfig config;

        mutex m;
    public:
        DenseFeatureMatcher(const DenseMatchingConfig &config);

        DenseFeatureMatcher(const GlobalConfig &config);

        DenseFeatureMatches matchKeyPointsPair(shared_ptr<FrameRGBD> f1, shared_ptr<FrameRGBD> f2, Transform trans);
    };

}
#endif //GraphFusion_FEATURE_MATCHER_H
