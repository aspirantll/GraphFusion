//
// Created by liulei on 2020/6/5.
//

#ifndef GraphFusion_FEATURE_MATCHER_H
#define GraphFusion_FEATURE_MATCHER_H

#include <vector>

#include "base_feature.h"
#include "feature2d.h"
#include "../datastructure/frame_types.h"
#include "../datastructure/point_types.h"
#include "orb_matcher.h"

using namespace std;

namespace rtf {

    class SIFTMatchingConfig {
    public:
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

        // Confidence threshold for geometric verification.
        double confidence = 0.999;

        // search radius for projection
        float search_radius = 15;
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

        FeatureMatches matchKeyPointsWithProjection(SIFTFeaturePoints& k1, SIFTFeaturePoints& k2, Transform T);
    };

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

    class ORBMatchingConfig {
    public:
        string vocTxtPath = "/home/liulei/codes/ORB_SLAM2-master/Vocabulary/ORBvoc.txt";
        float nnratio = 0.6;
        bool checkOri = true;
        int search_radius = 3;
    };

    class ORBFeatureMatcher{
    protected:
        ORBMatchingConfig config;
        shared_ptr<ORBVocabularyHelper> bowHelper;
        shared_ptr<ORBMatcher> matcher;
    public:
        ORBFeatureMatcher();

        ORBFeatureMatcher(ORBMatchingConfig config);

        FeatureMatches matchKeyPointsPair(ORBFeaturePoints& k1, ORBFeaturePoints& k2);

        FeatureMatches matchKeyPointsWithProjection(ORBFeaturePoints& k1, ORBFeaturePoints& k2, Transform T);

    };

}
#endif //GraphFusion_FEATURE_MATCHER_H
