//
// Created by liulei on 2020/6/6.
//

#ifndef GraphFusion_REGISTRATIONS_H
#define GraphFusion_REGISTRATIONS_H

#include <glog/logging.h>
#include<Eigen/StdVector>

#include "../../datastructure/context.h"
#include "../../datastructure/point_types.h"
#include "../../feature/feature_point.h"
#include "../../tool/geo_util.h"
#include "../../tool/image_util.h"
#include "../../tool/point_util.h"
#include "../../tool/timer.h"
#include "../solver/ransac.h"
#include "../../datastructure/view_graph.h"
#include "../../tool/map_reduce.h"
#include "../solver/matrix_conversion.h"
#include "../../tool/visual_index_hashing.h"
#include "../../feature/feature_matcher.h"

/**
 * the file for tracking method based on frame_to_frame
 */

namespace rtf {

    class RegReport {
    public:
        bool success = false;
        int iterations = 0;
        int pointsNum;
        double cost;
        Transform T;
        vector<FeatureKeypoint> kxs;
        vector<FeatureKeypoint> kys;


        double avgCost() {
            return cost / pointsNum;
        }


        void printReport() {
            cout << "-------------------------------------------------------------------------" << endl;
            cout << "success: " << success << endl;
            cout << "pointsNum: " << pointsNum << endl;
            cout << "iterations: " << iterations << endl;
            cout << "cost: " << cost << endl;
            cout << "avg cost: " << cost / pointsNum << endl;
            cout << "-------------------------------------------------------------------------" << endl;
        }

    };


    class BARegistration {
    private:
        float rmsThreshold;
        float relaxtion;
        float distTh;
        float minInlierRatio;
        float minInliers;

        float3x3 cudaK;
        CUDAMatrixs *cudaPoints;
        CUDAMatrixs *cudaPixels;
        CUDAMatrixc *cudaMask;
        CUDAMatrixc *cudaMaskBak;
        Summator *costSummator;
        Summator *hSummator;
        Summator *mSummator;
        Summator *bSummator;
        MatrixX points;
        MatrixX pixels;
        vector<FeatureKeypoint> *kxs;
        vector<FeatureKeypoint> *kys;

        RegReport bundleAdjustment(Rotation R, Translation t, int iterations);

    public:
        BARegistration(const GlobalConfig &config);

        RegReport multiViewBundleAdjustment(ViewGraph &viewGraph, const vector<int>& cc);

        RegReport bundleAdjustment(Transform initT, shared_ptr<Camera> cx, shared_ptr<Camera> cy,
                                   vector<FeatureKeypoint> &kxs, vector<FeatureKeypoint> &kys,
                                   bool robust = false);

        void alloc(shared_ptr<Camera> cx, shared_ptr<Camera> cy, vector<FeatureKeypoint> &kxs, vector<FeatureKeypoint> &kys);

        RegReport bundleAdjustment(Transform initT, bool robust = false, int iterations = 100);

        void bundleAdjustmentThread(Transform initT, bool robust, RegReport* report, cudaStream_t curStream);

        void free();
    };


    class RANSAC2DConfig : public RANSACConfig {
    public:
        // min matches num for every pair
        int kMinMatches = 25;
        // min inliers num for every pair
        int kMinInliers = 15;
        // rms threshold
        double rmsThreshold = 100;

        RANSAC2DConfig() {

        }

        RANSAC2DConfig(const GlobalConfig &globalConfig) {
            // max iterations
            numOfTrials = globalConfig.numOfTrials;
            // residual upper limit
            upperBoundResidual = globalConfig.upperBoundResidual;
            // don't abort probability
            confidence = globalConfig.confidence;
            // multiplier for dynamic iterations
            lambda = globalConfig.lambda;

            // down step factor
            stepFactor = globalConfig.stepFactor;
            // inliers ratio threshold
            irThreshold = globalConfig.irThreshold;
            // adaptive ending delta
            aeDelta = globalConfig.aeDelta;
            // allow difference for inlier ratio
            irDelta = globalConfig.irDelta;
            // min inlier ratio
            minInlierRatio = globalConfig.minInlierRatio;
            // min matches num for every pair
            kMinMatches = globalConfig.kMinMatches;
            // min inliers num for every pair
            kMinInliers = globalConfig.kMinInliers;
            // rms threshold
            rmsThreshold = globalConfig.rmsThreshold;
        }
    };

    /**
     * epipolar geometry
     */
    class RANSAC2DReport {
    public:
        bool success = false;
        double maxResidual = 0;
        int pointNum = 0;
        int iterations = 0;
        vector<int> inliers;
        vector<double> xDs;
        vector<double> yDs;
        vector<int> kps1, kps2;
        Transform T;


        void printReport() {
            cout << "-------------------------------------------------------------------------" << endl;
            cout << "success: " << success << endl;
            if (true) {
                cout << "iterations: " << iterations << endl;
                cout << "inliers: " << inliers.size() << endl;
                cout << "pointsNum: " << pointNum << endl;
                cout << "maxResidual: " << maxResidual << endl;
            }

            cout << "-------------------------------------------------------------------------" << endl;
        }
    };

    class PnPRegistration {
    private:
        RANSAC2DConfig config;
    public:
        PnPRegistration(const GlobalConfig &config);

        RANSAC2DReport registrationFunction(FeatureMatches &featureMatches);

        void registrationFunctionThread(FeatureMatches *featureMatches, RANSAC2DReport *report);
    };

    class LocalRegistration {
    protected:
        GlobalConfig globalConfig;

        SIFTFeatureMatcher* matcher;
        ViewGraph localViewGraph;
        DBoWHashing* localDBoWHashing;
        PnPRegistration* pnpRegistration;
        SIFTVocabulary* siftVocabulary;
        mutex printMutex;

        Transform velocity;

        int kpNum;
        vector<int> startIndexes;
        vector<vector<pair<int, Point3D>>> correlations;

        void updateCorrelations();

        void updateLocalEdges();

        void registrationPnPBA(FeatureMatches* featureMatches, Edge* edge);

        void registrationWithMotion(SIFTFeaturePoints& f1, SIFTFeaturePoints& f2, Edge* edge);

        void registrationPairEdge(SIFTFeaturePoints* f1, SIFTFeaturePoints* f2, Edge* edge, cudaStream_t curStream);

        void registrationLocalEdges(vector<int>& overlapFrames, EigenVector(Edge)& edges);
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        LocalRegistration(const GlobalConfig &config, SIFTVocabulary* siftVocabulary);

        ViewGraph& getViewGraph();

        void localTrack(shared_ptr<Frame> frame);

        bool needMerge();

        bool isRemain();

        shared_ptr<KeyFrame> mergeFramesIntoKeyFrame();

        ~LocalRegistration();
    };


    class GlobalRegistration {
    protected:
        GlobalConfig globalConfig;

        SIFTFeatureMatcher* matcher;
        ViewGraph viewGraph;
        DBoWHashing* dBoWHashing;
        PnPRegistration* pnpRegistration;
        SIFTVocabulary * siftVocabulary;
        mutex printMutex;

        EigenVector(Edge) edges;

        float3 lastPos;
        bool notLost;// the status for tracking
        int lostNum;

        vector<pair<int, int> > loops;
        set<pair<int, int> > loopCandidates;

        void registrationPnPBA(FeatureMatches* featureMatches, Edge* edge, cudaStream_t curStream);

        void registrationPairEdge(FeatureMatches featureMatches, Edge* edge, cudaStream_t curStream);

        void registrationEdges(shared_ptr<KeyFrame> keyframe, vector<int>& overlapFrames, vector<int>& innerIndexes, EigenVector(Edge)& edges);

        bool loopClosureDetection();

        bool mergeViewGraph();

        void updateLostFrames();

    public:

        GlobalRegistration(const GlobalConfig &config, SIFTVocabulary* siftVocabulary);

        void insertKeyFrames(shared_ptr<KeyFrame> frame);

        int registration(bool opt = true);

        ViewGraph &getViewGraph();

        ~GlobalRegistration();
    };
}
#endif //GraphFusion_REGISTRATIONS_H
