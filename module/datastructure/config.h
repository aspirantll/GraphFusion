//
// Created by liulei on 2020/6/5.
//

#ifndef GraphFusion_CONFIG_H
#define GraphFusion_CONFIG_H

#include <string>
#include "camera.h"
#include "cuda_types.h"


using namespace std;

namespace rtf {
    class GlobalConfig {
    public:
        /* base */
        string workspace;

        /* input source */
        int filterFrames = 10;
        ushort minDepth = 100;
        ushort maxDepth = 3000;
        bool alignedColor = true;
        int parallelSize = -1;

        /* ransac */
        // max iterations
        int numOfTrials = 250;
        // start residuals
        float maxEGResidual = 0.0001;
        float maxHomoResidual = 0.0001;
        float maxPnPResidual = 5.991;
        // residual upper limit
        float upperBoundResidual = 0.001;
        // don't abort probability
        float confidence = 0.99;
        // multiplier for dynamic iterations
        float lambda = 3;

        // down step factor
        float stepFactor = 0.5;
        // inliers ratio threshold
        float irThreshold = 0.8;
        // adaptive ending delta
        float aeDelta = 0.00001;
        // allow difference for inlier ratio
        float irDelta = 0.02;
        // min inlier ratio
        float minInlierRatio = 0.3;
        // min matches num for every pair
        int kMinMatches = 25;
        // min inliers num for every pair
        int kMinInliers = 15;
        // rms threshold
        float rmsThreshold = 100;

        /* local and global */
        // the upper bound for average registration cost for every pair
        float maxAvgCost = 15;
        // relaxtion for bundle adjustment
        float relaxtion = 1;
        // cost threshold
        float costThreshold = 2;

        /* ------------------voxel fusion --------------------------*/
        /* hash params*/
        unsigned int hashNumBuckets = 800000;
        unsigned int hashMaxCollisionLinkedListSize = 7;
        unsigned int numSDFBlocks = 200000;

        float virtualVoxelSize =  0.010f;

        float maxIntegrationDistance = 3.0f;
        float truncScale = 0.06f;
        float truncation = 0.02f;
        unsigned int integrationWeightSample = 1;
        unsigned int integrationWeightMax = 99999999;

        float3 streamingVoxelExtents = make_float3(1.0f, 1.0f, 1.0f);
        int3 streamingGridDimensions = make_int3(257, 257, 257);
        int3 streamingMinGridPos = make_int3(257, 257, 257);

        /** raycast params */
        unsigned int width = 1280;
        unsigned int height = 720;

        int splatMinimum = 1;

        float rayIncrementFactor = 0.8f;
        float thresSampleDistFactor = 50.5f;
        float thresDistFactor = 50.0f;
        bool  useGradients = false;

        /** marching cube params */
        unsigned int maxNumTriangles = 10000000;
        float threshMarchingCubesFactor = 10.0f;
        float threshMarchingCubes2Factor = 10.0f;

        // voc search
        string vocTxtPath;
        float minScore = 0.25;
        int numThreads = 3;
        int numChecks = 256;
        int numNeighs = 10;
        int maxNumFeatures = 8192;
        int overlapNum = 5;

        // dense match
        int neigh = 2; // search neigh range
        int windowRadius = 5;// NCC radius
        float sigmaSpatial = -1;
        float sigmaColor = 0.2f;
        float deltaNormalTh = 0.2;
        float nccTh = 0.5;
        int downSampleScale = 16;

        // view graph
        int chunkSize = 30;
        float maxPointError = 0.01f;
        int frameNeighs = 1;

        bool fuse = false;
        int kpFuseTh = 2000;
        float fuseScore = 0.7;
        bool downSample = false;
        int downSampleGridSize = 10;


        GlobalConfig(const string &workspace);

        void loadFromFile(const string &file);

        void saveToFile(const string &file);

    };

    class BaseConfig {
    private:
        static shared_ptr<BaseConfig> config;

        BaseConfig(const GlobalConfig& globalConfig);
    public:
        string workspace;
        CameraModelType cameraModelType = CameraModelType::PINHOLE;
        bool fuse;
        int kpFuseTh;
        float fuseScore;
        bool downSample;
        bool downSampleGridSize;

        static void initInstance(const GlobalConfig& globalConfig);

        static shared_ptr<BaseConfig> getInstance();
    };

}

#endif //GraphFusion_CONFIG_H
