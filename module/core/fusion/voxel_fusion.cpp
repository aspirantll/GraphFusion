//
// Created by liulei on 2020/8/18.
//

#include "voxel_fusion.h"
#include "../../processor/frame_converters.h"
#include <pcl/registration/transforms.h>

namespace rtf {
    VoxelFusion::VoxelFusion(const GlobalConfig& globalConfig) {
        HashParams hashParams;
        RayCastParams rayCastParams;
        MarchingCubesParams marchingCubesParams;

        hashParams.m_hashNumBuckets = globalConfig.hashNumBuckets;
        hashParams.m_hashBucketSize = HASH_BUCKET_SIZE;
        hashParams.m_hashMaxCollisionLinkedListSize = globalConfig.hashMaxCollisionLinkedListSize;
        hashParams.m_SDFBlockSize = SDF_BLOCK_SIZE;
        hashParams.m_numSDFBlocks = globalConfig.numSDFBlocks;
        hashParams.m_virtualVoxelSize = globalConfig.virtualVoxelSize;
        hashParams.m_maxIntegrationDistance = globalConfig.maxIntegrationDistance;
        hashParams.m_truncation = globalConfig.truncation;
        hashParams.m_truncScale = globalConfig.truncScale;
        hashParams.m_integrationWeightSample = globalConfig.integrationWeightSample;
        hashParams.m_integrationWeightMax = globalConfig.integrationWeightMax;
        hashParams.m_streamingVoxelExtents = globalConfig.streamingVoxelExtents;
        hashParams.m_streamingGridDimensions = globalConfig.streamingGridDimensions;
        hashParams.m_streamingMinGridPos = globalConfig.streamingMinGridPos;

        sceneRep = new CUDASceneRepHashSDF(hashParams);

        rayCastParams.m_maxNumVertices = hashParams.m_numSDFBlocks*6;
        rayCastParams.m_splatMinimum = globalConfig.splatMinimum;
        rayCastParams.m_width = globalConfig.width;
        rayCastParams.m_height = globalConfig.height;
        rayCastParams.m_minDepth = globalConfig.minDepth;
        rayCastParams.m_maxDepth = globalConfig.maxDepth;
        rayCastParams.m_rayIncrement = hashParams.m_truncation*globalConfig.rayIncrementFactor;
        rayCastParams.m_thresSampleDist = rayCastParams.m_rayIncrement*globalConfig.thresSampleDistFactor;
        rayCastParams.m_thresDist = rayCastParams.m_rayIncrement*globalConfig.thresDistFactor;
        rayCastParams.m_useGradients = globalConfig.useGradients;

        rayCast = new CUDARayCastSDF(rayCastParams);

        marchingCubesParams.m_maxNumTriangles = globalConfig.maxNumTriangles;
        marchingCubesParams.m_threshMarchingCubes = globalConfig.threshMarchingCubesFactor*hashParams.m_virtualVoxelSize;
        marchingCubesParams.m_threshMarchingCubes2 = globalConfig.threshMarchingCubes2Factor*hashParams.m_virtualVoxelSize;
        marchingCubesParams.m_sdfBlockSize = SDF_BLOCK_SIZE;
        marchingCubesParams.m_hashBucketSize = HASH_BUCKET_SIZE;
        marchingCubesParams.m_hashNumBuckets = hashParams.m_hashNumBuckets;

        marchingCubesHashSdf = new CUDAMarchingCubesHashSDF(marchingCubesParams);
    }


    Mesh* VoxelFusion::integrateFrames(ViewGraph& viewGraph) {
        auto camera = viewGraph[0].getCamera();
        int width = camera->getWidth();
        int height = camera->getHeight();

        sceneRep->reset();
        Eigen::Matrix3f intrinsicTrans = camera->getK().cast<float>();
        Eigen::Matrix3f intrinsicTransInv = camera->getReverseK().cast<float>();
        rayCast->setRayCastIntrinsics(width, height, MatrixConversion::toCUDA(intrinsicTrans), MatrixConversion::toCUDA(intrinsicTransInv));
        marchingCubesHashSdf->clearMeshBuffer();
        int mergeCount = 0;
        Transform trans;
        for(const shared_ptr<KeyFrame>& kf: viewGraph.getSourceFrames()) {
            Transform baseTrans = viewGraph.getFrameTransform(kf->getIndex());
            if(!viewGraph.isVisible(kf->getIndex())) {
                continue;
            }
            for(const shared_ptr<Frame>& frame: kf->getFrames()) {
                if(!frame->isVisible()) continue;
                frame->reloadImages();
                FrameConverters::convertImageType(frame);
                cv::cuda::GpuMat rgbImg(height, width, CV_8UC4);
                cv::cuda::GpuMat depthImg(height, width, CV_32F);
                rgbImg.upload(*frame->getRGBImage());
                depthImg.upload(*frame->getDepthImage());
                CUDAFrame cudaFrame(*frame, depthImg, rgbImg);
                trans = baseTrans*frame->getTransform();
                cudaFrame.setTransform(trans);
                sceneRep->integrate(cudaFrame.transformation, cudaFrame);
                frame->releaseImages();
                mergeCount ++;
            }

        }
        cout << "merge count:" << mergeCount << endl;
        cout << "free count:" << sceneRep->getHeapFreeCount() << endl;

        rayCast->render(sceneRep->getHashData(), sceneRep->getHashParams(), MatrixConversion::toCUDA(trans));
        marchingCubesHashSdf->extractIsoSurface(sceneRep->getHashData(), sceneRep->getHashParams(), rayCast->getRayCastData());
        return marchingCubesHashSdf->getMeshData();
    }

    void VoxelFusion::saveMesh(std::string filename) {
        marchingCubesHashSdf->saveMesh(filename);
    }


    VoxelFusion::~VoxelFusion() {
        if (sceneRep != nullptr) {
            delete sceneRep;
            sceneRep = nullptr;
        }
        if (rayCast != nullptr) {
            delete rayCast;
            rayCast = nullptr;
        }
        if (marchingCubesHashSdf != nullptr) {
            delete marchingCubesHashSdf;
            marchingCubesHashSdf = nullptr;
        }
    }
}

