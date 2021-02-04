//
// Created by liulei on 2020/8/27.
//

#include "online_reconstruction.h"
#include <opencv2/core/eigen.hpp>

namespace rtf {
    OnlineReconstruction::OnlineReconstruction(const GlobalConfig &globalConfig) : globalConfig(globalConfig) {
        extractor = new SIFTFeatureExtractor();
        denseMatcher = new DenseFeatureMatcher(globalConfig);

        siftVocabulary = new SIFTVocabulary();
        siftVocabulary->loadFromTextFile(globalConfig.vocTxtPath);

        localRegistration = new LocalRegistration(globalConfig, siftVocabulary);
        globalRegistration = new GlobalRegistration(globalConfig, siftVocabulary);

//        viewer.run();
        frameCounter = 0;
    }

    VoxelFusion* OnlineReconstruction::getVoxelFusion() {
        if(voxelFusion == nullptr) {
            voxelFusion = new VoxelFusion(globalConfig);
        }
        return voxelFusion;
    }

    void OnlineReconstruction::appendFrame(shared_ptr<FrameRGBD> frameRGBD) {
        // initialize frame
        frameRGBD->setFrameIndex(frameCounter++);
        shared_ptr<Frame> frame = allocate_shared<Frame>(Eigen::aligned_allocator<Frame>(), frameRGBD);
        // extract sift feature points
        extractor->extractFeatures(frameRGBD, frame->getKps());
        if(frame->getKps().empty()) return;

        // track local frames
        localRegistration->localTrack(frame);
        if(localRegistration->needMerge()) {
            shared_ptr<ViewCluster> cluster = localRegistration->mergeFramesIntoCluster();
            globalRegistration->insertViewCluster(cluster);
        }
    }

    void OnlineReconstruction::finalOptimize(bool opt) {
        if(localRegistration->isRemain()) {
            shared_ptr<ViewCluster> cluster = localRegistration->mergeFramesIntoCluster();
            globalRegistration->insertViewCluster(cluster);
        }

        globalRegistration->registration(opt);
//        auto meshData = getVoxelFusion()->integrateFrames(getViewGraph());
//        cout << "show final mesh" << endl;
//        updateViewer(meshData);
    }

    ViewGraph &OnlineReconstruction::getViewGraph() {
        return globalRegistration->getViewGraph();
    }

    void OnlineReconstruction::updateViewer(Mesh *mesh) {
//        viewer.setMesh(mesh);
//        PointUtil::savePLYMesh(
//                BaseConfig::getInstance()->workspace + "/mesh_" + to_string(viewGraph.getFramesNum()) + ".ply", *mesh);
    }

    bool OnlineReconstruction::closed() {
//        return viewer.closed();
    }

    void OnlineReconstruction::saveMesh(string path) {
        voxelFusion->saveMesh(path);
    }


    OnlineReconstruction::~OnlineReconstruction() {
        delete extractor;
        delete denseMatcher;
        delete localRegistration;
        delete globalRegistration;
        if(voxelFusion)
            delete voxelFusion;
    }

}
