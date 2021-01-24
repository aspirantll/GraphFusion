//
// Created by liulei on 2020/8/27.
//

#include "online_reconstruction.h"
#include <opencv2/core/eigen.hpp>

namespace rtf {
    OnlineReconstruction::OnlineReconstruction(const GlobalConfig &globalConfig) : globalConfig(globalConfig) {
        extractor = new ORBFeatureExtractor();
        denseMatcher = new DenseFeatureMatcher(globalConfig);

        orbVocabulary = new ORBVocabulary();
        orbVocabulary->loadFromTextFile(globalConfig.vocTxtPath);

        localRegistration = new LocalRegistration(globalConfig, orbVocabulary);
        globalRegistration = new GlobalRegistration(globalConfig, orbVocabulary);

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
        // extract orb feature points
        extractor->extractFeatures(frameRGBD, frame->getKps());
        if(frame->getKps().empty()) return;

        // track the keyframes database
//        std::thread kfTracker(bind(&GlobalRegistration::trackKeyFrames, globalRegistration, placeholders::_1), frame);
        // track local frames
        localRegistration->localTrack(frame);
        // merging graph
        if(localRegistration->needMerge()) {
            Timer merger = Timer::startTimer("merge frames");
            shared_ptr<KeyFrame> kf = localRegistration->mergeFramesIntoKeyFrame();
            merger.stopTimer();
//            kfTracker.join();
            Timer insertKF = Timer::startTimer("insert frames");
            globalRegistration->insertKeyFrames(kf);
            insertKF.stopTimer();
        }else {
//            kfTracker.join();
        }

    }

    void OnlineReconstruction::finalOptimize(bool opt) {
        if(localRegistration->isRemain()) {
            shared_ptr<KeyFrame> kf = localRegistration->mergeFramesIntoKeyFrame();
            globalRegistration->insertKeyFrames(kf);
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
