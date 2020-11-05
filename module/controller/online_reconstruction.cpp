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
        lastFrameIndex = 0;
        frameCounter = 0;
    }


    void OnlineReconstruction::computeBow(SIFTFeaturePoints& sf) {
        auto & bowVec = sf.getMBowVec();
        auto & featVec = sf.getMFeatVec();
        if(bowVec.empty()) {
            vector<vector<float>> vCurrentDesc;
            toDescriptorVector(sf.getDescriptors(), vCurrentDesc);
            siftVocabulary->transform(vCurrentDesc,bowVec,featVec,4);
        }
    }

    bool OnlineReconstruction::needMerge() {
        return frameCounter-lastFrameIndex>=globalConfig.chunkSize;
    }

    VoxelFusion* OnlineReconstruction::getVoxelFusion() {
        if(voxelFusion == nullptr) {
            voxelFusion = new VoxelFusion(globalConfig);
        }
        return voxelFusion;
    }

    void OnlineReconstruction::appendFrame(shared_ptr<FrameRGBD> frame) {
        // initialize frame
        frame->setFrameIndex(frameCounter++);
        shared_ptr<FrameRGBDT> frameRGBDT = allocate_shared<FrameRGBDT>(Eigen::aligned_allocator<FrameRGBDT>(), frame);
        shared_ptr<KeyFrame> keyframe = allocate_shared<KeyFrame>(Eigen::aligned_allocator<KeyFrame>());
        keyframe->addFrame(frameRGBDT);
        // extract sift feature points
        extractor->extractFeatures(frame, keyframe->getKps());
        computeBow(keyframe->getKps());

        // track the keyframes database
        std::thread kfTracker(bind(&GlobalRegistration::trackKeyFrames, globalRegistration, placeholders::_1), keyframe);
        // track local frames
        localRegistration->localTrack(keyframe);
        // merging graph
        if(needMerge()) {
            Timer merger = Timer::startTimer("merge frames");
            shared_ptr<KeyFrame> kf = localRegistration->mergeFramesIntoKeyFrame();
            merger.stopTimer();
            kfTracker.join();
            globalRegistration->insertKeyFrames(kf);
            lastFrameIndex = frameCounter;
        }else {
            kfTracker.join();
        }
    }

    void OnlineReconstruction::finalOptimize(bool opt) {
        shared_ptr<KeyFrame> kf = localRegistration->mergeFramesIntoKeyFrame();
        globalRegistration->insertKeyFrames(kf);
        lastFrameIndex = frameCounter;

        globalRegistration->registration(opt);
        /*auto meshData = voxelFusion->integrateFrames(viewGraph);
        cout << "show final mesh" << endl;
        updateViewer(meshData);*/
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
