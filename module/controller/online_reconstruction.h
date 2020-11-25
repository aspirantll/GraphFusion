//
// Created by liulei on 2020/8/27.
//

#ifndef GraphFusion_OFFLINE_RECONSTRUCTION_H
#define GraphFusion_OFFLINE_RECONSTRUCTION_H


#include "../datastructure/config.h"
#include "../feature/feature2d.h"
#include "../feature/feature_matcher.h"
#include "../core/registration/registrations.h"
#include "../core/fusion/voxel_fusion.h"
#include "../tool/viewer_util.h"
#include "../tool/thread_pool.h"
#include "../tool/visual_index_hashing.h"

namespace rtf {
    class OnlineReconstruction {
    private:
        GlobalConfig globalConfig;
        SIFTFeatureExtractor* extractor = nullptr;
        DenseFeatureMatcher* denseMatcher = nullptr;

        LocalRegistration* localRegistration = nullptr;
        GlobalRegistration* globalRegistration = nullptr;
        VoxelFusion * voxelFusion = nullptr;

        SIFTVocabulary * siftVocabulary = nullptr;

        int lastFrameIndex;
        int frameCounter;

        VoxelFusion* getVoxelFusion();
    public:
        OnlineReconstruction(const GlobalConfig &globalConfig);

        void appendFrame(shared_ptr<FrameRGBD> frame);

        void finalOptimize(bool opt);

        ViewGraph &getViewGraph();

        void updateViewer(Mesh * mesh);

        void saveMesh(string path);

        bool closed();

        ~OnlineReconstruction();
    };
}


#endif //GraphFusion_OFFLINE_RECONSTRUCTION_H
