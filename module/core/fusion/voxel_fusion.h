//
// Created by liulei on 2020/8/18.
//

#ifndef RTF_VOXEL_FUSION_H
#define RTF_VOXEL_FUSION_H

#include "cuda_scene_rep.h"
#include "../../datastructure/view_graph.h"

namespace rtf {
    class VoxelFusion {
    private:
        CUDASceneRepHashSDF *sceneRep = nullptr;
        CUDARayCastSDF *rayCast = nullptr;
        CUDAMarchingCubesHashSDF *marchingCubesHashSdf = nullptr;
    public:
        VoxelFusion(const GlobalConfig& globalConfig);

        Mesh* integrateFrames(ViewGraph& viewGraph);

        void saveMesh(std::string filename);

        ~VoxelFusion();
    };
}



#endif //RTF_VOXEL_FUSION_H
