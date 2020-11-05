//
// Created by liulei on 2020/10/12.
//

#ifndef RTF_DENSE_MATCH_CUH
#define RTF_DENSE_MATCH_CUH

#include "../datastructure/view_graph.h"
#include "../core/solver/cuda_matrix.h"
#include "../core/solver/matrix_conversion.h"

namespace rtf {

    __align__(16)
    typedef struct DenseMatchParams{
        int width;
        int height;
        int neigh; // search neigh range
        int windowRadius;// NCC radius
        float sigmaSpatial;
        float sigmaColor;
        float deltaNormalTh;
        float nccTh;
        int downSampleScale;

        float3x3 K;
        float3x3 invK;
        float4x4 trans;
    } DenseMatchParams;

    void updateDenseMatchParams(const DenseMatchParams& denseMatchParams);

    void bindTextureParams(uchar* grayImg1, float* depthImg1, float4* normalImg1
            , uchar* grayImg2, float* depthImg2, float4* normalImg2, int width, int height);

    void leftDenseMatch(CUDAMatrixl& matches, CUDAMatrixs& matchScores);

    void rightDenseMatch(CUDAMatrixl& matches);

    void crossCheck(CUDAMatrixl& leftMatches, CUDAMatrixl& rightMatches, CUDAMatrixs& matchScores, CUDAMatrixc& mask);
}

#endif //RTF_DENSE_MATCH_CUH
