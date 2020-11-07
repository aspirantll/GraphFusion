//
// Created by liulei on 2020/10/15.
//

#ifndef GraphFusion_VISUAL_INDEX_HASHING_CUH
#define GraphFusion_VISUAL_INDEX_HASHING_CUH

#include "../datastructure/cuda_types.h"
#include "../core/solver/cuda_matrix.h"
#include "../core/solver/matrix_conversion.h"

namespace rtf {
    typedef struct CUDABoW {
        int imageId;
        CUDAPtru words;
    } CUDABoW;


    void wordsCount(CUDAPtrArray<CUDABoW>& voc, CUDAArrayu& cur, CUDAArrayu& wordCounts);
}

#endif //GraphFusion_VISUAL_INDEX_HASHING_CUH
