//
// Created by liulei on 2020/10/15.
//

#ifndef RTF_VISUAL_INDEX_HASHING_CUH
#define RTF_VISUAL_INDEX_HASHING_CUH

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

#endif //RTF_VISUAL_INDEX_HASHING_CUH
