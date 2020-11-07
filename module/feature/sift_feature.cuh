//
// Created by liulei on 2020/9/9.
//

#ifndef GraphFusion_SIFT_FEATURE_CUH
#define GraphFusion_SIFT_FEATURE_CUH

#include "../datastructure/cuda_types.h"
#include "feature_point.h"

namespace rtf {
    typedef struct CUDAScore {
        float score;
        unsigned int refIndex;
        unsigned int curIndex;
    } CUDAScore;

    typedef CUDAMatrix<unsigned char> CUDAFeatures;
    typedef CUDAArray<unsigned int> CUDAFeatureIndexes;
    typedef CUDAArray<int> CUDAMatchIndexes;
    typedef CUDAMatrix<CUDAScore> CUDAMatchScores;


    void findBestMatches(CUDAFeatures& refDesc, CUDAFeatures& curDesc, CUDAFeatureIndexes& refInd
            , CUDAFeatureIndexes& curInd, float distMax, float ratioMax, CUDAMatchIndexes& rowMatch, CUDAMatchIndexes& colMatch);
}


#endif //GraphFusion_SIFT_FEATURE_CUH
