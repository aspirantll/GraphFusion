//
// Created by liulei on 2020/11/12.
//

#ifndef GRAPHFUSION_ICP_MULTIVIEW_CUH
#define GRAPHFUSION_ICP_MULTIVIEW_CUH
#include "../../tool/map_reduce.h"
#include "../solver/cuda_matrix.h"

using namespace rtf;

class CUDAEdge {
public:
    long indexX;
    long indexY;
    long sumIndexX;
    long sumIndexY;
    long costIndex;
    long count;
    CUDAPtrs kx;
    CUDAPtrs ky;
    float3x3 intrinsicX;
    float3x3 intrinsicY;
    float4x4 transform;

};

typedef CUDAVector<CUDAEdge> CUDAEdgeVector;

class CUDALMSummators {
public:
    CUDAPtrs H;
    CUDAPtrs M;
    CUDAPtrs b;
};

class LMSumMats {
public:
    Eigen::Matrix<Scalar, 6, 6> H;
    Eigen::Matrix<Scalar, 6, 1> M;
    Eigen::Matrix<Scalar, 6, 1> b;
};

class LMSummators{
public:
    Summator * H = nullptr;
    Summator * M = nullptr;
    Summator * b = nullptr;

    LMSummators(int n) {
        H = new Summator(n, 6, 6);
        M = new Summator(n, 6, 1);
        b = new Summator(n, 6, 1);
    }

    CUDALMSummators uploadToCUDA() {
        CUDALMSummators summators;
        summators.H = *H->dataMat;
        summators.M = *M->dataMat;
        summators.b = *b->dataMat;
        return summators;
    }

    LMSumMats sum() {
        return {H->sum(), M->sum(), b->sum()};
    }

    ~LMSummators() {
        if(H) delete H;
        if(M) delete M;
        if(b) delete b;
    }
};

void computeMVICPCostAndJacobi(CUDAEdgeVector &edges, CUDAVector<CUDALMSummators>& gtSummators, CUDAVector<CUDALMSummators>& deltaSummators, Summator& costSummator);

void computeMVICPCost(CUDAEdgeVector &edges, Summator& costSummator);


#endif //GRAPHFUSION_ICP_MULTIVIEW_CUH
