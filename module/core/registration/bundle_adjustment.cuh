//
// Created by liulei on 2020/7/15.
//

#ifndef GraphFusion_BUNDLE_ADJUSTMENT_H
#define GraphFusion_BUNDLE_ADJUSTMENT_H

#include "../../tool/map_reduce.h"
#include "../solver/cuda_matrix.h"

namespace rtf {
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

        LMSumMats& mulWeight(Scalar weight) {
            H *= weight;
            M *= weight;
            b *= weight;
            return (*this);
        }
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

    void computeBACostAndJacobi(CUDAMatrixs& objectPoints, CUDAMatrixs& tarPixels, float4x4& T, float3x3& K, CUDAMatrixc& mask, Summator& costSummator, Summator& hSummator, Summator& mSummator, Summator& bSummator);

    void computeBACost(CUDAMatrixs& objectPoints, CUDAMatrixs& tarPixels, float4x4& T, float3x3& K, CUDAMatrixc& mask, Summator& costSummator);

    void computerInliers(Summator& costSummator, CUDAMatrixc& inliers, CudaScalar th);

    void computeMVBACostAndJacobi(CUDAEdgeVector &edges, CUDAVector<CUDALMSummators>& gtSummators, CUDAVector<CUDALMSummators>& deltaSummators, Summator& costSummator);

    void computeMVBACost(CUDAEdgeVector &edges, Summator& costSummator);


}

#endif //GraphFusion_BUNDLE_ADJUSTMENT_H
