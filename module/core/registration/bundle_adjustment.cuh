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
        long indexZ;
        CUDAPtrs kx;
        CUDAPtrs ky;
        float3x3 intrinsic;
        float4x4 transform;
        float4x4 transformInv;

        float3 median;
        Scalar scale;
    };

    typedef CUDAVector<CUDAEdge> CUDAEdgeVector;

    class LMSumMats {
    public:
        MatrixX H;
        VectorX M;
        VectorX b;
        Scalar cost;

        shared_ptr<CUDAMatrixs> cH, cM, cb;
        Scalar * cCost;

        LMSumMats(int n) {
            H.resize(n, n);
            M.resize(n, 1);
            b.resize(n, 1);

            H.setZero();
            M.setZero();
            b.setZero();

            cost = 0;
        }

        void uploadToCUDA() {
            cH = make_shared<CUDAMatrixs>(H);
            cM = make_shared<CUDAMatrixs>(M);
            cb = make_shared<CUDAMatrixs>(b);
            CUDA_CHECKED_CALL(cudaMalloc(&cCost, sizeof(Scalar)));
            CUDA_CHECKED_CALL(cudaMemcpyAsync(cCost, &cost, sizeof(Scalar), cudaMemcpyHostToDevice, stream));
        }

        void downloadToHost() {
            cH->download(H);
            cM->download(M);
            cb->download(b);
            CUDA_CHECKED_CALL(cudaMemcpyAsync(&cost, cCost, sizeof(Scalar), cudaMemcpyDeviceToHost, stream));
            CUDA_SYNCHRONIZE();

            cH->setZero();
            cM->setZero();
            cb->setZero();
            CUDA_CHECKED_CALL(cudaMemsetAsync(cCost, 0, sizeof(Scalar), stream));
        }

        LMSumMats& mulWeight(Scalar weight) {
            H *= weight;
            M *= weight;
            b *= weight;
            return (*this);
        }
    };

    void computeBACostAndJacobi(CUDAMatrixs& objectPoints, CUDAMatrixs& tarPixels, float4x4& T, float3x3& K, CUDAMatrixc& mask, Summator& costSummator, Summator& hSummator, Summator& mSummator, Summator& bSummator);

    void computeBACost(CUDAMatrixs& objectPoints, CUDAMatrixs& tarPixels, float4x4& T, float3x3& K, CUDAMatrixc& mask, Summator& costSummator);

    void computerInliers(Summator& costSummator, CUDAMatrixc& inliers, Scalar th);

    void computeMVBACostAndJacobi(CUDAEdgeVector &edges, LMSumMats& sumMats);

    void computeMVBACost(CUDAEdgeVector &edges, Scalar& cost);


}

#endif //GraphFusion_BUNDLE_ADJUSTMENT_H
