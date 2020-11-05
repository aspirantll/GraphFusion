//
// Created by liulei on 2020/7/16.
//

#include "epipolar_geometry.cuh"
#include "../solver/cuda_math.h"

namespace rtf {
    __global__ void computeRMSKernel(CUDAPtrs mat, CUDAPtrs meanPoint, CUDAPtrs summator) {
        long index = threadIdx.x + blockIdx.x*blockDim.x;

        if(index >= mat.getRows()) return;

        float point[3]={mat(index, 0), mat(index, 1), mat(index, 2)};
        float item = (point[0]-meanPoint[0])*(point[0]-meanPoint[0]) + (point[1]-meanPoint[1])*(point[1]-meanPoint[1])
                + (point[2]-meanPoint[2])*(point[2]-meanPoint[2]);

        summator.set(index, 0, item);
    }

    __global__ void transformPointKernel(CUDAPtrs mat, CUDAPtrs transform) {
        long index = threadIdx.x + blockIdx.x*blockDim.x;

        if(index >= mat.getRows()) return;

        float point[3]={mat(index, 0), mat(index, 1), mat(index, 2)};

        float x = transform(0,0)*point[0]+transform(0,1)*point[1]+transform(0,2)*point[2];
        float y = transform(1,0)*point[0]+transform(1,1)*point[1]+transform(1,2)*point[2];
        float z = transform(2,0)*point[0]+transform(2,1)*point[1]+transform(2,2)*point[2];

        mat.set(index, 0, x);
        mat.set(index, 1, y);
        mat.set(index, 2, z);
    }

    __global__ void computeAMatrixKernel(CUDAPtrs x, CUDAPtrs y, CUDAPtrs A) {
        long index = threadIdx.x + blockIdx.x*blockDim.x;

        if(index >= x.getRows()) return;

        float pointX[3]={x(index, 0), x(index, 1), x(index, 2)};
        float pointY[3]={y(index, 0), y(index, 1), y(index, 2)};

        A.set(index, 0, pointX[0]*pointY[0]);
        A.set(index, 1, pointX[0]*pointY[1]);
        A.set(index, 2, pointX[0]*pointY[2]);
        A.set(index, 3, pointX[1]*pointY[0]);
        A.set(index, 4, pointX[1]*pointY[1]);
        A.set(index, 5, pointX[1]*pointY[2]);
        A.set(index, 6, pointX[2]*pointY[0]);
        A.set(index, 7, pointX[2]*pointY[1]);
        A.set(index, 8, pointX[2]*pointY[2]);
    }

    __global__ void computeResidualKernel(CUDAPtrs x, CUDAPtrs y, CUDAPtrs E, CUDAPtrs residuals) {
        long index = threadIdx.x + blockIdx.x*blockDim.x;

        if(index >= x.getRows()) return;

        float pointX[3]={x(index, 0), x(index, 1), x(index, 2)};
        float pointY[3]={y(index, 0), y(index, 1), y(index, 2)};

        float xtEy = pointY[0]*(E(0,0)*pointX[0]+E(1,0)*pointX[1]+E(2,0)*pointX[2])
                    + pointY[1]*(E(0,1)*pointX[0]+E(1,1)*pointX[1]+E(2,1)*pointX[2])
                    + pointY[2]*(E(0,2)*pointX[0]+E(1,2)*pointX[1]+E(2,2)*pointX[2]);

        float E1y = E(0,0)*pointY[0]+E(0,1)*pointY[1]+E(0,2)*pointY[2];
        float E2y = E(1,0)*pointY[0]+E(1,1)*pointY[1]+E(1,2)*pointY[2];
        float Et1x = E(0,0)*pointX[0]+E(1,0)*pointX[1]+E(2,0)*pointX[2];
        float Et2x = E(0,1)*pointX[0]+E(1,1)*pointX[1]+E(2,1)*pointX[2];

        float residual = xtEy*xtEy/(E1y*E1y + E2y*E2y + Et1x*Et1x + Et2x*Et2x);
//        printf("cuda: %ld %f\n", index, residual);
        residuals.set(index, 0, residual);
    }

    float computeRMS(CUDAMatrixs& mat, CUDAMatrixs& meanPoint, Summator& summator) {
        long n = mat.getRows();
        // compute rms
        CUDA_LINE_BLOCK(n);
        computeRMSKernel <<< grid, block, 0, stream >>> (mat, meanPoint, *summator.dataMat);
        CUDA_CHECKED_NO_ERROR();
        return summator.sum(n, 1, 1)(0,0);
    }

    void transformPoints(CUDAMatrixs& mat, CUDAMatrixs& transform) {
        long n = mat.getRows();
        // compute rms
        CUDA_LINE_BLOCK(n);
        // transform all points
        transformPointKernel <<< grid, block, 0, stream >>> (mat, transform);
        CUDA_CHECKED_NO_ERROR();
    }


    void computeAMatrix(CUDAMatrixs& x, CUDAMatrixs& y, CUDAMatrixs& A) {
        long n = x.getRows();
        A.resize(n, 9);

        CUDA_LINE_BLOCK(n);
        computeAMatrixKernel <<< grid, block, 0, stream >>> (x, y, A);
        CUDA_CHECKED_NO_ERROR();
    }

    void computeResidualsCUDA(CUDAMatrixs& x, CUDAMatrixs& y, CUDAMatrixs& E, CUDAMatrixs& residual) {
        long n = x.getRows();

        CUDA_LINE_BLOCK(n);
        computeResidualKernel <<< grid, block, 0, stream >>> (x, y, E, residual);
        CUDA_CHECKED_NO_ERROR();
    }


    __global__ void computeHomoAMatrixKernel(CUDAPtrs x, CUDAPtrs y, CUDAPtrs A) {
        long index = threadIdx.x + blockIdx.x * blockDim.x;

        if (index >= x.getRows()) return;

        float pointX[3]={x(index, 0), x(index, 1), x(index, 2)};
        float pointY[3]={y(index, 0), y(index, 1), y(index, 2)};

        const float s_0 = pointY[0];
        const float s_1 = pointY[1];
        const float d_0 = pointX[0];
        const float d_1 = pointX[1];

        A.set(index, 0,  s_0);
        A.set(index, 1,  s_1);
        A.set(index, 2,  1);
        A.set(index, 3,  0);
        A.set(index, 4,  0);
        A.set(index, 5,  0);
        A.set(index, 6,  -s_0 * d_0);
        A.set(index, 7,  -s_1 * d_0);
        A.set(index, 8,  -d_0);

        A.set(index+x.getRows(), 0,  0);
        A.set(index+x.getRows(), 1,  0);
        A.set(index+x.getRows(), 2,  0);
        A.set(index+x.getRows(), 3,  -s_0);
        A.set(index+x.getRows(), 4,  -s_1);
        A.set(index+x.getRows(), 5,  -1);
        A.set(index+x.getRows(), 6,  s_0 * d_1);
        A.set(index+x.getRows(), 7,  s_1 * d_1);
        A.set(index+x.getRows(), 8,  d_1);
    }

    void computeHomoAMatrix(CUDAMatrixs& x, CUDAMatrixs& y, CUDAMatrixs& A) {
        long n = x.getRows();
        A.resize(2*n, 9);

        CUDA_LINE_BLOCK(n);
        computeHomoAMatrixKernel <<< grid, block, 0, stream >>> (x, y, A);
        CUDA_CHECKED_NO_ERROR();
    }

    __global__ void computeHomoResidualKernel(CUDAPtrs x, CUDAPtrs y, CUDAPtrs H, CUDAPtrs residuals) {
        long index = threadIdx.x + blockIdx.x*blockDim.x;

        if(index >= x.getRows()) return;

        float pointX[3]={x(index, 0), x(index, 1), x(index, 2)};
        float pointY[3]={y(index, 0), y(index, 1), y(index, 2)};

        const float H_00 = H(0, 0);
        const float H_01 = H(0, 1);
        const float H_02 = H(0, 2);
        const float H_10 = H(1, 0);
        const float H_11 = H(1, 1);
        const float H_12 = H(1, 2);
        const float H_20 = H(2, 0);
        const float H_21 = H(2, 1);
        const float H_22 = H(2, 2);

        const float s_0 = pointY[0];
        const float s_1 = pointY[1];
        const float d_0 = pointX[0];
        const float d_1 = pointX[1];

        const float pd_0 = H_00 * s_0 + H_01 * s_1 + H_02;
        const float pd_1 = H_10 * s_0 + H_11 * s_1 + H_12;
        const float pd_2 = H_20 * s_0 + H_21 * s_1 + H_22;

        const float inv_pd_2 = 1.0 / pd_2;
        const float dd_0 = d_0 - pd_0 * inv_pd_2;
        const float dd_1 = d_1 - pd_1 * inv_pd_2;

        float residual = dd_0 * dd_0 + dd_1 * dd_1;
//        printf("cuda: %ld %f\n", index, residual);
        residuals.set(index, 0, residual);
    }


    void computeHomoResidualsCUDA(CUDAMatrixs& x, CUDAMatrixs& y, CUDAMatrixs& H, CUDAMatrixs& residual) {
        long n = x.getRows();

        CUDA_LINE_BLOCK(n);
        computeHomoResidualKernel <<< grid, block, 0, stream >>> (x, y, H, residual);
        CUDA_CHECKED_NO_ERROR();
    }

}