//
// Created by liulei on 2020/11/12.
//

#include "icp_multiview.cuh"

constexpr Scalar kHuberWeight = 0.1;

__device__ Scalar computeHuberWeight(Scalar squared_residual, Scalar huber_parameter) {
    return (squared_residual < huber_parameter * huber_parameter) ? 1 : (huber_parameter / sqrtf(squared_residual));
}

__device__ Scalar ComputeHuberCost(Scalar squared_residual, Scalar huber_parameter) {
    if (squared_residual < huber_parameter * huber_parameter) {
        return 0.5 * squared_residual;
    } else {
        return huber_parameter * (sqrtf(squared_residual) - 0.5 * huber_parameter);
    }
}

__device__ void composeICPJacobi(Scalar* hat, Scalar* jacobi) {
    // for t
    jacobi[0] = 1;
    jacobi[1] = 0;
    jacobi[2] = 0;

    jacobi[6] = 0;
    jacobi[7] = 1;
    jacobi[8] = 0;

    jacobi[12] = 0;
    jacobi[13] = 0;
    jacobi[14] = 1;

    // for R
    jacobi[3] = -hat[0];
    jacobi[4] = -hat[1];
    jacobi[5] = -hat[2];

    jacobi[9] = -hat[3];
    jacobi[10] = -hat[4];
    jacobi[11] = -hat[5];

    jacobi[15] = -hat[6];
    jacobi[16] = -hat[7];
    jacobi[17] = -hat[8];
}

__device__ void transformPoint(float4x4 T, Scalar* point, Scalar *tPoint) {
    Scalar x = T(0,0)*point[0] + T(0,1)*point[1] + T(0,2)*point[2] + T(0,3);
    Scalar y = T(1,0)*point[0] + T(1,1)*point[1] + T(1,2)*point[2] + T(1,3);
    Scalar z = T(2,0)*point[0] + T(2,1)*point[1] + T(2,2)*point[2] + T(2,3);

    tPoint[0] = x;
    tPoint[1] = y;
    tPoint[2] = z;
}

__device__ void projectPoint(float3x3 k, Scalar* point, Scalar* pixel) {
    Scalar x = k(0,0)*point[0]+k(0,1)*point[1]+k(0,2)*point[2];
    Scalar y = k(1,0)*point[0]+k(1,1)*point[1]+k(1,2)*point[2];
    Scalar z = k(2,0)*point[0]+k(2,1)*point[1]+k(2,2)*point[2];

    pixel[0] = x/z;
    pixel[1] = y/z;
}

__device__ void hatMatrix(Scalar* point, Scalar* hat) {
    hat[0] = 0;
    hat[1] = -point[2];
    hat[2] = point[1];
    hat[3] = point[2];
    hat[4] = 0;
    hat[5] = -point[0];
    hat[6] = -point[1];
    hat[7] = point[0];
    hat[8] = 0;
}

// x = u/fx*d-cx/fx*d, y=v/fy*d-cy/fy*d, z=d
__device__ void unproject(float3x3 K, Scalar* pixel, Scalar *dst) {
    Scalar fx=K(0, 0), fy=K(1, 1), cx=K(0, 2), cy=K(1, 2);
    dst[0] = pixel[2]*(pixel[0]-cx)/fx;
    dst[1] = pixel[2]*(pixel[1]-cy)/fy;
    dst[2] = pixel[2];
}

__device__ void computeICPHMb(CUDALMSummators summators, long index, Scalar weight, Scalar* jacobi, Scalar* residual, Scalar jacobiWeight) {
    Scalar * H = summators.H.data+index*36;
    Scalar * M = summators.M.data+index*6;
    Scalar * b = summators.b.data+index*6;
    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            H[j*6+i] = jacobi[i]*jacobi[j] + jacobi[i+6]*jacobi[j+6]+jacobi[i+12]*jacobi[j+12];
        }
        M[i] = weight*H[i*6+i];
        b[i] = -weight*jacobiWeight*(jacobi[i]*residual[0]+jacobi[i+6]*residual[1]+jacobi[i+12]*residual[2]);
    }
}

__global__ void computeMVICPCostAndJacobiForEdge(CUDAEdge edge, CUDALMSummators summatorsX, CUDALMSummators summatorsY, CUDALMSummators deltaSummators, CUDAPtrs costSummator) {
    // obtain parameters from
    long index = threadIdx.x + blockIdx.x*blockDim.x;

    long sumIndexX = edge.sumIndexX+index;
    long sumIndexY = edge.sumIndexY+index;
    long costIndex = edge.costIndex+index;
    CUDAPtrs kx = edge.kx;
    CUDAPtrs ky = edge.ky;
    float3x3 intrinsicX = edge.intrinsicX;
    float3x3 intrinsicY = edge.intrinsicY;
    float4x4 transform = edge.transform;

    if(index>=kx.getRows()) return;

    Scalar py[3]={ky(index, 0), ky(index, 1), ky(index, 2)},
            px[3] = {kx(index, 0), kx(index, 1), kx(index, 2)};
    Scalar qx[3], qy[3], hatMat[9], residual[3], jacobi[18];
    unproject(intrinsicY, py, qy);
    transformPoint(transform, qy, qy);

    // compute jacobi
    hatMatrix(qy, hatMat);
    composeICPJacobi(hatMat, jacobi);

    unproject(intrinsicX, px, qx);
    // compute residual and cost
    residual[0] = qx[0] - qy[0];
    residual[1] = qx[1] - qy[1];
    residual[2] = qx[2] - qy[2];

    Scalar weight = computeHuberWeight(residual[0]*residual[0]+residual[1]*residual[1]+residual[2]*residual[2], kHuberWeight);
    Scalar cost = ComputeHuberCost(residual[0]*residual[0]+residual[1]*residual[1]+residual[2]*residual[2], kHuberWeight);

    costSummator.data[costIndex]=cost;
    // compute H,M,b
    computeICPHMb(summatorsX, sumIndexX, weight, jacobi, residual, 1.0);
    computeICPHMb(summatorsY, sumIndexY, weight, jacobi, residual, -1.0);
    computeICPHMb(deltaSummators, index, weight, jacobi, residual, 1.0);
}

__global__ void computeMVICPCostForEdge(CUDAEdge edge, CUDAPtrs costSummator) {
    // obtain parameters from
    long index = threadIdx.x + blockIdx.x*blockDim.x;

    long costIndex = edge.costIndex+index;
    CUDAPtrs kx = edge.kx;
    CUDAPtrs ky = edge.ky;
    float3x3 intrinsicX = edge.intrinsicX;
    float3x3 intrinsicY = edge.intrinsicY;
    float4x4 transform = edge.transform;

    if(index>=kx.getRows()) return;

    Scalar py[3]={ky(index, 0), ky(index, 1), ky(index, 2)},
            px[3] = {kx(index, 0), kx(index, 1), kx(index, 2)};

    Scalar qx[3], qy[3], residual[3];
    unproject(intrinsicY, py, qy);
    transformPoint(transform, qy, qy);

    unproject(intrinsicX, px, qx);
    // compute residual and cost
    residual[0] = qx[0] - qy[0];
    residual[1] = qx[1] - qy[1];
    residual[2] = qx[2] - qy[2];

    Scalar cost = ComputeHuberCost(residual[0]*residual[0]+residual[1]*residual[1]+residual[2]*residual[2], kHuberWeight);

    costSummator.data[costIndex]=cost;
}


void computeMVICPCostAndJacobi(CUDAEdgeVector &edges, CUDAVector<CUDALMSummators>& gtSummators, CUDAVector<CUDALMSummators>& deltaSummators, Summator& costSummator) {
    for(long index=0; index<edges.getNum(); index++) {
        CUDA_LINE_BLOCK(edges[index].count);

        computeMVICPCostAndJacobiForEdge<<<grid, block, 0, stream>>>(edges[index], gtSummators[edges[index].indexX], gtSummators[edges[index].indexY], deltaSummators[index], *costSummator.dataMat);

        CUDA_CHECKED_NO_ERROR();
    }
}

void computeMVICPCost(CUDAEdgeVector &edges, Summator& costSummator) {
    for(long index=0; index<edges.getNum(); index++) {
        CUDA_LINE_BLOCK(edges[index].count);

        computeMVICPCostForEdge<<<grid, block, 0, stream>>>(edges[index], *costSummator.dataMat);

        CUDA_CHECKED_NO_ERROR();
    }
}