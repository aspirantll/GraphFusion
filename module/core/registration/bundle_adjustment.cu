//
// Created by liulei on 2020/6/12.
//

#include "bundle_adjustment.cuh"

namespace rtf {
    constexpr float kHuberWeight = 1.2;
    __device__ float computeHuberWeight(float residual_x, float residual_y, float huber_parameter) {
        float squared_residual = residual_x * residual_x + residual_y * residual_y;
        return (squared_residual < huber_parameter * huber_parameter) ? 1 : (huber_parameter / sqrtf(squared_residual));
    }

    __device__ float ComputeHuberCost(float residual_x, float residual_y, float huber_parameter) {
        float squared_residual = residual_x * residual_x + residual_y * residual_y;
        if (squared_residual < huber_parameter * huber_parameter) {
            return 0.5 * squared_residual;
        } else {
            return huber_parameter * (sqrtf(squared_residual) - 0.5 * huber_parameter);
        }
    }

    __device__ void projectJacobi(float3x3 k, float* point, float* proJacobi) {
        float x = k(0,0)*point[0]+k(0,1)*point[1]+k(0,2)*point[2];
        float y = k(1,0)*point[0]+k(1,1)*point[1]+k(1,2)*point[2];
        float z = k(2,0)*point[0]+k(2,1)*point[1]+k(2,2)*point[2];

        proJacobi[0] = z*k(0,0)-x*k(2,0)/(z*z);
        proJacobi[1] = z*k(0,1)-x*k(2,1)/(z*z);
        proJacobi[2] = z*k(0,2)-x*k(2,2)/(z*z);
        proJacobi[3] = z*k(1,0)-y*k(2,0)/(z*z);
        proJacobi[4] = z*k(1,1)-y*k(2,1)/(z*z);
        proJacobi[5] = z*k(1,2)-y*k(2,2)/(z*z);
    }

    __device__ void transformPointOnlyRotation(float4x4 T, float* point, float *tPoint) {
        float x = T(0,0)*point[0] + T(0,1)*point[1] + T(0,2)*point[2];
        float y = T(1,0)*point[0] + T(1,1)*point[1] + T(1,2)*point[2];
        float z = T(2,0)*point[0] + T(2,1)*point[1] + T(2,2)*point[2];

        tPoint[0] = x;
        tPoint[1] = y;
        tPoint[2] = z;
    }


    __device__ void transformPoint(float4x4 T, float* point, float *tPoint) {
        float x = T(0,0)*point[0] + T(0,1)*point[1] + T(0,2)*point[2] + T(0,3);
        float y = T(1,0)*point[0] + T(1,1)*point[1] + T(1,2)*point[2] + T(1,3);
        float z = T(2,0)*point[0] + T(2,1)*point[1] + T(2,2)*point[2] + T(2,3);

        tPoint[0] = x;
        tPoint[1] = y;
        tPoint[2] = z;
    }

    __device__ void projectPoint(float3x3 k, float* point, float* pixel) {
        float x = k(0,0)*point[0]+k(0,1)*point[1]+k(0,2)*point[2];
        float y = k(1,0)*point[0]+k(1,1)*point[1]+k(1,2)*point[2];
        float z = k(2,0)*point[0]+k(2,1)*point[1]+k(2,2)*point[2];

        pixel[0] = x/z;
        pixel[1] = y/z;
    }

    __device__ void hatMatrix(float* point, float* hat) {
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
    __device__ void unproject(float3x3 K, float* pixel, float *dst) {
        float fx=K(0, 0), fy=K(1, 1), cx=K(0, 2), cy=K(1, 2);
        dst[0] = pixel[2]*(pixel[0]-cx)/fx;
        dst[1] = pixel[2]*(pixel[1]-cy)/fy;
        dst[2] = pixel[2];
    }

    __device__ void computeJacobi(float* proJacobi, float* hat, float* jacobi) {
        // for R
        jacobi[0] = -(proJacobi[0]*hat[0]+proJacobi[1]*hat[3]+proJacobi[2]*hat[6]);
        jacobi[1] = -(proJacobi[0]*hat[1]+proJacobi[1]*hat[4]+proJacobi[2]*hat[7]);
        jacobi[2] = -(proJacobi[0]*hat[2]+proJacobi[1]*hat[5]+proJacobi[2]*hat[8]);

        jacobi[6] = -(proJacobi[3]*hat[0]+proJacobi[4]*hat[3]+proJacobi[5]*hat[6]);
        jacobi[7] = -(proJacobi[3]*hat[1]+proJacobi[4]*hat[4]+proJacobi[5]*hat[7]);
        jacobi[8] = -(proJacobi[3]*hat[2]+proJacobi[4]*hat[5]+proJacobi[5]*hat[8]);

        // for t
        jacobi[3] = proJacobi[0];
        jacobi[4] = proJacobi[1];
        jacobi[5] = proJacobi[2];

        jacobi[9] = proJacobi[3];
        jacobi[10] = proJacobi[4];
        jacobi[11] = proJacobi[5];
    }


    __device__ void computeDeltaLie(float *jacobi, float *residual, float* delta) {
        delta[0] = fabs(residual[0]/jacobi[0]+residual[1]/jacobi[6]);
        delta[1] = fabs(residual[0]/jacobi[1]+residual[1]/jacobi[7]);
        delta[2] = fabs(residual[0]/jacobi[2]+residual[1]/jacobi[8]);
        delta[3] = fabs(residual[0]/jacobi[3]+residual[1]*jacobi[9]); // jacobi is 0
        delta[4] = fabs(residual[0]*jacobi[4]+residual[1]/jacobi[10]);
        delta[5] = fabs(residual[0]/jacobi[5]+residual[1]/jacobi[11]);

    }



    __global__ void computeCostAndJacobi(CUDAPtrs points, CUDAPtrs pixels, float4x4 T, float3x3 K, CUDAPtrc mask, CUDAPtrs costSummator, CUDAPtrs hSummator, CUDAPtrs mSummator, CUDAPtrs bSummator) {
        long index = threadIdx.x + blockIdx.x*blockDim.x;

        if(index>=points.getRows()) return;
        if(mask[index]) {
            float point[3]={points(index, 0), points(index, 1), points(index, 2)},
                    pixel[2] = {pixels(index, 0), pixels(index, 1)}
            , rePixel[2], transPoint[3], rotatePoint[3], proJacobi[6], hatMat[9], residual[2], jacobi[12];
            // copy point and pixel
            transformPoint(T, point, transPoint);

            // compute jacobi
            projectJacobi(K, transPoint, proJacobi);
            transformPointOnlyRotation(T, point, rotatePoint);
            hatMatrix(rotatePoint, hatMat);
            computeJacobi(proJacobi, hatMat, jacobi);

            projectPoint(K, transPoint, rePixel);
            // compute residual and cost
            residual[0] = rePixel[0] - pixel[0];
            residual[1] = rePixel[1] - pixel[1];

            float weight = computeHuberWeight(residual[0], residual[1], kHuberWeight);
            float cost = ComputeHuberCost(residual[0], residual[1], kHuberWeight);

            costSummator.data[index]=cost;
            // compute H,M,b
            float * H = hSummator.data+index*36;
            float * M = mSummator.data+index*6;
            float * b = bSummator.data+index*6;
            for(int i=0; i<6; i++) {
                for(int j=0; j<6; j++) {
                    H[j*6+i] = jacobi[i]*jacobi[j] + jacobi[i+6]*jacobi[j+6];
                }
                M[i] = weight*H[i*6+i];
                b[i] = -weight*(jacobi[i]*residual[0]+jacobi[i+6]*residual[1]);
            }
        }else {
            // compute H,M,b
            costSummator.data[index]=0;
            float * H = hSummator.data+index*36;
            float * M = mSummator.data+index*6;
            float * b = bSummator.data+index*6;
            for(int i=0; i<6; i++) {
                for(int j=0; j<6; j++) {
                    H[j*6+i] = 0;
                }
                M[i] = 0;
                b[i] = 0;
            }
        }

    }

    __global__ void computeCost(CUDAPtrs points, CUDAPtrs pixels, float4x4 T, float3x3 K, CUDAPtrc mask, CUDAPtrs costSummator) {
        long index = threadIdx.x + blockIdx.x*blockDim.x;

        if(index>=points.getRows()) return;

        if(mask[index]) {
            float point[3]={points(index, 0), points(index, 1), points(index, 2)},
                    pixel[2] = {pixels(index, 0), pixels(index, 1)};

            float rePixel[2], transPoint[3], residual[2];
            transformPoint(T, point, transPoint);
            projectPoint(K, transPoint, rePixel);
            // compute residual and cost
            residual[0] = rePixel[0] - pixel[0];
            residual[1] = rePixel[1] - pixel[1];

            float cost = ComputeHuberCost(residual[0], residual[1], kHuberWeight);

            costSummator.data[index]=cost;
        }else {
            costSummator.data[index] = 0;
        }
    }


    void computeBACostAndJacobi(CUDAMatrixs& objectPoints, CUDAMatrixs& tarPixels, float4x4& T, float3x3& K, CUDAMatrixc& mask, Summator& costSummator, Summator& hSummator, Summator& mSummator, Summator& bSummator) {
        long n = objectPoints.getRows();
        // invoke kernel
        CUDA_LINE_BLOCK(n);

        computeCostAndJacobi<<<grid, block, 0, stream>>>(objectPoints, tarPixels, T, K, mask, *costSummator.dataMat, *hSummator.dataMat, *mSummator.dataMat, *bSummator.dataMat);

        CUDA_CHECKED_NO_ERROR();
    }

    void computeBACost(CUDAMatrixs& objectPoints, CUDAMatrixs& tarPixels, float4x4& T, float3x3& K, CUDAMatrixc& mask, Summator& costSummator) {
        long n = objectPoints.getRows();
        // invoke kernel
        CUDA_LINE_BLOCK(n);

        computeCost<<<grid, block, 0, stream>>>(objectPoints, tarPixels, T, K, mask, *costSummator.dataMat);

        CUDA_CHECKED_NO_ERROR();
    }

    __global__ void computerInliersKernel(CUDAPtrs cost, CUDAPtrc inliers, float th) {
        long index = threadIdx.x + blockIdx.x*blockDim.x;

        if(index>=cost.getRows()) return;
        inliers.setIndex(index, cost[index]>0&&cost[index]<th);
    }

    void computerInliers(Summator& costSummator, CUDAMatrixc& inliers, float th) {
        // invoke kernel
        CUDA_LINE_BLOCK(costSummator.length);

        computerInliersKernel<<<grid, block, 0, stream>>>(*costSummator.dataMat, inliers, th);

        CUDA_CHECKED_NO_ERROR();
    }

    __device__ void composeJacobi(float* proJacobi, float* hat, float* jacobi) {
        // for t
        jacobi[0] = proJacobi[0];
        jacobi[1] = proJacobi[1];
        jacobi[2] = proJacobi[2];

        jacobi[6] = proJacobi[3];
        jacobi[7] = proJacobi[4];
        jacobi[8] = proJacobi[5];

        // for R
        jacobi[3] = -(proJacobi[0]*hat[0]+proJacobi[1]*hat[3]+proJacobi[2]*hat[6]);
        jacobi[4] = -(proJacobi[0]*hat[1]+proJacobi[1]*hat[4]+proJacobi[2]*hat[7]);
        jacobi[5] = -(proJacobi[0]*hat[2]+proJacobi[1]*hat[5]+proJacobi[2]*hat[8]);

        jacobi[9] = -(proJacobi[3]*hat[0]+proJacobi[4]*hat[3]+proJacobi[5]*hat[6]);
        jacobi[10] = -(proJacobi[3]*hat[1]+proJacobi[4]*hat[4]+proJacobi[5]*hat[7]);
        jacobi[11] = -(proJacobi[3]*hat[2]+proJacobi[4]*hat[5]+proJacobi[5]*hat[8]);
    }

    __device__ void computeHMb(CUDALMSummators summators, long index, float weight, float* jacobi, float* residual, float jacobiWeight) {
        float * H = summators.H.data+index*36;
        float * M = summators.M.data+index*6;
        float * b = summators.b.data+index*6;
        for(int i=0; i<6; i++) {
            for(int j=0; j<6; j++) {
                H[j*6+i] = jacobi[i]*jacobi[j] + jacobi[i+6]*jacobi[j+6];
            }
            M[i] = weight*H[i*6+i];
            b[i] = -weight*jacobiWeight*(jacobi[i]*residual[0]+jacobi[i+6]*residual[1]);
        }
    }


    __global__ void computeMVCostAndJacobiForEdge(CUDAEdge edge, CUDALMSummators summatorsX, CUDALMSummators summatorsY, CUDALMSummators deltaSummators, CUDAPtrs costSummator) {
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

        float point[3]={ky(index, 0), ky(index, 1), ky(index, 2)},
                pixel[2] = {kx(index, 0), kx(index, 1)};
        float rePixel[2], transPoint[3], proJacobi[6], hatMat[9], residual[2], jacobi[12];
        unproject(intrinsicY, point, transPoint);
        transformPoint(transform, transPoint, transPoint);

        // compute jacobi
        projectJacobi(intrinsicX, transPoint, proJacobi);
        hatMatrix(transPoint, hatMat);
        composeJacobi(proJacobi, hatMat, jacobi);

        projectPoint(intrinsicX, transPoint, rePixel);
        // compute residual and cost
        residual[0] = rePixel[0] - pixel[0];
        residual[1] = rePixel[1] - pixel[1];

        float weight = computeHuberWeight(residual[0], residual[1], kHuberWeight);
        float cost = ComputeHuberCost(residual[0], residual[1], kHuberWeight);

        costSummator.data[costIndex]=cost;
//        printf("%f\n", cost);
        // compute H,M,b
        computeHMb(summatorsX, sumIndexX, weight, jacobi, residual, 1.0);
        computeHMb(summatorsY, sumIndexY, weight, jacobi, residual, -1.0);
        computeHMb(deltaSummators, index, weight, jacobi, residual, 1.0);
    }

    __global__ void computeMVCostForEdge(CUDAEdge edge, CUDAPtrs costSummator) {
        // obtain parameters from
        long index = threadIdx.x + blockIdx.x*blockDim.x;

        long costIndex = edge.costIndex+index;
        CUDAPtrs kx = edge.kx;
        CUDAPtrs ky = edge.ky;
        float3x3 intrinsicX = edge.intrinsicX;
        float3x3 intrinsicY = edge.intrinsicY;
        float4x4 transform = edge.transform;

        if(index>=kx.getRows()) return;

        float point[3]={ky(index, 0), ky(index, 1), ky(index, 2)},
                pixel[2] = {kx(index, 0), kx(index, 1)};

        float rePixel[2], transPoint[3], residual[2];
        unproject(intrinsicY, point, transPoint);
        transformPoint(transform, transPoint, transPoint);

        projectPoint(intrinsicX, transPoint, rePixel);
        // compute residual and cost
        residual[0] = rePixel[0] - pixel[0];
        residual[1] = rePixel[1] - pixel[1];

        float cost = ComputeHuberCost(residual[0], residual[1], kHuberWeight);

        costSummator.data[costIndex]=cost;
    }


    void computeMVBACostAndJacobi(CUDAEdgeVector &edges, CUDAVector<CUDALMSummators>& gtSummators, CUDAVector<CUDALMSummators>& deltaSummators, Summator& costSummator) {
        for(long index=0; index<edges.getNum(); index++) {
            CUDA_LINE_BLOCK(edges[index].count);

            computeMVCostAndJacobiForEdge<<<grid, block, 0, stream>>>(edges[index], gtSummators[edges[index].indexX], gtSummators[edges[index].indexY], deltaSummators[index], *costSummator.dataMat);

            CUDA_CHECKED_NO_ERROR();
        }
    }

    void computeMVBACost(CUDAEdgeVector &edges, Summator& costSummator) {
        for(long index=0; index<edges.getNum(); index++) {
            CUDA_LINE_BLOCK(edges[index].count);

            computeMVCostForEdge<<<grid, block, 0, stream>>>(edges[index], *costSummator.dataMat);

            CUDA_CHECKED_NO_ERROR();
        }
    }

}
