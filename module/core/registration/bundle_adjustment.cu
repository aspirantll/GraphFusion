//
// Created by liulei on 2020/6/12.
//

#include "bundle_adjustment.cuh"

namespace rtf {
    constexpr Scalar kHuberWeight = 1.2;
    __device__ Scalar computeHuberWeight(Scalar residual_x, Scalar residual_y, Scalar huber_parameter) {
        Scalar squared_residual = residual_x * residual_x + residual_y * residual_y;
        return (squared_residual < huber_parameter * huber_parameter) ? 1 : (huber_parameter / sqrtf(squared_residual));
    }

    __device__ Scalar ComputeHuberCost(Scalar residual_x, Scalar residual_y, Scalar huber_parameter) {
        Scalar squared_residual = residual_x * residual_x + residual_y * residual_y;
        if (squared_residual < huber_parameter * huber_parameter) {
            return 0.5 * squared_residual;
        } else {
            return huber_parameter * (sqrtf(squared_residual) - 0.5 * huber_parameter);
        }
    }

    __device__ void projectJacobi(float3x3 k, Scalar* point, Scalar* proJacobi) {
        Scalar x = k(0,0)*point[0]+k(0,1)*point[1]+k(0,2)*point[2];
        Scalar y = k(1,0)*point[0]+k(1,1)*point[1]+k(1,2)*point[2];
        Scalar z = k(2,0)*point[0]+k(2,1)*point[1]+k(2,2)*point[2];

        proJacobi[0] = z*k(0,0)-x*k(2,0)/(z*z);
        proJacobi[1] = z*k(0,1)-x*k(2,1)/(z*z);
        proJacobi[2] = z*k(0,2)-x*k(2,2)/(z*z);
        proJacobi[3] = z*k(1,0)-y*k(2,0)/(z*z);
        proJacobi[4] = z*k(1,1)-y*k(2,1)/(z*z);
        proJacobi[5] = z*k(1,2)-y*k(2,2)/(z*z);
    }

    __device__ void transformPointOnlyRotation(float4x4 T, Scalar* point, Scalar *tPoint) {
        Scalar x = T(0,0)*point[0] + T(0,1)*point[1] + T(0,2)*point[2];
        Scalar y = T(1,0)*point[0] + T(1,1)*point[1] + T(1,2)*point[2];
        Scalar z = T(2,0)*point[0] + T(2,1)*point[1] + T(2,2)*point[2];

        tPoint[0] = x;
        tPoint[1] = y;
        tPoint[2] = z;
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

    __device__ void computeJacobi(Scalar* proJacobi, Scalar* hat, Scalar* jacobi) {
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


    __device__ void computeDeltaLie(Scalar *jacobi, Scalar *residual, Scalar* delta) {
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
            Scalar point[3]={points(index, 0), points(index, 1), points(index, 2)},
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

            Scalar weight = computeHuberWeight(residual[0], residual[1], kHuberWeight);
            Scalar cost = ComputeHuberCost(residual[0], residual[1], kHuberWeight);

            costSummator.data[index]=cost;
            // compute H,M,b
            Scalar * H = hSummator.data+index*36;
            Scalar * M = mSummator.data+index*6;
            Scalar * b = bSummator.data+index*6;
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
            Scalar * H = hSummator.data+index*36;
            Scalar * M = mSummator.data+index*6;
            Scalar * b = bSummator.data+index*6;
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
            Scalar point[3]={points(index, 0), points(index, 1), points(index, 2)},
                    pixel[2] = {pixels(index, 0), pixels(index, 1)};

            Scalar rePixel[2], transPoint[3], residual[2];
            transformPoint(T, point, transPoint);
            projectPoint(K, transPoint, rePixel);
            // compute residual and cost
            residual[0] = rePixel[0] - pixel[0];
            residual[1] = rePixel[1] - pixel[1];

            Scalar cost = ComputeHuberCost(residual[0], residual[1], kHuberWeight);

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

    __global__ void computerInliersKernel(CUDAPtrs cost, CUDAPtrc inliers, Scalar th) {
        long index = threadIdx.x + blockIdx.x*blockDim.x;

        if(index>=cost.getRows()) return;
        inliers.setIndex(index, cost[index]>0&&cost[index]<th);
    }

    void computerInliers(Summator& costSummator, CUDAMatrixc& inliers, Scalar th) {
        // invoke kernel
        CUDA_LINE_BLOCK(costSummator.length);

        computerInliersKernel<<<grid, block, 0, stream>>>(*costSummator.dataMat, inliers, th);

        CUDA_CHECKED_NO_ERROR();
    }

    __device__ void composeJacobi(Scalar* proJacobi, Scalar* hat, Scalar* jacobi) {
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

    __device__ void computeHMb(Scalar* H, Scalar* M, Scalar* b, Scalar weight, Scalar* jacobi, Scalar* residual) {
        for(int i=0; i<6; i++) {
            for(int j=0; j<6; j++) {
                H[j*6+i] = jacobi[i]*jacobi[j] + jacobi[i+6]*jacobi[j+6];
            }
            M[i] = weight*H[i*6+i];
            b[i] = -weight*(jacobi[i]*residual[0]+jacobi[i+6]*residual[1]);
        }
    }


    __global__ void computeMVCostAndJacobiForEdge(CUDAEdge edge, CUDAPtrs H, CUDAPtrs M, CUDAPtrs b, Scalar* cost) {
        // obtain parameters from
        long index = threadIdx.x + blockIdx.x*blockDim.x;

        CUDAPtrs kx = edge.kx;
        CUDAPtrs ky = edge.ky;
        const float3x3 intrinsicX = edge.intrinsicX;
        const float3x3 intrinsicY = edge.intrinsicY;
        const float4x4 transform = edge.transform;
        const int x = edge.indexX;
        const int y = edge.indexY;
        const int z = edge.indexZ;
        const int n = H.rows;

        if(index>=kx.getRows()) return;

        Scalar point[3]={ky(index, 0), ky(index, 1), ky(index, 2)},
                pixel[2] = {kx(index, 0), kx(index, 1)};
        Scalar rePixel[2], transPoint[3], proJacobi[6], hatMat[9], residual[2], jacobi[12];
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

        Scalar weight = computeHuberWeight(residual[0], residual[1], kHuberWeight);
        Scalar huberCost = ComputeHuberCost(residual[0], residual[1], kHuberWeight);

        // compute H,M,b
        Scalar tH[36], tM[6], tb[6];
        computeHMb(tH, tM, tb, weight, jacobi, residual);

        for(int i=0; i<6; i++) {
            for(int j=0; j<6; j++) {
                Scalar value = tH[6*j+i];
                atomicAdd(&H.data[(6*x+j)*n+i+6*x], value);
                atomicAdd(&H.data[(6*x+j)*n+i+6*y], -value);
                atomicAdd(&H.data[(6*x+j)*n+i+6*z], -value);

                atomicAdd(&H.data[(6*y+j)*n+i+6*x], -value);
                atomicAdd(&H.data[(6*y+j)*n+i+6*y], value);
                atomicAdd(&H.data[(6*y+j)*n+i+6*z], value);

                atomicAdd(&H.data[(6*z+j)*n+i+6*x], -value);
                atomicAdd(&H.data[(6*z+j)*n+i+6*y], value);
                atomicAdd(&H.data[(6*z+j)*n+i+6*z], value);
            }
            atomicAdd(&M.data[6*x+i], -tM[i]);
            atomicAdd(&M.data[6*y+i], tM[i]);
            atomicAdd(&M.data[6*z+i], tM[i]);

            atomicAdd(&b.data[6*x+i], -tb[i]);
            atomicAdd(&b.data[6*y+i], tb[i]);
            atomicAdd(&b.data[6*z+i], tb[i]);
        }

        atomicAdd(cost, huberCost);
    }

    __global__ void computeMVCostForEdge(CUDAEdge edge, Scalar* cost) {
        // obtain parameters from
        long index = threadIdx.x + blockIdx.x*blockDim.x;

        CUDAPtrs kx = edge.kx;
        CUDAPtrs ky = edge.ky;
        float3x3 intrinsicX = edge.intrinsicX;
        float3x3 intrinsicY = edge.intrinsicY;
        float4x4 transform = edge.transform;

        if(index>=kx.getRows()) return;

        Scalar point[3]={ky(index, 0), ky(index, 1), ky(index, 2)},
                pixel[2] = {kx(index, 0), kx(index, 1)};

        Scalar rePixel[2], transPoint[3], residual[2];
        unproject(intrinsicY, point, transPoint);
        transformPoint(transform, transPoint, transPoint);

        projectPoint(intrinsicX, transPoint, rePixel);
        // compute residual and cost
        residual[0] = rePixel[0] - pixel[0];
        residual[1] = rePixel[1] - pixel[1];

        atomicAdd(cost, ComputeHuberCost(residual[0], residual[1], kHuberWeight));
    }


    void computeMVBACostAndJacobi(CUDAEdgeVector &edges, LMSumMats& sumMats) {
        for(long index=0; index<edges.getNum(); index++) {
            CUDA_LINE_BLOCK(edges[index].kx.rows);

            computeMVCostAndJacobiForEdge<<<grid, block, 0, stream>>>(edges[index], *sumMats.cH, *sumMats.cM, *sumMats.cb, sumMats.cCost);

            CUDA_CHECKED_NO_ERROR();
        }
    }

    void computeMVBACost(CUDAEdgeVector &edges, Scalar& cost) {
        cost = 0;
        Scalar *cCost;
        CUDA_CHECKED_CALL(cudaMalloc(&cCost, sizeof(Scalar)));
        CUDA_CHECKED_CALL(cudaMemcpy(cCost, &cost, sizeof(Scalar), cudaMemcpyHostToDevice));
        for(long index=0; index<edges.getNum(); index++) {
            CUDA_LINE_BLOCK(edges[index].kx.rows);

            computeMVCostForEdge<<<grid, block, 0, stream>>>(edges[index], cCost);

            CUDA_CHECKED_NO_ERROR();
        }
        CUDA_CHECKED_CALL(cudaMemcpy(&cost, cCost, sizeof(Scalar), cudaMemcpyDeviceToHost));
        CUDA_CHECKED_CALL(cudaFree(cCost));
    }

}
