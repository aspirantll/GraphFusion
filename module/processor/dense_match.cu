//
// Created by liulei on 2020/10/12.
//
#include "dense_match.cuh"

namespace rtf {

    // rgb, depth and normals
    texture<uchar, cudaTextureType2D, cudaReadModeElementType> grayTextureRef1;
    texture<float, cudaTextureType2D, cudaReadModeElementType> depthTextureRef1;
    texture<float4, cudaTextureType2D, cudaReadModeElementType> normalTextureRef1;

    texture<uchar, cudaTextureType2D, cudaReadModeElementType> grayTextureRef2;
    texture<float, cudaTextureType2D, cudaReadModeElementType> depthTextureRef2;
    texture<float4, cudaTextureType2D, cudaReadModeElementType> normalTextureRef2;

    __constant__ DenseMatchParams denseMatchParams;

    void updateDenseMatchParams(const DenseMatchParams& params) {
        size_t size;
        CUDA_CHECKED_CALL(cudaGetSymbolSize(&size, denseMatchParams));
        CUDA_CHECKED_CALL(cudaMemcpyToSymbolAsync(denseMatchParams, &params, size, 0, cudaMemcpyHostToDevice, stream));
        CUDA_CHECKED_CALL(cudaStreamSynchronize(stream));

        denseMatchParams.width = params.width;
        denseMatchParams.height = params.height;
    }

    void bindTextureParams(uchar* grayImg1, float* depthImg1, float4* normalImg1
                           , uchar* grayImg2, float* depthImg2, float4* normalImg2, int width, int height) {
        cutilSafeCall(cudaBindTexture2D(0, &grayTextureRef1, grayImg1, &grayTextureRef1.channelDesc, width, height, sizeof(uchar) * width));
        cutilSafeCall(cudaBindTexture2D(0, &depthTextureRef1, depthImg1, &depthTextureRef1.channelDesc, width, height, sizeof(float)*width));
        cutilSafeCall(cudaBindTexture2D(0, &normalTextureRef1, normalImg1, &normalTextureRef1.channelDesc, width, height, sizeof(float4)*width));

        cutilSafeCall(cudaBindTexture2D(0, &grayTextureRef2, grayImg2, &grayTextureRef2.channelDesc, width, height, sizeof(uchar) * width));
        cutilSafeCall(cudaBindTexture2D(0, &depthTextureRef2, depthImg2, &depthTextureRef2.channelDesc, width, height, sizeof(float)*width));
        cutilSafeCall(cudaBindTexture2D(0, &normalTextureRef2, normalImg2, &normalTextureRef2.channelDesc, width, height, sizeof(float4)*width));

        grayTextureRef1.filterMode = cudaFilterModePoint;
        grayTextureRef1.addressMode[0] = cudaAddressModeBorder;
        grayTextureRef1.addressMode[1] = cudaAddressModeBorder;
        grayTextureRef1.addressMode[2] = cudaAddressModeBorder;

        depthTextureRef1.filterMode = cudaFilterModePoint;
        normalTextureRef1.filterMode = cudaFilterModePoint;

        grayTextureRef2.filterMode = cudaFilterModePoint;
        grayTextureRef2.addressMode[0] = cudaAddressModeBorder;
        grayTextureRef2.addressMode[1] = cudaAddressModeBorder;
        grayTextureRef2.addressMode[2] = cudaAddressModeBorder;

        depthTextureRef2.filterMode = cudaFilterModePoint;
        normalTextureRef2.filterMode = cudaFilterModePoint;
    }

    __device__ inline float computeBilateralWeight(const float row_diff, const float col_diff,
                                    const float color1,
                                    const float color2) {

        const float spatial_normalization_ = 1.0f / (2.0f * denseMatchParams.sigmaSpatial * denseMatchParams.sigmaSpatial);
        const float color_normalization_ = 1.0f / (2.0f * denseMatchParams.sigmaColor * denseMatchParams.sigmaColor);
        const float spatial_dist_squared =
                row_diff * row_diff + col_diff * col_diff;
        const float color_dist = color1 - color2;
        return exp(-spatial_dist_squared * spatial_normalization_ -
                   color_dist * color_dist * color_normalization_);
    }

    __device__ inline float computeNCC(int x1, int y1, int x2, int y2, int radius) {
        const int width = denseMatchParams.width, height = denseMatchParams.height;
        if(x1<0||x1>=width||y1<0||y1>=height||x2<0||x2>=width||y2<0||y2>=height) return -1;
        float colorSum1=0, colorSquaredSum1=0;
        float colorSum2=0, colorSquaredSum2=0;
        float colorSum12=0, bilateralWeightSum1 = 0, bilateralWeightSum2=0;
        float centerColor1 = tex2D(grayTextureRef1, x1, y1);
        float centerColor2 = tex2D(grayTextureRef2, x2, y2);

        for (int row = -radius; row <= radius; row += 1) {
            for (int col = -radius; col <= radius; col += 1) {
                const float color1 = tex2D(grayTextureRef1, x1 + row, y1 + col);
                const float color2 = tex2D(grayTextureRef2, x2 + row, y2 + col);

                const float bilateralWeight1 = computeBilateralWeight(
                        row, col, centerColor1, color1);
                const float bilateralWeight2 = computeBilateralWeight(
                        row, col, centerColor2, color2);

                // sum
                colorSum1 += bilateralWeight1 * color1;
                colorSum2 += bilateralWeight2 * color2;
                colorSquaredSum1 += bilateralWeight1 * color1 * color1;
                colorSquaredSum2 += bilateralWeight2 * color2 * color2;
                colorSum12 += bilateralWeight1*color1*color2;
                bilateralWeightSum1 += bilateralWeight1;
                bilateralWeightSum2 += bilateralWeight2;
            }
        }

        colorSum1 /= bilateralWeightSum1;
        colorSum2 /= bilateralWeightSum2;
        colorSquaredSum1 /= bilateralWeightSum1;
        colorSquaredSum2 /= bilateralWeightSum2;
        colorSum12 /= bilateralWeightSum1;

        const float colorVar1 = colorSquaredSum1 - colorSum1*colorSum1;
        const float colorVar2 = colorSquaredSum2 - colorSum2*colorSum2;

        // Based on Jensen's Inequality for convex functions, the variance
        // should always be larger than 0. Do not make this threshold smaller.
        const float kMinVar = 1e-5f;
        if(colorVar1<kMinVar||colorVar2<kMinVar) {
            return -1;
        }else {
            const float colorCovar = colorSum12-colorSum1*colorSum2;
            const float colorVar12 = sqrt(colorVar1*colorVar2);
            return max(-1.0f, colorCovar/colorVar12);
        }
    }

    /**
     *
     * @param matchIndexes
     * @param trans
     */
    __global__ void leftDenseMatchKernel(CUDAPtrl matchIndexes, CUDAPtrs matchScores) {
        int x = threadIdx.x + blockIdx.x*blockDim.x;
        int y = threadIdx.y + blockIdx.y*blockDim.y;
        const int windowRadius = denseMatchParams.windowRadius;
        const int neigh = denseMatchParams.neigh;

        const int width = denseMatchParams.width, height = denseMatchParams.height;
        if(x>=width||y>=height) return;
        // transform point to another camera coordinate system
        int transX, transY;
        {
            float depth = tex2D(depthTextureRef1, x, y);
            float3 point = denseMatchParams.invK*make_float3(x, y, 1)*depth;
            float3 transHPixel = denseMatchParams.K*make_float3(denseMatchParams.trans*make_float4(point, 1));
            transX = int(transHPixel.x/transHPixel.z+0.5);
            transY = int(transHPixel.y/transHPixel.z+0.5);

            if(transX>=width||transY>=height) {
                matchIndexes.set(x, y, -1);
                return;
            };
        }

        // compare normal
        float3 normal = make_float3(tex2D(normalTextureRef1, x, y));
        float3 transNormal = denseMatchParams.trans*normal;
        const float normalTh = denseMatchParams.deltaNormalTh;
        float bestNCC = -1;
        int bestX = -1, bestY = -1;
        for(int i=-neigh; i<=neigh; i++) {
            for (int j = -neigh; j <= neigh; j++) {
                float3 aNormal = make_float3(tex2D(normalTextureRef2, transX+i, transY+j));
                if(1-dot(transNormal, aNormal)<normalTh) {
                    float curNCC = computeNCC(x, y, transX+i, transY+j, windowRadius);
                    if(curNCC > bestNCC) {
                        bestX = transX+i;
                        bestY = transY+j;
                        bestNCC = curNCC;
                    }
                }
            }
        }

        if(bestX>=0&&bestY>=0) {
            matchIndexes.set(x, y, matchIndexes.convert1DIndex(bestX, bestY));
            matchScores.set(x, y, bestNCC);
        }else {
            matchIndexes.set(x, y, -1);
            matchScores.set(x, y, -1);
        }
    }


    __global__ void rightDenseMatchKernel(CUDAPtrl matchIndexes) {
        int x = threadIdx.x + blockIdx.x*blockDim.x;
        int y = threadIdx.y + blockIdx.y*blockDim.y;
        const int windowRadius = denseMatchParams.windowRadius;
        const int neigh = denseMatchParams.neigh;

        const int width = denseMatchParams.width, height = denseMatchParams.height;
        if(x>=width||y>=height) return;

        // transform point to another camera coordinate system
        int transX, transY;
        {
            float depth = tex2D(depthTextureRef2, x, y);
            float3 point = denseMatchParams.invK*make_float3(x, y, 1)*depth;
            float3 transHPixel = denseMatchParams.K*make_float3(denseMatchParams.trans*make_float4(point, 1));
            transX = int(transHPixel.x/transHPixel.z+0.5);
            transY = int(transHPixel.y/transHPixel.z+0.5);
        }

        // compare normal
        float3 normal = make_float3(tex2D(normalTextureRef2, x, y));
        float3 transNormal = denseMatchParams.trans*normal;
        const float normalTh = denseMatchParams.deltaNormalTh;
        float bestNCC = -1;
        int bestX = -1, bestY = -1;
        for(int i=-neigh; i<=neigh; i++) {
            for (int j = -neigh; j <= neigh; j++) {
                float3 aNormal = make_float3(tex2D(normalTextureRef1, transX+i, transY+j));
                if(1-dot(transNormal, aNormal)<normalTh) {
                    float curNCC = computeNCC(transX+i, transY+j, x, y, windowRadius);
                    if(curNCC > bestNCC) {
                        bestX = transX+i;
                        bestY = transY+j;
                        bestNCC = curNCC;
                    }
                }
            }
        }

        if(bestX>=0&&bestY>=0) {
            matchIndexes.set(x, y, matchIndexes.convert1DIndex(bestX, bestY));
        }else {
            matchIndexes.set(x, y, -1);
        }
    }

    void leftDenseMatch(CUDAMatrixl& matches, CUDAMatrixs& matchScores) {
        const int width = denseMatchParams.width, height = denseMatchParams.height;
        CUDA_MAT_BLOCK(width, height);
        leftDenseMatchKernel<<<grid, block, 0, stream>>>(matches, matchScores);
        CUDA_CHECKED_NO_ERROR();
    }

    void rightDenseMatch(CUDAMatrixl& matches) {
        const int width = denseMatchParams.width, height = denseMatchParams.height;
        CUDA_MAT_BLOCK(width, height);
        rightDenseMatchKernel<<<grid, block, 0, stream>>>(matches);
        CUDA_CHECKED_NO_ERROR();
    }

    __global__ void crossCheckKernel(CUDAPtrl left, CUDAPtrl right, CUDAPtrs scores, CUDAPtrc mask) {
        const int tidX = threadIdx.x, tidY = threadIdx.y;
        int x = tidX + blockIdx.x*blockDim.x;
        int y = tidY + blockIdx.y*blockDim.y;

        const int width = denseMatchParams.width, height = denseMatchParams.height;
        if(x>=width||y>=height) return;

        uchar maskBit = 0;
        long curIndex = left.convert1DIndex(x, y);
        long matchIndex = left(x, y);
        if(matchIndex!=-1) {
            long matchX, matchY;
            right.convert2DIndex(matchIndex, &matchX, &matchY);
            if(matchX<0||matchX>=width||matchY<0||matchY>=height) {
                maskBit = 0;
            }else {
                maskBit = curIndex==right(matchX, matchY);
            }
        }

        __shared__ uchar maskBlock[BLOCK_X*BLOCK_Y];
        __shared__ float scoreBlock[BLOCK_X*BLOCK_Y];
        maskBlock[tidX*BLOCK_Y+tidY] = maskBit;
        scoreBlock[tidX*BLOCK_Y+tidY] = scores(x, y);
        __syncthreads();
        const int scale = denseMatchParams.downSampleScale;
        if(tidX%scale==0&&tidY%scale==0) {
            int bestI=-1, bestJ=-1;
            float bestScore = -1;
            for(int i=tidX; i<tidX+scale; i++) {
                for(int j=tidY; j<tidY+scale; j++) {
                    if(maskBlock[i*BLOCK_Y+j]) {
                        float curScore = scoreBlock[i*BLOCK_Y+j];
                        if(curScore>bestScore) {
                            bestI=i;
                            bestJ=j;
                            bestScore = curScore;
                        }
                    }
                }
            }
            for(int i=tidX; i<tidX+scale; i++) {
                for (int j = tidY; j < tidY + scale; j++) {
                    maskBlock[i*BLOCK_Y+j] = bestI==i&&bestJ==j;
                }
            }
        }
        __syncthreads();
        mask.set(x, y, maskBlock[tidX*BLOCK_Y+tidY]);
    }

    void crossCheck(CUDAMatrixl& leftMatches, CUDAMatrixl& rightMatches, CUDAMatrixs& matchScore, CUDAMatrixc& mask) {
        const int width = denseMatchParams.width, height = denseMatchParams.height;
        CUDA_MAT_BLOCK(width, height);
        crossCheckKernel<<<grid, block, 0, stream>>>(leftMatches, rightMatches, matchScore, mask);
        CUDA_CHECKED_NO_ERROR();
    }
}

