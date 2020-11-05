//
// Created by liulei on 2020/9/9.
//

#include "sift_feature.cuh"
#include <device_functions.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../datastructure/cuda_types.h"

typedef unsigned int uint;
typedef unsigned char uchar;

namespace rtf {

    __global__ void computeMatchScoreKernel(CUDAPtr<uchar> refDesc, CUDAPtr<uchar> curDesc, CUDAPtr<uint> refInd
                                            , CUDAPtr<uint> curInd, CUDAPtr<CUDAScore> scores) {
        long x = blockIdx.x*blockDim.x + threadIdx.x;
        long y = blockIdx.y*blockDim.y + threadIdx.y;
        long rows = refInd.rows;
        long cols = curInd.rows;

        uint rowX = refInd[x];
        uint rowY = curInd[x];

        if(x>=rows || y>=cols) return;
        float score = 0;
        for(int j=0; j<refDesc.cols; j++) {
            score += refDesc(rowX,j) * curDesc(rowY, j);
        }
        CUDAScore cudaScore;
        cudaScore.score = score;
        cudaScore.refIndex = x;
        cudaScore.curIndex = y;

        scores.set(x, y, cudaScore);
    }

    void computeMatchScore(CUDAFeatures& refDesc, CUDAFeatures& curDesc, CUDAFeatureIndexes& refInd
            , CUDAFeatureIndexes& curInd, CUDAMatrix<CUDAScore>& scores) {
        long rows = refInd.getSizes();
        long cols = curInd.getSizes();

        CUDA_MAT_BLOCK(rows, cols);
        computeMatchScoreKernel<<<grid, block, 0, stream>>>(refDesc, curDesc, refInd, curInd, scores);
        CUDA_CHECKED_NO_ERROR();
    }


    __global__ void filteredRowBestMatchesKernel(CUDAPtr<CUDAScore> scores, int n, float distMax, float ratioMax, CUDAPtr<int> bestMatches) {
        long row = blockIdx.x;
        __shared__ float maxScore[BLOCK_X*2];
        __shared__ float subMaxScore[BLOCK_X*2];
        __shared__ int maxInd[BLOCK_X*2];

        float dotMax = 0, subMax = 0;
        int rowInd=-1, colInd=-1;
        for(int j=threadIdx.x; j<n; j+=threadIdx.x) {
            CUDAScore s = scores[row*n+j];
            if(rowInd!=-1&&rowInd!=s.refIndex) {
                printf("error rowId:%d\n", s.refIndex);
            }
            if(s.score > dotMax) {
                subMax = dotMax;
                dotMax = s.score;
                rowInd = s.refIndex;
                colInd = s.curIndex;
            }else {
                subMax = max(subMax, s.score);
            }
        }

        maxScore[threadIdx.x] = dotMax;
        subMaxScore[threadIdx.x] = subMax;
        maxInd[threadIdx.x] = colInd;

        __syncthreads();

        if(threadIdx.x == 0) {
            for(int i=1; i<blockIdx.x; i++) {
                if(maxScore[i] > dotMax) {
                    subMax = dotMax;
                    dotMax = maxScore[i];
                    colInd = maxInd[i];
                }else {
                    subMax = max(subMax, maxScore[i]);
                }
            }
            float dist =  acos(min(dotMax * 0.000003814697265625f, 1.0f));
            float distN = acos(min(subMax * 0.000003814697265625f, 1.0f));
            if(bestMatches[rowInd]!=-1) {
                printf("warning: repeat to match for row %d\n", rowInd);
            }
            bestMatches.data[rowInd] = (dist < distMax) && (dist < distN * ratioMax) ? colInd : -1;
        }
    }

    void filteredRowBestMatches(CUDAMatchScores& scores, int m, int n,  float distMax, float ratioMax, CUDAMatchIndexes& bestMatches) {
        dim3 grid(m);
        dim3 block(BLOCK_X*2);
        filteredRowBestMatchesKernel<<<grid,block, 0, stream>>>(scores, n, distMax, ratioMax, bestMatches);
        CUDA_CHECKED_NO_ERROR();
    }


    __global__ void filteredColBestMatchesKernel(CUDAPtr<CUDAScore> scores, int m, float distMax, float ratioMax,  CUDAPtr<int> bestMatches) {
        long col = blockIdx.x;
        __shared__ float maxScore[BLOCK_X*2];
        __shared__ float subMaxScore[BLOCK_X*2];
        __shared__ int maxInd[BLOCK_X*2];

        float dotMax = 0, subMax = 0;
        int rowInd=-1, colInd=-1;
        for(int i=threadIdx.x; i < m; i+=threadIdx.x) {
            CUDAScore s = scores(i, col);
            if(colInd!=-1&&colInd!=s.curIndex) {
                printf("error colId:%d\n", s.curIndex);
            }
            if(s.score > dotMax) {
                subMax = dotMax;
                dotMax = s.score;
                rowInd = s.refIndex;
                colInd = s.curIndex;
            }else {
                subMax = max(subMax, s.score);
            }
        }

        maxScore[threadIdx.x] = dotMax;
        subMaxScore[threadIdx.x] = subMax;
        maxInd[threadIdx.x] = rowInd;

        __syncthreads();

        if(threadIdx.x == 0) {
            for(int i=1; i<blockIdx.x; i++) {
                if(maxScore[i] > dotMax) {
                    subMax = dotMax;
                    dotMax = maxScore[i];
                    rowInd = maxInd[i];
                }else {
                    subMax = max(subMax, maxScore[i]);
                }
            }
            float dist =  acos(min(dotMax * 0.000003814697265625f, 1.0f));
            float distN = acos(min(subMax * 0.000003814697265625f, 1.0f));
            if(bestMatches[colInd]!=-1) {
                printf("warning: repeat to match for col %d\n", colInd);
            }
            bestMatches.data[colInd] = (dist < distMax) && (dist < distN * ratioMax) ? rowInd : -1;
        }
    }

    void filteredColBestMatches(CUDAMatchScores& scores, int m, int n, float distMax, float ratioMax, CUDAMatchIndexes& bestMatches) {
        dim3 grid(n);
        dim3 block(BLOCK_X*2);
        filteredColBestMatchesKernel<<<grid,block, 0, stream>>>(scores, m, distMax, ratioMax, bestMatches);
        CUDA_CHECKED_NO_ERROR();
    }


    void findBestMatches(CUDAFeatures& refDesc, CUDAFeatures& curDesc, CUDAFeatureIndexes& refInd
            , CUDAFeatureIndexes& curInd, float distMax, float ratioMax, CUDAMatchIndexes& rowMatch, CUDAMatchIndexes& colMatch) {
        long rows = refInd.size;
        long cols = curInd.size;

        CUDAMatchScores scores(rows, cols);
        computeMatchScore(refDesc, curDesc, refInd, curInd, scores);
        filteredRowBestMatches(scores, rows, cols, distMax, ratioMax, rowMatch);
        filteredColBestMatches(scores, rows, cols, distMax, ratioMax,  colMatch);
    }
}

