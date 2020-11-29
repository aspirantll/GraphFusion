//
// Created by liulei on 2020/7/15.
//

#include "map_reduce.h"

namespace rtf {
    __global__ void reduceSum(CudaScalar* data, long len, CudaScalar* result, long elementSize) {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int n = bid*blockDim.x+tid;
        extern __shared__ CudaScalar threadData[];
        if(n<len) {
            for(int i=0; i<elementSize; i++) {
                threadData[tid*elementSize+i] = data[n*elementSize+i];
            }
        }else {
            for(int i=0; i<elementSize; i++) {
                threadData[tid*elementSize+i] = 0;
            }
        }
        __syncthreads();

        for (int offset=blockDim.x>>1; offset>0; offset>>=1) {
            if(tid<offset) {
                for(int i=0; i<elementSize; i++) {
                    threadData[tid*elementSize+i] += threadData[(tid+offset)*elementSize+i];
                }
            }
            __syncthreads();
        }

        if(tid==0) {
            for(int i=0; i<elementSize; i++) {
                result[bid*elementSize+i] = threadData[i];
            }
        }
    }

    Summator::Summator(long length, long rows, long cols): length(length), rows(rows), cols(cols) {
        long totalNum = length;
        // alloc cuda memery
        dataMat = make_shared<CUDAArrays>(totalNum * rows * cols);
    }

    Summator::~Summator() {

    }


    MatrixX Summator::sum(long length, long rows, long cols) {
        length = length<=0?this->length:length;
        rows = rows<=0?this->rows:rows;
        cols = cols<=0?this->cols:cols;

        long gridNum = length;
        int len;
        do {
            len = gridNum;
            gridNum = (gridNum+blockDim-1)/blockDim;
            dim3 block(blockDim);
            dim3 grid(gridNum);
            reduceSum<<<grid,block,sizeof(CudaScalar)*blockDim*rows*cols>>>(dataMat->data, len, dataMat->data, rows*cols);
            CUDA_CHECKED_NO_ERROR();
        } while (gridNum>1);

        Eigen::MatrixXf result(rows, cols);
        dataMat->download(result.data(), 0, rows*cols);
        return result.cast<Scalar>();
    }


    MatrixX Summator::sum(vector<MatrixX>& elements) {
        long rows=elements[0].rows(), cols=elements[0].cols(), elementSize = rows*cols;
        long n = elements.size();
        CHECK_LE(n*elementSize, length*rows*cols);


        // upload data
        long offset = 0;
        for(auto element: elements) {
            Eigen::MatrixXf elementf = element.cast<CudaScalar>();
            dataMat->upload(elementf.data(), offset, elementSize);
            offset += elementSize;
        }

        return sum(n, rows, cols);
    }


    MatrixX Summator::sum(Eigen::MatrixXf elements, int wise) {
        long rows=elements.rows(), cols=elements.cols(), length;

        Eigen::MatrixXf mat = elements;

        switch (wise) {
            case 0:
                length = cols;
                cols = 1;
                break;
            case 1:
                mat = elements.transpose();
                length = rows;
                rows = 1;
                break;
            default:
                length = rows*cols;
                rows = 1;
                cols = 1;
        }
        dataMat->upload(elements.data(), 0, length*rows*cols);
        return sum(length, rows, cols);
    }

    MatrixX Summator::sum(CUDAMatrixs elements, int wise) {
        Eigen::MatrixXf data;
        elements.download(data);
        return sum(data, wise);
    }



}