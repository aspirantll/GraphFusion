//
// Created by liulei on 2020/10/15.
//
#include "visual_index_hashing.cuh"

namespace rtf {


    __global__ void wordsCountKernel(CUDAPtrArray<CUDABoW> voc, CUDAPtru cur, CUDAPtru counts) {
        const int tid = threadIdx.x;
        const int index = threadIdx.x + blockIdx.x*blockDim.x;
        if(index>=voc.getNum()) return;

        CUDABoW bow = voc[index];
        const unsigned int imageId = bow.imageId;
        CUDAPtru ref = bow.words;

        const int curLength = cur.size();
        const int refLength = ref.size();
        // copy cur to shared memory
        extern __shared__ unsigned int curWords[];
        for(int i=tid; i<curLength; i+=blockDim.x) {
            curWords[i] = cur[i];
        }
        __syncthreads();
        unsigned int count = 0;
        int i=0, j=0;
        while(i<curLength&&j<refLength) {
            unsigned int curWordId = curWords[i];
            unsigned int refWordId = ref[j];
            if(curWordId==refWordId) {
                i++;
                j++;
                count++;
            }else if(curWordId>refWordId) {
                j++;
            }else {
                i++;
            }
        }

        counts.setIndex(index, count);
    }


    void wordsCount(CUDAPtrArray<CUDABoW>& voc, CUDAArrayu& cur, CUDAArrayu& wordCounts) {
        int length = voc.getNum();
        CUDA_LINE_BLOCK(length);
        wordsCountKernel<<<grid, block, sizeof(unsigned int)*cur.getSizes(), stream>>>(voc, cur, wordCounts);
        CUDA_CHECKED_NO_ERROR();
    }
}
