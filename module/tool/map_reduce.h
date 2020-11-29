//
// Created by liulei on 2020/7/15.
//

#ifndef GraphFusion_MAP_REDUCE_H
#define GraphFusion_MAP_REDUCE_H

#include <initializer_list>
#include <Eigen/Core>

#include "../datastructure/cuda_types.h"

using namespace std;

typedef void (*func_t)(rtf::CUDAArrays, long start, long len, long size);

namespace rtf {
    class Summator {
    public:
        long length;
        long rows, cols;
        const long blockDim = BLOCK_X*BLOCK_Y;

        shared_ptr<CUDAArrays> dataMat;
        /**
         * constructor for summator
         * @param elementNum
         * @param stepRange
         */
        Summator(long maxElementNum, long rows, long cols);

        /**
         * destory the object
         */
        ~Summator();

        /**
         * @param length
         * @param rows
         * @param cols
         * @return
         */
        MatrixX sum(long length=-1, long rows=-1, long cols=-1);


        /**
         * function for sum
         * @param elements
         * @return
         */
        MatrixX sum(vector<MatrixX>& elements);

        /**
         *
         * @param elements
         * @param wise 0-rows, 1-cols, 2-element
         * @return
         */
        MatrixX sum(Eigen::MatrixXf, int wise);


        /**
         *
         * @param elements
         * @param wise 0-rows, 1-cols, 2-element
         * @return
         */
        MatrixX sum(CUDAMatrixs elements, int wise);
    };

    typedef CUDAPtrArray<CUDAPtrs> Summators;
}


#endif //GraphFusion_MAP_REDUCE_H
