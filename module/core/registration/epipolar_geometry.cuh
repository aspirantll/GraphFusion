//
// Created by liulei on 2020/7/16.
//

#ifndef RTF_EG_H
#define RTF_EG_H

#include "../../datastructure/cuda_types.h"
#include "../../tool/map_reduce.h"

namespace rtf {

    /**
     * compute RMS for points
     * @param mat
     * @param meanPoint
     * @param summator
     * @return
     */
    float computeRMS(CUDAMatrixs &mat, CUDAMatrixs &meanPoint, Summator &summator);

    /**
     * transform points
     * @param mat
     * @param transform
     */
    void transformPoints(CUDAMatrixs &mat, CUDAMatrixs &transform);


    /**
     * compute matrix A
     * @param x
     * @param y
     * @param A
     */
    void computeAMatrix(CUDAMatrixs &x, CUDAMatrixs &y, CUDAMatrixs &A);

    /**
     * compute residual for points
     * @param x
     * @param y
     * @param E
     * @param residual
     */
    void computeResidualsCUDA(CUDAMatrixs &x, CUDAMatrixs &y, CUDAMatrixs &E, CUDAMatrixs &residual);

    /**
     * compute A for homography estimate
     * @param x
     * @param y
     * @param A
     */
    void computeHomoAMatrix(CUDAMatrixs &x, CUDAMatrixs &y, CUDAMatrixs &A);

    /**
     * compute residual for homography
     * @param x
     * @param y
     * @param H
     * @param residual
     */
    void computeHomoResidualsCUDA(CUDAMatrixs &x, CUDAMatrixs &y, CUDAMatrixs &H, CUDAMatrixs &residual);
}


#endif //RTF_EG_H
