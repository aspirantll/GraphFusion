//
// Created by liulei on 2020/10/14.
//

#ifndef RTF_IMAGE_CUDA_CUH
#define RTF_IMAGE_CUDA_CUH

#include <opencv2/core/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "../datastructure/cuda_types.h"
#include "../core/solver/cuda_matrix.h"

using namespace std;

namespace rtf {
    namespace DepthFilterCUDA {
        void medianFilter(vector<shared_ptr<cv::Mat>> depthImages, shared_ptr<cv::Mat> output,  ushort minDepth, ushort maxDepth);
    };

    void cvtColor(cv::Mat& img, cv::Mat& out, int code);

    void imageConversion(shared_ptr<cv::Mat> depthImg, shared_ptr<cv::Mat> depthOut, shared_ptr<cv::Mat> rgbImg, shared_ptr<cv::Mat> rgbOut, shared_ptr<cv::Mat> normal, float3x3 invK, float ds, ushort minDepth, ushort maxDepth);

    void convertToRawBits(shared_ptr<cv::Mat> rgbImg, vector<uint8_t>& rawBits);

    void computeNormalFromDepthImg(shared_ptr<cv::Mat> depthImg, shared_ptr<cv::Mat> normalImg, float3x3 K);
}


#endif //RTF_IMAGE_CUDA_CUH

