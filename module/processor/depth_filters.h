//
// Created by liulei on 2020/6/4.
//

#ifndef RTF_DEPTH_FILTERS_H
#define RTF_DEPTH_FILTERS_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include "../datastructure/frame_types.h"
#ifdef CUDA_ENABLED
#include "image_cuda.cuh"
#endif


namespace rtf {
    namespace DepthFilter {
        /**
         * find median point for each pixel locations
         * @param depthImages
         * @param minDepth
         * @param maxDepth
         * @return
         */
        shared_ptr<cv::Mat> medianFilter(vector<shared_ptr<FrameDepth>> depthImages, ushort minDepth, ushort maxDepth);


        void bilateralFilter(shared_ptr<FrameDepth> frame);

    }


    namespace DepthFilterCPU {
        /**
         * the cpu implement of median filter
         * @param depthImages
         * @param output
         * @param minDepth
         * @param maxDepth
         */
        void medianFilter(vector<shared_ptr<cv::Mat>> depthImages, shared_ptr<cv::Mat> output,  ushort minDepth, ushort maxDepth);
    }
}


#endif //RTF_DEPTH_FILTERS_H
