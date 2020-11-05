//
// Created by liulei on 2020/6/4.
//

#include "depth_filters.h"

using namespace cv;
namespace rtf {
    void DepthFilterCPU::medianFilter(vector<shared_ptr<Mat>> depthImages, shared_ptr<Mat> output,  ushort minDepth, ushort maxDepth) {
        int width = output->size[1];
        int height = output->size[0];

        #pragma omp parallel for schedule(dynamic) //Using OpenMP to try to parallelise the loop
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                vector<ushort> tmp_vec;
                for (int ii = 0; ii < depthImages.size(); ii++) {
                    ushort tmp = (depthImages[ii])->at<ushort>(i, j);
                    tmp_vec.push_back(tmp);
                }
                sort(tmp_vec.begin(), tmp_vec.end());
                int mid = tmp_vec.size() / 2;
                if (tmp_vec[mid] > maxDepth || tmp_vec[mid] < minDepth)
                    tmp_vec[mid] = 0;
                output->at<ushort>(i, j) = tmp_vec[mid];
            }
        }
    }

    shared_ptr<Mat> DepthFilter::medianFilter(vector<shared_ptr<FrameDepth>> depthFrames, ushort minDepth, ushort maxDepth) {
        if(depthFrames.size() <= 0) {
            return nullptr;
        }
        shared_ptr<Camera> camera = depthFrames[0]->getCamera();

        int width = camera->getWidth();
        int height = camera->getHeight();

        vector<shared_ptr<Mat>> depthImages;
        for(auto depthFrame: depthFrames) {
            depthImages.push_back(depthFrame->getDepthImage());
        }

        //filter for depth
        shared_ptr<Mat> depthImage = make_shared<Mat>(Mat::zeros(Size(width, height), CV_16UC1));
        #ifdef CUDA_ENABLED
        DepthFilterCUDA::medianFilter(depthImages, depthImage, minDepth, maxDepth);
        #else
        DepthFilterCPU::medianFilter(depthImages, depthImage, minDepth, maxDepth);
        #endif
        return  depthImage;

    }


    void DepthFilter::bilateralFilter(shared_ptr<FrameDepth> frame) {
        cv::bilateralFilter(*frame->getDepthImage()*255, *frame->getDepthImage(), 50, 12.5, 50);
        *frame->getDepthImage() /= 255;
    }

}

