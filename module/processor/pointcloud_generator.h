//
// Created by liulei on 2020/7/14.
//

#ifndef RTF_POINTCLOUD_GENERATOR_H
#define RTF_POINTCLOUD_GENERATOR_H
#include "../datastructure/cuda_types.h"
#include "../datastructure/camera.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/gpu/containers/device_array.h>
#include <opencv2/cudaimgproc.hpp>

namespace rtf {
    class PointCloudGenerator {
    public:
        static void generate(shared_ptr<pcl::PointCloud< pcl::PointXYZRGBNormal>> pointCloud, shared_ptr<Camera> camera, shared_ptr<cv::Mat> rgbImage, shared_ptr<cv::Mat> depthImage, shared_ptr<cv::Mat> normalImage, double minDepth, double maxDepth);
    };
}


#endif //RTF_POINTCLOUD_GENERATOR_H
