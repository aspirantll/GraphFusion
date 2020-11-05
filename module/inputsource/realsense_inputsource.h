//
// Created by liulei on 2019/11/12.
//

#ifndef GAUSSIANFUSION_REALSENSEINPUTSOURCE_H
#define GAUSSIANFUSION_REALSENSEINPUTSOURCE_H

#include <iostream>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <algorithm>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <glog/logging.h>
#include <mutex>
#include "../datastructure/frame_types.h"
#include "../processor/depth_filters.h"

using namespace std;

namespace rtf {

    class RealsenseConfig {
    public:
        int filterFrames;
        ushort minDepth;
        ushort maxDepth;
        bool alignedColor = true;
        int parallelSize = -1;
    };

    class RealsenseInputSource {
    private:
        rs2::context context;
        vector<mutex *> pipeLocks;
        vector<int> cursors;
        vector<rs2::pipeline> pipelines;
        vector<rs2::config> configs;
        RealsenseConfig config;
        u_int32_t seqNo = 0;

        rs2::align* alignToColor;
        rs2::align* alignToDepth;
        rs2::spatial_filter* spat_filter;
        rs2::temporal_filter* temp_filter;
        // Declare filters
        rs2::decimation_filter* dec_filter;  // Decimation - reduces depth frame density
        rs2::threshold_filter* thr_filter;   // Threshold  - removes values outside recommended range
        rs2::disparity_transform* depth_to_disparity;
        rs2::disparity_transform* disparity_to_depth;

        /**
         * initialize the cameras and test pipeline
         */
        void initCameras();

        /**
         * start pipeline for camera
         * @param cameraPos
         * @return
         */
        rs2::pipeline route(int cameraPos);

        shared_ptr<FrameRGBD> getFrameByPipe(rs2::pipeline pipe, bool filtered, bool converted=true);
        /**
         * return filtered depth Frames
         * @param pipe
         */
        shared_ptr<FrameRGBD> filteredFrame(rs2::pipeline pipe);
        /**
         * return simple depth, rgb frames and depth units
         * @param pipe
         */
        shared_ptr<FrameRGBD> simpleFrame(rs2::pipeline pipe);

    public:
        RealsenseInputSource();

        RealsenseInputSource(RealsenseConfig config);

        ~RealsenseInputSource();

        /**
         * when plug a new camera ,or new begin end
         */
        void initialize();

        /**
         * only obtain from one camera
         * @return
         */
        shared_ptr<FrameRGBD> waitFrame(int cameraPos = 0, bool filtered= false, bool converted=true);

        /**
         * obtain from multi-cameras
         * @return
         */
        vector<shared_ptr<FrameRGBD>> waitMultiCamFrames(bool filtered= false, bool converted=true);

        /**
         * get the num of devices
         * @return
         */
        int getDevicesNum();
    };

}


#endif //GAUSSIANFUSION_REALSENSEINPUTSOURCE_H
