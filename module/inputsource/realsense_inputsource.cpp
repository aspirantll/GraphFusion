//
// Created by liulei on 2019/11/12.
//

#include "realsense_inputsource.h"
#include "../processor/frame_converters.h"

using namespace cv;

namespace rtf {
    RealsenseInputSource::RealsenseInputSource() {
        this->config.alignedColor = true;
        initialize();
    }

    RealsenseInputSource::RealsenseInputSource(RealsenseConfig config) {
        this->config = config;
        initialize();
    }

    RealsenseInputSource::~RealsenseInputSource() {
        delete alignToColor;
        delete alignToDepth;
        delete spat_filter;
        delete temp_filter;
        delete depth_to_disparity;
        delete disparity_to_depth;
        delete dec_filter;
        delete thr_filter;
    }

    void RealsenseInputSource::initCameras() {
        CameraFactory::readCameras();
        bool alignedColor = config.alignedColor;
        //for each config, then initialize camera and test pipeline
        for(auto cfg: configs) {
            rs2::pipeline pipe(context);
            pipe.start(cfg);

            auto frameSet = pipe.wait_for_frames();
            auto depth = frameSet.get_depth_frame();
            double depthScale = rs2::sensor_from_frame(depth)->get_option(RS2_OPTION_DEPTH_UNITS);
            auto depthSensor = rs2::sensor_from_frame(frameSet.get_color_frame());
            depthSensor->set_option(RS2_OPTION_EXPOSURE, 300);

            // obtain internal parameters
            auto depth_stream_file = pipe.get_active_profile().get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
            auto color_stream_file = pipe.get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();

            rs2_intrinsics intrinsics = alignedColor? color_stream_file.get_intrinsics():depth_stream_file.get_intrinsics();

            // initialize camera
            string serNum = pipe.get_active_profile().get_device().get_info(
                    RS2_CAMERA_INFO_SERIAL_NUMBER);
            double cx = intrinsics.ppx;
            double cy = intrinsics.ppy;
            double fx = intrinsics.fx;
            double fy = intrinsics.fy;
            int width = intrinsics.width;
            int height = intrinsics.height;

            shared_ptr<Camera> camera = make_shared<Camera>(Camera(serNum, fx, fy, cx, cy, width, height, depthScale));
            CameraFactory::addCamera(camera);
        }

    }

    void RealsenseInputSource::initialize() {
        auto devices = context.query_devices();
        cout << "the num of devices:" << devices.size() << endl;
        for (auto dev : devices) {
            string serNum = dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
            rs2::config cfg;
            cfg.enable_device(serNum);
            cfg.disable_all_streams();
            cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16);
            cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8);
            configs.push_back(cfg);
        }

        initCameras();

        int parallelSize = config.parallelSize;
        if(parallelSize <= 0 || parallelSize >configs.size()) {
            parallelSize = configs.size();
        }

        for(int i=0; i<parallelSize; i++) {
            rs2::pipeline pipe(context);
            rs2::config cfg = configs[i];
            pipe.start(cfg);
            pipelines.push_back(pipe);
            cursors.push_back(i);
            pipeLocks.push_back(new mutex);
        }

        // initialize
        alignToColor = new rs2::align(RS2_STREAM_COLOR);
        alignToDepth = new rs2::align(RS2_STREAM_DEPTH);
        spat_filter = new rs2::spatial_filter();
        temp_filter = new rs2::temporal_filter();
        depth_to_disparity = new rs2::disparity_transform(true);
        disparity_to_depth = new rs2::disparity_transform(false);
        dec_filter = new rs2::decimation_filter();
        thr_filter = new rs2::threshold_filter();
    }

    rs2::pipeline RealsenseInputSource::route(int cameraPos) {
        int pipePos = cameraPos;
        int configPos = cameraPos;
        if(config.parallelSize > 0) {//no limit
            pipePos = cameraPos % config.parallelSize;
        }

        rs2::pipeline pipe = pipelines[pipePos];
        rs2::config cfg = configs[configPos];

        mutex * lock = pipeLocks[pipePos];
        lock->lock();
        int cursor = cursors[pipePos];
        if(cameraPos != cursor) {
            pipe.stop();
            pipe.start(cfg);
            cursors[pipePos] = cameraPos;
        }
        lock->unlock();

        return pipe;


    }

    shared_ptr<FrameRGBD> RealsenseInputSource::filteredFrame(rs2::pipeline pipe) {
        //configs
        int filterFrames = config.filterFrames;

        vector<shared_ptr<FrameDepth>> depthFrames;
        shared_ptr<Mat> rgbImage;
        shared_ptr<Camera> camera;

        for(int i=0; i<filterFrames; i++) {
            shared_ptr<FrameRGBD> frame = simpleFrame(pipe);
            if (i == filterFrames / 2) {
                rgbImage = frame->getRGBImage();
            }
            if (i == 0) {
                camera = frame->getCamera();
            }
            depthFrames.push_back(frame);
        }

        shared_ptr<Mat> depthImage = DepthFilter::medianFilter(depthFrames, config.minDepth, config.maxDepth);
        shared_ptr<FrameRGBD> frame = make_shared<FrameRGBD>(FrameRGBD(seqNo, camera, rgbImage, depthImage));
        frame->setDepthBounds(config.minDepth, config.maxDepth);
        return frame;
    }

    shared_ptr<FrameRGBD> RealsenseInputSource::simpleFrame(rs2::pipeline pipe) {
        bool alignedColor = config.alignedColor;

        string serNum = pipe.get_active_profile().get_device().get_info(
                RS2_CAMERA_INFO_SERIAL_NUMBER);
        shared_ptr<Camera> camera = CameraFactory::getCamera(serNum);

        int width = camera->getWidth();
        int height = camera->getHeight();

        auto frameSet = pipe.wait_for_frames();

        if (alignedColor) {
            frameSet = alignToColor->process(frameSet);
        } else {
            frameSet = alignToDepth->process(frameSet);
        }

        shared_ptr<Mat> rgbImage = make_shared<Mat>(Mat());
        shared_ptr<Mat> depthImage = make_shared<Mat>(Mat());

        auto color = frameSet.get_color_frame();
        Mat(Size(width, height), CV_8UC3, (void *) color.get_data(), Mat::AUTO_STEP).copyTo(*rgbImage);
        // use SDK spatial_filter temporal_filter

        rs2::frame filtered = frameSet.get_depth_frame();
        /* Apply filters.
            The implemented flow of the filters pipeline is in the following order:
            1. apply decimation filter
            2. apply threshold filter
            3. transform the scene into disparity domain
            4. apply spatial filter
            5. apply temporal filter
            6. revert the results back (if step Disparity filter was applied
            to depth domain (each post processing block is optional and can be applied independantly).
        */
//        filtered = dec_filter->process(filtered);
        filtered = thr_filter->process(filtered);
        filtered = depth_to_disparity->process(filtered);
        filtered = spat_filter->process(filtered);
        filtered = temp_filter->process(filtered);
        filtered = disparity_to_depth->process(filtered);

        Mat(Size(width, height), CV_16UC1, (void *) filtered.get_data(), Mat::AUTO_STEP).copyTo(*depthImage);

        auto frame = make_shared<FrameRGBD>(FrameRGBD(seqNo, camera, rgbImage, depthImage));
        frame->setDepthBounds(config.minDepth, config.maxDepth);
        return frame;
    }

    shared_ptr<FrameRGBD> RealsenseInputSource::getFrameByPipe(rs2::pipeline pipe, bool filtered, bool converted) {
        auto frame = filtered? filteredFrame(pipe): simpleFrame(pipe);
        if(converted)
            FrameConverters::convertImageType(frame);
        return frame;
    }


    shared_ptr<FrameRGBD> RealsenseInputSource::waitFrame(int cameraPos, bool filtered, bool converted) {
        LOG_ASSERT(cameraPos >=0) << "cameraPos must be positive integer";
        shared_ptr<FrameRGBD> frame = getFrameByPipe(route(cameraPos), filtered, converted);
        seqNo++;
        return frame;
    }

    vector<shared_ptr<FrameRGBD>> RealsenseInputSource::waitMultiCamFrames(bool filtered, bool converted) {
        vector<shared_ptr<FrameRGBD>> multiFrames;
        for (int pos=0; pos<configs.size(); pos++) {
            multiFrames.push_back(getFrameByPipe(route(pos), filtered, converted));
        }
        seqNo++;
        return multiFrames;
    }

    int RealsenseInputSource::getDevicesNum() {
        return configs.size();
    }
}




