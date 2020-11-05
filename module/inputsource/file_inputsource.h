//
// Created by liulei on 2019/11/12.
//

#ifndef GAUSSIANFUSION_FILESOURCE_H
#define GAUSSIANFUSION_FILESOURCE_H

#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <glog/logging.h>

#include "../datastructure/frame_types.h"

using namespace std;
namespace rtf {
    class FileInputSource {
    protected:
        string dirPath;
        vector<int> ptrVec;
        vector<vector<YAML::Node>> frames;
    public:
        FileInputSource(bool init=true);

        /**
         * when plug a new camera ,or new begin end
         */
        virtual void initialize();

        /**
         * only obtain from one camera
         * @return
         */
        shared_ptr<FrameRGBD> waitFrame(int cameraPos = 0, int frameId=-1, bool converted=true);

        /**
         * get the num of frames for a camera
         * @param cameraPos
         * @return
         */
        int getFrameNum(int cameraPos = 0);

        /**
         * obtain from multi-cameras
         * @return
         */
        vector<shared_ptr<FrameRGBD>> waitMultiCamFrames();

        /**
         * get the num of devices
         * @return
         */
        int getDevicesNum();
    };


    class ETH3DInputSource: public FileInputSource {
    protected:
        string dirPath;
        vector<int> ptrVec;
        vector<vector<YAML::Node>> frames;

    public:
        ETH3DInputSource();
        /**
         * when plug a new camera ,or new begin end
         */
        void initialize();
    };


    class BFInputSource: public FileInputSource {
    protected:
        string dirPath;
        vector<int> ptrVec;
        vector<vector<YAML::Node>> frames;

    public:
        BFInputSource();
        /**
         * when plug a new camera ,or new begin end
         */
        void initialize();

    };

    class TUMInputSource: public FileInputSource {
    protected:
        string dirPath;
        vector<int> ptrVec;
        vector<vector<YAML::Node>> frames;

    public:
        TUMInputSource();
        /**
         * when plug a new camera ,or new begin end
         */
        void initialize();

    };
}



#endif //GAUSSIANFUSION_FILESOURCE_H
