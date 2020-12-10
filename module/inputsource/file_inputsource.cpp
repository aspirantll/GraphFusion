//
// Created by liulei on 2019/11/12.
//

#include "file_inputsource.h"
#include "../processor/frame_converters.h"

using namespace cv;
namespace rtf {
    FileInputSource::FileInputSource(bool init) {
        this->dirPath = BaseConfig::getInstance()->workspace;
        if(init) {
            initialize();
        }
    }

    void FileInputSource::initialize() {
        CameraFactory::readCameras();
        for(int i=0; i<getDevicesNum(); i++) {
            frames.emplace_back();
            ptrVec.push_back(0);
        }

        char filePath[100];
        sprintf(filePath, "%s/frames_%d.yaml", dirPath.c_str(), 0);
        LOG_ASSERT(FileUtil::exist(filePath))<<"file path error or camera model save path error!";

        for(int i=0; FileUtil::exist(filePath); i++) {
            YAML::Node cur = YAMLUtil::loadYAML(filePath);
            for(auto item: cur) {
                FrameBase frame = FrameBase(item);
                shared_ptr<Camera> camera = frame.getCamera();
                if(camera == nullptr) {
                    LOG(ERROR) << "cannot find camera for item:" << cur << endl;
                }else {
                    int index = CameraFactory::locateCamera(camera->getSerNum());
                    frames[index].push_back(item);
                }
            }
            sprintf(filePath, "%s/frames_%d.yaml", dirPath.c_str(), i+1);
        }

    }

    shared_ptr<FrameRGBD> FileInputSource::waitFrame(int cameraPos, int frameId, bool converted) {
        if(cameraPos >= getDevicesNum()) {
            throw runtime_error("no such camera!!!");
        }

        if(frameId==-1) {
            frameId = ptrVec[cameraPos];
            ptrVec[cameraPos] += 1;
        }

        vector<YAML::Node> framesForCamera = frames[cameraPos];
        auto frame = make_shared<FrameRGBD>(FrameRGBD(framesForCamera[frameId]));
        if(converted)
            FrameConverters::convertImageType(frame);
        return frame;
    }

    vector<shared_ptr<FrameRGBD>> FileInputSource::waitMultiCamFrames() {
        vector<shared_ptr<FrameRGBD>> frameVec;
        for(int pos=0; pos<FileInputSource::getDevicesNum(); pos++) {
            frameVec.push_back(waitFrame(pos));
        }
        return frameVec;
    }

    int FileInputSource::getDevicesNum() {
        return CameraFactory::getCameraNum();
    }


    int FileInputSource::getFrameNum(int cameraPos) {
        return frames[cameraPos].size();
    }



    ETH3DInputSource::ETH3DInputSource(): FileInputSource(false) {
        FileInputSource::dirPath = BaseConfig::getInstance()->workspace;
        initialize();
    }

    void ETH3DInputSource::initialize() {
        string serNum = "eth3d-camera";
        FileInputSource::frames.emplace_back();
        FileInputSource::ptrVec.push_back(0);

        LOG_ASSERT(FileUtil::exist(FileInputSource::dirPath+"/associated.txt"))<<"file path error or camera model save path error!";
        ifstream associateFile(FileInputSource::dirPath+"/associated.txt", ios::in | ios::binary);
        string line;
        long frameId = 0;
        // read frames
        while(getline(associateFile, line)) {
            YAML::Node node;

            auto parts = StringUtil::split(line, ' ');
            node["frameId"] = frameId++;
            node["camera"] = serNum;
            node["rgbImage"] = parts[1];
            node["depthImage"] = parts[3];
            node["minDepth"] = 500;
            node["maxDepth"] = 15000;

            frames[0].push_back(node);
        }
        associateFile.close();
        cv::Mat mat = cv::imread(FileInputSource::dirPath + "/" + FileInputSource::frames[0][0]["depthImage"].as<string>(), IMREAD_ANYDEPTH);
        int width = mat.cols, height = mat.rows;

        ifstream calibrationFile(FileInputSource::dirPath+"/calibration.txt", ios::in | ios::binary);
        getline(calibrationFile, line);
        auto parts = StringUtil::split(line, ' ');
        double fx = StringUtil::toDouble(parts[0]);
        double fy = StringUtil::toDouble(parts[1]);
        double cx = StringUtil::toDouble(parts[2]);
        double cy = StringUtil::toDouble(parts[3]);

        calibrationFile.close();
        shared_ptr<Camera> camera = make_shared<Camera>(serNum, fx, fy, cx, cy, width, height, 1.0/5000);
        CameraFactory::addCamera(camera);

    }


    BFInputSource::BFInputSource(): FileInputSource(false) {
        FileInputSource::dirPath = BaseConfig::getInstance()->workspace;
        initialize();
    }

    void BFInputSource::initialize() {
        string serNum = "bf-camera";
        FileInputSource::frames.emplace_back();
        FileInputSource::ptrVec.push_back(0);

        LOG_ASSERT(FileUtil::exist(FileInputSource::dirPath+"/info.txt"))<<" file path error or camera model save path error!";
        ifstream calibrationFile(FileInputSource::dirPath+"/info.txt", ios::in | ios::binary);
        int width, height, frameNums;
        double fx, fy, cx, cy, depthScale;
        string line;
        while(!calibrationFile.eof()) {
            getline(calibrationFile, line);
            auto parts = StringUtil::split(line, '=');
            if(parts.empty()) continue;
            if(parts[0] == "m_depthWidth") {
                width = StringUtil::toInt(parts[1]);
            }else if(parts[0] == "m_depthHeight") {
                height = StringUtil::toInt(parts[1]);
            }else if(parts[0] == "m_depthShift") {
                depthScale = 1/StringUtil::toDouble(parts[1]);
            }else if(parts[0] == "m_calibrationColorIntrinsic") {
                fx = StringUtil::toDouble(parts[1]);
                cx = StringUtil::toDouble(parts[3]);
                fy = StringUtil::toDouble(parts[6]);
                cy = StringUtil::toDouble(parts[7]);
            }else if(parts[0] == "m_frames.size") {
                frameNums = StringUtil::toInt(parts[1]);
            }
        }
        calibrationFile.close();
        shared_ptr<Camera> camera = make_shared<Camera>(serNum, fx, fy, cx, cy, width, height, depthScale);
        CameraFactory::addCamera(camera);

        char filePrefix[20];
        // read frames
        for(int id=0; id<frameNums; id++) {
            YAML::Node node;
            sprintf(filePrefix, "/frame-%06d", id);

            node["frameId"] = id;
            node["camera"] = serNum;
            node["rgbImage"] = string(filePrefix)+".color.jpg";
            node["depthImage"] = string(filePrefix)+".depth.png";
            node["minDepth"] = 100;
            node["maxDepth"] = 3000;

            FileInputSource::frames[0].emplace_back(node);
        }
    }


    TUMInputSource::TUMInputSource(): FileInputSource(false) {
        FileInputSource::dirPath = BaseConfig::getInstance()->workspace;
        initialize();
    }

    void TUMInputSource::initialize() {
        FileInputSource::frames.emplace_back();
        FileInputSource::ptrVec.push_back(0);

        string serNum;
        int width, height;
        double fx, fy, cx, cy, depthScale=1.0/5000, distCoef[5]={0.0};
        string path = BaseConfig::getInstance()->workspace;
        if (path.find("freiburg0")!=string::npos) {
            serNum = "tum-camera-0";
            width=640, height=480;
            fx=481.20, fy=-480.00, cx=319.50, cy=239.50;
            distCoef[0]=distCoef[1]=distCoef[2]=distCoef[3]=distCoef[4]=0.0;
        } else if (path.find("freiburg1")!=string::npos) {
            serNum = "tum-camera-1";
            width=640, height=480;
            fx=517.3, fy=516.5, cx=318.6, cy=255.3;
            distCoef[0]=0.2624, distCoef[1]=-0.9531, distCoef[2]=-0.0054, distCoef[3]=0.0026, distCoef[4]=1.1633;
        }else if (path.find("freiburg2")!=string::npos) {
            serNum = "tum-camera-2";
            width=640, height=480;
            fx=520.9, fy=521.0, cx=325.1, cy=249.7;
            distCoef[0]=0.2312, distCoef[1]=-0.7849, distCoef[2]=-0.0033, distCoef[3]=-0.0001, distCoef[4]=0.9172;
        }else if (path.find("freiburg3")!=string::npos) {
            serNum = "tum-camera-3";
            width=640, height=480;
            fx=535.4, fy=539.2, cx=320.1, cy=247.6;
            distCoef[0]=distCoef[1]=distCoef[2]=distCoef[3]=distCoef[4]=0.0;
        }

        shared_ptr<Camera> camera = make_shared<Camera>(serNum, fx, fy, cx, cy, width, height, depthScale, distCoef);
        CameraFactory::addCamera(camera);

        LOG_ASSERT(FileUtil::exist(FileInputSource::dirPath+"/associate.txt"))<<" associate file path error!";

        ifstream associateFile(FileInputSource::dirPath+"/associate.txt", ios::in | ios::binary);

        // skip header
        string line, dine;

        int id = 0;
        while(!associateFile.eof()) {
            getline(associateFile, line);

            auto parts = StringUtil::split(line, ' ');

            if(parts.size() != 4) {
                break;
            }

            YAML::Node node;
            node["frameId"] = id++;
            node["camera"] = serNum;
            node["rgbImage"] = "/"+parts[1];
            node["depthImage"] = "/"+parts[3];
            node["minDepth"] = 500;
            node["maxDepth"] = 25000;

            FileInputSource::frames[0].emplace_back(node);

        }
    }
}

