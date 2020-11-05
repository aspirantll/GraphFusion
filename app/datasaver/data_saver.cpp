//
// Created by liulei on 2019/11/12.
//


#include <thread>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <condition_variable>
#include <mutex>
#include <time.h>
#include "../../module/tool/file_util.h"
#include "../../module/inputsource/realsense_inputsource.h"


using namespace std;
using namespace cv;
using namespace rtf;

volatile bool quit = false;
volatile bool save = false;
bool filtered = false;
ushort maxDepth = 2000;
ushort minDepth = 100;
int filterFrames = 10;
int saveCount = 0;
int parallelSize = 1;

vector<bool> endFlags;
vector<mutex *> mtxs; // global mutex.
vector<condition_variable *> conditions; // global conditions
vector<YAML::Node> yamls;

// reset endFlags
void resetEndFlags() {
    for(int i=0; i<endFlags.size(); i++) {
        endFlags[i] = false;
    }
}

bool saved() {
    for(int i=0; i<endFlags.size(); i++) {
       if(!endFlags[i]) return false;
    }
    return true;
}


void notifyAll() {
    for(auto condition : conditions) {
        condition->notify_all();
    }
}

//鼠标回调函数
void onMouse(int event, int x, int y, int flags, void* userdata) {
    switch (event) {
        case EVENT_LBUTTONDBLCLK:
            save = true;
            notifyAll();
            break;
    }
}


//保存图像
void saveFrame(RealsenseInputSource *inputSource, string parameterFilePath, string workspace, int groupNum) {
    for(int i=groupNum; i<inputSource->getDevicesNum(); i += parallelSize) {
        shared_ptr<FrameRGBD> filteredFrame = inputSource->waitFrame(i, filtered, false);
        yamls[groupNum].push_back(filteredFrame->serialize());
        YAMLUtil::saveYAML(parameterFilePath, yamls[groupNum]);
    }

}

//保存线程工作函数
void saveLoop(RealsenseInputSource *inputSource, string workspace, int groupNum) {
    string parameterFilePath = workspace + "/frames_" + to_string(groupNum) + ".yaml";
    unique_lock<mutex> lock(*mtxs[groupNum]);

    while(!quit) {
        conditions[groupNum]->wait(lock);

        if (save) {// save frame
            time_t start = time(NULL);
            cout << "begin to save frame " << saveCount << " for " << groupNum << endl;

            saveFrame(inputSource, parameterFilePath, workspace, groupNum);

            cout << "saved frame " << saveCount << " for " << groupNum << endl;

            cout << "total time：" << (double)(time(NULL) - start) <<endl;

            endFlags[groupNum] = true;
            if(groupNum==0) {
                while(!saved());
                saveCount++;
                save = false;
                resetEndFlags();
            }

        }
    }


}


//工作循环
void loop(RealsenseInputSource *inputSource) {
    //CloudViewerPtr viewer = Visualization::createSimpleCloudViewer("3d viewer");

    string winName = "rgb";
    while (!quit) {
        shared_ptr<FrameRGBD> frames = inputSource->waitFrame(0, false, false);
        namedWindow(winName);
        setMouseCallback(winName, onMouse, nullptr); //类型转换

        do {
            cv::imshow("rgb", *frames->getRGBImage());
            cv::imshow("depth", (*frames->getDepthImage()) * 100);
            //viewer->showCloud(frames->pointCloud);
            cv::waitKey(1);
        }while (save);
    }
}

void waitQuitCommand() {
    char ch;
    while ((ch = getchar()) != 'q') {
        if (!save && ch == 's') {
            save = true;
            notifyAll();
        }
    }

    quit = true;
    notifyAll();
}


int main(int argc, char *argv[]) {

    //load inputSource
    string workspace = argv[1];
    FileUtil::createDirectory(workspace);

    BaseConfig::initInstance(workspace);

    RealsenseConfig config;
    config.filterFrames = filterFrames;
    config.minDepth = minDepth;
    config.maxDepth = maxDepth;
    config.alignedColor = true;
    config.parallelSize = parallelSize;
    RealsenseInputSource *inputSource = new RealsenseInputSource(config);

    //创建线程
    vector<thread *> threadVec;

    threadVec.emplace_back(new thread(loop, inputSource));
    threadVec.emplace_back(new thread(waitQuitCommand));

    for(int i=0; i<parallelSize; i++) {
        mtxs.emplace_back(new mutex());
        conditions.emplace_back(new condition_variable());
        endFlags.emplace_back(false);
        yamls.emplace_back();

        threadVec.emplace_back(new thread(saveLoop, inputSource, workspace, i));
    }

    for(thread * th : threadVec) {
        th->join();
    }

    return 0;
}
