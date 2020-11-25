//
// Created by liulei on 2019/11/12.
//


#include <thread>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <mutex>
#include "../../module/tool/file_util.h"
#include "../../module/inputsource/realsense_inputsource.h"
#include "../../module/inputsource/file_inputsource.h"


using namespace std;
using namespace cv;
using namespace rtf;

int main(int argc, char *argv[]) {

    //load inputSource
    string workspace = argv[1];
    string type = argv[2];
    FileUtil::createDirectory(workspace);
    BaseConfig::initInstance(workspace);

    FileInputSource * inputSource;
    if(type == "bf") {
        inputSource = new BFInputSource();
    } else if(type == "TUM") {
        inputSource = new TUMInputSource();
    } else {
        throw runtime_error("invalid data type");
    }

    int n = inputSource->getFrameNum();
    string file = workspace + "/frames_0.yaml";
    YAML::Node nodes;
    for(int i=0; i<n; i++) {
        auto frame = inputSource->waitFrame(0, -1, false);
        nodes.push_back(frame->serialize());
        cout << "finished to convert " << i+1 << "/" << n << endl;
    }

    YAMLUtil::saveYAML(file, nodes);

    delete inputSource;
    return 0;
}
