//
// Created by liulei on 2020/6/5.
//

#include "../../module/datastructure/config.h"
#include "../../module/processor/image_cuda.cuh"
#include "../../module/inputsource/file_inputsource.h"
#include "../../module/core/solver/matrix_conversion.h"
#include "../../module/tool/image_util.h"

string workspace = "/media/liulei/Data/dataset/TUM/rgbd_dataset_freiburg0_living_room_traj0";
double minDepth = 0.1;
double maxDepth = 3;

using namespace rtf;

int main() {
    GlobalConfig globalConfig(workspace);
    globalConfig.kMinMatches = 25;
    globalConfig.virtualVoxelSize = 0.01f;
    globalConfig.kMinInliers = 15;
    globalConfig.rmsThreshold = 30;
    globalConfig.irThreshold = 0.7;
    globalConfig.maxAvgCost = 100;
    globalConfig.width = 640;
    globalConfig.height = 480;

    FileInputSource * fileInputSource = new TUMInputSource();
    cout << "device_num: " << fileInputSource->getDevicesNum() << endl;
    cout << "frame_num: " << fileInputSource->getFrameNum() << endl;
    // visiualize
    for(int i=0; i<fileInputSource->getFrameNum(); i++) {
        auto frame = fileInputSource->waitFrame();
        shared_ptr<cv::Mat> normal = frame->getNormalImage();
        cv::Mat visNormal = ImageUtil::visualizeNormal(normal);
        cv::imshow("normal", visNormal);
        cv::waitKey();
    }

    return 0;
}