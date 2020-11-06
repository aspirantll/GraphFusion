//
// Created by liulei on 2020/6/5.
//

#include <string>
#include <pcl/point_cloud.h>
#include <pcl/registration/transforms.h>
#include "../../module/feature/feature2d.h"
#include "../../module/inputsource/file_inputsource.h"
#include "../../module/feature/feature_matcher.h"
#include "../../module/core/registration/registrations.h"
#include "../../module/core/fusion/voxel_fusion.h"
#include "../../module/processor/depth_filters.h"
#include "../../module/controller/online_reconstruction.h"
#include "../../module/processor/frame_converters.h"

using namespace std;
using namespace rtf;
using namespace pcl;

string workspace;
double minDepth = 0.1;
double maxDepth = 3.09294;

void saveATP(ViewGraph& viewGraph, GlobalConfig& globalConfig) {
    if(!FileUtil::exist(workspace+"/associate.txt")) {
        return;
    }
    ifstream gt(workspace+"/associate.txt", ios::in | ios::binary);
    ofstream estimate(workspace+"/online_estimate_" + to_string(globalConfig.overlapNum) + ".txt", ios::out | ios::binary);

    string line;
    for(const shared_ptr<KeyFrame>& kf: viewGraph.getSourceFrames()) {
        Transform baseTrans = viewGraph.getFrameTransform(kf->getIndex());
        for(shared_ptr<Frame> frame: kf->getFrames()) {
            getline(gt, line);
            while (line.empty()) {
                getline(gt, line);
            }
            auto gtParts = StringUtil::split(line, ' ');

            if(!viewGraph.isVisible(kf->getIndex())||!frame->isVisible()) continue;
            Transform trans = baseTrans*frame->getTransform();
            Rotation R;
            Translation t;
            GeoUtil::T2Rt(trans, R, t);
            Eigen::Quaternion<Scalar> q(R);
            estimate << gtParts[0] << " " << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }

    }
}

void saveResult(ViewGraph& viewGraph) {
    int count = 0;
    for(const shared_ptr<KeyFrame>& kf: viewGraph.getSourceFrames()) {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        Transform baseTrans = viewGraph.getFrameTransform(kf->getIndex());
        for(shared_ptr<Frame> frame: kf->getFrames()) {
            frame->reloadImages();
            FrameConverters::convertImageType(frame);
            if(!viewGraph.isVisible(kf->getIndex())||!frame->isVisible()) continue;
            Transform trans = baseTrans*frame->getTransform();
            auto pc = frame->calculatePointCloud();
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            pcl::transformPointCloud(*pc, *transformed_cloud, trans);
            PointUtil::savePLYPointCloud(workspace + "/frame_" + to_string(frame->getFrameIndex()) + ".ply", *transformed_cloud);
            *pointCloud += *transformed_cloud;
            count++;
            frame->releaseImages();
        }
        PointUtil::savePLYPointCloud(workspace + "/keyframe_" + to_string(kf->getIndex()) + ".ply", *pointCloud);
        cout << "merge cloud num is " << count << endl;
    }
}

int main(int argc, char* argv[]) {
    workspace = argv[1];
    string configFile = argv[2];
    GlobalConfig globalConfig(workspace);
    globalConfig.loadFromFile(configFile);
    workspace = globalConfig.workspace;

//    string savePath = workspace + "/online_result_mesh_" + to_string(globalConfig.overlapNum) + ".ply";
    string savePath = "/home/liulei/桌面/online_result_mesh_" + to_string(globalConfig.overlapNum) + ".ply";
//    if(FileUtil::exist(savePath)) return 0;
//    freopen((workspace+"/online_out.txt").c_str(),"w",stdout);

    FileInputSource * fileInputSource = new FileInputSource();
    cout << "device_num: " << fileInputSource->getDevicesNum() << endl;
    cout << "frame_num: " << fileInputSource->getFrameNum() << endl;

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    OnlineReconstruction onlineRecon(globalConfig);
    for(int i=0; i<fileInputSource->getFrameNum(); i++) {
        shared_ptr<FrameRGBD> frame = fileInputSource->waitFrame(0, i);
        frame->setDepthBounds(minDepth, maxDepth);
        onlineRecon.appendFrame(frame);
        frame->releaseImages();
    }
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    cout << "mean tracked time: " << ttrack/fileInputSource->getFrameNum() << endl;
    cout << "finish to online reconstruction: " << ttrack << endl;
//    onlineRecon.getViewGraph().print();
//    YAMLUtil::saveYAML(workspace+"/online.yaml", onlineRecon.getViewGraph().serialize());
    onlineRecon.finalOptimize(true);
    onlineRecon.saveMesh(savePath);
//    saveResult(onlineRecon.getViewGraph());
    saveATP(onlineRecon.getViewGraph(), globalConfig);

//    while(!onlineRecon.closed());

    return 0;
}