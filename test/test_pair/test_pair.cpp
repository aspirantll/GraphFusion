//
// Created by liulei on 2020/6/5.
//

#include <string>
#include <pcl/registration/transforms.h>
#include <opencv2/core/eigen.hpp>

#include "../../module/feature/feature2d.h"
#include "../../module/inputsource/file_inputsource.h"
#include "../../module/feature/feature_matcher.h"
#include "../../module/core/registration/registrations.h"

using namespace std;
using namespace rtf;
using namespace pcl;

string workspace = "/media/liulei/Data/dataset/TUM/rgbd_dataset_freiburg1_room";
double minDepth = 0.1;
double maxDepth = 3;

int main() {
    GlobalConfig globalConfig(workspace);
    globalConfig.kMinMatches = 25;
    globalConfig.virtualVoxelSize = 0.01f;
    globalConfig.kMinInliers = 15;
    globalConfig.rmsThreshold = 30;
    globalConfig.irThreshold = 0.7;
    globalConfig.maxPnPResidual = 5.991;
    globalConfig.maxAvgCost = 100;
    globalConfig.width = 640;
    globalConfig.height = 480;

    FileInputSource * fileInputSource = new FileInputSource();
    cout << "device_num: " << fileInputSource->getDevicesNum() << endl;
    cout << "frame_num: " << fileInputSource->getFrameNum() << endl;

    SIFTFeatureExtractor extractor;
    auto ref = allocate_shared<Frame>(Eigen::aligned_allocator<Frame>(), fileInputSource->waitFrame(0, 1));
//    ref->setDepthBounds(minDepth, maxDepth);
    extractor.extractFeatures(ref, ref->getKps());

    auto cur = allocate_shared<Frame>(Eigen::aligned_allocator<Frame>(), fileInputSource->waitFrame(0, 4));
//    cur->setDepthBounds(minDepth, maxDepth);
    extractor.extractFeatures(cur, cur->getKps());

    SIFTFeatureMatcher matcher;
    FeatureMatches featureMatches = matcher.matchKeyPointsPair(ref->getKps(), cur->getKps());
    ImageUtil::drawMatches(featureMatches, ref, cur, workspace+"/matches.png");

    time_t start = clock();
    BARegistration baRegistration(globalConfig);
    PnPRegistration pnPRegistration(globalConfig);

    auto pnpReport = pnPRegistration.registrationFunction(featureMatches);
    vector<FeatureKeypoint> kxs, kys;
    featureIndexesToPoints(featureMatches.getKx(), pnpReport.kps1, kxs);
    featureIndexesToPoints(featureMatches.getKy(), pnpReport.kps2, kys);
    ImageUtil::drawMatches(kxs, kys, ref, cur, workspace+"/pnp_inlier.png");

    clock_t s = clock();
    cout << pnpReport.T << endl;
    auto baReport = baRegistration.bundleAdjustment(pnpReport.T, ref->getCamera(), cur->getCamera(), kxs, kys);
    baReport.printReport();
    cout << "ba: " << double(clock()-s)/CLOCKS_PER_SEC << "s" << endl;
    cout << "pnp+ba: " << double(clock()-start)/CLOCKS_PER_SEC << "s" << endl;
    if(baReport.success) {
        auto pc = ref->calculatePointCloud();
        auto curPc = cur->calculatePointCloud();
        pcl::transformPointCloud(*curPc, *curPc, baReport.T);
        *pc += *curPc;
        PointUtil::savePLYPointCloud(workspace+"/pnp.ply", *pc);
    }
    return 0;
}
