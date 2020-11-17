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
double maxDepth = 4;

GlobalConfig globalConfig(workspace);
EGRegistration *egRegistration;
HomographyRegistration *homoRegistration;
PnPRegistration *pnpRegistration;

void registrationEGBA(FeatureMatches *featureMatches, Edge *edge, cudaStream_t curStream) {
    stream = curStream;
    RANSAC2DReport eg = egRegistration->registrationFunction(*featureMatches);
    BAReport ba;
    if (eg.success) {
        vector<FeatureKeypoint> kxs, kys;
        featureIndexesToPoints(featureMatches->getKx(), eg.kps1, kxs);
        featureIndexesToPoints(featureMatches->getKy(), eg.kps2, kys);
        BARegistration baRegistration(globalConfig);
        ba = baRegistration.bundleAdjustment(eg.T, featureMatches->getCx(), featureMatches->getCy(), kxs, kys, true);
        if (ba.success) {
            double cost = ba.avgCost();
            if (!isnan(cost) && cost < globalConfig.maxAvgCost) {
                edge->setKxs(kxs);
                edge->setKys(kys);
                edge->setTransform(ba.T);
                edge->setCost(cost);
            }
        }
    }

    cout << "-------------------" << featureMatches->getFIndexX() << "-" << featureMatches->getFIndexY()  << "-global-eg+ba---------------------------------" << endl;
    eg.printReport();
    ba.printReport();
}

void registrationHomoBA(FeatureMatches *featureMatches, Edge *edge, cudaStream_t curStream) {
    stream = curStream;
    RANSAC2DReport homo = homoRegistration->registrationFunction(*featureMatches);
    BAReport ba;
    if (homo.success) {
        vector<FeatureKeypoint> kxs, kys;
        featureIndexesToPoints(featureMatches->getKx(), homo.kps1, kxs);
        featureIndexesToPoints(featureMatches->getKy(), homo.kps2, kys);

        BARegistration baRegistration(globalConfig);
        ba = baRegistration.bundleAdjustment(homo.T, featureMatches->getCx(), featureMatches->getCy(), kxs, kys, true);
        if (ba.success) {
            double cost = ba.avgCost();
            if (!isnan(cost) && cost < globalConfig.maxAvgCost) {
                edge->setKxs(kxs);
                edge->setKys(kys);
                edge->setTransform(ba.T);
                edge->setCost(cost);
            }
        }
    }

    cout << "-------------------" << featureMatches->getFIndexX() << "-" << featureMatches->getFIndexY()  << "-global-homo+ba---------------------------------" << endl;
    homo.printReport();
    ba.printReport();
}

void registrationPnPBA(FeatureMatches *featureMatches, Edge *edge, cudaStream_t curStream) {
    stream = curStream;
    RANSAC2DReport pnp = pnpRegistration->registrationFunction(*featureMatches);
    BAReport ba;
    if (pnp.success) {
        vector<FeatureKeypoint> kxs, kys;
        featureIndexesToPoints(featureMatches->getKx(), pnp.kps1, kxs);
        featureIndexesToPoints(featureMatches->getKy(), pnp.kps2, kys);

        BARegistration baRegistration(globalConfig);
        ba = baRegistration.bundleAdjustment(pnp.T, featureMatches->getCx(), featureMatches->getCy(), kxs, kys);
        if (ba.success) {
            double cost = ba.avgCost();
            if (!isnan(cost) && cost < globalConfig.maxAvgCost) {
                edge->setKxs(kxs);
                edge->setKys(kys);
                edge->setTransform(ba.T);
                edge->setCost(cost);
            }
        }
    }

    cout << "-------------------" << featureMatches->getFIndexX() << "-" << featureMatches->getFIndexY()  << "-global-pnp+ba---------------------------------" << endl;
    pnp.printReport();
    ba.printReport();
}

void registrationPairEdge(FeatureMatches featureMatches, Edge *edge, cudaStream_t curStream, bool near) {
    stream = curStream;
    clock_t start = clock();
    registrationPnPBA(&featureMatches, edge, curStream);
    /*if(near&&edge->isUnreachable()) {
        registrationEGBA(&featureMatches, edge, curStream);
        if(edge->isUnreachable()) {
            registrationHomoBA(&featureMatches, edge, curStream);
        }
    }*/
    cout << "time:" << double(clock()-start)/CLOCKS_PER_SEC << endl;
}

int main() {
    globalConfig.loadFromFile("test/test_online/online_pnp.yaml");

    FileInputSource * fileInputSource = new FileInputSource();
    cout << "device_num: " << fileInputSource->getDevicesNum() << endl;
    cout << "frame_num: " << fileInputSource->getFrameNum() << endl;

    SIFTFeatureExtractor extractor;
    auto ref = allocate_shared<Frame>(Eigen::aligned_allocator<Frame>(), fileInputSource->waitFrame(0, 0));
    ref->setDepthBounds(minDepth, maxDepth);
    extractor.extractFeatures(ref, ref->getKps());

    auto cur = allocate_shared<Frame>(Eigen::aligned_allocator<Frame>(), fileInputSource->waitFrame(0, 1));
    cur->setDepthBounds(minDepth, maxDepth);
    extractor.extractFeatures(cur, cur->getKps());

    auto fur = allocate_shared<Frame>(Eigen::aligned_allocator<Frame>(), fileInputSource->waitFrame(0, 2));
    fur->setDepthBounds(minDepth, maxDepth);
    extractor.extractFeatures(fur, fur->getKps());

    SIFTFeatureMatcher matcher;
    FeatureMatches featureMatches = matcher.matchKeyPointsPair(ref->getKps(), cur->getKps());
    ImageUtil::drawMatches(featureMatches, ref, cur, workspace+"/matches.png");
/*    int dotSum = 0;
    for(int i=0; i<featureMatches.size(); i++) {
        FeatureMatch match = featureMatches.getMatch(i);
        Eigen::Matrix<uint8_t, 1, -1, Eigen::RowMajor> d1 = featureMatches.getFp1().getDescriptors().row(match.getPX());
        Eigen::Matrix<uint8_t, 1, -1, Eigen::RowMajor> d2 = featureMatches.getFp2().getDescriptors().row(match.getPY());
        int dist = d1.dot(d2);
        cout << dist << endl;
        dotSum += dist;
    }
    cout << "mean: " << dotSum/featureMatches.size() << endl;*/

    egRegistration = new EGRegistration(globalConfig);
    homoRegistration = new HomographyRegistration(globalConfig);
    pnpRegistration = new PnPRegistration(globalConfig);

    Edge edge = Edge::UNREACHABLE;
    registrationPairEdge(featureMatches, &edge, 0, true);

    if(!edge.isUnreachable()) {
        auto pc = ref->calculatePointCloud();
        auto curPc = cur->calculatePointCloud();
        pcl::transformPointCloud(*curPc, *curPc, edge.getTransform());
        *pc += *curPc;
        PointUtil::savePLYPointCloud("/home/liulei/桌面/pnp01.ply", *pc);
    }

    {
        auto pc = cur->calculatePointCloud();
        auto curPc = fur->calculatePointCloud();
        pcl::transformPointCloud(*curPc, *curPc, edge.getTransform());
        *pc += *curPc;
        PointUtil::savePLYPointCloud("/home/liulei/桌面/vecity12.ply", *pc);
    }

    time_t start = clock();
    featureMatches = matcher.matchKeyPointsWithProjection(cur, fur, edge.getTransform());
    cout << "match:" << double(clock()-start)/CLOCKS_PER_SEC << endl;
    ImageUtil::drawMatches(featureMatches, cur, fur, workspace+"/motion_matches.png");

    vector<FeatureKeypoint> kxs, kys;
    featureMatchesToPoints(featureMatches, kxs, kys);

    BARegistration baRegistration(globalConfig);
    BAReport ba = baRegistration.bundleAdjustment(edge.getTransform(), featureMatches.getCx(), featureMatches.getCy(), kxs, kys, true);
    cout << "motion time:" << double(clock()-start)/CLOCKS_PER_SEC << endl;
    cout << "--------------------------------Motion------------------------------------------" << endl;
    ba.printReport();
    if (ba.success) {
        double cost = ba.avgCost();
        if (!isnan(cost) && cost < globalConfig.maxAvgCost) {
            auto pc = cur->calculatePointCloud();
            auto curPc = fur->calculatePointCloud();
            pcl::transformPointCloud(*curPc, *curPc, edge.getTransform());
            *pc += *curPc;
            PointUtil::savePLYPointCloud("/home/liulei/桌面/motion12.ply", *pc);
        }
    }

    ImageUtil::drawMatches(kxs, kys, cur, fur, workspace+"/motion_inliers.png");
    return 0;
}
