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

string workspace = "/home/liulei/workspace/small";
double minDepth = 0.1;
double maxDepth = 3;

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
    registrationPnPBA(&featureMatches, edge, curStream);
    if(near&&edge->isUnreachable()) {
        registrationEGBA(&featureMatches, edge, curStream);
        if(edge->isUnreachable()) {
            registrationHomoBA(&featureMatches, edge, curStream);
        }
    }
}

int main() {
    globalConfig.loadFromFile("test/test_online/online_pnp.yaml");

    FileInputSource * fileInputSource = new FileInputSource();
    cout << "device_num: " << fileInputSource->getDevicesNum() << endl;
    cout << "frame_num: " << fileInputSource->getFrameNum() << endl;

    SIFTFeatureExtractor extractor;
    auto ref = allocate_shared<Frame>(Eigen::aligned_allocator<Frame>(), fileInputSource->waitFrame(0, 46));
//    ref->setDepthBounds(minDepth, maxDepth);
    extractor.extractFeatures(ref, ref->getKps());

    auto cur = allocate_shared<Frame>(Eigen::aligned_allocator<Frame>(), fileInputSource->waitFrame(0, 47));
//    cur->setDepthBounds(minDepth, maxDepth);
    extractor.extractFeatures(cur, cur->getKps());

    SIFTFeatureMatcher matcher;
    FeatureMatches featureMatches = matcher.matchKeyPointsPair(ref->getKps(), cur->getKps());
    ImageUtil::drawMatches(featureMatches, ref, cur, workspace+"/matches.png");

    time_t start = clock();
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
        PointUtil::savePLYPointCloud("/home/liulei/桌面/pnp.ply", *pc);
    }
    return 0;
}
