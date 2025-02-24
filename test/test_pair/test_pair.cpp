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

string workspace = "/media/liulei/Data/dataset/TUM/rgbd_dataset_freiburg0_living_room_traj0";
double minDepth = 0.1;
double maxDepth = 4;

GlobalConfig globalConfig(workspace);
PnPRegistration *pnpRegistration;

void registrationPnPBA(FeatureMatches *featureMatches, ConnectionCandidate *edge, cudaStream_t curStream) {
    stream = curStream;
    RANSAC2DReport pnp = pnpRegistration->registrationFunction(*featureMatches);
    RegReport ba;
    if (pnp.success) {
        {
            vector<FeatureKeypoint> kxs, kys;
            BARegistration baRegistration(globalConfig);
            if(pnp.inliers.size()<100) {
                FeatureMatches matches = SIFTFeatureMatcher().matchKeyPointsWithProjection(featureMatches->getFp1(), featureMatches->getFp2(), pnp.T);
                featureMatchesToPoints(matches, kxs, kys);

                ba = baRegistration.bundleAdjustment(pnp.T, matches.getCx(), matches.getCy(), kxs, kys, true);
            }else {
                featureIndexesToPoints(featureMatches->getKx(), pnp.kps1, kxs);
                featureIndexesToPoints(featureMatches->getKy(), pnp.kps2, kys);

                ba = baRegistration.bundleAdjustment(pnp.T, featureMatches->getCx(), featureMatches->getCy(), kxs, kys);
            }
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
    }

    cout << "-------------------" << featureMatches->getFIndexX() << "-" << featureMatches->getFIndexY()  << "-global-pnp+ba---------------------------------" << endl;
    pnp.printReport();
    ba.printReport();
}

void registrationPairEdge(FeatureMatches featureMatches, ConnectionCandidate *edge, cudaStream_t curStream, bool near) {
    stream = curStream;
    clock_t start = clock();
    registrationPnPBA(&featureMatches, edge, curStream);
    cout << "time:" << double(clock()-start)/CLOCKS_PER_SEC << endl;
}

int refIndex = 1140;
int curIndex = 1142;

int main() {
    globalConfig.loadFromFile("test/test_online/online_pnp.yaml");
    BaseConfig::initInstance(globalConfig);

    FileInputSource * fileInputSource = new TUMInputSource();
    cout << "device_num: " << fileInputSource->getDevicesNum() << endl;
    cout << "frame_num: " << fileInputSource->getFrameNum() << endl;

    SIFTFeatureExtractor extractor;
    auto ref = allocate_shared<Frame>(Eigen::aligned_allocator<Frame>(), fileInputSource->waitFrame(0, refIndex));
    ref->setDepthBounds(minDepth, maxDepth);
    extractor.extractFeatures(ref, ref->getKps());

    auto cur = allocate_shared<Frame>(Eigen::aligned_allocator<Frame>(), fileInputSource->waitFrame(0, curIndex));
    cur->setDepthBounds(minDepth, maxDepth);
    extractor.extractFeatures(cur, cur->getKps());
    ImageUtil::drawKeypoints(cur->getKps(), cur, workspace+"/kp1.png");

    SIFTFeatureMatcher matcher;
    FeatureMatches featureMatches = matcher.matchKeyPointsPair(ref->getKps(), cur->getKps());
    ImageUtil::drawMatches(featureMatches, ref, cur, workspace+"/matches.png");

    pnpRegistration = new PnPRegistration(globalConfig);

    ConnectionCandidate edge;
    Timer timer = Timer::startTimer("registration");
    registrationPairEdge(featureMatches, &edge, 0, true);
    timer.stopTimer();

    if(!edge.isUnreachable()) {
        ImageUtil::drawMatches(edge.getKxs(), edge.getKys(), ref, cur, workspace+"/projection_matches.png");
        auto pc = ref->calculatePointCloud();
        auto curPc = cur->calculatePointCloud();
        pcl::transformPointCloud(*curPc, *curPc, edge.getTransform());
        *pc += *curPc;
        PointUtil::savePLYPointCloud("/home/liulei/桌面/pnp01.ply", *pc);
    }

    ofstream estimate(workspace+"/online_estimate_pair.txt", ios::out | ios::binary);
    estimate << refIndex << " " << setprecision(9) << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 1 << endl;
    Rotation R;
    Translation t;
    GeoUtil::T2Rt(edge.getTransform(), R, t);
    Eigen::Quaternion<Scalar> q(R);
    estimate << curIndex << " " << setprecision(9) << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

    return 0;
}
