//
// Created by liulei on 2020/11/12.
//
#include "registrations.h"
#include "../../processor/downsample.h"
#include "../.././thirdparty/ICPCUDA/ICPOdometry.h"

namespace rtf {
    PairwiseICP::PairwiseICP(const GlobalConfig& config) {
        rmsThreshold = config.rmsThreshold;
        relaxtion = config.relaxtion;
        distTh = 0.01;
        minInliers = config.kMinInliers;
    }

    RegReport PairwiseICP::icp(Transform trans, shared_ptr<Frame> fx, shared_ptr<Frame> fy) {
        shared_ptr<Camera> camera = fx->getCamera();
        ICPOdometry icpOdom(camera->getWidth(), camera->getHeight(), camera->getCx(), camera->getCy(), camera->getFx(), camera->getFy());
        icpOdom.initICPModel(fx->getDepthImage()->ptr<float>());
        icpOdom.initICP(fy->getDepthImage()->ptr<float>());

        icpOdom.getIncrementalTransformation(trans, 96, 96);

        RegReport report;
        report.success = icpOdom.lastInliers>minInliers;
        report.cost = icpOdom.lastError;
        report.inlierNum = icpOdom.lastInliers;
        report.T = trans;
        report.iterations = 1;
        return report;
    }
}
