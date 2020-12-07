//
// Created by liulei on 2020/10/11.
//

#ifndef GraphFusion_DOWNSAMPLE_H
#define GraphFusion_DOWNSAMPLE_H

#include "../feature/feature_point.h"

using namespace std;
namespace rtf {
    vector<int> downSampleFeatureMatches(FeatureMatches& fm, float gridSize);

    vector<int> downSampleFeatureMatches(FeatureMatches& fm, vector<int> inlier, Transform trans, float gridSize);

    void downSampleFeatureMatches(vector<FeatureKeypoint> &kxs, vector<FeatureKeypoint> &kys, shared_ptr<Camera> camera, Transform trans, float gridSize);
}


#endif //GraphFusion_DOWNSAMPLE_H
