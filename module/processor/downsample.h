//
// Created by liulei on 2020/10/11.
//

#ifndef GraphFusion_DOWNSAMPLE_H
#define GraphFusion_DOWNSAMPLE_H

#include "../feature/feature_point.h"

using namespace std;
namespace rtf {
    vector<int> downSampleFeatureMatches(FeatureMatches& fm, float gridSize);
}


#endif //GraphFusion_DOWNSAMPLE_H
