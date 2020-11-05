//
// Created by liulei on 2020/10/11.
//

#ifndef RTF_DOWNSAMPLE_H
#define RTF_DOWNSAMPLE_H

#include "../feature/feature_point.h"

using namespace std;
namespace rtf {
    vector<int> downSampleFeatureMatches(FeatureMatches& fm, float gridSize);
}


#endif //RTF_DOWNSAMPLE_H
