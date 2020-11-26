//
// Created by liulei on 2020/11/26.
//

#ifndef GRAPHFUSION_OPTIMIZERT_H
#define GRAPHFUSION_OPTIMIZERT_H

#include "../../datastructure/view_graph.h"

namespace rtf {
    class Optimizer {
    public:
        static void poseGraphOptimize(ViewGraph &viewGraph, const vector<pair<int, int> >& loops);

        static void poseGraphOptimizeCeres(ViewGraph &viewGraph, const vector<pair<int, int> >& loops);
    };
}


#endif //GRAPHFUSION_OPTIMIZERT_H
