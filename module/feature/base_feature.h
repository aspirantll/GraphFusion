//
// Created by liulei on 2020/6/5.
//

#ifndef GraphFusion_BASE_FEATURE_H
#define GraphFusion_BASE_FEATURE_H

#include "feature_point.h"
#include "../tool/image_util.h"
#include "../datastructure/view_graph.h"

namespace rtf {


    template<class S, class T> class FeatureExtractor {
    public:
        virtual void extractFeatures(shared_ptr<S> s, T& t) = 0;
    };


    template<class K> class FeatureMatcher {
    public:

        virtual FeatureMatches matchKeyPointsPair(K& k1, K& k2) = 0;

        vector<FeatureMatches> match(vector<K>& ks, FeatureMatchStrategy strategy=FeatureMatchStrategy::SEQUENCE) {
            int frameNum = ks.size();
            // extract the features for every frame
            vector<FeatureMatches> siftVec;
            if(strategy == FeatureMatchStrategy::SEQUENCE) {
                for(int i=0; i<frameNum-1; i++) {
                    siftVec.emplace_back(matchKeyPointsPair(ks[i], ks[i+1]));
                }
            } else if(strategy == FeatureMatchStrategy::FULL) {
                for(int i=0; i<frameNum; i++) {
                    for(int j=0; j<i; j++) {
                        siftVec.emplace_back(matchKeyPointsPair(ks[j], ks[i]));
                    }
                }
            }
            return siftVec;
        }


    };

}


#endif //GraphFusion_BASE_FEATURE_H
