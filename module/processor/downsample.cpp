//
// Created by liulei on 2020/10/11.
//

#include "downsample.h"

namespace rtf {


    vector<int> downSampleFeatureMatches(FeatureMatches& fm, float gridSize) {
        int num = fm.size();
        FeatureKeypoints kxs = fm.getKx();
        FeatureKeypoints kys = fm.getKy();
        shared_ptr<Camera> cx = fm.getCx();
        shared_ptr<Camera> cy = fm.getCy();
        int width = cx->getWidth(), height=cy->getHeight();
        int rows = floor(width/gridSize), cols = floor(height/gridSize);

        map<int, vector<int>> gridGroup;
        for(int i=0; i<num; i++) {
            int ind = fm.getMatch(i).getPX();
            auto kp = kxs[ind];
            int gridx = floor(kp->x/gridSize);
            int gridy = floor(kp->y/gridSize);
            int gridIndex = gridx*cols+gridy;
            if(!gridGroup.count(gridIndex)) {
                gridGroup.insert(map<int, vector<int>>::value_type(gridIndex, vector<int>()));
            }
            gridGroup[gridIndex].emplace_back(i);
        }

        vector<int> filteredIndexes;
        for(auto mit: gridGroup) {
            int selectedIndex = -1;
            double angle = numeric_limits<double>::infinity();
            for(int ind: mit.second) {
                auto match = fm.getMatch(ind);
                shared_ptr<SIFTFeatureKeypoint> kx = dynamic_pointer_cast<SIFTFeatureKeypoint>(kxs[match.getPX()]);
                shared_ptr<SIFTFeatureKeypoint> ky = dynamic_pointer_cast<SIFTFeatureKeypoint>(kys[match.getPY()]);
                double curAngle = abs(kx->ComputeOrientation()-ky->ComputeOrientation());
                if(curAngle<angle) {
                    selectedIndex = ind;
                    angle = curAngle;
                }
            }
            filteredIndexes.emplace_back(selectedIndex);
        }
        return filteredIndexes;
    }
}