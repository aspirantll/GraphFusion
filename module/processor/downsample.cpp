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

    vector<int> downSampleFeatureMatches(FeatureMatches& fm, vector<int> inlier, Transform trans, float gridSize) {
        int num = inlier.size();
        FeatureKeypoints kxs = fm.getKx();
        FeatureKeypoints kys = fm.getKy();
        shared_ptr<Camera> cx = fm.getCx();
        shared_ptr<Camera> cy = fm.getCy();
        int height=cy->getHeight();
        int cols = floor(height/gridSize);

        map<int, vector<int>> gridGroup;
        for(int i=0; i<num; i++) {
            int ind = fm.getMatch(inlier[i]).getPX();
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
            double dist = numeric_limits<double>::infinity();
            for(int ind: mit.second) {
                auto match = fm.getMatch(ind);
                shared_ptr<FeatureKeypoint> kx = kxs[match.getPX()];
                shared_ptr<FeatureKeypoint> ky = kys[match.getPY()];

                Vector3 py = cy->getCameraModel()->unproject(ky->x, ky->y, ky->z);
                Vector2 qy = cx->getCameraModel()->project((trans*py.homogeneous()).block<3,1>(0,0));
                double curDist = (qy-kx->toVector2()).norm();
                if(curDist < dist) {
                    selectedIndex = ind;
                    dist = curDist;
                }
            }
            filteredIndexes.emplace_back(selectedIndex);
        }
        return filteredIndexes;
    }

    void downSampleFeatureMatches(vector<FeatureKeypoint> &kxs, vector<FeatureKeypoint> &kys, shared_ptr<Camera> camera, Transform trans, float gridSize) {
        int num = kxs.size();
        int height=camera->getHeight();
        int cols = floor(height/gridSize);

        map<int, vector<int>> gridGroup;
        for(int i=0; i<num; i++) {
            auto kp = kxs[i];
            int gridx = floor(kp.x/gridSize);
            int gridy = floor(kp.y/gridSize);
            int gridIndex = gridx*cols+gridy;
            if(!gridGroup.count(gridIndex)) {
                gridGroup.insert(map<int, vector<int>>::value_type(gridIndex, vector<int>()));
            }
            gridGroup[gridIndex].emplace_back(i);
        }

        vector<int> filteredIndexes;
        for(auto mit: gridGroup) {
            int selectedIndex = -1;
            double dist = numeric_limits<double>::infinity();
            for(int ind: mit.second) {
                const FeatureKeypoint& kx = kxs[ind];
                const FeatureKeypoint& ky = kys[ind];

                Vector3 py = camera->getCameraModel()->unproject(ky.x, ky.y, ky.z);
                Vector2 qy = camera->getCameraModel()->project((trans*py.homogeneous()).block<3,1>(0,0));
                double curDist = (qy-kx.toVector2()).norm();
                if(curDist < dist) {
                    selectedIndex = ind;
                    dist = curDist;
                }
            }
            filteredIndexes.emplace_back(selectedIndex);
        }

        vector<FeatureKeypoint> nKxs(kxs.begin(), kxs.end()), nKys(kys.begin(), kys.end());
        kxs.clear();
        kys.clear();
        for(int ind: filteredIndexes) {
            kxs.emplace_back(nKxs[ind]);
            kys.emplace_back(nKys[ind]);
        }
    }
}