//
// Created by liulei on 2020/8/31.
//

#ifndef GraphFusion_VIEWER_UTIL_H
#define GraphFusion_VIEWER_UTIL_H

#include <pcl/PolygonMesh.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;

class Viewer {
private:
    shared_ptr<pcl::visualization::PCLVisualizer> viewer;

public:
    Viewer();
    void setMesh(pcl::PolygonMesh* mesh);
    void spin();
    void run();
    bool closed();
};


#endif //GraphFusion_VIEWER_UTIL_H
