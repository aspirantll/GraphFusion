//
// Created by liulei on 2020/8/31.
//

#include "viewer_util.h"
#include <thread>

Viewer::Viewer() {
    viewer = make_shared<pcl::visualization::PCLVisualizer>("online scanner");
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(0.001);
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 0, 0, 0, 0.01, 0, -0.01, 0);
}

void Viewer::setMesh(pcl::PolygonMesh* mesh) {
    string id = "simple mesh";
    if(viewer->contains(id)) {
        viewer->removePolygonMesh(id);
    }
    viewer->addPolygonMesh(*mesh, id);
    viewer->spinOnce(100);
}

void Viewer::spin() {
    viewer->spin();
}

void Viewer::run() {

}

bool Viewer::closed() {
    return viewer->wasStopped();
}