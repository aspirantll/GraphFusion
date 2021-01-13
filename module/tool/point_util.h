//
// Created by liulei on 2020/6/8.
//

#ifndef GraphFusion_POINT_UTIL_H
#define GraphFusion_POINT_UTIL_H

#include "../datastructure/point_types.h"
#include "../datastructure/camera.h"

#include <vector>
#include <opencv2/core/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>

using namespace std;

using Mesh = pcl::PolygonMesh;

namespace rtf {
    namespace PointUtil {
        cv::Mat vec2Mat(vector<Point3D> &pointVec);

        cv::Mat vec2Mat(vector<Point2D> &pixelVec);

        cv::Mat point2Mat(Point2D pixel);

        cv::Mat point2Mat(Point3D point);

        MatrixX vec2Matrix(const vector<Point3D> &pointVec);

        MatrixX vec2Matrix(const vector<Point2D> &pixelVec);

        Vector3 transformPoint(Vector3 point, Transform trans);

        Point3D transformPixel(Point3D pixel, Transform trans, shared_ptr<Camera> camera);

        Vector3 transformPixelToPoint(Point3D pixel, Transform trans, shared_ptr<Camera> camera);

        bool savePLYPointCloud(string path, pcl::PointCloud<pcl::PointXYZRGBNormal>& pointCloud);

        bool savePLYPointCloud(string path, pcl::PointCloud<pcl::PointXYZRGB>& pointCloud);

        bool loadPLYPointCloud(string path, pcl::PointCloud<pcl::PointXYZRGB>& pointCloud);

        bool savePLYMesh(string path, Mesh& mesh);

        bool loadPLYMesh(string path, Mesh& mesh);
    }
}


#endif //GraphFusion_POINT_UTIL_H
