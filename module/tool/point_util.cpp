//
// Created by liulei on 2020/6/8.
//

#include "point_util.h"
#include <pcl/io/ply_io.h>

namespace rtf {
    namespace PointUtil {
        cv::Mat vec2Mat(vector<Point3D> &pointVec) {
            cv::Mat mat(pointVec.size(), 3, CV_64FC1);
            for(int i=0; i<pointVec.size(); i++) {
                Point3D point = pointVec[i];
                mat.at<double>(i, 0) = point.x;
                mat.at<double>(i, 1) = point.y;
                mat.at<double>(i, 2) = point.z;
            }
            return mat;
        }

        cv::Mat vec2Mat(vector<Point2D> &pixelVec) {
            cv::Mat mat(pixelVec.size(), 2, CV_64FC1);
            for(int i=0; i<pixelVec.size(); i++) {
                Point2D pixel = pixelVec[i];
                mat.at<double>(i, 0) = pixel.x;
                mat.at<double>(i, 1) = pixel.y;
            }
            return mat;
        }

        cv::Mat point2Mat(Point2D pixel) {
            cv::Mat mat(2, 1, CV_64FC1);
            mat.at<double>(0, 0) = pixel.x;
            mat.at<double>(1, 0) = pixel.y;
            return mat;
        }

        cv::Mat point2Mat(Point3D point) {
            cv::Mat mat(3, 1, CV_64FC1);
            mat.at<double>(0, 0) = point.x;
            mat.at<double>(1, 0) = point.y;
            mat.at<double>(2, 0) = point.z;
            return mat;
        }


        MatrixX vec2Matrix(const vector<Point3D> &pointVec) {
            MatrixX matrix(pointVec.size(), 3);
            for(int i=0; i<pointVec.size(); i++) {
                matrix.row(i) = pointVec[i].toVector3();
            }
            return matrix;
        }

        MatrixX vec2Matrix(const vector<Point2D> &pixelVec) {
            MatrixX matrix(pixelVec.size(), 2);
            for(int i=0; i<pixelVec.size(); i++) {
                matrix.row(i) = pixelVec[i].toVector2();
            }
            return matrix;
        }

        Vector3 transformPoint(Vector3 point, Transform trans) {
            return trans.block<3, 3>(0,0)*point + trans.block<3, 1>(0,3);
        }

        Point3D transformPixel(Point3D pixel, Transform trans, shared_ptr<Camera> camera) {
            Vector3 p = transformPoint(camera->getCameraModel()->unproject(pixel.x, pixel.y, pixel.z), trans);
            p = camera->getK()*p;
            return Point3D(p.x()/p.z(), p.y()/p.z(), p.z());
        }



        bool savePLYPointCloud(string path, pcl::PointCloud<pcl::PointXYZRGBNormal>& pointCloud) {
            pcl::PCLPointCloud2 blob;
            pcl::toPCLPointCloud2(pointCloud, blob);
            return pcl::io::savePLYFile(path, blob, Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), false, false)!=-1;
        }

        bool savePLYPointCloud(string path, pcl::PointCloud<pcl::PointXYZRGB>& pointCloud) {
            pcl::PCLPointCloud2 blob;
            pcl::toPCLPointCloud2(pointCloud, blob);
            return pcl::io::savePLYFile(path, blob, Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), false, false)!=-1;
        }

        bool loadPLYPointCloud(string path, pcl::PointCloud<pcl::PointXYZRGB>& pointCloud) {
            return pcl::io::loadPLYFile(path, pointCloud)!=-1;
        }


        bool savePLYMesh(string path, Mesh& mesh) {
            return pcl::io::savePLYFile(path, mesh)!=-1;
        }

        bool loadPLYMesh(string path, Mesh& mesh) {
            return pcl::io::loadPLYFile(path, mesh)!=-1;
        }


    }
}