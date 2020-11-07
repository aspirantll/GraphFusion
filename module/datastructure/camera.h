//
// Created by liulei on 2020/5/18.
//

#ifndef GraphFusion_CAMERA_H
#define GraphFusion_CAMERA_H

#include "base_types.h"
#include "../tool/file_util.h"
#include "../tool/yaml_util.h"
#include <opencv2/core.hpp>

#include <map>
#include <memory>
#include <iostream>
#include <Eigen/Core>

using namespace std;

namespace rtf {
    enum CameraModelType {
        DEFAULT,
        PINHOLE,
        GENERIC
    };

    class CameraModel {
    public:

        //2D to 3D
        virtual Vector3 unproject(Vector2 pixel, double depth) = 0;
        Vector3 unproject(double x, double y, double depth);


        //3D to 2D
        virtual Vector2 project(Vector3 point) = 0;
        Vector2 project(double x, double y, double z);

        virtual Vector2 projectWithJacobi(Vector3 point, Eigen::Matrix<Scalar, 2, 3>& jacobi) = 0;

        //return ray
        virtual Vector3 getRay(Vector2 pixel) = 0;
        Vector3 getRay(double x, double y);

        //get centre direction of camera
        virtual Vector3 getCentreDirection() = 0;

        //get border direction of camera
        virtual vector<Vector3> getBorderDirection() = 0;
    };



    class PinholeCameraModel : public CameraModel{
    private:
        Intrinsic K;
        Intrinsic reverseK;
        int width;
        int height;
        double depthScale;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        PinholeCameraModel(Intrinsic K, Intrinsic reverseK, int width, int height, double depthScale);

        //2D to 3D
        Vector3 unproject(Vector2 pixel, double depth) override ;

        //3D to 2D
        Vector2 project(Vector3 point) override ;

        Vector2 projectWithJacobi(Vector3 point, Eigen::Matrix<Scalar, 2, 3>& jacobi);

        //get ray
        Vector3 getRay(Vector2 pixel) override ;

        //get centre direction of camera
        Vector3 getCentreDirection() override ;

        //get border direction of camera
        vector<Vector3> getBorderDirection() override ;


    };


    class Camera : public Serializable{
    private:
        // serNum for camera
        string serNum;
        // the intrinsics for pinhole model
        double fx;
        double fy;
        double cx;
        double cy;
        // image size
        int width;
        int height;
        // the depth scale
        double depthScale;
        double distCoef[5] = {0.0};

        // camera model
        shared_ptr<PinholeCameraModel> pinholeCameraModel;

    public:
        // align box
        int alignLeftMargin = 0;
        int alignRightMargin = 0;
        int alignTopMargin = 0;
        int alignBottomMargin = 0;

        Camera(YAML::Node serNode);

        Camera(string serNum, double fx, double fy, double cx, double cy, int width, int height, double depthScale=-1, double distCoef[5]={0});

        string getSerNum();

        Intrinsic getK();

        Intrinsic getReverseK();

        cv::Mat getDistCoef();

        double getFx() const;

        double getFy() const;

        double getCx() const;

        double getCy() const;

        int getWidth();

        int getHeight();

        double getDepthScale();

        shared_ptr<CameraModel> getCameraModel(CameraModelType type=DEFAULT);

        YAML::Node serialize();
    };


    class CameraFactory {
    private:
        static vector<shared_ptr<Camera>> cameras;

        static const string filename;

    public:
        static void readCameras();

        static void writeCameras();

        static bool addCamera(shared_ptr<Camera> camera);

        static void delCamera(string serNum);

        static shared_ptr<Camera> getCamera(string serNum);

        static int locateCamera(string serNum);

        static int getCameraNum();
    };
}


#endif //GraphFusion_CAMERA_H
