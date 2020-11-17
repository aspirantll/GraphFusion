//
// Created by liulei on 2020/5/18.
//

#include "camera.h"
#include "../controller/config.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

namespace rtf {
    /* CameraModel start */
    Vector3 CameraModel::unproject(double x, double y, double depth) {
        Vector2 pixel;
        pixel << x, y;
        return this->unproject(pixel, depth);
    }

    Vector2 CameraModel::project(double x, double y, double z) {
        Vector3 point;
        point << x, y, z;
        return this->project(point);
    }

    Vector3 CameraModel::getRay(double x, double y) {
        Vector2 pixel;
        pixel << x, y;
        return this->getRay(pixel);
    }
    /* CameraModel end */


    /* PinholeCameraModel start */

    PinholeCameraModel::PinholeCameraModel(Intrinsic K, Intrinsic reverseK,int width, int height, double depthScale) {
        this->K = K;
        this->reverseK = reverseK;
        this->depthScale = depthScale>0? depthScale:0.001;
        this->width = width;
        this->height = height;
    }

    //2D to 3D
    Vector3 PinholeCameraModel::unproject(Vector2 pixel, double depth)  {
        Vector3 uv(pixel[0] * depth, pixel[1] * depth, depth);
        return reverseK * uv;
    }

    //3D to 2D
    Vector2 PinholeCameraModel::project(Vector3 point) {
        Vector3 hPixel = K * point;
        Vector2 pixel;
        pixel << hPixel[0]/hPixel[2], hPixel[1]/hPixel[2];
        return pixel;
    }


    Vector2 PinholeCameraModel::projectWithJacobi(Vector3 point, Eigen::Matrix<Scalar, 2, 3>& jacobi) {
        Vector3 hPixel = K * point;

        jacobi.row(0) = K.row(0)*hPixel[2] - hPixel[0]/(hPixel[2]*hPixel[2])*K.row(2);
        jacobi.row(1) = K.row(1)*hPixel[2] - hPixel[1]/(hPixel[2]*hPixel[2])*K.row(2);

        Vector2 pixel;
        pixel << hPixel[0]/hPixel[2], hPixel[1]/hPixel[2];
        return pixel;
    }

    //get ray
    Vector3 PinholeCameraModel::getRay(Vector2 pixel) {
        Vector3 hPixel;
        hPixel << int(pixel[0]), int(pixel[1]), 1;
        return reverseK * hPixel;
    }

    //get centre direction of camera
    Vector3 PinholeCameraModel::getCentreDirection() {
        Vector2 center;
        center << width/2, height/2;
        return this->getRay(center);
    }

    //get border direction of camera
    vector<Vector3> PinholeCameraModel::getBorderDirection() {
        int w[] = {1, width-1, 1, width-1};
        int h[] = {1, 1, height-1, height-1};

        vector<Vector3> borderRays;
        for(int i=0; i<4; i++) {
            Vector2 pixel;
            pixel << w[i], h[i];
            borderRays.push_back(this->getRay(pixel));
        }
        return borderRays;
    }
    /* PinholeCameraModel end */


    /* Camera start */

    Camera::Camera(YAML::Node serNode) {
        this->serNum = serNode["serNum"].as<string>();
        this->fx = serNode["fx"].as<double>();
        this->fy = serNode["fy"].as<double>();
        this->cx = serNode["cx"].as<double>();
        this->cy = serNode["cy"].as<double>();
        this->width = serNode["width"].as<int>();
        this->height = serNode["height"].as<int>();
        this->depthScale = serNode["depthScale"].as<double>();
        YAMLUtil::baseArrayDeserialize(serNode["distCoef"], distCoef);
        pinholeCameraModel = make_shared<PinholeCameraModel>(PinholeCameraModel(this->getK(), this->getReverseK(), width, height, depthScale));
        computeBounds();
    }

    Camera::Camera(string serNum, double fx, double fy, double cx, double cy, int width, int height, double depthScale, double distCoef[5])
            : serNum(serNum), fx(fx), fy(fy), cx(cx), cy(cy), width(width), height(height), depthScale(depthScale){
        if(distCoef) {
            this->distCoef[0] = distCoef[0];
            this->distCoef[1] = distCoef[1];
            this->distCoef[2] = distCoef[2];
            this->distCoef[3] = distCoef[3];
            this->distCoef[4] = distCoef[4];
        }

        pinholeCameraModel = make_shared<PinholeCameraModel>(PinholeCameraModel(this->getK(), this->getReverseK(), width, height, depthScale));
        computeBounds();
    }

    void Camera::computeBounds() {
        if(distCoef[0]!=0.0)
        {
            cv::Mat mat(4,2,CV_32F);
            mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
            mat.at<float>(1,0)=width; mat.at<float>(1,1)=0.0;
            mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=height;
            mat.at<float>(3,0)=width; mat.at<float>(3,1)=height;

            cv::Mat K;
            cv::eigen2cv(getK(), K);
            // Undistort corners
            mat=mat.reshape(2);
            cv::undistortPoints(mat, mat, K, getDistCoef(), cv::Mat(), K);
            mat=mat.reshape(1);

            minX = min(mat.at<float>(0,0),mat.at<float>(2,0));
            maxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
            minY = min(mat.at<float>(0,1),mat.at<float>(1,1));
            maxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

        }
        else
        {
            minX = 0.0f;
            maxX = width;
            minY = 0.0f;
            maxY = height;
        }
    }

    string Camera::getSerNum() {
        return this->serNum;
    }

    Intrinsic Camera::getK() {
        Intrinsic K;
        K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
        return K;
    }

    Intrinsic Camera::getReverseK() {
        Intrinsic reverseK;
        reverseK << 1/fx, 0, -cx/fx, 0, 1/fy, -cy/fy, 0, 0, 1;
        return reverseK;
    }

    cv::Mat Camera::getDistCoef() {
        cv::Mat distCoefMat(5, 1, CV_32F);
        distCoefMat.at<float>(0) = distCoef[0];
        distCoefMat.at<float>(1) = distCoef[1];
        distCoefMat.at<float>(2) = distCoef[2];
        distCoefMat.at<float>(3) = distCoef[3];
        distCoefMat.at<float>(4) = distCoef[4];
        return distCoefMat;
    }

    int Camera::getWidth() {
        return this->width;
    }

    int Camera::getHeight() {
        return this->height;
    }

    double Camera::getDepthScale() {
        return this->depthScale;
    }

    shared_ptr<CameraModel> Camera::getCameraModel(CameraModelType type) {
        if(type == DEFAULT) {
            type = BaseConfig::getInstance()->cameraModelType;
        }

        switch (type) {
            case PINHOLE:
                return this->pinholeCameraModel;
            default:
                throw invalid_argument("invalid camera model type");
        }

    }

    YAML::Node Camera::serialize() {
        YAML::Node node;

        node["serNum"] = this->serNum;
        node["fx"] = this->fx;
        node["fy"] = this->fy;
        node["cx"] = this->cx;
        node["cy"] = this->cy;
        node["width"] = this->width;
        node["height"] = this->height;
        node["depthScale"] = this->depthScale;
        node["distCoef"] = YAMLUtil::baseArraySerialize(distCoef, 5);

        return node;
    }

    double Camera::getFx() const {
        return fx;
    }

    double Camera::getFy() const {
        return fy;
    }

    double Camera::getCx() const {
        return cx;
    }

    double Camera::getCy() const {
        return cy;
    }

    float Camera::getMinX() const {
        return minX;
    }

    float Camera::getMaxX() const {
        return maxX;
    }

    float Camera::getMinY() const {
        return minY;
    }

    float Camera::getMaxY() const {
        return maxY;
    }
    /* Camera end */


    /* CameraFactory start */

    // initialize members
    vector<shared_ptr<Camera>> CameraFactory::cameras = vector<shared_ptr<Camera>>();
    const string CameraFactory::filename = "camera.yaml";

    // member function implements
    void CameraFactory::readCameras() {
        string baseDir = BaseConfig::getInstance()->workspace;
        string cameraYamlPath = FileUtil::joinPath({baseDir, filename});
        if(FileUtil::exist(cameraYamlPath)) {
            YAML::Node root = YAMLUtil::loadYAML(cameraYamlPath);

            for(auto cameraNode=root.begin(); cameraNode!=root.end(); ++cameraNode) {
                CameraFactory::cameras.push_back(make_shared<Camera>(Camera(*cameraNode)));
            }
        }
    }

    void CameraFactory::writeCameras() {
        string baseDir = BaseConfig::getInstance()->workspace;
        string cameraYamlPath = FileUtil::joinPath({baseDir, filename});
        YAML::Node root;

        for(auto & camera : CameraFactory::cameras) {
            root.push_back(camera->serialize());
        }

        YAMLUtil::saveYAML(cameraYamlPath, root);
    }

    bool CameraFactory::addCamera(shared_ptr<Camera> camera) {
        if(getCamera(camera->getSerNum()) != nullptr) {
            return false;
        }

        CameraFactory::cameras.push_back(camera);

        writeCameras();
        return true;
    }

    void CameraFactory::delCamera(string serNum) {
        // delete element when for each
        auto cameraIter=CameraFactory::cameras.begin();

        while(cameraIter!=CameraFactory::cameras.end()) {
            if((*cameraIter)->getSerNum() == serNum) {
                cameraIter = CameraFactory::cameras.erase(cameraIter);
            } else {
                ++cameraIter;
            }
        }
        writeCameras();
    }

    shared_ptr<Camera> CameraFactory::getCamera(string serNum) {
        for(auto & cameraPtr : CameraFactory::cameras) {
            if(cameraPtr->getSerNum() == serNum) {
                return cameraPtr;
            }
        }

        return nullptr;
    }

    int CameraFactory::locateCamera(string serNum) {
        for(int i=0; i < CameraFactory::cameras.size(); i++) {
            if(CameraFactory::cameras[i]->getSerNum() == serNum) {
                return i;
            }
        }

        return -1;
    }

    int CameraFactory::getCameraNum() {
        return CameraFactory::cameras.size();
    }
    /* CameraFactory start */
}

