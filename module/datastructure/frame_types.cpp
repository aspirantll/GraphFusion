//
// Created by liulei on 2020/5/19.
//

#include "frame_types.h"
#include "config.h"
#include "../processor/pointcloud_generator.h"

namespace rtf {
    /** FrameBase start */
    FrameBase::FrameBase(YAML::Node serNode) {
        this->frameId = serNode["frameId"].as<int>();
        string serNum = serNode["camera"].as<string>();
        this->camera = CameraFactory::getCamera(serNum);
    }

    FrameBase::FrameBase(u_int32_t frameId, shared_ptr<Camera> camera) {
        this->frameId = frameId;
        this->camera = camera;
    }

    u_int32_t FrameBase::getId() {
        return frameId;
    }

    shared_ptr<Camera> FrameBase::getCamera() {
        return this->camera;
    }

    shared_ptr<CameraModel> FrameBase::getCameraModel() {
        return camera->getCameraModel();
    }

    YAML::Node FrameBase::serialize() {
        YAML::Node node;

        node["frameId"] = this->frameId;
        node["camera"] = this->camera->getSerNum();

        return node;
    }

    u_int32_t FrameBase::getFrameIndex() const {
        return frameIndex;
    }

    void FrameBase::setFrameIndex(u_int32_t frameIndex) {
        FrameBase::frameIndex = frameIndex;
    }

    /** FrameBase end */


    /** FrameRGB start */
    FrameRGB::FrameRGB(YAML::Node serNode) : FrameBase(serNode) {
        // load image
        string baseDir = BaseConfig::getInstance()->workspace;
        string relativePath = serNode["rgbImage"].as<string>();
        rgbPath = FileUtil::joinPath({baseDir, relativePath});
        this->rgbImage = make_shared<cv::Mat>(imread(rgbPath, cv::IMREAD_COLOR));
    }


    FrameRGB::FrameRGB(int frameId, shared_ptr<Camera> camera, shared_ptr<cv::Mat> rgbImage, string rgbPath) : FrameBase(frameId, camera) {
        this->rgbImage = rgbImage;
        this->rgbPath = rgbPath;
    }

    void FrameRGB::reloadRGBImage() {
        if(rgbPath.empty()) {
            string baseDir = BaseConfig::getInstance()->workspace;
            string serNum = this->camera->getSerNum();
            string relativePath = FileUtil::joinPath(
                    {this->subdir, serNum + "_" + to_string(this->frameId) + this->suffix});
            rgbPath = FileUtil::joinPath({baseDir, relativePath});
        }

        this->rgbImage = make_shared<cv::Mat>(imread(rgbPath, cv::IMREAD_COLOR));
    }

    void FrameRGB::releaseRGBImage() {
        this->rgbImage->release();
        this->rgbImage = nullptr;
    }

    shared_ptr<cv::Mat> FrameRGB::getRGBImage() {
        return this->rgbImage;
    }

    YAML::Node FrameRGB::serialize() {
        string baseDir = BaseConfig::getInstance()->workspace;
        // invoke the parent method
        YAML::Node node = FrameBase::serialize();
        // if target dir not exist, then create
        string targetDir = FileUtil::joinPath({baseDir, this->subdir});
        if (!FileUtil::exist(targetDir)) {
            FileUtil::createDirectory(targetDir);
        }

        // save image
        string serNum = this->camera->getSerNum();
        string relativePath = FileUtil::joinPath(
                {this->subdir, serNum + "_" + to_string(this->frameId) + this->suffix});
        string rgbFilePath = FileUtil::joinPath({baseDir, relativePath});
        if(!FileUtil::exist(rgbFilePath)&&rgbImage!=nullptr)
            imwrite(rgbFilePath, *this->rgbImage);

        // serialize node
        node["rgbImage"] = relativePath;

        return node;
    }

    void FrameRGB::setRgbImage(const shared_ptr<cv::Mat> &rgbImage) {
        FrameRGB::rgbImage = rgbImage;
    }

    /** FrameRGB end */

    /** FrameDepth start */
    FrameDepth::FrameDepth(YAML::Node serNode) : FrameBase(serNode) {
        // load depth
        string baseDir = BaseConfig::getInstance()->workspace;
        string relativePath = serNode["depthImage"].as<string>();
        depthPath = FileUtil::joinPath({baseDir, relativePath});
        this->depthImage = make_shared<cv::Mat>(imread(depthPath, cv::IMREAD_ANYDEPTH));

        this->minDepth = serNode["minDepth"].as<double>();
        this->maxDepth = serNode["maxDepth"].as<double>();
        normalImage = make_shared<cv::Mat>();
    }

    FrameDepth::FrameDepth(int frameId, shared_ptr<Camera> camera, shared_ptr<cv::Mat> depthImage, string depthPath)
            : FrameBase(frameId, camera) {
        this->depthImage = depthImage;
        this->depthPath = depthPath;
        normalImage = make_shared<cv::Mat>();
    }

    double FrameDepth::getDepth(Point2D pixel) {
        return depthImage->at<float>(pixel.y, pixel.x);
    }

    void FrameDepth::reloadDepthImage() {
        if(depthPath.empty()) {
            string baseDir = BaseConfig::getInstance()->workspace;
            string serNum = this->camera->getSerNum();
            string relativePath = FileUtil::joinPath(
                    {this->subdir, serNum + "_" + to_string(this->frameId) + this->suffix});
            depthPath = FileUtil::joinPath({baseDir, relativePath});
        }

        this->depthImage = make_shared<cv::Mat>(imread(depthPath, cv::IMREAD_ANYDEPTH));
        this->minDepth /= camera->getDepthScale();
        this->maxDepth /= camera->getDepthScale();
    }

    void FrameDepth::releaseDepthImage() {
        this->depthImage->release();
        this->depthImage = nullptr;
    }

    bool FrameDepth::inDepthMask(Point2D pixel) {
        double depth = getDepth(pixel);
        return depth >= minDepth && depth <= maxDepth;
    }

    void FrameDepth::setDepthBounds(double minDepth, double maxDepth) {
        this->minDepth = minDepth;
        this->maxDepth = maxDepth;
    }


    shared_ptr<cv::Mat> FrameDepth::getDepthImage() {
        return this->depthImage;
    }

    YAML::Node FrameDepth::serialize() {
        string baseDir = BaseConfig::getInstance()->workspace;
        // invoke the parent method
        YAML::Node node = FrameBase::serialize();
        // if target dir not exist, then create
        string targetDir = FileUtil::joinPath({baseDir, this->subdir});
        if (!FileUtil::exist(targetDir)) {
            FileUtil::createDirectory(targetDir);
        }

        // save image
        string serNum = this->camera->getSerNum();
        string relativePath = FileUtil::joinPath(
                {this->subdir, serNum + "_" + to_string(this->frameId) + this->suffix});
        string depthPath = FileUtil::joinPath({baseDir, relativePath});
        if(!FileUtil::exist(depthPath))
            imwrite(depthPath, *this->depthImage);

        // serialize node
        node["depthImage"] = relativePath;
        node["minDepth"] = minDepth;
        node["maxDepth"] = maxDepth;
        return node;
    }

    double FrameDepth::getMinDepth() const {
        return minDepth;
    }

    double FrameDepth::getMaxDepth() const {
        return maxDepth;
    }

    void FrameDepth::setDepthImage(const shared_ptr<cv::Mat> &depthImage) {
        FrameDepth::depthImage = depthImage;
    }

    shared_ptr<cv::Mat> FrameDepth::getNormalImage() {
        return normalImage;
    }

    /** FrameDepth end */


    /** FrameRGBD start */
    FrameRGBD::FrameRGBD(YAML::Node serNode) : FrameRGB(serNode), FrameDepth(serNode), FrameBase(serNode) {

    }

    FrameRGBD::FrameRGBD(shared_ptr<FrameRGBD> other) : FrameRGB(*other),
                                                        FrameDepth(*other),
                                                        FrameBase(*other) {

    }

    FrameRGBD::FrameRGBD(int frameId, shared_ptr<Camera> camera, shared_ptr<cv::Mat> rgbImage, shared_ptr<cv::Mat> depthImage)
            : FrameRGB(frameId, camera, rgbImage), FrameDepth(frameId, camera, depthImage), FrameBase(frameId, camera) {

    }

    shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal>> FrameRGBD::calculatePointCloud() {
        shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal>> pointCloudPtr = make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>(
                pcl::PointCloud<pcl::PointXYZRGBNormal>());
        PointCloudGenerator::generate(pointCloudPtr, camera, rgbImage, depthImage, normalImage, minDepth, maxDepth);

        return pointCloudPtr;
    }

    void FrameRGBD::releaseImages() {
        rgbImage->release();
        depthImage->release();
        normalImage->release();
    }

    void FrameRGBD::reloadImages() {
        reloadRGBImage();
        reloadDepthImage();
    }

    YAML::Node FrameRGBD::serialize() {

        YAML::Node rgbNode = FrameRGB::serialize();
        YAML::Node depthNode = FrameDepth::serialize();

        return YAMLUtil::mergeNodes({rgbNode, depthNode});
    }
    /** FrameRGBD end */


    /** FrameRGBDT begin */
    FrameRGBDT::FrameRGBDT(YAML::Node serNode): FrameRGBD(serNode), FrameBase(serNode) {

    }

    void FrameRGBDT::setTransform(const Transform &transform) {
        T = SE3(transform);
    }


    Transform FrameRGBDT::getTransform() {
        return T.matrix();
    }

    SE3 FrameRGBDT::getSE() {
        return T;
    }

    Rotation FrameRGBDT::getRotation() {
        return T.rotationMatrix();
    }


    Translation FrameRGBDT::getTranslation() {
        return T.translation();
    }

    YAML::Node FrameRGBDT::serialize() {
        // invoke the parent method
        YAML::Node node = FrameRGBD::serialize();
        return node;
    }

    FrameRGBDT::FrameRGBDT(int frameId, shared_ptr<Camera> &camera, shared_ptr<cv::Mat> &rgbImage,
                           shared_ptr<cv::Mat> &depthImage, const Transform& t) : FrameRGBD(frameId, camera, rgbImage,
                                                                                             depthImage), FrameBase(frameId, camera) {
        T = SE3(t);
    }

    FrameRGBDT::FrameRGBDT(shared_ptr<FrameRGBD> other, const Transform& t) : FrameRGBD(other), FrameBase(*other) {
        T = SE3(t);
    }

    FrameRGBDT::FrameRGBDT(shared_ptr<FrameRGBD> other): FrameRGBD(other), FrameBase(*other) {
        T = SE3(Transform::Identity());
    }


}