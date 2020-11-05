//
// Created by liulei on 2020/5/10.
//

#ifndef RTF_FRAME_TYPES_H
#define RTF_FRAME_TYPES_H

#include "base_types.h"
#include "camera.h"
#include "../tool/geo_util.h"
#include "../tool/file_util.h"
#include "../tool/string_util.h"
#include "../controller/config.h"
#include "point_types.h"

#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <pcl/point_cloud.h>

#include "../tool/point_util.h"

using namespace std;


namespace rtf {
    class FrameBase: public Serializable{
    protected:
        u_int32_t frameId;
        u_int32_t frameIndex;
        shared_ptr<Camera> camera;
        FrameBase() {}
    public:
        /**
         * deserialize obj from string
         * @param serStr serialized string
         * @param baseDir the dir for saving internal files
         */
        FrameBase(YAML::Node serNode);

        FrameBase(u_int32_t frameId, shared_ptr<Camera> camera);

        u_int32_t getId();

        shared_ptr<Camera> getCamera();

        shared_ptr<CameraModel> getCameraModel();

        u_int32_t getFrameIndex() const;

        void setFrameIndex(u_int32_t frameIndex);

        YAML::Node serialize();

    };


    class FrameRGB: virtual public FrameBase {
    private:
        const string subdir = "images";
        const string suffix = ".png";

    protected:
        shared_ptr<cv::Mat> rgbImage;

        FrameRGB() {}
    public:
        FrameRGB(YAML::Node serNode);

        FrameRGB(int frameId, shared_ptr<Camera> camera, shared_ptr<cv::Mat> rgbImage);

        void reloadRGBImage();

        void releaseRGBImage();

        shared_ptr<cv::Mat> getRGBImage();

        void setRgbImage(const shared_ptr<cv::Mat> &rgbImage);

        YAML::Node serialize();

    };


    class FrameDepth: virtual public FrameBase {
    private:
        const string subdir = "depths";
        const string suffix = ".png";

    protected:
        shared_ptr<cv::Mat> depthImage;
        shared_ptr<cv::Mat> normalImage;
        double minDepth = 0.1;
        double maxDepth = 2;
        int top = 10;
        int bottom = 10;
        int left = 200;
        int right = 10;

        FrameDepth() {}
    public:
        FrameDepth(YAML::Node serNode);

        FrameDepth(int frameId, shared_ptr<Camera> camera, shared_ptr<cv::Mat> depthImage);

        void reloadDepthImage();

        void releaseDepthImage();

        bool inAlignBox(Point2D pixel);

        void setAlignMargin(int left, int right, int top, int bottom);

        double getDepth(Point2D pixel);

        bool inDepthMask(Point2D pixel);

        void setDepthBounds(double minDepth, double maxDepth);

        void setDepthImage(const shared_ptr<cv::Mat> &depthImage);

        shared_ptr<cv::Mat> getNormalImage();

        shared_ptr<cv::Mat> getDepthImage();

        double getMinDepth() const;

        double getMaxDepth() const;

        YAML::Node serialize();
    };

    class FrameRGBD: public FrameRGB, public FrameDepth {
    protected:
        FrameRGBD() {}
    public:
        FrameRGBD(YAML::Node serNode);

        FrameRGBD(shared_ptr<FrameRGBD> other);

        FrameRGBD(int frameId, shared_ptr<Camera> camera, shared_ptr<cv::Mat> rgbImage, shared_ptr<cv::Mat> depthImage);

        shared_ptr<pcl::PointCloud <pcl::PointXYZRGBNormal>> calculatePointCloud();

        void releaseImages();

        YAML::Node serialize();
    };

    class FrameRGBDT: public FrameRGBD {
    protected:
        Transform T;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        FrameRGBDT(YAML::Node serNode);
        FrameRGBDT(shared_ptr<FrameRGBD> other);
        FrameRGBDT(shared_ptr<FrameRGBD> other,const Transform& t);

        FrameRGBDT(int frameId, shared_ptr<Camera> &camera, shared_ptr<cv::Mat> &rgbImage,
                   shared_ptr<cv::Mat> &depthImage,const Transform& t);

        void setTransform(const Transform &transform);

        Transform getTransform();

        Rotation getRotation();

        Translation getTranslation();

        YAML::Node serialize();
    };
}

#endif //RTF_FRAME_TYPES_H
