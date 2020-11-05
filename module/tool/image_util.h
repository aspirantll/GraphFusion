//
// Created by liulei on 2020/6/16.
//

#ifndef RTF_IMAGE_UTIL_H
#define RTF_IMAGE_UTIL_H

#include <opencv2/core/core.hpp>
#include "../feature/feature_point.h"

using namespace std;

namespace rtf {

    namespace ImageUtil {

        /**
         * draw key points and save to path
         * @param featurePoints
         * @param path
         */
        void drawKeypoints(SIFTFeaturePoints& featurePoints, shared_ptr<FrameRGB> frame, string path);

        /**
         *
         * @tparam F
         * @param featureMatches
         * @param path
         */
        void drawMatches(FeatureMatches& featureMatches, shared_ptr<FrameRGB> frame1, shared_ptr<FrameRGB> frame2, string path);

        void drawMatches(vector<FeatureKeypoint>& k1, vector<FeatureKeypoint>& k2, shared_ptr<FrameRGB> frame1, shared_ptr<FrameRGB> frame2, string path);

        cv::Mat drawFrame(cv::Mat& im, vector<FeatureKeypoint> &key, int state, int frameIndex, int lostNum);
        /**
         * convert rgb image to raw bits
         * @param img
         * @param rawBits
         */
        void convertToRawBits(const cv::Mat& img, vector<uint8_t>& rawBits);

        /**
         * convert rgb to grey
         * @param r
         * @param g
         * @param b
         * @return
         */
        uint8_t convertRGBToGrey(uint8_t r, uint8_t g, uint8_t b);

        cv::Mat visualizeNormal(shared_ptr<cv::Mat> normalImg);
    }
}


#endif //RTF_IMAGE_UTIL_H
