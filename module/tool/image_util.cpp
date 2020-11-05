//
// Created by liulei on 2020/6/16.
//

#include "image_util.h"
#include "../core/solver/cuda_math.h"
/**
Extract the luminance channel L from a RGBF image.
Luminance is calculated from the sRGB model using a D65 white point, using the Rec.709 formula :
L = ( 0.2126 * r ) + ( 0.7152 * g ) + ( 0.0722 * b )
Reference :
A Standard Default Color Space for the Internet - sRGB.
[online] http://www.w3.org/Graphics/Color/sRGB
*/
#define LUMA_REC709(r, g, b)    (0.2126F * r + 0.7152F * g + 0.0722F * b)

#define GREY(r, g, b) ((uint8_t)(LUMA_REC709(r, g, b) + 0.5F))

using namespace cv;

namespace rtf {
    namespace ImageUtil {
        void drawKeypoints(SIFTFeaturePoints &featurePoints, shared_ptr<FrameRGB> frame, string path) {
            Mat drawed;
            std::vector<KeyPoint> keypoints;
            for (auto kp: featurePoints.getKeyPoints()) {
                keypoints.emplace_back(Point2f(kp->x, kp->y), 0, -1, 0, 0, -1);
            }
            cv::drawKeypoints(*frame->getRGBImage(), keypoints, drawed);
            imwrite(path, drawed);
        }

        void drawMatches(FeatureMatches &featureMatches, shared_ptr<FrameRGB> frame1, shared_ptr<FrameRGB> frame2,
                         string path) {
            auto img1 = frame1->getRGBImage();
            auto img2 = frame2->getRGBImage();

            vector<DMatch> matches;
            for (auto match: featureMatches.getMatches()) {
                matches.emplace_back(match.getPX(), match.getPY(), 0);
            }

            std::vector<KeyPoint> kp1;
            for (auto kp: featureMatches.getKx()) {
                kp1.emplace_back(Point2f(kp->x, kp->y), 0, -1, 0, 0, -1);
            }

            std::vector<KeyPoint> kp2;
            for (auto kp: featureMatches.getKy()) {
                kp2.emplace_back(Point2f(kp->x, kp->y), 0, -1, 0, 0, -1);
            }

            Mat goodImgMatch;
            drawMatches(*img1, kp1, *img2, kp2, matches, goodImgMatch);
            imwrite(path, goodImgMatch);
        }

        void drawMatches(vector<FeatureKeypoint> &k1, vector<FeatureKeypoint> &k2, shared_ptr<FrameRGB> frame1,
                         shared_ptr<FrameRGB> frame2, string path) {
            auto img1 = frame1->getRGBImage();
            auto img2 = frame2->getRGBImage();

            vector<DMatch> matches;
            for (int i = 0; i < k1.size(); i++) {
                matches.emplace_back(i, i, 0);
            }

            std::vector<KeyPoint> kp1;
            for (auto kp: k1) {
                kp1.emplace_back(Point2f(kp.x, kp.y), 0, -1, 0, 0, -1);
            }

            std::vector<KeyPoint> kp2;
            for (auto kp: k2) {
                kp2.emplace_back(Point2f(kp.x, kp.y), 0, -1, 0, 0, -1);
            }

            Mat goodImgMatch;
            drawMatches(*img1, kp1, *img2, kp2, matches, goodImgMatch);
            imwrite(path, goodImgMatch);
        }

        void drawTextInfo(cv::Mat &im, int nState, int frameIndex, int kpNum, int lostNum, cv::Mat &imText) {
            stringstream s;
            s << "Frame Index: " << frameIndex << ", state:" << (nState == 0 ? "LOST" : "Tracked") << ", Matches: "
              << kpNum << ", LOST num:" << lostNum;
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

            imText = cv::Mat(im.rows + textSize.height + 10, im.cols, im.type());
            im.copyTo(imText.rowRange(0, im.rows).colRange(0, im.cols));
            imText.rowRange(im.rows, imText.rows) = cv::Mat::zeros(textSize.height + 10, im.cols, im.type());
            cv::putText(imText, s.str(), cv::Point(5, imText.rows - 5), cv::FONT_HERSHEY_PLAIN, 1,
                        cv::Scalar(255, 255, 255), 1, 8);
        }

        cv::Mat drawFrame(cv::Mat &im, vector<FeatureKeypoint> &key, int state, int frameIndex, int lostNum) {
            cv::cvtColor(im, im, COLOR_RGBA2BGR);
            const float r = 5;
            const int n = key.size();
            for (int i = 0; i < n; i++) {
                cv::Point2f pt1, pt2;
                pt1.x = key[i].x - r;
                pt1.y = key[i].y - r;
                pt2.x = key[i].x + r;
                pt2.y = key[i].y + r;

                cv::rectangle(im, pt1, pt2, cv::Scalar(0, 255, 0));
                cv::circle(im, cv::Point2f(key[i].x, key[i].y), 2, cv::Scalar(255, 0, 0), -1);
            }

            cv::Mat imWithInfo;
            drawTextInfo(im, state, frameIndex, key.size(), lostNum, imWithInfo);

            return imWithInfo;
        }


        void convertToRawBits(const Mat &img, vector<uint8_t> &rawBits) {
            int height = img.rows;
            int width = img.cols;
            rawBits.resize(height * width, 0);

            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    Vec4b rgba = img.at<Vec4b>(i, j);
                    rawBits[i * width + j] = convertRGBToGrey(rgba[0], rgba[1], rgba[2]);
                }
            }
        }


        uint8_t convertRGBToGrey(uint8_t r, uint8_t g, uint8_t b) {
            return GREY(r, g, b);
        }

        cv::Mat visualizeNormal(shared_ptr<cv::Mat> normalImg) {
            const int rows = normalImg->rows, cols = normalImg->cols;
            cv::Mat visNormal(rows, cols, CV_8UC3);
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    float3 n = make_float3(normalImg->at<float4>(row, col));
                    float3 vn = (n * 0.5f + 0.5f) * 255;
                    visNormal.at<uchar3>(row, col) = make_uchar3((uint) vn.x, (uint) vn.y, (uint) vn.z);
                }
            }
            return visNormal;
        }
    }
}