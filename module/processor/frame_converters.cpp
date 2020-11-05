//
// Created by liulei on 2020/5/21.
//

#include "frame_converters.h"
#include "image_cuda.cuh"
#include "../core/solver/matrix_conversion.h"

using namespace cv;
namespace rtf {

    void FrameConverters::convertImageType(shared_ptr<FrameRGBD> frame) {
        auto depthImg = frame->getDepthImage();
        auto rgbImg = frame->getRGBImage();
        auto normalImg = frame->getNormalImage();
        auto camera = frame->getCamera();

        shared_ptr<Mat> depthOut = make_shared<Mat>(depthImg->rows, depthImg->cols, CV_32F);
        shared_ptr<Mat> rgbOut = make_shared<Mat>(rgbImg->rows, rgbImg->cols, CV_8UC4);
        float3x3 invK = MatrixConversion::toCUDA(camera->getReverseK());

        imageConversion(depthImg, depthOut, rgbImg, rgbOut, normalImg, invK, camera->getDepthScale(), frame->getMinDepth(),
                        frame->getMaxDepth());

        frame->setRgbImage(rgbOut);
        frame->setDepthImage(depthOut);

        frame->setDepthBounds(camera->getDepthScale()*frame->getMinDepth(), camera->getDepthScale()*frame->getMaxDepth());
    }
}