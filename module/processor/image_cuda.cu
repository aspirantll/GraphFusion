//
// Created by liulei on 2020/6/4.
//

#include "image_cuda.cuh"

#define LUMA_REC709(r, g, b)	(0.2126F * r + 0.7152F * g + 0.0722F * b)

#define GREY(r, g, b) ((uint8_t)(LUMA_REC709(r, g, b) + 0.5F))
using namespace cv;

namespace rtf {
    texture<float, cudaTextureType2D, cudaReadModeElementType> depthRef;

    __global__ void medianKernel(ushort **pSrcImg, ushort *pDstImg, int width, int height, int imgNum, ushort minDepth,
                                 ushort maxDepth) {
        int col = threadIdx.x + blockDim.x * blockIdx.x;
        int row = threadIdx.y + blockDim.y * blockIdx.y;
        int index = row * width + col;

        ushort *p = new ushort[imgNum], temp;
        // load the elements
        for (int i = 0; i < imgNum; i++) {
            p[i] = pSrcImg[i][index];
        }

        for (int i = 0; i < imgNum / 2 + 1; i++) {
            for (int j = imgNum - 1; j > i; j--) {
                if (p[j] < p[j - 1]) {
                    temp = p[j];
                    p[j] = p[j - 1];
                    p[j - 1] = temp;
                }
            }
        }

        ushort median = p[imgNum / 2];
        if (median < minDepth || median > maxDepth) {
            median = 0;
        }

        pDstImg[index] = median;
        delete[] p;
    }


    namespace DepthFilterCUDA {


        void
        medianFilter(vector<shared_ptr<Mat>> depthImages, shared_ptr<Mat> output, ushort minDepth, ushort maxDepth) {
            int frameNum = depthImages.size();
            int width = output->size[0];
            int height = output->size[1];

            ushort **pImg = (ushort **) malloc(sizeof(ushort *) * frameNum);

            ushort **pDevice;
            ushort *pDeviceData;
            ushort *pDstImgData;

            cudaMalloc(&pDstImgData, width * height * sizeof(ushort));
            cudaMalloc(&pDevice, sizeof(ushort *) * frameNum);
            cudaMalloc(&pDeviceData, sizeof(ushort) * width * height * frameNum);

            for (int i = 0; i < frameNum; i++) {
                pImg[i] = pDeviceData + i * width * height;
                cudaMemcpyAsync(pImg[i], depthImages[i]->data, sizeof(ushort) * width * height, cudaMemcpyHostToDevice, stream);
            }

            CUDA_CHECKED_CALL(cudaMemcpyAsync(pDevice, pImg, sizeof(ushort *) * frameNum, cudaMemcpyHostToDevice, stream));

            CUDA_IMG_BLOCK(width, height);
            medianKernel<<<grid, block, 0, stream>>>(pDevice, pDstImgData, width, height, frameNum, minDepth, maxDepth);
            CUDA_CHECKED_NO_ERROR();

            CUDA_CHECKED_CALL(
                    cudaMemcpyAsync(output->data, pDstImgData, width * height * sizeof(ushort), cudaMemcpyDeviceToHost, stream));

            cudaFree(pDevice);
            cudaFree(pDeviceData);
            cudaFree(pDstImgData);
            free(pImg);
        }
    }


    __global__ void depthConversionKernel(cv::cuda::PtrStepSz<ushort> src, cv::cuda::PtrStepSz<float> dst, float ds, ushort minDepth, ushort maxDepth) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if(x>=src.cols||y>=src.rows) return;

        ushort depth = src(y, x);
        if(depth<=minDepth||depth>=maxDepth) {
            dst(y, x) = 0.0;
        }else {
            dst(y, x) = depth * ds;
        }
    }

    __global__ void computeNormalKernel(cv::cuda::PtrStepSz<float4> normalImg, float3x3 invK) {
        const int x = threadIdx.x + blockDim.x*blockIdx.x;
        const int y = threadIdx.y + blockDim.y*blockIdx.y;
        const int width = normalImg.cols, height = normalImg.rows;

        if(x>=width||y>=height) return;
        float d = tex2D(depthRef, x, y);
        float d1 = tex2D(depthRef, x+1, y);
        float d2 = tex2D(depthRef, x, y+1);

        float3 p = invK * make_float3(x, y, 1) * d;
        float3 p1 = invK * make_float3(x + 1, y, 1) * d1;
        float3 p2 = invK * make_float3(x, y + 1, 1) * d2;

        float3 v1 = p1-p;
        float3 v2 = p2-p;

        float3 normal = cross(v1, v2);
        // normalized to unit length
        float m = sqrt(dot(normal, normal));
        if(m!=0) {
            normal = normal/m;
        }
        normalImg(y, x) = make_float4(normal, 0);
    }

    void cvtColor(Mat& img, Mat& out, int code) {
        cv::cuda::GpuMat COut(img.rows, img.cols, CV_8UC4), CImg;
        CImg.upload(img);

        cv::cuda::cvtColor(CImg, COut, code);
        COut.download(out);
    }


    void imageConversion(shared_ptr<Mat> depthImage, shared_ptr<Mat> depthOut, shared_ptr<Mat> rgbImage,
                         shared_ptr<Mat> rgbOut, shared_ptr<Mat> normal, float3x3 invK, float ds, ushort minDepth, ushort maxDepth) {
        int width = depthImage->cols, height = depthImage->rows;

        cv::cuda::GpuMat depthCImg;
        depthCImg.upload(*depthImage);

        cvtColor(*rgbImage, *rgbOut, COLOR_BGR2RGBA);

        cv::cuda::GpuMat depthCOut(depthCImg.rows, depthCImg.cols, CV_32F);
        CUDA_IMG_BLOCK(width, height);
        depthConversionKernel<<<grid, block, 0, stream>>>(depthCImg, depthCOut, ds, minDepth, maxDepth);
        CUDA_CHECKED_NO_ERROR();

        depthCOut.download(*depthOut);

        // compute normal
        cutilSafeCall(cudaBindTexture2D(0, &depthRef, depthCOut.ptr<float>(), &depthRef.channelDesc, width, height, sizeof(float) * width));
        cv::cuda::GpuMat normalCImg(height, width, CV_32FC4);
        computeNormalKernel<<<grid, block, 0, stream>>>(normalCImg, invK);
        CUDA_CHECKED_NO_ERROR();
        normal->create(height, width, CV_32FC4);
        normalCImg.download(*normal);
    }



    __global__ void convertToRawBitsKernel(cv::cuda::PtrStepSz<uchar4> rgbImg, uint8_t * deviceBits) {
        int x = threadIdx.x + blockDim.x * blockIdx.x;
        int y = threadIdx.y + blockDim.y * blockIdx.y;

        if(x>=rgbImg.cols||y>=rgbImg.rows) return;

        uchar4 rgba = rgbImg(y, x);
        deviceBits[y*rgbImg.cols+x] = GREY(rgba.x, rgba.y, rgba.z);
    }

    void convertToRawBits(shared_ptr<Mat> rgbImg, vector<uint8_t>& rawBits) {
        int width = rgbImg->cols;
        int height = rgbImg->rows;

        rawBits.resize(width*height);

        uint8_t * deviceBits;
        CUDA_CHECKED_CALL(cudaMalloc(&deviceBits, width*height*sizeof(uint8_t)));

        cv::cuda::GpuMat rgbCImg;
        rgbCImg.upload(*rgbImg);

        CUDA_IMG_BLOCK(width, height);
        convertToRawBitsKernel<<<grid, block, 0, stream>>>(rgbCImg, deviceBits);
        CUDA_CHECKED_NO_ERROR();

        CUDA_CHECKED_CALL(cudaMemcpyAsync(rawBits.data(), deviceBits, width*height*sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECKED_CALL(cudaFree(deviceBits));
    }


    void computeNormalFromDepthImg(shared_ptr<cv::Mat> depthImg, shared_ptr<cv::Mat> normalImg, float3x3 K) {
        const int height=depthImg->rows, width = depthImg->cols;
        cv::cuda::GpuMat depthGImg(height, width, CV_32F), normalGImg(height, width, CV_32FC4);
        depthGImg.upload(*depthImg);
        cutilSafeCall(cudaBindTexture2D(0, &depthRef, depthGImg.ptr<float>(), &depthRef.channelDesc, width, height, sizeof(float) * width));
        CUDA_MAT_BLOCK(width, height);
        computeNormalKernel<<<grid, block, 0, stream>>>(normalGImg, K);
        CUDA_CHECKED_NO_ERROR();
        normalImg->create(height, width, CV_32FC4);
        normalGImg.download(*normalImg);
    }
}