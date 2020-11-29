//
// Created by liulei on 2020/7/14.
//

#include "pointcloud_generator.h"


namespace rtf {
    __global__ void generatePointCloud(pcl::gpu::PtrSz<pcl::PointXYZRGBNormal> cloudDevice, cv::cuda::PtrStepSz<uchar4> rgbImg, cv::cuda::PtrStepSz<float> depthImg, cv::cuda::PtrStepSz<float4> normalImg, CUDAPtrs kInv, int* pointCount) {
        // compute row and col
        int index = threadIdx.x + blockDim.x * blockIdx.x;
        int width=depthImg.cols, height=depthImg.rows;
        int row = index/height, col = index%height;
        if(row<0||row>=width||col<0||col>=height) return;
        double depth = depthImg(col, row);

        __shared__ int localCounter;
        if (threadIdx.x == 0) localCounter = 0;
        __syncthreads();

        int addrLocal = -1;
        if(depth!=0) {
            addrLocal = atomicAdd(&localCounter, 1);
        }

        __syncthreads();

        __shared__ int addrGlobal;
        if (threadIdx.x == 0 && localCounter > 0) {
            addrGlobal = atomicAdd(pointCount, localCounter);
        }
        __syncthreads();

        if (addrLocal != -1) {
            int address = addrGlobal + addrLocal;
            if(address>0&&address<height*width) {
                pcl::PointXYZRGBNormal* point = cloudDevice.data + address;
                point->x = depth*(kInv(0,0)*row+kInv(0,1)*col+kInv(0,2));
                point->y = depth*(kInv(1,0)*row+kInv(1,1)*col+kInv(1,2));
                point->z = depth*(kInv(2,0)*row+kInv(2,1)*col+kInv(2,2));

                uchar4 rgba = rgbImg(col, row);
                point->r = rgba.x;
                point->g = rgba.y;
                point->b = rgba.z;

                float4 normal = normalImg(col, row);
                point->normal_x = normal.x;
                point->normal_y = normal.y;
                point->normal_z = normal.z;
            }
        }
    }

    void PointCloudGenerator::generate(shared_ptr<pcl::PointCloud< pcl::PointXYZRGBNormal>> pointCloud, shared_ptr<Camera> camera, shared_ptr<cv::Mat> rgbImage, shared_ptr<cv::Mat> depthImage, shared_ptr<cv::Mat> normalImage, double minDepth, double maxDepth) {
        auto cameraModel = camera->getCameraModel();
        int width=camera->getWidth(), height=camera->getHeight();

        pcl::gpu::DeviceArray<pcl::PointXYZRGBNormal> cloudDevice(width*height);
        cv::cuda::GpuMat depthImg, rgbImg, normalImg;
        depthImg.upload(*depthImage);
        rgbImg.upload(*rgbImage);
        normalImg.upload(*normalImage);

        CUDAMatrixs kInv(3, 3);
        Eigen::MatrixXf tempK = camera->getReverseK().cast<CudaScalar>();

        kInv.upload(tempK);

        int * devicePointCount;
        CUDA_CHECKED_CALL(cudaMalloc(&devicePointCount, sizeof(int)));
        int pointCount = 0;
        CUDA_CHECKED_CALL(cudaMemcpyAsync(devicePointCount, &pointCount, sizeof(int), cudaMemcpyHostToDevice, stream));

        CUDA_LINE_BLOCK(width*height);
        generatePointCloud<<<grid, block, 0, stream>>>(cloudDevice, rgbImg, depthImg, normalImg, kInv, devicePointCount);
        CUDA_CHECKED_NO_ERROR();

        CUDA_CHECKED_CALL(cudaMemcpyAsync(&pointCount, devicePointCount, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECKED_CALL(cudaFree(devicePointCount));

        pointCloud->resize(width*height);
        cloudDevice.download(pointCloud->points.data());
        pointCloud->resize(pointCount);
    }
}