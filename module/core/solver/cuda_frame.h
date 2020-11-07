//
// Created by liulei on 2020/8/18.
//

#ifndef GraphFusion_CUDA_FRAME_H
#define GraphFusion_CUDA_FRAME_H

#include "cuda_matrix.h"
#include "cuda_math.h"
#include "matrix_conversion.h"
#include "../../datastructure/cuda_types.h"
#include "../../datastructure/frame_types.h"
#include "../../datastructure/view_graph.h"

__device__ inline static float4 convertHomogeneous(float3 src) {
    return make_float4(src.x, src.y, src.z, 1.0);
}


__device__ inline static float3 convertNonHomogeneous(float4 src) {
    return make_float3(src.x/src.w,src.y/src.w,src.z/src.w);
}


class CUDAFrame {
public:
    uchar4 * colorData;
    float * depthData;
    int imageWidth;
    int imageHeight;
    float minDepth;
    float maxDepth;
    mat4f transformation;
    mat4f transformationInverse;
    mat3f intrinsic;
    mat3f intrinsicInverse;

    __host__ __device__ CUDAFrame(rtf::FrameRGBD & frame, cv::cuda::GpuMat& depthImg, cv::cuda::GpuMat& rgbImg)  {
        colorData = rgbImg.ptr<uchar4>();
        depthData = depthImg.ptr<float>();
        imageHeight = frame.getCamera()->getHeight();
        imageWidth = frame.getCamera()->getWidth();
        minDepth = frame.getMinDepth();
        maxDepth = frame.getMaxDepth();

        Eigen::Matrix3f intrinsicTrans = frame.getCamera()->getK().cast<float>();
        intrinsic = MatrixConversion::toCUDA(intrinsicTrans);
        Eigen::Matrix3f intrinsicTransInv = frame.getCamera()->getReverseK().cast<float>();
        intrinsicInverse = MatrixConversion::toCUDA(intrinsicTransInv);
    }


    __host__ void setTransform(const Transform& trans) {
        Eigen::Matrix4f  trans4f = trans.cast<float>();
        transformation = MatrixConversion::toCUDA(trans4f);
        Eigen::Matrix4f trans4fInv = rtf::GeoUtil::reverseTransformation(trans).cast<float>();
        transformationInverse = MatrixConversion::toCUDA(trans4fInv);
    }

    __device__ inline bool isInCameraFrustumApprox(float3 globalPos) const {
        float3 localPos = transformToLocal(globalPos);
        float3 projZO = cameraToProj(localPos);
        projZO = projZO * 0.95;
        return !(projZO.x < -1.0f || projZO.x > 1.0f || projZO.y < -1.0f || projZO.y > 1.0f || projZO.z < 0.0f || projZO.z > 1.0f);
    }

    __device__ inline float3 project(float3 pos) const {
        float3 pixel = intrinsic*pos;
        pixel.x /= pixel.z;
        pixel.y /= pixel.z;
        return pixel;
    }


    __device__ inline float3 cameraToProj(float3& pos) const	{
        float3 pImage = project(pos);

        pImage.x = (2.0f*pImage.x - (imageWidth- 1.0f))/(imageWidth- 1.0f);
        pImage.y = ((imageHeight-1.0f) - 2.0f*pImage.y)/(imageHeight-1.0f);
        pImage.z = cameraToProjZ(pos.z);

        return pImage;
    }


    __device__ inline double cameraToProjZ(double z) const	{
        return (z - minDepth)/(maxDepth - minDepth);
    }

    __device__ inline float3 unProject(int x, int y, float d) const {
        float3 point = make_float3(x, y, 1.0);
        point = intrinsicInverse*point;
        return point*d;
    }

    __device__ inline float3 transformToGlobal(float3 point) const {
        float4 hPoint = convertHomogeneous(point);
        hPoint = transformation*hPoint;
        return convertNonHomogeneous(hPoint);
    }

    __device__ inline float3 transformToLocal(float3 point) const {
        float4 hPoint = convertHomogeneous(point);
        hPoint = transformationInverse*hPoint;
        return convertNonHomogeneous(hPoint);
    }
};



#endif //GraphFusion_CUDA_FRAME_H
