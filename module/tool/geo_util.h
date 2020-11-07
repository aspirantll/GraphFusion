//
// Created by vipl on 2020/1/15.
//

#ifndef GraphFusionUSION_GEOUTIL_H
#define GraphFusionUSION_GEOUTIL_H

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <sophus/rxso3.hpp>
#include <sophus/se2.hpp>
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>
#include <sophus/so2.hpp>
#include <sophus/so3.hpp>
#include <cmath>

#include "../datastructure/base_types.h"

namespace rtf {
    namespace GeoUtil {
        //D.Chen warped from vis
        // Averages the given Sophus::SE3 transformations.
        Sophus::SE3d AverageSE3(int count, Sophus::SE3d *transformations);
        Sophus::SE3d AverageSE3(Sophus::SE3d *transformations,int circleNum,int circleSize);

        /**
         * convert T to R and t
         * @param T
         * @param R
         * @param t
         */
        void T2Rt(const Transform T, Rotation &R, Translation &t);


        /**
         * convert R and t to T
         * @param T
         * @param R
         * @param t
         */
        void Rt2T(const Rotation R, const Translation t, Transform &T);

        /**
         * convert R and t to project matrix
         * @param P: 3x4
         * @param R: 3x3
         * @param t: 3x1
         */
        void Rt2P(const Rotation &R, const Translation &t, Eigen::Matrix<Scalar, 3, 4> &P);

        /**
         * reverse the transformation
         * @param src
         * @return
         */
        Transform reverseTransformation(Transform src);

        /**
         * calculate depth for 3d point
         * @param P
         * @param point3D
         * @return
         */
        double calculateDepth(const Eigen::Matrix<Scalar, 3, 4> &P, const Vector3 &point3D);

        /**
         * compute camera position
         * @param trans
         * @param ow
         */
        void computeOW(Transform trans, Vector3& ow);

        /**
         * check rotation is orthogonal
         * @param trans
         * @return
         */
        bool validateTransform(Transform trans);
    }
}



#endif //GraphFusionUSION_GEOUTIL_H
