//
// Created by vipl on 2020/1/15.
//

#include "geo_util.h"

namespace rtf {
    //D.Chen warped from vis
    /// Averages the given Sophus::SE3 transformations.
    Sophus::SE3d GeoUtil::AverageSE3(int count, Sophus::SE3d *transformations) {
        Eigen::Matrix<double , 3, 3> accumulated_rotations;
        accumulated_rotations.setZero();
        Eigen::Matrix<double, 3, 1> accumulated_translations;
        accumulated_translations.setZero();

        for (int i = 0; i < count; ++ i) {
            accumulated_rotations += transformations[i].so3().matrix();
            accumulated_translations += transformations[i].translation();
        }

        Sophus::SE3d result;
        Eigen::JacobiSVD<Eigen::Matrix<double, 3, 3>> svd(accumulated_rotations, Eigen::ComputeFullU | Eigen::ComputeFullV);
        result.setRotationMatrix((svd.matrixU() * svd.matrixV().transpose()));
        result.translation() = (accumulated_translations / (static_cast<double>(1) * count));
        return result;
    }

    Sophus::SE3d GeoUtil::AverageSE3(Sophus::SE3d *transformations,int circleNum,int circleSize) {

        Eigen::Matrix<double , 3, 3> accumulated_rotations;
        accumulated_rotations.setZero();
        Eigen::Matrix<double, 3, 1> accumulated_translations;
        accumulated_translations.setZero();
        double halfCircle = circleSize/2.0;
        double weight = exp(-(circleNum-halfCircle)*(circleNum-halfCircle)/2);
        accumulated_rotations += transformations[0].so3().matrix()*weight;
        accumulated_rotations += transformations[1].so3().matrix();
        accumulated_translations += transformations[0].translation()*weight;
        accumulated_translations += transformations[1].translation();
        Sophus::SE3d result;
        Eigen::JacobiSVD<Eigen::Matrix<double, 3, 3>> svd(accumulated_rotations, Eigen::ComputeFullU | Eigen::ComputeFullV);
        result.setRotationMatrix((svd.matrixU() * svd.matrixV().transpose()));
        result.translation() = (accumulated_translations / (static_cast<double>(1) * (weight+1)));
        return result;
    }


    void GeoUtil::T2Rt(const Transform T, Rotation &R, Translation &t) {
        R << T(0,0), T(0,1), T(0,2),
                T(1,0), T(1,1), T(1,2),
                T(2,0), T(2,1), T(2,2);
        t << T(0,3), T(1,3), T(2,3);
    }


    void GeoUtil::Rt2T(const Rotation R, const Translation t, Transform &T) {
        T << R(0,0), R(0,1), R(0,2),t(0),
                R(1,0), R(1,1), R(1,2),t(1),
                R(2,0), R(2,1), R(2,2),t(2),
                0,0,0,1;
    }


    void GeoUtil::Rt2P(const Rotation &R, const Translation &t, Eigen::Matrix<Scalar, 3, 4>  &P) {
        P << R(0,0), R(0,1), R(0,2),t(0),
                R(1,0), R(1,1), R(1,2),t(1),
                R(2,0), R(2,1), R(2,2),t(2);
    }

    Transform GeoUtil::reverseTransformation(Transform src) {
        Rotation R;
        Translation t;
        Transform dst;
        T2Rt(src, R, t);
        Rt2T(R.transpose(), -R.transpose()*t, dst);
        return dst;
    }


    double GeoUtil::calculateDepth(const Eigen::Matrix<Scalar, 3, 4> &P, const Vector3 &point3D) {
        const double z = P.row(2).dot(point3D.homogeneous());
        return z * P.col(2).norm();
    }


    void GeoUtil::computeOW(Transform trans, Vector3& ow) {
        auto Rt = trans.block<3,3>(0,0).transpose();
        auto t = trans.block<3,1>(0, 3);
        ow = -Rt*t;
    }

    bool GeoUtil::validateTransform(Transform trans) {
        Rotation R;
        Translation t;
        T2Rt(trans, R, t);
        return trans(3,3)==1&&Sophus::isOrthogonal(R);
    }

}
