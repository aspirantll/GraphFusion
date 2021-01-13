//
// Created by liulei on 2020/11/12.
//
#include "registrations.h"
#include "../.././thirdparty/ICPCUDA/ICPOdometry.h"

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/autodiff_cost_function.h>

struct CeresPose
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CeresPose(const SE3& pose)
    {
        t = pose.translation();
        q = Eigen::Quaternion<Scalar>(pose.unit_quaternion());
    }
    SE3 returnPose() const { return SE3(q,t); }

    Vector3 t; ///< translation
    Eigen::Quaternion<Scalar> q; ///< quaternion
    bool flagFixPose = false;
    void setPoseFixed() { flagFixPose = true; }
};

class PointPairErrorItem {
public:

    PointPairErrorItem(const Eigen::Vector3d& px, const Eigen::Vector3d& py): px(px), py(py){}

    template <typename T>
    bool operator()(const T* const p_ptr, const T* const q_ptr,
                    T* residuals_ptr) const
    {
        Eigen::Map<const Eigen::Matrix<T, 3, 1> > p(p_ptr);
        Eigen::Map<const Eigen::Quaternion<T> > q(q_ptr);

        Eigen::Matrix<T, 3, 1> qy = q*py.cast<T>();

        Eigen::Map<Eigen::Matrix<T, 3, 1> > residuals(residuals_ptr);
        residuals.template block<3, 1>(0, 0) = qy-px.cast<T>();

        return true;
    }
    static ceres::CostFunction* Create(const Eigen::Vector3d& px, const Eigen::Vector3d& py)
    {
        return new ceres::AutoDiffCostFunction<PointPairErrorItem, 3, 3, 4>(new PointPairErrorItem(px, py));
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    Eigen::Vector3d px;
    Eigen::Vector3d py;
};

namespace rtf {
    PairwiseICP::PairwiseICP(const GlobalConfig& config) {
        rmsThreshold = config.rmsThreshold;
        relaxtion = config.relaxtion;
        distTh = 0.01;
        minInliers = config.kMinInliers;
    }

    RegReport PairwiseICP::icp(Transform trans, shared_ptr<Frame> fx, shared_ptr<Frame> fy) {
        shared_ptr<Camera> camera = fx->getCamera();
        ICPOdometry icpOdom(camera->getWidth(), camera->getHeight(), camera->getCx(), camera->getCy(), camera->getFx(), camera->getFy());
        icpOdom.initICPModel(fx->getDepthImage()->ptr<float>());
        icpOdom.initICP(fy->getDepthImage()->ptr<float>());

        icpOdom.getIncrementalTransformation(trans, 96, 96);

        RegReport report;
        report.success = icpOdom.lastInliers>minInliers;
        report.cost = icpOdom.lastError;
        report.inlierNum = icpOdom.lastInliers;
        report.T = trans;
        report.iterations = 1;
        return report;
    }

    RegReport PairwiseICP::icp(Transform initT, shared_ptr<Camera> cx, shared_ptr<Camera> cy,
                  vector<FeatureKeypoint> &kxs, vector<FeatureKeypoint> &kys) {
        int n = kxs.size();
        ceres::Problem problem;
        ceres::LossFunction *loss_function = nullptr;
        ceres::LocalParameterization *quaternion_local_parameterization = new ceres::EigenQuaternionParameterization;

        CeresPose pose((SE3(initT)));
        for (int i = 0; i < n; i++) {
            Vector3 px = cx->getCameraModel()->unproject(kxs[i].x, kxs[i].y, kxs[i].z);
            Vector3 py = cy->getCameraModel()->unproject(kys[i].x, kys[i].y, kys[i].z);

            ceres::CostFunction *cost_function = PointPairErrorItem::Create(px, py);

            problem.AddResidualBlock(cost_function, loss_function,
                                     pose.t.data(), pose.q.coeffs().data());
            problem.SetParameterization(pose.q.coeffs().data(), quaternion_local_parameterization);
        }

        ceres::Solver::Options options;
        options.max_num_iterations = 200;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.num_threads = 1;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << '\n';
        RegReport report;
        report.success = summary.IsSolutionUsable();
        report.cost = summary.final_cost;
        report.T = pose.returnPose().matrix();
        report.iterations = summary.iterations.size();

        if(report.success) {
            vector<FeatureKeypoint> bKxs(kxs.begin(), kxs.end()), bKys(kys.begin(), kys.end());
            kxs.clear();
            kys.clear();
            for (int i = 0; i < n; i++) {
                Vector3 px = cx->getCameraModel()->unproject(bKxs[i].x, bKxs[i].y, bKxs[i].z);
                Vector3 py = cy->getCameraModel()->unproject(bKys[i].x, bKys[i].y, bKys[i].z);

                Vector3 qy = PointUtil::transformPoint(py, report.T);
                if((px-qy).norm()<0.01) {
                    kxs.emplace_back(bKxs[i]);
                    kys.emplace_back(bKys[i]);
                }
            }
        }
        return report;
    }
}
