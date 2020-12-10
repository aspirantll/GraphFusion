//
// Created by liulei on 2020/11/22.
//

#include "optimizer.h"
#include "../../processor/downsample.h"
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
using CeresPoseVector = std::vector<CeresPose,Eigen::aligned_allocator<CeresPose>>;

class PoseGraph3dErrorTerm {
public:

    PoseGraph3dErrorTerm(const CeresPose& t_ab_measured, const Eigen::Matrix<double,6,6>& sqrt_information)
            : t_ab_measured_(t_ab_measured), sqrt_information_(sqrt_information){}

    template <typename T>
    bool operator()(const T* const p_a_ptr, const T* const q_a_ptr,
                    const T* const p_b_ptr, const T* const q_b_ptr,
                    T* residuals_ptr) const
    {
        Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_a(p_a_ptr);
        Eigen::Map<const Eigen::Quaternion<T> > q_a(q_a_ptr);

        Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_b(p_b_ptr);
        Eigen::Map<const Eigen::Quaternion<T> > q_b(q_b_ptr);

        // Compute the relative transformation between the two frames.
        Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
        Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;

        // Represent the displacement between the two frames in the A frame.
        Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (p_b - p_a);

        // Compute the error between the two orientation estimates.
        Eigen::Quaternion<T> delta_q = t_ab_measured_.q.template cast<T>() * q_ab_estimated.conjugate();

        // Compute the residuals.
        // [ position         ]   [ delta_p          ]
        // [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
        Eigen::Map<Eigen::Matrix<T, 6, 1> > residuals(residuals_ptr);
        residuals.template block<3, 1>(0, 0) = p_ab_estimated - t_ab_measured_.t.template cast<T>();
        residuals.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();

        // Scale the residuals by the measurement uncertainty.
        residuals.applyOnTheLeft(sqrt_information_.cast<T>());

        return true;
    }
    static ceres::CostFunction* Create(const CeresPose& t_ab_measured, const Eigen::Matrix<double,6,6>& sqrt_information)
    {
        return new ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm, 6, 3, 4, 3, 4>(new PoseGraph3dErrorTerm(t_ab_measured, sqrt_information));
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    // The measurement for the position of B relative to A in the A frame.
    const CeresPose t_ab_measured_;
    // The square root of the measurement information matrix.
    const Eigen::Matrix<double,6,6> sqrt_information_;
};

class NormErrorTerm {
public:

    NormErrorTerm(const double weight, const Eigen::Matrix<double,6,6>& sqrt_information): weight(weight), sqrt_information_(sqrt_information){}

    template <typename T>
    bool operator()(const T* const p_ptr, const T* const q_ptr,
                    T* residuals_ptr) const
    {
        Eigen::Map<const Eigen::Matrix<T, 3, 1> > p(p_ptr);
        Eigen::Map<const Eigen::Matrix<T, 3, 1> > q(q_ptr);

        Eigen::Map<Eigen::Matrix<T, 6, 1> > residuals(residuals_ptr);
        residuals.template block<3, 1>(0, 0) = weight * p;
        residuals.template block<3, 1>(3, 0) = weight * 2 * q;

        // Scale the residuals by the measurement uncertainty.
        residuals.applyOnTheLeft(sqrt_information_.cast<T>());

        return true;
    }
    static ceres::CostFunction* Create(const double weight, const Eigen::Matrix<double,6,6>& sqrt_information)
    {
        return new ceres::AutoDiffCostFunction<NormErrorTerm, 6, 3, 4>(new NormErrorTerm(weight, sqrt_information));
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    double weight;
    // The square root of the measurement information matrix.
    const Eigen::Matrix<double,6,6> sqrt_information_;
};


constexpr Scalar kHuberWeight = 1.2;
inline __device__ Scalar computeHuberWeight(Scalar residual_x, Scalar residual_y, Scalar huber_parameter) {
    Scalar squared_residual = residual_x * residual_x + residual_y * residual_y;
    return (squared_residual < huber_parameter * huber_parameter) ? 1 : (huber_parameter / sqrtf(squared_residual));
}

// global bundle adjustment
struct BAObservation {
    rtf::Point3D point;
    rtf::Point2D pixel;
    int order; //0-normal 1-inverse
    Scalar fx, fy, cx, cy;

    double computeErr(SE3 se) {
        Vector3 p = se.rotationMatrix()*point.toVector3()+se.translation();
        // project into image pixel
        const double predicted_x = (fx*p[0]+cx*p[2])/p[2];
        const double predicted_y = (fy*p[1]+cy*p[2])/p[2];

        // The error is the difference between the predicted and observed position.
        return computeHuberWeight(predicted_x - pixel.x, predicted_y - pixel.y, kHuberWeight);
    }
};

class ReprojectionErrorWithQuaternions {
public:
    // (u, v): the position of the observation with respect to the image
    // center point.
    ReprojectionErrorWithQuaternions(const BAObservation& observation): observation(observation) {}

    template <typename T>
    bool operator()(const T* const p_i_ptr, const T* const q_i_ptr,
                    const T* const p_j_ptr, const T* const q_j_ptr,
                    const T* const p_k_ptr, const T* const q_k_ptr,
                    T* residuals) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1> > pi(p_i_ptr);
        Eigen::Map<const Eigen::Quaternion<T> > qi(q_i_ptr);

        Eigen::Map<const Eigen::Matrix<T, 3, 1> > pj(p_j_ptr);
        Eigen::Map<const Eigen::Quaternion<T> > qj(q_j_ptr);

        Eigen::Map<const Eigen::Matrix<T, 3, 1> > pk(p_k_ptr);
        Eigen::Map<const Eigen::Quaternion<T> > qk(q_k_ptr);

        Sophus::SE3<T> sei(qi, pi), sej(qj, pj), sek(qk, pk);

        Sophus::SE3<T> trans = observation.order==0? sei.inverse()*sej*sek: sek.inverse()*sej.inverse()*sei;

        Eigen::Matrix<T, 3, 1> p = trans.rotationMatrix()*observation.point.toVector3()+trans.translation();
        // project into image pixel
        const T predicted_x = (observation.fx*p[0]+observation.cx*p[2])/p[2];
        const T predicted_y = (observation.fy*p[1]+observation.cy*p[2])/p[2];

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - observation.pixel.x;
        residuals[1] = predicted_y - observation.pixel.y;

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(BAObservation observation) {
        return (new ceres::AutoDiffCostFunction<
                ReprojectionErrorWithQuaternions, 2, 3, 4, 3, 4, 3, 4>(
                new ReprojectionErrorWithQuaternions(observation)));
    }

    BAObservation observation;
};

namespace rtf {

    void Optimizer::poseGraphOptimizeCeres(ViewGraph &viewGraph, const vector<pair<int, int> >& loops) {
        //Loss Function can be changed!
        // ceres::LossFunction* loss_function = new ceres::CauchyLoss(1.0);
        // ceres::LossFunction* loss_function = new ceres::HuberLoss(0.2);
        ceres::Problem problem;
        ceres::LossFunction* loss_function = nullptr;
        ceres::LocalParameterization* quaternion_local_parameterization = new ceres::EigenQuaternionParameterization;
        // collect cere poses
        CeresPoseVector ceresVectorPoses;
        for (int i = 0; i < viewGraph.getNodesNum(); i++) {
            ceresVectorPoses.emplace_back(viewGraph[i].getGtSE());
        }
        ceresVectorPoses[0].setPoseFixed();
        // 2. add edges in connected components
        for(int i=0; i<viewGraph.getNodesNum()-1; i++) {
            const Eigen::Matrix<double, 6, 6> sqrt_information = Eigen::Matrix<double, 6, 6>::Identity();
            ceres::CostFunction* cost_function = PoseGraph3dErrorTerm::Create((viewGraph[i].getGtSE().inverse()*viewGraph[i+1].getGtSE()), sqrt_information);

            CeresPose& T_W_i = ceresVectorPoses[i];
            CeresPose& T_W_j = ceresVectorPoses[i+1];

            problem.AddResidualBlock(cost_function, loss_function,
                                      T_W_i.t.data(),T_W_i.q.coeffs().data(),
                                      T_W_j.t.data(),T_W_j.q.coeffs().data());
            problem.SetParameterization(T_W_i.q.coeffs().data(), quaternion_local_parameterization);
            problem.SetParameterization(T_W_j.q.coeffs().data(), quaternion_local_parameterization);
        }

        // loops
        for(int i=0; i<loops.size(); i++) {
            int refIndex = loops[i].first;
            int curIndex = loops[i].second;
            Edge connection = viewGraph.getEdge(refIndex, curIndex);
            if (!connection.isUnreachable()) {
                for(int j=0; j<3; j++) {
                    const Eigen::Matrix<double, 6, 6> sqrt_information = Eigen::Matrix<double, 6, 6>::Identity();
                    ceres::CostFunction* cost_function = PoseGraph3dErrorTerm::Create(connection.getSE(), sqrt_information);

                    CeresPose& T_W_i = ceresVectorPoses[refIndex];
                    CeresPose& T_W_j = ceresVectorPoses[curIndex];

                    problem.AddResidualBlock(cost_function, loss_function,
                                             T_W_i.t.data(),T_W_i.q.coeffs().data(),
                                             T_W_j.t.data(),T_W_j.q.coeffs().data());
                    problem.SetParameterization(T_W_i.q.coeffs().data(), quaternion_local_parameterization);
                    problem.SetParameterization(T_W_j.q.coeffs().data(), quaternion_local_parameterization);
                }
            }
        }

        ceres::Solver::Options options;
        options.max_num_iterations = 200;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.num_threads = 1;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << '\n';

        for(int i=0; i<viewGraph.getNodesNum(); i++) {
            viewGraph[i].setGtTransform(ceresVectorPoses[i].returnPose().matrix());
        }
    }

    void Optimizer::globalBundleAdjustmentCeres(ViewGraph &viewGraph, const vector<int>& cc) {
        int m = cc.size();
        if(m<=1) return;

        CeresPoseVector ceresPoseVector;
        for(int c: cc) {
            ceresPoseVector.emplace_back(viewGraph[c].getGtSE());
        }

        for(int i=0; i<m; i++) {
            for(int j=i+1; j<m; j++) {
                if(viewGraph.existEdge(cc[i], cc[j])) {
                    ceresPoseVector.emplace_back(ceresPoseVector[j].returnPose().inverse() * ceresPoseVector[i].returnPose() * viewGraph.getEdgeSE(cc[i], cc[j]));
                }
            }
        }
        ceresPoseVector[0].setPoseFixed();

        ceres::Problem problem;
        ceres::LossFunction* loss_function = new ceres::HuberLoss(kHuberWeight);
        ceres::LocalParameterization* quaternion_local_parameterization = new ceres::EigenQuaternionParameterization;

        shared_ptr<Camera> camera = viewGraph[0].getCamera();
        shared_ptr<CameraModel> cameraModel = camera->getCameraModel();

        BAObservation baseObservation;
        baseObservation.fx = camera->getFx();
        baseObservation.fy = camera->getFy();
        baseObservation.cx = camera->getCx();
        baseObservation.cy = camera->getCy();

        int k = m;
        int kpNum = 0;
        double err = 0;
        for(int i=0; i<m; i++) {
            CeresPose& Ti = ceresPoseVector[i];
            for(int j=i+1; j<m; j++) {
                CeresPose& Tj = ceresPoseVector[j];

                Edge edge = viewGraph.getEdge(cc[i], cc[j]);
                if(!edge.isUnreachable()) {
                    CeresPose& Tk = ceresPoseVector[k];

                    SE3 se = edge.getSE();
                    vector<FeatureKeypoint> kxs = edge.getKxs(), kys = edge.getKys();
                    downSampleFeatureMatches(kxs, kys, camera, edge.getTransform(), 100, 100);
                    int pairNum = kxs.size();
                    kpNum += pairNum;
                    for(int l=0; l<pairNum; l++) {
                        FeatureKeypoint px = kxs[l];
                        FeatureKeypoint py = kys[l];

                        BAObservation observation1 = baseObservation;
                        observation1.pixel = px;
                        observation1.point = cameraModel->unproject(py.x, py.y, py.z);
                        observation1.order = 0;

                        ceres::CostFunction* cost_function1 = ReprojectionErrorWithQuaternions::Create(observation1);

                        problem.AddResidualBlock(cost_function1, loss_function,
                                                 Ti.t.data(),Ti.q.coeffs().data(),
                                                 Tj.t.data(),Tj.q.coeffs().data(),
                                                 Tk.t.data(),Tk.q.coeffs().data());

                        err += observation1.computeErr(se);
                        BAObservation observation2 = baseObservation;
                        observation1.pixel = py;
                        observation1.point = cameraModel->unproject(px.x, px.y, px.z);
                        observation1.order = 1;

                        ceres::CostFunction* cost_function2 = ReprojectionErrorWithQuaternions::Create(observation2);

                        problem.AddResidualBlock(cost_function2, loss_function,
                                                 Ti.t.data(),Ti.q.coeffs().data(),
                                                 Tj.t.data(),Tj.q.coeffs().data(),
                                                 Tk.t.data(),Tk.q.coeffs().data());

                        err += observation2.computeErr(se.inverse());

                        problem.SetParameterization(Ti.q.coeffs().data(), quaternion_local_parameterization);
                        problem.SetParameterization(Tj.q.coeffs().data(), quaternion_local_parameterization);
                        problem.SetParameterization(Tk.q.coeffs().data(), quaternion_local_parameterization);
                    }

                    const Eigen::Matrix<double, 6, 6> sqrt_information = Eigen::Matrix<double, 6, 6>::Identity();
                    ceres::CostFunction* cost_function = NormErrorTerm::Create(10, sqrt_information);
                    problem.AddResidualBlock(cost_function, nullptr,
                                             Tk.t.data(),Tk.q.coeffs().data());
                    k++;
                }
            }
        }
        cout << "err:" << err <<endl;

        ceres::Solver::Options options;
        options.max_num_iterations = 5;
        options.num_threads = 1;
        options.eta = 1e-2;
        options.max_solver_time_in_seconds = 1e32;
        options.use_nonmonotonic_steps = false;

        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.dogleg_type = ceres::TRADITIONAL_DOGLEG;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.use_inner_iterations = false;
        options.preconditioner_type = ceres::JACOBI;
        options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
        options.dense_linear_algebra_library_type = ceres::EIGEN;
        options.use_explicit_schur_complement = false;
        options.use_mixed_precision_solves = false;
        options.max_num_refinement_iterations = 0;

        options.gradient_tolerance = 1e-16;
        options.function_tolerance = 1e-5;
        ceres::Solver::Summary summary;
        Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n";

        // update transformation
        for(int i=0; i<m; i++) {
            viewGraph[cc[i]].setGtTransform(ceresPoseVector[i].returnPose().matrix());
        }
    }
}