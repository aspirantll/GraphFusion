//
// Created by liulei on 2020/11/22.
//

#include "optimizer.h"

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/edge_se3.h"

typedef g2o::BlockSolver< g2o::BlockSolverTraits<g2o::VertexSE3::Dimension, -1>> SlamBlockSolver;
typedef g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/autodiff_cost_function.h>

struct CeresConstraint
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CeresConstraint(const Sophus::SE3d& T_ji, size_t i_, size_t j_):T_j_i(T_ji),i(i_),j(j_){}
    Sophus::SE3d T_j_i; //transformation from i to j
    size_t i, j; //keyframeId
};

struct CeresPose
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CeresPose(const Sophus::SE3d& pose)
    {
        t = pose.translation();
        q = Eigen::Quaterniond(pose.unit_quaternion());
    }
    Sophus::SE3d returnPose() const { return Sophus::SE3d(q,t); }
    Eigen::Vector3d t; ///< translation
    Eigen::Quaterniond q; ///< quaternion
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
        // Eigen::Quaternion<T> delta_q =  t_ab_measured_.unit_quaternion().template cast<T>() * q_ab_estimated.conjugate();

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

namespace rtf {

    void Optimizer::poseGraphOptimize(ViewGraph &viewGraph, const vector<pair<int, int> >& loops) {
        // pose graph optimization
        // How the problem is mapped to g2o:
        // The nodes get the global_T_frame transformation.
        // The edges get A as "from" (vertices()[0]),
        //               B as "to" (vertices()[1]), and
        //               A_tr_B as measurement.
        SlamLinearSolver* linearSolver = new SlamLinearSolver();
        SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(blockSolver);
        solver->setUserLambdaInit(1e-16);

        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm( solver );
        optimizer.setVerbose( false );

        //1.add frame as node
        for (int i = 0; i < viewGraph.getNodesNum(); i++) {
            g2o::VertexSE3* node = new g2o::VertexSE3();
            node->setId(i);
            node->setEstimate(Eigen::Isometry3d(GeoUtil::reverseTransformation(viewGraph[i].getGtTransform()).cast<double>()));
            optimizer.addVertex(node);
        }

        // Fix the first pose to account for gauge freedom.
        optimizer.vertex(0)->setFixed(true);

        // 2. add edges in connected components
        for(int i=0; i<viewGraph.getNodesNum()-1; i++) {
            g2o::EdgeSE3* edge = new g2o::EdgeSE3();
            edge->vertices()[0] = optimizer.vertex(i);
            edge->vertices()[1] = optimizer.vertex(i+1);
            Transform trans = viewGraph[i+1].getGtTransform().reverse()*viewGraph[i].getGtTransform();
            edge->setMeasurement(Eigen::Isometry3d(trans.cast<double>()));
            edge->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
            optimizer.addEdge(edge);
        }

        // loops
        for(int i=0; i<loops.size(); i++) {
            int refIndex = loops[i].first;
            int curIndex = loops[i].second;
            Edge connection = viewGraph.getEdge(refIndex, curIndex);
            if (!connection.isUnreachable()) {
                g2o::EdgeSE3* edge = new g2o::EdgeSE3();
                edge->vertices()[0] = optimizer.vertex(refIndex);
                edge->vertices()[1] = optimizer.vertex(curIndex);
                edge->setMeasurement(Eigen::Isometry3d(connection.getTransform().reverse().cast<double>()));
                edge->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
                optimizer.addEdge(edge);
            }
        }

        optimizer.save("/home/liulei/result_before.g2o");
        //3. pose graph optimization
        optimizer.initializeOptimization();
        constexpr int kMaxIterations = 20;
        optimizer.optimize(kMaxIterations);
        optimizer.save("/home/liulei/result_after.g2o");

        //4. update global transformation
        for(int i=0; i<viewGraph.getNodesNum(); i++) {
            Transform globalToFrame = reinterpret_cast<const g2o::VertexSE3*>(optimizer.vertex(i))->estimate().matrix().cast<float>();
            viewGraph[i].setGtTransform(GeoUtil::reverseTransformation(globalToFrame));
        }
    }

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
            Sophus::SE3f se3Pose(viewGraph[i].getGtTransform());
            ceresVectorPoses.emplace_back(se3Pose.cast<double>());
        }
        ceresVectorPoses[0].setPoseFixed();
        // 2. add edges in connected components
        for(int i=0; i<viewGraph.getNodesNum(); i++) {
            for(int j=i+1; j<viewGraph.getNodesNum(); j++) {
                Edge connection = viewGraph.getEdge(i, j);
                if (!connection.isUnreachable()) {
                    Transform trans = connection.getTransform();
                    const Eigen::Matrix<double, 6, 6> sqrt_information = Eigen::Matrix<double, 6, 6>::Identity();
                    ceres::CostFunction* cost_function = PoseGraph3dErrorTerm::Create(Sophus::SE3f(trans).cast<double>(), sqrt_information);

                    CeresPose& T_W_i = ceresVectorPoses[i];
                    CeresPose& T_W_j = ceresVectorPoses[j];

                    problem.AddResidualBlock(cost_function, loss_function,
                                             T_W_i.t.data(),T_W_i.q.coeffs().data(),
                                             T_W_j.t.data(),T_W_j.q.coeffs().data());
                    problem.SetParameterization(T_W_i.q.coeffs().data(), quaternion_local_parameterization);
                    problem.SetParameterization(T_W_j.q.coeffs().data(), quaternion_local_parameterization);
                }
            }
        }

        // loops
        for(int i=0; i<loops.size(); i++) {
            int refIndex = loops[i].first;
            int curIndex = loops[i].second;
            Edge connection = viewGraph.getEdge(refIndex, curIndex);
            if (!connection.isUnreachable()) {
                for(int j=0; j<10; j++) {
                    Transform trans = connection.getTransform();
                    const Eigen::Matrix<double, 6, 6> sqrt_information = Eigen::Matrix<double, 6, 6>::Identity();
                    ceres::CostFunction* cost_function = PoseGraph3dErrorTerm::Create(Sophus::SE3f(trans).cast<double>(), sqrt_information);

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
        std::cerr << summary.FullReport() << '\n';

        for(int i=0; i<viewGraph.getNodesNum(); i++) {
            viewGraph[i].setGtTransform(ceresVectorPoses[i].returnPose().matrix().cast<Scalar>());
        }
    }
}