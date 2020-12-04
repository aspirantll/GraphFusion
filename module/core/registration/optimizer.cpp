//
// Created by liulei on 2020/11/22.
//

#include "optimizer.h"
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/autodiff_cost_function.h>

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


// global bundle adjustment
// Return a random number sampled from a uniform distribution in the range
// [0,1].
inline double RandDouble() {
    double r = static_cast<double>(rand());
    return r / RAND_MAX;
}

// Marsaglia Polar method for generation standard normal (pseudo)
// random numbers http://en.wikipedia.org/wiki/Marsaglia_polar_method
inline double RandNormal() {
    double x1, x2, w;
    do {
        x1 = 2.0 * RandDouble() - 1.0;
        x2 = 2.0 * RandDouble() - 1.0;
        w = x1 * x1 + x2 * x2;
    } while ( w >= 1.0 || w == 0.0 );

    w = sqrt((-2.0 * log(w)) / w);
    return x1 * w;
}

struct PointPair {
    int index;
    rtf::Point3D point;
    rtf::Point2D pixel;
};


void collectCorrespondences(vector<vector<pair<int, PointPair>>>& correlations, vector<bool>& visited, int u, vector<int>& corrIndexes, vector<PointPair>& corr) {
    for(int i=0; i<correlations[u].size(); i++) {
        int v = correlations[u][i].first;
        if(!visited[v]) {
            visited[v] = true;
            corrIndexes.emplace_back(v);
            corr.emplace_back(correlations[u][i].second);
            collectCorrespondences(correlations, visited, v, corrIndexes, corr);
        }
    }
}

namespace rtf {
    typedef Eigen::Map<Eigen::VectorXd> VectorRef;
    typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;

    void PerturbPoint3(const double sigma, double* point) {
        for (int i = 0; i < 3; ++i) {
            point[i] += RandNormal() * sigma;
        }
    }

    double Median(std::vector<double>* data) {
        int n = data->size();
        std::vector<double>::iterator mid_point = data->begin() + n / 2;
        std::nth_element(data->begin(), mid_point, data->end());
        return *mid_point;
    }

    BALProblem::BALProblem(ViewGraph &viewGraph, const vector<int>& cc) {
        // collect points
        int m = cc.size();
        int kpNum = 0;
        vector<int> startIndexes(m);
        for(int i=0; i<m; i++) {
            shared_ptr<KeyFrame> kf = viewGraph[cc[i]].getFrames()[0];
            startIndexes[i] = kpNum;
            kpNum += kf->getKps().getKeyPoints().size();
        }

        // foreach edge
        vector<vector<pair<int, PointPair>>> correlations(kpNum);
        shared_ptr<Camera> camera = viewGraph[0].getCamera();
        for(int i=0; i<m; i++) {
            for (int j = i + 1; j < m; j++) {
                Edge edge = viewGraph.getEdge(cc[i], cc[j]);
                if(!edge.isUnreachable()) {
                    for(int k=0; k<edge.getKxs().size(); k++) {
                        FeatureKeypoint px = edge.getKxs()[k];
                        FeatureKeypoint py = edge.getKys()[k];

                        Vector3 qx = PointUtil::transformPoint(camera->getCameraModel()->unproject(px.x, px.y, px.z), viewGraph[cc[i]].getGtTransform());
                        Vector3 qy = PointUtil::transformPoint(camera->getCameraModel()->unproject(py.x, py.y, py.z), viewGraph[cc[j]].getGtTransform());
                        if((qx-qy).norm()<0.03) {
                            int ix = startIndexes[i] + px.getIndex();
                            int iy = startIndexes[j] + py.getIndex();

                            PointPair p1;
                            p1.index = j;
                            p1.pixel = py;
                            p1.point = qy;

                            correlations[ix].emplace_back(make_pair(iy, p1));

                            PointPair p2;
                            p2.index = i;
                            p2.pixel = px;
                            p2.point = qx;
                            correlations[iy].emplace_back(make_pair(ix, p2));
                        }
                    }
                }
            }
        }

        // foreach edge
        vector<bool> visited(correlations.size(), false);
        EigenVector(Vector3) points;
        vector<vector<PointPair>>  obs;
        int obsCount = 0;
        for(int i=0; i<m; i++) {
            int nodeIndex = cc[i];
            SIFTFeaturePoints &sift = viewGraph[nodeIndex].getFrames()[0]->getKps();
            for(int j=0; j<sift.getKeyPoints().size(); j++) {
                int curIndex = startIndexes[nodeIndex]+j;
                if(!visited[curIndex]) {
                    vector<PointPair> corr;
                    vector<int> corrIndexes;

                    collectCorrespondences(correlations, visited, curIndex, corrIndexes, corr);
                    if(!corr.empty()) {
                        Vector3 pos = Vector3::Zero();
                        for(const PointPair& c: corr) {
                            pos += c.point.toVector3();
                        }
                        pos /= corr.size();

                        points.emplace_back(pos);
                        obs.emplace_back(corr);
                        obsCount += corr.size();
                    }
                }
            }
        }

        // add parameters
        num_cameras_ = m;
        num_points_ = points.size();
        num_observations_ = obsCount;
        cout << "camera num:" << num_cameras_ << ", point num:" << num_points_ << ", observation num:" << num_observations_ << endl;

        point_index_ = new int[num_observations_];
        camera_index_ = new int[num_observations_];
        observations_ = new double[2 * num_observations_];

        num_parameters_ = 7 * num_cameras_ + 3 * num_points_;
        parameters_ = new double[num_parameters_];

        // add obs
        int obsIndex = 0;
        for(int i=0; i<num_points_; i++) {
            for(PointPair& p: obs[i]) {
                camera_index_[obsIndex] = p.index;
                point_index_[obsIndex] = i;
                observations_[2*obsIndex] = p.pixel.x;
                observations_[2*obsIndex+1] = p.pixel.y;
                obsIndex++;
            }
        }

        // add parameters
        for(int i=0; i<num_cameras_; i++) {
            SE3 se = viewGraph[cc[i]].getGtSE().inverse();
            parameters_[7*i + 0] = 1.0;
            parameters_[7*i + 1] = se.unit_quaternion().x();
            parameters_[7*i + 2] = se.unit_quaternion().y();
            parameters_[7*i + 3] = se.unit_quaternion().z();
            parameters_[7*i + 4] = se.translation().x();
            parameters_[7*i + 5] = se.translation().y();
            parameters_[7*i + 6] = se.translation().z();
            parameters_[7*i + 7] = viewGraph[cc[i]].getCamera()->getFx();
            parameters_[7*i + 8] = 0;
            parameters_[7*i + 9] = 0;
        }
        
        for(int i=0; i<num_points_; i++) {
            parameters_[7*num_cameras_+3*i+0] = points[i].x();
            parameters_[7*num_cameras_+3*i+1] = points[i].y();
            parameters_[7*num_cameras_+3*i+2] = points[i].z();
        }
    }

    void BALProblem::CameraToAngleAxisAndCenter(const double* camera,
                                                double* angle_axis,
                                                double* center) const {
        VectorRef angle_axis_ref(angle_axis, 3);
        ceres::QuaternionToAngleAxis(camera, angle_axis);

        // c = -R't
        Eigen::VectorXd inverse_rotation = -angle_axis_ref;
        ceres::AngleAxisRotatePoint(inverse_rotation.data(),
                             camera + 4,
                             center);
        VectorRef(center, 3) *= -1.0;
    }

    void BALProblem::AngleAxisAndCenterToCamera(const double* angle_axis,
                                                const double* center,
                                                double* camera) const {
        ConstVectorRef angle_axis_ref(angle_axis, 3);
        ceres::AngleAxisToQuaternion(angle_axis, camera);

        // t = -R * c
        ceres::AngleAxisRotatePoint(angle_axis,
                             center,
                             camera + 4);
        VectorRef(camera + 4, 3) *= -1.0;
    }


    void BALProblem::Normalize() {
        // Compute the marginal median of the geometry.
        std::vector<double> tmp(num_points_);
        double* points = mutable_points();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < num_points_; ++j) {
                tmp[j] = points[3 * j + i];
            }
            median(i) = Median(&tmp);
        }

        for (int i = 0; i < num_points_; ++i) {
            VectorRef point(points + 3 * i, 3);
            tmp[i] = (point - median).lpNorm<1>();
        }

        const double median_absolute_deviation = Median(&tmp);

        // Scale so that the median absolute deviation of the resulting
        // reconstruction is 100.
        scale = 100.0 / median_absolute_deviation;

        VLOG(2) << "median: " << median.transpose();
        VLOG(2) << "median absolute deviation: " << median_absolute_deviation;
        VLOG(2) << "scale: " << scale;

        // X = scale * (X - median)
        for (int i = 0; i < num_points_; ++i) {
            VectorRef point(points + 3 * i, 3);
            point = scale * (point - median);
        }

        double* cameras = mutable_cameras();
        double angle_axis[3];
        double center[3];
        for (int i = 0; i < num_cameras_; ++i) {
            double* camera = cameras + camera_block_size() * i;
            CameraToAngleAxisAndCenter(camera, angle_axis, center);
            // center = scale * (center - median)
            VectorRef(center, 3) = scale * (VectorRef(center, 3) - median);
            AngleAxisAndCenterToCamera(angle_axis, center, camera);
        }
    }

    void BALProblem::Denormalize() {
        double* points = mutable_points();
        // X = X/scale + median
        for (int i = 0; i < num_points_; ++i) {
            VectorRef point(points + 3 * i, 3);
            point = point/scale + median;
        }

        double* cameras = mutable_cameras();
        double angle_axis[3];
        double center[3];
        for (int i = 0; i < num_cameras_; ++i) {
            double* camera = cameras + camera_block_size() * i;
            CameraToAngleAxisAndCenter(camera, angle_axis, center);
            // center = center/scale + median
            VectorRef(center, 3) = VectorRef(center, 3)/scale + median;
            AngleAxisAndCenterToCamera(angle_axis, center, camera);
        }
    }

    void BALProblem::Perturb(const double rotation_sigma,
                             const double translation_sigma,
                             const double point_sigma) {
        CHECK_GE(point_sigma, 0.0);
        CHECK_GE(rotation_sigma, 0.0);
        CHECK_GE(translation_sigma, 0.0);

        double* points = mutable_points();
        if (point_sigma > 0) {
            for (int i = 0; i < num_points_; ++i) {
                PerturbPoint3(point_sigma, points + 3 * i);
            }
        }

        for (int i = 0; i < num_cameras_; ++i) {
            double* camera = mutable_cameras() + camera_block_size() * i;

            double angle_axis[3];
            double center[3];
            // Perturb in the rotation of the camera in the angle-axis
            // representation.
            CameraToAngleAxisAndCenter(camera, angle_axis, center);
            if (rotation_sigma > 0.0) {
                PerturbPoint3(rotation_sigma, angle_axis);
            }
            AngleAxisAndCenterToCamera(angle_axis, center, camera);

            if (translation_sigma > 0.0) {
                PerturbPoint3(translation_sigma, camera + 4);
            }
        }
    }

    BALProblem::~BALProblem() {
        delete []point_index_;
        delete []camera_index_;
        delete []observations_;
        delete []parameters_;
    }

    // Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 10 parameters. 4 for rotation, 3 for
// translation, 1 for focal length and 2 for radial distortion. The
// principal point is not modeled (i.e. it is assumed be located at
// the image center).
    struct SnavelyReprojectionErrorWithQuaternions {
        // (u, v): the position of the observation with respect to the image
        // center point.
        SnavelyReprojectionErrorWithQuaternions(double observed_x, double observed_y, shared_ptr<Camera> camera)
                : observed_x(observed_x), observed_y(observed_y), camera(move(camera)) {}

        template <typename T>
        bool operator()(const T* const trans,
                        const T* const point,
                        T* residuals) const {
            // trans[0,1,2,3] is are the rotation of the camera as a quaternion.
            //
            // We use QuaternionRotatePoint as it does not assume that the
            // quaternion is normalized, since one of the ways to run the
            // bundle adjuster is to let Ceres optimize all 4 quaternion
            // parameters without a local parameterization.
            T p[3];
            ceres::QuaternionRotatePoint(trans, point, p);

            p[0] += trans[4];
            p[1] += trans[5];
            p[2] += trans[6];

            // project into image pixel
            const T predicted_x = (camera->getFx()*p[0]+camera->getCx()*p[2])/p[2];
            const T predicted_y = (camera->getFy()*p[1]+camera->getCy()*p[2])/p[2];

            // The error is the difference between the predicted and observed position.
            residuals[0] = predicted_x - observed_x;
            residuals[1] = predicted_y - observed_y;

            return true;
        }

        // Factory to hide the construction of the CostFunction object from
        // the client code.
        static ceres::CostFunction* Create(const double observed_x,
                                           const double observed_y, shared_ptr<Camera> camera) {
            return (new ceres::AutoDiffCostFunction<
                    SnavelyReprojectionErrorWithQuaternions, 2, 7, 3>(
                    new SnavelyReprojectionErrorWithQuaternions(observed_x,
                                                                observed_y, camera)));
        }

        double observed_x;
        double observed_y;
        shared_ptr<Camera> camera;
    };

    void SetLinearSolver(ceres::Solver::Options* options) {
        CHECK(StringToLinearSolverType("sparse_schur",
                                       &options->linear_solver_type));
        CHECK(StringToPreconditionerType("jacobi",
                                         &options->preconditioner_type));
        CHECK(StringToVisibilityClusteringType("canonical_views",
                                               &options->visibility_clustering_type));
        CHECK(StringToSparseLinearAlgebraLibraryType(
                "suite_sparse",
                &options->sparse_linear_algebra_library_type));
        CHECK(StringToDenseLinearAlgebraLibraryType(
                "eigen",
                &options->dense_linear_algebra_library_type));
        options->use_explicit_schur_complement = false;
        options->use_mixed_precision_solves = false;
        options->max_num_refinement_iterations = 0;
    }

    void SetMinimizerOptions(ceres::Solver::Options* options) {
        options->max_num_iterations = 5;
        options->minimizer_progress_to_stdout = true;
        options->num_threads = 1;
        options->eta = 1e-2;
        options->max_solver_time_in_seconds = 1e32;
        options->use_nonmonotonic_steps = false;

        CHECK(StringToTrustRegionStrategyType("levenberg_marquardt",
                                              &options->trust_region_strategy_type));
        CHECK(StringToDoglegType("traditional_dogleg", &options->dogleg_type));
        options->use_inner_iterations = false;
    }

    void SetSolverOptionsFromFlags(BALProblem* bal_problem,
                                   ceres::Solver::Options* options) {
        SetMinimizerOptions(options);
        SetLinearSolver(options);
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
            ceresVectorPoses.emplace_back(viewGraph[i].getGtSE().cast<double>());
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
        BALProblem bal_problem(viewGraph, cc);

        shared_ptr<Camera> camera = viewGraph[0].getCamera();
        ceres::Problem problem;
        srand(0);
        bal_problem.Normalize();
        bal_problem.Perturb(0.0, 0.0,0.0);

        const int point_block_size = bal_problem.point_block_size();
        const int camera_block_size = bal_problem.camera_block_size();
        double* points = bal_problem.mutable_points();
        double* cameras = bal_problem.mutable_cameras();

        // Observations is 2*num_observations long array observations =
        // [u_1, u_2, ... , u_n], where each u_i is two dimensional, the x
        // and y positions of the observation.
        const double* observations = bal_problem.observations();
        for (int i = 0; i < bal_problem.num_observations(); ++i) {
            ceres::CostFunction* cost_function;
            // Each Residual block takes a point and a camera as input and
            // outputs a 2 dimensional residual.
            cost_function = SnavelyReprojectionErrorWithQuaternions::Create(
                    observations[2 * i + 0],
                    observations[2 * i + 1], camera);
            
            // If enabled use Huber's loss function.
            ceres::LossFunction* loss_function = NULL;

            // Each observation correponds to a pair of a camera and a point
            // which are identified by camera_index()[i] and point_index()[i]
            // respectively.
            double* camera =
                    cameras + camera_block_size * bal_problem.camera_index()[i];
            double* point = points + point_block_size * bal_problem.point_index()[i];
            problem.AddResidualBlock(cost_function, loss_function, camera, point);
        }

        ceres::LocalParameterization* camera_parameterization =
                new ceres::ProductParameterization(
                        new ceres::QuaternionParameterization(),
                        new ceres::IdentityParameterization(3));
        for (int i = 0; i < bal_problem.num_cameras(); ++i) {
            problem.SetParameterization(cameras + camera_block_size * i,
                                        camera_parameterization);
        }

        ceres::Solver::Options options;
        SetSolverOptionsFromFlags(&bal_problem, &options);
        options.gradient_tolerance = 1e-16;
        options.function_tolerance = 1e-16;
        ceres::Solver::Summary summary;
        Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n";

        // update transformation
        bal_problem.Denormalize();
        for(int i=0; i<cc.size(); i++) {
            const double* camera = bal_problem.cameras()+i*bal_problem.camera_block_size();
            Eigen::Quaterniond q(camera[0], camera[1], camera[2], camera[3]);
            Vector3 t;
            t << camera[4], camera[5], camera[6];
            viewGraph[cc[i]].setGtSE(SE3(q, t).inverse());
        }
    }
}