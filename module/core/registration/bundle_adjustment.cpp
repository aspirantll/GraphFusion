//
// Created by liulei on 2020/6/12.
//

#include <utility>
#include "registrations.h"
#include "bundle_adjustment.cuh"
#include "../../processor/downsample.h"

namespace rtf {

    void computeP(Vector6 r, Vector6 M, double lambda, Vector6 &p) {
        for (int i = 0; i < 6; i++) {
            p(i) = r(i) / (M(i) + lambda);
        }
    }

    BARegistration::BARegistration(const GlobalConfig &config) {
        rmsThreshold = config.rmsThreshold;
        relaxtion = config.relaxtion;
        distTh = config.maxPnPResidual;
        minInliers = config.kMinInliers;

    }


    RegReport BARegistration::bundleAdjustment(Transform initT, shared_ptr<Camera> cx, shared_ptr<Camera> cy,
                                               vector<FeatureKeypoint> &kxs, vector<FeatureKeypoint> &kys,
                                               bool robust) {
        alloc(std::move(cx), std::move(cy), kxs, kys);
        RegReport report = bundleAdjustment(initT, robust);
        free();
        return report;
    }

    void BARegistration::alloc(shared_ptr<Camera> cx, shared_ptr<Camera> cy, vector<FeatureKeypoint> &kxs,
                               vector<FeatureKeypoint> &kys) {
        this->kxs = &kxs;
        this->kys = &kys;
        this->camera = cx;

        // compose parameters for ba
        const int n = kxs.size();
        points.resize(n, 3);
        pixels.resize(n, 2);

        for (int k = 0; k < kxs.size(); k++) {
            const FeatureKeypoint &px = kxs[k];
            const FeatureKeypoint &py = kys[k];

            pixels.row(k) << px.x, px.y;
            points.row(k) = cy->getCameraModel()->unproject(py.x, py.y, py.z);
        }


        vector<unsigned char> mask(n, 1);
        cudaMask = new CUDAMatrixc(mask);
        cudaMaskBak = new CUDAMatrixc(mask);
        cudaPoints = new CUDAMatrixs(points);
        cudaPixels = new CUDAMatrixs(pixels);
        cudaK = MatrixConversion::toCUDA(cx->getK());
        costSummator = new Summator(n, 1, 1);
        hSummator = new Summator(n, 6, 6);
        mSummator = new Summator(n, 6, 1);
        bSummator = new Summator(n, 6, 1);

    }

    RegReport BARegistration::bundleAdjustment(Transform initT, bool robust, int iterations) {
        if (kxs->empty() || kys->empty()) {
            RegReport report;
            report.success = false;
            return report;
        }

        const int pointNum = pixels.rows();
        Vector2 mean = pixels.colwise().mean();
        double rms = 0;
        for (int k = 0; k < pixels.rows(); k++) {
            rms += (pixels.row(k).transpose() - mean).squaredNorm() / pointNum;
        }
        rms = sqrt(rms);
        if (rms < rmsThreshold) {
            RegReport report;
            report.success = false;
            return report;
        }

        Rotation R;
        Translation t;
        GeoUtil::T2Rt(initT, R, t);

        RegReport report;
        if (robust) {
            const int its = 4;
            for (int it = 0; it < its; it++) {
                report = bundleAdjustment(R, t, 10);
                // exchange the pointer
                CUDAMatrixc *temp = cudaMask;
                cudaMask = cudaMaskBak;
                cudaMaskBak = temp;
                if (report.iterations < iterations) {
                    break;
                }
            }

            // update keypoints
            vector<unsigned char> mask;
            cudaMask->download(mask);
            vector<int> selected_indexes;
            for (int i = 0; i < mask.size(); i++) {
                if (mask[i]) {
                    selected_indexes.emplace_back(i);
                }
            }

            report.success = selected_indexes.size() >= minInliers;
            if (report.success) {
                report = bundleAdjustment(R, t, iterations);

                vector<FeatureKeypoint> bKxs(kxs->begin(), kxs->end()), bKys(kys->begin(), kys->end());
                kxs->clear();
                kys->clear();
                for (int ind: selected_indexes) {
                    kxs->emplace_back(bKxs[ind]);
                    kys->emplace_back(bKys[ind]);
                }
            }
        } else {
            report = bundleAdjustment(R, t, iterations);
            report.success = true;
        }
        report.inlierNum = kxs->size();
        return report;
    }

    void
    BARegistration::bundleAdjustmentThread(Transform initT, bool robust, RegReport *report, cudaStream_t curStream) {
        stream = curStream;
        *report = bundleAdjustment(initT, robust);
    }

    void BARegistration::free() {
        delete cudaPoints;
        delete cudaPixels;
        delete cudaMask;
        delete cudaMaskBak;
        delete costSummator;
        delete hSummator;
        delete mSummator;
        delete bSummator;
    }

    RegReport BARegistration::bundleAdjustment(Rotation R, Translation t, int iterations) {
        RegReport report;
        report.pointsNum = cudaPoints->getRows();

        // Levenberg-Marquardt optimization algorithm.
        constexpr double kEpsilon = 1e-12;
        constexpr int max_lm_attempts = 50;
        constexpr int max_inner_iterations = 100;


        SE3 se(R, t);
        Vector6 initSeVec = se.log();

        double lambda = -1;
        for (int i = 0; i < iterations; ++i) {
            // so3 to R and t
            float4x4 cudaT = MatrixConversion::toCUDA(se.matrix());
            //compute jacobi matrix and cost
            computeBACostAndJacobi(*cudaPoints, *cudaPixels, cudaT, cudaK, *cudaMask, *costSummator, *hSummator,
                                   *mSummator, *bSummator);  // should always return true
            computerInliers(*costSummator, *cudaMaskBak, distTh);
            // Accumulate H and b.
            double cost = costSummator->sum()(0, 0);
            Eigen::Matrix<Scalar, 6, 6> H(hSummator->sum());
            Vector6 M(mSummator->sum());
            Vector6 b(bSummator->sum());
            if (lambda < 0) {
                constexpr float kInitialLambdaFactor = 0.01;
                lambda = kInitialLambdaFactor * 0.5 * H.trace();
            }
            Vector6 init_r = b, init_P;
            computeP(init_r, M, lambda, init_P);
            bool update_accepted = false;
            for (int lm_iteration = 0; lm_iteration < max_lm_attempts; ++lm_iteration) {
                // Solve the system.
                Vector6 r = init_r, p = init_P, finalDeltaVec = Vector6::Zero(), g, deltaVec = Vector6::Zero();
                // alpha_n = r^T * p
                double alpha_d, alpha_n = r.transpose() * p, beta_n;

                // Run PCG inner iterations to determine pcg_delta
                double prev_r_norm = numeric_limits<double>::infinity();
                int num_iterations_without_improvement = 0;

                double smallest_r_norm = numeric_limits<double>::infinity();
                for (int step = 0; step < max_inner_iterations; ++step) {
                    if (step > 0) {
                        // Set pcg_alpha_n_ to pcg_beta_n_ by swapping the pointers (since we
                        // don't need to preserve pcg_beta_n_).
                        // NOTE: This is wrong in the Opt paper, it says "beta" only instead of
                        //       "beta_n" which is something different.
                        std::swap(alpha_n, beta_n);
                    }

                    // Run PCG step 1 & 2
                    g = H * p;
                    alpha_d = p.transpose() * g;
                    // TODO: Default to 1 or to 0 if denominator is near-zero? stop optimization if that happens?
                    double alpha =
                            (alpha_d >= 1e-35f) ? (alpha_n / alpha_d) : 0;
                    deltaVec += alpha * p;
                    r = r - alpha * (g + lambda * p);
                    computeP(r, M, lambda, g);
                    beta_n = r.transpose() * g;

                    // Check for convergence of the inner iterations
                    double r_norm = sqrt(beta_n);
                    if (r_norm < smallest_r_norm) {
                        smallest_r_norm = r_norm;
                        finalDeltaVec = deltaVec;
                        if (r_norm == 0) {
                            break;
                        }
                    }

                    if (r_norm < prev_r_norm - 1e-3) {  // TODO: Make this threshold a parameter
                        num_iterations_without_improvement = 0;
                    } else {
                        ++num_iterations_without_improvement;
                        if (num_iterations_without_improvement >= 3) {
                            break;
                        }
                    }
                    prev_r_norm = r_norm;

                    // This (and some computations from step 2) is not necessary in the last
                    // iteration since the result is already computed in pcg_final_delta.
                    // NOTE: For best speed, could make a special version of step 2 (templated)
                    //       which excludes the unnecessary operations. Probably not very relevant though.
                    if (step < max_inner_iterations - 1) {
                        // TODO: Default to 1 or to 0 if denominator is near-zero? stop optimization if that happens?
                        double beta = (alpha_n >= 1e-35f) ? (beta_n / alpha_n) : 0;
                        p = g + beta * p;
                    }
                }  // end loop over PCG inner iterations
                // Compute the test state (constrained to the calibrated image area).
                Vector6 testSeVec = (SE3::exp(finalDeltaVec) * se).log();
                for (int j = 3; j < 6; j++) {
                    if (testSeVec(j) < initSeVec(j) - relaxtion) {
                        testSeVec(j) = initSeVec(j) - relaxtion;
                    }
                    if (testSeVec(j) > initSeVec(j) + relaxtion) {
                        testSeVec(j) = initSeVec(j) + relaxtion;
                    }
                }

                SE3 testSe = SE3::exp(testSeVec);
                // Compute the test cost.
                float4x4 testT = MatrixConversion::toCUDA(testSe.matrix());
                computeBACost(*cudaPoints, *cudaPixels, testT, cudaK, *cudaMask, *costSummator);
                double test_cost = costSummator->sum()(0, 0);

                if (!isnan(test_cost) && test_cost < cost) {
                    lambda *= 0.5;
                    se = testSe;
                    update_accepted = true;
                    break;
                } else {
                    lambda *= 2;
                }
            }

            report.cost = cost;
            report.iterations = i + 1;

            if (!update_accepted) {
                break;
            }

            if (cost < kEpsilon) {
                break;
            }
        }
        report.T = se.matrix();
        report.success = true;
        return report;
    }


    void computePX(VectorX r, VectorX M, double lambda, VectorX &p) {
        p.resize(r.rows(), r.cols());
        for (int i = 0; i < r.rows(); i++) {
            p(i) = r(i) / (M(i) + lambda);
        }
    }

    MatrixX featureKeypoints2Matrix(vector<FeatureKeypoint> &features) {
        int n = features.size();

        MatrixX matrix(n, 3);
        for (int i = 0; i < n; i++) {
            matrix.row(i) = features[i].toVector3();
        }
        return std::move(matrix);
    }

    void plusDeltaToSEVec(const VectorX& initSEsVec, double relaxtion, const TransformVector& transSEs, const VectorX& finalDeltaVec, VectorX& testSEsVec) {
        int tranNum = transSEs.size();
        testSEsVec.resize(tranNum*6);
        for(int i=0; i<tranNum; i++) {
            Vector6 testSeVec = (SE3::exp(finalDeltaVec.block<6,1>(6*i, 0))*SE3(transSEs[i])).log();
            Vector6 initSeVec = initSEsVec.block<6,1>(6*i, 0);
            for (int j = 3; j < 6; j++) {
                if (testSeVec(j) < initSeVec(j) - relaxtion) {
                    testSeVec(j) = initSeVec(j) - relaxtion;
                }
                if (testSeVec(j) > initSeVec(j) + relaxtion) {
                    testSeVec(j) = initSeVec(j) + relaxtion;
                }
            }
            testSEsVec.block<6, 1>(6*i, 0) = testSeVec;
        }

    }

    void computeTransFromLie(const VectorX& transSEs, TransformVector& transVec,  CUDAEdgeVector& cudaEdgeVector) {
        // compute transformation from se vector
        for(int j=0; j < transVec.size(); j++) {
            transVec[j] = SE3::exp(transSEs.block<6,1>(j*6, 0)).matrix();
        }

        // compute relative transformation for edges
        for(int j=0; j < cudaEdgeVector.getNum(); j++) {
            CUDAEdge edge = cudaEdgeVector[j];
            Transform trans = transVec[edge.indexX].inverse()*transVec[edge.indexY]*transVec[edge.indexZ];
            Transform transInv = trans.inverse();

            edge.transform = MatrixConversion::toCUDA(trans);
            edge.transformInv = MatrixConversion::toCUDA(transInv);
        }
    }
}
