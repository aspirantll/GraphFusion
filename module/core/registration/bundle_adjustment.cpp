//
// Created by liulei on 2020/6/12.
//

#include <utility>

#include "registrations.h"
#include "bundle_adjustment.cuh"

namespace rtf {

    void computeP(Vector6 r, Vector6 M, double lambda, Vector6 &p) {
        for (int i = 0; i < 6; i++) {
            p(i) = r(i) / (M(i) + lambda);
        }
    }

    Transform vec2Matrix(Vector6 &soVec) {
        Vector3 so3Vec;
        so3Vec << soVec(0), soVec(1), soVec(2);

        Rotation R = Sophus::SO3<Scalar>::exp(so3Vec).matrix();
        Translation tVec;
        tVec << soVec(3), soVec(4), soVec(5);

        Transform T;
        GeoUtil::Rt2T(R, tVec, T);
        return T;
    }

    // compute
    SEVector solvePCGIteration(double lambda, double relaxtion, int max_inner_iterations, LMSumMats& lmSumMats, const SEVector& initSE3Vec, const SEVector& se3Vec) {
        SEVector init_r = lmSumMats.b, init_P;
        computeP(init_r, lmSumMats.M, lambda, init_P);

        // Solve the system.
        SEVector r = init_r, p = init_P, finalDeltaVec = SEVector::Zero(), g, deltaVec = SEVector::Zero();
        // alpha_n = r^T * p
        double alpha_d, alpha_n = r.transpose() * p, beta_n;

        // Run PCG inner iterations to determine pcg_delta
        double prev_r_norm = numeric_limits<double>::infinity();
        int num_iterations_without_improvement = 0;

        double initial_r_norm = numeric_limits<double>::quiet_NaN();
        double smallest_r_norm = numeric_limits<double>::infinity();
        for (int step = 0; step < max_inner_iterations; ++ step) {
            if (step > 0) {
                // Set pcg_alpha_n_ to pcg_beta_n_ by swapping the pointers (since we
                // don't need to preserve pcg_beta_n_).
                // NOTE: This is wrong in the Opt paper, it says "beta" only instead of
                //       "beta_n" which is something different.
                std::swap(alpha_n, beta_n);
            }

            // Run PCG step 1 & 2
            g = lmSumMats.H * p;
            alpha_d = p.transpose() * g;
            // TODO: Default to 1 or to 0 if denominator is near-zero? stop optimization if that happens?
            double alpha =
                    (alpha_d >= 1e-35f) ? (alpha_n/alpha_d) : 0;
            deltaVec += alpha * p;
            r = r - alpha*(g + lambda*p);
            computeP(r, lmSumMats.M, lambda, g);
            beta_n = r.transpose() * g;

            // Check for convergence of the inner iterations
            double r_norm = sqrt(beta_n);
            if (step == 0) {
                initial_r_norm = r_norm;
            }
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
                ++ num_iterations_without_improvement;
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
                p = g + beta*p;
            }
        }  // end loop over PCG inner iterations

        // Compute the test state (constrained to the calibrated image area).
        SEVector testSe3 = se3Vec + finalDeltaVec;

        for(int j=0; j<3; j++) {
            if(testSe3(j)<initSE3Vec(j)-relaxtion) {
                testSe3(j) = initSE3Vec(j)-relaxtion;
            }
            if(testSe3(j)>initSE3Vec(j)+relaxtion) {
                testSe3(j) = initSE3Vec(j)+relaxtion;
            }
        }

        return testSe3;
    }

    BARegistration::BARegistration(const GlobalConfig &config) {
        rmsThreshold = config.rmsThreshold;
        relaxtion = config.relaxtion;
        distTh = config.maxPnPResidual;
        minInlierRatio = config.minInlierRatio;
        minInliers = config.kMinInliers;

    }


    BAReport BARegistration::bundleAdjustment(Transform initT, shared_ptr<Camera> cx, shared_ptr<Camera> cy,
                                              vector<FeatureKeypoint> &kxs, vector<FeatureKeypoint> &kys, bool robust) {
        alloc(std::move(cx), std::move(cy), kxs, kys);
        BAReport report = bundleAdjustment(initT, robust);
        free();
        return report;
    }

    void BARegistration::alloc(shared_ptr<Camera> cx, shared_ptr<Camera> cy, vector<FeatureKeypoint> &kxs,
                               vector<FeatureKeypoint> &kys) {
        this->kxs = &kxs;
        this->kys = &kys;

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

    BAReport BARegistration::bundleAdjustment(Transform initT, bool robust, int iterations) {
        if (kxs->empty() || kys->empty()) {
            BAReport report;
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
//        cout << "rms:" << rms << endl;
        if (rms < rmsThreshold) {
            BAReport report;
            report.success = false;
            return report;
        }

        Rotation R;
        Translation t;
        GeoUtil::T2Rt(initT, R, t);

        BAReport report;
        if (robust) {
            const int its = 4;
            for (int it = 0; it < its; it++) {
                report = bundleAdjustment(R, t, iterations);
                // exchange the pointer
                CUDAMatrixc * temp = cudaMask;
                cudaMask = cudaMaskBak;
                cudaMaskBak = temp;
            }

            // update keypoints
            vector<unsigned char> mask;
            cudaMask->download(mask);
            vector<FeatureKeypoint> bKxs(kxs->begin(), kxs->end()), bKys(kys->begin(), kys->end());
            kxs->clear();
            kys->clear();
            for (int i = 0; i < mask.size(); i++) {
                if (mask[i]) {
                    kxs->emplace_back(bKxs[i]);
                    kys->emplace_back(bKys[i]);
                }
            }
            report.success = kxs->size() >= max(pointNum*minInlierRatio, minInliers);
        } else {
            report = bundleAdjustment(R, t, iterations);
            report.success = true;
        }
        return report;
    }

    void BARegistration::bundleAdjustmentThread(Transform initT, bool robust, BAReport* report, cudaStream_t curStream) {
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

    BAReport BARegistration::bundleAdjustment(Rotation &R, Translation &t, int iterations) {
        BAReport report;
        report.pointsNum = cudaPoints->getRows();

        // Levenberg-Marquardt optimization algorithm.
        constexpr double kEpsilon = 1e-12;
        constexpr int max_lm_attempts = 50;
        constexpr int max_inner_iterations = 100;

        Sophus::SO3<Scalar> initSo3(R);
        Vector3 initSo3Vec = initSo3.log();
        Vector6 soVec;
        soVec << initSo3Vec, t;

        double lambda = -1;
        for (int i = 0; i < iterations; ++i) {
            // so3 to R and t
            float4x4 cudaT = MatrixConversion::toCUDA(vec2Matrix(soVec));
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
                Vector6 testSo3 = soVec + finalDeltaVec;

                for (int j = 0; j < 3; j++) {
                    if (testSo3(j) < initSo3Vec(j) - relaxtion) {
                        testSo3(j) = initSo3Vec(j) - relaxtion;
                    }
                    if (testSo3(j) > initSo3Vec(j) + relaxtion) {
                        testSo3(j) = initSo3Vec(j) + relaxtion;
                    }
                }
                // Compute the test cost.
                float4x4 testT = MatrixConversion::toCUDA(vec2Matrix(testSo3));
                computeBACost(*cudaPoints, *cudaPixels, testT, cudaK, *cudaMask, *costSummator);
                double test_cost = costSummator->sum()(0, 0);

                if (test_cost < cost) {
                    lambda *= 0.5;
                    soVec = testSo3;
                    update_accepted = true;
                    break;
                } else {
                    lambda *= 2;
                }
            }

            Vector3 so3Vec;
            so3Vec << soVec(0), soVec(1), soVec(2);
            R = Sophus::SO3<Scalar>::exp(so3Vec).matrix();
            t << soVec(3), soVec(4), soVec(5);
            report.cost = cost;
            report.iterations = i + 1;

            if (!update_accepted) {
                report.success = cost < kEpsilon;
                break;
            }

            if (cost < kEpsilon) {
                report.success = true;
                break;
            }
        }
        GeoUtil::Rt2T(R, t, report.T);
        return report;
    }


    constexpr Scalar kHuberWeight = 1.2;
    template <typename Scalar> Scalar ComputeHuberCost(Scalar squared_residual, Scalar huber_parameter) {
        if (squared_residual < huber_parameter * huber_parameter) {
            return static_cast<Scalar>(0.5) * squared_residual;
        } else {
            return huber_parameter * (sqrtf(squared_residual) - static_cast<Scalar>(0.5) * huber_parameter);
        }
    }

    MatrixX featureKeypoints2Matrix(vector<FeatureKeypoint>& features) {
        int n = features.size();

        MatrixX matrix(n, 3);
        for(int i=0; i<n; i++) {
            matrix.row(i) = features[i].toVector3();
        }
        return matrix;
    }

    void computeTransFromLie(const vector<SEVector, Eigen::aligned_allocator<SEVector>>& gtSEs,
                             const vector<SEVector, Eigen::aligned_allocator<SEVector>>& deltaSEs,
                             TransformVector& gtTransVec, TransformVector& deltaTransVec,  CUDAEdgeVector& cudaEdgeVector) {
        // compute transformation from se vector
        for(int j=0; j < gtSEs.size(); j++) {
            gtTransVec[j] << Sophus::SE3<Scalar>::exp(gtSEs[j]).matrix();
        }
        for(int j=0; j < deltaSEs.size(); j++) {
            deltaTransVec[j] << Sophus::SE3<Scalar>::exp(deltaSEs[j]).matrix();
        }
        // compute relative transformation for edges
        for(int j=0; j < cudaEdgeVector.getNum(); j++) {
            CUDAEdge edge = cudaEdgeVector[j];
            Transform trans = GeoUtil::reverseTransformation(gtTransVec[edge.indexX])*gtTransVec[edge.indexY]*deltaTransVec[j];
            edge.transform = MatrixConversion::toCUDA(trans);
        }
    }

    BAReport BARegistration::multiViewBundleAdjustment(ViewGraph &viewGraph, const vector<int>& cc, TransformVector& gtTransVec, double costThreshold) {
        int poseNum = cc.size();
        BAReport report;
        //1. determine points num and collect edge for each pose
        CUDAEdgeVector cudaEdgeVector;
        TransformVector deltaTransVec;
        vector<int> gTCount(poseNum);
        vector<int> deltaCount;
        int totalCount = 0;
        // initialize gtCount and cudaGtTransVec
        for(int i=0; i<poseNum; i++) {
            gTCount[i] = 0;
        }
        for(int i=0; i<poseNum; i++) {
            for(int j=i+1; j<poseNum; j++) {
                Edge edge = viewGraph.getEdge(cc[i], cc[j]);
                if(!edge.isUnreachable()&&(costThreshold==-1||edge.getCost()<=costThreshold)) {
                    vector<FeatureKeypoint>& kx = edge.getKxs();
                    vector<FeatureKeypoint>& ky = edge.getKys();
                    // construct the cuda edge
                    CUDAEdge cudaEdge;
                    cudaEdge.indexX = i;
                    cudaEdge.indexY = j;
                    cudaEdge.sumIndexX = gTCount[i];
                    cudaEdge.sumIndexY = gTCount[j];
                    cudaEdge.costIndex = totalCount;

                    auto * kxPtr = new CUDAMatrixs(featureKeypoints2Matrix(kx));
                    auto * kyPtr = new CUDAMatrixs(featureKeypoints2Matrix(ky));

                    cudaEdge.kx = *kxPtr;
                    cudaEdge.ky = *kyPtr;

                    cudaEdge.intrinsicX = MatrixConversion::toCUDA(viewGraph[cc[i]].getK());
                    cudaEdge.intrinsicY = MatrixConversion::toCUDA(viewGraph[cc[j]].getK());

                    // count
                    int curCount = kx.size();
                    cudaEdge.count = curCount;
                    gTCount[i] += curCount;
                    gTCount[j] += curCount;
                    totalCount += curCount;
                    deltaCount.emplace_back(curCount);

                    auto deltaTrans = GeoUtil::reverseTransformation(gtTransVec[j]) * gtTransVec[i] *
                            edge.getTransform();
                    deltaTransVec.emplace_back(deltaTrans);

                    cudaEdge.transform = MatrixConversion::toCUDA(edge.getTransform());

                    cudaEdgeVector.addItem(cudaEdge);
                }
            }
        }
        report.pointsNum = totalCount;
        int m = gtTransVec.size();
        int n = deltaTransVec.size();
        cout << "frame num:" << m << endl;
        cout << "pose num:" << n << endl;
        if(totalCount==0) {
            report.success = false;
            return report;
        }

        //2. initialize SE and summators
        vector<SEVector, Eigen::aligned_allocator<SEVector>> gtSEs(m);
        vector<LMSummators *> gtLMSummators(m);
        CUDAVector<CUDALMSummators> gtCudaSummators;
        for(int i=0; i<m; i++) {
            gtSEs[i] = Sophus::SE3<Scalar>(gtTransVec[i]).log();
            gtLMSummators[i] = new LMSummators(gTCount[i]);
            gtCudaSummators.addItem(gtLMSummators[i]->uploadToCUDA());
        }

        vector<SEVector, Eigen::aligned_allocator<SEVector>> deltaSEs(n);
        vector<LMSummators *> deltaLMSummators(n);
        CUDAVector<CUDALMSummators> deltaCudaSummators;
        for(int i=0; i < n; i++) {
            deltaSEs[i] = Sophus::SE3<Scalar>(deltaTransVec[i]).log();
            deltaLMSummators[i] = new LMSummators(deltaCount[i]);
            deltaCudaSummators.addItem(deltaLMSummators[i]->uploadToCUDA());
        }

        Summator costSummator(totalCount, 1, 1);

        // 3.global bundle adjustment
        // Levenberg-Marquardt optimization algorithm.
        constexpr double kEpsilon = 1e-12;
        constexpr int kMaxIterations = 100;
        constexpr int max_lm_attempts = 50;
        constexpr int max_inner_iterations = 100;

        // initialize the variables
        vector<SEVector, Eigen::aligned_allocator<SEVector>> initGtSEs = gtSEs, initDeltaSEs = deltaSEs;
        TransformVector testGtTransVec(m), testDeltaTransVec(n);
        vector<SEVector, Eigen::aligned_allocator<SEVector>> testGtSEs = gtSEs,testDeltaSEs = deltaSEs;

        // compute the relative transform
        computeTransFromLie(gtSEs, deltaSEs, gtTransVec, deltaTransVec, cudaEdgeVector);

        double lambda = -1;
        for (int i = 0; i < kMaxIterations; ++i) {
            //compute jacobi matrix and cost
            computeMVBACostAndJacobi(cudaEdgeVector, gtCudaSummators, deltaCudaSummators, costSummator);
            // Accumulate H and b.
            double cost = costSummator.sum()(0,0);
            double deltaCost = 0;
            for(const auto& deltaSE: deltaSEs) {
                deltaCost += ComputeHuberCost(deltaSE.squaredNorm(), kHuberWeight);
            }
            cout << "ba cost:" << cost << ", delta cost:" << deltaCost << endl;
            cost += deltaCost;
            double meanTrace = 0;
            int num = 0;
            vector<LMSumMats> gtLMSumMatsVector;
            for(int j=0; j < m; j++) {
                if(gTCount[j]!=0) {
                    auto mat = gtLMSummators[j]->sum();
                    gtLMSumMatsVector.emplace_back(mat);
                    meanTrace += mat.H.trace();
                    num ++;
                } else {
                    gtLMSumMatsVector.emplace_back(LMSumMats());
                }
            }

            vector<LMSumMats> deltaLMSumMatsVector;
            for(auto* ptr: deltaLMSummators) {
                auto mat = ptr->sum();
                deltaLMSumMatsVector.emplace_back(mat);
                meanTrace += mat.H.trace();
                num ++;
            }
            meanTrace /= double(num);

            if (lambda < 0) {
                constexpr double kInitialLambdaFactor = 0.01;
                lambda = kInitialLambdaFactor * 0.5 * meanTrace;
            }

            bool update_accepted = false;
            for (int lm_iteration = 0; lm_iteration < max_lm_attempts; ++ lm_iteration) {
                for(int j=1; j < m; j++) { // it should fix the first one
                    testGtSEs[j] << solvePCGIteration(lambda, relaxtion, max_inner_iterations, gtLMSumMatsVector[j], initGtSEs[j], gtSEs[j]);
                }
                for(int j=0; j < n; j++) {
                    testDeltaSEs[j] << solvePCGIteration(lambda, relaxtion, max_inner_iterations, deltaLMSumMatsVector[j], initDeltaSEs[j], deltaSEs[j]);
                }

                // compute relative transformation for edges
                computeTransFromLie(testGtSEs, testDeltaSEs, testGtTransVec, testDeltaTransVec, cudaEdgeVector);

                // Compute the test cost.
                computeMVBACost(cudaEdgeVector, costSummator);
                double test_cost = costSummator.sum()(0,0);
                for(const auto& deltaSE: testDeltaSEs) {
                    test_cost += ComputeHuberCost(deltaSE.squaredNorm(), kHuberWeight);
                }

                if (test_cost < cost) {
                    lambda *= 0.5;
                    gtSEs = testGtSEs;
                    deltaSEs = testDeltaSEs;
                    gtTransVec = testGtTransVec;
                    deltaTransVec = testDeltaTransVec;
                    update_accepted = true;
                    break;
                } else {
                    lambda *= 2;
                }
            }

            report.cost = cost;
            report.iterations = i+1;

            if (!update_accepted||cost < kEpsilon) {
                break;
            }
        }


        for(auto *ptr: gtLMSummators) {
            delete ptr;
        }
        for(auto *ptr: deltaLMSummators) {
            delete ptr;
        }

        report.success = true;
        return report;
    }
}
