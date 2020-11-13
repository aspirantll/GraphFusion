//
// Created by liulei on 2020/11/12.
//

#include "registrations.h"
#include "icp_multiview.cuh"

namespace rtf {
    MultiViewICP::MultiViewICP(const GlobalConfig& config) {
        rmsThreshold = config.rmsThreshold;
        relaxtion = config.relaxtion;
    }

    void computeP1(Vector6 r, Vector6 M, double lambda, Vector6 &p) {
        for (int i = 0; i < 6; i++) {
            p(i) = r(i) / (M(i) + lambda);
        }
    }

    // compute
    SEVector solvePCGIteration1(double lambda, double relaxtion, int max_inner_iterations, LMSumMats& lmSumMats, const SEVector& initSE3Vec, const SEVector& se3Vec) {
        SEVector init_r = lmSumMats.b, init_P;
        computeP1(init_r, lmSumMats.M, lambda, init_P);

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
            computeP1(r, lmSumMats.M, lambda, g);
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

    constexpr Scalar kHuberWeight1 = 1.2;
    template <typename Scalar> Scalar ComputeHuberCost1(Scalar squared_residual, Scalar huber_parameter) {
        if (squared_residual < huber_parameter * huber_parameter) {
            return static_cast<Scalar>(0.5) * squared_residual;
        } else {
            return huber_parameter * (sqrtf(squared_residual) - static_cast<Scalar>(0.5) * huber_parameter);
        }
    }

    MatrixX featureKeypoints2Matrix1(vector<FeatureKeypoint>& features) {
        int n = features.size();

        MatrixX matrix(n, 3);
        for(int i=0; i<n; i++) {
            matrix.row(i) = features[i].toVector3();
        }
        return matrix;
    }

    void computeTransFromLie1(const vector<SEVector, Eigen::aligned_allocator<SEVector>>& gtSEs,
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



    BAReport MultiViewICP::multiViewICP(ViewGraph &viewGraph, const vector<int>& cc, TransformVector& gtTransVec, double costThreshold) {
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

                    auto * kxPtr = new CUDAMatrixs(featureKeypoints2Matrix1(kx));
                    auto * kyPtr = new CUDAMatrixs(featureKeypoints2Matrix1(ky));

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
        computeTransFromLie1(gtSEs, deltaSEs, gtTransVec, deltaTransVec, cudaEdgeVector);

        double lambda = -1;
        for (int i = 0; i < kMaxIterations; ++i) {
            //compute jacobi matrix and cost
            computeMVICPCostAndJacobi(cudaEdgeVector, gtCudaSummators, deltaCudaSummators, costSummator);
            // Accumulate H and b.
            double cost = costSummator.sum()(0,0);
            double deltaCost = 0;
            for(const auto& deltaSE: deltaSEs) {
                deltaCost += ComputeHuberCost1(deltaSE.squaredNorm(), kHuberWeight1);
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
                    testGtSEs[j] << solvePCGIteration1(lambda, relaxtion, max_inner_iterations, gtLMSumMatsVector[j], initGtSEs[j], gtSEs[j]);
                }
                for(int j=0; j < n; j++) {
                    testDeltaSEs[j] << solvePCGIteration1(lambda, relaxtion, max_inner_iterations, deltaLMSumMatsVector[j], initDeltaSEs[j], deltaSEs[j]);
                }

                // compute relative transformation for edges
                computeTransFromLie1(testGtSEs, testDeltaSEs, testGtTransVec, testDeltaTransVec, cudaEdgeVector);

                // Compute the test cost.
                computeMVICPCost(cudaEdgeVector, costSummator);
                double test_cost = costSummator.sum()(0,0);
                for(const auto& deltaSE: testDeltaSEs) {
                    test_cost += ComputeHuberCost1(deltaSE.squaredNorm(), kHuberWeight1);
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
