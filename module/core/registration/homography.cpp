//
// Created by liulei on 2020/9/26.
//

#include "registrations.h"
#include "epipolar_geometry.cuh"
#include "../../tool/math.h"

namespace rtf {
    class HomographyMatrixEstimator : public Estimator<Point3D, Point3D, Matrix3> {
    protected:
        shared_ptr<CUDAMatrixs> cudaX;
        shared_ptr<CUDAMatrixs> cudaY;
        shared_ptr<CUDAMatrixs> cudaA;
        shared_ptr<CUDAMatrixs> cudaH;
        shared_ptr<CUDAMatrixs> cudaTransformX;
        shared_ptr<CUDAMatrixs> cudaTransformY;
        shared_ptr<CUDAMatrixs> cudaResidual;
        shared_ptr<Summator> summator;
    public:

        HomographyMatrixEstimator(long pointsNum) {
            cudaX = make_shared<CUDAMatrixs>(pointsNum, 3);
            cudaY = make_shared<CUDAMatrixs>(pointsNum, 3);
            cudaA = make_shared<CUDAMatrixs>(2 * pointsNum, 9);
            cudaH = make_shared<CUDAMatrixs>(3, 3);
            cudaTransformX = make_shared<CUDAMatrixs>(3, 3);
            cudaTransformY = make_shared<CUDAMatrixs>(3, 3);
            cudaResidual = make_shared<CUDAMatrixs>(pointsNum, 1);
            summator = make_shared<Summator>(pointsNum, 1, 3);
        }

        void normalizePoints(MatrixX &data, CUDAMatrixs &mat, CUDAMatrixs &cudaTransform, Summator &summator) {
            int n = mat.getRows();

            // compute mean value point
            MatrixX meanPoint = summator.sum(data, 1) / n;
            CUDAMatrixs cudaMeanPoint(meanPoint);

            // compute rms
            double rms = sqrt(computeRMS(mat, cudaMeanPoint, summator) / n);

            // compose the transform
            double normFactor = sqrt(2.0) / rms;
            Matrix3 transform;
            transform << normFactor, 0, -normFactor * meanPoint(0, 0), 0, normFactor, -normFactor *
                                                                                      meanPoint(0, 1), 0, 0, 1;
            cudaTransform.upload(transform);

            // transform all points
            transformPoints(mat, cudaTransform);
            MatrixX normalizedMat;
            mat.download(normalizedMat);

        }

    public:
        /** must define this virances*/
        static int kMinNumSamples;

        typedef Point3D I1;
        typedef Point3D I2;
        typedef Matrix3 M;


        EstimateReport estimate(vector<Point3D> &x, vector<Point3D> &y) override {
            EstimateReport report;
            report.x = x;
            report.y = y;

            // upload the point matrix
            MatrixX xMatrix = PointUtil::vec2Matrix(x);
            MatrixX yMatrix = PointUtil::vec2Matrix(y);

            cudaX->upload(xMatrix);
            cudaY->upload(yMatrix);

            // normalize for x
            normalizePoints(xMatrix, *cudaX, *cudaTransformX, *summator);
            // normalize for y
            normalizePoints(yMatrix, *cudaY, *cudaTransformY, *summator);

            //download data from device
            MatrixX xTrans, yTrans;
            cudaTransformX->download(xTrans);
            cudaTransformY->download(yTrans);

            // compute matrix A
            computeHomoAMatrix(*cudaX, *cudaY, *cudaA);
            MatrixX A;
            cudaA->download(A);

            // solve essential matrix problem
            Eigen::JacobiSVD<MatrixX> svd(A.transpose() * A, Eigen::ComputeFullU | Eigen::ComputeFullV);
            VectorX nullSpace = svd.matrixV().col(8);
            const Eigen::Map<const Matrix3> Ht(nullSpace.data());

            // denormalize for homography matrix
            report.model = xTrans.inverse() * Ht.transpose() * yTrans;

            report.success = true;
            report.numOfTrials = 1;
            return report;
        }

        vector<double> computeResiduals(Matrix3 &model, vector<Point3D> &tx, vector<Point3D> &ty) override {
            CHECK_EQ(tx.size(), ty.size());
            int n = tx.size();

            cudaH->upload(model);
            cudaX->upload(PointUtil::vec2Matrix(tx));
            cudaY->upload(PointUtil::vec2Matrix(ty));

            computeHomoResidualsCUDA(*cudaX, *cudaY, *cudaH, *cudaResidual);

            vector<Scalar> residuals(n);
            cudaResidual->download(residuals.data(), 0, n);
            return vector<double>(residuals.begin(), residuals.end());
        }
    };

    int HomographyMatrixEstimator::kMinNumSamples = 4;

    HomographyRegistration::HomographyRegistration(const GlobalConfig &config) : config(config) {
        this->config.maxResidual = config.maxHomoResidual;
    }

    HomographyRegistration::HomographyRegistration(const RANSAC2DConfig &config) : config(config) {}


    double ComputeOppositeOfMinor(const Matrix3 &matrix, const size_t row,
                                  const size_t col) {
        const size_t col1 = col == 0 ? 1 : 0;
        const size_t col2 = col == 2 ? 1 : 2;
        const size_t row1 = row == 0 ? 1 : 0;
        const size_t row2 = row == 2 ? 1 : 2;
        return (matrix(row1, col2) * matrix(row2, col1) -
                matrix(row1, col1) * matrix(row2, col2));
    }

    Matrix3 ComputeHomographyRotation(const Matrix3 &H_normalized,
                                              const Translation &tstar,
                                              const Vector3 &n,
                                              const double v) {
        return H_normalized *
               (Matrix3::Identity() - (2.0 / v) * tstar * n.transpose());
    }


    void HomographyRegistration::decomposeHomographyMatrix(Matrix3 &H, Intrinsic kx, Intrinsic ky,
                                                           vector<Point3D> &x, vector<Point3D> &y, Rotation &R,
                                                           Translation &t, vector<double> &xDs,
                                                           vector<double> &yDs) {
        CHECK_EQ(x.size(), y.size());
        // Remove calibration from homography.
        Matrix3 H_normalized = kx.inverse() * H * ky;

        // Remove scale from normalized homography.
        Eigen::JacobiSVD<Matrix3> hmatrix_norm_svd(H_normalized);
        H_normalized.array() /= hmatrix_norm_svd.singularValues()[1];

        const Matrix3 S =
                H_normalized.transpose() * H_normalized - Matrix3::Identity();

        std::vector<Rotation> RCandidates;
        std::vector<Translation> tCandidates;
        std::vector<Vector3> nCandidates;

        // Check if H is rotation matrix.
        const double kMinInfinityNorm = 1e-10;
        if (S.lpNorm<Eigen::Infinity>() < kMinInfinityNorm) {
            RCandidates = {H_normalized};
            tCandidates = {Translation::Zero()};
            nCandidates = {Translation::Zero()};
        } else {
            const double M00 = ComputeOppositeOfMinor(S, 0, 0);
            const double M11 = ComputeOppositeOfMinor(S, 1, 1);
            const double M22 = ComputeOppositeOfMinor(S, 2, 2);

            const double rtM00 = std::sqrt(M00);
            const double rtM11 = std::sqrt(M11);
            const double rtM22 = std::sqrt(M22);

            const double M01 = ComputeOppositeOfMinor(S, 0, 1);
            const double M12 = ComputeOppositeOfMinor(S, 1, 2);
            const double M02 = ComputeOppositeOfMinor(S, 0, 2);

            const int e12 = SignOfNumber(M12);
            const int e02 = SignOfNumber(M02);
            const int e01 = SignOfNumber(M01);

            const double nS00 = std::abs(S(0, 0));
            const double nS11 = std::abs(S(1, 1));
            const double nS22 = std::abs(S(2, 2));

            const std::array<double, 3> nS{{nS00, nS11, nS22}};
            const size_t idx =
                    std::distance(nS.begin(), std::max_element(nS.begin(), nS.end()));

            Vector3 np1;
            Vector3 np2;
            if (idx == 0) {
                np1[0] = S(0, 0);
                np2[0] = S(0, 0);
                np1[1] = S(0, 1) + rtM22;
                np2[1] = S(0, 1) - rtM22;
                np1[2] = S(0, 2) + e12 * rtM11;
                np2[2] = S(0, 2) - e12 * rtM11;
            } else if (idx == 1) {
                np1[0] = S(0, 1) + rtM22;
                np2[0] = S(0, 1) - rtM22;
                np1[1] = S(1, 1);
                np2[1] = S(1, 1);
                np1[2] = S(1, 2) - e02 * rtM00;
                np2[2] = S(1, 2) + e02 * rtM00;
            } else if (idx == 2) {
                np1[0] = S(0, 2) + e01 * rtM11;
                np2[0] = S(0, 2) - e01 * rtM11;
                np1[1] = S(1, 2) + rtM00;
                np2[1] = S(1, 2) - rtM00;
                np1[2] = S(2, 2);
                np2[2] = S(2, 2);
            }

            const double traceS = S.trace();
            const double v = 2.0 * std::sqrt(1.0 + traceS - M00 - M11 - M22);

            const double ESii = SignOfNumber(S(idx, idx));
            const double r_2 = 2 + traceS + v;
            const double nt_2 = 2 + traceS - v;

            const double r = std::sqrt(r_2);
            const double n_t = std::sqrt(nt_2);

            const Vector3 n1 = np1.normalized();
            const Vector3 n2 = np2.normalized();

            const double half_nt = 0.5 * n_t;
            const double esii_t_r = ESii * r;

            const Translation t1_star = half_nt * (esii_t_r * n2 - n_t * n1);
            const Translation t2_star = half_nt * (esii_t_r * n1 - n_t * n2);

            const Rotation R1 =
                    ComputeHomographyRotation(H_normalized, t1_star, n1, v);
            const Translation t1 = R1 * t1_star;

            const Rotation R2 =
                    ComputeHomographyRotation(H_normalized, t2_star, n2, v);
            const Translation t2 = R2 * t2_star;

            RCandidates = {R1, R1, R2, R2};
            tCandidates = {t1, -t1, t2, -t2};
            nCandidates = {-n1, n1, -n2, n2};
        }

        int n = x.size();
        Eigen::VectorXi supportVec(n);

        int bestSupportPointsNum = 0;
        for (int i = 0; i < RCandidates.size(); ++i) {
            Rotation RCandidate = RCandidates[i];
            Translation tCandidate = tCandidates[i];

            Eigen::Matrix<Scalar, 3, 4> projectMatrix1, projectMatrix2 = Eigen::Matrix<Scalar, 3, 4>::Identity();
            GeoUtil::Rt2P(RCandidate, tCandidate, projectMatrix1);

            vector<double> xDepths(n), yDepths(n);
            for (int j = 0; j < n; ++j) {
                xDepths[j] = -1;
                yDepths[j] = -1;
            }

            const double kMinDepth = std::numeric_limits<double>::epsilon();
            const double kMaxDepth = 1000.0f * (RCandidate.transpose() * tCandidate).norm();

            // find support points
            for (int j = 0; j < n; ++j) {
                supportVec[j] = 0;
                Vector3 point1 = x[j].toVector3();
                Vector3 point2 = y[j].toVector3();

                // compute 3d point
                Matrix4 A;

                A.row(0) = point1(0) * projectMatrix1.row(2) - projectMatrix1.row(0);
                A.row(1) = point1(1) * projectMatrix1.row(2) - projectMatrix1.row(1);
                A.row(2) = point2(0) * projectMatrix2.row(2) - projectMatrix2.row(0);
                A.row(3) = point2(1) * projectMatrix2.row(2) - projectMatrix2.row(1);

                Eigen::JacobiSVD<Matrix4> svd(A, Eigen::ComputeFullV);

                Vector3 point3D = svd.matrixV().col(3).hnormalized();

                // compute depth for point
                const double depth1 = GeoUtil::calculateDepth(projectMatrix1, point3D);
                if (depth1 > kMinDepth && depth1 < kMaxDepth) {
                    const double depth2 = GeoUtil::calculateDepth(projectMatrix2, point3D);
                    if (depth2 > kMinDepth && depth2 < kMaxDepth) {
                        xDepths[j] = depth1;
                        yDepths[j] = depth2;
                        supportVec[j] = 1;
                    }
                }
            }

            int supportPointsNum = supportVec.sum();
            if (supportPointsNum >= bestSupportPointsNum) {
                R = RCandidate;
                t = tCandidate;
                xDs = xDepths;
                yDs = yDepths;
                bestSupportPointsNum = supportPointsNum;
            }
        }
    }


    RANSAC2DReport HomographyRegistration::registrationFunction(FeatureMatches &featureMatches) {
        RANSAC2DReport report;
        report.pointNum = featureMatches.size();
        if (featureMatches.size() < config.kMinMatches) {
            report.success = false;
            return report;
        }

        auto kx = featureMatches.getKx();
        auto ky = featureMatches.getKy();

        Rotation R;
        Translation t;

        shared_ptr<Camera> cx = featureMatches.getCx();
        shared_ptr<Camera> cy = featureMatches.getCy();
        vector<Point3D> pxs, pys;

        for (int i = 0; i < featureMatches.size(); i++) {
            FeatureMatch match = featureMatches.getMatch(i);

            uint32_t indx = match.getPX();
            uint32_t indy = match.getPY();

            auto px = kx[indx];
            auto py = ky[indy];

            pxs.emplace_back(cx->getCameraModel()->getRay((int) px->x, (int) px->y));
            pys.emplace_back(cy->getCameraModel()->getRay((int) py->x, (int) py->y));
        }

        HomographyMatrixEstimator estimator(featureMatches.size());
        SimpleSupportMeasurer supportMeasurer;
        typedef LORANSAC<HomographyMatrixEstimator, HomographyMatrixEstimator, RandomSampler<Point3D, Point3D>> EMRANSAC;
        EMRANSAC loransac(config, estimator, estimator, supportMeasurer);
        EMRANSAC::BASE::RANSACReport r = loransac.estimate(pxs, pys);
        if (r.success) {
            if(r.support.inlierRatio()<config.minInlierRatio) {
                r.success = false;
            }else {
                decomposeHomographyMatrix(r.model, cx->getK(), cy->getK(), pxs, pys, R, t, report.xDs, report.yDs);
                //make sure that initR is orthogonal matrix
                if(!Sophus::isOrthogonal(R)) {
                    Eigen::AngleAxis<Scalar> rotationVector(R);
                    if((R - rotationVector.toRotationMatrix()).norm()<1e-4) {
                        R = rotationVector.toRotationMatrix();
                    }else {
                        r.success = false;
                    }
                }

                if(R.determinant()<0) {
                    R.col(2) *= -1;
                }
            }
        }

        report.success = r.success;
        report.inliers = r.support.inlierIndexes;
        report.maxResidual = r.maxResidual;

        vector<int> inliers = report.inliers;
        vector<double> xDs = report.xDs, yDs = report.yDs;
        if (!report.success || inliers.size() < config.kMinInliers) {
            report.success = false;
            return report;
        }
        // compute scale for some points
        int count = 0;
        double scale = 0;

        for (int k = 0; k < featureMatches.size(); k++) {
            FeatureMatch match = featureMatches.getMatch(k);

            uint32_t indx = match.getPX();
            uint32_t indy = match.getPY();

            const FeatureKeypoint &px = *kx[indx];
            const FeatureKeypoint &py = *ky[indy];

            double dx = px.z;
            double dy = py.z;

            if (xDs[k] > 0 && yDs[k] > 0) {
                scale += (dx / xDs[k] + dy / yDs[k]) / 2;
                count++;
            }

        }
        scale /= count;

        // compose parameters for ba
        vector<int> &kps1 = report.kps1, &kps2 = report.kps2;

        for (int k = 0; k < inliers.size(); k++) {
            int index = inliers[k];
            FeatureMatch match = featureMatches.getMatch(index);

            uint32_t indx = match.getPX();
            uint32_t indy = match.getPY();

            const FeatureKeypoint &px = *kx[indx];
            const FeatureKeypoint &py = *ky[indy];

            kps1.emplace_back(indx);
            kps2.emplace_back(indy);
        }

        if (kps1.size() < config.kMinInliers) {
            report.success = false;
            return report;
        }

        GeoUtil::Rt2T(R, t * scale, report.T);
        report.success = true;
        return report;
    }
}
