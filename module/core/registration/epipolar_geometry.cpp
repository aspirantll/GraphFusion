//
// Created by liulei on 2020/6/20.
//


#include "registrations.h"
#include "epipolar_geometry.cuh"

namespace rtf {


    class EssentialMatrixEstimator: public Estimator<Point3D, Point3D, Matrix3> {
    protected:
        shared_ptr<CUDAMatrixs> cudaX;
        shared_ptr<CUDAMatrixs> cudaY;
        shared_ptr<CUDAMatrixs> cudaA;
        shared_ptr<CUDAMatrixs> cudaE;
        shared_ptr<CUDAMatrixs> cudaTransformX;
        shared_ptr<CUDAMatrixs> cudaTransformY;
        shared_ptr<CUDAMatrixs> cudaResidual;
        shared_ptr<Summator> summator;
    public:

        EssentialMatrixEstimator(long pointsNum) {
            cudaX = make_shared<CUDAMatrixs>(pointsNum, 3);
            cudaY = make_shared<CUDAMatrixs>(pointsNum, 3);
            cudaA = make_shared<CUDAMatrixs>(pointsNum, 9);
            cudaE = make_shared<CUDAMatrixs>(3, 3);
            cudaTransformX = make_shared<CUDAMatrixs>(3, 3);
            cudaTransformY = make_shared<CUDAMatrixs>(3, 3);
            cudaResidual = make_shared<CUDAMatrixs>(pointsNum, 1);
            summator = make_shared<Summator>(pointsNum, 1, 3);
        }

        void normalizePoints(MatrixX& data, CUDAMatrixs& mat, CUDAMatrixs& cudaTransform, Summator& summator) {
            int n = mat.getRows();

            // compute mean value point
            MatrixX meanPoint = summator.sum(data, 1)/n;
            CUDAMatrixs cudaMeanPoint(meanPoint);

            // compute rms
            double rms = sqrt(computeRMS(mat, cudaMeanPoint, summator) / n);

            // compose the transform
            double normFactor = sqrt(2.0)/rms;
            Matrix3 transform;
            transform << normFactor, 0, -normFactor*meanPoint(0,0), 0, normFactor, -normFactor*meanPoint(0,1), 0, 0, 1;
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
            normalizePoints(xMatrix, *cudaX,*cudaTransformX, *summator);
            // normalize for y
            normalizePoints(yMatrix, *cudaY, *cudaTransformY, *summator);

            //download data from device
            MatrixX xTrans, yTrans;
            cudaTransformX->download(xTrans);
            cudaTransformY->download(yTrans);

            // compute matrix A
            computeAMatrix(*cudaX, *cudaY, *cudaA);
            MatrixX A;
            cudaA->download(A);

            // solve essential matrix problem
            Eigen::JacobiSVD<MatrixX> svd(A.transpose() * A, Eigen::ComputeFullU | Eigen::ComputeFullV );
            VectorX e = svd.matrixV().col(8);
            const Eigen::Map<const Matrix3> eMatrix(e.data());

            // Enforcing the internal constraint that two singular values must be equal
            // and one must be zero.
            Eigen::JacobiSVD<Matrix3> eMatrixSVD(
                    eMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Vector3 singularValues = eMatrixSVD.singularValues();
            singularValues(0) = (singularValues(0) + singularValues(1)) / 2.0;
            singularValues(1) = singularValues(0);
            singularValues(2) = 0.0;
            // denormalize for essential matrix
            report.model = xTrans.transpose() * eMatrixSVD.matrixU() * singularValues.asDiagonal() * eMatrixSVD.matrixV().transpose() * yTrans;

            report.success = true;
            report.numOfTrials = 1;
            return report;
        }

        vector<double> computeResiduals(Matrix3 &model, vector<Point3D> &tx, vector<Point3D> &ty) override {
            CHECK_EQ(tx.size(), ty.size());
            int n = tx.size();

            cudaE->upload(model);
            cudaX->upload(PointUtil::vec2Matrix(tx));
            cudaY->upload(PointUtil::vec2Matrix(ty));

            computeResidualsCUDA(*cudaX, *cudaY, *cudaE, *cudaResidual);

            vector<float> residuals(n);
            cudaResidual->download(residuals.data(), 0, n);
            return vector<double>(residuals.begin(), residuals.end());
        }
    };
    int EssentialMatrixEstimator::kMinNumSamples = 8;

    EGRegistration::EGRegistration(const GlobalConfig &config) : config(config) {
        this->config.maxResidual = config.maxEGResidual;
    }

    EGRegistration::EGRegistration(const RANSAC2DConfig &config) : config(config) {}

    void EGRegistration::updateStartResidual(double residual) {
        config.maxResidual = residual;
    }

    RANSAC2DReport EGRegistration::filterMatches(FeatureMatches &featureMatches) {
        RANSAC2DReport report;
        report.pointNum = featureMatches.size();
        if(featureMatches.size() < config.kMinMatches) {
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

        for(int i=0; i<featureMatches.size(); i++) {
            FeatureMatch match = featureMatches.getMatch(i);

            uint32_t indx = match.getPX();
            uint32_t indy = match.getPY();

            auto px = kx[indx];
            auto py = ky[indy];

            pxs.emplace_back(cx->getCameraModel()->getRay((int)px->x, (int)px->y));
            pys.emplace_back(cy->getCameraModel()->getRay((int)py->x, (int)py->y));
        }

        EssentialMatrixEstimator estimator(featureMatches.size());
        SimpleSupportMeasurer supportMeasurer;
        typedef LORANSAC<EssentialMatrixEstimator, EssentialMatrixEstimator, RandomSampler<Point3D, Point3D>> EMRANSAC;
        EMRANSAC loransac(config, estimator, estimator, supportMeasurer);
        EMRANSAC::BASE::RANSACReport r = loransac.estimate(pxs, pys);
        if(r.success) {
            if(r.support.inlierRatio()<config.minInlierRatio) {
                r.success = false;
            }
        }

        report.success = r.success;
        report.inliers = r.support.inlierIndexes;
        report.maxResidual = r.maxResidual;

        vector<int> inliers = report.inliers;
        if(!report.success || inliers.size() < config.kMinInliers) {
            report.success = false;
            return report;
        }

        report.success = true;
        return report;
    }

    void EGRegistration::decomposeEssentialMatrix(Matrix3& E, vector<Point3D> &x, vector<Point3D> &y, Matrix3 &R, Translation& t, vector<double>& xDs, vector<double>& yDs) {
        CHECK_EQ(x.size(), y.size());
        int n = x.size();
        // solve all R and t
        Eigen::JacobiSVD<Matrix3> svd(
                E, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Matrix3 U = svd.matrixU();
        Matrix3 V = svd.matrixV().transpose();

        if (U.determinant() < 0) {
            U *= -1;
        }
        if (V.determinant() < 0) {
            V *= -1;
        }

        Matrix3 W;
        W << 0, 1, 0, -1, 0, 0, 0, 0, 1;

        Rotation R1 = U * W * V, R2 = U * W.transpose() * V;
        Translation t1 = U.col(2).normalized();
        // find best R and t from all possible projection matrix combinations.
        const std::array<Rotation, 4> RCandidates{{R1, R2, R1, R2}};
        const std::array<Translation, 4> tCandidates{{t1, t1, -t1, -t1}};

        Eigen::VectorXi supportVec(n);

        int bestSupportPointsNum = 0;
        for (int i = 0; i < RCandidates.size(); ++i) {
            Rotation RCandidate = RCandidates[i];
            Translation tCandidate = tCandidates[i];

            Eigen::Matrix<Scalar, 3, 4>  projectMatrix1, projectMatrix2=Eigen::Matrix<Scalar, 3, 4>::Identity();
            GeoUtil::Rt2P(RCandidate, tCandidate, projectMatrix1);

            vector<double> xDepths(n), yDepths(n);
            for(int j=0; j<n; ++j) {
                xDepths[j] = -1;
                yDepths[j] = -1;
            }

            const double kMinDepth = std::numeric_limits<double>::epsilon();
            const double kMaxDepth = 1000.0f * (RCandidate.transpose()*tCandidate).norm();

            // find support points
            for(int j=0; j<n; ++j) {
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


    RANSAC2DReport EGRegistration::registrationFunction(FeatureMatches& featureMatches) {
        RANSAC2DReport report;
        report.pointNum = featureMatches.size();
        if(featureMatches.size() < config.kMinMatches) {
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

        for(int i=0; i<featureMatches.size(); i++) {
            FeatureMatch match = featureMatches.getMatch(i);

            uint32_t indx = match.getPX();
            uint32_t indy = match.getPY();

            auto px = kx[indx];
            auto py = ky[indy];

            pxs.emplace_back(cx->getCameraModel()->getRay((int)px->x, (int)px->y));
            pys.emplace_back(cy->getCameraModel()->getRay((int)py->x, (int)py->y));
        }

        EssentialMatrixEstimator estimator(featureMatches.size());
        SimpleSupportMeasurer supportMeasurer;
        typedef LORANSAC<EssentialMatrixEstimator, EssentialMatrixEstimator, RandomSampler<Point3D, Point3D>> EMRANSAC;
        EMRANSAC loransac(config, estimator, estimator, supportMeasurer);
        EMRANSAC::BASE::RANSACReport r = loransac.estimate(pxs, pys);
        if(r.success) {
            if(r.support.inlierRatio()<config.minInlierRatio) {
                r.success = false;
            }else {
                decomposeEssentialMatrix(r.model, pxs, pys, R, t, report.xDs, report.yDs);
            }
        }

        report.success = r.success;
        report.inliers = r.support.inlierIndexes;
        report.maxResidual = r.maxResidual;

        vector<int> inliers = report.inliers;
        vector<double> xDs = report.xDs, yDs = report.yDs;
        if(!report.success || inliers.size() < config.kMinInliers) {
            report.success = false;
            return report;
        }
        // compute scale for some points
        int count = 0;
        double scale = 0;

        for(int k=0; k < featureMatches.size(); k++) {
            FeatureMatch match = featureMatches.getMatch(k);

            uint32_t indx = match.getPX();
            uint32_t indy = match.getPY();

            const FeatureKeypoint& px = *kx[indx];
            const FeatureKeypoint& py = *ky[indy];

            double dx = px.z;
            double dy = py.z;

            if(xDs[k] > 0 && yDs[k] > 0) {
                scale += (dx/xDs[k] + dy / yDs[k]) / 2;
                count ++;
            }

        }
        scale /= count;

        // compose parameters for ba
        vector<int> &kps1 = report.kps1, &kps2 = report.kps2;

        for(int k=0; k<inliers.size(); k++) {
            int index = inliers[k];
            FeatureMatch match = featureMatches.getMatch(index);

            uint32_t indx = match.getPX();
            uint32_t indy = match.getPY();

            const FeatureKeypoint& px = *kx[indx];
            const FeatureKeypoint& py = *ky[indy];

            kps1.emplace_back(indx);
            kps2.emplace_back(indy);
        }

        if(kps1.size() < config.kMinInliers) {
            report.success = false;
            return report;
        }

        GeoUtil::Rt2T(R, t*scale, report.T);
        report.success = true;
        return report;
    }
}
