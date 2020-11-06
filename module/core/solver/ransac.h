/**
 created by liulei on 2020/06/20
*/


#include "base_solver.h"
#include <glog/logging.h>

namespace rtf {

    class RANSACConfig {
    public:
        // max iterations
        int numOfTrials = 250;
        // start residuals
        double maxResidual = 0.0001;
        // residual upper limit
        double upperBoundResidual = 0.001;
        // don't abort probability
        double confidence = 0.99;
        // multiplier for dynamic iterations
        double lambda = 3;

        // down step factor
        double stepFactor = 0.5;
        // inliers ratio threshold
        double irThreshold = 0.8;
        // adaptive ending delta
        double aeDelta = 0.00001;
        // allow difference for inlier ratio
        double irDelta = 0.02;
        // min inlier ratio
        double minInlierRatio = 0.3;
        // fast stop inlier ratio threshold
        double minStopInlinerRatio = 0.3;
    };

    template <class Estimator, class Sampler, class SupportMeasurer=SimpleSupportMeasurer>
    class RANSAC {
    public:
        typedef typename Estimator::M Model;
        typedef typename Estimator::I1 I1;
        typedef typename Estimator::I2 I2;

        class RANSACReport: public Estimator::EstimateReport {
        public:
            typedef typename Estimator::EstimateReport BASE;
            Support support;

            void printReport() {
                cout << "-------------------------------------------------------------------------" << endl;
                BASE::printReport();
                cout << "numOfInliers: " << support.numOfInliers << endl;
                cout << "-------------------------------------------------------------------------" << endl;
            }
        };
    protected:
        RANSACConfig config;
        Estimator estimator;
        Sampler sampler;
        SupportMeasurer supportMeasurer;

        virtual void updateReport(RANSACReport &report, Support &curSupport, Model model) {
            report.support = curSupport;
            report.model = model;
        }


        int computeNumTrials(Support &support) {
            const double nom = 1 - config.confidence;
            if (nom <= 0) {
                return numeric_limits<int>::max();
            }

            const double denom = 1 - std::pow(support.inlierRatio(), sampler.numOfSamples);
            if (denom <= 0) {
                return 1;
            }

            return static_cast<size_t>(
                    std::ceil(std::log(nom) / std::log(denom) * config.lambda));
        }

    public:
        RANSAC(const RANSACConfig config, Estimator &estimator, SupportMeasurer &supportMeasurer)
                : config(config), estimator(estimator), sampler(Sampler(Estimator::kMinNumSamples)), supportMeasurer(supportMeasurer) {
            CHECK_LE(config.confidence, 1);
            CHECK_GE(config.maxResidual, 0);
            CHECK_GT(config.lambda, 0);
            CHECK_GT(config.numOfTrials, 0);
        }

        RANSACConfig& getConfig() {
            return config;
        }

        virtual RANSACReport estimate(vector<I1>& x, vector<I2>& y) {
            CHECK_EQ(x.size(), y.size());
            int totalNum = x.size();

            RANSACReport report;
            report.success = false;
            report.x = x;
            report.y = y;
            report.numOfTrials = 0;
            report.maxResidual = config.maxResidual;
            report.support.numOfInliers = 0;
            report.support.totalNum = totalNum;

            if(totalNum < Estimator::kMinNumSamples) {
                return report;
            }

            int dynamicNumOfTrial = config.numOfTrials;
            for(int i=0; i<dynamicNumOfTrial; i++) {
                report.numOfTrials++;

                vector<typename Estimator::I1> sx;
                vector<typename Estimator::I2> sy;

                sampler.sample(x, y, sx, sy);

                typename Estimator::EstimateReport estimateReport = estimator.estimate(sx, sy);
                vector<double> residuals = estimator.computeResiduals(estimateReport.model, x, y);
                Support support = supportMeasurer.evaluate(residuals, config.maxResidual);
                if(supportMeasurer.compare(support, report.support)) {
                    this->updateReport(report, support, estimateReport.model);
                    dynamicNumOfTrial = min(config.numOfTrials, this->computeNumTrials(support));
                }
            }

            report.success = true;

            return report;
        }
    };

    template <class Estimator, class LocalEstimator, class Sampler, class SupportMeasurer=SimpleSupportMeasurer>
    class LORANSAC: public RANSAC<Estimator, Sampler, SupportMeasurer> {
    protected:
        LocalEstimator localEstimator;
    public:
        typedef RANSAC<Estimator, Sampler, SupportMeasurer> BASE;

        LORANSAC(const RANSACConfig config, Estimator &estimator, LocalEstimator localEstimator
                , SupportMeasurer &supportMeasurer) : BASE(config, estimator, supportMeasurer),
                                                  localEstimator(localEstimator) {}

        void updateReport(typename BASE::RANSACReport &report, Support &curSupport, typename BASE::Model model) {
            // Estimate locally optimized model from inliers.
            if (curSupport.numOfInliers > Estimator::kMinNumSamples &&
                curSupport.numOfInliers >= LocalEstimator::kMinNumSamples) {

                // extract the inliers
                vector<typename Estimator::I1> xInliers;
                vector<typename Estimator::I2> yInliers;
                for (int index: curSupport.inlierIndexes) {
                    xInliers.push_back(report.x[index]);
                    yInliers.push_back(report.y[index]);
                }

                // estimate
                typename Estimator::EstimateReport localReport = localEstimator.estimate(xInliers, yInliers);
                vector<double> residuals = localEstimator.computeResiduals(localReport.model, report.x, report.y);

                Support localSupport = this->supportMeasurer.evaluate(residuals, this->config.maxResidual);

                if(this->supportMeasurer.compare(localSupport, report.support)) {
                    curSupport = localSupport;
                    model = localReport.model;
                }
            }

            BASE::updateReport(report, curSupport, model);
        }

    };

    template <class Estimator, class LocalEstimator, class Sampler, class SupportMeasurer=SimpleSupportMeasurer>
    class AdaptiveLORANSAC: public LORANSAC<Estimator, LocalEstimator, Sampler, SupportMeasurer> {
    public:
        typedef LORANSAC<Estimator, LocalEstimator, Sampler, SupportMeasurer> PARENT;
        typedef typename PARENT::BASE BASE;

        typedef typename BASE::Model model;
        typedef typename BASE::I1 I1;
        typedef typename BASE::I2 I2;

        typedef typename BASE::RANSACReport RANSACReport;


        AdaptiveLORANSAC(const RANSACConfig config, Estimator &estimator, LocalEstimator localEstimator
                , SupportMeasurer &supportMeasurer) : PARENT(config, estimator, localEstimator, supportMeasurer){
            CHECK_GE(config.irThreshold, 0);
            CHECK_LT(config.stepFactor, 1);
            CHECK_GT(config.stepFactor, 0);
            CHECK_GT(config.aeDelta, 0);
            CHECK_GT(config.irDelta, 0);
            CHECK_LE(config.irDelta+config.irThreshold, 1);
        }


        RANSACReport estimate(vector<I1>& x, vector<I2>& y) {
            RANSACReport report = PARENT::estimate(x,y);
            if(fabs(report.support.inlierRatio()-this->config.irThreshold)<this->config.irDelta) {
                return report;
            }
            bool dec = report.support.inlierRatio() > this->config.irThreshold;
            double stepFactor = dec? this->config.stepFactor: 1/this->config.stepFactor;

            RANSACReport high, low, cur=report, last;
            // find bound points
            do{
                last = cur;
                this->config.maxResidual *= stepFactor;
                cur = PARENT::estimate(x,y);
                if(fabs(cur.support.inlierRatio()-this->config.irThreshold)<this->config.irDelta) {
                    return cur;
                }
                if(this->config.maxResidual>=this->config.upperBoundResidual/10&&cur.support.inlierRatio()<this->config.minStopInlinerRatio) {
                    cur.success = false;
                    return cur;
                }
            }while (fabs(last.maxResidual-cur.maxResidual)>this->config.aeDelta&&(cur.support.inlierRatio()-this->config.irThreshold)*(last.support.inlierRatio()-this->config.irThreshold)>0);

            //binary search
            high = cur.support.inlierRatio()>this->config.irThreshold? cur:last;
            low = cur.support.inlierRatio()<this->config.irThreshold? cur:last;
            while(fabs(high.maxResidual-low.maxResidual)>this->config.aeDelta) {
                this->config.maxResidual = (high.maxResidual+low.maxResidual)/2;
                cur = PARENT::estimate(x,y);
                if(fabs(cur.support.inlierRatio()-this->config.irThreshold)<this->config.irDelta) {
                    return cur;
                }
                if(cur.support.inlierRatio()>this->config.irThreshold) {
                    high = cur;
                } else {
                    low = cur;
                }
            }

            if(high.maxResidual>=this->config.upperBoundResidual) {
                if(low.maxResidual<this->config.upperBoundResidual) {
                    this->config.maxResidual = low.maxResidual;
                    return low;
                } else {
                    this->config.maxResidual = this->config.upperBoundResidual;
                    high.success = false;
                }
            } else {
                this->config.maxResidual = high.maxResidual;
            }
            return high;
        }


    };

}