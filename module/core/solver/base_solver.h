//
// Created by liulei on 2020/6/20.
//

#ifndef GraphFusion_BASE_SOLVER_H
#define GraphFusion_BASE_SOLVER_H

#include <vector>
#include <bits/stdint-uintn.h>
#include "../../tool/random.h"

using namespace std;

namespace rtf {
    template <class I1, class I2, class M> class Estimator {
    public:
        typedef I1 IX;
        typedef I2 IY;
        typedef M Model;

        class EstimateReport {
        public:
            vector<I1> x;
            vector<I2> y;

            bool success;
            M model;
            int numOfTrials;
            double maxResidual;

            void printReport() {
                cout << "success: " << success << endl;
                cout << "numOfTrials: " << numOfTrials << endl;
                cout << "maxResidual: " << maxResidual << endl;
                cout << "totalNum: " << x.size() << endl;
            }
        };

        virtual EstimateReport estimate(vector<I1>& x, vector<I2>& y) = 0;

        virtual vector<double> computeResiduals(M &model, vector<I1>& tx, vector<I2>& ty) = 0;
    };




    template <class I1, class I2> class Sampler {
    public:
        int numOfSamples = -1;

        explicit Sampler(int numOfSamples) {
            this->numOfSamples = numOfSamples;
        }

        virtual vector<int> sampleIndexes(int totalNum) = 0;

        void sample(vector<I1>& x, vector<I2>& y, vector<I1>& sx, vector<I2>& sy) {
            CHECK_GT(numOfSamples, 0);

            int totalNum = min(x.size(), y.size());
            vector<int> indexes = sampleIndexes(totalNum);

            sx.clear();
            sy.clear();
            for(int index: indexes) {
                sx.push_back(x[index]);
                sy.push_back(y[index]);
            }
        }
    };

    template <class I1, class I2> class RandomSampler: public Sampler<I1, I2> {
    public:
        explicit RandomSampler(int numOfSamples) : Sampler<I1, I2>(numOfSamples) {
            SetPRNGSeed();
        }

        vector<int> sampleIndexes(int totalNum) {
            vector<int> sampleIndexes(totalNum);
            for(int i=0; i<totalNum; i++) {
                sampleIndexes[i] = i;
            }

            Shuffle(static_cast<uint32_t >(this->numOfSamples), &sampleIndexes);

            std::vector<int> sampledIndexes(this->numOfSamples);
            for (size_t i = 0; i < this->numOfSamples; ++i) {
                sampledIndexes[i] = sampleIndexes[i];
            }

            return sampledIndexes;
        }
    };





    class Support {
    public:
        vector<int> inlierIndexes;
        int numOfInliers = 0;
        int totalNum = 1;
        vector<double> residuals;

        double inlierRatio() {
            return (double)numOfInliers / totalNum;
        }
    };

    class SupportMeasurer {
    public:
        virtual Support evaluate(vector<double> residuals, double maxResidual) = 0;

        virtual bool compare(Support &support1, Support &support2) = 0;
    };

    class SimpleSupportMeasurer: public SupportMeasurer {
    public:
        Support evaluate(vector<double> residuals, double maxResidual) {
            Support support;
            support.totalNum = residuals.size();
            support.numOfInliers = 0;
            support.inlierIndexes.clear();
            support.residuals = residuals;

            for(int i=0; i<residuals.size(); i++) {
                if(residuals[i] <= maxResidual) {
                    support.inlierIndexes.push_back(i);
                    support.numOfInliers ++;
                }
            }
            return support;
        }

        bool compare(Support &support1, Support &support2) {
            return support1.inlierRatio() >= support2.inlierRatio();
        }
    };


}

#endif //GraphFusion_BASE_SOLVER_H
