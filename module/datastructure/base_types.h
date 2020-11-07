//
// Created by liulei on 2020/5/18.
//

#ifndef GraphFusion_BASE_TYPES_H
#define GraphFusion_BASE_TYPES_H

#include <string>
#include <mutex>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <glog/logging.h>
#include <Eigen/Core>
#include <Eigen/StdVector>

#define EigenVector(type) vector<type, Eigen::aligned_allocator<type>>
using namespace std;

typedef float Scalar;

typedef Eigen::Matrix<Scalar, -1, -1> MatrixX;
typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
typedef Eigen::Matrix<Scalar, -1, 3> MatrixX3;
typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;

typedef Eigen::Matrix<Scalar, -1, 1> VectorX;
typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
typedef Eigen::Matrix<Scalar, 4, 1> Vector4;
typedef Eigen::Matrix<Scalar, 6, 1> Vector6;

typedef Matrix3 Intrinsic;
typedef Matrix4 Transform;
typedef Matrix3 Rotation;
typedef Vector3 Translation;
typedef Vector6 SEVector;

typedef EigenVector(Transform) TransformVector;

namespace rtf {
    class Serializable {

        /**
         * serialize obj to string
         * @param baseDir the dir for saving internal files
         * @return
         */
        virtual YAML::Node serialize() = 0;

    };


    template<class T> class EigenUpperTriangularMatrix {
    public:
        int n;
        EigenVector(T) matVec;
        T defaultValue;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EigenUpperTriangularMatrix() {}

        EigenUpperTriangularMatrix(vector<T>& data, int n, T defaultValue): matVec(data), n(n), defaultValue(defaultValue) {
            CHECK_EQ(data.size(), n*(n-1)/2);
        }

        EigenUpperTriangularMatrix(int n, T defaultValue): n(n), defaultValue(defaultValue) {
            matVec.resize(n*(n-1)/2, defaultValue);
        }

        void resize(int n,  T defaultValue) {
            this->n = n;
            this->defaultValue = defaultValue;
            matVec.resize(n*(n-1)/2, defaultValue);
        }

        int getN() {
            return n;
        }

        void extend() {
            for(int i=0; i<n; i++) {
                matVec.emplace_back(defaultValue);
            }
            n++;
        }

        int computeIndex(int i, int j) {
            CHECK_GE(i, 0);
            CHECK_LT(i, n);

            CHECK_GE(j, 0);
            CHECK_LT(j, n);

            CHECK_NE(i, j);

            if(i>j) return computeIndex(j,i);

            return j*(j-1)/2+i;
        }

        T &operator() (int i, int j) {
            if(i==j) {
                return defaultValue;
            }
            return matVec[computeIndex(i, j)];
        }
    };

    template <class T> class UpperTriangularMatrix {
    protected:
        int n;
        vector<T> matVec;
        T defaultValue;
    public:
        UpperTriangularMatrix() {}

        UpperTriangularMatrix(vector<T>& data, int n, T defaultValue): matVec(data), n(n), defaultValue(defaultValue) {
            CHECK_EQ(data.size(), n*(n-1)/2);
        }

        UpperTriangularMatrix(int n, T defaultValue): n(n), defaultValue(defaultValue) {
            for(int i=0; i<n*(n-1)/2; i++) {
                matVec.emplace_back(defaultValue);
            }
        }

        int getN() {
            return n;
        }

        int computeIndex(int i, int j) {
            CHECK_GE(i, 0);
            CHECK_LT(i, n);

            CHECK_GE(j, 0);
            CHECK_LT(j, n);

            CHECK_NE(i, j);

            if(i>j) return computeIndex(j,i);

            return j*(j-1)/2+i;
        }

        T &operator() (int i, int j) {
            if(i==j) {
                return defaultValue;
            }
            return matVec[computeIndex(i, j)];
        }
    };


    class Runnable {
    protected:
        virtual void reset() = 0;
        void resetIfRequested();
        bool mbResetRequested;
        std::mutex mMutexReset;

        bool checkFinish();
        void setFinish();
        void setAcceptRequest(bool flag);

        virtual bool checkNewRequest() = 0;

        bool mbFinishRequested;
        bool mbFinished;
        std::mutex mMutexFinish;

        bool mbStopped;
        bool mbStopRequested;
        bool mbNotStop;
        std::mutex mMutexStop;

        bool mbAcceptRequest;
        std::mutex mMutexAccept;
    public:
        Runnable();
        // Thread Synch
        void requestStop();
        void requestReset();
        bool stop();
        bool isStopped();
        bool stopRequested();
        bool setNotStop(bool flag);
        bool acceptRequest();

        void requestFinish();
        bool isFinished();

        // main function
        void run();
        virtual void doTask() = 0;
    };
}
#endif //GraphFusion_BASE_TYPES_H
