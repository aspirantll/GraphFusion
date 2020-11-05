//
// Created by liulei on 2020/6/5.
//

#ifndef RTF_FEATURE_POINT_H
#define RTF_FEATURE_POINT_H

#include <vector>
#include <DBoW2/BowVector.h>
#include <DBoW2/FeatureVector.h>
#include "../datastructure/frame_types.h"

using namespace std;

namespace rtf {
    class FeatureKeypoint : public Point3D{
    protected:
        int index;
    public:
        FeatureKeypoint();
        FeatureKeypoint(YAML::Node serNode);
        FeatureKeypoint(double x, double y, double z);

        int getIndex() const;

        void setIndex(int index);

        YAML::Node serialize() override;
    };

    typedef vector<shared_ptr<FeatureKeypoint>> FeatureKeypoints;

    template <class T>
    class FeatureDescriptors: public Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, public Serializable {
    public:
        typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> BASE;
        FeatureDescriptors() {}
        FeatureDescriptors(const long &x, const long &y) : BASE(x, y) {}

        YAML::Node serialize() override {
            YAML::Node node;

            for(int i=0; i<this->rows(); i++) {
                for(int j=0; j<this->cols(); j++) {
                    node.push_back((*this)(i, j));
                }
            }

            return node;
        }
    };


    template <class T>
    class FeaturePoints: public Serializable {
    protected:
        int fIndex;
        shared_ptr<Camera> camera;
        FeatureKeypoints keyPoints;
        FeatureDescriptors<T> descriptors;
        DBoW2::BowVector mBowVec;
        DBoW2::FeatureVector mFeatVec;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        FeaturePoints() {

        }

        void deserialize(YAML::Node serNode) {
            camera = CameraFactory::getCamera(serNode["camera"].as<string>());
            YAMLUtil::vectorDeserialize(serNode["keyPoints"], keyPoints);
            YAMLUtil::matrixDeserialize(serNode["descriptors"], descriptors);
        }

        int getFIndex() const {
            return fIndex;
        }

        void setFIndex(int fIndex) {

            FeaturePoints::fIndex = fIndex;
        }

        void setCamera(const shared_ptr<Camera> &camera) {
            FeaturePoints::camera = camera;
        }

        shared_ptr<Camera> getCamera() {
            return this->camera;
        }

        FeatureKeypoints& getKeyPoints() {
            return this->keyPoints;
        }

        FeatureDescriptors<T>& getDescriptors() {
            return this->descriptors;
        }

        DBoW2::BowVector &getMBowVec() {
            return this->mBowVec;
        }

        DBoW2::FeatureVector &getMFeatVec() {
            return this->mFeatVec;
        }

        YAML::Node serialize() override {
            YAML::Node node;
            node["camera"] = camera->getSerNum();
            node["keyPoints"] = YAMLUtil::vectorSerialize(keyPoints);
            node["descriptors"] = descriptors.serialize();
            //todo serialize the bow vector
            return node;
        }
    };

    enum FeatureMatchStrategy {
        SEQUENCE,
        FULL
    };

    class FeatureMatch {
    protected:
        uint32_t px;
        uint32_t py;
    public:
        FeatureMatch() {}

        FeatureMatch(uint32_t px, uint32_t py) {
            this->px = px;
            this->py = py;
        }

        uint32_t getPX() {
            return px;
        }

        uint32_t getPY() {
            return py;
        }
    };

    class FeatureMatches {
    protected:
        FeaturePoints<uint8_t>* fp1;
        FeaturePoints<uint8_t>* fp2;
        vector<FeatureMatch> matches;
    public:
        FeatureMatches();

        FeatureMatches(FeaturePoints<uint8_t> &fp1, FeaturePoints<uint8_t> &fp2,
                       const vector<FeatureMatch> &matches);

        int getFIndexX() const;

        int getFIndexY() const;

        shared_ptr<Camera> getCx();

        shared_ptr<Camera> getCy();

        FeatureKeypoints &getKx();

        FeatureKeypoints &getKy();

        void setMatches(vector<FeatureMatch>& matches);

        void setFp1(FeaturePoints<uint8_t> &fp1);

        void setFp2(FeaturePoints<uint8_t> &fp2);

        FeaturePoints<uint8_t> &getFp1();

        FeaturePoints<uint8_t> &getFp2();

        void addMatch(FeatureMatch match);

        FeatureMatch getMatch(int index) const;

        vector<FeatureMatch> &getMatches();

        int size() const;
    };

    class SIFTFeatureKeypoint: public FeatureKeypoint {
    public:
        SIFTFeatureKeypoint();
        SIFTFeatureKeypoint(const float x, const float y, const float z);
        SIFTFeatureKeypoint(const float x, const float y, const float z, const float scale,
                            const float orientation);
        SIFTFeatureKeypoint(const float x, const float y, const float z, const float a11,
                            const float a12, const float a21, const float a22);

        static SIFTFeatureKeypoint FromParameters(const float x, const float y, const float depth,
                                                  const float scale_x,
                                                  const float scale_y,
                                                  const float orientation,
                                                  const float shear);

        // Rescale the feature location and shape size by the given scale factor.
        void Rescale(const float scale);
        void Rescale(const float scale_x, const float scale_y);

        // Compute similarity shape parameters from affine shape.
        float ComputeScale() const;
        float ComputeScaleX() const;
        float ComputeScaleY() const;
        float ComputeOrientation() const;
        float ComputeShear() const;

        // Affine shape of the feature.
        float a11;
        float a12;
        float a21;
        float a22;
    };

    typedef std::vector<SIFTFeatureKeypoint> SIFTFeatureKeypoints;
    typedef FeatureDescriptors<uint8_t>     SIFTFeatureDescriptors;
    typedef FeaturePoints<uint8_t> SIFTFeaturePoints;


    class ORBFeatureKeypoint: public FeatureKeypoint {
    public:
        float angle; //!< computed orientation of the keypoint (-1 if not applicable);
        float response; //!< the response by which the most strong keypoints have been selected. Can be used for the further sorting or subsampling
        float size;
        int octave; //!< octave (pyramid layer) from which the keypoint has been extracted
        int class_id; //!< object class (if the keypoints need to be clustered by an object they belong to)

        ORBFeatureKeypoint();

        ORBFeatureKeypoint(cv::KeyPoint& kp);
    };

    typedef std::vector<ORBFeatureKeypoint> ORBFeatureKeypoints;
    typedef FeatureDescriptors<uint8_t> ORBFeatureDescriptors;
    typedef FeaturePoints<uint8_t> ORBFeaturePoints;


    void featureIndexesToPoints(const FeatureKeypoints& features, const vector<int>& featureIndexes, vector<FeatureKeypoint>& points);

    void downFeatureToSift(const FeatureKeypoints& src, SIFTFeatureKeypoints& target);


    class DenseFeatureKeypoint: public FeatureKeypoint {
    public:
        float nx;
        float ny;
        float nz;
    };

    class DenseFeatureMatches{
    protected:
        shared_ptr<Camera> cx;
        shared_ptr<Camera> cy;

        vector<FeatureKeypoint> kx;
        vector<FeatureKeypoint> ky;
    public:
        DenseFeatureMatches();

        DenseFeatureMatches(const shared_ptr<Camera> &cx, const shared_ptr<Camera> &cy);

        DenseFeatureMatches(const shared_ptr<Camera> &cx, const shared_ptr<Camera> &cy, vector<FeatureKeypoint> &kx,
                            vector<FeatureKeypoint> &ky);

        shared_ptr<Camera> &getCx();

        shared_ptr<Camera> &getCy();

        vector<FeatureKeypoint> &getKx();

        vector<FeatureKeypoint> &getKy();

        int size() const;
    };
}

#endif //RTF_FEATURE_POINT_H
