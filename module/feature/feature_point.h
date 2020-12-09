//
// Created by liulei on 2020/6/5.
//

#ifndef GraphFusion_FEATURE_POINT_H
#define GraphFusion_FEATURE_POINT_H

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


#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64
    class FeaturePoints: public Serializable {
    protected:
        int fIndex;
        shared_ptr<Camera> camera;
        FeatureKeypoints keyPoints;
        FeatureDescriptors<uint8_t> descriptors;
        DBoW2::BowVector mBowVec;
        DBoW2::FeatureVector mFeatVec;

        float minX;
        float maxX;
        float minY;
        float maxY;
        float gridElementWidthInv;
        float gridElementHeightInv;
        vector<int> grid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        FeaturePoints();

        void deserialize(YAML::Node serNode);

        int getFIndex() const;

        void setFIndex(int fIndex);

        void setCamera(const shared_ptr<Camera> &camera);

        shared_ptr<Camera> getCamera();

        shared_ptr<FeatureKeypoint> getKeypoint(int index);

        void setBounds(float minX, float maxX, float minY, float maxY);

        bool posInGrid(const shared_ptr<FeatureKeypoint> &kp, int &posX, int &posY);

        void assignFeaturesToGrid();

        void fuseFeaturePoints(FeatureKeypoints& fps, vector<Eigen::Matrix<uint8_t, 1, -1, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::Matrix<uint8_t, 1, -1, Eigen::RowMajor>>>& descs);

        vector<int> getFeaturesInArea(const float &x, const float  &y, const float  &r);

        FeatureKeypoints& getKeyPoints();

        FeatureDescriptors<uint8_t>& getDescriptors();

        DBoW2::BowVector &getMBowVec();

        DBoW2::FeatureVector &getMFeatVec();

        bool empty();

        int size();

        float getMinX() const;

        float getMaxX() const;

        float getMinY() const;

        float getMaxY() const;

        YAML::Node serialize() override;
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
        FeaturePoints* fp1;
        FeaturePoints* fp2;
        vector<FeatureMatch> matches;
    public:
        FeatureMatches();

        FeatureMatches(FeaturePoints &fp1, FeaturePoints &fp2,
                       const vector<FeatureMatch> &matches);

        int getFIndexX() const;

        int getFIndexY() const;

        shared_ptr<Camera> getCx();

        shared_ptr<Camera> getCy();

        FeatureKeypoints &getKx();

        FeatureKeypoints &getKy();

        void setMatches(vector<FeatureMatch>& matches);

        void setFp1(FeaturePoints &fp1);

        void setFp2(FeaturePoints &fp2);

        FeaturePoints &getFp1();

        FeaturePoints &getFp2();

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
    typedef FeaturePoints SIFTFeaturePoints;

    void featureIndexesToPoints(const FeatureKeypoints& features, const vector<int>& featureIndexes, vector<FeatureKeypoint>& points);

    void featureMatchesToPoints(FeatureMatches& featureMatches, vector<FeatureKeypoint>& kxs, vector<FeatureKeypoint>& kys);

    void featureMatchesToPoints(FeatureMatches& featureMatches, vector<int> inliers, vector<FeatureKeypoint>& kxs, vector<FeatureKeypoint>& kys);

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

#endif //GraphFusion_FEATURE_POINT_H
