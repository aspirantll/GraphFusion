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
    template <class T>
    class FeaturePoints: public Serializable {
    protected:
        int fIndex;
        shared_ptr<Camera> camera;
        FeatureKeypoints keyPoints;
        FeatureDescriptors<T> descriptors;
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
            setBounds(camera->getMinX(), camera->getMaxX(), camera->getMinY(), camera->getMaxY());
        }

        shared_ptr<Camera> getCamera() {
            return this->camera;
        }

        shared_ptr<FeatureKeypoint> getKeypoint(int index) {
            CHECK_LT(index, this->keyPoints.size());
            return this->keyPoints[index];
        }

        void setBounds(float minX, float maxX, float minY, float maxY) {
            this->minX = minX;
            this->maxX = maxX;
            this->minY = minY;
            this->maxY = maxY;
            gridElementWidthInv= static_cast<float>(FRAME_GRID_COLS) / (maxX-minX);
            gridElementHeightInv= static_cast<float>(FRAME_GRID_ROWS) / (maxY-minY);
        }

        bool posInGrid(const shared_ptr<FeatureKeypoint> &kp, int &posX, int &posY) {
            posX = round((kp->x-camera->getMinX()) * gridElementWidthInv);
            posY = round((kp->y-camera->getMinY()) * gridElementHeightInv);

            //Keypoint's coordinates are undistorted, which could cause to go out of the image
            if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
                return false;

            return true;
        }

        void assignFeaturesToGrid() {
            int nReserve = size()/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
            for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
                for (unsigned int j=0; j<FRAME_GRID_ROWS;j++) {
                    grid[i][j].clear();
                    grid[i][j].reserve(nReserve);
                }

            for(int i=0;i<size();i++) {
                const shared_ptr<FeatureKeypoint> kp = getKeypoint(i);

                int nGridPosX, nGridPosY;
                if(posInGrid(kp,nGridPosX,nGridPosY))
                    grid[nGridPosX][nGridPosY].push_back(i);
            }
        }

        vector<int> getFeaturesInArea(const float &x, const float  &y, const float  &r) {
            vector<int> vIndices;
            vIndices.reserve(size());

            const int nMinCellX = max(0,(int)floor((x-camera->getMinX()-r)*gridElementWidthInv));
            if(nMinCellX>=FRAME_GRID_COLS)
                return vIndices;

            const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-camera->getMinX()+r)*gridElementWidthInv));
            if(nMaxCellX<0)
                return vIndices;

            const int nMinCellY = max(0,(int)floor((y-camera->getMinY()-r)*gridElementHeightInv));
            if(nMinCellY>=FRAME_GRID_ROWS)
                return vIndices;

            const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-camera->getMinY()+r)*gridElementHeightInv));
            if(nMaxCellY<0)
                return vIndices;

            for(int ix = nMinCellX; ix<=nMaxCellX; ix++) {
                for(int iy = nMinCellY; iy<=nMaxCellY; iy++) {
                    const vector<int> vCell = grid[ix][iy];
                    if(vCell.empty())
                        continue;

                    for(int j=0, jend=vCell.size(); j<jend; j++) {
                        const shared_ptr<FeatureKeypoint> kp = getKeypoint(vCell[j]);

                        const float distx = kp->x-x;
                        const float disty = kp->y-y;

                        if(fabs(distx)<r && fabs(disty)<r)
                            vIndices.push_back(vCell[j]);
                    }
                }
            }

            return vIndices;
        }

        void fuseFeaturePoints(FeatureKeypoints& fps, vector<Eigen::Matrix<T, 1, -1, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::Matrix<T, 1, -1, Eigen::RowMajor>>>& descs) {
            int nReserve = size()/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
            for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
                for (unsigned int j=0; j<FRAME_GRID_ROWS;j++) {
                    grid[i][j].clear();
                    grid[i][j].reserve(nReserve);
                }

            int num = fps.size();
            for (int i = 0; i < num; i++) {
                shared_ptr<FeatureKeypoint> fp = fps[i];
                Eigen::Matrix<T, 1, -1, Eigen::RowMajor> desc = descs[i];

                int index = keyPoints.size();
                fp->setIndex(index);
                keyPoints.emplace_back(fp);
                descriptors.conservativeResize(descriptors.rows() + 1, 128);
                descriptors.row(index) = desc;

                int nGridPosX, nGridPosY;
                if (posInGrid(fp, nGridPosX, nGridPosY))
                    grid[nGridPosX][nGridPosY].push_back(index);

            }
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

        bool empty() {
            return this->keyPoints.empty();
        }

        int size() {
            return this->keyPoints.size();
        }

        float getMinX() const {
            return minX;
        }

        float getMaxX() const {
            return maxX;
        }

        float getMinY() const {
            return minY;
        }

        float getMaxY() const {
            return maxY;
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
