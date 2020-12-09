//
// Created by liulei on 2020/6/5.
//

#include "feature_point.h"

#include <utility>

namespace rtf {
    FeatureMatches::FeatureMatches() {}

    FeatureMatches::FeatureMatches(FeaturePoints &fp1, FeaturePoints &fp2,
                                   const vector<FeatureMatch> &matches) : fp1(&fp1), fp2(&fp2), matches(matches) {}

    shared_ptr<Camera> FeatureMatches::getCx(){
        return fp1->getCamera();
    }

    shared_ptr<Camera> FeatureMatches::getCy(){
        return fp2->getCamera();
    }

    FeatureKeypoints &FeatureMatches::getKx() {
        return fp1->getKeyPoints();
    }

    FeatureKeypoints &FeatureMatches::getKy() {
        return fp2->getKeyPoints();
    }

    void FeatureMatches::setMatches(vector<FeatureMatch>& matches) {
        FeatureMatches::matches = matches;
    }

    void FeatureMatches::addMatch(FeatureMatch match) {
        matches.push_back(match);
    }

    FeatureMatch FeatureMatches::getMatch(int index) const {
        return matches[index];
    }

    vector<FeatureMatch> &FeatureMatches::getMatches() {
        return this->matches;
    }

    int FeatureMatches::size() const {
        return this->matches.size();
    }

    int FeatureMatches::getFIndexX() const {
        return fp1->getFIndex();
    }

    int FeatureMatches::getFIndexY() const {
        return fp2->getFIndex();
    }

    void FeatureMatches::setFp1(FeaturePoints &fp1) {
        FeatureMatches::fp1 = &fp1;
    }

    void FeatureMatches::setFp2(FeaturePoints &fp2) {
        FeatureMatches::fp2 = &fp2;
    }

    FeaturePoints &FeatureMatches::getFp1() {
        return *fp1;
    }

    FeaturePoints &FeatureMatches::getFp2() {
        return *fp2;
    }


    /** FeatureKeypoint */
    FeatureKeypoint::FeatureKeypoint()
            : FeatureKeypoint(0, 0, 0) {}

    FeatureKeypoint::FeatureKeypoint(YAML::Node serNode): Point3D(serNode) {
    }
    FeatureKeypoint::FeatureKeypoint(double x, double y, double z)
            : Point3D(x,y,z) {}

    YAML::Node FeatureKeypoint::serialize() {
        YAML::Node node = Point3D::serialize();
        return node;
    }

    int FeatureKeypoint::getIndex() const {
        return index;
    }

    void FeatureKeypoint::setIndex(int index) {
        FeatureKeypoint::index = index;
    }

    FeaturePoints::FeaturePoints() {

    }

    void FeaturePoints::deserialize(YAML::Node serNode) {
        camera = CameraFactory::getCamera(serNode["camera"].as<string>());
        YAMLUtil::vectorDeserialize(serNode["keyPoints"], keyPoints);
        YAMLUtil::matrixDeserialize(serNode["descriptors"], descriptors);
    }

    int FeaturePoints::getFIndex() const {
        return fIndex;
    }

    void FeaturePoints::setFIndex(int fIndex) {

        FeaturePoints::fIndex = fIndex;
    }

    void FeaturePoints::setCamera(const shared_ptr<Camera> &camera) {
        FeaturePoints::camera = camera;
        setBounds(camera->getMinX(), camera->getMaxX(), camera->getMinY(), camera->getMaxY());
    }

    shared_ptr<Camera> FeaturePoints::getCamera() {
        return this->camera;
    }

    shared_ptr<FeatureKeypoint> FeaturePoints::getKeypoint(int index) {
        CHECK_LT(index, this->keyPoints.size());
        return this->keyPoints[index];
    }

    void FeaturePoints::setBounds(float minX, float maxX, float minY, float maxY) {
        this->minX = minX;
        this->maxX = maxX;
        this->minY = minY;
        this->maxY = maxY;
        gridElementWidthInv= static_cast<float>(FRAME_GRID_COLS) / (maxX-minX);
        gridElementHeightInv= static_cast<float>(FRAME_GRID_ROWS) / (maxY-minY);
    }

    bool FeaturePoints::posInGrid(const shared_ptr<FeatureKeypoint> &kp, int &posX, int &posY) {
        posX = round((kp->x-minX) * gridElementWidthInv);
        posY = round((kp->y-minY) * gridElementHeightInv);

        //Keypoint's coordinates are undistorted, which could cause to go out of the image
        if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
            return false;

        return true;
    }

    void FeaturePoints::assignFeaturesToGrid() {
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

    void FeaturePoints::fuseFeaturePoints(FeatureKeypoints& fps, vector<Eigen::Matrix<uint8_t, 1, -1, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::Matrix<uint8_t, 1, -1, Eigen::RowMajor>>>& descs) {
        int num = fps.size();
        for(int i=0; i<num; i++) {
            shared_ptr<FeatureKeypoint> fp = fps[i];
            Eigen::Matrix<uint8_t, 1, -1, Eigen::RowMajor> desc = descs[i];

            bool newFp = true;
            if(num>2000) {
                vector<int> indices = getFeaturesInArea(fp->x, fp->y, 3);
                float bestDist = 0;

                Eigen::Matrix<int, 1, -1, Eigen::RowMajor> d2 = desc.cast<int>();
                for(int ind: indices)  {
                    Eigen::Matrix<int, 1, -1, Eigen::RowMajor> d1 = getDescriptors().row(ind).cast<int>();

                    const float dist = d1.dot(d2); // dot product for distance

                    if(dist>bestDist) {
                        bestDist=dist;
                    }
                }
                float dist =  acos(min(bestDist * 0.000003814697265625f, 1.0f));
                newFp = dist>0.4;
            }

            if(newFp) {
                int index = keyPoints.size();
                fp->setIndex(index);
                keyPoints.emplace_back(fp);
                descriptors.conservativeResize(descriptors.rows()+1, 128);
                descriptors.row(index) = desc;

                int nGridPosX, nGridPosY;
                if(posInGrid(fp,nGridPosX,nGridPosY))
                    grid[nGridPosX][nGridPosY].push_back(index);
            }else {
                keyPoints[bestInd] = fp;
                descriptors.row(bestInd) = desc;
            }

        }
    }

    vector<int> FeaturePoints::getFeaturesInArea(const float &x, const float  &y, const float  &r) {
        vector<int> vIndices;
        vIndices.reserve(size());

        const int nMinCellX = max(0,(int)floor((x-minX-r)*gridElementWidthInv));
        if(nMinCellX>=FRAME_GRID_COLS)
            return vIndices;

        const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-minX+r)*gridElementWidthInv));
        if(nMaxCellX<0)
            return vIndices;

        const int nMinCellY = max(0,(int)floor((y-minY-r)*gridElementHeightInv));
        if(nMinCellY>=FRAME_GRID_ROWS)
            return vIndices;

        const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-minY+r)*gridElementHeightInv));
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

    FeatureKeypoints& FeaturePoints::getKeyPoints() {
        return this->keyPoints;
    }

    FeatureDescriptors<uint8_t>& FeaturePoints::getDescriptors() {
        return this->descriptors;
    }

    DBoW2::BowVector &FeaturePoints::getMBowVec() {
        return this->mBowVec;
    }

    DBoW2::FeatureVector &FeaturePoints::getMFeatVec() {
        return this->mFeatVec;
    }

    bool FeaturePoints::empty() {
        return this->keyPoints.empty();
    }

    int FeaturePoints::size() {
        return this->keyPoints.size();
    }

    float FeaturePoints::getMinX() const {
        return minX;
    }

    float FeaturePoints::getMaxX() const {
        return maxX;
    }

    float FeaturePoints::getMinY() const {
        return minY;
    }

    float FeaturePoints::getMaxY() const {
        return maxY;
    }

    YAML::Node FeaturePoints::serialize() {
        YAML::Node node;
        node["camera"] = camera->getSerNum();
        node["keyPoints"] = YAMLUtil::vectorSerialize(keyPoints);
        node["descriptors"] = descriptors.serialize();
        //todo serialize the bow vector
        return node;
    }

    /** SIFTFeatureKeypoint */

    SIFTFeatureKeypoint::SIFTFeatureKeypoint()
            : FeatureKeypoint() {}

    SIFTFeatureKeypoint::SIFTFeatureKeypoint(const float x, const float y, const float z)
            : FeatureKeypoint(x, y, z) {}

    SIFTFeatureKeypoint::SIFTFeatureKeypoint(const float x, const float y, const float z,
                                             const float scale, const float orientation)
            : FeatureKeypoint(x, y, z) {
        CHECK_GE(scale, 0.0);
        const float scale_cos_orientation = scale * std::cos(orientation);
        const float scale_sin_orientation = scale * std::sin(orientation);
        a11 = scale_cos_orientation;
        a12 = -scale_sin_orientation;
        a21 = scale_sin_orientation;
        a22 = scale_cos_orientation;
    }

    SIFTFeatureKeypoint::SIFTFeatureKeypoint(const float x, const float y, const float z,
                                             const float a11_, const float a12_,
                                             const float a21_, const float a22_)
            : FeatureKeypoint(x, y, z), a11(a11_), a12(a12_), a21(a21_), a22(a22_) {}

    SIFTFeatureKeypoint SIFTFeatureKeypoint::FromParameters(const float x, const float y, const float z,
                                                            const float scale_x,
                                                            const float scale_y,
                                                            const float orientation,
                                                            const float shear) {
        return SIFTFeatureKeypoint(x, y, z, scale_x * std::cos(orientation),
                               -scale_y * std::sin(orientation + shear),
                               scale_x * std::sin(orientation),
                               scale_y * std::cos(orientation + shear));
    }

    void SIFTFeatureKeypoint::Rescale(const float scale) {
        Rescale(scale, scale);
    }

    void SIFTFeatureKeypoint::Rescale(const float scale_x, const float scale_y) {
        CHECK_GT(scale_x, 0);
        CHECK_GT(scale_y, 0);
        x *= scale_x;
        y *= scale_y;
        a11 *= scale_x;
        a12 *= scale_y;
        a21 *= scale_x;
        a22 *= scale_y;
    }

    float SIFTFeatureKeypoint::ComputeScale() const {
        return (ComputeScaleX() + ComputeScaleY()) / 2.0f;
    }

    float SIFTFeatureKeypoint::ComputeScaleX() const {
        return std::sqrt(a11 * a11 + a21 * a21);
    }

    float SIFTFeatureKeypoint::ComputeScaleY() const {
        return std::sqrt(a12 * a12 + a22 * a22);
    }

    float SIFTFeatureKeypoint::ComputeOrientation() const {
        return std::atan2(a21, a11);
    }

    float SIFTFeatureKeypoint::ComputeShear() const {
        return std::atan2(-a12, a22) - ComputeOrientation();
    }

    void featureIndexesToPoints(const FeatureKeypoints& features, const vector<int>& featureIndexes, vector<FeatureKeypoint>& points) {
        points.clear();
        for(auto ind: featureIndexes) {
            points.emplace_back(*features[ind]);
        }
    }

    void featureMatchesToPoints(FeatureMatches& featureMatches, vector<FeatureKeypoint>& kxs, vector<FeatureKeypoint>& kys) {
        kxs.clear();
        kys.clear();

        for(int i=0; i<featureMatches.size(); i++) {
            FeatureMatch match = featureMatches.getMatch(i);
            kxs.emplace_back(*featureMatches.getKx()[match.getPX()]);
            kys.emplace_back(*featureMatches.getKy()[match.getPY()]);
        }
    }

    void featureMatchesToPoints(FeatureMatches& featureMatches, vector<int> inliers, vector<FeatureKeypoint>& kxs, vector<FeatureKeypoint>& kys) {
        kxs.clear();
        kys.clear();

        for(int i=0; i<inliers.size(); i++) {
            FeatureMatch match = featureMatches.getMatch(inliers[i]);
            kxs.emplace_back(*featureMatches.getKx()[match.getPX()]);
            kys.emplace_back(*featureMatches.getKy()[match.getPY()]);
        }
    }

    void downFeatureToSift(const FeatureKeypoints& src, SIFTFeatureKeypoints& target) {
        target.clear();
        target.reserve(src.size());
        for(auto kpPtr: src) {
            target.emplace_back(*dynamic_pointer_cast<SIFTFeatureKeypoint>(kpPtr));
        }
    }


    DenseFeatureMatches::DenseFeatureMatches() {}

    DenseFeatureMatches::DenseFeatureMatches(const shared_ptr<Camera> &cx, const shared_ptr<Camera> &cy) : cx(cx), cy(cy) {}

    DenseFeatureMatches::DenseFeatureMatches(const shared_ptr<Camera> &cx, const shared_ptr<Camera> &cy,
                                             vector<FeatureKeypoint> &kx, vector<FeatureKeypoint> &ky) : cx(cx), cy(cy), kx(kx), ky(ky){}

    shared_ptr<Camera> &DenseFeatureMatches::getCx() {
        return cx;
    }

    shared_ptr<Camera> &DenseFeatureMatches::getCy() {
        return cy;
    }

    vector<FeatureKeypoint> &DenseFeatureMatches::getKx() {
        return kx;
    }

    vector<FeatureKeypoint> &DenseFeatureMatches::getKy() {
        return ky;
    }

    int DenseFeatureMatches::size() const {
        return this->kx.size();
    }

}