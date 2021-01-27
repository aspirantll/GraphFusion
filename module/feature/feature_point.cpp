//
// Created by liulei on 2020/6/5.
//

#include "feature_point.h"

#include <utility>

namespace rtf {
    FeatureMatches::FeatureMatches() {}

    FeatureMatches::FeatureMatches(FeaturePoints<uint8_t> &fp1, FeaturePoints<uint8_t> &fp2,
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

    void FeatureMatches::setFp1(FeaturePoints<uint8_t> &fp1) {
        FeatureMatches::fp1 = &fp1;
    }

    void FeatureMatches::setFp2(FeaturePoints<uint8_t> &fp2) {
        FeatureMatches::fp2 = &fp2;
    }

    FeaturePoints<uint8_t> &FeatureMatches::getFp1() {
        return *fp1;
    }

    FeaturePoints<uint8_t> &FeatureMatches::getFp2() {
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