//
// Created by liulei on 2020/5/10.
//

#ifndef GraphFusion_POINT_TYPES_H
#define GraphFusion_POINT_TYPES_H

#include <Eigen/Core>
#include "base_types.h"

namespace rtf {
    class Point2D{
    public:
        Scalar x;
        Scalar y;

        Point2D() {

        }

        Point2D(const Vector2& src) {
            this->x = src[0];
            this->y = src[1];
        }

        Point2D(Scalar x, Scalar y) {
            this->x = x;
            this->y = y;
        }

        Vector2 toVector2() const {
            Vector2 pointVector;
            pointVector << x, y;
            return pointVector;
        }
    };


    class Point3D: public Point2D, public Serializable {
    public:
        Scalar z;


        Point3D() {

        }

        Point3D(YAML::Node serNode) {
            x = serNode["x"].as<Scalar>();
            y = serNode["y"].as<Scalar>();
            z = serNode["z"].as<Scalar>();
        }

        Point3D(const Vector3& src) {
            this->x = src[0];
            this->y = src[1];
            this->z = src[2];
        }

        Point3D(Scalar x, Scalar y, Scalar z) {
            this->x = x;
            this->y = y;
            this->z = z;
        }

        Vector3 toVector3() const {
            Vector3 pointVector;
            pointVector << this->x, this->y, this->z;
            return pointVector;
        }

        YAML::Node serialize() {
            YAML::Node node;
            node["x"] = x;
            node["y"] = y;
            node["z"] = z;
            return node;
        }
    };


}

#endif //GraphFusion_POINT_TYPES_H
