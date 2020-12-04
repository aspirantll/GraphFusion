//
// Created by liulei on 2020/11/26.
//

#ifndef GRAPHFUSION_OPTIMIZERT_H
#define GRAPHFUSION_OPTIMIZERT_H

#include "registrations.h"
#include "../../datastructure/view_graph.h"

namespace rtf {
    class BALProblem {
    public:
        explicit BALProblem(ViewGraph &viewGraph, const vector<int>& cc);
        ~BALProblem();

        // Move the "center" of the reconstruction to the origin, where the
        // center is determined by computing the marginal median of the
        // points. The reconstruction is then scaled so that the median
        // absolute deviation of the points measured from the origin is
        // 100.0.
        //
        // The reprojection error of the problem remains the same.
        void Normalize();

        // Perturb the camera pose and the geometry with random normal
        // numbers with corresponding standard deviations.
        void Perturb(const double rotation_sigma,
                     const double translation_sigma,
                     const double point_sigma);

        int camera_block_size()      const { return 7; }
        int point_block_size()       const { return 3;                         }
        int num_cameras()            const { return num_cameras_;              }
        int num_points()             const { return num_points_;               }
        int num_observations()       const { return num_observations_;         }
        int num_parameters()         const { return num_parameters_;           }
        const int* point_index()     const { return point_index_;              }
        const int* camera_index()    const { return camera_index_;             }
        const double* observations() const { return observations_;             }
        const double* parameters()   const { return parameters_;               }
        const double* cameras()      const { return parameters_;               }
        double* mutable_cameras()          { return parameters_;               }
        double* mutable_points() {
            return parameters_  + camera_block_size() * num_cameras_;
        }

        void Denormalize();


    private:
        void CameraToAngleAxisAndCenter(const double* camera,
                                        double* angle_axis,
                                        double* center) const;

        void AngleAxisAndCenterToCamera(const double* angle_axis,
                                        const double* center,
                                        double* camera) const;
        int num_cameras_;
        int num_points_;
        int num_observations_;
        int num_parameters_;

        int* point_index_;
        int* camera_index_;
        double* observations_;
        // The parameter vector is laid out as follows
        // [camera_1, ..., camera_n, point_1, ..., point_m]
        double* parameters_;

        double scale = 0;
        Vector3 median = Vector3::Zero();
    };

    class Optimizer {
    public:
        static void poseGraphOptimizeCeres(ViewGraph &viewGraph, const vector<pair<int, int> >& loops);

        static void globalBundleAdjustmentCeres(ViewGraph &viewGraph, const vector<int>& cc);
    };
}


#endif //GRAPHFUSION_OPTIMIZERT_H
