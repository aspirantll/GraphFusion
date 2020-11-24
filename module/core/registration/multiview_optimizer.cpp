//
// Created by liulei on 2020/11/22.
//

#include "registrations.h"

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>

#include <memory>

using  namespace g2o;

typedef BlockSolver< BlockSolverTraits<VertexSE3::Dimension, -1>> SlamBlockSolver;
typedef LinearSolverPCG<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;


namespace rtf {
    MultiviewOptimizer::MultiviewOptimizer(const GlobalConfig &globalConfig) : globalConfig(globalConfig) {
        baRegistration = new BARegistration(globalConfig);
    }

    void MultiviewOptimizer::poseGraphOptimize(ViewGraph &viewGraph, const vector<int>& cc) {
        // pose graph optimization
        // How the problem is mapped to g2o:
        // The nodes get the global_T_frame transformation.
        // The edges get A as "from" (vertices()[0]),
        //               B as "to" (vertices()[1]), and
        //               A_tr_B as measurement.
        auto linearSolver = std::make_unique<SlamLinearSolver>();
        auto blockSolver = std::make_unique<SlamBlockSolver>(std::move(linearSolver));
        auto* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

        auto optimizer = new g2o::SparseOptimizer();
        optimizer->setAlgorithm( solver );
        optimizer->setVerbose( false );

        //1.add frame as node
        for (int i = 0; i < viewGraph.getNodesNum(); i++) {
            auto* node = new VertexSE3();
            node->setId(i);
            node->setEstimate(Eigen::Isometry3d(GeoUtil::reverseTransformation(viewGraph[i].nGtTrans).cast<double>()));
            optimizer->addVertex(node);
        }

        // Fix the first pose to account for gauge freedom.
        optimizer->vertex(0)->setFixed(true);

        // 2. add edges in connected components
        for(int i=0; i<viewGraph.getNodesNum(); i++) {
            int parent = viewGraph.getParent(i);
            if(parent!=-1) {
                Transform trans = GeoUtil::reverseTransformation(viewGraph.getEdgeTransform(parent, i));
                auto* edge = new EdgeSE3();
                edge->vertices()[0] = optimizer->vertex(parent);
                edge->vertices()[1] = optimizer->vertex(i);
                edge->setMeasurement(Eigen::Isometry3d(trans.cast<double>()));
                edge->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
                optimizer->addEdge(edge);
            }
        }
        for(int i=0; i<cc.size(); i++) {
            for (int j = i + 1; j < cc.size(); j++) {
                if(viewGraph.getParent(cc[i])==cc[j]||viewGraph.getParent(cc[j])==cc[i]) {
                    continue;
                }
                Edge connection = viewGraph.getEdge(cc[i], cc[j]);
                if (!connection.isUnreachable()) {
                    auto* edge = new EdgeSE3();
                    edge->vertices()[0] = optimizer->vertex(cc[i]);
                    edge->vertices()[1] = optimizer->vertex(cc[j]);
                    edge->setMeasurement(Eigen::Isometry3d(GeoUtil::reverseTransformation(connection.getTransform()).cast<double>()));
                    edge->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
                    optimizer->addEdge(edge);
                }
            }
        }

        optimizer->save("/home/liulei/result_before.g2o");
        //3. pose graph optimization
        optimizer->initializeOptimization();
        constexpr int kMaxIterations = 20;
        optimizer->optimize(kMaxIterations);
        optimizer->save("/home/liulei/result_after.g2o");

        //4. update global transformation
        for(int i=0; i<cc.size(); i++) {
            Transform globalToFrame = reinterpret_cast<const g2o::VertexSE3*>(optimizer->vertex(i))->estimate().matrix().cast<float>();
            viewGraph[cc[i]].nGtTrans = GeoUtil::reverseTransformation(globalToFrame);
            cout << "new:" << GeoUtil::reverseTransformation(globalToFrame) << endl;
        }

    }

    RegReport MultiviewOptimizer::optimize(ViewGraph &viewGraph, const vector<int>& cc) {
        // pose graph optimization
        poseGraphOptimize(viewGraph, cc);
        // multiview bundle adjustment
        return baRegistration->multiViewBundleAdjustment(viewGraph, cc);
    }
}