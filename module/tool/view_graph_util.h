//
// Created by liulei on 2020/8/6.
//

#ifndef GraphFusion_VIEW_GRAPH_UTIL_H
#define GraphFusion_VIEW_GRAPH_UTIL_H

#include "../datastructure/view_graph.h"

namespace rtf {
    namespace ViewGraphUtil {
        void dfs(ViewGraph& viewGraph, vector<bool>& visited, vector<int>& cc, int k, double minCost);

        // find connected Components for undirected graph by dfs
        vector<vector<int>> findConnectedComponents(ViewGraph& viewGraph, double minCost);

        /**
         * merge all circle containing u
         * @param viewGraph
         * @param minCost
         * @param u
         * @return
         */
        vector<int> findCircleComponent(ViewGraph& viewGraph, double minCost, int u);

            // find shortest path by prim
        bool findShortestPathTransVec(ViewGraph& viewGraph, vector<int>& cc,TransformVector& transVec);

        // merge nodes in the same connected component
        void mergeComponentNodes(ViewGraph& viewGraph, vector<int>& cc, const TransformVector& transVec, Node& node);

        void transformFeatureKeypoints(vector<FeatureKeypoint>& keypoints, const Rotation& R, const Translation& t);

        vector<Edge> findEdgesBetweenComponents(ViewGraph& viewGraph, vector<int>& cc1, const TransformVector& transVec1, vector<int>& cc2, const TransformVector& transVec2);

        Edge selectEdgeBetweenComponents(ViewGraph& viewGraph, vector<int>& cc1, const TransformVector& transVec1 , vector<int>& cc2, const TransformVector& transVec2);
    }
}


#endif //GraphFusion_VIEW_GRAPH_UTIL_H
