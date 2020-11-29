//
// Created by liulei on 2020/8/6.
//

#include "view_graph_util.h"

namespace rtf {
    namespace ViewGraphUtil {
        void dfs(ViewGraph& viewGraph, vector<bool>& visited, vector<int>& cc, int k, double minCost) {
            cc.emplace_back(k);
            visited[k] = true;
            int n = viewGraph.getNodesNum();
            for(int j=0; j<n; j++) {
                Edge edge = viewGraph.getEdge(k, j);
                if(!edge.isUnreachable()&&edge.getCost()<minCost&&!visited[j]) {
                    dfs(viewGraph, visited, cc, j, minCost);
                }
            }
        }

        // find connected Components for undirected graph by dfs
        vector<vector<int>> findConnectedComponents(ViewGraph& viewGraph, double minCost) {
            int n = viewGraph.getNodesNum();
            vector<bool> visited(n);
            for(int i=0; i<n; i++) visited[i] = false;

            vector<vector<int>> components;
            for(int i=0; i<n; i++) {
                if(!visited[i]) {
                    vector<int> cc;
                    dfs(viewGraph, visited, cc, i, minCost);
                    components.emplace_back(cc);
                }
            }
            return components;
        }

        void findCircle(ViewGraph& viewGraph, vector<bool>& visited, vector<int>& path, int k, set<pair<int, int> >& circleCandidates) {
            path.emplace_back(k);
            visited[k] = true;
            int n = viewGraph.getNodesNum();
            for(int j=0; j<n; j++) {
                Edge edge = viewGraph.getEdge(k, j);
                if(!edge.isUnreachable()) {
                    if(visited[j]) {
                        if(j == path[0]&&path.size()>=10) {
                            circleCandidates.insert(make_pair(path[0], min(path[1], path[path.size()-1])));
                        }
                    }else {
                        findCircle(viewGraph, visited, path, j, circleCandidates);
                    }
                }
            }
            path.erase(path.end()-1);
        }

        set<pair<int, int> > findLoopEdges(ViewGraph& viewGraph, int u) {
            int n = viewGraph.getNodesNum();
            vector<bool> visited(n);
            for(int i=0; i<n; i++) visited[i] = false;

            vector<int> path;
            set<pair<int, int> > circleCandidates;
            findCircle(viewGraph, visited, path, u, circleCandidates);
            return circleCandidates;
        }


        // find shortest path by prim
        bool findShortestPathTransVec(ViewGraph& viewGraph, vector<int>& cc,TransformVector& transVec) {
            int n = cc.size();
            int u = 0;

            vector<bool> fixed(n);
            vector<double> lowCost(n);
            vector<int> path(n);
            // initialize vectors
            for(int v=0; v<n; v++) {
                Edge edge = viewGraph.getEdge(cc[u],cc[v]);
                lowCost[v] = edge.isUnreachable()? numeric_limits<double>::infinity(): edge.getCost();
                path[v] = edge.isUnreachable()? -1: u;
                fixed[v] = false;
            }
            fixed[0] = true;

            // begin to update lowCost and path
            bool connected = true;
            for(int k=1; k<n-1; k++) {
                // find min cost index
                double minCost = numeric_limits<double>::infinity();
                int minIndex = -1;
                for(int v=0; v<n; v++) {
                    if(!fixed[v]&&lowCost[v]<minCost) {
                        minCost = lowCost[v];
                        minIndex = v;
                    }
                }
                if(minIndex==-1) {
                    connected = false;
                    break;
                }
                // fix min cost index path
                u = minIndex;
                fixed[u] = true;
                for(int v=0; v<n; v++) {
                    Edge edge = viewGraph.getEdge(cc[u],cc[v]);
                    if(!edge.isUnreachable()) {
                        double cost = edge.getCost();
                        if(!fixed[v]&&lowCost[v]>cost) {
                            lowCost[v] = cost;
                            path[v] = u;
                        }
                    }
                }
            }

            // compute transformation for every node
            for(int i=0; i<n; i++) {
                Transform trans = Transform::Identity();
                int k = i;
                int j = -1;
//                cout << viewGraph[cc[0]].getIndex() << " to " << viewGraph[cc[i]].getIndex() << " path:" << viewGraph[cc[i]].getIndex() << "<-";
                while((j=path[k])!=-1) {
//                    cout << viewGraph[cc[j]].getIndex() << "<-";
                    Edge edge = viewGraph.getEdge(cc[j], cc[k]);
                    LOG_ASSERT(!edge.isUnreachable()) << " error in compute transformation for connected components: the edge is unreachable!";
                    trans = edge.getTransform()*trans;
                    k = j;
                }
//                cout << endl;
                transVec.emplace_back(trans);
            }
            return connected;
        }

        // merge nodes in the same connected component
        void mergeComponentNodes(ViewGraph& viewGraph, vector<int>& cc, Node& node) {
            // pick the principal frame of first node as new node's the principal frame
            bool visible = true;
            for(int i=0; i<cc.size(); i++) {
                Node& cur = viewGraph[cc[i]];
                for(int j=0; j<cur.getFrames().size(); j++) {
                    const auto& frame = cur.getFrames()[j];
                    Transform trans = viewGraph[cc[i]].getGtTransform() * frame->getTransform();
                    frame->setTransform(trans);
                    node.addFrame(frame);
                }
                visible = visible&&cur.isVisible();
            }

            node.status = 0;
            node.setGtTransform(viewGraph[cc[0]].getGtTransform());
            node.setVisible(visible);
        }

        void transformFeatureKeypoints(vector<FeatureKeypoint>& keypoints, const Rotation& R, const Translation& t) {
            for(auto & keypoint : keypoints) {
                Vector3 srcPoint(keypoint.x*keypoint.z, keypoint.y*keypoint.z, keypoint.z);

                Vector3 dstPoint = R*srcPoint + t;

                keypoint.x = dstPoint.x()/dstPoint.z();
                keypoint.y = dstPoint.y()/dstPoint.z();
                keypoint.z = dstPoint.z();
            }
        }

        vector<Edge> findEdgesBetweenComponents(ViewGraph& viewGraph, vector<int>& cc1, const TransformVector& transVec1, vector<int>& cc2, const TransformVector& transVec2) {
            // find edges in two components
            vector<Edge> edges;
            // principal K for two components
            Intrinsic k1 = viewGraph[cc1[0]].getK();
            Intrinsic k2 = viewGraph[cc2[0]].getK();

            for(int i=0; i<cc1.size(); i++) {
                for(int j=0; j<cc2.size(); j++) {
                    int x = cc1[i], y = cc2[j];
                    Edge edge = viewGraph.getEdge(x, y);
                    if(!edge.isUnreachable()) {
                        Transform transX = transVec1[i];
                        Transform transY = transVec2[j];

                        Intrinsic kX = viewGraph[x].getK();
                        Intrinsic kY = viewGraph[y].getK();

                        // transform and k: p' = K(R*K^-1*p+t)
                        Rotation rX = k1*transX.block<3,3>(0,0)*kX.inverse();
                        Rotation rY = k2*transY.block<3,3>(0,0)*kY.inverse();
                        Translation tX = k1*transX.block<3,1>(0,3);
                        Translation tY = k2*transY.block<3,1>(0,3);

                        // transform key points
                        transformFeatureKeypoints(edge.getKxs(), rX, tY);
                        transformFeatureKeypoints(edge.getKys(), rY, tY);

                        // trans12 = trans1*relative_trans*trans2^-1
                        Transform relativeTrans = transX * edge.getTransform() * transY.inverse();
                        edge.setTransform(relativeTrans);

                        edges.emplace_back(edge);
                    }
                }
            }
            return edges;
        }

        Edge selectEdgeBetweenComponents(ViewGraph& viewGraph, vector<int>& cc1, vector<int>& cc2) {
            // principal K for two components
            Intrinsic k1 = viewGraph[cc1[0]].getK();
            Intrinsic k2 = viewGraph[cc2[0]].getK();

            int bestI=-1, bestJ=-1;
            Edge bestEdge;
            bestEdge.setCost(numeric_limits<double>::infinity());
            bestEdge.setUnreachable();
            for(int i=0; i<cc1.size(); i++) {
                for(int j=0; j<cc2.size(); j++) {
                    int x = cc1[i], y = cc2[j];
                    Edge edge = viewGraph.getEdge(x, y);
                    if(edgeCompare(edge, bestEdge)) {
                        bestEdge = edge;
                        bestI = i;
                        bestJ = j;
                    }
                }
            }
            if(!bestEdge.isUnreachable()&&!bestEdge.getKxs().empty()) {
                Transform transX = viewGraph[cc1[bestI]].getGtTransform();
                Transform transY = viewGraph[cc2[bestJ]].getGtTransform();

                Intrinsic kX = viewGraph[cc1[bestI]].getK();
                Intrinsic kY = viewGraph[cc2[bestJ]].getK();

                // transform and k: p' = K(R*K^-1*p+t)
                Rotation rX = k1*transX.block<3,3>(0,0)*kX.inverse();
                Rotation rY = k2*transY.block<3,3>(0,0)*kY.inverse();
                Translation tX = k1*transX.block<3,1>(0,3);
                Translation tY = k2*transY.block<3,1>(0,3);

                // transform key points
                transformFeatureKeypoints(bestEdge.getKxs(), rX, tX);
                transformFeatureKeypoints(bestEdge.getKys(), rY, tY);

                // trans12 = trans1*relative_trans*trans2^-1
                Transform relativeTrans = transX * bestEdge.getTransform() * transY.inverse();
                bestEdge.setTransform(relativeTrans);
            }

            return bestEdge;
        }
    }
}