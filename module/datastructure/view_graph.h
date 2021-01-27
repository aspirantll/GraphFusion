//
// Created by liulei on 2020/8/1.
//

#ifndef GraphFusion_VIEW_GRAPH_H
#define GraphFusion_VIEW_GRAPH_H

#include <vector>
#include <Eigen/Core>
#include "../feature/feature_point.h"
#include "yaml-cpp/yaml.h"

using namespace std;

namespace rtf {
    class Frame: public FrameRGBDT {
    protected:
        ORBFeaturePoints kps;

        bool visible;

    public:

        Frame(YAML::Node serNode);

        Frame(shared_ptr<FrameRGBD> frameRGBD);

        ORBFeaturePoints &getKps();

        void setKps(const ORBFeaturePoints &kps);

        void setFrameIndex(uint32_t frameIndex);

        bool isVisible() const;

        void setVisible(bool visible);
    };

    class KeyFrame {
    protected:
        vector<shared_ptr<Frame>> frames;
        vector<int> pathLengths;
        map<int, int> indexToInnerMap;
        SE3 transform;
        ORBFeaturePoints kps;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        KeyFrame();

        int getIndex();

        Intrinsic getK();

        shared_ptr<Camera> getCamera();

        Transform getTransform();

        Transform getTransform(int frameIndex);

        void setTransform(Transform trans);

        void addFrame(shared_ptr<Frame> frame, int pathLength=0);

        int getPathLength(int frameIndex);

        vector<shared_ptr<Frame>> &getFrames();

        shared_ptr<Frame> getFirstFrame();

        shared_ptr<Frame> getFrame(int frameIndex);

        ORBFeaturePoints &getKps();

        void setKps(const ORBFeaturePoints &kps);
    };

    class Edge {
    protected:
        vector<FeatureKeypoint> kxs;
        vector<FeatureKeypoint> kys;
        map<int, int> matchIndexesX;
        map<int, int> matchIndexesY;
        SE3 transform;
        double cost;
        bool unreachable = false;
    private:
        Edge(bool unreachable);

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        static Edge UNREACHABLE;

        Edge();

        vector<FeatureKeypoint> &getKxs();

        void setKxs(const vector<FeatureKeypoint> &kxs);

        vector<FeatureKeypoint> &getKys();

        void setKys(const vector<FeatureKeypoint> &kys);

        bool containKeypoint(int index);

        FeatureKeypoint getMatchKeypoint(int index);

        void setTransform(Transform transformation);

        void setCost(double cost);

        void setUnreachable();

        Transform getTransform();

        void setSE(SE3 t);

        SE3 getSE();

        double getCost();

        bool isUnreachable();

        Edge reverse();

        YAML::Node serialize();

    };


    class Node {
    protected:
        // the first frame is key frame
        vector<shared_ptr<KeyFrame>> frames;
        vector<int> frameIndexes;
        map<int, int> frameIndexesToInnerIndexes;
        vector<int> connections;
        // flag for vis
        bool visible = true;
        SE3 gtTrans;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        int status; //0-frame node 1-keyframe node

        Node();

        int getIndex();

        shared_ptr<Camera> getCamera();

        Intrinsic getK();

        Intrinsic getKInv();

        vector<shared_ptr<KeyFrame>> &getFrames();

        void addFrame(shared_ptr<KeyFrame> frame);

        vector<int> &getFrameIndexes();

        Transform getTransform(int frameIndex);

        shared_ptr<KeyFrame> getKeyFrame(int frameIndex);

        void setGtTransform(Transform trans);

        Transform getGtTransform();

        void setGtSE(SE3 gt);

        SE3 getGtSE();

        void addConnections(int v);

        vector<int> getConnections();

        void setVisible(bool visible);

        bool isVisible();
    };

    typedef vector<Node, Eigen::aligned_allocator<Node>> NodeVector;

    class ViewGraph {
    protected:
        NodeVector nodes;
        EigenUpperTriangularMatrix <Edge> *adjMatrix;

        vector<shared_ptr<KeyFrame>> sourceFrames;
        vector<int> frameNodeIndex;
        map<int,int> frameToInnerIndex;

        int curMaxRoot = 0;
        vector<int> parentIndexes;
        vector<int> rootIndexes;
        vector<int> nodePathLens;

        bool changeStatus = false;

        SE3 computeTransform(int u, map<int, int>& innerMap, vector<int>& cc, vector<bool>& visited);

        int computePathLens(int index);
    public:
        ViewGraph();

        ~ViewGraph();

        ViewGraph(int nodesNum);

        void reset(int nodesNum = 0, Edge defaultValue = Edge::UNREACHABLE);

        int getNodesNum();

        Transform getFrameTransform(int frameIndex);

        Edge &operator()(int i, int j);

        Node &operator[](int index);

        Node &extendNode(shared_ptr<KeyFrame> frame);

        shared_ptr<KeyFrame> indexFrame(int index);

        void addSourceFrame(shared_ptr<KeyFrame> frame);

        Edge getEdge(int i, int j);

        double getEdgeCost(int i, int j);

        Transform getEdgeTransform(int i, int j);

        SE3 getEdgeSE(int i, int j);

        void setEdgeTransform(int i, int j, Transform trans);

        bool existEdge(int i, int j);

        int getParent(int child);

        int getPathLen(int frameIndex);

        int getFramesNum();

        vector<shared_ptr<KeyFrame>> getSourceFrames();

        void updateNodeIndex(vector<vector<int>> &ccs);

        int findNodeIndexByFrameIndex(int frameIndex);

        int updateSpanningTree();

        void generateSpanningTree();

        vector<vector<int>> getConnectComponents();

        void computeGtTransforms();

        vector<int> getBestCovisibilityNodes(int index, int k);

        bool isVisible(int frameIndex);

        bool isChange();

        void check();

        void print();

    };

    bool edgeCompare(Edge &one, Edge &another);
}
#endif //GraphFusion_VIEW_GRAPH_H
