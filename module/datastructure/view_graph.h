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
    class Frame;
    class ViewCluster;

    class ConnectionCandidate {
    protected:
        vector<FeatureKeypoint> kxs;
        vector<FeatureKeypoint> kys;
        SE3 transform;
        double cost;
        bool unreachable = false;
    private:
        ConnectionCandidate(bool unreachable);

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        static ConnectionCandidate UNREACHABLE;

        ConnectionCandidate();

        vector<FeatureKeypoint> &getKxs();

        void setKxs(const vector<FeatureKeypoint> &kxs);

        vector<FeatureKeypoint> &getKys();

        void setKys(const vector<FeatureKeypoint> &kys);

        void setTransform(Transform transformation);

        void setCost(double cost);

        Transform getTransform();

        SE3 getSE();

        double getCost();

        bool isUnreachable();
    };

    template <class V>
    class Connection {
    protected:
        shared_ptr<V> h;
        shared_ptr<V> t;
        float pointWeight;
        SE3 transform;
        double cost;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Connection() {}

        Connection(const shared_ptr<V> &h, const shared_ptr<V> &t, float pointWeight, const SE3 &transform, double cost)
                : h(h), t(t), pointWeight(pointWeight), transform(transform), cost(cost) {}

        shared_ptr<V> getHead() {
            return h;
        }

        void setHead(const shared_ptr<V> &h) {
            Connection::h = h;
        }

        shared_ptr<V> getTail() {
            return t;
        }

        void setTail(const shared_ptr<V> &t) {
            Connection::t = t;
        }

        float getPointWeight() {
            return pointWeight;
        }

        void setPointWeight(float pointWeight) {
            Connection::pointWeight = pointWeight;
        }

        Transform getTransform() {
            return transform.matrix();
        }

        void setTransform(const Transform &transform) {
            Connection::transform = SE3(transform);
        }

        SE3 getSE() {
            return transform;
        }

        double getCost() {
            return cost;
        }

        void setCost(double cost) {
            Connection::cost = cost;
        }
    };

    typedef Connection<Frame> FrameConnection;
    typedef Connection<ViewCluster> ViewConnection;

    class Frame: public FrameRGBDT {
    protected:
        SIFTFeaturePoints kps;
        map<int, shared_ptr<FrameConnection>> connections;

        bool visible;

    public:

        Frame(YAML::Node serNode);

        Frame(shared_ptr<FrameRGBD> frameRGBD);

        SIFTFeaturePoints &getKps();

        void setKps(const SIFTFeaturePoints &kps);

        void setFrameIndex(uint32_t frameIndex);

        bool isVisible() const;

        void setVisible(bool visible);

        bool existConnection(int v);

        shared_ptr<FrameConnection> getConnection(int v);

        void addConnection(int v, shared_ptr<FrameConnection> con);

        vector<shared_ptr<FrameConnection>> getConnections();

        map<int, shared_ptr<FrameConnection>> getConnectionMap();
    };

    class ViewCluster {
    protected:
        vector<shared_ptr<Frame>> frames;
        vector<int> pathLengths;

        map<int, int> indexToInnerMap;
        map<int, shared_ptr<ViewConnection>> connections;

        int rootIndex;
        SE3 transform;
        bool visible = true;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ViewCluster();

        int getIndex();

        Intrinsic getK();

        shared_ptr<Camera> getCamera();

        Transform getFrameTransform(int frameIndex);

        SE3 getFrameSE(int frameIndex);

        Transform getTransform();

        void setTransform(Transform trans);

        SE3 getSE();

        void setSE(SE3 se3);

        void addFrame(shared_ptr<Frame> frame, int pathLength=0);

        int getPathLength(int frameIndex);

        vector<shared_ptr<Frame>> &getFrames();

        void setRootIndex(int index);

        shared_ptr<Frame> getRootFrame();

        shared_ptr<Frame> getFrame(int frameIndex);

        bool existConnection(int v);

        shared_ptr<ViewConnection> getConnection(int v);

        void addConnection(int v, shared_ptr<ViewConnection> con);

        vector<shared_ptr<ViewConnection>> getConnections();

        map<int, shared_ptr<ViewConnection>> getConnectionMap();

        void setVisible(bool visible);

        bool isVisible();

    };


    class ViewGraph {
    protected:
        vector<shared_ptr<Frame>> sourceFrames;
        vector<shared_ptr<ViewCluster>> nodes;
        vector<int> frameNodeIndex;
        map<int,int> frameToInnerIndex;

        int curMaxRoot = 0;
        vector<int> parentIndexes;
        vector<int> rootIndexes;
        vector<int> nodePathLens;

        SE3 computeTransform(int u, map<int, int>& innerMap, vector<int>& cc, vector<bool>& visited);

        int computePathLens(int index);
    public:
        ViewGraph();

        ~ViewGraph();

        ViewGraph(int nodesNum);

        void reset(int nodesNum = 0);

        int getNodesNum();

        int getFramesNum();

        Transform getViewTransform(int frameIndex);

        shared_ptr<ViewConnection> operator()(int i, int j);

        shared_ptr<ViewCluster> operator[](int index);

        void addSourceFrame(shared_ptr<Frame> frame);

        void extendNode(shared_ptr<ViewCluster> node);
        
        double getEdgeCost(int i, int j);

        SE3 getEdgeSE(int i, int j);

        bool existEdge(int i, int j);

        shared_ptr<Camera> getCamera();

        shared_ptr<Frame> getLastFrame();

        shared_ptr<Frame> getFirstFrame();

        int getParent(int child);

        int getPathLenByFrameIndex(int frameIndex);

        int getPathLenByNodeIndex(int nodeIndex);

        int findNodeIndexByFrameIndex(int frameIndex);

        shared_ptr<ViewCluster> findNodeByFrameIndex(int frameIndex);

        shared_ptr<Frame> findFrameByIndex(int frameIndex);

        int updateSpanningTree(int lastIndex);

        vector<vector<int>> getConnectComponents();

        vector<int> getBestCovisibilityNodes(int index, int k);

        int getMaxRoot();

        void optimizeBestRootNode();

        vector<int> maxConnectedComponent();

        bool isVisible(int frameIndex);

        void print();

    };
}
#endif //GraphFusion_VIEW_GRAPH_H
