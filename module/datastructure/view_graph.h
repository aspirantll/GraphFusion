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
    class Connection;
    class ViewCluster;


    class Frame: public FrameRGBDT {
    protected:
        SIFTFeaturePoints kps;

        bool visible;

    public:

        Frame(YAML::Node serNode);

        Frame(shared_ptr<FrameRGBD> frameRGBD);

        SIFTFeaturePoints &getKps();

        void setKps(const SIFTFeaturePoints &kps);

        void setFrameIndex(uint32_t frameIndex);

        bool isVisible() const;

        void setVisible(bool visible);
    };

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


    class Connection {
    protected:
        shared_ptr<ViewCluster> v;
        Vector3 p;
        float pointWeight;
        SE3 transform;
        double cost;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Connection();

        Connection(shared_ptr<ViewCluster> v, Vector3 p, float pointWeight, SE3 transform, double cost);

        void setViewCluster(const shared_ptr<ViewCluster> &v);

        void setNormPoint(const Vector3 &p);

        void setPointWeight(float pointWeight);

        void setTransform(const Transform &transform);

        void setSE(const SE3 &transform);

        void setCost(double cost);


        shared_ptr<ViewCluster> getViewCluster();

        Vector3 getNormPoint();

        float getPointWeight();

        Transform getTransform();

        SE3 getSE();

        double getCost();
    };


    class ViewCluster {
    protected:
        vector<shared_ptr<Frame>> frames;
        vector<int> pathLengths;

        map<int, int> indexToInnerMap;
        map<int, shared_ptr<Connection>> connections;

        int rootIndex;
        SE3 transform;
        bool visible = true;

        SIFTFeaturePoints kps;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ViewCluster();

        int getIndex();

        Intrinsic getK();

        shared_ptr<Camera> getCamera();

        Transform getFrameTransform(int frameIndex);

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

        shared_ptr<Connection> getConnection(int v);

        void addConnection(int v, shared_ptr<Connection> con);

        vector<shared_ptr<Connection>> getConnections();

        void setVisible(bool visible);

        bool isVisible();

        SIFTFeaturePoints &getKps();

        void setKps(const SIFTFeaturePoints &kps);
    };


    class ViewGraph {
    protected:
        vector<shared_ptr<ViewCluster>> sourceNodes;
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

        Transform getViewTransform(int frameIndex);

        shared_ptr<Connection> operator()(int i, int j);

        shared_ptr<ViewCluster> operator[](int index);

        void extendNode(shared_ptr<ViewCluster> node);
        
        double getEdgeCost(int i, int j);

        SE3 getEdgeSE(int i, int j);

        bool existEdge(int i, int j);

        shared_ptr<Camera> getCamera();

        int getParent(int child);

        int getPathLen(int frameIndex);

        int findNodeIndexByFrameIndex(int frameIndex);

        shared_ptr<ViewCluster> findNodeByFrameIndex(int frameIndex);

        int updateSpanningTree();

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
