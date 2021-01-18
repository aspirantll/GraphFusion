//
// Created by liulei on 2020/8/1.
//

#include "view_graph.h"
#include <utility>
#include "../processor/frame_converters.h"

namespace rtf {
    Frame::Frame(YAML::Node serNode): FrameRGBDT(serNode), FrameBase(serNode) {

    }

    Frame::Frame(shared_ptr<FrameRGBD> frameRGBD): FrameRGBDT(frameRGBD, Transform::Identity()), FrameBase(frameRGBD->getId(), frameRGBD->getCamera()) {
        setFrameIndex(frameRGBD->getFrameIndex());
        visible = true;
    }

    SIFTFeaturePoints &Frame::getKps() {
        return kps;
    }

    void Frame::setKps(const SIFTFeaturePoints &kps) {
        Frame::kps = kps;
    }

    void Frame::setFrameIndex(uint32_t frameIndex) {
        FrameBase::frameIndex = frameIndex;
        kps.setFIndex(frameIndex);
    }

    bool Frame::isVisible() const {
        return visible;
    }

    void Frame::setVisible(bool visible) {
        Frame::visible = visible;
    }

    KeyFrame::KeyFrame(){
        transform = SE3(Transform::Identity());
    }

    int KeyFrame::getIndex() {
        return frames[0]->getFrameIndex();
    }

    shared_ptr<Camera> KeyFrame::getCamera() {
        return frames[0]->getCamera();
    }

    Transform KeyFrame::getTransform() {
        return transform.matrix();
    }

    Transform KeyFrame::getTransform(int frameIndex) {
        return getFrame(frameIndex)->getTransform();
    }

    void KeyFrame::setTransform(Transform trans) {
        transform = SE3(trans);
    }

    Intrinsic KeyFrame::getK() {
        return getCamera()->getK();
    }

    void KeyFrame::addFrame(shared_ptr<Frame> frame, int pathLength) {
        indexToInnerMap.insert(map<int, int>::value_type(frame->getFrameIndex(), frames.size()));
        frames.emplace_back(frame);
        pathLengths.emplace_back(pathLength);
    }

    int KeyFrame::getPathLength(int frameIndex) {
        int innerIndex = indexToInnerMap[frameIndex];
        return pathLengths[innerIndex];
    }

    vector<shared_ptr<Frame>> &KeyFrame::getFrames() {
        return frames;
    }

    shared_ptr<Frame> KeyFrame::getFirstFrame() {
        return frames[0];
    }

    shared_ptr<Frame> KeyFrame::getFrame(int frameIndex) {
        int innerIndex = indexToInnerMap[frameIndex];
        return frames[innerIndex];
    }

    SIFTFeaturePoints &KeyFrame::getKps() {
        return kps;
    }

    void KeyFrame::setKps(const SIFTFeaturePoints &kps) {
        KeyFrame::kps = kps;
    }

    Transform Edge::getTransform() {
        return transform.matrix();
    }

    void Edge::setSE(SE3 t) {
        transform = t;
    }

    SE3 Edge::getSE() {
        return transform;
    }

    double Edge::getCost() {
        return cost;
    }

    Edge Edge::reverse() {
        Edge edge(this->unreachable);
        if(!edge.isUnreachable()) {
            edge.setKxs(kys);
            edge.setKys(kxs);
            edge.transform = transform.inverse();
            edge.cost = cost;
            edge.matchIndexesX = matchIndexesY;
            edge.matchIndexesY = matchIndexesX;
        }
        return edge;
    }

    bool Edge::isUnreachable() {
        return unreachable;
    }

    Edge::Edge() : Edge(false) {}

    Edge Edge::UNREACHABLE(true);

    void Edge::setUnreachable() {
        unreachable = true;
    }

    void Edge::setTransform(Transform transformation) {
        this->transform = SE3(transformation);
    }

    void Edge::setCost(double cost) {
        this->cost = cost;
        this->unreachable = false;
    }

    YAML::Node Edge::serialize() {
        YAML::Node node;
        node["cost"] = cost;
        node["unreachable"] = unreachable;

        return node;
    }

    Edge::Edge(bool unreachable) : unreachable(unreachable) {}

    vector<FeatureKeypoint> &Edge::getKxs() {
        return kxs;
    }

    void Edge::setKxs(const vector<FeatureKeypoint> &kxs) {
        matchIndexesX.clear();
        for(int i=0; i<kxs.size(); i++) {
            matchIndexesX.insert(map<int,int>::value_type(kxs[i].getIndex(), i));
        }
        Edge::kxs = kxs;
    }

    vector<FeatureKeypoint> &Edge::getKys() {
        return kys;
    }

    void Edge::setKys(const vector<FeatureKeypoint> &kys) {
        matchIndexesY.clear();
        for(int i=0; i<kys.size(); i++) {
            matchIndexesY.insert(map<int,int>::value_type(kys[i].getIndex(), i));
        }
        Edge::kys = kys;
    }

    bool Edge::containKeypoint(int index) {
        return matchIndexesX.count(index);
    }

    FeatureKeypoint Edge::getMatchKeypoint(int index) {
        int innerIndex = matchIndexesX[index];
        return kys[innerIndex];
    }

    shared_ptr<Camera> Node::getCamera() {
        return frames[0]->getCamera();
    }

    Intrinsic Node::getK() {
        return getCamera()->getK();
    }

    Intrinsic Node::getKInv() {
        return getCamera()->getReverseK();
    }

    vector<int>& Node::getFrameIndexes() {
        return frameIndexes;
    }

    Transform Node::getTransform(int frameIndex) {
        return getKeyFrame(frameIndex)->getTransform();
    }

    shared_ptr<KeyFrame> Node::getKeyFrame(int frameIndex) {
        int inner = frameIndexesToInnerIndexes[frameIndex];
        return frames[inner];
    }

    void Node::setGtTransform(Transform trans) {
        gtTrans = SE3(trans);
    }

    Transform Node::getGtTransform() {
        return gtTrans.matrix();
    }

    void Node::setGtSE(SE3 gt) {
        gtTrans = gt;
    }

    SE3 Node::getGtSE() {
        return gtTrans;
    }


    void Node::setVisible(bool visible) {
        this->visible = visible;
    }

    bool Node::isVisible() {
        return this->visible;
    }

    Node::Node() {}

    int Node::getIndex() {
        return frames[0]->getIndex();
    }

    vector<shared_ptr<KeyFrame>> &Node::getFrames() {
        return this->frames;
    }

    void Node::addFrame(shared_ptr<KeyFrame> frame) {
        int index = frame->getIndex();
        frames.emplace_back(frame);
        frameIndexes.emplace_back(index);
        frameIndexesToInnerIndexes.insert(map<int, int>::value_type(index, frames.size()-1));
    }

    ViewGraph::ViewGraph(): ViewGraph(0) {

    }

    ViewGraph::~ViewGraph() {
        delete adjMatrix;
    }

    ViewGraph::ViewGraph(int nodesNum) {
        adjMatrix = new EigenUpperTriangularMatrix<Edge>();
        reset(nodesNum);
    }

    void ViewGraph::reset(int nodesNum, Edge defaultValue) {
        if(nodesNum <= 0) {
            nodes.clear();
            adjMatrix->resize(0, defaultValue);
            sourceFrames.clear();
            frameNodeIndex.clear();
            parentIndexes.clear();
            rootIndexes.clear();
            curMaxRoot = -1;
        }else {
            nodes.resize(nodesNum, Node());
            adjMatrix->resize(nodesNum, defaultValue);
            parentIndexes.resize(nodesNum, -1);
            rootIndexes.resize(nodesNum);
            iota(rootIndexes.begin(), rootIndexes.end(), 0);
            curMaxRoot = 0;
        }
    }

    int ViewGraph::getNodesNum() {
        return nodes.size();
    }

    Transform ViewGraph::getFrameTransform(int frameIndex) {
        int innerIndex = frameToInnerIndex[frameIndex];
        int nodeIndex = frameNodeIndex[innerIndex];
        return nodes[nodeIndex].getGtTransform()*sourceFrames[innerIndex]->getTransform();
    }

    SE3 ViewGraph::computeTransform(int u, map<int, int>& innerMap, vector<int>& cc, vector<bool>& visited) {
        int nodeIndex = cc[u];
        if(visited[u]) return nodes[nodeIndex].getGtSE();
        int parent = parentIndexes[cc[u]];
        if(parent==-1) {
            nodes[nodeIndex].setGtTransform(Transform::Identity());
        }else {
            int v = innerMap[parent];
            SE3 trans = computeTransform(v, innerMap, cc, visited)*getEdgeSE(parent, nodeIndex);
            nodes[nodeIndex].setGtSE(trans);
        }
        visited[u] = true;
        return nodes[nodeIndex].getGtSE();
    }

    int ViewGraph::computePathLens(int index) {
        if(index < 0) return -1;
        if(nodePathLens[index]>=0) return nodePathLens[index];
        nodePathLens[index] = computePathLens(parentIndexes[index])+1;
        return nodePathLens[index];
    }

    Edge &ViewGraph::operator()(int i, int j) {
        return (*adjMatrix)(i, j);
    }

    Node &ViewGraph::operator[](int index) {
        return nodes[index];
    }

    Node& ViewGraph::extendNode(shared_ptr<KeyFrame> frame) {
        Node node;
        node.addFrame(frame);
        node.setGtTransform(Transform::Identity());
        node.status = 1;

        nodes.emplace_back(node);
        adjMatrix->extend();

        sourceFrames.emplace_back(frame);
        frameNodeIndex.emplace_back(nodes.size()-1);
        frameToInnerIndex.insert(map<int, int>::value_type(frame->getIndex(), sourceFrames.size()-1));
        parentIndexes.emplace_back(-1);
        rootIndexes.emplace_back(nodes.size()-1);
        nodePathLens.emplace_back(0);
        return nodes[nodes.size()-1];
    }

    shared_ptr<KeyFrame> ViewGraph::indexFrame(int index) {
        LOG_ASSERT(frameToInnerIndex.count(index)) << "error index";
        int innerIndex = frameToInnerIndex[index];
        return sourceFrames[innerIndex];
    }

    int ViewGraph::getFramesNum() {
        return sourceFrames.size();
    }

    vector<shared_ptr<KeyFrame>> ViewGraph::getSourceFrames() {
        return sourceFrames;
    }

    void ViewGraph::addSourceFrame(shared_ptr<KeyFrame> frame) {
        sourceFrames.emplace_back(frame);
        frameNodeIndex.emplace_back(-1);
        frameToInnerIndex.insert(map<int, int>::value_type(frame->getIndex(), sourceFrames.size()-1));
    }

    Edge ViewGraph::getEdge(int i, int j) {
        if(i<=j) return (*adjMatrix)(i,j);
        else {
            return (*adjMatrix)(i,j).reverse();
        }
    }

    double ViewGraph::getEdgeCost(int i, int j) {
        LOG_ASSERT(!(*this)(i,j).isUnreachable());
        return (*this)(i,j).getCost();
    }

    Transform ViewGraph::getEdgeTransform(int i, int j) {
        LOG_ASSERT(!(*this)(i,j).isUnreachable());
        Transform trans = (*this)(i, j).getTransform();
        return i<=j?trans:trans.inverse();
    }

    SE3 ViewGraph::getEdgeSE(int i, int j) {
        LOG_ASSERT(!(*this)(i,j).isUnreachable());
        SE3 trans = (*this)(i, j).getSE();
        return i<=j?trans:trans.inverse();
    }

    void ViewGraph::setEdgeTransform(int i, int j, Transform trans) {
        if(i<j) (*this)(i, j).setTransform(trans);
        else {
            (*this)(i, j).setTransform(trans.inverse());
        }
    }

    void ViewGraph::updateNodeIndex(vector<vector<int>>& ccs) {
        // collect all frame
        vector<vector<int>> frameIndexes(ccs.size());
        for(int i=0; i<ccs.size(); i++) {
            for(auto ind: ccs[i]) {
                for(int j=0; j<frameNodeIndex.size(); j++) {
                    if(frameNodeIndex[j]==ind) {
                        frameIndexes[i].emplace_back(j);
                    }
                }
            }
        }

        for(int i=0; i<ccs.size(); i++) {
            for(auto ind: frameIndexes[i]) {
                frameNodeIndex[ind] = i;
            }
        }
    }

    int ViewGraph::findNodeIndexByFrameIndex(int frameIndex) {
        int innerIndex = frameToInnerIndex[frameIndex];
        return frameNodeIndex[innerIndex];
    }

    int ViewGraph::updateSpanningTree() { // for last node
        // collect all edges for last node
        map<int, vector<pair<int, double>>> costsMap;
        int lastIndex = nodes.size()-1;
        for(int index=0; index<lastIndex; index++) {
            Edge& edge = (*this)(index, lastIndex);
            if(!edge.isUnreachable()) {
                int root = rootIndexes[index];
                if(!costsMap.count(root)) {
                    costsMap.insert(map<int, vector<pair<int, double>>>::value_type(root, vector<pair<int, double>>()));
                }
                costsMap[root].emplace_back(make_pair(index, edge.getCost()));
            }
        }

        // select root for last frame
        int lastRoot = -1;
        for(auto mit: costsMap) {
            if(lastRoot==-1||mit.first==curMaxRoot) {
                lastRoot = mit.first;
            }
        }

        // handle connected spanning trees
        for(auto mit: costsMap) {
            int curRoot = mit.first;
            vector<pair<int, double>>& costs = mit.second;
            int m = costs.size();

            // find cost edge from inputs and spanning tree
            vector<double> selectedCosts(2*m);
            for(int i=0; i<m; i++) {
                selectedCosts[i] = costs[i].second;
                int itp = parentIndexes[costs[i].first];
                if(itp == -1) {
                    selectedCosts[m+i] = lastRoot==costs[i].first?0:numeric_limits<double>::infinity();
                }else {
                    selectedCosts[m+i] = getEdgeCost(itp, costs[i].first);
                }
            }

            // sort the cost
            std::vector<size_t> idxs(selectedCosts.size());
            std::iota(idxs.begin(), idxs.end(), 0);
            std::sort(idxs.begin(), idxs.end(),
                      [&selectedCosts](size_t index_1, size_t index_2) { return selectedCosts[index_1] > selectedCosts[index_2]; });
            // delete m-1 edge
            vector<bool> selected(2*m, true);
            vector<bool> excluded(2*m, false);
            for(int i=0, k=0; i<2*m&&k<m-1; i++) {
                int idx = idxs[i];
                if(excluded[idx]) continue;
                selected[idx] = false; // delete
                excluded[idx] = true;
                if(idx>=m) excluded[idx-m] = true;
                else excluded[idx+m] = true;
                k++;
            }

            // compose spanning tree
            set<int> path;
            for(int i=0; i<m; i++) {
                bool f1 = selected[i];
                bool f2 = selected[m+i];
                if(f1&&f2) { // new frame edge
                    if(curRoot == lastRoot)
                        parentIndexes[lastIndex] = costs[i].first;

                    vector<int> pathSeq;
                    pathSeq.emplace_back(lastIndex);
                    path.insert(lastIndex);

                    int k  = costs[i].first;
                    do {
                        path.insert(k);
                        pathSeq.emplace_back(k);
                    }while((k=parentIndexes[k])!=-1);

                    if(curRoot != lastRoot) { // reverse root for non-main tree
                        for(int j=pathSeq.size()-1; j>0; j--) {
                            parentIndexes[pathSeq[j]] = pathSeq[j-1];
                        }
                    }
                }
            }
            for(int i=0; i<m; i++) {
                bool f1 = selected[i];
                bool f2 = selected[m+i];
                int refInnerIndex = costs[i].first;
                if(f1&&!f2&&!path.count(refInnerIndex)) { // update other frames
                    parentIndexes[refInnerIndex] = lastIndex;
                }
            }
        }

        if (!costsMap.empty()) {
            // update root
            bool endFlag = false;
            set<int> needUpdateRoot;
            needUpdateRoot.insert(lastIndex);
            nodePathLens[lastIndex] = -1;
            while(!endFlag) {
                endFlag = true;
                for(int i=0; i<parentIndexes.size(); i++) {
                    if(needUpdateRoot.count(parentIndexes[i])&&!needUpdateRoot.count(i)) {
                        needUpdateRoot.insert(i);
                        nodePathLens[i] = -1; // reset node path length
                        endFlag = false;
                    }
                }
            }

            for(int index: needUpdateRoot) {
                rootIndexes[index] = lastRoot;
                computePathLens(index);
            }

            // update visible root
            map<int, int> rootCounts;
            for(int i=0; i<rootIndexes.size(); i++) {
                int root = rootIndexes[i];
                if(!rootCounts.count(root)) {
                    rootCounts.insert(map<int, int>::value_type(root, 0));
                }else {
                    rootCounts[root]++;
                }
            }

            int maxRoot = -1;
            int maxCount = 0;
            for(auto mit: rootCounts) {
                if(mit.second>maxCount) {
                    maxRoot = mit.first;
                    maxCount = mit.second;
                }
            }

            curMaxRoot = maxRoot;

            // update transformation for node
            vector<int> cc;
            map<int, int> ccIndex;
            vector<bool> visited;
            for(int j=0; j<rootIndexes.size(); j++) {
                if(rootIndexes[j]==lastRoot) {
                    cc.emplace_back(j);
                    ccIndex.insert(map<int, int>::value_type(j, cc.size()-1));
                    visited.emplace_back(!needUpdateRoot.count(j));
                }
            }

            for(int j=0; j<cc.size(); j++) {
                computeTransform(j, ccIndex, cc, visited);
            }
        }

        int lostCount = 0;
        for(int i=0; i<rootIndexes.size(); i++) {
            nodes[i].setVisible(rootIndexes[i]==curMaxRoot);
            if(!nodes[i].isVisible()) {
                lostCount += nodes[i].getFrames().size();
            }
        }

        return lostCount;
    }

    void ViewGraph::generateSpanningTree() {
        // generate minimum spanning tree
        int n = nodes.size();
        parentIndexes.resize(n,-1);
        rootIndexes.resize(n);
        iota(rootIndexes.begin(), rootIndexes.end(), 0);

        vector<bool> fixed(n, false);
        vector<double> lowCost(n);
        for(int u=0; u<n; u++) {
            if(fixed[u]) continue;

            // initialize vectors
            for(int v=0; v<n; v++) {
                Edge edge = getEdge(u, v);
                if(!fixed[v]){
                    lowCost[v] = edge.isUnreachable()? numeric_limits<double>::infinity(): edge.getCost();
                    parentIndexes[v] = edge.isUnreachable()? -1: u;
                }
            }
            fixed[u] = true;

            // begin to update lowCost and path
            for(int k=u+1; k<n; k++) {
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
                    break;
                }
                // fix min cost index path
                int s = minIndex;
                fixed[s] = true;
                for(int v=0; v<n; v++) {
                    Edge edge = getEdge(s, v);
                    if(!edge.isUnreachable()) {
                        double cost = edge.getCost();
                        if(!fixed[v]&&lowCost[v]>cost) {
                            lowCost[v] = cost;
                            parentIndexes[v] = s;
                        }
                    }
                }
            }
        }

        // update root nodes
        map<int,int> rootCounter;
        for(int i=0; i<n; i++) {
            int u = i;
            while (parentIndexes[u]!=-1) u = parentIndexes[u];
            rootIndexes[i] = u;
            if(!rootCounter.count(u)) {
                rootCounter.insert(map<int, int>::value_type(u, 0));
            }
            rootCounter[u]++;
        }

        int maxRoot = 0;
        int maxCount = rootCounter[0];
        for(auto mit: rootCounter) {
            if(mit.second>maxCount) {
                maxCount = mit.second;
                maxRoot = mit.first;
            }
        }

        curMaxRoot = maxRoot;
    }

    int ViewGraph::getParent(int nodeIndex) {
        return parentIndexes[nodeIndex];
    }

    int ViewGraph::getPathLen(int frameIndex) {
        int nodeIndex = findNodeIndexByFrameIndex(frameIndex);
        return nodePathLens[nodeIndex];
    }


    bool ViewGraph::isVisible(int frameIndex) {
        int nodeIndex = findNodeIndexByFrameIndex(frameIndex);
        return nodes[nodeIndex].isVisible();
    }

    vector<vector<int>> ViewGraph::getConnectComponents() {
        map<int, vector<int>> ccMap;
        for(int i=0; i<rootIndexes.size(); i++) {
            int root = rootIndexes[i];
            if(!ccMap.count(root)) {
                ccMap.insert(map<int, vector<int>>::value_type(root, vector<int>()));
            }
            ccMap[root].emplace_back(i);
        }

        vector<vector<int>> ccs;
        for(auto mit: ccMap) {
            ccs.emplace_back(mit.second);
        }
        return ccs;
    }

    void ViewGraph::computeGtTransforms() {
        map<int, vector<int>> ccMap;
        for(int i=0; i<rootIndexes.size(); i++) {
            int root = rootIndexes[i];
            if(!ccMap.count(root)) {
                ccMap.insert(map<int, vector<int>>::value_type(root, vector<int>()));
            }
            ccMap[root].emplace_back(i);
        }

        auto mit = ccMap.begin();
        for(int i=0; i<ccMap.size(); i++, mit++) {
            vector<int>& cc = mit->second;
            vector<bool> visited(cc.size(), false);
            map<int, int> ccIndex;
            for(int j=0; j<cc.size(); j++) {
                ccIndex.insert(map<int, int>::value_type(cc[j], j));
            }
            for(int j=0; j<cc.size(); j++) {
                computeTransform(j, ccIndex, cc, visited);
            }
        }
    }

    bool ViewGraph::existEdge(int i, int j) {
        return !(*this)(i, j).isUnreachable();
    }

    void ViewGraph::check() {
        CHECK_EQ(nodes.size(), adjMatrix->getN());
    }

    void ViewGraph::print() {
        cout << "---------------------view graph begin------------------------------" << endl;
        // print nodes
        int n = getNodesNum();
        cout << "nodes:" << n << endl;
        for (int i = 0; i < n; i++) {
            cout << i << ": ";
            for (auto frame: nodes[i].getFrames()) {
                cout << frame->getIndex() << ",";
            }
            cout  << endl;
        }

        // print cost
        cout << "edges:" << endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                auto edge = (*this)(i, j);
                double cost = edge.isUnreachable() ? -1.0 : edge.getCost();
                int kpNum = edge.getKxs().size();
                cout << cost << "(" << kpNum << ")" << "\t";
            }
            cout << endl;
        }
        cout << "frameNodeIndex:" << endl;
        for(int i=0; i<getFramesNum(); i++) {
            cout << i << "-" << frameNodeIndex[i] << ",";
        }
        cout << endl;

        cout << "current root:" << curMaxRoot << endl;

        cout << "parents:" << endl;
        for(int i=0; i<n; i++) {
            cout << i << "-" << parentIndexes[i] << ",";
        }
        cout << endl;

        cout << "roots:" << endl;
        for(int i=0; i<n; i++) {
            cout << i << "-" << rootIndexes[i] << ",";
        }
        cout << endl;
        cout << "---------------------view graph end------------------------------" << endl;
    }

    bool edgeCompare(Edge& one, Edge& another) {
        if(one.isUnreachable()) return false;
        if(!one.isUnreachable()&&another.isUnreachable()) return true;
        float factor = 1; //(float)one.getKxs().size()/another.getKxs().size();
        return one.getCost() < factor*another.getCost();
    }
}

