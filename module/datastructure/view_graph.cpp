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

    ConnectionCandidate::ConnectionCandidate(bool unreachable) : unreachable(unreachable) {}


    ConnectionCandidate::ConnectionCandidate() : ConnectionCandidate(true) {}

    ConnectionCandidate ConnectionCandidate::UNREACHABLE(true);

    void ConnectionCandidate::setTransform(Transform transformation) {
        this->transform = SE3(transformation);
    }

    void ConnectionCandidate::setCost(double cost) {
        this->cost = cost;
        this->unreachable = false;
    }


    vector<FeatureKeypoint> &ConnectionCandidate::getKxs() {
        return kxs;
    }

    void ConnectionCandidate::setKxs(const vector<FeatureKeypoint> &kxs) {
        ConnectionCandidate::kxs = kxs;
    }

    vector<FeatureKeypoint> &ConnectionCandidate::getKys() {
        return kys;
    }

    void ConnectionCandidate::setKys(const vector<FeatureKeypoint> &kys) {
        ConnectionCandidate::kys = kys;
    }

    Transform ConnectionCandidate::getTransform() {
        return transform.matrix();
    }

    SE3 ConnectionCandidate::getSE() {
        return transform;
    }

    double ConnectionCandidate::getCost() {
        return cost;
    }

    bool ConnectionCandidate::isUnreachable() {
        return unreachable;
    }

    Connection::Connection() {}

    Connection::Connection(shared_ptr<ViewCluster> v, Vector3 p, float pointWeight, SE3 transform, double cost) : v(v), p(p), pointWeight(pointWeight), transform(transform), cost(cost) {}

    shared_ptr<ViewCluster> Connection::getViewCluster() {
        return v;
    }

    Vector3 Connection::getNormPoint() {
        return p;
    }

    float Connection::getPointWeight() {
        return pointWeight;
    }

    Transform Connection::getTransform() {
        return transform.matrix();
    }

    SE3 Connection::getSE() {
        return transform;
    }

    double Connection::getCost() {
        return cost;
    }

    void Connection::setViewCluster(const shared_ptr<ViewCluster> &v) {
        Connection::v = v;
    }

    void Connection::setNormPoint(const Vector3 &p) {
        Connection::p = p;
    }

    void Connection::setPointWeight(float pointWeight) {
        Connection::pointWeight = pointWeight;
    }

    void Connection::setTransform(const Transform &transform) {
        Connection::transform = SE3(transform);
    }

    void Connection::setSE(const SE3 &transform) {
        Connection::transform = transform;
    }

    void Connection::setCost(double cost) {
        Connection::cost = cost;
    }

    ViewCluster::ViewCluster(){
        transform = SE3(Transform::Identity());
        rootIndex = 0;
    }

    int ViewCluster::getIndex() {
        return frames[0]->getFrameIndex();
    }

    shared_ptr<Camera> ViewCluster::getCamera() {
        return frames[0]->getCamera();
    }

    Transform ViewCluster::getTransform() {
        return transform.matrix();
    }

    Transform ViewCluster::getFrameTransform(int frameIndex) {
        return getFrame(frameIndex)->getTransform();
    }

    void ViewCluster::setTransform(Transform trans) {
        transform = SE3(trans);
    }

    SE3 ViewCluster::getSE() {
        return transform;
    }

    void ViewCluster::setSE(SE3 se3) {
        transform = se3;
    }

    Intrinsic ViewCluster::getK() {
        return getCamera()->getK();
    }

    void ViewCluster::addFrame(shared_ptr<Frame> frame, int pathLength) {
        indexToInnerMap.insert(map<int, int>::value_type(frame->getFrameIndex(), frames.size()));
        frames.emplace_back(frame);
        pathLengths.emplace_back(pathLength);
    }

    int ViewCluster::getPathLength(int frameIndex) {
        int innerIndex = indexToInnerMap[frameIndex];
        return pathLengths[innerIndex];
    }

    vector<shared_ptr<Frame>> &ViewCluster::getFrames() {
        return frames;
    }

    void ViewCluster::setRootIndex(int index) {
        rootIndex = index;
    }

    shared_ptr<Frame> ViewCluster::getRootFrame() {
        return frames[rootIndex];
    }

    shared_ptr<Frame> ViewCluster::getFrame(int frameIndex) {
        int innerIndex = indexToInnerMap[frameIndex];
        return frames[innerIndex];
    }

    bool ViewCluster::existConnection(int v) {
        return connections.count(v);
    }

    shared_ptr<Connection> ViewCluster::getConnection(int v) {
        assert(existConnection(v));
        return connections[v];
    }

    void ViewCluster::addConnection(int v, shared_ptr<Connection> con) {
        connections.insert(map<int, shared_ptr<Connection>>::value_type(v, con));
    }

    vector<shared_ptr<Connection>> ViewCluster::getConnections() {
        vector<shared_ptr<Connection>> conVec;
        for(auto mit: connections) {
            conVec.emplace_back(mit.second);
        }
        return conVec;
    }

    map<int, shared_ptr<Connection>> ViewCluster::getConnectionMap() {
        return connections;
    }

    void ViewCluster::setVisible(bool visible) {
        this->visible = visible;
    }

    bool ViewCluster::isVisible() {
        return visible;
    }

    SIFTFeaturePoints &ViewCluster::getKps() {
        return kps;
    }

    void ViewCluster::setKps(const SIFTFeaturePoints &kps) {
        ViewCluster::kps = kps;
    }

    ViewGraph::ViewGraph(): ViewGraph(0) {

    }

    ViewGraph::~ViewGraph() {

    }

    ViewGraph::ViewGraph(int nodesNum) {
        reset(nodesNum);
    }

    void ViewGraph::reset(int nodesNum) {
        if(nodesNum <= 0) {
            sourceNodes.clear();
            frameNodeIndex.clear();
            parentIndexes.clear();
            rootIndexes.clear();
            curMaxRoot = 0;
        }else {
            sourceNodes.resize(nodesNum);
            parentIndexes.resize(nodesNum, -1);
            rootIndexes.resize(nodesNum);
            iota(rootIndexes.begin(), rootIndexes.end(), 0);
            curMaxRoot = 0;
        }
    }

    int ViewGraph::getNodesNum() {
        return sourceNodes.size();
    }

    Transform ViewGraph::getViewTransform(int frameIndex) {
        int innerIndex = frameToInnerIndex[frameIndex];
        return sourceNodes[innerIndex]->getTransform();
    }

    SE3 ViewGraph::computeTransform(int u, map<int, int>& innerMap, vector<int>& cc, vector<bool>& visited) {
        int nodeIndex = cc[u];
        if(visited[u]) return sourceNodes[nodeIndex]->getSE();
        int parent = parentIndexes[cc[u]];
        if(parent==-1) {
            sourceNodes[nodeIndex]->setTransform(Transform::Identity());
        }else {
            int v = innerMap[parent];
            SE3 trans = computeTransform(v, innerMap, cc, visited)*getEdgeSE(parent, nodeIndex);
            sourceNodes[nodeIndex]->setSE(trans);
        }
        visited[u] = true;
        return sourceNodes[nodeIndex]->getSE();
    }

    int ViewGraph::computePathLens(int index) {
        if(index == -1) return -1;
        assert(index<getNodesNum()&&index>=0);
        if(nodePathLens[index]>=0) return nodePathLens[index];
        nodePathLens[index] = computePathLens(parentIndexes[index])+1;
        return nodePathLens[index];
    }

    shared_ptr<Connection> ViewGraph::operator()(int i, int j) {
        shared_ptr<ViewCluster> view = (*this)[i];
        LOG_ASSERT(view->existConnection(j)) << " connection no exist: " << to_string(i) << "->" << to_string(j) << endl;
        return view->getConnection(j);
    }

    shared_ptr<ViewCluster> ViewGraph::operator[](int index) {
        LOG_ASSERT(index<getNodesNum()) << " view no exist: " << to_string(index) << endl;
        return sourceNodes[index];
    }

    void ViewGraph::extendNode(shared_ptr<ViewCluster> node) {
        sourceNodes.emplace_back(node);
        node->setTransform(Transform::Identity());

        frameNodeIndex.emplace_back(sourceNodes.size()-1);
        frameToInnerIndex.insert(map<int, int>::value_type(node->getIndex(), sourceNodes.size() - 1));
        parentIndexes.emplace_back(-1);
        rootIndexes.emplace_back(sourceNodes.size()-1);
        nodePathLens.emplace_back(0);
    }

    double ViewGraph::getEdgeCost(int i, int j) {
        return (*this)(i,j)->getCost();
    }

    SE3 ViewGraph::getEdgeSE(int i, int j) {
        return (*this)(i, j)->getSE();
    }

    int ViewGraph::findNodeIndexByFrameIndex(int frameIndex) {
        assert(frameToInnerIndex.count(frameIndex));
        int innerIndex = frameToInnerIndex[frameIndex];
        return frameNodeIndex[innerIndex];
    }

    shared_ptr<ViewCluster> ViewGraph::findNodeByFrameIndex(int frameIndex) {
        return sourceNodes[findNodeIndexByFrameIndex(frameIndex)];
    }

    int ViewGraph::updateSpanningTree() { // for last node
        // collect all edges for last node
        map<int, vector<pair<int, double>>> costsMap;
        int lastIndex = sourceNodes.size()-1;
        shared_ptr<ViewCluster> lastNode = (*this)[lastIndex];
        for(auto& cit: lastNode->getConnectionMap()) {
            int root = rootIndexes[cit.first];
            if(!costsMap.count(root)) {
                costsMap.insert(map<int, vector<pair<int, double>>>::value_type(root, vector<pair<int, double>>()));
            }
            costsMap[root].emplace_back(make_pair(cit.first, cit.second->getCost()));
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
            sourceNodes[i]->setVisible(rootIndexes[i]==curMaxRoot);
            if(!sourceNodes[i]->isVisible()) {
                lostCount += sourceNodes[i]->getFrames().size();
            }
        }

        return lostCount;
    }

    shared_ptr<Camera> ViewGraph::getCamera() {
        return (*this)[0]->getCamera();
    }

    int ViewGraph::getParent(int nodeIndex) {
        return parentIndexes[nodeIndex];
    }

    int ViewGraph::getPathLen(int frameIndex) {
        int nodeIndex = findNodeIndexByFrameIndex(frameIndex);
        assert(nodePathLens[nodeIndex]<=getNodesNum());
        return nodePathLens[nodeIndex];
    }


    bool ViewGraph::isVisible(int frameIndex) {
        int nodeIndex = findNodeIndexByFrameIndex(frameIndex);
        return sourceNodes[nodeIndex]->isVisible();
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

    vector<int> ViewGraph::getBestCovisibilityNodes(int index, int k) {
        typedef std::pair<int, int> WeightedView;

        std::vector<WeightedView> weighted_views;
        auto connections = (*this)[index]->getConnectionMap();
        for (auto& cit: connections) {
            weighted_views.push_back( std::make_pair( (int)(*this)[cit.first]->getConnections().size(), cit.first));
        }

        // sort by the number of connections in descending order
        std::sort(weighted_views.begin(), weighted_views.end(),
                  [](const WeightedView &x, const WeightedView &y) //TODO: fix x is becames null??? why???
                  {
                      return x.first > y.first;
                  }
        );

        std::vector<int> covisibility;
        covisibility.reserve(k);

        int i = 0;
        for (auto wv: weighted_views) {
            if (i++==k) break;

            covisibility.push_back(wv.second);
        }

        return covisibility;
    }

    int ViewGraph::getMaxRoot() {
        return curMaxRoot;
    }

    void ViewGraph::optimizeBestRootNode() {
        vector<int> cc = maxConnectedComponent();

        int m = cc.size();
        vector<int> nodeNums(getNodesNum(), 0);
        for(int ind: cc) {
            for(int p=ind; p!=-1; p=getParent(p)) nodeNums[p]++;
        }

        int maxPathLen = 0;
        int maxPathIndex = 0;
        for(int i=0; i<m; i++) {
            // compute path length in MST
            int pathLen = nodePathLens[cc[i]];
            int curPathLen = 0;
            for(int j=0, p=cc[i], q=0; p!=-1; q=nodeNums[p], p=getParent(p), j++) {
                curPathLen += (nodeNums[p]-q)*(nodePathLens[p]-j);
            }
            if(curPathLen > maxPathLen) {
                maxPathLen = curPathLen;
                maxPathIndex = cc[i];
            }
        }

        vector<int> path;
        for(int p=maxPathIndex; p!=-1; p=getParent(p)) {
            path.emplace_back(p);
        }

        for(int i=path.size()-1; i>0; i--) {
            parentIndexes[path[i]] = path[i-1];
        }
        parentIndexes[path[0]] = -1;
        curMaxRoot = maxPathIndex;

        // update root indexes
        for(int i=0; i<m; i++) {
            int pathLen = 0, p = cc[i], q = getParent(p);
            for(; q!=-1; p=q, q=getParent(p)) pathLen++;
            rootIndexes[cc[i]] = p;
            nodePathLens[cc[i]] = pathLen;
        }

        // compute transformations
        vector<bool> visited(cc.size(), false);
        map<int, int> ccIndex;
        for(int j=0; j<cc.size(); j++) {
            ccIndex.insert(map<int, int>::value_type(cc[j], j));
        }
        for(int j=0; j<cc.size(); j++) {
            computeTransform(j, ccIndex, cc, visited);
        }
    }

    vector<int> ViewGraph::maxConnectedComponent() {
        vector<int> cc;
        for(int i=0; i<getNodesNum(); i++) {
            if(rootIndexes[i]==curMaxRoot) {
                cc.emplace_back(i);
            }
        }
        return cc;
    }

    bool ViewGraph::existEdge(int i, int j) {
        shared_ptr<ViewCluster> view = (*this)[i];
        return view->existConnection(j);
    }

    void ViewGraph::print() {
        cout << "---------------------view graph begin------------------------------" << endl;
        // print nodes
        int n = getNodesNum();
        cout << "nodes:" << n << endl;
        for (int i = 0; i < n; i++) {
            cout << i << ": ";
            for (auto frame: sourceNodes[i]->getFrames()) {
                cout << frame->getFrameIndex() << ",";
            }
            cout  << endl;
            cout << "  |   ";
            for (auto con: sourceNodes[i]->getConnections()) {
                int index = con->getViewCluster()->getIndex();
                cout << "(" << findNodeIndexByFrameIndex(index) << " : " << index << "-" << con->getCost() << "-" << con->getPointWeight()<< "), ";
            }
            cout  << endl;
        }

        cout << "frameNodeIndex:" << endl;
        for(int i=0; i<getNodesNum(); i++) {
            cout << i << "-" << frameNodeIndex[i] << ",";
        }
        cout << endl;

        cout << "current root:" << curMaxRoot << endl;

        cout << "pathLens:" << endl;
        for(int i=0; i<n; i++) {
            cout << i << "-" << nodePathLens[i] << ",";
        }
        cout << endl;

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
}

