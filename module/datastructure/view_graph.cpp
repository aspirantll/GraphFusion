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
        transform.setIdentity();
    }

    int KeyFrame::getIndex() {
        return frames[0]->getFrameIndex();
    }

    shared_ptr<Camera> KeyFrame::getCamera() {
        return frames[0]->getCamera();
    }

    Transform KeyFrame::getTransform() {
        return transform;
    }

    Transform KeyFrame::getTransform(int frameIndex) {
        return getFrame(frameIndex)->getTransform();
    }

    void KeyFrame::setTransform(Transform trans) {
        transform = std::move(trans);
    }

    Intrinsic KeyFrame::getK() {
        return frames[0]->getCamera()->getK();
    }

    void KeyFrame::addFrame(shared_ptr<Frame> frame) {
        indexToInnerMap.insert(map<int, int>::value_type(frame->getFrameIndex(), frames.size()));
        frames.emplace_back(frame);
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

    Transform Edge::getTransform() {
        return transformation;
    }

    double Edge::getCost() {
        return cost;
    }

    Edge Edge::reverse() {
        Edge edge(this->unreachable);
        if(!edge.isUnreachable()) {
            edge.setKxs(kys);
            edge.setKys(kxs);
            edge.transformation << GeoUtil::reverseTransformation(transformation);
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
        this->transformation << transformation;
    }

    void Edge::setCost(double cost) {
        this->cost = cost;
        this->unreachable = false;
    }

    YAML::Node Edge::serialize() {
        YAML::Node node;
        node["transformation"] = YAMLUtil::matrixSerialize(transformation);
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
        return frames[0]->getK();
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

    void ViewGraph::setNodesAndEdges(NodeVector &nodes, EigenUpperTriangularMatrix<rtf::Edge> &adjMatrix) {
        this->nodes = nodes;
        *(this->adjMatrix) = adjMatrix;
    }


    void ViewGraph::reset(int nodesNum, Edge defaultValue) {
        if(nodesNum <= 0) {
            nodes.clear();
            adjMatrix->resize(0, defaultValue);
            sourceFrames.clear();
            frameNodeIndex.clear();
        }else {
            nodes.resize(nodesNum, Node());
            adjMatrix->resize(nodesNum, defaultValue);
        }
    }

    int ViewGraph::getNodesNum() {
        return nodes.size();
    }

    Transform ViewGraph::getFrameTransform(int frameIndex) {
        int innerIndex = frameToInnerIndex[frameIndex];
        int nodeIndex = frameNodeIndex[innerIndex];
        return nodes[nodeIndex].nGtTrans*sourceFrames[innerIndex]->getTransform();
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
        node.nGtTrans.setIdentity();
        node.status = 1;

        nodes.emplace_back(node);
        adjMatrix->extend();

        sourceFrames.emplace_back(frame);
        frameNodeIndex.emplace_back(nodes.size()-1);
        frameToInnerIndex.insert(map<int, int>::value_type(frame->getIndex(), sourceFrames.size()-1));
        return nodes[nodes.size()-1];
    }

    shared_ptr<KeyFrame> ViewGraph::indexFrame(int index) {
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


    bool ViewGraph::isVisible(int frameIndex) {
        int nodeIndex = findNodeIndexByFrameIndex(frameIndex);
        return nodes[nodeIndex].isVisible();
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
        cout << "---------------------view graph end------------------------------" << endl;
    }

    bool edgeCompare(Edge& one, Edge& another) {
        if(one.isUnreachable()) return false;
        if(!one.isUnreachable()&&another.isUnreachable()) return true;
        /*if(fabs(one.getCost()-another.getCost())<0.3) {
            return one.getKxs().size() > another.getKys().size();
        }*/
        return one.getCost() < another.getCost();
    }
}

