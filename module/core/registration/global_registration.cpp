//
// Created by liulei on 2020/8/4.
//

#include <utility>

#include "registrations.h"
#include "../../tool/view_graph_util.h"
#include "optimizer.h"

using namespace rtf::ViewGraphUtil;

namespace rtf {
    void GlobalRegistration::registrationPnPBA(FeatureMatches *featureMatches, Edge *edge, cudaStream_t curStream) {
        RANSAC2DReport pnp = pnpRegistration->registrationFunction(*featureMatches);
        RegReport ba;
        if (pnp.success) {
            BARegistration baRegistration(globalConfig);
            vector<FeatureKeypoint> kxs, kys;
            if(pnp.inliers.size()<100) {
                FeatureMatches matches = matcher->matchKeyPointsWithProjection(featureMatches->getFp1(), featureMatches->getFp2(), pnp.T);
                featureMatchesToPoints(matches, kxs, kys);

                ba = baRegistration.bundleAdjustment(pnp.T, matches.getCx(), matches.getCy(), kxs, kys, true);
            }else {
                featureIndexesToPoints(featureMatches->getKx(), pnp.kps1, kxs);
                featureIndexesToPoints(featureMatches->getKy(), pnp.kps2, kys);

                ba = baRegistration.bundleAdjustment(pnp.T, featureMatches->getCx(), featureMatches->getCy(), kxs, kys);
            }

            if (ba.success) {
                double cost = ba.avgCost();
                if (!isnan(cost) && cost < globalConfig.maxAvgCost) {
                    edge->setKxs(kxs);
                    edge->setKys(kys);
                    edge->setTransform(ba.T);
                    edge->setCost(cost);
                }
            }
        }

        printMutex.lock();
        cout << "-------------------" << featureMatches->getFIndexX() << "-" << featureMatches->getFIndexY()  << "-global-pnp+ba---------------------------------" << endl;
        pnp.printReport();
        ba.printReport();
        printMutex.unlock();
    }

    void GlobalRegistration::registrationPairEdge(FeatureMatches featureMatches, Edge *edge, cudaStream_t curStream) {
        stream = curStream;
        registrationPnPBA(&featureMatches, edge, curStream);

        if(featureMatches.getFIndexY()-featureMatches.getFIndexX()==globalConfig.chunkSize) { // registration
            shared_ptr<KeyFrame> ref = viewGraph.indexFrame(featureMatches.getFIndexX());
            shared_ptr<KeyFrame> cur = viewGraph.indexFrame(featureMatches.getFIndexY());
            vector<shared_ptr<Frame>> refFrames = ref->getFrames();
            int lastVisPos = refFrames.size()-1;
            for(; lastVisPos>0&&!refFrames[lastVisPos]->isVisible(); lastVisPos--);
            shared_ptr<Frame> lastRefFrame = refFrames[lastVisPos];
            shared_ptr<Frame> curFrame = cur->getFirstFrame();
            FeatureMatches frameMatches = matcher->matchKeyPointsPair(lastRefFrame->getKps(), curFrame->getKps());
            Edge frameEdge = Edge::UNREACHABLE;
            registrationPnPBA(&frameMatches, &frameEdge, curStream);

            if(!frameEdge.isUnreachable()&&(edge->isUnreachable()||(edge->getKxs().size()<frameEdge.getKxs().size()&&edge->getCost()>frameEdge.getCost()))) {
                Transform transX = lastRefFrame->getTransform();

                Intrinsic kX = viewGraph[0].getK();

                // transform and k: p' = K(R*K^-1*p+t)
                Rotation rX = kX*transX.block<3,3>(0,0)*kX.inverse();
                Translation tX = kX*transX.block<3,1>(0,3);

                // transform key points
                transformFeatureKeypoints(frameEdge.getKxs(), rX, tX);

                // trans12 = trans1*relative_trans*trans2^-1
                Transform relativeTrans = transX * frameEdge.getTransform();
                edge->setTransform(relativeTrans);
                edge->setKxs(frameEdge.getKxs());
                edge->setKys(frameEdge.getKys());
                edge->setCost(frameEdge.getCost());
            }
        }
    }

    bool detectLoop(set<pair<int, int> >& candidates) {
        bool c1 = candidates.size()>=2;

        int delta = 0;
        int count = 0;
        int lastIndex = -1;
        for(auto sit: candidates) {
            if(lastIndex!=-1) {
                delta += abs(sit.second-lastIndex);
                count ++;
            }
            lastIndex = sit.second;
        }
        bool c2 = delta/max(count, 1) < 10;

        return c1&&c2;
    }

    bool GlobalRegistration::loopClosureDetection() {
        set<pair<int, int>> candidates = ViewGraphUtil::findLoopEdges(viewGraph, viewGraph.getNodesNum() - 1);
        if(!candidates.empty()) {
            loopCandidates.insert(candidates.begin(), candidates.end());
        }else {
            if(detectLoop(loopCandidates)) {
                cout << "loop candidates" << endl;
                for(auto sit: loopCandidates) {
                    cout << sit.first << ", " << sit.second << endl;
                }
                cout << "--------------------------------------------" << endl;

                cout << "loop closure detected!!!" << endl;
                loops.insert(loops.end(), loopCandidates.begin(), loopCandidates.end());
                Optimizer::poseGraphOptimizeCeres(viewGraph, loops);
                loopCandidates.clear();
                return true;
            }
            loopCandidates.clear();
        }
        return false;
    }

    void GlobalRegistration::registrationEdges(shared_ptr<KeyFrame> cur, vector<int>& overlapFrames, vector<int>& innerIndexes, EigenVector(Edge)& edges) {
        const int k = globalConfig.overlapNum;
        const int lastNum = 1;
        const int n = viewGraph.getFramesNum()-1;
        vector<shared_ptr<KeyFrame>> frames = viewGraph.getSourceFrames();
        DBoW2::BowVector& bow = cur->getKps().getMBowVec();

        set<int> spAlreadyAddedKF;
        overlapFrames.reserve(k);
        edges.resize(k, Edge::UNREACHABLE);
        thread *threads[k];
        cudaStream_t streams[k];
        // last frame
        int index = 0;
        for (int i = 1; i <= lastNum && i <= k && i < n; i++) {
            shared_ptr<KeyFrame> refKeyFrame = frames[n - i];
            int refIndex = refKeyFrame->getIndex();
            spAlreadyAddedKF.insert(refIndex);
            overlapFrames.emplace_back(refIndex);

            FeatureMatches featureMatches = matcher->matchKeyPointsPair(refKeyFrame->getKps(),
                                                                        cur->getKps());
            cudaStreamCreate(&streams[index]);
            threads[index] = new thread(
                    bind(&GlobalRegistration::registrationPairEdge, this, placeholders::_1, placeholders::_2,
                         placeholders::_3), featureMatches, &edges[index], streams[index]);
            index++;
        }

        if (overlapFrames.size() < k) {
            Timer queryTimer = Timer::startTimer("query index");
            std::vector<MatchScore> imageScores = dBoWHashing->queryImages(lastPos, cur->getKps(), notLost,  lostNum > 0);
            queryTimer.stopTimer();
            // Return all those keyframes with a score higher than 0.75*bestScore
            for (auto it: imageScores) {
                const float &si = it.score;
                int refIndex = it.imageId;
                if (!spAlreadyAddedKF.count(refIndex)) {
                    shared_ptr<KeyFrame> refKeyFrame = viewGraph.indexFrame(refIndex);
                    spAlreadyAddedKF.insert(refIndex);
                    overlapFrames.emplace_back(refIndex);

                    FeatureMatches featureMatches = matcher->matchKeyPointsPair(refKeyFrame->getKps(),
                                                                                cur->getKps());
                    overlapFrames.emplace_back(refIndex);
                    spAlreadyAddedKF.insert(refIndex);
                    cudaStreamCreate(&streams[index]);
                    threads[index] = new thread(
                            bind(&GlobalRegistration::registrationPairEdge, this, placeholders::_1, placeholders::_2,
                                 placeholders::_3), featureMatches, &edges[index], streams[index]);
                    index++;
                }
                if (overlapFrames.size() >= k) break;
            }
        }

        for (int i = 0; i < index; i++) {
            threads[i]->join();
            CUDA_CHECKED_CALL(cudaStreamSynchronize(streams[i]));
            CUDA_CHECKED_CALL(cudaStreamDestroy(streams[i]));
        }
    }

    GlobalRegistration::GlobalRegistration(const GlobalConfig &globalConfig, SIFTVocabulary* siftVocabulary): siftVocabulary(siftVocabulary), globalConfig(globalConfig) {
        dBoWHashing = new DBoWHashing(globalConfig, siftVocabulary, true);
        matcher = new SIFTFeatureMatcher();
        pnpRegistration = new PnPRegistration(globalConfig);
    }

    bool GlobalRegistration::mergeViewGraph() {
        viewGraph.check();
        cout << "merge view graph..." << endl;
        // find connected component from view graph
        vector<vector<int>> connectedComponents = findConnectedComponents(viewGraph, globalConfig.costThreshold);
        // it is unnecessary to update again if the size of connected components is equals to the nodes of graph
        int n = connectedComponents.size();

        if (n == viewGraph.getNodesNum()) return false;
        // initialize new view graph
        NodeVector nodes(n);
        EigenUpperTriangularMatrix<Edge> adjMatrix(n, Edge::UNREACHABLE);
        // merge nodes and transformation
        for (int i = 0; i < n; i++) {
            if (connectedComponents[i].size() > 1) {
                cout << "multiview" << endl;
                BARegistration bundleAdjustment(globalConfig);
                bundleAdjustment.multiViewBundleAdjustment(viewGraph, connectedComponents[i]).printReport();
                mergeComponentNodes(viewGraph, connectedComponents[i], nodes[i]);
            } else {
                nodes[i] = viewGraph[connectedComponents[i][0]];
            }

        }
        for (int i = 0; i < n; i++) {
            vector<int> cc1 = connectedComponents[i];
            for (int j = i + 1; j < n; j++) {
                vector<int> cc2 = connectedComponents[j];
                adjMatrix(i, j) = selectEdgeBetweenComponents(viewGraph, cc1, cc2);
            }
        }

        // update view graph
        viewGraph.setNodesAndEdges(nodes, adjMatrix);
        viewGraph.updateNodeIndex(connectedComponents);
    }

    void GlobalRegistration::updateLostFrames() {
        vector<int> lostImageIds = dBoWHashing->lostImageIds();
        vector<int> updateIds;
        vector<float3> poses;
        for (int lostId: lostImageIds) {
            int nodeIndex = viewGraph.findNodeIndexByFrameIndex(lostId);
            if (viewGraph[nodeIndex].isVisible()) {
                updateIds.emplace_back(lostId);

                Transform trans = viewGraph.getFrameTransform(lostId);
                Vector3 ow;
                GeoUtil::computeOW(trans, ow);
                poses.emplace_back(make_float3(ow.x(), ow.y(), ow.z()));
            }
        }

        dBoWHashing->updateVisualIndex(updateIds, poses);
    }

    void GlobalRegistration::insertKeyFrames(shared_ptr<KeyFrame> keyframe) {
        siftVocabulary->computeBow(keyframe->getKps());
        viewGraph.extendNode(keyframe);

        if(viewGraph.getFramesNum()>1) {
            vector<int> overlapFrames;
            vector<int> innerIndexes;
            vector<Edge, Eigen::aligned_allocator<Edge>> pairEdges;
            registrationEdges(keyframe, overlapFrames, innerIndexes, pairEdges);

            map<int, int> bestEdgeMap;
            for(int i=0; i<overlapFrames.size(); i++) {
                if (!pairEdges[i].isUnreachable()) {
                    int refNodeIndex = viewGraph.findNodeIndexByFrameIndex(overlapFrames[i]);
                    if (bestEdgeMap.count(refNodeIndex)) {
                        double bestCost = pairEdges[bestEdgeMap[refNodeIndex]].getCost();
                        if (pairEdges[i].getCost() < bestCost) {
                            bestEdgeMap[refNodeIndex] = i;
                        }
                    } else {
                        bestEdgeMap.insert(map<int, int>::value_type(refNodeIndex, i));
                    }
                }
            }

            int curNodeIndex = viewGraph.getNodesNum() - 1;
            for (auto mit: bestEdgeMap) {
                int refNodeIndex = mit.first;
                int ind = mit.second;
                Edge &edge = viewGraph(refNodeIndex, curNodeIndex);
                Edge &bestEdge = pairEdges[ind];

                if (edgeCompare(bestEdge, edge)) {
                    edge.setKxs(bestEdge.getKxs());
                    edge.setKys(bestEdge.getKys());
                    edge.setTransform(bestEdge.getTransform());
                    edge.setCost(bestEdge.getCost());
                }

            }

            loopClosureDetection();

            if(viewGraph.getFramesNum()%globalConfig.chunkSize==0) {
                cout << "local optimization" << endl;
                vector<int> cc(viewGraph.getFramesNum());
                iota(cc.begin(), cc.end(), 0);
                BARegistration baRegistration(globalConfig);
                baRegistration.multiViewBundleAdjustment(viewGraph, cc).printReport();
            }
            Timer grTimer = Timer::startTimer("gr");
            lostNum = viewGraph.updateSpanningTree();
            grTimer.stopTimer();
            updateLostFrames();

            curNodeIndex = viewGraph.findNodeIndexByFrameIndex(keyframe->getIndex());
            if (viewGraph[curNodeIndex].isVisible()) {
                Transform trans = viewGraph.getFrameTransform(keyframe->getIndex());
                Vector3 ow;
                GeoUtil::computeOW(trans, ow);
                lastPos = make_float3(ow.x(), ow.y(), ow.z());
            }
            notLost = viewGraph[curNodeIndex].isVisible();
        }else {
            lastPos = make_float3(0, 0, 0);
            notLost = true;
        }

        dBoWHashing->addVisualIndex(lastPos, keyframe->getKps(), keyframe->getIndex(),  notLost);
    }

    int GlobalRegistration::registration(bool opt) {
        int n = viewGraph.getNodesNum();
        if(n<1) return true;

        cout << "------------------------compute global transform for view graph------------------------" << endl;
        vector<vector<int>> ccs = viewGraph.getConnectComponents();
//        viewGraph.computeGtTransforms();

        for(int i=0; i<ccs.size(); i++) {
            vector<int>& cc = ccs[i];
            if(cc.size()>1) {
                if(opt) {
                    BARegistration baRegistration(globalConfig);
                    baRegistration.multiViewBundleAdjustment(viewGraph, cc).printReport();
                }
            }
        }
        cout << "invisible count:" << lostNum << endl;
        return lostNum;
    }

    ViewGraph &GlobalRegistration::getViewGraph() {
        return viewGraph;
    }

    GlobalRegistration::~GlobalRegistration() {
        delete matcher;
        delete dBoWHashing;
        delete pnpRegistration;
    }
}

