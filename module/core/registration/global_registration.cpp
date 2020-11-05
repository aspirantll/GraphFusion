//
// Created by liulei on 2020/8/4.
//

#include <utility>

#include "registrations.h"
#include "../../tool/view_graph_util.h"

using namespace rtf::ViewGraphUtil;

namespace rtf {
    void GlobalRegistration::registrationEGBA(FeatureMatches *featureMatches, Edge *edge, cudaStream_t curStream) {
        stream = curStream;
        RANSAC2DReport eg = egRegistration->registrationFunction(*featureMatches);
        BAReport ba;
        if (eg.success) {
            vector<FeatureKeypoint> kxs, kys;
            featureIndexesToPoints(featureMatches->getKx(), eg.kps1, kxs);
            featureIndexesToPoints(featureMatches->getKy(), eg.kps2, kys);
            BARegistration baRegistration(globalConfig);
            ba = baRegistration.bundleAdjustment(eg.T, featureMatches->getCx(), featureMatches->getCy(), kxs, kys, true);
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
        cout << "-------------------" << featureMatches->getFIndexX() << "-" << featureMatches->getFIndexY()  << "-global-eg+ba---------------------------------" << endl;
        eg.printReport();
        if (ba.success) {
            ba.printReport();
        }
        printMutex.unlock();
    }

    void GlobalRegistration::registrationHomoBA(FeatureMatches *featureMatches, Edge *edge, cudaStream_t curStream) {
        stream = curStream;
        RANSAC2DReport homo = homoRegistration->registrationFunction(*featureMatches);
        BAReport ba;
        if (homo.success) {
            vector<FeatureKeypoint> kxs, kys;
            featureIndexesToPoints(featureMatches->getKx(), homo.kps1, kxs);
            featureIndexesToPoints(featureMatches->getKy(), homo.kps2, kys);

            BARegistration baRegistration(globalConfig);
            ba = baRegistration.bundleAdjustment(homo.T, featureMatches->getCx(), featureMatches->getCy(), kxs, kys, true);
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
        cout << "-------------------" << featureMatches->getFIndexX() << "-" << featureMatches->getFIndexY()  << "-global-homo+ba---------------------------------" << endl;
        homo.printReport();
        if (ba.success) {
            ba.printReport();
        }
        printMutex.unlock();
    }

    void GlobalRegistration::registrationPnPBA(FeatureMatches *featureMatches, Edge *edge, cudaStream_t curStream) {
        stream = curStream;
        RANSAC2DReport pnp = pnpRegistration->registrationFunction(*featureMatches);
        BAReport ba;
        if (pnp.success) {
            vector<FeatureKeypoint> kxs, kys;
            featureIndexesToPoints(featureMatches->getKx(), pnp.kps1, kxs);
            featureIndexesToPoints(featureMatches->getKy(), pnp.kps2, kys);

            BARegistration baRegistration(globalConfig);
            ba = baRegistration.bundleAdjustment(pnp.T, featureMatches->getCx(), featureMatches->getCy(), kxs, kys);
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
        if (ba.success) {
            ba.printReport();
        }
        printMutex.unlock();
    }

    void GlobalRegistration::registrationPairEdge(FeatureMatches featureMatches, Edge *edge, cudaStream_t curStream, bool near) {
        stream = curStream;
        registrationPnPBA(&featureMatches, edge, curStream);
        if(near&&edge->isUnreachable()) {
            registrationEGBA(&featureMatches, edge, curStream);
            if(edge->isUnreachable()) {
                registrationHomoBA(&featureMatches, edge, curStream);
            }
        }
    }

    void GlobalRegistration::registrationEdges(shared_ptr<KeyFrame> cur, vector<int>& overlapFrames, EigenVector(Edge)& edges) {
        const int k = globalConfig.overlapNum;
        const int lastNum = 1;
        const int n = viewGraph.getFramesNum();
        auto frames = viewGraph.getSourceFrames();

        set<int> spAlreadyAddedKF;
        overlapFrames.reserve(k);
        edges.resize(k, Edge::UNREACHABLE);
        thread *threads[k];
        cudaStream_t streams[k];
        // last frame
        int index = 0;
        for (int i = 1; i <= lastNum && i <= k && i <= n; i++) {
            shared_ptr<KeyFrame> ref = frames[n-i];
            int refIndex = ref->getIndex();
            spAlreadyAddedKF.insert(refIndex);
            overlapFrames.emplace_back(refIndex);

            FeatureMatches featureMatches = matcher->matchKeyPointsPair(ref->getKps(),
                                                                        cur->getKps());
            cudaStreamCreate(&streams[index]);
            threads[index] = new thread(
                    bind(&GlobalRegistration::registrationPairEdge, this, placeholders::_1, placeholders::_2,
                         placeholders::_3, placeholders::_4), featureMatches, &edges[index], streams[index], true);
            index++;
        }

        if (overlapFrames.size() < k) {
            std::vector<MatchScore> imageScores = dBoWHashing->queryImages(lastPos, cur->getKps(), notLost,  lostNum > 0);
            // Return all those keyframes with a score higher than 0.75*bestScore
            float minScoreToRetain = globalConfig.minScore;
            std::sort(imageScores.begin(), imageScores.end(), [=](MatchScore& ind1, MatchScore& ind2) {return ind1.imageId < ind2.imageId;});
            for (auto it: imageScores) {
                const float &si = it.score;
                if (si >= minScoreToRetain) {
                    int refIndex = it.imageId;
                    if (!spAlreadyAddedKF.count(refIndex)) {
                        cout << it.imageId << "," << it.score << endl;
                        shared_ptr<KeyFrame> ref = viewGraph.indexFrame(refIndex);
                        FeatureMatches featureMatches = matcher->matchKeyPointsPair(ref->getKps(),
                                                                                    cur->getKps());
                        overlapFrames.emplace_back(refIndex);
                        spAlreadyAddedKF.insert(refIndex);
                        cudaStreamCreate(&streams[index]);
                        threads[index] = new thread(
                                bind(&GlobalRegistration::registrationPairEdge, this, placeholders::_1, placeholders::_2,
                                     placeholders::_3, placeholders::_4), featureMatches, &edges[index], streams[index], false);
                        index++;
                    }
                    if (overlapFrames.size() >= k) break;
                }
            }
        }

        for (int i = 0; i < index; i++) {
            threads[i]->join();
            CUDA_CHECKED_CALL(cudaStreamSynchronize(streams[i]));
            CUDA_CHECKED_CALL(cudaStreamDestroy(streams[i]));
        }
    }

    GlobalRegistration::GlobalRegistration(const GlobalConfig &globalConfig, SIFTVocabulary* siftVocabulary): globalConfig(globalConfig) {
        dBoWHashing = new DBoWHashing(globalConfig, siftVocabulary, true);
        matcher = new SIFTFeatureMatcher();
        egRegistration = new EGRegistration(globalConfig);
        homoRegistration = new HomographyRegistration(globalConfig);
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
        vector<TransformVector> transforms(n);
        // merge nodes and transformation
        for (int i = 0; i < n; i++) {
            bool isConnected = findShortestPathTransVec(viewGraph, connectedComponents[i], transforms[i]);
            LOG_ASSERT(isConnected) << " occur a error: the view graph is not connected ";
            if (connectedComponents[i].size() > 1) {
                BARegistration baRegistration(globalConfig);
                baRegistration.multiViewBundleAdjustment(viewGraph, connectedComponents[i],
                                                         transforms[i], globalConfig.costThreshold);
                mergeComponentNodes(viewGraph, connectedComponents[i], transforms[i], nodes[i]);
            } else {
                nodes[i] = viewGraph[connectedComponents[i][0]];
            }

        }
        for (int i = 0; i < n; i++) {
            vector<int> cc1 = connectedComponents[i];
            TransformVector trans1 = transforms[i];
            for (int j = i + 1; j < n; j++) {
                vector<int> cc2 = connectedComponents[j];
                TransformVector trans2 = transforms[j];
                adjMatrix(i, j) = selectEdgeBetweenComponents(viewGraph, cc1, trans1, cc2, trans2);
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

    void GlobalRegistration::trackKeyFrames(shared_ptr<KeyFrame> keyframe) {
        if (viewGraph.getFramesNum() > 0) {
            clock_t start = clock();
            vector<int> overlapFrames;
            vector<Edge, Eigen::aligned_allocator<Edge>> edges;
            registrationEdges(keyframe, overlapFrames, edges);

            for(int i=0; i<overlapFrames.size(); i++) {
                if (!edges[i].isUnreachable()) {
                    int refNodeIndex = viewGraph.findNodeIndexByFrameIndex(overlapFrames[i]);
                    if (bestEdgeMap.count(refNodeIndex)) {
                        double bestCost = bestEdgeMap[refNodeIndex].getCost();
                        if (edges[i].getCost() < bestCost) {
                            bestEdgeMap[refNodeIndex] = edges[i];
                            bestCurIndexes[refNodeIndex] = keyframe->getIndex();
                            bestRefIndexes[refNodeIndex] = overlapFrames[i];
                        }
                    } else {
                        bestEdgeMap.insert(map<int, Edge>::value_type(refNodeIndex, edges[i]));
                        bestCurIndexes.insert(map<int, int>::value_type(refNodeIndex, keyframe->getIndex()));
                        bestRefIndexes.insert(map<int, int>::value_type(refNodeIndex, overlapFrames[i]));
                    }
                }
            }

            cout << "pair registration:" << double(clock() - start) / CLOCKS_PER_SEC << "s" << endl;
        }
    }

    void GlobalRegistration::insertKeyFrames(shared_ptr<KeyFrame> keyframe) {
        viewGraph.extendNode(keyframe);

        if(viewGraph.getFramesNum()>1) {
            int curNodeIndex = viewGraph.getNodesNum() - 1;
            for (auto mit: bestEdgeMap) {
                int refNodeIndex = mit.first;
                Edge &edge = viewGraph(refNodeIndex, curNodeIndex);
                Edge &bestEdge = mit.second;

                if (edgeCompare(bestEdge, edge)) {
                    Transform transX = viewGraph[refNodeIndex].getTransform(bestRefIndexes[refNodeIndex]);
                    Transform transY = keyframe->getTransform(bestCurIndexes[refNodeIndex]);

                    Intrinsic kX = viewGraph[refNodeIndex].getK();
                    Intrinsic kY = viewGraph[curNodeIndex].getK();

                    // transform and k: p' = K(R*K^-1*p+t)
                    Rotation rX = kX*transX.block<3,3>(0,0)*kX.inverse();
                    Rotation rY = kY*transY.block<3,3>(0,0)*kY.inverse();
                    Translation tX = kX*transX.block<3,1>(0,3);
                    Translation tY = kY*transY.block<3,1>(0,3);

                    // transform key points
                    transformFeatureKeypoints(bestEdge.getKxs(), rX, tY);
                    transformFeatureKeypoints(bestEdge.getKys(), rY, tY);

                    // trans12 = trans1*relative_trans*trans2^-1
                    Transform relativeTrans = transX * bestEdge.getTransform() * GeoUtil::reverseTransformation(transY);
                    edge.setTransform(relativeTrans);
                    edge.setKxs(bestEdge.getKxs());
                    edge.setKys(bestEdge.getKys());
                    edge.setCost(bestEdge.getCost());
                }
            }
            bestEdgeMap.clear();
            bestCurIndexes.clear();
            bestRefIndexes.clear();

            mergeViewGraph();
            lostNum = registration(false);
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
        vector<vector<int>> ccs = findConnectedComponents(viewGraph, globalConfig.maxAvgCost);
        // global registration for every connected component
        int visibleIndex = 0;
        int maxFrameNum = 0;
        for(int i=0; i<ccs.size(); i++) {
            int curFrameNum = 0;
            for(int j : ccs[i]) {
                curFrameNum += viewGraph[j].getFrames().size();
            }
            if(curFrameNum>maxFrameNum) {
                visibleIndex = i;
                maxFrameNum = curFrameNum;
            }
        }

        int invisibleCount = 0;
        for(int i=0; i<ccs.size(); i++) {
            vector<int>& cc = ccs[i];
            for(int j: cc) {
                viewGraph[j].setVisible(i==visibleIndex);
                if(!viewGraph[j].isVisible()) invisibleCount+=viewGraph[j].getFrames().size();
            }
            if(cc.size()>1) {
                TransformVector gtTransVec;
                findShortestPathTransVec(viewGraph, cc, gtTransVec);
                if(opt) {
                    BARegistration baRegistration(globalConfig);
                    BAReport report = baRegistration.multiViewBundleAdjustment(viewGraph, cc, gtTransVec);
                    report.printReport();
                }
                for (int j = 1; j < cc.size(); j++) {
                    viewGraph[cc[j]].nGtTrans = viewGraph[cc[0]].nGtTrans*gtTransVec[j];
                }
            }
        }
        cout << "invisible count:" << invisibleCount << endl;
        return invisibleCount;
    }

    ViewGraph &GlobalRegistration::getViewGraph() {
        return viewGraph;
    }

    GlobalRegistration::~GlobalRegistration() {
        delete matcher;
        delete dBoWHashing;
        delete egRegistration;
        delete homoRegistration;
        delete pnpRegistration;
    }
}

