//
// Created by liulei on 2020/8/4.
//

#include <utility>

#include "registrations.h"
#include "../../tool/view_graph_util.h"
#include "../../processor/downsample.h"
#include "optimizer.h"

using namespace rtf::ViewGraphUtil;

namespace rtf {

    void GlobalRegistration::registrationPairEdge(FeatureMatches *featureMatches, Edge *edge, cudaStream_t curStream, float weight) {
        stream = curStream;
        RANSAC2DReport pnp = pnpRegistration->registrationFunction(*featureMatches);
        RegReport ba;
        if (pnp.success) {
            BARegistration baRegistration(globalConfig);
            vector<FeatureKeypoint> kxs, kys;
            featureIndexesToPoints(featureMatches->getKx(), pnp.kps1, kxs);
            featureIndexesToPoints(featureMatches->getKy(), pnp.kps2, kys);

            ba = baRegistration.bundleAdjustment(pnp.T, featureMatches->getCx(), featureMatches->getCy(), kxs, kys);

            if (ba.success) {
                double cost = ba.avgCost();
                if (!isnan(cost) && cost < globalConfig.maxAvgCost) {
                    edge->setKxs(kxs);
                    edge->setKys(kys);
                    edge->setTransform(ba.T);
                    edge->setCost(cost*weight);
                }
            }
        }

        printMutex.lock();
        cout << "-------------------" << featureMatches->getFIndexX() << "-" << featureMatches->getFIndexY()  << "-global-pnp+ba---------------------------------" << endl;
        pnp.printReport();
        ba.printReport();
        printMutex.unlock();
    }

    void GlobalRegistration::registrationEdges(shared_ptr<KeyFrame> curKeyFrame, vector<int>& refKFIndexes, vector<int>& refInnerIndexes, vector<int>& curInnerIndexes, EigenVector(Edge)& pairEdges) {
        int k = refKFIndexes.size();
        pairEdges.resize(k, Edge::UNREACHABLE);
        thread *threads[k];
        cudaStream_t streams[k];

        for(int i=0; i<k; i++) {
            shared_ptr<KeyFrame> refKeyFrame = viewGraph.indexFrame(refKFIndexes[i]);
            shared_ptr<Frame> refFrame = refKeyFrame->getFrame(refInnerIndexes[i]);
            shared_ptr<Frame> curFrame = curKeyFrame->getFrame(curInnerIndexes[i]);
            FeatureMatches featureMatches = matcher->matchKeyPointsPair(refFrame->getKps(),
                                                                        curFrame->getKps());

            float weight = viewGraph.getPathLen(refKFIndexes[i])+refKeyFrame->getPathLength(refInnerIndexes[i])+curKeyFrame->getPathLength(curInnerIndexes[i])+1;
//            cudaStreamCreate(&streams[index]);
//            threads[index] = new thread(
//                    bind(&GlobalRegistration::registrationPairEdge, this, placeholders::_1, placeholders::_2,
//                         placeholders::_3), featureMatches, &edges[index], streams[index]);
            registrationPairEdge(&featureMatches, &pairEdges[i], stream, weight);
        }

        /*for (int i = 0; i < index; i++) {
            threads[i]->join();
            CUDA_CHECKED_CALL(cudaStreamSynchronize(streams[i]));
            CUDA_CHECKED_CALL(cudaStreamDestroy(streams[i]));
        }*/

        for(int i=0; i<k; i++) {
            if(!pairEdges[i].isUnreachable()) {
                shared_ptr<KeyFrame> refKeyFrame = viewGraph.indexFrame(refKFIndexes[i]);
                Transform refTrans = refKeyFrame->getTransform(refInnerIndexes[i]);
                Transform curTrans = curKeyFrame->getTransform(curInnerIndexes[i]);
                Transform trans = refTrans*pairEdges[i].getTransform()*curTrans.inverse();
                pairEdges[i].setTransform(trans);
            }
        }
    }

    void GlobalRegistration::findOverlapping(shared_ptr<KeyFrame> cur, vector<int>& refKFIndexes, vector<int>& refInnerIndexes, vector<int>& curInnerIndexes) {
        const int k = globalConfig.overlapNum;
        const int lastNum = 1;
        const int n = viewGraph.getFramesNum()-1;
        vector<shared_ptr<KeyFrame>> frames = viewGraph.getSourceFrames();
        DBoW2::BowVector& bow = cur->getKps().getMBowVec();

        set<int> spAlreadyAddedKF;
        refKFIndexes.reserve(k);
        refInnerIndexes.reserve(k);
        curInnerIndexes.reserve(k);
        // last frame
        int index = 0;
        for (int i = 1; i <= lastNum && i <= k && i < n; i++) {
            shared_ptr<KeyFrame> refKeyFrame = frames[n - i];
            int refIndex = refKeyFrame->getIndex();
            spAlreadyAddedKF.insert(refIndex);
            refKFIndexes.emplace_back(refIndex);
            pair<int, int> bestPair = selectBestOverlappingFrame(refKeyFrame, cur, siftVocabulary);
            refInnerIndexes.emplace_back(bestPair.first);
            curInnerIndexes.emplace_back(bestPair.second);
            index++;
        }

        if (refKFIndexes.size() < k) {
            Timer queryTimer = Timer::startTimer("query index");
            std::vector<MatchScore> imageScores = dBoWHashing->queryImages(lastPos, cur->getKps(), notLost,  lostNum > 0);

            queryTimer.stopTimer();
            float minScore = imageScores[0].score*globalConfig.matchFactorTh;
            std::sort(imageScores.begin(), imageScores.end(), [=](MatchScore& ind1, MatchScore& ind2) {
                if(ind1.score>minScore&&ind2.score>minScore) {
                    return viewGraph.getPathLen(ind1.imageId)<viewGraph.getPathLen(ind2.imageId);
                }else {
                    return ind1.score > ind2.score;
                }
            });
            // Return all those keyframes with a score higher than 0.75*bestScore
            for (auto it: imageScores) {
                const float &si = it.score;
                int refIndex = it.imageId;
                if (!spAlreadyAddedKF.count(refIndex)) {
                    shared_ptr<KeyFrame> refKeyFrame = viewGraph.indexFrame(refIndex);
                    spAlreadyAddedKF.insert(refIndex);
                    refKFIndexes.emplace_back(refIndex);
                    pair<int, int> bestPair = selectBestOverlappingFrame(refKeyFrame, cur, siftVocabulary);
                    refInnerIndexes.emplace_back(bestPair.first);
                    curInnerIndexes.emplace_back(bestPair.second);
                    index++;
                }
                if (refKFIndexes.size() >= k) break;
            }
        }
    }

    bool GlobalRegistration::checkLoopConsistency(const std::vector<int> &candidates,
                              std::vector<int> &consistentCandidates,
                              std::vector<ConsistentGroup> &consistentGroups,
                              const int covisibilityConsistencyTh) {
        // For each loop candidate check consistency with previous loop candidates
        // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
        // A group is consistent with a previous group if they share at least a keyframe
        // We must detect a consistent loop in several consecutive keyframes to accept it

        consistentCandidates.clear();
        consistentGroups.clear();

        std::vector<bool> prevConsistentGroupFlag(prevConsistentGroups.size(), false);


        for (int candidate : candidates) {

            // --- Create candidate group -----------
            std::set<int> candidateGroup;
            const auto &connections = viewGraph[candidate].getConnections();
            for (auto vi: connections) {
                candidateGroup.insert(vi);
            }
            candidateGroup.insert(candidate);


            // --- compare candidate grou against prevoius consistent groups -----------

            bool enoughConsistent = false;
            bool consistentForSomeGroup = false;

            for (size_t g = 0, iendG = prevConsistentGroups.size(); g < iendG; g++) {
                // find if candidate_group is consistent with any previous consistent group
                std::set<int> prevGroup = prevConsistentGroups[g].first;
                bool consistent = false;
                for (int vi: candidateGroup) {
                    if (prevGroup.count(vi)) {
                        consistent = true;
                        consistentForSomeGroup = true;
                        break;
                    }
                }

                if (consistent) {
                    int previousConsistency = prevConsistentGroups[g].second;
                    int currentConsistency = previousConsistency + 1;

                    if (!prevConsistentGroupFlag[g]) {
                        consistentGroups.push_back(std::make_pair(candidateGroup,
                                                                  currentConsistency));
                        prevConsistentGroupFlag[g] = true; //this avoid to include the same group more than once
                    }

                    if (currentConsistency >= covisibilityConsistencyTh && !enoughConsistent) {
                        consistentCandidates.push_back(candidate);
                        enoughConsistent = true; //this avoid to insert the same candidate more than once
                    }
                }
            }

            // If the group is not consistent with any previous group insert with consistency counter set to zero
            if (!consistentForSomeGroup) {
                consistentGroups.push_back(std::make_pair(candidateGroup, 0));
            }
        }

        return !consistentCandidates.empty();

    }

    GlobalRegistration::GlobalRegistration(const GlobalConfig &globalConfig, SIFTVocabulary* siftVocabulary): siftVocabulary(siftVocabulary), globalConfig(globalConfig) {
        dBoWHashing = new DBoWHashing(globalConfig, siftVocabulary, &viewGraph, true);
        matcher = new SIFTFeatureMatcher();
        pnpRegistration = new PnPRegistration(globalConfig);
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

    bool GlobalRegistration::loopClosureCorrection() {
        int curNodeIndex = viewGraph.getNodesNum()-1;
        shared_ptr<KeyFrame> curKeyFrame = viewGraph[curNodeIndex].getFrames()[0];
        auto& sf = curKeyFrame->getKps();
        // find min score
        float minScore = 1;
        int minIndex = -1;
        for(int ind: viewGraph[curNodeIndex].getConnections()) {
            assert(ind < curNodeIndex);
            auto& fp = viewGraph[ind].getFrames()[0]->getKps();
            float score = siftVocabulary->score(sf.getMBowVec(), fp.getMBowVec());
            if(score < minScore) {
                minScore = score;
                minIndex = ind;
            }
        }
        cout << "minScore:" << minScore << ", minIndex:" << minIndex << endl;

        std::vector<int> candidates = dBoWHashing->detectLoopClosures(sf, minScore);
        if (!candidates.empty()) {
            cout << "find loop candidates" << endl;
            std::vector<int> consistentCandidates;
            std::vector<ConsistentGroup> consistentGroups;

            if (checkLoopConsistency(candidates, consistentCandidates, consistentGroups)) {
                std::cout << " * * * loop closure detected * * *\n" << std::endl;

                for (int refNodeIndex : consistentCandidates) {
                    if(viewGraph.existEdge(refNodeIndex, curNodeIndex)) continue;
                    shared_ptr<KeyFrame> refKeyFrame = viewGraph[refNodeIndex].getFrames()[0];
                    pair<int, int> bestPair = selectBestOverlappingFrame(refKeyFrame, curKeyFrame, siftVocabulary);
                    shared_ptr<Frame> refFrame = refKeyFrame->getFrame(bestPair.first);
                    shared_ptr<Frame> curFrame = curKeyFrame->getFrame(bestPair.second);

                    FeatureMatches featureMatches = matcher->matchKeyPointsPair(refFrame->getKps(),
                                                                                curFrame->getKps());

                    Edge &edge = viewGraph(refNodeIndex, curNodeIndex);
                    float weight = viewGraph.getPathLen(refNodeIndex) + 1;
                    registrationPairEdge(&featureMatches, &edge, stream, weight);

                    if(!edge.isUnreachable()) {
                        viewGraph[refNodeIndex].addConnections(curNodeIndex);
                        viewGraph[curNodeIndex].addConnections(refNodeIndex);

                        loops.emplace_back(make_pair(refNodeIndex, curNodeIndex));
                    }
                }

                Optimizer::poseGraphOptimizeCeres(viewGraph, loops);
            }
            prevConsistentGroups = std::move(consistentGroups);
        }
    }

    void GlobalRegistration::insertKeyFrames(shared_ptr<KeyFrame> keyframe) {
        siftVocabulary->computeBow(keyframe->getKps());
        viewGraph.extendNode(keyframe);

        if(viewGraph.getFramesNum()>1) {
            vector<int> refKFIndexes, refInnerIndexes, curInnerIndexes;
            findOverlapping(keyframe, refKFIndexes, refInnerIndexes, curInnerIndexes);
            vector<Edge, Eigen::aligned_allocator<Edge>> pairEdges;
            registrationEdges(keyframe, refKFIndexes, refInnerIndexes, curInnerIndexes, pairEdges);

            map<int, int> bestEdgeMap;
            for(int i=0; i<refKFIndexes.size(); i++) {
                if (!pairEdges[i].isUnreachable()) {
                    int refNodeIndex = viewGraph.findNodeIndexByFrameIndex(refKFIndexes[i]);
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
                    if(edge.isUnreachable()) {
                        viewGraph[refNodeIndex].addConnections(curNodeIndex);
                        viewGraph[curNodeIndex].addConnections(refNodeIndex);
                    }

                    edge.setKxs(bestEdge.getKxs());
                    edge.setKys(bestEdge.getKys());
                    edge.setTransform(bestEdge.getTransform());
                    edge.setCost(bestEdge.getCost());
                }
            }

            Timer grTimer = Timer::startTimer("gr");
            lostNum = viewGraph.updateSpanningTree();
            grTimer.stopTimer();
            updateLostFrames();

//            loopClosureCorrection();

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

        for(int i=0; i<ccs.size(); i++) {
            vector<int>& cc = ccs[i];
            if(cc.size()>1) {
                if(opt) {
                    Optimizer::poseGraphOptimizeCeres(viewGraph);
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

