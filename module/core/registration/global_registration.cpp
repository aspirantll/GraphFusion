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
            selectBestOverlappingFrame(refKeyFrame, cur, siftVocabulary,refInnerIndexes, curInnerIndexes);
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
                    selectBestOverlappingFrame(refKeyFrame, cur, siftVocabulary, refInnerIndexes, curInnerIndexes);
                    index++;
                }
                if (refKFIndexes.size() >= k) break;
            }
        }
    }

    GlobalRegistration::GlobalRegistration(const GlobalConfig &globalConfig, SIFTVocabulary* siftVocabulary): siftVocabulary(siftVocabulary), globalConfig(globalConfig) {
        dBoWHashing = new DBoWHashing(globalConfig, siftVocabulary, true);
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

            loopClosureDetection();

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

