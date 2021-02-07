//
// Created by liulei on 2020/8/4.
//

#include <utility>

#include "registrations.h"
#include "optimizer.h"

namespace rtf {

    void GlobalRegistration::registrationPairEdge(FeatureMatches *featureMatches, ConnectionCandidate *edge, cudaStream_t curStream, float weight) {
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

    bool GlobalRegistration::checkLoopConsistency(const std::vector<int> &candidates,
                              std::vector<int> &consistentCandidates,
                              std::vector<ConsistentGroup> &consistentGroups,
                              const int covisibilityConsistencyTh) {
        // For each loop candidate check consistency with previous loop candidates
        // Each candidate expands a covisibility group (viewclusters connected to the loop candidate in the covisibility graph)
        // A group is consistent with a previous group if they share at least a viewcluster
        // We must detect a consistent loop in several consecutive viewclusters to accept it

        consistentCandidates.clear();
        consistentGroups.clear();

        std::vector<bool> prevConsistentGroupFlag(prevConsistentGroups.size(), false);


        for (int candidate : candidates) {

            // --- Create candidate group -----------
            std::set<int> candidateGroup;
            for (auto& vit: viewGraph.findFrameByIndex(candidate)->getConnectionMap()) {
                candidateGroup.insert(vit.first);
            }
            candidateGroup.insert(candidate);


            // --- compare candidate group against previous consistent groups -----------

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
        notLost = true;
    }

    void GlobalRegistration::updateLostFrames() {
        vector<int> lostImageIds = dBoWHashing->lostImageIds();
        vector<int> updateIds;
        vector<float3> poses;
        for (int lostId: lostImageIds) {
            int nodeIndex = viewGraph.findNodeIndexByFrameIndex(lostId);
            if (viewGraph[nodeIndex]->isVisible()) {
                updateIds.emplace_back(lostId);

                Transform trans = viewGraph.getViewTransform(lostId);
                Vector3 ow;
                GeoUtil::computeOW(trans, ow);
                poses.emplace_back(make_float3(ow.x(), ow.y(), ow.z()));
            }
        }

        dBoWHashing->updateVisualIndex(updateIds, poses);
    }

    float GlobalRegistration::computeMinScore(shared_ptr<Frame> frame) {
        auto connections = frame->getConnections();
        float minScore = 1;
        if(connections.empty()) {
            int lastFrameIndex = frame->getFrameIndex()-1;
            if(lastFrameIndex>0) {
                shared_ptr<Frame> lastFrame = viewGraph.findFrameByIndex(lastFrameIndex);
                minScore = siftVocabulary->score(frame->getKps().getMBowVec(), lastFrame->getKps().getMBowVec());
            }
        }else {
            for(auto& con: connections) {
                float score = siftVocabulary->score(frame->getKps().getMBowVec(), con->getT()->getKps().getMBowVec());
                if(score < minScore) {
                    minScore = score;
                }
            }
        }

        return minScore * 0.5;
    }

    void GlobalRegistration::insertViewCluster(shared_ptr<ViewCluster> cluster) {
        // collect all candidates
        map<shared_ptr<Frame>, map<int, double>> frameCandidates;
        map<int, int> nodeWeights;
        map<int, double> nodeBestScores;
        map<int, pair<int, int>> nodeBestPairs;
        for(shared_ptr<Frame> frame: cluster->getFrames()) {
            viewGraph.addSourceFrame(frame);
            if(viewGraph.getNodesNum()>1&&frame->isVisible()) {
                int curFrameIndex = frame->getFrameIndex();
                float minScore = computeMinScore(frame);
                map<int, double> candidates = dBoWHashing->findOverlappingFrames(lastPos, frame->getKps(), minScore, notLost);
                for(auto& mit: candidates) {
                    int frameIndex = mit.first;
                    if(frame->existConnection(frameIndex)) continue;
                    int nodeIndex = viewGraph.findNodeIndexByFrameIndex(frameIndex);
                    if(!nodeWeights.count(nodeIndex)) {
                        nodeWeights.insert(map<int,int>::value_type(nodeIndex, 1));
                        nodeBestScores.insert(map<int,double>::value_type(nodeIndex, mit.second));
                        nodeBestPairs.insert(map<int,pair<int, int>>::value_type(nodeIndex, make_pair(frameIndex, curFrameIndex)));
                    }else {
                        nodeWeights[nodeIndex]++;
                        if(nodeBestScores[nodeIndex]<mit.second) {
                            nodeBestScores[nodeIndex] = mit.second;
                            nodeBestPairs[nodeIndex] = make_pair(frameIndex, curFrameIndex);
                        }
                    }
                }

                frameCandidates.insert(map<shared_ptr<Frame>, map<int, double>>::value_type(frame, candidates));
            }
        }
        viewGraph.extendNode(cluster);

        int curNodeIndex = viewGraph.getNodesNum()-1;
        shared_ptr<ViewCluster> curNode = viewGraph[curNodeIndex];

        if(viewGraph.getNodesNum()>1) {
            // find best k pairwise from candidates
            vector<pair<int, int>> nodeWeightsVec;
            for (auto& mit: nodeWeights) {
                nodeWeightsVec.emplace_back(make_pair(mit.first, mit.second));
            }

            std::sort(nodeWeightsVec.begin(), nodeWeightsVec.end(),
                      [](const pair<int, int> &x, const pair<int, int> &y)
                      {
                          return x.second > y.second;
                      }
            );

            set<int> alreadySelectedNodeIndexes;
            vector<int> selectedNodeIndexes;
            vector<pair<int, int>> bestPairs;
            int lastNodeIndex = curNodeIndex-1;
            alreadySelectedNodeIndexes.insert(lastNodeIndex);
            selectedNodeIndexes.emplace_back(lastNodeIndex);
            bestPairs.emplace_back(selectBestOverlappingFrame(viewGraph[lastNodeIndex], cluster, siftVocabulary));

            for(int i=0; i<nodeWeightsVec.size() && alreadySelectedNodeIndexes.size() < globalConfig.overlapNum; i++) {
                int nodeIndex = nodeWeightsVec[i].first;
                if(alreadySelectedNodeIndexes.count(nodeIndex)) continue;
                alreadySelectedNodeIndexes.insert(nodeIndex);
                selectedNodeIndexes.emplace_back(nodeIndex);
                bestPairs.emplace_back(selectBestOverlappingFrame(viewGraph[nodeIndex], cluster, siftVocabulary));
            }

            // align pairwise frames
            for(int i=0; i<selectedNodeIndexes.size(); i++) {
                totalCount++;

                int refNodeIndex = selectedNodeIndexes[i];
                pair<int, int> bestPair = bestPairs[i];
                shared_ptr<Frame> refFrame = viewGraph.findFrameByIndex(bestPair.first);
                shared_ptr<Frame> curFrame = viewGraph.findFrameByIndex(bestPair.second);
                FeatureMatches featureMatches = matcher->matchKeyPointsPair(refFrame->getKps(),
                                                                            curFrame->getKps());

                ConnectionCandidate candidate;
                float weight = viewGraph.getPathLenByNodeIndex(refNodeIndex)+viewGraph[refNodeIndex]->getPathLength(bestPair.first)+cluster->getPathLength(bestPair.second)+1;
                registrationPairEdge(&featureMatches, &candidate, stream,  weight);

                if(!candidate.isUnreachable()) {
                    successCount++;
                    shared_ptr<ViewCluster> refNode = viewGraph[refNodeIndex];

                    SE3 finalSE = refFrame->getSE()*candidate.getSE()*curFrame->getSE().inverse();
                    {
                        vector<Point3D> points(candidate.getKys().begin(), candidate.getKys().end());

                        float pointWeight = PointUtil::computePointWeight(points, viewGraph.getCamera());

                        refNode->addConnection(curNodeIndex, allocate_shared<ViewConnection>(Eigen::aligned_allocator<ViewConnection>(), refNode, curNode, pointWeight, finalSE, candidate.getCost()));
                    }

                    {
                        vector<Point3D> points(candidate.getKys().begin(), candidate.getKys().end());

                        float pointWeight = PointUtil::computePointWeight(points, viewGraph.getCamera());

                        curNode->addConnection(refNodeIndex, allocate_shared<ViewConnection>(Eigen::aligned_allocator<ViewConnection>(), curNode, refNode, pointWeight, finalSE.inverse(), candidate.getCost()));
                    }

                    int matchNum = candidate.getKxs().size();
                    curFrame->addConnection(refFrame->getFrameIndex(), allocate_shared<FrameConnection>(Eigen::aligned_allocator<FrameConnection>(), curFrame, refFrame, matchNum, 0));
                    refFrame->addConnection(curFrame->getFrameIndex(), allocate_shared<FrameConnection>(Eigen::aligned_allocator<FrameConnection>(), refFrame, curFrame, matchNum, 0));
                }
            }

            lostNum = viewGraph.updateSpanningTree(curNodeIndex);
            updateLostFrames();
        }

        // loop correction
        /*for(auto& mit: frameCandidates) {
            shared_ptr<Frame> frame = mit.first;
            vector<int> candidates = mit.second;
            if (!candidates.empty()) {
                cout << "-------------------begin loop---------------------" << endl;
                std::vector<int> consistentCandidates;
                std::vector<ConsistentGroup> consistentGroups;

                if (checkLoopConsistency(candidates, consistentCandidates, consistentGroups)) {
                    for (int refFrameIndex : consistentCandidates) {
                        if(frame->existConnection(refFrameIndex)) continue;
                        shared_ptr<Frame> refFrame = viewGraph.findFrameByIndex(refFrameIndex);
                        FeatureMatches featureMatches = matcher->matchKeyPointsPair(refFrame->getKps(),
                                                                                    frame->getKps());

                        ConnectionCandidate candidate;
                        registrationPairEdge(&featureMatches, &candidate, stream,  1);

                        if(!candidate.isUnreachable()) {
                            {
                                vector<Point3D> points(candidate.getKys().begin(), candidate.getKys().end());

                                float weight = PointUtil::computePointWeight(points, viewGraph.getCamera());

                                refFrame->addConnection(frame->getFrameIndex(), allocate_shared<FrameConnection>(Eigen::aligned_allocator<FrameConnection>(), refFrame, frame, weight, candidate.getSE(), candidate.getCost()));
                            }

                            {
                                vector<Point3D> points(candidate.getKxs().begin(), candidate.getKxs().end());

                                float weight = PointUtil::computePointWeight(points, viewGraph.getCamera());

                                frame->addConnection(refFrame->getFrameIndex(), allocate_shared<FrameConnection>(Eigen::aligned_allocator<FrameConnection>(), frame, refFrame, weight, candidate.getSE().inverse(), candidate.getCost()));
                            }
                        }
                    }
                }

                cout << "---------------------end loop -------------------" << endl;
                prevConsistentGroups = std::move(consistentGroups);
            }
        }*/

        Optimizer::poseGraphOptimizeCeres(viewGraph);

        /*for(auto& mit: frameCandidates) {
            shared_ptr<Frame> curFrame = mit.first;
            for(auto& sit: mit.second) {
                int refFrameIndex = sit.first;
                int refNodeIndex = viewGraph.findNodeIndexByFrameIndex(refFrameIndex);
                shared_ptr<ViewCluster> refNode = viewGraph[refNodeIndex];
                shared_ptr<Frame> refFrame = viewGraph.findFrameByIndex(refFrameIndex);

                if(refNode->isVisible()&&refFrame->isVisible()&&!curFrame->existConnection(refFrame->getFrameIndex())) {
                    SE3 se = (refNode->getSE()*refFrame->getSE()).inverse()*(curNode->getSE()*curFrame->getSE());
                    FeatureMatches matches = matcher->matchKeyPointsPair(refFrame->getKps(), curFrame->getKps());

                    int count = 0;
                    for(int i=0; i<matches.size(); i++) {
                        FeatureMatch match = matches.getMatch(i);
                        shared_ptr<FeatureKeypoint> px = matches.getKx()[match.getPX()];
                        shared_ptr<FeatureKeypoint> py = matches.getKy()[match.getPY()];
                        Point3D rePixel = PointUtil::transformPixel(*py, se.matrix(), viewGraph.getCamera());
                        if((rePixel.toVector2()-px->toVector2()).squaredNorm()<globalConfig.maxPnPResidual) count++;
                    }
                    if(count>100) {
                        curFrame->addConnection(refFrame->getFrameIndex(), allocate_shared<FrameConnection>(Eigen::aligned_allocator<FrameConnection>(), curFrame, refFrame, count, sit.second));
                        refFrame->addConnection(curFrame->getFrameIndex(), allocate_shared<FrameConnection>(Eigen::aligned_allocator<FrameConnection>(), refFrame, curFrame, count, sit.second));
                    }
                }
            }
        }*/

        for(auto frame: cluster->getFrames()) {
            if(frame->isVisible()) {
                Transform trans = (cluster->getSE()*frame->getSE()).matrix();
                Vector3 ow;
                GeoUtil::computeOW(trans, ow);
                lastPos = make_float3(ow.x(), ow.y(), ow.z());
                dBoWHashing->addVisualIndex(lastPos, frame->getKps(), frame->getFrameIndex(), cluster->isVisible());
            }
        }
        notLost = viewGraph[curNodeIndex]->isVisible();
    }

    int GlobalRegistration::registration(bool opt) {
        int n = viewGraph.getNodesNum();
        if(n<1) return true;

        /*cout << "------------------------compute global transform for view graph------------------------" << endl;
        if(opt) {
            Optimizer::poseGraphOptimizeCeres(viewGraph);
        }*/
        cout << "invisible count:" << lostNum << endl;
        cout << "total registration count:" << totalCount << endl;
        cout << "success registration count:" << successCount << endl;
        cout << "fail registration count:" << totalCount-successCount << endl;
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

