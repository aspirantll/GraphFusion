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
                candidateGroup.insert(viewGraph.findNodeIndexByFrameIndex(vit.first));
            }
            candidateGroup.insert(viewGraph.findNodeIndexByFrameIndex(candidate));


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

    void GlobalRegistration::insertViewCluster(shared_ptr<ViewCluster> cluster) {
        viewGraph.extendNode(cluster);

        int maxFrameIndex = cluster->getFrames().back()->getFrameIndex();
        int curNodeIndex = viewGraph.getNodesNum()-1;
        shared_ptr<ViewCluster> curNode = cluster;
        for(auto frame: cluster->getFrames()) {
            SE3 headSE = frame->getSE();
            for(auto con: frame->getConnections()) {
                int tailIndex = con->getTail()->getFrameIndex();
                if(tailIndex>maxFrameIndex) continue;
                int tailNodeIndex = viewGraph.findNodeIndexByFrameIndex(tailIndex);
                if(tailNodeIndex==curNodeIndex) continue;

                shared_ptr<ViewCluster> tailNode = viewGraph[tailNodeIndex];
                SE3 tailSE = tailNode->getFrameSE(tailIndex);
                SE3 finalSE = headSE*con->getSE()*tailSE.inverse();
                double cost = (tailNode->getPathLength(tailIndex)+curNode->getPathLength(frame->getFrameIndex())+1)*con->getCost();
                if(curNode->existConnection(tailNodeIndex)) {
                    shared_ptr<ViewConnection> oCon = curNode->getConnection(tailNodeIndex);
                    if(oCon->getCost()>cost) {
                        oCon->setPointWeight(con->getPointWeight());
                        oCon->setTransform(con->getTransform());
                        oCon->setCost(cost);
                    }
                }else {
                    curNode->addConnection(tailNodeIndex, allocate_shared<ViewConnection>(Eigen::aligned_allocator<ViewConnection>(), curNode, tailNode, con->getPointWeight(), finalSE, cost));
                    tailNode->addConnection(curNodeIndex, allocate_shared<ViewConnection>(Eigen::aligned_allocator<ViewConnection>(), tailNode, curNode, con->getPointWeight(), finalSE.inverse(), cost));
                }
            }
        }

        lostNum = viewGraph.updateSpanningTree(curNodeIndex);
        updateLostFrames();

        for(auto frame: cluster->getFrames()) {
            if(frame->isVisible()) {
                Transform trans = (cluster->getSE()*frame->getSE()).matrix();
                Vector3 ow;
                GeoUtil::computeOW(trans, ow);
                float3 pos = make_float3(ow.x(), ow.y(), ow.z());
                dBoWHashing->addVisualIndex(pos, frame->getKps(), frame->getFrameIndex(), cluster->isVisible());
            }
        }

        Optimizer::poseGraphOptimizeCeres(viewGraph);
        viewGraph.print();
    }

    void GlobalRegistration::globalTrack(shared_ptr<Frame> frame, float minScore) {
        viewGraph.addSourceFrame(frame);
        if(viewGraph.getNodesNum()<=0) return;
        std::vector<int> candidates = dBoWHashing->detectLoopClosures(frame->getKps(), minScore);
        if (!candidates.empty()) {
            cout << "-------------------loop candidates---------------------" << endl;
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

            cout << "----------------------------------------" << endl;
            prevConsistentGroups = std::move(consistentGroups);
        }
    }

    int GlobalRegistration::registration(bool opt) {
        int n = viewGraph.getNodesNum();
        if(n<1) return true;

        cout << "------------------------compute global transform for view graph------------------------" << endl;
        if(opt) {
            Optimizer::poseGraphOptimizeCeres(viewGraph);
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

