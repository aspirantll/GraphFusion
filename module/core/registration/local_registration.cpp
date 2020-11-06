//
// Created by liulei on 2020/7/9.
//

#include "registrations.h"
#include "../../tool/view_graph_util.h"
#include "../../feature/feature_matcher.h"
#include <glog/logging.h>

#include <utility>

using namespace rtf::ViewGraphUtil;

namespace rtf {
    LocalRegistration::LocalRegistration(const GlobalConfig &globalConfig, SIFTVocabulary* siftVocabulary): globalConfig(globalConfig), siftVocabulary(siftVocabulary) {
        localViewGraph.reset(0);
        matcher = new SIFTFeatureMatcher();
        localDBoWVoc = new DBoWVocabulary();
        egRegistration = new EGRegistration(globalConfig);
        homoRegistration = new HomographyRegistration(globalConfig);
        pnpRegistration = new PnPRegistration(globalConfig);
    }

    void LocalRegistration::registrationEGBA(FeatureMatches *featureMatches, Edge *edge, cudaStream_t curStream) {
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
        cout << "-------------------" << featureMatches->getFIndexX() << "-" << featureMatches->getFIndexY()  << "-eg+ba---------------------------------" << endl;
        eg.printReport();
        if (ba.success) {
            ba.printReport();
        }
        printMutex.unlock();
    }

    void LocalRegistration::registrationHomoBA(FeatureMatches *featureMatches, Edge *edge, cudaStream_t curStream) {
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
        cout << "-------------------" << featureMatches->getFIndexX() << "-" << featureMatches->getFIndexY()  << "-homo+ba---------------------------------" << endl;
        homo.printReport();
        if (ba.success) {
            ba.printReport();
        }
        printMutex.unlock();
    }

    void LocalRegistration::registrationPnPBA(FeatureMatches *featureMatches, Edge *edge, cudaStream_t curStream) {
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
        cout << "-------------------" << featureMatches->getFIndexX() << "-" << featureMatches->getFIndexY()  << "-pnp+ba---------------------------------" << endl;
        pnp.printReport();
        if (ba.success) {
            ba.printReport();
        }
        printMutex.unlock();
    }

    void LocalRegistration::registrationPairEdge(FeatureMatches featureMatches, Edge *edge, cudaStream_t curStream, bool near) {
        stream = curStream;
        registrationPnPBA(&featureMatches, edge, curStream);
        if(near&&edge->isUnreachable()&&featureMatches.size()>globalConfig.kMinMatches) {
            registrationEGBA(&featureMatches, edge, curStream);
            if(edge->isUnreachable()) {
                registrationHomoBA(&featureMatches, edge, curStream);
            }
        }
    }

    void LocalRegistration::registrationLocalEdges(vector<int>& overlapFrames, EigenVector(Edge)& edges) {
        const int k = globalConfig.overlapNum;
        const int lastNum = 2;
        const int curIndex = localViewGraph.getFramesNum()-1;
        auto frames = localViewGraph.getSourceFrames();

        set<int> spAlreadyAddedKF;
        overlapFrames.reserve(k);
        edges.resize(k, Edge::UNREACHABLE);
        thread *threads[k];
        cudaStream_t streams[k];
        // last frame
        int index = 0;
        for (int i = 1; i <= lastNum && i <= k && i <= curIndex; i++) {
            int refIndex = curIndex - i;
            spAlreadyAddedKF.insert(refIndex);
            overlapFrames.emplace_back(refIndex);

            FeatureMatches featureMatches = matcher->matchKeyPointsPair(frames[refIndex]->getFirstFrame()->getKps(),
                                                                        frames[curIndex]->getFirstFrame()->getKps());
            cudaStreamCreate(&streams[index]);
            threads[index] = new thread(
                    bind(&LocalRegistration::registrationPairEdge, this, placeholders::_1, placeholders::_2,
                         placeholders::_3, placeholders::_4), featureMatches, &edges[index], streams[index], true);
            index++;
        }

        if (overlapFrames.size() < k) {
            auto cur = localViewGraph.indexFrame(curIndex);
            std::vector<MatchScore> imageScores;
            localDBoWVoc->query(siftVocabulary, &cur->getFirstFrame()->getKps().getMBowVec(), &imageScores);
            // Return all those keyframes with a score higher than 0.75*bestScore
            float minScoreToRetain = globalConfig.minScore;
            std::sort(imageScores.begin(), imageScores.end(), [=](MatchScore& ind1, MatchScore& ind2) {return ind1.imageId < ind2.imageId;});
            for (auto it: imageScores) {
                const float &si = it.score;
                if (si >= minScoreToRetain) {
                    int refIndex = it.imageId;
                    if (!spAlreadyAddedKF.count(refIndex)) {
                        FeatureMatches featureMatches = matcher->matchKeyPointsPair(frames[refIndex]->getFirstFrame()->getKps(),
                                                                                    frames[curIndex]->getFirstFrame()->getKps());
                        overlapFrames.emplace_back(refIndex);
                        spAlreadyAddedKF.insert(refIndex);
                        cudaStreamCreate(&streams[index]);
                        threads[index] = new thread(
                                bind(&LocalRegistration::registrationPairEdge, this, placeholders::_1, placeholders::_2,
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

    void LocalRegistration::updateLocalEdges() {
        clock_t start = clock();
        vector<int> overlapFrames;
        vector<Edge, Eigen::aligned_allocator<Edge>> edges;
        registrationLocalEdges(overlapFrames, edges);

        map<int, int> bestEdgeMap;
        for(int i=0; i<overlapFrames.size(); i++) {
            if (!edges[i].isUnreachable()) {
                int refNodeIndex = overlapFrames[i];
                if (bestEdgeMap.count(refNodeIndex)) {
                    double bestCost = edges[bestEdgeMap[refNodeIndex]].getCost();
                    if (edges[i].getCost() < bestCost) {
                        bestEdgeMap[refNodeIndex] = i;
                    }
                } else {
                    bestEdgeMap.insert(map<int, int>::value_type(refNodeIndex, i));
                }
            }
        }

        int curNodeIndex = localViewGraph.getNodesNum() - 1;
        for (auto mit: bestEdgeMap) {
            int refNodeIndex = mit.first;
            int ind = mit.second;
            Edge &edge = localViewGraph(refNodeIndex, curNodeIndex);
            Edge &bestEdge = edges[ind];

            if (edgeCompare(bestEdge, edge)) {
                edge.setKxs(bestEdge.getKxs());
                edge.setKys(bestEdge.getKys());
                edge.setTransform(bestEdge.getTransform());
                edge.setCost(bestEdge.getCost());
            }

        }

//        cout << "pair registration:" << double(clock() - start) / CLOCKS_PER_SEC << "s" << endl;

    }

    void LocalRegistration::localTrack(shared_ptr<Frame> frame) {
        shared_ptr<KeyFrame> keyframe = allocate_shared<KeyFrame>(Eigen::aligned_allocator<KeyFrame>());
        keyframe->addFrame(frame);
        localViewGraph.extendNode(keyframe);
        if (localViewGraph.getFramesNum() > 1) {
            updateLocalEdges();
        }
        localDBoWVoc->add(localViewGraph.getFramesNum()-1, &frame->getKps().getMBowVec());
    }

    shared_ptr<KeyFrame> LocalRegistration::mergeFramesIntoKeyFrame() {
        //1. local optimization
        const int n = localViewGraph.getNodesNum();
        TransformVector gtTransVec;
        vector<vector<int>> connectedComponents = findConnectedComponents(localViewGraph, globalConfig.maxAvgCost);
        findShortestPathTransVec(localViewGraph, connectedComponents[0], gtTransVec);

        BARegistration baRegistration(globalConfig);
        baRegistration.multiViewBundleAdjustment(localViewGraph, connectedComponents[0], gtTransVec);

        // 2. initialize key frame
        set<int> visibleSet(connectedComponents[0].begin(), connectedComponents[0].end());
        shared_ptr<KeyFrame> keyframe = allocate_shared<KeyFrame>(Eigen::aligned_allocator<KeyFrame>());
        keyframe->setTransform(Transform::Identity());
        for(int i=0; i<n; i++) {
            shared_ptr<Frame> frame = localViewGraph[i].getFrames()[0]->getFirstFrame();
            frame->setTransform(gtTransVec[i]);
            frame->setVisible(visibleSet.count(i));
            keyframe->addFrame(frame);
        }

        // reset
        localViewGraph.reset(0);
        localDBoWVoc->clear();

        if(keyframe->getIndex() == 180) {
            cout << "ky:" << keyframe->getTransform() << endl;
        }
        return keyframe;
    }

    ViewGraph& LocalRegistration::getViewGraph() {
        return localViewGraph;
    }

    LocalRegistration::~LocalRegistration() {
        delete matcher;
        delete localDBoWVoc;
        delete egRegistration;
        delete homoRegistration;
        delete pnpRegistration;
    }
}




