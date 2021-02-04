//
// Created by liulei on 2020/7/9.
//

#include "registrations.h"
#include "optimizer.h"
#include <glog/logging.h>
#include <utility>

namespace rtf {
    LocalRegistration::LocalRegistration(const GlobalConfig &globalConfig, SIFTVocabulary* siftVocabulary): globalConfig(globalConfig), siftVocabulary(siftVocabulary) {
        localViewGraph.reset(0);
        matcher = new SIFTFeatureMatcher();
        localDBoWHashing = new DBoWHashing(globalConfig, siftVocabulary, &localViewGraph, false);
        pnpRegistration = new PnPRegistration(globalConfig);
    }

    void LocalRegistration::registrationPairEdge(SIFTFeaturePoints* f1, SIFTFeaturePoints* f2, ConnectionCandidate *edge, cudaStream_t curStream) {
        stream = curStream;
        FeatureMatches featureMatches = matcher->matchKeyPointsPair(*f1, *f2);
        RANSAC2DReport pnp = pnpRegistration->registrationFunction(featureMatches);
        RegReport ba;
        if (pnp.success) {
            BARegistration baRegistration(globalConfig);
            vector<FeatureKeypoint> kxs, kys;
            if(pnp.inliers.size()<100) {
                FeatureMatches matches = matcher->matchKeyPointsWithProjection(featureMatches.getFp1(), featureMatches.getFp2(), pnp.T);
                featureMatchesToPoints(matches, kxs, kys);

                ba = baRegistration.bundleAdjustment(pnp.T, matches.getCx(), matches.getCy(), kxs, kys, true);
            }else {
                featureIndexesToPoints(featureMatches.getKx(), pnp.kps1, kxs);
                featureIndexesToPoints(featureMatches.getKy(), pnp.kps2, kys);

                ba = baRegistration.bundleAdjustment(pnp.T, featureMatches.getCx(), featureMatches.getCy(), kxs, kys);
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
        cout << "-------------------" << featureMatches.getFIndexX() << "-" << featureMatches.getFIndexY()  << "-pnp+ba---------------------------------" << endl;
        pnp.printReport();
        if (ba.success) {
            ba.printReport();
        }
        printMutex.unlock();
    }

    void LocalRegistration::registrationLocalEdges(shared_ptr<Frame> curFrame) {
        clock_t start = clock();
        overlapFrames.clear();
        edges.clear();

        const int k = globalConfig.overlapNum;
        const int lastNum = 1;
        const int curIndex = localViewGraph.getNodesNum();

        set<int> spAlreadyAddedKF;
        overlapFrames.reserve(k);
        edges.resize(k, ConnectionCandidate::UNREACHABLE);
        thread *threads[k];
        cudaStream_t streams[k];
        // last frame
        int index = 0;
        for (int i = 1; i <= lastNum && i <= k && i <= curIndex; i++) {
            int refIndex = curIndex - i;
            spAlreadyAddedKF.insert(refIndex);
            overlapFrames.emplace_back(refIndex);

//            cudaStreamCreate(&streams[index]);
//            threads[index] = new thread(
//                    bind(&LocalRegistration::registrationPairEdge, this, placeholders::_1, placeholders::_2,
//                         placeholders::_3, placeholders::_4), &frames[refIndex]->getFirstFrame()->getKps(), &frames[curIndex]->getRootFrame()->getKps(), &edges[index], streams[index]);
            registrationPairEdge(&localViewGraph[refIndex]->getRootFrame()->getKps(), &curFrame->getKps(), &edges[index], stream);
            index++;
        }

        if (overlapFrames.size() < k) {
            std::vector<MatchScore> imageScores = localDBoWHashing->queryImages(make_float3(0,0,0), curFrame->getKps());
            // Return all those viewclusters with a score higher than 0.75*bestScore
            float minScoreToRetain = globalConfig.minScore;
            std::sort(imageScores.begin(), imageScores.end(), [=](MatchScore& ind1, MatchScore& ind2) {return ind1.imageId < ind2.imageId;});
            for (auto it: imageScores) {
                const float &si = it.score;
                if (si >= minScoreToRetain) {
                    int refIndex = it.imageId;
                    if (!spAlreadyAddedKF.count(refIndex)) {
                        overlapFrames.emplace_back(refIndex);
                        spAlreadyAddedKF.insert(refIndex);
//                        cudaStreamCreate(&streams[index]);
//                        threads[index] = new thread(
//                                bind(&LocalRegistration::registrationPairEdge, this, placeholders::_1, placeholders::_2,
//                                     placeholders::_3, placeholders::_4), &frames[refIndex]->getFirstFrame()->getKps(), &frames[curIndex]->getRootFrame()->getKps(), &edges[index], streams[index]);
                        registrationPairEdge(&localViewGraph[refIndex]->getRootFrame()->getKps(), &curFrame->getKps(), &edges[index], stream);

                        index++;
                    }
                    if (overlapFrames.size() >= k) break;
                }
            }
        }

        /*for (int i = 0; i < index; i++) {
            threads[i]->join();
            CUDA_CHECKED_CALL(cudaStreamSynchronize(streams[i]));
            CUDA_CHECKED_CALL(cudaStreamDestroy(streams[i]));
        }*/

        for (int i=0; i<overlapFrames.size(); i++) {
            int refNodeIndex = overlapFrames[i];
            ConnectionCandidate& candidate = edges[i];
            if(candidate.isUnreachable()) continue;
            double cost = localViewGraph.getPathLenByNodeIndex(refNodeIndex) * candidate.getCost();
            shared_ptr<Frame> refFrame = localViewGraph[refNodeIndex]->getRootFrame();
            {
                vector<Point3D> points(candidate.getKys().begin(), candidate.getKys().end());

                float weight = PointUtil::computePointWeight(points, localViewGraph.getCamera());

                refFrame->addConnection(curFrame->getFrameIndex(), allocate_shared<FrameConnection>(Eigen::aligned_allocator<FrameConnection>(), refFrame, curFrame, weight, candidate.getSE(), cost));
            }

            {
                vector<Point3D> points(candidate.getKys().begin(), candidate.getKys().end());

                float weight = PointUtil::computePointWeight(points, localViewGraph.getCamera());

                curFrame->addConnection(refFrame->getFrameIndex(), allocate_shared<FrameConnection>(Eigen::aligned_allocator<FrameConnection>(), curFrame, refFrame, weight, candidate.getSE().inverse(), cost));
            }
        }

        cout << "pair registration:" << double(clock() - start) / CLOCKS_PER_SEC << "s" << endl;
    }

    void LocalRegistration::updateCorrelations(int lastNodeIndex) {
        shared_ptr<Camera> camera = localViewGraph[lastNodeIndex]->getCamera();
        for(int i=0; i<overlapFrames.size(); i++) {
            ConnectionCandidate& edge = edges[i];
            int refNodeIndex = overlapFrames[i];
            if(!edge.isUnreachable()) {
                for(int k=0; k<edge.getKxs().size(); k++) {
                    FeatureKeypoint px = edge.getKxs()[k];
                    FeatureKeypoint py = edge.getKys()[k];

                    Vector3 qx = PointUtil::transformPoint(camera->getCameraModel()->unproject(px.x, px.y, px.z), localViewGraph[refNodeIndex]->getTransform());
                    Vector3 qy = PointUtil::transformPoint(camera->getCameraModel()->unproject(py.x, py.y, py.z), localViewGraph[lastNodeIndex]->getTransform());
                    if((qx-qy).norm()<globalConfig.maxPointError) {
                        int ix = startIndexes[refNodeIndex] + px.getIndex();
                        int iy = startIndexes[lastNodeIndex] + py.getIndex();

                        if(correlations[ix].empty()&&correlations[iy].empty()) kpNum++;

                        correlations[ix].emplace_back(make_pair(iy, PointUtil::transformPixel(py, localViewGraph[lastNodeIndex]->getTransform(), camera)));
                        correlations[iy].emplace_back(make_pair(ix, PointUtil::transformPixel(px, localViewGraph[refNodeIndex]->getTransform(), camera)));
                    }
                }
            }
        }
    }

    void LocalRegistration::updateLocalEdges(shared_ptr<Frame> frame) {
        int minFrameIndex = localViewGraph.getFirstFrame()->getFrameIndex();
        for(auto& con: frame->getConnections()) {
            int headFrameIndex = con->getHead()->getFrameIndex();
            int tailFrameIndex = con->getTail()->getFrameIndex();
            if(tailFrameIndex<minFrameIndex) continue;

            int curNodeIndex = localViewGraph.findNodeIndexByFrameIndex(headFrameIndex);
            int refNodeIndex = localViewGraph.findNodeIndexByFrameIndex(tailFrameIndex);
            shared_ptr<ViewCluster> curNode = localViewGraph[curNodeIndex];
            shared_ptr<ViewCluster> refNode = localViewGraph[refNodeIndex];

            refNode->addConnection(curNodeIndex, allocate_shared<ViewConnection>(Eigen::aligned_allocator<ViewConnection>(), refNode, curNode, con->getPointWeight(), con->getSE().inverse(), con->getCost()));
            curNode->addConnection(refNodeIndex, allocate_shared<ViewConnection>(Eigen::aligned_allocator<ViewConnection>(), curNode, refNode, con->getPointWeight(), con->getSE(), con->getCost()));
        }
    }

    void LocalRegistration::extendFrame(shared_ptr<Frame> frame) {
        shared_ptr<ViewCluster> cluster = allocate_shared<ViewCluster>(Eigen::aligned_allocator<ViewCluster>());
        cluster->addFrame(frame);
        localViewGraph.extendNode(cluster);
        startIndexes.emplace_back(correlations.size());
        correlations.resize(correlations.size()+frame->getKps().size());
        localDBoWHashing->addVisualIndex(make_float3(0,0,0), frame->getKps(), localViewGraph.getNodesNum() - 1);

        if(localViewGraph.getNodesNum()>1) {
            updateLocalEdges(frame);
            int lastNodeIndex = localViewGraph.findNodeIndexByFrameIndex(frame->getFrameIndex());
            localViewGraph.updateSpanningTree(lastNodeIndex);
            updateCorrelations(lastNodeIndex);
        }

    }



    void LocalRegistration::localTrack(shared_ptr<Frame> frame) {
        siftVocabulary->computeBow(frame->getKps());

        if (localViewGraph.getFramesNum() > 0) {
            registrationLocalEdges(frame);
        }

        localViewGraph.addSourceFrame(frame);
        extendFrame(frame);
    }

    void collectCorrespondences(vector<vector<pair<int, Point3D>>>& correlations, vector<bool>& visited, int u, vector<int>& corrIndexes, vector<Point3D>& corr) {
        for(int i=0; i<correlations[u].size(); i++) {
            int v = correlations[u][i].first;
            if(!visited[v]) {
                visited[v] = true;
                corrIndexes.emplace_back(v);
                corr.emplace_back(correlations[u][i].second);
                collectCorrespondences(correlations, visited, v, corrIndexes, corr);
            }
        }
    }

    bool LocalRegistration::needMerge() {
        const int n = localViewGraph.getNodesNum();
        if(n<=0) return false;
        bool c1 = localViewGraph[n-1]->getIndex() - localViewGraph[0]->getIndex() + 1>=globalConfig.chunkSize;
        bool c2 = localViewGraph[n-1]->getIndex() - localViewGraph[0]->getIndex() + 1>=2*globalConfig.chunkSize;
        bool c3 = kpNum > 1000;

        return (c1&&c3)||c2;
    }

    bool LocalRegistration::isRemain() {
        return localViewGraph.getNodesNum()>0;
    }

    shared_ptr<ViewCluster> LocalRegistration::mergeFramesIntoCluster() {
        localViewGraph.print();
        //1. local optimization
        const int n = localViewGraph.getNodesNum();
//        localViewGraph.optimizeBestRootNode();
        Optimizer::poseGraphOptimizeCeres(localViewGraph);
        shared_ptr<Frame> lastFrame = localViewGraph.getLastFrame();

        // 2. initialize key frame
        vector<int> cc = localViewGraph.maxConnectedComponent();
        assert(cc.size()>=1);
        set<int> visibleSet(cc.begin(), cc.end());
        shared_ptr<ViewCluster> viewcluster = allocate_shared<ViewCluster>(Eigen::aligned_allocator<ViewCluster>());
        viewcluster->setTransform(Transform::Identity());
        viewcluster->setRootIndex(localViewGraph.getMaxRoot());
        for(int i=0; i<n; i++) {
            shared_ptr<Frame> frame = localViewGraph[i]->getRootFrame();
            frame->setVisible(visibleSet.count(i));
            frame->setTransform(localViewGraph[i]->getTransform());
            // compute path length in MST
            int pathLen = localViewGraph.getPathLenByFrameIndex(frame->getFrameIndex());
            viewcluster->addFrame(frame, pathLen);
        }

        // reset
        for(shared_ptr<Frame> frame: viewcluster->getFrames()) {
            frame->releaseImages();
        }
        localViewGraph.reset(0);
        localDBoWHashing->clear();

        kpNum = 0;
        startIndexes.clear();
        correlations.clear();
        overlapFrames.clear();
        edges.clear();

        return viewcluster;
    }

    ViewGraph& LocalRegistration::getViewGraph() {
        return localViewGraph;
    }

    LocalRegistration::~LocalRegistration() {
        delete matcher;
        delete localDBoWHashing;
        delete pnpRegistration;
    }
}




