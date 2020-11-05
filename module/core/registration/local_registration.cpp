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
    LocalRegistration::LocalRegistration(const GlobalConfig &globalConfig, SIFTVocabulary* siftVocabulary): globalConfig(globalConfig) {
        localViewGraph.reset(0);
        matcher = new SIFTFeatureMatcher();
        localDBoWHashing = new DBoWHashing(globalConfig, siftVocabulary, false);
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

        /*printMutex.lock();
        cout << "-------------------" << featureMatches->getFIndexX() << "-" << featureMatches->getFIndexY()  << "-eg+ba---------------------------------" << endl;
        eg.printReport();
        if (ba.success) {
            ba.printReport();
        }
        printMutex.unlock();*/
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

        /*printMutex.lock();
        cout << "-------------------" << featureMatches->getFIndexX() << "-" << featureMatches->getFIndexY()  << "-homo+ba---------------------------------" << endl;
        homo.printReport();
        if (ba.success) {
            ba.printReport();
        }
        printMutex.unlock();*/
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

        /*printMutex.lock();
        cout << "-------------------" << featureMatches->getFIndexX() << "-" << featureMatches->getFIndexY()  << "-pnp+ba---------------------------------" << endl;
        pnp.printReport();
        if (ba.success) {
            ba.printReport();
        }
        printMutex.unlock();*/
    }

    void LocalRegistration::registrationPairEdge(FeatureMatches featureMatches, Edge *edge, cudaStream_t curStream) {
        stream = curStream;
        registrationPnPBA(&featureMatches, edge, curStream);
        if(edge->isUnreachable()&&featureMatches.size()>globalConfig.kMinMatches*2) {
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

            FeatureMatches featureMatches = matcher->matchKeyPointsPair(frames[refIndex]->getKps(),
                                                                        frames[curIndex]->getKps());
            cudaStreamCreate(&streams[index]);
            threads[index] = new thread(
                    bind(&LocalRegistration::registrationPairEdge, this, placeholders::_1, placeholders::_2,
                         placeholders::_3), featureMatches, &edges[index], streams[index]);
            index++;
        }

        if (overlapFrames.size() < k) {
            auto cur = localViewGraph.getSourceFrames()[curIndex];
            std::vector<MatchScore> imageScores = localDBoWHashing->queryImages(make_float3(0,0,0), cur->getKps());
            // Return all those keyframes with a score higher than 0.75*bestScore
            float minScoreToRetain = globalConfig.minScore;
            std::sort(imageScores.begin(), imageScores.end(), [=](MatchScore& ind1, MatchScore& ind2) {return ind1.imageId < ind2.imageId;});
            for (auto it: imageScores) {
                const float &si = it.score;
                if (si >= minScoreToRetain) {
                    int refIndex = it.imageId;
                    if (!spAlreadyAddedKF.count(refIndex)) {
                        FeatureMatches featureMatches = matcher->matchKeyPointsPair(frames[refIndex]->getKps(),
                                                                                    frames[curIndex]->getKps());
                        overlapFrames.emplace_back(refIndex);
                        spAlreadyAddedKF.insert(refIndex);
                        cudaStreamCreate(&streams[index]);
                        threads[index] = new thread(
                                bind(&LocalRegistration::registrationPairEdge, this, placeholders::_1, placeholders::_2,
                                     placeholders::_3), featureMatches, &edges[index], streams[index]);
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

    void LocalRegistration::localTrack(shared_ptr<KeyFrame> keyframe) {
        localViewGraph.extendNode(keyframe);
        if (localViewGraph.getFramesNum() > 1) {
            updateLocalEdges();
        }
        localDBoWHashing->addVisualIndex(make_float3(0,0,0), keyframe->getKps(), localViewGraph.getFramesNum()-1);
    }

    void transformFeaturePoint(FeatureKeypoint &fp, shared_ptr<Camera> camera, Transform trans) {
        Vector3 point = camera->getCameraModel()->unproject(fp.x, fp.y, fp.z);
        Rotation R;
        Translation t;
        GeoUtil::T2Rt(trans, R, t);
        Vector3 transPoint = R*point+t;
        Vector3 pixel = camera->getK()*transPoint;

        fp.x = pixel.x()/pixel.z();
        fp.y = pixel.y()/pixel.z();
        fp.z = pixel.z();
    }

    void dfs(ViewGraph& viewGraph, vector<bool>& visited, vector<vector<bool>>& done, int i, int j, int k, int &count, TransformVector gtTrans) {
        FeatureKeypoint fp = viewGraph.getEdge(i,j).getMatchKeypoint(k);
        if(done[j][fp.getIndex()]) return;
        visited[j] = true;
        done[j][fp.getIndex()] = true;
        count++;
        for(int m=0; m<viewGraph.getNodesNum(); m++) {
            if(visited[m]) continue;
            Edge edge = viewGraph.getEdge(j, m);
            if(!edge.isUnreachable()&&edge.containKeypoint(fp.getIndex())) {
                dfs(viewGraph, visited, done, j, m, fp.getIndex(), count, gtTrans);
            }
        }
    }

    shared_ptr<KeyFrame> LocalRegistration::mergeFramesIntoKeyFrame() {
        //1. local optimization
        const int n = localViewGraph.getNodesNum();
        vector<int> cc(n);
        iota(cc.begin(), cc.end(), 0);

        TransformVector gtTransVec;
        findShortestPathTransVec(localViewGraph, cc, gtTransVec);

        BARegistration baRegistration(globalConfig);
        baRegistration.multiViewBundleAdjustment(localViewGraph, cc, gtTransVec);

        // 2. initialize key frame
        shared_ptr<KeyFrame> keyframe = allocate_shared<KeyFrame>(Eigen::aligned_allocator<KeyFrame>());
        keyframe->setTransform(Transform::Identity());
        for(int i=0; i<n; i++) {
            shared_ptr<FrameRGBDT> frame = localViewGraph[i].getFrames()[0]->getFrames()[0];
            frame->setTransform(gtTransVec[i]);
            keyframe->addFrame(frame);
        }

        // 2. collect matching key points
        SIFTFeaturePoints &sf = keyframe->getKps();
        sf.setCamera(localViewGraph[0].getCamera());
        sf.setFIndex(localViewGraph[0].getIndex());
        FeatureKeypoints& kps = sf.getKeyPoints();
        FeatureDescriptors<uint8_t>& descriptors = sf.getDescriptors();
        vector<vector<bool>> done(n);
        for(int i=0; i<n; i++) {
            shared_ptr<KeyFrame> kf = localViewGraph[i].getFrames()[0];
            done[i].resize(kf->getKps().getKeyPoints().size(), false);
        }
        // foreach edge
        map<int, int> counter;
        vector<int> selectedNodes;
        vector<int> selectedIndexes;
        for(int i=0; i<n; i++) {
            for(int j=i+1; j<n; j++) {
                Edge edge = localViewGraph.getEdge(i, j);
                if(!edge.isUnreachable()) {
                    for(FeatureKeypoint fp: edge.getKxs()) {
                        int k = fp.getIndex();
                        if(!done[i][k]) {
                            vector<bool> visit(n, false);
                            visit[i] = true;
                            done[i][k] = true;
                            int count = 1;
                            dfs(localViewGraph, visit, done, i, j, k, count, gtTransVec);
                            if(count>2) {
                                transformFeaturePoint(fp, localViewGraph[i].getCamera(), gtTransVec[i]);
                                fp.setIndex(kps.size());
                                kps.emplace_back(make_shared<FeatureKeypoint>(fp));
                                selectedNodes.emplace_back(i);
                                selectedIndexes.emplace_back(k);
                            }
                        }
                    }
                }
            }
        }

        const int kpNum = selectedIndexes.size();
        cout << "kpNUm:" << kpNum << endl;
        descriptors.resize(kpNum, 128);
        for(int i=0; i<kpNum; i++) {
            descriptors.row(i) = localViewGraph[selectedNodes[i]].getFrames()[0]->getKps().getDescriptors().row(selectedIndexes[i]);
        }
        localDBoWHashing->computeBow(sf);

        // reset
        localViewGraph.reset(0);
        localDBoWHashing->clear();

        return keyframe;
    }

    LocalRegistration::~LocalRegistration() {
        delete matcher;
        delete localDBoWHashing;
        delete egRegistration;
        delete homoRegistration;
        delete pnpRegistration;
    }
}




