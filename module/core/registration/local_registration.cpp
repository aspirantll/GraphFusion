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

    void LocalRegistration::registrationLocalEdges() {
        const int k = globalConfig.overlapNum;
        const int lastNum = 2;
        const int curIndex = localViewGraph.getNodesNum()-1;

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
            registrationPairEdge(&localViewGraph[refIndex]->getRootFrame()->getKps(), &localViewGraph[curIndex]->getRootFrame()->getKps(), &edges[index], stream);
            index++;
        }

        if (overlapFrames.size() < k) {
            auto cur = localViewGraph[curIndex];
            std::vector<MatchScore> imageScores = localDBoWHashing->queryImages(make_float3(0,0,0), cur->getKps());
            // Return all those keyframes with a score higher than 0.75*bestScore
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
                        registrationPairEdge(&localViewGraph[refIndex]->getRootFrame()->getKps(), &localViewGraph[curIndex]->getRootFrame()->getKps(), &edges[index], stream);

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
    }

    void LocalRegistration::updateCorrelations() {
        int curNodeIndex = localViewGraph.getNodesNum()-1;
        shared_ptr<Camera> camera = localViewGraph[curNodeIndex]->getCamera();
        for(int i=0; i<edges.size(); i++) {
            ConnectionCandidate& edge = edges[i];
            int refNodeIndex = overlapFrames[i];
            if(!edge.isUnreachable()) {
                for(int k=0; k<edge.getKxs().size(); k++) {
                    FeatureKeypoint px = edge.getKxs()[k];
                    FeatureKeypoint py = edge.getKys()[k];

                    Vector3 qx = PointUtil::transformPoint(camera->getCameraModel()->unproject(px.x, px.y, px.z), localViewGraph[refNodeIndex]->getTransform());
                    Vector3 qy = PointUtil::transformPoint(camera->getCameraModel()->unproject(py.x, py.y, py.z), localViewGraph[curNodeIndex]->getTransform());
                    if((qx-qy).norm()<globalConfig.maxPointError) {
                        int ix = startIndexes[refNodeIndex] + px.getIndex();
                        int iy = startIndexes[curNodeIndex] + py.getIndex();

                        if(correlations[ix].empty()&&correlations[iy].empty()) kpNum++;

                        correlations[ix].emplace_back(make_pair(iy, PointUtil::transformPixel(py, localViewGraph[curNodeIndex]->getTransform(), camera)));
                        correlations[iy].emplace_back(make_pair(ix, PointUtil::transformPixel(px, localViewGraph[refNodeIndex]->getTransform(), camera)));
                    }
                }
            }
        }
    }

    void LocalRegistration::updateLocalEdges() {
        clock_t start = clock();
        overlapFrames.clear();
        edges.clear();
        registrationLocalEdges();

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
            ConnectionCandidate& candidate = edges[ind];

            shared_ptr<ViewCluster> curNode = localViewGraph[curNodeIndex];
            shared_ptr<ViewCluster> refNode = localViewGraph[refNodeIndex];

            if(localViewGraph.existEdge(refNodeIndex, curNodeIndex)) {
                shared_ptr<Connection> edge = localViewGraph(refNodeIndex, curNodeIndex);
                if(candidate.getCost() < edge->getCost()) {
                    {
                        Vector3 p;
                        float weight;
                        vector<Point3D> points(candidate.getKys().begin(), candidate.getKys().end());

                        PointUtil::meanFeatures(points, localViewGraph.getCamera(), p, weight);

                        edge->setNormPoint(p);
                        edge->setPointWeight(weight);
                        edge->setTransform(candidate.getTransform());
                        edge->setCost(candidate.getCost());
                    }

                    {
                        shared_ptr<Connection> rEdge = localViewGraph(curNodeIndex, refNodeIndex);
                        Vector3 p;
                        float weight;
                        vector<Point3D> points(candidate.getKxs().begin(), candidate.getKxs().end());

                        PointUtil::meanFeatures(points, localViewGraph.getCamera(), p, weight);

                        rEdge->setNormPoint(p);
                        rEdge->setPointWeight(weight);
                        rEdge->setSE(candidate.getSE().inverse());
                        rEdge->setCost(candidate.getCost());
                    }
                }
            }else {
                {
                    Vector3 p;
                    float weight;
                    vector<Point3D> points(candidate.getKys().begin(), candidate.getKys().end());

                    PointUtil::meanFeatures(points, localViewGraph.getCamera(), p, weight);

                    localViewGraph[refNodeIndex]->addConnection(curNodeIndex, allocate_shared<Connection>(Eigen::aligned_allocator<Connection>(), curNode, p, weight, candidate.getSE(), candidate.getCost()));
                }

                {
                    Vector3 p;
                    float weight;
                    vector<Point3D> points(candidate.getKxs().begin(), candidate.getKxs().end());

                    PointUtil::meanFeatures(points, localViewGraph.getCamera(), p, weight);

                    localViewGraph[curNodeIndex]->addConnection(refNodeIndex, allocate_shared<Connection>(Eigen::aligned_allocator<Connection>(), refNode, p, weight, candidate.getSE().inverse(), candidate.getCost()));
                }
            }

        }

        cout << "pair registration:" << double(clock() - start) / CLOCKS_PER_SEC << "s" << endl;

    }

    void LocalRegistration::localTrack(shared_ptr<Frame> frame) {
        shared_ptr<ViewCluster> keyframe = allocate_shared<ViewCluster>(Eigen::aligned_allocator<ViewCluster>());
        siftVocabulary->computeBow(frame->getKps());
        keyframe->addFrame(frame);
        keyframe->setKps(frame->getKps());
        localViewGraph.extendNode(keyframe);
        startIndexes.emplace_back(correlations.size());
        correlations.resize(correlations.size()+frame->getKps().size());
        if (localViewGraph.getNodesNum() > 1) {
            updateLocalEdges();
            localViewGraph.updateSpanningTree();
            updateCorrelations();
        }
        localDBoWHashing->addVisualIndex(make_float3(0,0,0), keyframe->getKps(), localViewGraph.getNodesNum()-1);
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
        bool c1 = localViewGraph[n-1]->getIndex() - localViewGraph[0]->getIndex() + 1>=globalConfig.chunkSize;
        bool c2 = localViewGraph[n-1]->getIndex() - localViewGraph[0]->getIndex() + 1>=2*globalConfig.chunkSize;
        bool c3 = kpNum > 1000;

        return (c1&&c3)||c2;
    }

    bool LocalRegistration::isRemain() {
        return localViewGraph.getNodesNum()>0;
    }

    shared_ptr<ViewCluster> LocalRegistration::mergeFramesIntoKeyFrame() {
        //1. local optimization
        const int n = localViewGraph.getNodesNum();
//        localViewGraph.optimizeBestRootNode();
        Optimizer::poseGraphOptimizeCeres(localViewGraph);

        // 2. initialize key frame
        vector<int> cc = localViewGraph.maxConnectedComponent();
        assert(cc.size()>=1);
        set<int> visibleSet(cc.begin(), cc.end());
        shared_ptr<ViewCluster> keyframe = allocate_shared<ViewCluster>(Eigen::aligned_allocator<ViewCluster>());
        keyframe->setTransform(Transform::Identity());
        keyframe->setRootIndex(localViewGraph.getMaxRoot());
        for(int i=0; i<n; i++) {
            shared_ptr<Frame> frame = localViewGraph[i]->getRootFrame();
            frame->setVisible(visibleSet.count(i));
            // compute path length in MST
            int pathLen = localViewGraph.getPathLen(frame->getFrameIndex());
            keyframe->addFrame(frame, pathLen);
        }

        //3. collect keypoints
        SIFTFeaturePoints &sf = keyframe->getKps();
        int m = cc.size();
        if(m > 1) {
            Optimizer::poseGraphOptimizeCeres(localViewGraph);
            // update transforms
            vector<shared_ptr<Frame>>& frames = keyframe->getFrames();
            for(int i=0; i<cc.size(); i++) {
                frames[cc[i]]->setTransform(localViewGraph[cc[i]]->getTransform());
            }

            sf.setCamera(localViewGraph.getCamera());
            sf.setFIndex(localViewGraph[0]->getIndex());

            // foreach edge
            shared_ptr<CameraModel> cameraModel = localViewGraph.getCamera()->getCameraModel();
            vector<bool> visited(correlations.size(), false);
            FeatureKeypoints kp;
            vector<Eigen::Matrix<uint8_t, 1, -1, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::Matrix<uint8_t, 1, -1, Eigen::RowMajor>>> desc;
            Scalar minX=numeric_limits<Scalar>::infinity(), maxX=0, minY=numeric_limits<Scalar>::infinity(), maxY=0;
            for(int i=0; i<m; i++) {
                int nodeIndex = cc[i];
                SIFTFeaturePoints &sift = localViewGraph[nodeIndex]->getRootFrame()->getKps();
                for(int j=0; j<sift.getKeyPoints().size(); j++) {
                    int curIndex = startIndexes[nodeIndex]+j;
                    if(!visited[curIndex]) {
                        shared_ptr<SIFTFeatureKeypoint> fp = make_shared<SIFTFeatureKeypoint>(*dynamic_pointer_cast<SIFTFeatureKeypoint>(sift.getKeyPoints()[j]));
                        vector<Point3D> corr;
                        vector<int> corrIndexes;

                        collectCorrespondences(correlations, visited, curIndex, corrIndexes, corr);
                        if(!corr.empty()) {
                            Vector3 point = Vector3::Zero();
                            for(const Point3D& c: corr) {
                                point += c.toVector3();
                            }
                            point /= corr.size();
                            fp->x = point.x();
                            fp->y = point.y();
                            fp->z = point.z();

                            kp.emplace_back(fp);
                            desc.emplace_back(sift.getDescriptors().row(j));

                            minX = min(fp->x, minX);
                            maxX = max(fp->x, maxX);
                            minY = min(fp->y, minY);
                            maxY = max(fp->y, maxY);
                        }
                    }
                }
            }

            sf.setBounds(minX, maxX, minY, maxY);
            sf.fuseFeaturePoints(kp, desc);
        }else if(m==1){
            sf = localViewGraph[cc[0]]->getRootFrame()->getKps();
        }



        cout << "----------------------------------------" << endl;
        cout << "frame index:" << keyframe->getIndex() << endl;
        cout << "visible frame num:" << m << endl;
        cout << "feature num:" << sf.size() << endl;

        // reset
        for(shared_ptr<Frame> frame: keyframe->getFrames()) {
            frame->releaseImages();
        }
        localViewGraph.reset(0);
        localDBoWHashing->clear();

        kpNum = 0;
        startIndexes.clear();
        correlations.clear();

        return keyframe;
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




