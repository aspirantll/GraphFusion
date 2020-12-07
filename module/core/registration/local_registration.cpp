//
// Created by liulei on 2020/7/9.
//

#include "registrations.h"
#include "optimizer.h"
#include "../../tool/view_graph_util.h"
#include "../../processor/downsample.h"
#include <glog/logging.h>

#include <utility>

using namespace rtf::ViewGraphUtil;

namespace rtf {
    LocalRegistration::LocalRegistration(const GlobalConfig &globalConfig, SIFTVocabulary* siftVocabulary): globalConfig(globalConfig), siftVocabulary(siftVocabulary) {
        localViewGraph.reset(0);
        matcher = new SIFTFeatureMatcher();
        localDBoWHashing = new DBoWHashing(globalConfig, siftVocabulary, false);
        pnpRegistration = new PnPRegistration(globalConfig);
    }

    void LocalRegistration::registrationPnPBA(FeatureMatches *featureMatches, Edge *edge) {
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
        cout << "-------------------" << featureMatches->getFIndexX() << "-" << featureMatches->getFIndexY()  << "-pnp+ba---------------------------------" << endl;
        pnp.printReport();
        if (ba.success) {
            ba.printReport();
        }
        printMutex.unlock();
    }

    void LocalRegistration::registrationWithMotion(SIFTFeaturePoints& f1, SIFTFeaturePoints& f2, Edge* edge) {
        if(velocity(3, 3)!=1) return;

        FeatureMatches featureMatches = matcher->matchKeyPointsWithProjection(f1, f2, velocity);

        vector<FeatureKeypoint> kxs, kys;
        featureMatchesToPoints(featureMatches, kxs, kys);

        BARegistration baRegistration(globalConfig);
        RegReport ba = baRegistration.bundleAdjustment(velocity, featureMatches.getCx(), featureMatches.getCy(), kxs, kys, true);
        if (ba.success) {
            double cost = ba.avgCost();
            if (!isnan(cost) && cost < globalConfig.maxAvgCost) {
                edge->setKxs(kxs);
                edge->setKys(kys);
                edge->setTransform(ba.T);
                edge->setCost(cost);
            }
        }

        printMutex.lock();
        if (ba.success) {
            cout << "-------------------" << f1.getFIndex() << "-" << f2.getFIndex()  << "-motion+ba---------------------------------" << endl;
            ba.printReport();
        }
        printMutex.unlock();
    }

    void LocalRegistration::registrationPairEdge(SIFTFeaturePoints* f1, SIFTFeaturePoints* f2, Edge *edge, cudaStream_t curStream) {
        stream = curStream;
        FeatureMatches featureMatches = matcher->matchKeyPointsPair(*f1, *f2);
        registrationPnPBA(&featureMatches, edge);
        bool near = f2->getFIndex()-f1->getFIndex() <=1;
        if(near&&edge->isUnreachable()) {
            registrationWithMotion(*f1, *f2, edge);
        }

        if(near&&!edge->isUnreachable()) {
            velocity = edge->getTransform();
        }else if(near){
            velocity.setZero();
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

//            cudaStreamCreate(&streams[index]);
//            threads[index] = new thread(
//                    bind(&LocalRegistration::registrationPairEdge, this, placeholders::_1, placeholders::_2,
//                         placeholders::_3, placeholders::_4), &frames[refIndex]->getFirstFrame()->getKps(), &frames[curIndex]->getFirstFrame()->getKps(), &edges[index], streams[index]);
            registrationPairEdge(&frames[refIndex]->getFirstFrame()->getKps(), &frames[curIndex]->getFirstFrame()->getKps(), &edges[index], stream);
            index++;
        }

        if (overlapFrames.size() < k) {
            auto cur = localViewGraph.indexFrame(curIndex);
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
//                                     placeholders::_3, placeholders::_4), &frames[refIndex]->getFirstFrame()->getKps(), &frames[curIndex]->getFirstFrame()->getKps(), &edges[index], streams[index]);
                        registrationPairEdge(&frames[refIndex]->getFirstFrame()->getKps(), &frames[curIndex]->getFirstFrame()->getKps(), &edges[index], stream);

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
        shared_ptr<Camera> camera = localViewGraph[curNodeIndex].getCamera();
        for(int refNodeIndex=0; refNodeIndex<curNodeIndex; refNodeIndex++) {
            Edge edge = localViewGraph.getEdge(refNodeIndex, curNodeIndex);
            if(!edge.isUnreachable()) {
                for(int k=0; k<edge.getKxs().size(); k++) {
                    FeatureKeypoint px = edge.getKxs()[k];
                    FeatureKeypoint py = edge.getKys()[k];

                    Vector3 qx = PointUtil::transformPoint(camera->getCameraModel()->unproject(px.x, px.y, px.z), localViewGraph[refNodeIndex].getGtTransform());
                    Vector3 qy = PointUtil::transformPoint(camera->getCameraModel()->unproject(py.x, py.y, py.z), localViewGraph[curNodeIndex].getGtTransform());
                    if((qx-qy).norm()<globalConfig.maxPointError) {
                        int ix = startIndexes[refNodeIndex] + px.getIndex();
                        int iy = startIndexes[curNodeIndex] + py.getIndex();

                        if(correlations[ix].empty()&&correlations[iy].empty()) kpNum++;
                        correlations[ix].emplace_back(make_pair(iy, PointUtil::transformPixel(py, localViewGraph[curNodeIndex].getGtTransform(), camera)));
                        correlations[iy].emplace_back(make_pair(ix, PointUtil::transformPixel(px, localViewGraph[refNodeIndex].getGtTransform(), camera)));
                    }
                }
            }
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

        cout << "pair registration:" << double(clock() - start) / CLOCKS_PER_SEC << "s" << endl;

    }

    void LocalRegistration::localTrack(shared_ptr<Frame> frame) {
        shared_ptr<KeyFrame> keyframe = allocate_shared<KeyFrame>(Eigen::aligned_allocator<KeyFrame>());
        siftVocabulary->computeBow(frame->getKps());
        keyframe->addFrame(frame);
        keyframe->setKps(frame->getKps());
        localViewGraph.extendNode(keyframe);
        startIndexes.emplace_back(correlations.size());
        correlations.resize(correlations.size()+frame->getKps().size());
        if (localViewGraph.getFramesNum() > 1) {
            updateLocalEdges();
            localViewGraph.updateSpanningTree();
            updateCorrelations();
        }
        localDBoWHashing->addVisualIndex(make_float3(0,0,0), keyframe->getKps(), localViewGraph.getFramesNum()-1);
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
        bool c1 = localViewGraph[n-1].getIndex() - localViewGraph[0].getIndex() + 1>=globalConfig.chunkSize;
        bool c2 = localViewGraph[n-1].getIndex() - localViewGraph[0].getIndex() + 1>=2*globalConfig.chunkSize;
        bool c3 = kpNum > 1000;

        return (c1&&c3)||c2;
    }

    bool LocalRegistration::isRemain() {
        return localViewGraph.getNodesNum()>0;
    }

    shared_ptr<KeyFrame> LocalRegistration::mergeFramesIntoKeyFrame() {
        //1. local optimization
        const int n = localViewGraph.getNodesNum();
        vector<vector<int>> connectedComponents = findConnectedComponents(localViewGraph, globalConfig.maxAvgCost);

        // 2. initialize key frame
        set<int> visibleSet(connectedComponents[0].begin(), connectedComponents[0].end());
        shared_ptr<KeyFrame> keyframe = allocate_shared<KeyFrame>(Eigen::aligned_allocator<KeyFrame>());
        keyframe->setTransform(Transform::Identity());
        for(int i=0; i<n; i++) {
            shared_ptr<Frame> frame = localViewGraph[i].getFrames()[0]->getFirstFrame();
            frame->setVisible(visibleSet.count(i));
            keyframe->addFrame(frame);
        }


        //3. collect keypoints
        SIFTFeaturePoints &sf = keyframe->getKps();
        int m = connectedComponents[0].size();
        if(m > 1) {
            cout << "multiview" << endl;
            BARegistration baRegistration(globalConfig);
            baRegistration.multiViewBundleAdjustment(localViewGraph, connectedComponents[0]).printReport();
//            Optimizer::globalBundleAdjustmentCeres(localViewGraph, connectedComponents[0]);
            // update transforms
            vector<shared_ptr<Frame>>& frames = keyframe->getFrames();
            for(int i=0; i<connectedComponents[0].size(); i++) {
                frames[connectedComponents[0][i]]->setTransform(localViewGraph[connectedComponents[0][i]].getGtTransform());
            }

            sf.setCamera(localViewGraph[0].getCamera());
            sf.setFIndex(localViewGraph[0].getIndex());
            FeatureKeypoints& kps = sf.getKeyPoints();
            FeatureDescriptors<uint8_t>& descriptors = sf.getDescriptors();

            // foreach edge
            shared_ptr<CameraModel> cameraModel = localViewGraph[0].getCamera()->getCameraModel();
            vector<double> scores;
            vector<bool> visited(correlations.size(), false);
            vector<Eigen::Matrix<uint8_t, 1, -1, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::Matrix<uint8_t, 1, -1, Eigen::RowMajor>>> desc;
            float minX=numeric_limits<float>::infinity(), maxX=0, minY=numeric_limits<float>::infinity(), maxY=0;
            for(int i=0; i<m; i++) {
                int nodeIndex = connectedComponents[0][i];
                SIFTFeaturePoints &sift = localViewGraph[nodeIndex].getFrames()[0]->getKps();
                for(int j=0; j<sift.getKeyPoints().size(); j++) {
                    int curIndex = startIndexes[nodeIndex]+j;
                    if(!visited[curIndex]) {
                        shared_ptr<SIFTFeatureKeypoint> fp = make_shared<SIFTFeatureKeypoint>(*dynamic_pointer_cast<SIFTFeatureKeypoint>(sift.getKeyPoints()[j]));
                        vector<Point3D> corr;
                        vector<int> corrIndexes;

                        collectCorrespondences(correlations, visited, curIndex, corrIndexes, corr);
                        if(!corr.empty()) {
                            Vector3 point = Vector3::Zero();
                            Vector3 mean = Vector3::Zero();
                            EigenVector(Vector3) points;
                            for(const Point3D& c: corr) {
                                point += c.toVector3();
                                points.emplace_back(cameraModel->unproject(c.x, c.y, c.z));
                                mean += points.back();
                            }
                            point /= corr.size();
                            mean /= corr.size();

                            double rms = 0;
                            for(const Vector3& p: points) {
                                rms += (p-mean).squaredNorm();
                            }

                            fp->x = point.x();
                            fp->y = point.y();
                            fp->z = point.z();

                            scores.emplace_back(sqrt(rms/corr.size()));
                            kps.emplace_back(fp);
                            desc.emplace_back(sift.getDescriptors().row(j));

                            minX = min(fp->x, minX);
                            maxX = max(fp->x, maxX);
                            minY = min(fp->y, minY);
                            maxY = max(fp->y, maxY);
                        }
                    }
                }
            }

            if(sf.size()>2400) {
                cout << "before:" << sf.size() <<endl;
                int gridSize = 20;
                int num = sf.size();
                int width = maxX-minX, height=maxY-minY;
                int rows = floor(width/gridSize), cols = floor(height/gridSize);

                map<int, vector<int>> gridGroup;
                for(int i=0; i<num; i++) {
                    auto kp = kps[i];
                    int gridx = floor(kp->x-minX/gridSize);
                    int gridy = floor(kp->y-minY/gridSize);
                    int gridIndex = gridx*cols+gridy;
                    if(!gridGroup.count(gridIndex)) {
                        gridGroup.insert(map<int, vector<int>>::value_type(gridIndex, vector<int>()));
                    }
                    gridGroup[gridIndex].emplace_back(i);
                }

                FeatureKeypoints bKps(kps.begin(), kps.end());
                vector<Eigen::Matrix<uint8_t, 1, -1, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::Matrix<uint8_t, 1, -1, Eigen::RowMajor>>> bDesc(desc.begin(), desc.end());

                kps.clear();
                desc.clear();
                for(auto mit: gridGroup) {
                    int selectedIndex = -1;
                    double score = numeric_limits<double>::infinity();
                    for(int ind: mit.second) {
                        if(scores[ind]<score) {
                            selectedIndex = ind;
                            score = scores[ind];
                        }
                    }
                    kps.emplace_back(bKps[selectedIndex]);
                    desc.emplace_back(bDesc[selectedIndex]);
                }
            }

            descriptors.resize(desc.size(), 128);
            for(int i=0; i<desc.size(); i++) {
                descriptors.row(i) = desc[i];
            }

            sf.setBounds(minX, maxX, minY, maxY);
            sf.assignFeaturesToGrid();
        }else if(m==1){
            sf = localViewGraph[connectedComponents[0][0]].getFrames()[0]->getKps();
        }



        cout << "----------------------------------------" << endl;
        cout << "frame index:" << keyframe->getIndex() << endl;
        cout << "visible frame num:" << m << endl;
        cout << "feature num:" << sf.size() << endl;

        // reset
        localViewGraph.reset(0);
        localDBoWHashing->clear();
        velocity.setZero();

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




