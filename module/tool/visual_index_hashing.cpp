//
// Created by liulei on 2020/9/23.
//

#include "visual_index_hashing.h"
#include "../core/solver/cuda_math.h"

#define EQUAL3(pos1, pos2) \
     (pos1.x==pos2.x&&pos1.y==pos2.y&&pos1.z==pos2.z)


namespace rtf {
    void toDescriptorVector(SIFTFeatureDescriptors & desc, vector<vector<float>>&converted) {
        converted.reserve(desc.rows());
        for(int i=0; i<desc.rows(); i++) {
            vector<float> row(desc.cols());
            for(int j=0; j<desc.cols(); j++) {
                row[j] = desc(i, j);
            }
            converted.emplace_back(row);
        }
    }

    void SIFTVocabulary::computeBow(SIFTFeaturePoints& sf) {
        auto & bowVec = sf.getMBowVec();
        auto & featVec = sf.getMFeatVec();
        if(bowVec.empty()) {
            vector<vector<float>> vCurrentDesc;
            toDescriptorVector(sf.getDescriptors(), vCurrentDesc);
            transform(vCurrentDesc, bowVec, featVec, 4);
        }
    }
    MatchScore::MatchScore() {}
    MatchScore::MatchScore(int imageId, float score) : imageId(imageId), score(score) {}

    void DBoWVocabulary::add(int imageId, SIFTFeaturePoints* sf) {
        imageIds.emplace_back(imageId);
        cpuVoc.emplace_back(make_pair(imageId, sf));
        // upload gpu
        vector<uint> words = sf->getMBowVec().words();
        CUDAArrayu* cudaWords = new CUDAArrayu(words);
        ptrHolder.emplace_back(cudaWords);

        CUDABoW cudaBoW;
        cudaBoW.imageId = imageId;
        cudaBoW.words = *cudaWords;
        gpuVoc.addItem(cudaBoW);
    }

    void DBoWVocabulary::remove(int index) {
        imageIds.erase(imageIds.begin()+index);
        cpuVoc.erase(cpuVoc.begin()+index);
        delete ptrHolder[index];
        ptrHolder.erase(ptrHolder.begin()+index);
        gpuVoc.removeItem(index);
    }

    int DBoWVocabulary::size() {
        return cpuVoc.size();
    }

    void DBoWVocabulary::clear() {
        for(auto *it: ptrHolder) {
            delete it;
        }

        ptrHolder.clear();
        cpuVoc.clear();
        gpuVoc.clear();
        imageIds.clear();
    }

    DBoWVocabulary::~DBoWVocabulary() {

    }

    DBoWHashing::DBoWHashing(const GlobalConfig &globalConfig, SIFTVocabulary * siftVocabulary, bool hashing): siftVocabulary(siftVocabulary) {
        config.vocTxtPath = globalConfig.vocTxtPath;
        config.numNeighs = globalConfig.numNeighs;
        config.numThreads = globalConfig.numThreads;
        config.hashing = hashing;
        initialize();
    }

    void DBoWHashing::initialize() {
        items = new HashItem[config.maxVINum];
        for (int i = 0; i < config.maxVINum; i++) {
            items[i].pos = make_int3(0, 0, 0);
            items[i].ptr = -1;
        }
        if(config.hashing) {
            featureCatas = new DBoWVocabulary[config.maxVINum];
            vocTh = 0;
        }

        featureCata = new DBoWVocabulary();
        prepared = false;

        queryPool = new ThreadPool(config.numThreads);
    }

    DBoWHashing::~DBoWHashing() {
        delete items;
        delete queryPool;
        for(int i=0; i<vocTh&&config.hashing; i++) {
            featureCatas[i].clear();
        }
        featureCata->clear();
    }

    uint DBoWHashing::hashFunction(int3 pos) {
        const int p0 = 73856093;
        const int p1 = 19349669;
        const int p2 = 83492791;

        int res = ((pos.x * p0) ^ (pos.y * p1) ^ (pos.z * p2)) %
                  config.maxVINum;
        if (res < 0) res += config.maxVINum;
        return (uint) res;
    }

    int3 DBoWHashing::worldToPos(float3 wPos) {
        const float3 p = wPos/config.voxelSize;
        return make_int3(p + make_float3(sign(p)) * 0.5f);
    }

    void DBoWHashing::addVisualIndex(float3 wPos, SIFTFeaturePoints &sf, int imageId, bool notLost) {
        auto* lVisualIndex = featureCata;
        if(config.hashing&&notLost) {
            // locate virtual index
            int3 pos = worldToPos(wPos);
            uint hashCode = hashFunction(pos);
            HashItem* item = items+hashCode;
            bool find = true;
            while(item->ptr!=-1&&!EQUAL3(item->pos, pos)) {
                uint offset = (item-items+1)%config.maxVINum;
                if(offset==hashCode) {
                    find = false;
                    break;
                }
                item = items + offset;
            }

            if(!find||item->ptr==-1) {
                while(item->ptr!=-1) {
                    uint offset = (item-items+1)%config.maxVINum;
                    if(offset==hashCode) {
                        throw runtime_error("no enough hash item for visual index");
                    }
                    item = items + offset;
                }
                item->pos = pos;
                item->ptr = vocTh;
                vocTh++;
            }
//            cout << "visual Index:" << item->ptr << endl;
            lVisualIndex = featureCatas + item->ptr;
        }else {
            prepared = true;
        }

        lVisualIndex->add(imageId, &sf);
    }


    void DBoWHashing::queryVisualIndex(vector<DBoWVocabulary*> vocs, SIFTFeaturePoints* sf, vector<MatchScore>* imageScores) {
        DBoW2::BowVector &bow = sf->getMBowVec();
        map<int, int> indexMap;
        map<int, int> sharedWordFrames;
        vector<int> startVec;
        int start = 0;
        for(int i=0; i<vocs.size(); i++) {
            DBoWVocabulary* voc = vocs[i];
            int num = voc->size();
            vector<uint> wordCounts(num, 0);
            {
                CUDAArrayu cudaWordCounts(num);
                vector<uint> curWords = sf->getMBowVec().words();
                CUDAArrayu cudaCurWords(curWords);
                CUDAPtrArray<CUDABoW> gpuVoc = voc->gpuVoc.uploadToCUDA();
                wordsCount(gpuVoc, cudaCurWords, cudaWordCounts);
                cudaWordCounts.download(wordCounts);
            }

            for(int j=0; j<num; j++) {
                if(wordCounts[j]) {
                    sharedWordFrames.insert(map<int,int>::value_type(start+j, wordCounts[j]));
                    indexMap.insert(map<int,int>::value_type(start+j, i));
                }
            }
            startVec.emplace_back(start);
            start += num;
        }



        // filter by maxWordCount and minWordCount
        int maxCommonWords = 0;
        for (auto &sharedWordNode : sharedWordFrames) {
            if (sharedWordNode.second > maxCommonWords) {
                maxCommonWords = sharedWordNode.second;
            }
        }
        int minCommonWords = maxCommonWords * 0.6f;
        imageScores->clear();
        for (auto &sharedWordFrame : sharedWordFrames) {
            if (sharedWordFrame.second >= minCommonWords) {
                int ind = sharedWordFrame.first;
                int vocInd = indexMap[ind];
                ind = ind - startVec[vocInd];
                auto item = (*vocs[vocInd]).cpuVoc[ind];
                DBoW2::BowVector &ibow = item.second->getMBowVec();
                float score = siftVocabulary->score(bow, ibow);
                imageScores->emplace_back(item.first, score);
            }
        }

        std::sort(imageScores->begin(), imageScores->end(), [=](MatchScore& ind1, MatchScore& ind2) {return ind1.score > ind2.score;});
    }

    vector<MatchScore> DBoWHashing::queryImages(float3 wPos, SIFTFeaturePoints& sf, bool notLost, bool hasLost) {
        set<int> viInds;
        if(config.hashing&&notLost) {
            viInds.clear();
            int3 pos = worldToPos(wPos);
            int deltas[3] = {-1, 0, 1};
            for(auto i: deltas) {
                for(auto j: deltas) {
                    for(auto k: deltas) {
                        int3 p = make_int3(pos.x+i, pos.y+j, pos.z+k);
                        uint hashCode = hashFunction(p);
                        HashItem* item = items+hashCode;
                        bool find = true;
                        while(item->ptr!=-1&&!EQUAL3(item->pos, pos)) {
                            uint offset = (item-items+1)%config.maxVINum;
                            if(offset==hashCode) {
                                find = false;
                                break;
                            }
                            item = items + offset;
                        }
                        if(find&&item->ptr!=-1) {
                            viInds.insert(item->ptr);
                        }
                    }
                }
            }
        }else if(!notLost&&config.hashing) {
            for(int i=vocTh-1; i>=0; i--) {
                viInds.insert(i);
            }
        }

        // when lost pose, need to search all images; and if prepared, then search featureCata
        vector<vector<MatchScore>> cMatchScores;
        vector<future<void>> invokeFutures;
        if(prepared) {// hava image in visual index
            cMatchScores.resize(viInds.empty()?1:2);
            vector<DBoWVocabulary*> vocs;
            vocs.emplace_back(featureCata);
            invokeFutures.emplace_back(queryPool->enqueue(bind(&DBoWHashing::queryVisualIndex
                    , this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), vocs, &sf, &cMatchScores[viInds.empty()?0:1]));
        }else {
            cMatchScores.resize(1);
        }

        vector<DBoWVocabulary*> vocs;
        for(uint viInd: viInds) {
            vocs.emplace_back(featureCatas+viInd);
        }
        if(!vocs.empty()) {
            invokeFutures.emplace_back(queryPool->enqueue(bind(&DBoWHashing::queryVisualIndex
                    , this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), vocs, &sf, &cMatchScores[0]));
        }

        for(int i=0; i<invokeFutures.size(); i++) {
            invokeFutures[i].wait();
        }

        vector<MatchScore> matchScore;
        vector<int> inds(cMatchScores.size(), 0);
        int start = 0;
        if(!notLost&&prepared) {
            for(start=0; start<2&&start<config.numNeighs; start++) {
                int maxJ = -1;
                float maxScore = 0;
                for(int j=0; j<cMatchScores.size()-1; j++) {
                    int k = inds[j];
                    if(k>=cMatchScores[j].size()) continue;
                    if(maxScore<cMatchScores[j][k].score) {
                        maxScore = cMatchScores[j][k].score;
                        maxJ = j;
                    }
                }
                if(maxJ==-1) break;
                matchScore.emplace_back(cMatchScores[maxJ][inds[maxJ]]);
                inds[maxJ]++;
            }
        }else if(hasLost&&prepared) {
            int maxJ = cMatchScores.size()-1;
            for(start=0; start<1&&start<cMatchScores[maxJ].size()&&start<config.numNeighs; start++) {
                matchScore.emplace_back(cMatchScores[maxJ][inds[maxJ]]);
                inds[maxJ]++;
            }
        }

        for(int i=start; i<config.numNeighs; i++) {
            int maxJ = -1;
            float maxScore = 0;
            for(int j=0; j<cMatchScores.size(); j++) {
                int k = inds[j];
                if(k>=cMatchScores[j].size()) continue;
                if(maxScore<cMatchScores[j][k].score) {
                    maxScore = cMatchScores[j][k].score;
                    maxJ = j;
                }
            }
            if(maxJ==-1) break;
            matchScore.emplace_back(cMatchScores[maxJ][inds[maxJ]]);
            inds[maxJ]++;
        }

        return matchScore;
    }


    vector<int> DBoWHashing::lostImageIds() {
        return featureCata->imageIds;
    }

    void DBoWHashing::updateVisualIndex(vector<int> imageIds, vector<float3> poses) {
        for(int i=0; i<imageIds.size(); i++) {
            vector<int>& curIds = featureCata->imageIds;
            for(vector<int>::const_iterator cit=curIds.begin(); cit!=curIds.end(); cit++) {
                if(*cit==imageIds[i]) {
                    int j = cit-curIds.begin();
                    auto cpuFeature = featureCata->cpuVoc[j];
                    featureCata->remove(j);
                    addVisualIndex(poses[i], *cpuFeature.second, cpuFeature.first);
                    break;
                }
            }
        }
        if(featureCata->imageIds.empty()) {
            prepared = false;
        }
    }


    void DBoWHashing::clear() {
        featureCata->clear();
        if(config.hashing) {
            for(int i=0; i<vocTh; i++) {
                featureCatas[i].clear();
            }
        }
    }
}