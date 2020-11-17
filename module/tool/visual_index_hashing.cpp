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

    void DBoWVocabulary::add(int imageId, DBoW2::BowVector* sf) {
        imageIds.emplace_back(imageId);
        cpuVoc.emplace_back(make_pair(imageId, sf));
        // upload gpu
        vector<uint> words = sf->words();
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

    void DBoWVocabulary::query(SIFTVocabulary* siftVocabulary, DBoW2::BowVector* bow, vector<MatchScore>* imageScores) {
        int num = size();

        vector<uint> wordCounts(num, 0);
        {
            CUDAArrayu cudaWordCounts(num);
            vector<uint> curWords = bow->words();
            CUDAArrayu cudaCurWords(curWords);
            CUDAPtrArray<CUDABoW> gpu = gpuVoc.uploadToCUDA();
            wordsCount(gpu, cudaCurWords, cudaWordCounts);
            cudaWordCounts.download(wordCounts);
        }

        map<int, int> sharedWordFrames;
        for(int i=0; i<num; i++) {
            if(wordCounts[i]) {
                sharedWordFrames.insert(map<int,int>::value_type(i, wordCounts[i]));
            }
        }


        // filter by maxWordCount and minWordCount
        int maxCommonWords = 0;
        for (auto &sharedWordNode : sharedWordFrames) {
            if (sharedWordNode.second > maxCommonWords) {
                maxCommonWords = sharedWordNode.second;
            }
        }
        int minCommonWords = maxCommonWords * 0.8f;
        imageScores->clear();
        for (auto &sharedWordFrame : sharedWordFrames) {
            if (sharedWordFrame.second >= minCommonWords) {
                int ind = sharedWordFrame.first;
                DBoW2::BowVector *ibow = cpuVoc[ind].second;
                float score = siftVocabulary->score(*bow, *ibow);
                imageScores->emplace_back(cpuVoc[ind].first, score);
            }
        }
        std::sort(imageScores->begin(), imageScores->end(), [=](MatchScore& ind1, MatchScore& ind2) {return ind1.score > ind2.score;});
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
        for(auto *it: ptrHolder) {
            delete it;
        }
    }

    void ComposeDBoWVocabulary::add(DBoWVocabulary* voc, int imageId) {
        vocs.emplace_back(voc);
        imageIds.emplace_back(imageId);
    }

    void ComposeDBoWVocabulary::add(shared_ptr<KeyFrame> keyframe) {
        DBoWVocabulary* voc = new DBoWVocabulary();
        for(shared_ptr<Frame> frame: keyframe->getFrames()) {
            voc->add(frame->getFrameIndex(), &frame->getKps().getMBowVec());
        }
        vocs.emplace_back(voc);
        imageIds.emplace_back(keyframe->getIndex());
    }
    void ComposeDBoWVocabulary::remove(int index, bool free) {
        if(free) {
            delete vocs[index];
        }
        vocs.erase(vocs.begin()+index);
        imageIds.erase(imageIds.begin()+index);
    }
    int ComposeDBoWVocabulary::size() {
        return vocs.size();
    }
    void ComposeDBoWVocabulary::clear() {
        for(DBoWVocabulary* voc: vocs) {
            delete voc;
        }
        vocs.clear();
        imageIds.clear();
    }
    ComposeDBoWVocabulary::~ComposeDBoWVocabulary() {
        clear();
    }

    DBoWHashing::DBoWHashing(const GlobalConfig &globalConfig, bool hashing) {
        config.vocTxtPath = globalConfig.vocTxtPath;
        config.numNeighs = globalConfig.numNeighs;
        config.numThreads = globalConfig.numThreads;
        config.numChecks = globalConfig.numChecks;
        config.maxNumFeatures = globalConfig.maxNumFeatures;
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
            featureCatas = new ComposeDBoWVocabulary[config.maxVINum];
            vocTh = 0;
        }

        featureCata = new ComposeDBoWVocabulary();
        prepared = false;

        queryPool = new ThreadPool(config.numThreads);
    }

    DBoWHashing::~DBoWHashing() {
        delete items;
        delete queryPool;
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

    void DBoWHashing::addVisualIndex(float3 wPos, DBoWVocabulary* voc, int imageId, bool notLost) {
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

        lVisualIndex->add(voc, imageId);
    }

    void DBoWHashing::addVisualIndex(float3 wPos, shared_ptr<KeyFrame> sf, bool notLost) {
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

        lVisualIndex->add(sf);
    }


    void DBoWHashing::computeMatchScores(const vector<ComposeDBoWVocabulary*> &vocs, DBoW2::BowVector* bow, vector<vector<MatchScore>>& imageScores) {
        int num = vocs.size();

        vector<vector<vector<uint>>> wordCounts(num);
        vector<vector<int>> imageIds;
        for(int i=0; i<num; i++){
            ComposeDBoWVocabulary* voc = vocs[i];
            int kNum = voc->size();
            wordCounts[i].resize(kNum);
            imageIds.emplace_back(voc->imageIds);
            for(int j=0; j<kNum; j++) {
                int bNum = voc->vocs[j]->size();
                CUDAArrayu cudaWordCounts(bNum);
                vector<uint> curWords = bow->words();
                CUDAArrayu cudaCurWords(curWords);
                CUDAPtrArray<CUDABoW> gpuVoc = voc->vocs[j]->gpuVoc.uploadToCUDA();
                wordsCount(gpuVoc, cudaCurWords, cudaWordCounts);
                cudaWordCounts.download(wordCounts[i][j]);
            }
        }

        int maxCommonWords = 0;
        for(int i=0; i<num; i++) {
            int kNum = wordCounts[i].size();
            for(int j=0; j<kNum; j++) {
                int wNum = wordCounts[i][j].size();
                for(int k=0; k<wNum; k++) {
                    maxCommonWords = max(maxCommonWords, wordCounts[i][j][k]);
                }
            }
        }
        int minCommonWords = maxCommonWords * 0.8f;
        for(int i=0; i<num; i++) {
            int kNum = wordCounts[i].size();
            imageScores[i].resize(kNum);
            for(int j=0; j<kNum; j++) {
                int wNum = wordCounts[i][j].size();
                float score = 0;
                for(int k=0; k<wNum; k++) {
                    score += (wordCounts[i][j][k] >= minCommonWords);
                }
                imageScores[i][j].imageId = imageIds[i][j];
                imageScores[i][j].score = score;
            }
            std::sort(imageScores[i].begin(), imageScores[i].end(), [=](MatchScore& ind1, MatchScore& ind2) {return ind1.score > ind2.score;});
        }
    }

    vector<MatchScore> DBoWHashing::queryImages(float3 wPos, DBoW2::BowVector* bow, bool notLost, bool hasLost) {
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
        vector<ComposeDBoWVocabulary*> nearVocs;

        for(uint viInd: viInds) {
            nearVocs.emplace_back(featureCatas+viInd);
            cMatchScores.emplace_back();

        }
        if(prepared) {// hava image in visual index
            nearVocs.emplace_back(featureCata);
            cMatchScores.emplace_back();
        }



        computeMatchScores(nearVocs, bow, cMatchScores);

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
            for(start=0; start<2&&start<cMatchScores[maxJ].size()&&start<config.numNeighs; start++) {
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
                    DBoWVocabulary* voc = featureCata->vocs[j];
                    int imageId = featureCata->imageIds[j];
                    featureCata->remove(j, false);
                    addVisualIndex(poses[i], voc, imageId);
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