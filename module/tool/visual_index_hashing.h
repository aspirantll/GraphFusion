//
// Created by liulei on 2020/9/23.
//

#ifndef RTF_VISUAL_INDEX_HASHING_H
#define RTF_VISUAL_INDEX_HASHING_H

#include "thread_pool.h"
#include "visual_index_hashing.cuh"
#include "../feature/feature_point.h"

#include <DBoW2/TemplatedVocabulary.h>
#include <DBoW2/FSIFT.h>

typedef DBoW2::TemplatedVocabulary<DBoW2::FSIFT::TDescriptor, DBoW2::FSIFT>
        SIFTVocabulary;

namespace rtf {
    class VIHConfig {
    public:
        int maxVINum = 500;
        float voxelSize = 2;
        int numThreads = 3;
        int numChecks = 256;
        int numNeighs = 10;
        int maxNumFeatures = 8192;
        bool hashing = true;
        string vocTxtPath;
    };

    typedef struct HashItem {
        int3 pos;
        int ptr;
    } HashItem;


    typedef struct MatchScore {
        int imageId;
        float score;

        MatchScore(int imageId, float score);
    } MatchScore;

    class DBoWVocabulary {
    public:
        vector<CUDAArrayu*> ptrHolder;
        vector<pair<int, SIFTFeaturePoints*>> cpuVoc;
        CUDAVector<CUDABoW> gpuVoc;
        vector<int> imageIds;

        void add(int imageId, SIFTFeaturePoints* sf);
        void remove(int index);
        int size();
        void clear();
        ~DBoWVocabulary();
    };

    class DBoWHashing {
    private:
        HashItem * items;
        SIFTVocabulary * siftVocabulary;
        DBoWVocabulary * featureCatas;
        DBoWVocabulary * featureCata;
        bool prepared;
        uint vocTh;
        VIHConfig config;

        ThreadPool* queryPool;
    public:
        DBoWHashing(const GlobalConfig& config, SIFTVocabulary * siftVocabulary,  bool hashing=true);

        void initialize();

        ~DBoWHashing();

        void computeBow(SIFTFeaturePoints& sf);

        uint hashFunction(int3 pos);

        int3 worldToPos(float3 wPos);

        void addVisualIndex(float3 wPos, SIFTFeaturePoints& sf, int imageId, bool notLost=true);

        void queryVisualIndex(DBoWVocabulary* voc, SIFTFeaturePoints* sf, vector<MatchScore>* imageScores);

        vector<MatchScore> queryImages(float3 wPos, SIFTFeaturePoints& sf, bool notLost=true, bool hasLost=false);

        vector<int> lostImageIds();

        void updateVisualIndex(vector<int> imageIds, vector<float3> poses);

        void clear();
    };
}


#endif //RTF_VISUAL_INDEX_HASHING_H
