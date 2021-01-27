//
// Created by liulei on 2020/9/23.
//

#ifndef GraphFusion_VISUAL_INDEX_HASHING_H
#define GraphFusion_VISUAL_INDEX_HASHING_H

#include "thread_pool.h"
#include "visual_index_hashing.cuh"
#include "../feature/feature_point.h"
#include "../datastructure/view_graph.h"

#include <DBoW2/TemplatedVocabulary.h>
#include <DBoW2/FSIFT.h>

namespace rtf {
    class SIFTVocabulary : public DBoW2::TemplatedVocabulary<DBoW2::FSIFT::TDescriptor, DBoW2::FSIFT>{
    public:
        void computeBow(SIFTFeaturePoints& sf);
    };

    class VIHConfig {
    public:
        int maxVINum = 500;
        float voxelSize = 2;
        int numThreads = 3;
        int numNeighs = 10;
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

        MatchScore();
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
        ViewGraph* viewGraph;

        HashItem * items;
        SIFTVocabulary * siftVocabulary;
        DBoWVocabulary * featureCatas;
        DBoWVocabulary * featureCata;
        bool prepared;
        uint vocTh;
        VIHConfig config;

        ThreadPool* queryPool;
    public:
        DBoWHashing(const GlobalConfig& config, SIFTVocabulary * siftVocabulary, ViewGraph* viewGraph, bool hashing=true);

        void initialize();

        ~DBoWHashing();

        uint hashFunction(int3 pos);

        int3 worldToPos(float3 wPos);

        void addVisualIndex(float3 wPos, SIFTFeaturePoints& sf, int imageId, bool notLost=true);

        void queryVisualIndex(vector<DBoWVocabulary*> vocs, SIFTFeaturePoints* sf, vector<MatchScore>* imageScores);

        vector<MatchScore> queryImages(float3 wPos, SIFTFeaturePoints& sf, bool notLost=true, bool hasLost=false);

        vector<int> detectLoopClosures(SIFTFeaturePoints& sf, float minScore=1);

        vector<int> lostImageIds();

        void updateVisualIndex(vector<int> imageIds, vector<float3> poses);

        void clear();
    };

    pair<int, int> selectBestOverlappingFrame(shared_ptr<KeyFrame> ref, shared_ptr<KeyFrame> cur, SIFTVocabulary* siftVocabulary);
}


#endif //GraphFusion_VISUAL_INDEX_HASHING_H
