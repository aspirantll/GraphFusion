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

        MatchScore();
        MatchScore(int imageId, float score);
    } MatchScore;

    class DBoWVocabulary {
    public:
        vector<CUDAArrayu*> ptrHolder;
        vector<pair<int, DBoW2::BowVector*>> cpuVoc;
        CUDAVector<CUDABoW> gpuVoc;
        vector<int> imageIds;

        void add(int imageId, DBoW2::BowVector* bow);
        void remove(int index);
        void query(SIFTVocabulary* siftVocabulary, DBoW2::BowVector* bow, vector<MatchScore>* imageScores);
        int size();
        void clear();
        ~DBoWVocabulary();
    };

    class ComposeDBoWVocabulary {
    public:
        vector<DBoWVocabulary *> vocs;
        vector<int> imageIds;

        void add(DBoWVocabulary* voc, int imageId);
        void add(shared_ptr<KeyFrame> keyframe);
        void remove(int index, bool free);
        int size();
        void clear();
        ~ComposeDBoWVocabulary();
    };


    class DBoWHashing {
    private:
        HashItem * items;
        ComposeDBoWVocabulary * featureCatas = nullptr;
        ComposeDBoWVocabulary * featureCata = nullptr;
        bool prepared;
        uint vocTh;
        VIHConfig config;

        ThreadPool* queryPool;

        void computeMatchScores(const vector<ComposeDBoWVocabulary*> &voc, DBoW2::BowVector* bow, vector<vector<MatchScore>>& imageScores);
    public:
        DBoWHashing(const GlobalConfig& config, bool hashing=true);

        void initialize();

        ~DBoWHashing();

        uint hashFunction(int3 pos);

        int3 worldToPos(float3 wPos);

        void addVisualIndex(float3 wPos, DBoWVocabulary* voc, int imageId, bool notLost=true);

        void addVisualIndex(float3 wPos, shared_ptr<KeyFrame> sf, bool notLost=true);

        vector<MatchScore> queryImages(float3 wPos, DBoW2::BowVector* bow, bool notLost=true, bool hasLost=false);

        vector<int> lostImageIds();

        void updateVisualIndex(vector<int> imageIds, vector<float3> poses);

        void clear();
    };
}


#endif //GraphFusion_VISUAL_INDEX_HASHING_H
