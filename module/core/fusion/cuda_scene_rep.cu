/**
 * this file is based on https://github.com/niessner/BundleFusion.git
 */

#include "cuda_scene_rep.h"
#include <cuda_texture_types.h>
#include <pcl/Vertices.h>
#include <pcl/conversions.h>
#include "tables.h"

///////////////
// Host part //
///////////////

__device__ __host__
HashDataStruct::HashDataStruct() {
	d_heap = NULL;
	d_heapCounter = NULL;
	d_hash = NULL;
	d_hashDecision = NULL;
	d_hashDecisionPrefix = NULL;
	d_hashCompactified = NULL;
	d_hashCompactifiedCounter = NULL;
	d_SDFBlocks = NULL;
	d_hashBucketMutex = NULL;
	m_bIsOnGPU = false;
}

__host__
void HashDataStruct::allocate(const HashParams &params, bool dataOnGPU) {
	m_bIsOnGPU = dataOnGPU;
	if (m_bIsOnGPU) {
		cutilSafeCall(cudaMalloc(&d_heap, sizeof(unsigned int) * params.m_numSDFBlocks));
		cutilSafeCall(cudaMalloc(&d_heapCounter, sizeof(unsigned int)));
		cutilSafeCall(cudaMalloc(&d_hash, sizeof(HashEntry) * params.m_hashNumBuckets * params.m_hashBucketSize));
		cutilSafeCall(cudaMalloc(&d_hashDecision, sizeof(int) * params.m_hashNumBuckets * params.m_hashBucketSize));
		cutilSafeCall(
				cudaMalloc(&d_hashDecisionPrefix, sizeof(int) * params.m_hashNumBuckets * params.m_hashBucketSize));
		cutilSafeCall(
				cudaMalloc(&d_hashCompactified, sizeof(HashEntry) * params.m_hashNumBuckets * params.m_hashBucketSize));
		cutilSafeCall(cudaMalloc(&d_hashCompactifiedCounter, sizeof(int)));
		cutilSafeCall(cudaMalloc(&d_SDFBlocks,
								 sizeof(Voxel) * params.m_numSDFBlocks * params.m_SDFBlockSize * params.m_SDFBlockSize *
								 params.m_SDFBlockSize));
		cutilSafeCall(cudaMalloc(&d_hashBucketMutex, sizeof(int) * params.m_hashNumBuckets));
	} else {
		d_heap = new unsigned int[params.m_numSDFBlocks];
		d_heapCounter = new unsigned int[1];
		d_hash = new HashEntry[params.m_hashNumBuckets * params.m_hashBucketSize];
		d_hashDecision = new int[params.m_hashNumBuckets * params.m_hashBucketSize];
		d_hashDecisionPrefix = new int[params.m_hashNumBuckets * params.m_hashBucketSize];
		d_hashCompactifiedCounter = new int[1];
		d_hashCompactified = new HashEntry[params.m_hashNumBuckets * params.m_hashBucketSize];
		d_SDFBlocks = new Voxel[params.m_numSDFBlocks * params.m_SDFBlockSize * params.m_SDFBlockSize *
								params.m_SDFBlockSize];
		d_hashBucketMutex = new int[params.m_hashNumBuckets];
	}

	updateParams(params);
}

extern "C" void updateConstantHashParams(const HashParams& params) {

	size_t size;
	CUDA_CHECKED_CALL(cudaGetSymbolSize(&size, c_hashParams));
	CUDA_CHECKED_CALL(cudaMemcpyToSymbol(c_hashParams, &params, size, 0, cudaMemcpyHostToDevice));
	CUDA_CHECKED_CALL(cudaDeviceSynchronize());
}

__host__
void HashDataStruct::updateParams(const HashParams &params) {
	if (m_bIsOnGPU) {
		updateConstantHashParams(params);
	}
}

__host__
void HashDataStruct::free() {
	if (m_bIsOnGPU) {
		cutilSafeCall(cudaFree(d_heap));
		cutilSafeCall(cudaFree(d_heapCounter));
		cutilSafeCall(cudaFree(d_hash));
		cutilSafeCall(cudaFree(d_hashDecision));
		cutilSafeCall(cudaFree(d_hashDecisionPrefix));
		cutilSafeCall(cudaFree(d_hashCompactified));
		cutilSafeCall(cudaFree(d_hashCompactifiedCounter));
		cutilSafeCall(cudaFree(d_SDFBlocks));
		cutilSafeCall(cudaFree(d_hashBucketMutex));
	} else {
		if (d_heap) delete[] d_heap;
		if (d_heapCounter) delete[] d_heapCounter;
		if (d_hash) delete[] d_hash;
		if (d_hashDecision) delete[] d_hashDecision;
		if (d_hashDecisionPrefix) delete[] d_hashDecisionPrefix;
		if (d_hashCompactified) delete[] d_hashCompactified;
		if (d_hashCompactifiedCounter) delete[] d_hashCompactifiedCounter;
		if (d_SDFBlocks) delete[] d_SDFBlocks;
		if (d_hashBucketMutex) delete[] d_hashBucketMutex;
	}

	d_hash = NULL;
	d_heap = NULL;
	d_heapCounter = NULL;
	d_hashDecision = NULL;
	d_hashDecisionPrefix = NULL;
	d_hashCompactified = NULL;
	d_hashCompactifiedCounter = NULL;
	d_SDFBlocks = NULL;
	d_hashBucketMutex = NULL;
}

__host__
HashDataStruct HashDataStruct::copyToCPU() const {
	HashParams params;

	HashDataStruct hashData;
	hashData.allocate(params, false);    //allocate the data on the CPU
	cutilSafeCall(
			cudaMemcpy(hashData.d_heap, d_heap, sizeof(unsigned int) * params.m_numSDFBlocks, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(hashData.d_heapCounter, d_heapCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	cutilSafeCall(
			cudaMemcpy(hashData.d_hash, d_hash, sizeof(HashEntry) * params.m_hashNumBuckets * params.m_hashBucketSize,
					   cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(hashData.d_hashDecision, d_hashDecision,
							 sizeof(int) * params.m_hashNumBuckets * params.m_hashBucketSize, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(hashData.d_hashDecisionPrefix, d_hashDecisionPrefix,
							 sizeof(int) * params.m_hashNumBuckets * params.m_hashBucketSize, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(hashData.d_hashCompactified, d_hashCompactified,
							 sizeof(HashEntry) * params.m_hashNumBuckets * params.m_hashBucketSize,
							 cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(hashData.d_SDFBlocks, d_SDFBlocks,
							 sizeof(Voxel) * params.m_numSDFBlocks * params.m_SDFBlockSize * params.m_SDFBlockSize *
							 params.m_SDFBlockSize, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(hashData.d_hashBucketMutex, d_hashBucketMutex, sizeof(int) * params.m_hashNumBuckets,
							 cudaMemcpyDeviceToHost));

	return hashData;    //TODO MATTHIAS look at this (i.e,. when does memory get destroyed ; if it's in the destructer it would kill everything here
}


/////////////////
// Device part //
/////////////////
__device__
const HashParams &HashDataStruct::params() const {
	return c_hashParams;
}

//! see teschner et al. (but with correct prime values)
__device__
uint HashDataStruct::computeHashPos(const int3 &virtualVoxelPos) const {
	const int p0 = 73856093;
	const int p1 = 19349669;
	const int p2 = 83492791;

	int res = ((virtualVoxelPos.x * p0) ^ (virtualVoxelPos.y * p1) ^ (virtualVoxelPos.z * p2)) %
			  c_hashParams.m_hashNumBuckets;
	if (res < 0) res += c_hashParams.m_hashNumBuckets;
	return (uint) res;
}

//merges two voxels (v0 is the input voxel, v1 the currently stored voxel)
__device__
void HashDataStruct::combineVoxel(const Voxel &v0, const Voxel &v1, Voxel &out) const {

	//v.color = (10*v0.weight * v0.color + v1.weight * v1.color)/(10*v0.weight + v1.weight);	//give the currently observed color more weight
	//v.color = (v0.weight * v0.color + v1.weight * v1.color)/(v0.weight + v1.weight);
	//out.color = 0.5f * (v0.color + v1.color);	//exponential running average


	float3 c0 = make_float3(v0.color.x, v0.color.y, v0.color.z);
	float3 c1 = make_float3(v1.color.x, v1.color.y, v1.color.z);

	//float3 res = (c0+c1)/2;
	//float3 res = (c0 * (float)v0.weight + c1 * (float)v1.weight) / ((float)v0.weight + (float)v1.weight);
	//float3 res = c1;
	if (v0.weight == 0) out.color = v1.color;
	else {
		float3 res = 0.5f * c0 + 0.5f * c1;
		out.color.x = (uchar)(res.x + 0.5f);
		out.color.y = (uchar)(res.y + 0.5f);
		out.color.z = (uchar)(res.z + 0.5f);
	}


	out.sdf = (v0.sdf * (float) v0.weight + v1.sdf * (float) v1.weight) / ((float) v0.weight + (float) v1.weight);
	//out.weight = min(c_hashParams.m_integrationWeightMax, (unsigned int)v0.weight + (unsigned int)v1.weight);
	out.weight = min((float) c_hashParams.m_integrationWeightMax, v0.weight + v1.weight);
}

__device__
void HashDataStruct::combineVoxelDepthOnly(const Voxel &v0, const Voxel &v1, Voxel &out) const {
	out.sdf = (v0.sdf * (float) v0.weight + v1.sdf * (float) v1.weight) / ((float) v0.weight + (float) v1.weight);
	out.weight = min((float) c_hashParams.m_integrationWeightMax, v0.weight + v1.weight);
}


//! returns the truncation of the SDF for a given distance value
__device__
float HashDataStruct::getTruncation(float z) const {
	return c_hashParams.m_truncation + c_hashParams.m_truncScale * z;
}


__device__
float3 HashDataStruct::worldToVirtualVoxelPosFloat(const float3 &pos) const {
	return pos / c_hashParams.m_virtualVoxelSize;
}

__device__
int3 HashDataStruct::worldToVirtualVoxelPos(const float3 &pos) const {
	//const float3 p = pos*g_VirtualVoxelResolutionScalar;
	const float3 p = pos / c_hashParams.m_virtualVoxelSize;
	return make_int3(p + make_float3(sign(p)) * 0.5f);
}

__device__
int3 HashDataStruct::virtualVoxelPosToSDFBlock(int3 virtualVoxelPos) const {
	if (virtualVoxelPos.x < 0) virtualVoxelPos.x -= SDF_BLOCK_SIZE - 1;
	if (virtualVoxelPos.y < 0) virtualVoxelPos.y -= SDF_BLOCK_SIZE - 1;
	if (virtualVoxelPos.z < 0) virtualVoxelPos.z -= SDF_BLOCK_SIZE - 1;

	return make_int3(
			virtualVoxelPos.x / SDF_BLOCK_SIZE,
			virtualVoxelPos.y / SDF_BLOCK_SIZE,
			virtualVoxelPos.z / SDF_BLOCK_SIZE);
}

// Computes virtual voxel position of corner sample position
__device__
int3 HashDataStruct::SDFBlockToVirtualVoxelPos(const int3 &sdfBlock) const {
	return sdfBlock * SDF_BLOCK_SIZE;
}

__device__
float3 HashDataStruct::virtualVoxelPosToWorld(const int3 &pos) const {
	return make_float3(pos) * c_hashParams.m_virtualVoxelSize;
}

__device__
float3 HashDataStruct::SDFBlockToWorld(const int3 &sdfBlock) const {
	return virtualVoxelPosToWorld(SDFBlockToVirtualVoxelPos(sdfBlock));
}

__device__
int3 HashDataStruct::worldToSDFBlock(const float3 &worldPos) const {
	return virtualVoxelPosToSDFBlock(worldToVirtualVoxelPos(worldPos));
}

__device__
bool HashDataStruct::isSDFBlockInCameraFrustumApprox(const int3 &sdfBlock, CUDAFrame &frame) {
	float3 posWorld = virtualVoxelPosToWorld(SDFBlockToVirtualVoxelPos(sdfBlock)) +
					  c_hashParams.m_virtualVoxelSize * 0.5f * (SDF_BLOCK_SIZE - 1.0f);
	return frame.isInCameraFrustumApprox(posWorld);
}

//! computes the (local) virtual voxel pos of an index; idx in [0;511]
__device__
uint3 HashDataStruct::delinearizeVoxelIndex(uint idx) const {
	uint x = idx % SDF_BLOCK_SIZE;
	uint y = (idx % (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE)) / SDF_BLOCK_SIZE;
	uint z = idx / (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE);
	return make_uint3(x, y, z);
}

//! computes the linearized index of a local virtual voxel pos; pos in [0;7]^3
__device__
uint HashDataStruct::linearizeVoxelPos(const int3 &virtualVoxelPos) const {
	return
			virtualVoxelPos.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE +
			virtualVoxelPos.y * SDF_BLOCK_SIZE +
			virtualVoxelPos.x;
}

__device__
int HashDataStruct::virtualVoxelPosToLocalSDFBlockIndex(const int3 &virtualVoxelPos) const {
	int3 localVoxelPos = make_int3(
			virtualVoxelPos.x % SDF_BLOCK_SIZE,
			virtualVoxelPos.y % SDF_BLOCK_SIZE,
			virtualVoxelPos.z % SDF_BLOCK_SIZE);

	if (localVoxelPos.x < 0) localVoxelPos.x += SDF_BLOCK_SIZE;
	if (localVoxelPos.y < 0) localVoxelPos.y += SDF_BLOCK_SIZE;
	if (localVoxelPos.z < 0) localVoxelPos.z += SDF_BLOCK_SIZE;

	return linearizeVoxelPos(localVoxelPos);
}

__device__
int HashDataStruct::worldToLocalSDFBlockIndex(const float3 &world) const {
	int3 virtualVoxelPos = worldToVirtualVoxelPos(world);
	return virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos);
}


//! returns the hash entry for a given worldPos; if there was no hash entry the returned entry will have a ptr with FREE_ENTRY set
__device__
HashEntry HashDataStruct::getHashEntry(const float3 &worldPos) const {
	//int3 blockID = worldToSDFVirtualVoxelPos(worldPos)/SDF_BLOCK_SIZE;	//position of sdf block
	int3 blockID = worldToSDFBlock(worldPos);
	return getHashEntryForSDFBlockPos(blockID);
}


__device__
void HashDataStruct::deleteHashEntry(uint id) {
	deleteHashEntry(d_hash[id]);
}

__device__
void HashDataStruct::deleteHashEntry(HashEntry &hashEntry) {
	hashEntry.pos = make_int3(0);
	hashEntry.offset = 0;
	hashEntry.ptr = FREE_ENTRY;
}

__device__
bool HashDataStruct::voxelExists(const float3 &worldPos) const {
	HashEntry hashEntry = getHashEntry(worldPos);
	return (hashEntry.ptr != FREE_ENTRY);
}

__device__
void HashDataStruct::deleteVoxel(Voxel &v) const {
	v.color = make_uchar4(0, 0, 0, 0);
	v.weight = 0.0f;
	v.sdf = 0.0f;
}

__device__
void HashDataStruct::deleteVoxel(uint id) {
	deleteVoxel(d_SDFBlocks[id]);
}


__device__
Voxel HashDataStruct::getVoxel(const float3 &worldPos) const {
	HashEntry hashEntry = getHashEntry(worldPos);
	Voxel v;
	if (hashEntry.ptr == FREE_ENTRY) {
		deleteVoxel(v);
	} else {
		int3 virtualVoxelPos = worldToVirtualVoxelPos(worldPos);
		v = d_SDFBlocks[hashEntry.ptr + virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos)];
	}
	return v;
}

__device__
Voxel HashDataStruct::getVoxel(const int3 &virtualVoxelPos) const {
	HashEntry hashEntry = getHashEntryForSDFBlockPos(virtualVoxelPosToSDFBlock(virtualVoxelPos));
	Voxel v;
	if (hashEntry.ptr == FREE_ENTRY) {
		deleteVoxel(v);
	} else {
		v = d_SDFBlocks[hashEntry.ptr + virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos)];
	}
	return v;
}

__device__
void HashDataStruct::setVoxel(const int3 &virtualVoxelPos, Voxel &voxelInput) const {
	HashEntry hashEntry = getHashEntryForSDFBlockPos(virtualVoxelPosToSDFBlock(virtualVoxelPos));
	if (hashEntry.ptr != FREE_ENTRY) {
		d_SDFBlocks[hashEntry.ptr + virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos)] = voxelInput;
	}
}

//! returns the hash entry for a given sdf block id; if there was no hash entry the returned entry will have a ptr with FREE_ENTRY set
__device__
HashEntry HashDataStruct::getHashEntryForSDFBlockPos(const int3 &sdfBlock) const {
	uint h = computeHashPos(sdfBlock);            //hash bucket
	uint hp = h * HASH_BUCKET_SIZE;    //hash position

	HashEntry entry;
	entry.pos = sdfBlock;
	entry.offset = 0;
	entry.ptr = FREE_ENTRY;

	for (uint j = 0; j < HASH_BUCKET_SIZE; j++) {
		uint i = j + hp;
		HashEntry curr = d_hash[i];
		if (curr.pos.x == entry.pos.x && curr.pos.y == entry.pos.y && curr.pos.z == entry.pos.z &&
			curr.ptr != FREE_ENTRY) {
			return curr;
		}
	}

#ifdef HANDLE_COLLISIONS
	const uint idxLastEntryInBucket = (h + 1) * HASH_BUCKET_SIZE - 1;
	int i = idxLastEntryInBucket;    //start with the last entry of the current bucket
	HashEntry curr;
	//traverse list until end: memorize idx at list end and memorize offset from last element of bucket to list end

	unsigned int maxIter = 0;
	uint g_MaxLoopIterCount = c_hashParams.m_hashMaxCollisionLinkedListSize;
#pragma unroll 1
	while (maxIter < g_MaxLoopIterCount) {
		curr = d_hash[i];

		if (curr.pos.x == entry.pos.x && curr.pos.y == entry.pos.y && curr.pos.z == entry.pos.z &&
			curr.ptr != FREE_ENTRY) {
			return curr;
		}

		if (curr.offset == 0) {    //we have found the end of the list
			break;
		}
		i = idxLastEntryInBucket + curr.offset;                        //go to next element in the list
		i %= (HASH_BUCKET_SIZE * c_hashParams.m_hashNumBuckets);    //check for overflow

		maxIter++;
	}
#endif
	return entry;
}

//for histogram (no collision traversal)
__device__
unsigned int HashDataStruct::getNumHashEntriesPerBucket(unsigned int bucketID) {
	unsigned int h = 0;
	for (uint i = 0; i < HASH_BUCKET_SIZE; i++) {
		if (d_hash[bucketID * HASH_BUCKET_SIZE + i].ptr != FREE_ENTRY) {
			h++;
		}
	}
	return h;
}

//for histogram (collisions traversal only)
__device__
unsigned int HashDataStruct::getNumHashLinkedList(unsigned int bucketID) {
	unsigned int listLen = 0;

#ifdef HANDLE_COLLISIONS
	const uint idxLastEntryInBucket = (bucketID + 1) * HASH_BUCKET_SIZE - 1;
	unsigned int i = idxLastEntryInBucket;    //start with the last entry of the current bucket
	//int offset = 0;
	HashEntry curr;
	curr.offset = 0;
	//traverse list until end: memorize idx at list end and memorize offset from last element of bucket to list end

	unsigned int maxIter = 0;
	uint g_MaxLoopIterCount = c_hashParams.m_hashMaxCollisionLinkedListSize;
#pragma unroll 1
	while (maxIter < g_MaxLoopIterCount) {
		//offset = curr.offset;
		//curr = getHashEntry(g_Hash, i);
		curr = d_hash[i];

		if (curr.offset == 0) {    //we have found the end of the list
			break;
		}
		i = idxLastEntryInBucket + curr.offset;        //go to next element in the list
		i %= (HASH_BUCKET_SIZE * c_hashParams.m_hashNumBuckets);    //check for overflow
		listLen++;

		maxIter++;
	}
#endif

	return listLen;
}


__device__
uint HashDataStruct::consumeHeap() {
	uint addr = atomicSub(&d_heapCounter[0], 1);
	//TODO MATTHIAS check some error handling?
	return d_heap[addr];
}

__device__
void HashDataStruct::appendHeap(uint ptr) {
	uint addr = atomicAdd(&d_heapCounter[0], 1);
	//TODO MATTHIAS check some error handling?
	d_heap[addr + 1] = ptr;
}

//pos in SDF block coordinates
__device__
void HashDataStruct::allocBlock(const int3 &pos) {
	uint h = computeHashPos(pos);                //hash bucket
	uint hp = h * HASH_BUCKET_SIZE;    //hash position

	int firstEmpty = -1;
	for (uint j = 0; j < HASH_BUCKET_SIZE; j++) {
		uint i = j + hp;
		const HashEntry &curr = d_hash[i];

		//in that case the SDF-block is already allocated and corresponds to the current position -> exit thread
		if (curr.pos.x == pos.x && curr.pos.y == pos.y && curr.pos.z == pos.z && curr.ptr != FREE_ENTRY) {
			return;
		}

		//store the first FREE_ENTRY hash entry
		if (firstEmpty == -1 && curr.ptr == FREE_ENTRY) {
			firstEmpty = i;
		}
	}


#ifdef HANDLE_COLLISIONS
	//updated variables as after the loop
	const uint idxLastEntryInBucket = (h + 1) * HASH_BUCKET_SIZE - 1;    //get last index of bucket
	uint i = idxLastEntryInBucket;                                            //start with the last entry of the current bucket
	//int offset = 0;
	HashEntry curr;
	curr.offset = 0;
	//traverse list until end: memorize idx at list end and memorize offset from last element of bucket to list end
	//int k = 0;

    unsigned int maxIter = 0;
	uint g_MaxLoopIterCount = c_hashParams.m_hashMaxCollisionLinkedListSize;
#pragma  unroll 1
	while (maxIter < g_MaxLoopIterCount) {
        //offset = curr.offset;
        curr = d_hash[i];    //TODO MATTHIAS do by reference

		if (curr.pos.x == pos.x && curr.pos.y == pos.y && curr.pos.z == pos.z && curr.ptr != FREE_ENTRY) {
			return;
		}
		if (curr.offset == 0) {    //we have found the end of the list
			break;
		}

		i = idxLastEntryInBucket + curr.offset;        //go to next element in the list
		i %= (HASH_BUCKET_SIZE * c_hashParams.m_hashNumBuckets);    //check for overflow

		maxIter++;
	}
#endif

	if (firstEmpty != -1) {    //if there is an empty entry and we haven't allocated the current entry before
		//int prevValue = 0;
		//InterlockedExchange(d_hashBucketMutex[h], LOCK_ENTRY, prevValue);	//lock the hash bucket
		int prevValue = atomicExch(&d_hashBucketMutex[h], LOCK_ENTRY);
		if (prevValue != LOCK_ENTRY) {    //only proceed if the bucket has been locked
			HashEntry &entry = d_hash[firstEmpty];
			entry.pos = pos;
			entry.offset = NO_OFFSET;
			long index = consumeHeap();
			entry.ptr = index * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;    //memory alloc
		}
		return;
	}

#ifdef HANDLE_COLLISIONS
	//if (i != idxLastEntryInBucket) return;
	int offset = 0;
	//linear search for free entry

	maxIter = 0;
#pragma  unroll 1
	while (maxIter < g_MaxLoopIterCount) {
		offset++;
		i = (idxLastEntryInBucket + offset) %
			(HASH_BUCKET_SIZE * c_hashParams.m_hashNumBuckets);    //go to next hash element
		if ((offset % HASH_BUCKET_SIZE) == 0)
			continue;            //cannot insert into a last bucket element (would conflict with other linked lists)
		curr = d_hash[i];
		//if (curr.pos.x == pos.x && curr.pos.y == pos.y && curr.pos.z == pos.z && curr.ptr != FREE_ENTRY) {
		//	return;
		//}
		if (curr.ptr == FREE_ENTRY) {    //this is the first free entry
			//int prevValue = 0;
			//InterlockedExchange(g_HashBucketMutex[h], LOCK_ENTRY, prevValue);	//lock the original hash bucket
			int prevValue = atomicExch(&d_hashBucketMutex[h], LOCK_ENTRY);
			if (prevValue != LOCK_ENTRY) {
				HashEntry lastEntryInBucket = d_hash[idxLastEntryInBucket];
				h = i / HASH_BUCKET_SIZE;
				//InterlockedExchange(g_HashBucketMutex[h], LOCK_ENTRY, prevValue);	//lock the hash bucket where we have found a free entry
				prevValue = atomicExch(&d_hashBucketMutex[h], LOCK_ENTRY);
				if (prevValue != LOCK_ENTRY) {    //only proceed if the bucket has been locked
					HashEntry &entry = d_hash[i];
					entry.pos = pos;
					entry.offset = lastEntryInBucket.offset;
					long index = consumeHeap();
					entry.ptr = index * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;    //memory alloc

					lastEntryInBucket.offset = offset;
					d_hash[idxLastEntryInBucket] = lastEntryInBucket;
					//setHashEntry(g_Hash, idxLastEntryInBucket, lastEntryInBucket);
				}
			}
			return;    //bucket was already locked
		}

		maxIter++;
	}
#endif
}


//!inserts a hash entry without allocating any memory: used by streaming: TODO MATTHIAS check the atomics in this function
__device__
bool HashDataStruct::insertHashEntry(HashEntry entry) {
	uint h = computeHashPos(entry.pos);
	uint hp = h * HASH_BUCKET_SIZE;

	for (uint j = 0; j < HASH_BUCKET_SIZE; j++) {
		uint i = j + hp;
		//const HashEntry& curr = d_hash[i];
		int prevWeight = 0;
		//InterlockedCompareExchange(hash[3*i+2], FREE_ENTRY, LOCK_ENTRY, prevWeight);
		prevWeight = atomicCAS(&d_hash[i].ptr, FREE_ENTRY, LOCK_ENTRY);
		if (prevWeight == FREE_ENTRY) {
			d_hash[i] = entry;
			//setHashEntry(hash, i, entry);
			return true;
		}
	}

#ifdef HANDLE_COLLISIONS
	//updated variables as after the loop
	const uint idxLastEntryInBucket = (h + 1) * HASH_BUCKET_SIZE - 1;    //get last index of bucket

	uint i = idxLastEntryInBucket;                                            //start with the last entry of the current bucket
	HashEntry curr;

	unsigned int maxIter = 0;
	//[allow_uav_condition]
	uint g_MaxLoopIterCount = c_hashParams.m_hashMaxCollisionLinkedListSize;
#pragma  unroll 1
	while (maxIter <
		   g_MaxLoopIterCount) {                                    //traverse list until end // why find the end? we you are inserting at the start !!!
		//curr = getHashEntry(hash, i);
		curr = d_hash[i];    //TODO MATTHIAS do by reference
		if (curr.offset == 0) break;                                    //we have found the end of the list
		i = idxLastEntryInBucket + curr.offset;                            //go to next element in the list
		i %= (HASH_BUCKET_SIZE * c_hashParams.m_hashNumBuckets);    //check for overflow

		maxIter++;
	}

	maxIter = 0;
	int offset = 0;
#pragma  unroll 1
	while (maxIter <
		   g_MaxLoopIterCount) {                                                    //linear search for free entry
		offset++;
		uint i = (idxLastEntryInBucket + offset) %
				 (HASH_BUCKET_SIZE * c_hashParams.m_hashNumBuckets);    //go to next hash element
		if ((offset % HASH_BUCKET_SIZE) == 0)
			continue;                                        //cannot insert into a last bucket element (would conflict with other linked lists)

		int prevWeight = 0;
		//InterlockedCompareExchange(hash[3*i+2], FREE_ENTRY, LOCK_ENTRY, prevWeight);		//check for a free entry
		uint *d_hashUI = (uint *) d_hash;
		prevWeight = prevWeight = atomicCAS(&d_hashUI[3 * idxLastEntryInBucket + 1], (uint) FREE_ENTRY,
											(uint) LOCK_ENTRY);
		if (prevWeight ==
			FREE_ENTRY) {                                                        //if free entry found set prev->next = curr & curr->next = prev->next
			//[allow_uav_condition]
			//while(hash[3*idxLastEntryInBucket+2] == LOCK_ENTRY); // expects setHashEntry to set the ptr last, required because pos.z is packed into the same value -> prev->next = curr -> might corrput pos.z

			HashEntry lastEntryInBucket = d_hash[idxLastEntryInBucket];            //get prev (= lastEntry in Bucket)

			int newOffsetPrev =
					(offset << 16) | (lastEntryInBucket.pos.z & 0x0000ffff);    //prev->next = curr (maintain old z-pos)
			int oldOffsetPrev = 0;
			//InterlockedExchange(hash[3*idxLastEntryInBucket+1], newOffsetPrev, oldOffsetPrev);	//set prev offset atomically
			uint *d_hashUI = (uint *) d_hash;
			oldOffsetPrev = prevWeight = atomicExch(&d_hashUI[3 * idxLastEntryInBucket + 1], newOffsetPrev);
			entry.offset = oldOffsetPrev
					>> 16;                                                    //remove prev z-pos from old offset

			//setHashEntry(hash, i, entry);														//sets the current hashEntry with: curr->next = prev->next
			d_hash[i] = entry;
			return true;
		}

		maxIter++;
	}
#endif

	return false;
}


//! deletes a hash entry position for a given sdfBlock index (returns true uppon successful deletion; otherwise returns false)
__device__
bool HashDataStruct::deleteHashEntryElement(const int3 &sdfBlock) {
	uint h = computeHashPos(sdfBlock);    //hash bucket
	uint hp = h * HASH_BUCKET_SIZE;        //hash position

	for (uint j = 0; j < HASH_BUCKET_SIZE; j++) {
		uint i = j + hp;
		const HashEntry &curr = d_hash[i];
		if (curr.pos.x == sdfBlock.x && curr.pos.y == sdfBlock.y && curr.pos.z == sdfBlock.z &&
			curr.ptr != FREE_ENTRY) {
#ifndef HANDLE_COLLISIONS
			const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
            appendHeap(curr.ptr / linBlockSize);
            //heapAppend.Append(curr.ptr / linBlockSize);
            deleteHashEntry(i);
            return true;
#endif
#ifdef HANDLE_COLLISIONS
			if (curr.offset != 0) {    //if there was a pointer set it to the next list element
				//int prevValue = 0;
				//InterlockedExchange(bucketMutex[h], LOCK_ENTRY, prevValue);	//lock the hash bucket
				int prevValue = atomicExch(&d_hashBucketMutex[h], LOCK_ENTRY);
				if (prevValue == LOCK_ENTRY) return false;
				if (prevValue != LOCK_ENTRY) {
					const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
					appendHeap(curr.ptr / linBlockSize);
					//heapAppend.Append(curr.ptr / linBlockSize);
					int nextIdx = (i + curr.offset) % (HASH_BUCKET_SIZE * c_hashParams.m_hashNumBuckets);
					//setHashEntry(hash, i, getHashEntry(hash, nextIdx));
					d_hash[i] = d_hash[nextIdx];
					deleteHashEntry(nextIdx);
					return true;
				}
			} else {
				const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
				appendHeap(curr.ptr / linBlockSize);
				//heapAppend.Append(curr.ptr / linBlockSize);
				deleteHashEntry(i);
				return true;
			}
#endif    //HANDLE_COLLSISION
		}
	}
#ifdef HANDLE_COLLISIONS
	const uint idxLastEntryInBucket = (h + 1) * HASH_BUCKET_SIZE - 1;
	int i = idxLastEntryInBucket;
	HashEntry curr;
	curr = d_hash[i];
	int prevIdx = i;
	i = idxLastEntryInBucket + curr.offset;                            //go to next element in the list
	i %= (HASH_BUCKET_SIZE * c_hashParams.m_hashNumBuckets);    //check for overflow

	unsigned int maxIter = 0;
	uint g_MaxLoopIterCount = c_hashParams.m_hashMaxCollisionLinkedListSize;

#pragma  unroll 1
	while (maxIter < g_MaxLoopIterCount) {
		curr = d_hash[i];
		//found that dude that we need/want to delete
		if (curr.pos.x == sdfBlock.x && curr.pos.y == sdfBlock.y && curr.pos.z == sdfBlock.z &&
			curr.ptr != FREE_ENTRY) {
			//int prevValue = 0;
			//InterlockedExchange(bucketMutex[h], LOCK_ENTRY, prevValue);	//lock the hash bucket
			int prevValue = atomicExch(&d_hashBucketMutex[h], LOCK_ENTRY);
			if (prevValue == LOCK_ENTRY) return false;
			if (prevValue != LOCK_ENTRY) {
				const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
				appendHeap(curr.ptr / linBlockSize);
				//heapAppend.Append(curr.ptr / linBlockSize);
				deleteHashEntry(i);
				HashEntry prev = d_hash[prevIdx];
				prev.offset = curr.offset;
				//setHashEntry(hash, prevIdx, prev);
				d_hash[prevIdx] = prev;
				return true;
			}
		}

		if (curr.offset == 0) {    //we have found the end of the list
			return false;    //should actually never happen because we need to find that guy before
		}
		prevIdx = i;
		i = idxLastEntryInBucket + curr.offset;        //go to next element in the list
		i %= (HASH_BUCKET_SIZE * c_hashParams.m_hashNumBuckets);    //check for overflow

		maxIter++;
	}
#endif    // HANDLE_COLLSISION
	return false;
}


#define T_PER_BLOCK 16

texture<float, cudaTextureType2D, cudaReadModeElementType> depthTextureRef;
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> colorTextureRef;

void bindInputDepthColorTextures(const CUDAFrame& frame)
{
	int width = frame.imageWidth, height = frame.imageHeight;
	cutilSafeCall(cudaBindTexture2D(0, &depthTextureRef, frame.depthData, &depthTextureRef.channelDesc, width, height, sizeof(float)*width));
	cutilSafeCall(cudaBindTexture2D(0, &colorTextureRef, frame.colorData, &colorTextureRef.channelDesc, width, height, sizeof(uchar4)*width));

	depthTextureRef.filterMode = cudaFilterModePoint;
	colorTextureRef.filterMode = cudaFilterModePoint;
}

__global__ void resetHeapKernel(HashDataStruct hashData) 
{
	const HashParams& hashParams = c_hashParams;
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx == 0) {
		hashData.d_heapCounter[0] = hashParams.m_numSDFBlocks - 1;	//points to the last element of the array
	}
	
	if (idx < hashParams.m_numSDFBlocks) {

		hashData.d_heap[idx] = hashParams.m_numSDFBlocks - idx - 1;
		uint blockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
		uint base_idx = idx * blockSize;
		for (uint i = 0; i < blockSize; i++) {
			hashData.deleteVoxel(base_idx+i);
		}
	}
}

__global__ void resetHashKernel(HashDataStruct hashData) 
{
	const HashParams& hashParams = c_hashParams;
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < hashParams.m_hashNumBuckets * HASH_BUCKET_SIZE) {
		hashData.deleteHashEntry(hashData.d_hash[idx]);
		hashData.deleteHashEntry(hashData.d_hashCompactified[idx]);
	}
}


__global__ void resetHashBucketMutexKernel(HashDataStruct hashData) 
{
	const HashParams& hashParams = c_hashParams;
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < hashParams.m_hashNumBuckets) {
		hashData.d_hashBucketMutex[idx] = FREE_ENTRY;
	}
}

void resetCUDA(HashDataStruct& hashData, const HashParams& hashParams)
{
	{
		//resetting the heap and SDF blocks
		const dim3 gridSize((hashParams.m_numSDFBlocks + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
		const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);

		resetHeapKernel<<<gridSize, blockSize>>>(hashData);

		CUDA_CHECKED_NO_ERROR();
	}

	{
		//resetting the hash
		const dim3 gridSize((HASH_BUCKET_SIZE * hashParams.m_hashNumBuckets + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
		const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);

		resetHashKernel<<<gridSize, blockSize>>>(hashData);
		CUDA_CHECKED_NO_ERROR();
	}

	{
		//resetting the mutex
		const dim3 gridSize((hashParams.m_hashNumBuckets + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
		const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);

		resetHashBucketMutexKernel<<<gridSize, blockSize>>>(hashData);

		CUDA_CHECKED_NO_ERROR();

	}


}

void resetHashBucketMutexCUDA(HashDataStruct& hashData, const HashParams& hashParams)
{
	const dim3 gridSize((hashParams.m_hashNumBuckets + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
	const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);

	resetHashBucketMutexKernel<<<gridSize, blockSize>>>(hashData);

	CUDA_CHECKED_NO_ERROR();

}


__device__
unsigned int linearizeChunkPos(const int3& chunkPos)
{
	int3 p = chunkPos-c_hashParams.m_streamingMinGridPos;
	return  p.z * c_hashParams.m_streamingGridDimensions.x * c_hashParams.m_streamingGridDimensions.y +
			p.y * c_hashParams.m_streamingGridDimensions.x +
			p.x;
}

__device__
int3 worldToChunks(const float3& posWorld)
{
	float3 p;
	p.x = posWorld.x/c_hashParams.m_streamingVoxelExtents.x;
	p.y = posWorld.y/c_hashParams.m_streamingVoxelExtents.y;
	p.z = posWorld.z/c_hashParams.m_streamingVoxelExtents.z;

	float3 s;
	s.x = (float)sign(p.x);
	s.y = (float)sign(p.y);
	s.z = (float)sign(p.z);

	return make_int3(p+s*0.5f);
}

__global__ void allocKernel(HashDataStruct hashData, CUDAFrame frame)
{
    const HashParams& hashParams = c_hashParams;
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < frame.imageWidth && y < frame.imageHeight)
	{
        float d = tex2D(depthTextureRef, x, y);

		if (d == MINF || d == 0.0f)	return;
		if (d >= hashParams.m_maxIntegrationDistance) return;

		float t = hashData.getTruncation(d);
        float minDepth = min(hashParams.m_maxIntegrationDistance, d-t);
        float maxDepth = min(hashParams.m_maxIntegrationDistance, d+t);
		if (minDepth >= maxDepth) return;

		float3 rayMin = frame.unProject(x, y, minDepth);
		rayMin = hashParams.m_rigidTransform * rayMin;
		float3 rayMax = frame.unProject(x, y, maxDepth);
		rayMax = hashParams.m_rigidTransform * rayMax;

		float3 rayDir = normalize(rayMax - rayMin);

		int3 idCurrentVoxel = hashData.worldToSDFBlock(rayMin);
		int3 idEnd = hashData.worldToSDFBlock(rayMax);

		float3 step = make_float3(sign(rayDir));
		float3 boundaryPos = hashData.SDFBlockToWorld(idCurrentVoxel+make_int3(clamp(step, 0.0, 1.0f)))-0.5f*hashParams.m_virtualVoxelSize;
		float3 tMax = (boundaryPos-rayMin)/rayDir;
		float3 tDelta = (step*SDF_BLOCK_SIZE*hashParams.m_virtualVoxelSize)/rayDir;
		int3 idBound = make_int3(make_float3(idEnd)+step);

		//#pragma unroll
		//for(int c = 0; c < 3; c++) {
		//	if (rayDir[c] == 0.0f) { tMax[c] = PINF; tDelta[c] = PINF; }
		//	if (boundaryPos[c] - rayMin[c] == 0.0f) { tMax[c] = PINF; tDelta[c] = PINF; }
		//}
		if (rayDir.x == 0.0f) { tMax.x = PINF; tDelta.x = PINF; }
		if (boundaryPos.x - rayMin.x == 0.0f) { tMax.x = PINF; tDelta.x = PINF; }

		if (rayDir.y == 0.0f) { tMax.y = PINF; tDelta.y = PINF; }
		if (boundaryPos.y - rayMin.y == 0.0f) { tMax.y = PINF; tDelta.y = PINF; }

		if (rayDir.z == 0.0f) { tMax.z = PINF; tDelta.z = PINF; }
		if (boundaryPos.z - rayMin.z == 0.0f) { tMax.z = PINF; tDelta.z = PINF; }


		unsigned int iter = 0; // iter < g_MaxLoopIterCount
		unsigned int g_MaxLoopIterCount = 1024;	//TODO MATTHIAS MOVE TO GLOBAL APP STATE
#pragma unroll 1
		while(iter < g_MaxLoopIterCount) {
			//check if it's in the frustum and not checked out
			if (hashData.isSDFBlockInCameraFrustumApprox(idCurrentVoxel, frame)) {
				hashData.allocBlock(idCurrentVoxel);
			}

			// Traverse voxel grid
			if(tMax.x < tMax.y && tMax.x < tMax.z)	{
				idCurrentVoxel.x += step.x;
				if(idCurrentVoxel.x == idBound.x) return;
				tMax.x += tDelta.x;
			}
			else if(tMax.z < tMax.y) {
				idCurrentVoxel.z += step.z;
				if(idCurrentVoxel.z == idBound.z) return;
				tMax.z += tDelta.z;
			}
			else	{
				idCurrentVoxel.y += step.y;
				if(idCurrentVoxel.y == idBound.y) return;
				tMax.y += tDelta.y;
			}

			iter++;
		}
	}
}

void allocCUDA(HashDataStruct& hashData, const HashParams& hashParams, const CUDAFrame& frame)
{
	const dim3 gridSize((frame.imageWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (frame.imageHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	allocKernel<<<gridSize, blockSize>>>(hashData, frame);

	CUDA_CHECKED_NO_ERROR();

}



__global__ void fillDecisionArrayKernel(HashDataStruct hashData, CUDAFrame frame)
{
	const HashParams& hashParams = c_hashParams;
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < hashParams.m_hashNumBuckets * HASH_BUCKET_SIZE) {
		hashData.d_hashDecision[idx] = 0;
		if (hashData.d_hash[idx].ptr != FREE_ENTRY) {
			if (hashData.isSDFBlockInCameraFrustumApprox(hashData.d_hash[idx].pos, frame))
			{
				hashData.d_hashDecision[idx] = 1;	//yes
			}
		}
	}
}

void fillDecisionArrayCUDA(HashDataStruct& hashData, const HashParams& hashParams, const CUDAFrame& frame)
{
	const dim3 gridSize((HASH_BUCKET_SIZE * hashParams.m_hashNumBuckets + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
	const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);

	fillDecisionArrayKernel<<<gridSize, blockSize>>>(hashData, frame);

	CUDA_CHECKED_NO_ERROR();

}

__global__ void compactifyHashKernel(HashDataStruct hashData)
{
	const HashParams& hashParams = c_hashParams;
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < hashParams.m_hashNumBuckets * HASH_BUCKET_SIZE) {
		if (hashData.d_hashDecision[idx] == 1) {
			hashData.d_hashCompactified[hashData.d_hashDecisionPrefix[idx]-1] = hashData.d_hash[idx];
		}
	}
}

void compactifyHashCUDA(HashDataStruct& hashData, const HashParams& hashParams)
{
	const dim3 gridSize((HASH_BUCKET_SIZE * hashParams.m_hashNumBuckets + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
	const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);

	compactifyHashKernel<<<gridSize, blockSize>>>(hashData);

	CUDA_CHECKED_NO_ERROR();

}

#define COMPACTIFY_HASH_THREADS_PER_BLOCK 256
//#define COMPACTIFY_HASH_SIMPLE
__global__ void compactifyHashAllInOneKernel(HashDataStruct hashData, CUDAFrame frame)
{
	const HashParams& hashParams = c_hashParams;
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ int localCounter;
	if (threadIdx.x == 0) localCounter = 0;
	__syncthreads();

	int addrLocal = -1;
	if (idx < hashParams.m_hashNumBuckets * HASH_BUCKET_SIZE) {
		if (hashData.d_hash[idx].ptr != FREE_ENTRY) {
			if (hashData.isSDFBlockInCameraFrustumApprox(hashData.d_hash[idx].pos, frame))
			{
				addrLocal = atomicAdd(&localCounter, 1);
			}
		}
	}

	__syncthreads();

	__shared__ int addrGlobal;
	if (threadIdx.x == 0 && localCounter > 0) {
		addrGlobal = atomicAdd(hashData.d_hashCompactifiedCounter, localCounter);
	}
	__syncthreads();

	if (addrLocal != -1) {
		const unsigned int addr = addrGlobal + addrLocal;
		hashData.d_hashCompactified[addr] = hashData.d_hash[idx];
	}
}

unsigned int compactifyHashAllInOneCUDA(HashDataStruct& hashData, const HashParams& hashParams, const CUDAFrame &frame)
{
	const unsigned int threadsPerBlock = COMPACTIFY_HASH_THREADS_PER_BLOCK;
	const dim3 gridSize((HASH_BUCKET_SIZE * hashParams.m_hashNumBuckets + threadsPerBlock - 1) / threadsPerBlock, 1);
	const dim3 blockSize(threadsPerBlock, 1);

	cutilSafeCall(cudaMemset(hashData.d_hashCompactifiedCounter, 0, sizeof(int)));
	compactifyHashAllInOneKernel << <gridSize, blockSize >> >(hashData, frame);
	unsigned int res = 0;
	cutilSafeCall(cudaMemcpy(&res, hashData.d_hashCompactifiedCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));

	CUDA_CHECKED_NO_ERROR();

	return res;
}

template<bool deIntegrate = false>
__global__ void integrateDepthMapKernel(HashDataStruct hashData, CUDAFrame frame) {
	const HashParams& hashParams = c_hashParams;
	const HashEntry& entry = hashData.d_hashCompactified[blockIdx.x];

	int3 pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos);

	uint i = threadIdx.x;	//inside of an SDF block
	int3 pi = pi_base + make_int3(hashData.delinearizeVoxelIndex(i));
	float3 pf = hashData.virtualVoxelPosToWorld(pi);

	pf = hashParams.m_rigidTransformInverse * pf;
	float3 pixel = frame.project(pf);
	uint2 screenPos = make_uint2((uint)pixel.x, (uint)pixel.y);


	if (screenPos.x < frame.imageWidth && screenPos.y < frame.imageHeight) {	//on screen

		//float depth = g_InputDepth[screenPos];
		float depth = tex2D(depthTextureRef, screenPos.x, screenPos.y);
		uchar4 color_uc = tex2D(colorTextureRef, screenPos.x, screenPos.y);
		float3 color = make_float3(color_uc.x, color_uc.y, color_uc.z);

		if (color.x != MINF && depth != MINF) { // valid depth and color
		//if (depth != MINF) {	//valid depth

			if (depth < hashParams.m_maxIntegrationDistance) {
				float depthZeroOne = frame.cameraToProjZ(depth);

				float sdf = depth - pf.z;
				float truncation = hashData.getTruncation(depth);
				//if (sdf > -truncation)
				if (abs(sdf) < truncation)
				{
					if (sdf >= 0.0f) {
						sdf = fminf(truncation, sdf);
					} else {
						sdf = fmaxf(-truncation, sdf);
					}

					float weightUpdate = max(hashParams.m_integrationWeightSample * 1.5f * (1.0f-depthZeroOne), 1.0f);
					weightUpdate = 1.0f;	//TODO remove that again

					Voxel curr;	//construct current voxel
					curr.sdf = sdf;
					curr.weight = weightUpdate;

					curr.color = make_uchar4(color.x, color.y, color.z, 255);

					uint idx = entry.ptr + i;

					const Voxel& oldVoxel = hashData.d_SDFBlocks[idx];
					Voxel newVoxel;

					float3 oldColor = make_float3(oldVoxel.color.x, oldVoxel.color.y, oldVoxel.color.z);
					float3 currColor = make_float3(curr.color.x, curr.color.y, curr.color.z);

					if (!deIntegrate) {	//integration
						//hashData.combineVoxel(hashData.d_SDFBlocks[idx], curr, newVoxel);
						float3 res;
						if (oldVoxel.weight == 0) res = currColor;
						//else res = (currColor + oldColor) / 2;
						else res = 0.2f * currColor + 0.8f * oldColor;
						//float3 res = (currColor*curr.weight + oldColor*oldVoxel.weight) / (curr.weight + oldVoxel.weight);
						res = make_float3(round(res.x), round(res.y), round(res.z));
						res = fmaxf(make_float3(0.0f), fminf(res, make_float3(254.5f)));
						//newVoxel.color.x = (uchar)(res.x + 0.5f);	newVoxel.color.y = (uchar)(res.y + 0.5f);	newVoxel.color.z = (uchar)(res.z + 0.5f);
						newVoxel.color = make_uchar4(res.x, res.y, res.z, 255);
						newVoxel.sdf = (curr.sdf*curr.weight + oldVoxel.sdf*oldVoxel.weight) / (curr.weight + oldVoxel.weight);
						newVoxel.weight = min((float)c_hashParams.m_integrationWeightMax, curr.weight + oldVoxel.weight);
					}
					else {				//deintegration
						//float3 res = 2 * c0 - c1;
						float3 res = (oldColor*oldVoxel.weight - currColor*curr.weight) / (oldVoxel.weight - curr.weight);
						res = make_float3(round(res.x), round(res.y), round(res.z));
						res = fmaxf(make_float3(0.0f), fminf(res, make_float3(254.5f)));
						//newVoxel.color.x = (uchar)(res.x + 0.5f);	newVoxel.color.y = (uchar)(res.y + 0.5f);	newVoxel.color.z = (uchar)(res.z + 0.5f);
						newVoxel.color = make_uchar4(res.x, res.y, res.z, 255);
						newVoxel.sdf = (oldVoxel.sdf*oldVoxel.weight - curr.sdf*curr.weight) / (oldVoxel.weight - curr.weight);
						newVoxel.weight = max(0.0f, oldVoxel.weight - curr.weight);
						if (newVoxel.weight <= 0.001f) {
							newVoxel.sdf = 0.0f;
							newVoxel.color = make_uchar4(0,0,0,0);
							newVoxel.weight = 0.0f;
						}
					}

					hashData.d_SDFBlocks[idx] = newVoxel;
				}
			}
		}
	}
}


void integrateDepthMapCUDA(HashDataStruct& hashData, const HashParams& hashParams, const CUDAFrame& frame)
{
	const unsigned int threadsPerBlock = SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;
	const dim3 gridSize(hashParams.m_numOccupiedBlocks, 1);
	const dim3 blockSize(threadsPerBlock, 1);

	integrateDepthMapKernel<false> <<<gridSize, blockSize>>>(hashData, frame);

	CUDA_CHECKED_NO_ERROR();

}

void deIntegrateDepthMapCUDA(HashDataStruct& hashData, const HashParams& hashParams, const CUDAFrame& frame)
{
	const unsigned int threadsPerBlock = SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;
	const dim3 gridSize(hashParams.m_numOccupiedBlocks, 1);
	const dim3 blockSize(threadsPerBlock, 1);

	integrateDepthMapKernel<true> <<<gridSize, blockSize >>>(hashData, frame);

	CUDA_CHECKED_NO_ERROR();

}



__global__ void starveVoxelsKernel(HashDataStruct hashData) {

	const uint idx = blockIdx.x;
	const HashEntry& entry = hashData.d_hashCompactified[idx];

	//is typically exectued only every n'th frame
	int weight = hashData.d_SDFBlocks[entry.ptr + threadIdx.x].weight;
	weight = max(0, weight-1);
	hashData.d_SDFBlocks[entry.ptr + threadIdx.x].weight = weight;
}

void starveVoxelsKernelCUDA(HashDataStruct& hashData, const HashParams& hashParams)
{
	const unsigned int threadsPerBlock = SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;
	const dim3 gridSize(hashParams.m_numOccupiedBlocks, 1);
	const dim3 blockSize(threadsPerBlock, 1);

	starveVoxelsKernel<<<gridSize, blockSize>>>(hashData);

	CUDA_CHECKED_NO_ERROR();

}


//__shared__ float	shared_MinSDF[SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2];
__shared__ uint		shared_MaxWeight[SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2];


__global__ void garbageCollectIdentifyKernel(HashDataStruct hashData) {

	const unsigned int hashIdx = blockIdx.x;
	const HashEntry& entry = hashData.d_hashCompactified[hashIdx];

	//uint h = hashData.computeHashPos(entry.pos);
	//hashData.d_hashDecision[hashIdx] = 1;
	//if (hashData.d_hashBucketMutex[h] == LOCK_ENTRY)	return;

	//if (entry.ptr == FREE_ENTRY) return; //should never happen since we did compactify before
	//const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

	const unsigned int idx0 = entry.ptr + 2*threadIdx.x+0;
	const unsigned int idx1 = entry.ptr + 2*threadIdx.x+1;

	Voxel v0 = hashData.d_SDFBlocks[idx0];
	Voxel v1 = hashData.d_SDFBlocks[idx1];

	//if (v0.weight == 0)	v0.sdf = PINF;
	//if (v1.weight == 0)	v1.sdf = PINF;

	//shared_MinSDF[threadIdx.x] = min(fabsf(v0.sdf), fabsf(v1.sdf));	//init shared memory
	shared_MaxWeight[threadIdx.x] = max(v0.weight, v1.weight);

#pragma unroll 1
	for (uint stride = 2; stride <= blockDim.x; stride <<= 1) {
		__syncthreads();
		if ((threadIdx.x  & (stride-1)) == (stride-1)) {
			//shared_MinSDF[threadIdx.x] = min(shared_MinSDF[threadIdx.x-stride/2], shared_MinSDF[threadIdx.x]);
			shared_MaxWeight[threadIdx.x] = max(shared_MaxWeight[threadIdx.x-stride/2], shared_MaxWeight[threadIdx.x]);
		}
	}

	__syncthreads();

	if (threadIdx.x == blockDim.x - 1) {
		uint maxWeight = shared_MaxWeight[threadIdx.x];

		if (maxWeight == 0) {
			hashData.d_hashDecision[hashIdx] = 1;
		} else {
			hashData.d_hashDecision[hashIdx] = 0; 
		}
	}
}
 
void garbageCollectIdentifyCUDA(HashDataStruct& hashData, const HashParams& hashParams) {
	
	const unsigned int threadsPerBlock = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2;
	const dim3 gridSize(hashParams.m_numOccupiedBlocks, 1);
	const dim3 blockSize(threadsPerBlock, 1);

	garbageCollectIdentifyKernel<<<gridSize, blockSize>>>(hashData);

	CUDA_CHECKED_NO_ERROR();

}


__global__ void garbageCollectFreeKernel(HashDataStruct hashData) {

	//const uint hashIdx = blockIdx.x;
	const uint hashIdx = blockIdx.x*blockDim.x + threadIdx.x;


	if (hashIdx < c_hashParams.m_numOccupiedBlocks && hashData.d_hashDecision[hashIdx] != 0) {	//decision to delete the hash entry

		const HashEntry& entry = hashData.d_hashCompactified[hashIdx];
		//if (entry.ptr == FREE_ENTRY) return; //should never happen since we did compactify before

		if (hashData.deleteHashEntryElement(entry.pos)) {	//delete hash entry from hash (and performs heap append)
			const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

			#pragma unroll 1
			for (uint i = 0; i < linBlockSize; i++) {	//clear sdf block: CHECK TODO another kernel?
				hashData.deleteVoxel(entry.ptr + i);
			}
		}
	}
}


void garbageCollectFreeCUDA(HashDataStruct& hashData, const HashParams& hashParams) {
	
	const unsigned int threadsPerBlock = T_PER_BLOCK*T_PER_BLOCK;
	const dim3 gridSize((hashParams.m_numOccupiedBlocks + threadsPerBlock - 1) / threadsPerBlock, 1);
	const dim3 blockSize(threadsPerBlock, 1);

	garbageCollectFreeKernel<<<gridSize, blockSize>>>(hashData);

	CUDA_CHECKED_NO_ERROR();

}


/** raycast */
__device__ __host__
RayCastData::RayCastData() {
    d_depth = NULL;
    d_depth4 = NULL;
    d_normals = NULL;
    d_colors = NULL;

    d_vertexBuffer = NULL;

    d_rayIntervalSplatMinArray = NULL;
    d_rayIntervalSplatMaxArray = NULL;
}

extern "C" void updateConstantRayCastParams(const RayCastParams& params) {

	size_t size;
	cutilSafeCall(cudaGetSymbolSize(&size, c_rayCastParams));
	cutilSafeCall(cudaMemcpyToSymbol(c_rayCastParams, &params, size, 0, cudaMemcpyHostToDevice));

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

}

__host__
void RayCastData::updateParams(const RayCastParams &params) {
	updateConstantRayCastParams(params);
}

/////////////////
// Device part //
/////////////////
__device__
const RayCastParams &RayCastData::params() const {
    return c_rayCastParams;
}

__device__
float RayCastData::frac(float val) const {
    return (val - floorf(val));
}

__device__
float3 RayCastData::frac(const float3 &val) const {
    return make_float3(frac(val.x), frac(val.y), frac(val.z));
}

__device__
bool RayCastData::trilinearInterpolationSimpleFastFast(const HashDataStruct &hash, const float3 &pos, float &dist,
                                                       uchar3 &color) const {
    const float oSet = c_hashParams.m_virtualVoxelSize;
    const float3 posDual = pos - make_float3(oSet / 2.0f, oSet / 2.0f, oSet / 2.0f);
    float3 weight = frac(hash.worldToVirtualVoxelPosFloat(pos));

    dist = 0.0f;
    float3 colorFloat = make_float3(0.0f, 0.0f, 0.0f);
    Voxel v = hash.getVoxel(posDual + make_float3(0.0f, 0.0f, 0.0f));
    if (v.weight == 0) return false;
    float3 vColor = make_float3(v.color.x, v.color.y, v.color.z);
    dist += (1.0f - weight.x) * (1.0f - weight.y) * (1.0f - weight.z) * v.sdf;
    colorFloat += (1.0f - weight.x) * (1.0f - weight.y) * (1.0f - weight.z) * vColor;
    v = hash.getVoxel(posDual + make_float3(oSet, 0.0f, 0.0f));
    if (v.weight == 0) return false;
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    dist += weight.x * (1.0f - weight.y) * (1.0f - weight.z) * v.sdf;
    colorFloat += weight.x * (1.0f - weight.y) * (1.0f - weight.z) * vColor;
    v = hash.getVoxel(posDual + make_float3(0.0f, oSet, 0.0f));
    if (v.weight == 0) return false;
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    dist += (1.0f - weight.x) * weight.y * (1.0f - weight.z) * v.sdf;
    colorFloat += (1.0f - weight.x) * weight.y * (1.0f - weight.z) * vColor;
    v = hash.getVoxel(posDual + make_float3(0.0f, 0.0f, oSet));
    if (v.weight == 0) return false;
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    dist += (1.0f - weight.x) * (1.0f - weight.y) * weight.z * v.sdf;
    colorFloat += (1.0f - weight.x) * (1.0f - weight.y) * weight.z * vColor;
    v = hash.getVoxel(posDual + make_float3(oSet, oSet, 0.0f));
    if (v.weight == 0) return false;
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    dist += weight.x * weight.y * (1.0f - weight.z) * v.sdf;
    colorFloat += weight.x * weight.y * (1.0f - weight.z) * vColor;
    v = hash.getVoxel(posDual + make_float3(0.0f, oSet, oSet));
    if (v.weight == 0) return false;
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    dist += (1.0f - weight.x) * weight.y * weight.z * v.sdf;
    colorFloat += (1.0f - weight.x) * weight.y * weight.z * vColor;
    v = hash.getVoxel(posDual + make_float3(oSet, 0.0f, oSet));
    if (v.weight == 0) return false;
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    dist += weight.x * (1.0f - weight.y) * weight.z * v.sdf;
    colorFloat += weight.x * (1.0f - weight.y) * weight.z * vColor;
    v = hash.getVoxel(posDual + make_float3(oSet, oSet, oSet));
    if (v.weight == 0) return false;
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    dist += weight.x * weight.y * weight.z * v.sdf;
    colorFloat += weight.x * weight.y * weight.z * vColor;

    color = make_uchar3(colorFloat.x, colorFloat.y, colorFloat.z);//v.color;

    return true;
}

__device__
float RayCastData::findIntersectionLinear(float tNear, float tFar, float dNear, float dFar) const {
    return tNear + (dNear / (dNear - dFar)) * (tFar - tNear);
}

// d0 near, d1 far
__device__
bool RayCastData::findIntersectionBisection(const HashDataStruct &hash, const float3 &worldCamPos, const float3 &worldDir,
                                            float d0, float r0, float d1, float r1, float &alpha, uchar3 &color) const {
    float a = r0;
    float aDist = d0;
    float b = r1;
    float bDist = d1;
    float c = 0.0f;

#pragma unroll 1
    for (uint i = 0; i < nIterationsBisection; i++) {
        c = findIntersectionLinear(a, b, aDist, bDist);

        float cDist;
        if (!trilinearInterpolationSimpleFastFast(hash, worldCamPos + c * worldDir, cDist, color)) return false;

        if (aDist * cDist > 0.0) {
            a = c;
            aDist = cDist;
        }
        else {
            b = c;
            bDist = cDist;
        }
    }

    alpha = c;

    return true;
}


__device__
float3 RayCastData::gradientForPoint(const HashDataStruct &hash, const float3 &pos) const {
    const float voxelSize = c_hashParams.m_virtualVoxelSize;
    float3 offset = make_float3(voxelSize, voxelSize, voxelSize);

    float distp00;
    uchar3 colorp00;
    trilinearInterpolationSimpleFastFast(hash, pos - make_float3(0.5f * offset.x, 0.0f, 0.0f), distp00, colorp00);
    float dist0p0;
    uchar3 color0p0;
    trilinearInterpolationSimpleFastFast(hash, pos - make_float3(0.0f, 0.5f * offset.y, 0.0f), dist0p0, color0p0);
    float dist00p;
    uchar3 color00p;
    trilinearInterpolationSimpleFastFast(hash, pos - make_float3(0.0f, 0.0f, 0.5f * offset.z), dist00p, color00p);

    float dist100;
    uchar3 color100;
    trilinearInterpolationSimpleFastFast(hash, pos + make_float3(0.5f * offset.x, 0.0f, 0.0f), dist100, color100);
    float dist010;
    uchar3 color010;
    trilinearInterpolationSimpleFastFast(hash, pos + make_float3(0.0f, 0.5f * offset.y, 0.0f), dist010, color010);
    float dist001;
    uchar3 color001;
    trilinearInterpolationSimpleFastFast(hash, pos + make_float3(0.0f, 0.0f, 0.5f * offset.z), dist001, color001);

    float3 grad = make_float3((distp00 - dist100) / offset.x, (dist0p0 - dist010) / offset.y,
                              (dist00p - dist001) / offset.z);

    float l = length(grad);
    if (l == 0.0f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    return -grad / l;
}

__device__
void RayCastData::traverseCoarseGridSimpleSampleAll(const HashDataStruct& hash, const float3& worldCamPos, const float3& worldDir, const float3& camDir, const int3& dTid, float minInterval, float maxInterval) const
{
    const RayCastParams& rayCastParams = c_rayCastParams;

    // Last Sample
    RayCastSample lastSample; lastSample.sdf = 0.0f; lastSample.alpha = 0.0f; lastSample.weight = 0; // lastSample.color = int3(0, 0, 0);
    const float depthToRayLength = 1.0f/camDir.z; // scale factor to convert from depth to ray length

    float rayCurrent = depthToRayLength * max(rayCastParams.m_minDepth, minInterval);	// Convert depth to raylength
    float rayEnd = depthToRayLength * min(rayCastParams.m_maxDepth, maxInterval);		// Convert depth to raylength
    //float rayCurrent = depthToRayLength * rayCastParams.m_minDepth;	// Convert depth to raylength
    //float rayEnd = depthToRayLength * rayCastParams.m_maxDepth;		// Convert depth to raylength

#pragma unroll 1
    while(rayCurrent < rayEnd)
    {
        float3 currentPosWorld = worldCamPos+rayCurrent*worldDir;
        float dist;	uchar3 color;

        if(trilinearInterpolationSimpleFastFast(hash, currentPosWorld, dist, color))
        {
            if(lastSample.weight > 0 && lastSample.sdf > 0.0f && dist < 0.0f)// current sample is always valid here
                //if(lastSample.weight > 0 && ((lastSample.sdf > 0.0f && dist < 0.0f) || (lastSample.sdf < 0.0f && dist > 0.0f))) //hack for top down video
            {

                float alpha; // = findIntersectionLinear(lastSample.alpha, rayCurrent, lastSample.sdf, dist);
                uchar3 color2;
                bool b = findIntersectionBisection(hash, worldCamPos, worldDir, lastSample.sdf, lastSample.alpha, dist, rayCurrent, alpha, color2);

                float3 currentIso = worldCamPos+alpha*worldDir;
                if(b && abs(lastSample.sdf - dist) < rayCastParams.m_thresSampleDist)
                {
                    if(abs(dist) < rayCastParams.m_thresDist)
                    {
                        float depth = alpha / depthToRayLength; // Convert ray length to depth depthToRayLength

                        d_depth[dTid.y*rayCastParams.m_width+dTid.x] = depth;
                        d_depth4[dTid.y*rayCastParams.m_width+dTid.x] = make_float4(depthToCamera(dTid.x, dTid.y, depth), 1.0f);
                        d_colors[dTid.y*rayCastParams.m_width+dTid.x] = make_float4(color2.x/255.f, color2.y/255.f, color2.z/255.f, 1.0f);

                        if(rayCastParams.m_useGradients)
                        {
                            float3 normal = make_float3(0,0,0)-gradientForPoint(hash, currentIso);
                            float4 n = rayCastParams.m_viewMatrix * make_float4(normal, 0.0f);
                            d_normals[dTid.y*rayCastParams.m_width+dTid.x] = make_float4(n.x, n.y, n.z, 1.0f);
                        }

                        return;
                    }
                }
            }

            lastSample.sdf = dist;
            lastSample.alpha = rayCurrent;
            // lastSample.color = color;
            lastSample.weight = 1;
            rayCurrent += rayCastParams.m_rayIncrement;
        } else {
            lastSample.weight = 0;
            rayCurrent += rayCastParams.m_rayIncrement;
        }


    }

}
__global__ void computeNormalsDevice(float4* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	d_output[y*width+x] = make_float4(MINF, MINF, MINF, MINF);

	if(x > 0 && x < width-1 && y > 0 && y < height-1)
	{
		const float4 CC = d_input[(y+0)*width+(x+0)];
		const float4 PC = d_input[(y+1)*width+(x+0)];
		const float4 CP = d_input[(y+0)*width+(x+1)];
		const float4 MC = d_input[(y-1)*width+(x+0)];
		const float4 CM = d_input[(y+0)*width+(x-1)];

		if(CC.x != MINF && PC.x != MINF && CP.x != MINF && MC.x != MINF && CM.x != MINF)
		{
			const float3 n = cross(make_float3(PC)-make_float3(MC), make_float3(CP)-make_float3(CM));
			const float  l = length(n);

			if(l > 0.0f)
			{
				d_output[y*width+x] = make_float4(n/-l, 1.0f);
			}
		}
	}
}

extern "C" void computeNormals(float4* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeNormalsDevice<<<gridSize, blockSize>>>(d_output, d_input, width, height);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

texture<float, cudaTextureType2D, cudaReadModeElementType> rayMinTextureRef;
texture<float, cudaTextureType2D, cudaReadModeElementType> rayMaxTextureRef;

__global__ void renderKernel(HashDataStruct hashData, RayCastData rayCastData)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	const RayCastParams& rayCastParams = c_rayCastParams;

	if (x < rayCastParams.m_width && y < rayCastParams.m_height) {
		rayCastData.d_depth[y*rayCastParams.m_width+x] = MINF;
		rayCastData.d_depth4[y*rayCastParams.m_width+x] = make_float4(MINF,MINF,MINF,MINF);
		rayCastData.d_normals[y*rayCastParams.m_width+x] = make_float4(MINF,MINF,MINF,MINF);
		rayCastData.d_colors[y*rayCastParams.m_width+x] = make_float4(MINF,MINF,MINF,MINF);

		float3 camDir = normalize(RayCastData::depthToCamera(x, y, 1.0f));
		float3 worldCamPos = rayCastParams.m_viewMatrixInverse * make_float3(0.0f, 0.0f, 0.0f);
		float4 w = rayCastParams.m_viewMatrixInverse * make_float4(camDir, 0.0f);
		float3 worldDir = normalize(make_float3(w.x, w.y, w.z));

		float minInterval = tex2D(rayMinTextureRef, x, y);
		float maxInterval = tex2D(rayMaxTextureRef, x, y);

		//float minInterval = rayCastParams.m_minDepth;
		//float maxInterval = rayCastParams.m_maxDepth;

		//if (minInterval == 0 || minInterval == MINF) minInterval = rayCastParams.m_minDepth;
		//if (maxInterval == 0 || maxInterval == MINF) maxInterval = rayCastParams.m_maxDepth;
		//TODO MATTHIAS: shouldn't this return in the case no interval is found?
		if (minInterval == 0 || minInterval == MINF) return;
		if (maxInterval == 0 || maxInterval == MINF) return;

		minInterval = max(minInterval, rayCastParams.m_minDepth);
		maxInterval = min(maxInterval, rayCastParams.m_maxDepth);

		// debugging
		//if (maxInterval < minInterval) {
		//	printf("ERROR (%d,%d): [ %f, %f ]\n", x, y, minInterval, maxInterval);
		//}

		rayCastData.traverseCoarseGridSimpleSampleAll(hashData, worldCamPos, worldDir, camDir, make_int3(x,y,1), minInterval, maxInterval);
	}
}

extern "C" void renderCS(const HashDataStruct& hashData, const RayCastData &rayCastData, const RayCastParams &rayCastParams)
{

	const dim3 gridSize((rayCastParams.m_width + T_PER_BLOCK - 1)/T_PER_BLOCK, (rayCastParams.m_height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	cudaBindTexture2D(0, &rayMinTextureRef, rayCastData.d_rayIntervalSplatMinArray, &depthTextureRef.channelDesc, rayCastParams.m_width, rayCastParams.m_height, sizeof(float)*rayCastParams.m_width);
	cudaBindTexture2D(0, &rayMaxTextureRef, rayCastData.d_rayIntervalSplatMaxArray, &depthTextureRef.channelDesc, rayCastParams.m_width, rayCastParams.m_height, sizeof(float)*rayCastParams.m_width);

	renderKernel<<<gridSize, blockSize>>>(hashData, rayCastData);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

/////////////////////////////////////////////////////////////////////////
// ray interval splatting
/////////////////////////////////////////////////////////////////////////

__global__ void resetRayIntervalSplatKernel(RayCastData data)
{
	uint idx = blockIdx.x + blockIdx.y * NUM_GROUPS_X;
	data.d_vertexBuffer[idx] = make_float4(MINF);
}

extern "C" void resetRayIntervalSplatCUDA(RayCastData& data, const RayCastParams& params)
{
	const dim3 gridSize(NUM_GROUPS_X, (params.m_maxNumVertices + NUM_GROUPS_X - 1) / NUM_GROUPS_X, 1); // ! todo check if need third dimension?
	const dim3 blockSize(1, 1, 1);

	resetRayIntervalSplatKernel<<<gridSize, blockSize>>>(data);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

__global__ void rayIntervalSplatKernel(HashDataStruct hashData, RayCastData rayCastData)
{
	uint idx = blockIdx.x + blockIdx.y * NUM_GROUPS_X;

	const HashEntry& entry = hashData.d_hashCompactified[idx];
	const RayCastParams& rayCastParams = c_rayCastParams;
	if (entry.ptr != FREE_ENTRY) {
		float3 posWorld = hashData.virtualVoxelPosToWorld(hashData.SDFBlockToVirtualVoxelPos(entry.pos)) +
						  c_hashParams.m_virtualVoxelSize * 0.5f * (SDF_BLOCK_SIZE - 1.0f);
		if (rayCastData.isInCameraFrustumApprox(rayCastParams.m_viewMatrixInverse, posWorld)) return;
		const RayCastParams &params = c_rayCastParams;
		const float4x4& viewMatrix = params.m_viewMatrix;

		float3 worldCurrentVoxel = hashData.SDFBlockToWorld(entry.pos);

		float3 MINV = worldCurrentVoxel - c_hashParams.m_virtualVoxelSize / 2.0f;

		float3 maxv = MINV+SDF_BLOCK_SIZE*c_hashParams.m_virtualVoxelSize;

		//float3 proj000 = DepthCameraData::cameraToKinectProj(viewMatrix * make_float3(MINV.x, MINV.y, MINV.z));
		//float3 proj100 = DepthCameraData::cameraToKinectProj(viewMatrix * make_float3(maxv.x, MINV.y, MINV.z));
		//float3 proj010 = DepthCameraData::cameraToKinectProj(viewMatrix * make_float3(MINV.x, maxv.y, MINV.z));
		//float3 proj001 = DepthCameraData::cameraToKinectProj(viewMatrix * make_float3(MINV.x, MINV.y, maxv.z));
		//float3 proj110 = DepthCameraData::cameraToKinectProj(viewMatrix * make_float3(maxv.x, maxv.y, MINV.z));
		//float3 proj011 = DepthCameraData::cameraToKinectProj(viewMatrix * make_float3(MINV.x, maxv.y, maxv.z));
		//float3 proj101 = DepthCameraData::cameraToKinectProj(viewMatrix * make_float3(maxv.x, MINV.y, maxv.z));
		//float3 proj111 = DepthCameraData::cameraToKinectProj(viewMatrix * make_float3(maxv.x, maxv.y, maxv.z));
		float3 proj000 = RayCastData::cameraToDepthProj(viewMatrix * make_float3(MINV.x, MINV.y, MINV.z));
		float3 proj100 = RayCastData::cameraToDepthProj(viewMatrix * make_float3(maxv.x, MINV.y, MINV.z));
		float3 proj010 = RayCastData::cameraToDepthProj(viewMatrix * make_float3(MINV.x, maxv.y, MINV.z));
		float3 proj001 = RayCastData::cameraToDepthProj(viewMatrix * make_float3(MINV.x, MINV.y, maxv.z));
		float3 proj110 = RayCastData::cameraToDepthProj(viewMatrix * make_float3(maxv.x, maxv.y, MINV.z));
		float3 proj011 = RayCastData::cameraToDepthProj(viewMatrix * make_float3(MINV.x, maxv.y, maxv.z));
		float3 proj101 = RayCastData::cameraToDepthProj(viewMatrix * make_float3(maxv.x, MINV.y, maxv.z));
		float3 proj111 = RayCastData::cameraToDepthProj(viewMatrix * make_float3(maxv.x, maxv.y, maxv.z));

		// Tree Reduction Min
		float3 min00 = fminf(proj000, proj100);
		float3 min01 = fminf(proj010, proj001);
		float3 min10 = fminf(proj110, proj011);
		float3 min11 = fminf(proj101, proj111);

		float3 min0 = fminf(min00, min01);
		float3 min1 = fminf(min10, min11);

		float3 minFinal = fminf(min0, min1);

		// Tree Reduction Max
		float3 max00 = fmaxf(proj000, proj100);
		float3 max01 = fmaxf(proj010, proj001);
		float3 max10 = fmaxf(proj110, proj011);
		float3 max11 = fmaxf(proj101, proj111);

		float3 max0 = fmaxf(max00, max01);
		float3 max1 = fmaxf(max10, max11);

		float3 maxFinal = fmaxf(max0, max1);

		float depth = maxFinal.z;
		float * rayArray = rayCastData.d_rayIntervalSplatMaxArray;
		if(params.m_splatMinimum == 1) {
			depth = minFinal.z;
			rayArray = rayCastData.d_rayIntervalSplatMinArray;
		}
		float depthWorld = RayCastData::depthProjToCameraZ(depth);
		for(uint x=(uint)ceil(minFinal.x); x<maxFinal.x&&x<rayCastParams.m_width; x++) {
			for(uint y=(uint)ceil(minFinal.y); y<maxFinal.y&&y<rayCastParams.m_height; y++) {
				rayArray[y*rayCastParams.m_width+x] = depth;
			}
		}
	}
}

extern "C" void rayIntervalSplatCUDA(const HashDataStruct& hashData, const RayCastData &rayCastData, const RayCastParams &rayCastParams)
{

	const dim3 gridSize(NUM_GROUPS_X, (rayCastParams.m_numOccupiedSDFBlocks + NUM_GROUPS_X - 1) / NUM_GROUPS_X, 1);
	const dim3 blockSize(1, 1, 1);

	rayIntervalSplatKernel<<<gridSize, blockSize>>>(hashData, rayCastData);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


/** cube */
__device__ __host__
MarchingCubesData::MarchingCubesData() {
    d_params = NULL;
    d_triangles = NULL;
    d_numTriangles = NULL;
    m_bIsOnGPU = false;
}

/////////////////
// Device part //
/////////////////
__device__
void MarchingCubesData::extractIsoSurfaceAtPosition(const float3 &worldPos, const HashDataStruct &hashData,
                                                    const RayCastData &rayCastData) {
    const HashParams &hashParams = c_hashParams;
    const MarchingCubesParams &params = *d_params;

    if (params.m_boxEnabled == 1) {
        if (!isInBoxAA(params.m_minCorner, params.m_maxCorner, worldPos)) return;
    }

    const float isolevel = 0.0f;

    const float P = hashParams.m_virtualVoxelSize / 2.0f;
    const float M = -P;

    float3 p000 = worldPos + make_float3(M, M, M);
    float dist000;
    uchar3 color000;
    bool valid000 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p000, dist000, color000);
    float3 p100 = worldPos + make_float3(P, M, M);
    float dist100;
    uchar3 color100;
    bool valid100 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p100, dist100, color100);
    float3 p010 = worldPos + make_float3(M, P, M);
    float dist010;
    uchar3 color010;
    bool valid010 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p010, dist010, color010);
    float3 p001 = worldPos + make_float3(M, M, P);
    float dist001;
    uchar3 color001;
    bool valid001 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p001, dist001, color001);
    float3 p110 = worldPos + make_float3(P, P, M);
    float dist110;
    uchar3 color110;
    bool valid110 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p110, dist110, color110);
    float3 p011 = worldPos + make_float3(M, P, P);
    float dist011;
    uchar3 color011;
    bool valid011 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p011, dist011, color011);
    float3 p101 = worldPos + make_float3(P, M, P);
    float dist101;
    uchar3 color101;
    bool valid101 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p101, dist101, color101);
    float3 p111 = worldPos + make_float3(P, P, P);
    float dist111;
    uchar3 color111;
    bool valid111 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p111, dist111, color111);

    if (!valid000 || !valid100 || !valid010 || !valid001 || !valid110 || !valid011 || !valid101 || !valid111) return;

    uint cubeindex = 0;
    if (dist010 < isolevel) cubeindex += 1;
    if (dist110 < isolevel) cubeindex += 2;
    if (dist100 < isolevel) cubeindex += 4;
    if (dist000 < isolevel) cubeindex += 8;
    if (dist011 < isolevel) cubeindex += 16;
    if (dist111 < isolevel) cubeindex += 32;
    if (dist101 < isolevel) cubeindex += 64;
    if (dist001 < isolevel) cubeindex += 128;

    const float thres = params.m_threshMarchingCubes;
    float distArray[] = {dist000, dist100, dist010, dist001, dist110, dist011, dist101, dist111};
    for (uint k = 0; k < 8; k++) {
        for (uint l = 0; l < 8; l++) {
            if (distArray[k] * distArray[l] < 0.0f) {
                if (abs(distArray[k]) + abs(distArray[l]) > thres) return;
            } else {
                if (abs(distArray[k] - distArray[l]) > thres) return;
            }
        }
    }

    if (abs(dist000) > params.m_threshMarchingCubes2) return;
    if (abs(dist100) > params.m_threshMarchingCubes2) return;
    if (abs(dist010) > params.m_threshMarchingCubes2) return;
    if (abs(dist001) > params.m_threshMarchingCubes2) return;
    if (abs(dist110) > params.m_threshMarchingCubes2) return;
    if (abs(dist011) > params.m_threshMarchingCubes2) return;
    if (abs(dist101) > params.m_threshMarchingCubes2) return;
    if (abs(dist111) > params.m_threshMarchingCubes2) return;

    if (edgeTable[cubeindex] == 0 || edgeTable[cubeindex] == 255) return; // added by me edgeTable[cubeindex] == 255

    Voxel v = hashData.getVoxel(worldPos);

    Vertex vertlist[12];
    if (edgeTable[cubeindex] & 1) vertlist[0] = vertexInterp(isolevel, p010, p110, dist010, dist110, v.color, v.color);
    if (edgeTable[cubeindex] & 2) vertlist[1] = vertexInterp(isolevel, p110, p100, dist110, dist100, v.color, v.color);
    if (edgeTable[cubeindex] & 4) vertlist[2] = vertexInterp(isolevel, p100, p000, dist100, dist000, v.color, v.color);
    if (edgeTable[cubeindex] & 8) vertlist[3] = vertexInterp(isolevel, p000, p010, dist000, dist010, v.color, v.color);
    if (edgeTable[cubeindex] & 16) vertlist[4] = vertexInterp(isolevel, p011, p111, dist011, dist111, v.color, v.color);
    if (edgeTable[cubeindex] & 32) vertlist[5] = vertexInterp(isolevel, p111, p101, dist111, dist101, v.color, v.color);
    if (edgeTable[cubeindex] & 64) vertlist[6] = vertexInterp(isolevel, p101, p001, dist101, dist001, v.color, v.color);
    if (edgeTable[cubeindex] & 128)
        vertlist[7] = vertexInterp(isolevel, p001, p011, dist001, dist011, v.color, v.color);
    if (edgeTable[cubeindex] & 256)
        vertlist[8] = vertexInterp(isolevel, p010, p011, dist010, dist011, v.color, v.color);
    if (edgeTable[cubeindex] & 512)
        vertlist[9] = vertexInterp(isolevel, p110, p111, dist110, dist111, v.color, v.color);
    if (edgeTable[cubeindex] & 1024)
        vertlist[10] = vertexInterp(isolevel, p100, p101, dist100, dist101, v.color, v.color);
    if (edgeTable[cubeindex] & 2048)
        vertlist[11] = vertexInterp(isolevel, p000, p001, dist000, dist001, v.color, v.color);

    for (int i = 0; triTable[cubeindex][i] != -1; i += 3) {
        Triangle t;
        t.v0 = vertlist[triTable[cubeindex][i + 0]];
        t.v1 = vertlist[triTable[cubeindex][i + 1]];
        t.v2 = vertlist[triTable[cubeindex][i + 2]];

        appendTriangle(t);
    }
}
using Vertex = MarchingCubesData::Vertex;
using Triangle = MarchingCubesData::Triangle;

__device__
Vertex MarchingCubesData::vertexInterp(float isolevel, const float3 &p1, const float3 &p2, float d1, float d2,
                                       const uchar4 &c1, const uchar4 &c2) const {
    Vertex r1;
    r1.p = p1;
    r1.c = make_float3(c1.x, c1.y, c1.z) / 255.f;
    Vertex r2;
    r2.p = p2;
    r2.c = make_float3(c2.x, c2.y, c2.z) / 255.f;

    if (abs(isolevel - d1) < 0.00001f) return r1;
    if (abs(isolevel - d2) < 0.00001f) return r2;
    if (abs(d1 - d2) < 0.00001f) return r1;

    float mu = (isolevel - d1) / (d2 - d1);

    Vertex res;
    res.p.x = p1.x + mu * (p2.x - p1.x); // Positions
    res.p.y = p1.y + mu * (p2.y - p1.y);
    res.p.z = p1.z + mu * (p2.z - p1.z);

    res.c.x = (float) (c1.x + mu * (float) (c2.x - c1.x)) / 255.f; // Color
    res.c.y = (float) (c1.y + mu * (float) (c2.y - c1.y)) / 255.f;
    res.c.z = (float) (c1.z + mu * (float) (c2.z - c1.z)) / 255.f;

    return res;
}

__device__
bool MarchingCubesData::isInBoxAA(const float3 &minCorner, const float3 &maxCorner, const float3 &pos) const {
    if (pos.x < minCorner.x || pos.x > maxCorner.x) return false;
    if (pos.y < minCorner.y || pos.y > maxCorner.y) return false;
    if (pos.z < minCorner.z || pos.z > maxCorner.z) return false;

    return true;
}

__device__
uint MarchingCubesData::append() {
    uint addr = atomicAdd(d_numTriangles, 1);
    //TODO check
    return addr;
}

__device__
void MarchingCubesData::appendTriangle(const Triangle &t) {
    if (*d_numTriangles >= d_params->m_maxNumTriangles) {
        *d_numTriangles = d_params->m_maxNumTriangles;
        return; // todo
    }

    uint addr = append();

    if (addr >= d_params->m_maxNumTriangles) {
        printf("marching cubes exceeded max number of triangles (addr, #tri, max#tri): (%d, %d, %d)\n", addr,
               *d_numTriangles, d_params->m_maxNumTriangles);
        *d_numTriangles = d_params->m_maxNumTriangles;
        return; // todo
    }

    Triangle &triangle = d_triangles[addr];
    triangle.v0 = t.v0;
    triangle.v1 = t.v1;
    triangle.v2 = t.v2;
    return;
}

/** marching cube cuda*/
__global__ void resetMarchingCubesKernel(MarchingCubesData data) {
    *data.d_numTriangles = 0;
}

__global__ void extractIsoSurfaceKernel(HashDataStruct hashData, RayCastData rayCastData, MarchingCubesData data) {
    uint idx = blockIdx.x;

    const HashEntry &entry = hashData.d_hash[idx];
    if (entry.ptr != FREE_ENTRY) {
        int3 pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos);
        int3 pi = pi_base + make_int3(threadIdx);
        float3 worldPos = hashData.virtualVoxelPosToWorld(pi);

        data.extractIsoSurfaceAtPosition(worldPos, hashData, rayCastData);
    }
}

extern "C" void resetMarchingCubesCUDA(MarchingCubesData &data) {
    const dim3 blockSize(1, 1, 1);
    const dim3 gridSize(1, 1, 1);

    resetMarchingCubesKernel<<<gridSize, blockSize>>>(data);

#ifdef _DEBUG
    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg(__FUNCTION__);
#endif
}

extern "C" void
extractIsoSurfaceCUDA(const HashDataStruct &hashData, const RayCastData &rayCastData, const MarchingCubesParams &params,
                      MarchingCubesData &data) {
    const dim3 gridSize(params.m_hashNumBuckets * params.m_hashBucketSize, 1, 1);
    const dim3 blockSize(params.m_sdfBlockSize, params.m_sdfBlockSize, params.m_sdfBlockSize);

    extractIsoSurfaceKernel<<<gridSize, blockSize>>>(hashData, rayCastData, data);

#ifdef _DEBUG
    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg(__FUNCTION__);
#endif
}

void CUDAMarchingCubesHashSDF::create(const MarchingCubesParams &params) {
    m_params = params;
    m_data.allocate(m_params);

    resetMarchingCubesCUDA(m_data);
}


void CUDAMarchingCubesHashSDF::extractIsoSurface(const HashDataStruct &hashData, const HashParams &hashParams,
                                                 const RayCastData &rayCastData, const vec3f &minCorner,
                                                 const vec3f &maxCorner, bool boxEnabled) {
    resetMarchingCubesCUDA(m_data);

    m_params.m_maxCorner = maxCorner;
    m_params.m_minCorner = minCorner;
    m_params.m_boxEnabled = boxEnabled;
    m_data.updateParams(m_params);

    extractIsoSurfaceCUDA(hashData, rayCastData, m_params, m_data);
    copyTrianglesToCPU();
}


Mesh * CUDAMarchingCubesHashSDF::getMeshData() {
	return m_meshData;
}