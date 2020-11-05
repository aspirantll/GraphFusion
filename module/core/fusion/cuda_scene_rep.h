/**
 * this file is based on https://github.com/niessner/BundleFusion.git
 */

#ifndef CUDA_SCENE_REP_H
#define CUDA_SCENE_REP_H

#include "../solver/cuda_frame.h"
#include "../solver/matrix_conversion.h"
#include <pcl/PolygonMesh.h>

#define HANDLE_COLLISIONS
#define SDF_BLOCK_SIZE 8
#define HASH_BUCKET_SIZE 4
#define NUM_GROUPS_X 1024

#ifndef MINF
#define MINF __int_as_float(0xff800000)
#endif

#ifndef PINF
#define PINF __int_as_float(0x7f800000)
#endif


//status flags for hash entries
static const int LOCK_ENTRY = -1;
static const int FREE_ENTRY = -2;
static const int NO_OFFSET = 0;


__align__(16)    //has to be aligned to 16 bytes
typedef struct HashParams {
	HashParams() {
	}

	float4x4 m_rigidTransform;
	float4x4 m_rigidTransformInverse;

	unsigned int m_hashNumBuckets;
	unsigned int m_hashBucketSize;
	unsigned int m_hashMaxCollisionLinkedListSize;
	unsigned int m_numSDFBlocks;

	int m_SDFBlockSize;
	float m_virtualVoxelSize;
	unsigned int m_numOccupiedBlocks;    //occupied blocks in the viewing frustum

	float m_maxIntegrationDistance;
	float m_truncScale;
	float m_truncation;
	unsigned int m_integrationWeightSample;
	unsigned int m_integrationWeightMax;

	float3 m_streamingVoxelExtents;
	int3 m_streamingGridDimensions;
	int3 m_streamingMinGridPos;

} HashParams;

static __constant__ HashParams c_hashParams;
extern "C" void updateConstantHashParams(const HashParams &params);


__align__(16)
typedef struct HashEntry {
	int3 pos;        //hash position (lower left corner of SDFBlock))
	int ptr;        //pointer into heap to SDFBlock
	uint offset;        //offset for collisions


	__device__ void operator=(const struct HashEntry &e) {
		((int*)this)[0] = ((const int*)&e)[0];
		((int*)this)[1] = ((const int*)&e)[1];
		((int*)this)[2] = ((const int*)&e)[2];
		((int*)this)[3] = ((const int*)&e)[3];
		((int*)this)[4] = ((const int*)&e)[4];
    }
} HashEntry;

//__align__(8)
typedef struct Voxel {
	float sdf;        //signed distance function
	float weight;        //accumulated sdf weight
	uchar4 color;        //color

	//unsigned short sdf;
	//unsigned short weight;
	//uchar4	color;

	__device__ void operator=(const struct Voxel &v) {
		((int *) this)[0] = ((const int *) &v)[0];
		((int *) this)[1] = ((const int *) &v)[1];
		((int *) this)[2] = ((const int *) &v)[2];

		//((long long*)this)[0] = ((const long long*)&v)[0];	//8 bytes

		//this needs align, which unfortunately is problematic as __align__(16) would require more memory...
		//((long long*)this)[0] = ((const long long*)&v)[0];	//8 bytes
		//((int*)this)[2] = ((const int*)&v)[2];				//4 bytes
	}

} Voxel;

typedef struct HashDataStruct {

	///////////////
	// Host part //
	///////////////

	__device__ __host__
	HashDataStruct();

	__host__
	void allocate(const HashParams &params, bool dataOnGPU = true);

	__host__
	void updateParams(const HashParams &params);

	__host__
	void free();

	__host__
	HashDataStruct copyToCPU() const;



	/////////////////
	// Device part //
	/////////////////
	__device__
	const HashParams &params() const;

	//! see teschner et al. (but with correct prime values)
	__device__
	uint computeHashPos(const int3 &virtualVoxelPos) const;
	//merges two voxels (v0 is the input voxel, v1 the currently stored voxel)
	__device__
	void combineVoxel(const Voxel &v0, const Voxel &v1, Voxel &out) const;

	__device__
	void combineVoxelDepthOnly(const Voxel &v0, const Voxel &v1, Voxel &out) const;


	//! returns the truncation of the SDF for a given distance value
	__device__
	float getTruncation(float z) const;


	__device__
	float3 worldToVirtualVoxelPosFloat(const float3 &pos) const;

	__device__
	int3 worldToVirtualVoxelPos(const float3 &pos) const;

	__device__
	int3 virtualVoxelPosToSDFBlock(int3 virtualVoxelPos) const;

	// Computes virtual voxel position of corner sample position
	__device__
	int3 SDFBlockToVirtualVoxelPos(const int3 &sdfBlock) const;

	__device__
	float3 virtualVoxelPosToWorld(const int3 &pos) const;

	__device__
	float3 SDFBlockToWorld(const int3 &sdfBlock) const;

	__device__
	int3 worldToSDFBlock(const float3 &worldPos) const;

	__device__
	bool isSDFBlockInCameraFrustumApprox(const int3 &sdfBlock, CUDAFrame &frame);

	//! computes the (local) virtual voxel pos of an index; idx in [0;511]
	__device__
	uint3 delinearizeVoxelIndex(uint idx) const;

	//! computes the linearized index of a local virtual voxel pos; pos in [0;7]^3
	__device__
	uint linearizeVoxelPos(const int3 &virtualVoxelPos) const;

	__device__
	int virtualVoxelPosToLocalSDFBlockIndex(const int3 &virtualVoxelPos) const;

	__device__
	int worldToLocalSDFBlockIndex(const float3 &world) const;


	//! returns the hash entry for a given worldPos; if there was no hash entry the returned entry will have a ptr with FREE_ENTRY set
	__device__
	HashEntry getHashEntry(const float3 &worldPos) const;


	__device__
	void deleteHashEntry(uint id);

	__device__
	void deleteHashEntry(HashEntry &hashEntry);

	__device__
	bool voxelExists(const float3 &worldPos) const;

	__device__
	void deleteVoxel(Voxel &v) const;

	__device__
	void deleteVoxel(uint id);

	__device__
	Voxel getVoxel(const float3 &worldPos) const;

	__device__
	Voxel getVoxel(const int3 &virtualVoxelPos) const;

	__device__
	void setVoxel(const int3 &virtualVoxelPos, Voxel &voxelInput) const;

	//! returns the hash entry for a given sdf block id; if there was no hash entry the returned entry will have a ptr with FREE_ENTRY set
	__device__
	HashEntry getHashEntryForSDFBlockPos(const int3 &sdfBlock) const;

	//for histogram (no collision traversal)
	__device__
	unsigned int getNumHashEntriesPerBucket(unsigned int bucketID);

	//for histogram (collisions traversal only)
	__device__
	unsigned int getNumHashLinkedList(unsigned int bucketID);


	__device__
	uint consumeHeap();

	__device__
	void appendHeap(uint ptr);

	//pos in SDF block coordinates
	__device__
	void allocBlock(const int3 &pos);
	//!inserts a hash entry without allocating any memory: used by streaming: TODO MATTHIAS check the atomics in this function
	__device__
	bool insertHashEntry(HashEntry entry);



	//! deletes a hash entry position for a given sdfBlock index (returns true uppon successful deletion; otherwise returns false)
	__device__
	bool deleteHashEntryElement(const int3 &sdfBlock);


	uint *d_heap;                        //heap that manages free memory
	uint *d_heapCounter;                //single element; used as an atomic counter (points to the next free block)
	int *d_hashDecision;                //
	int *d_hashDecisionPrefix;        //
	HashEntry *d_hash;                        //hash that stores pointers to sdf blocks
	HashEntry *d_hashCompactified;            //same as before except that only valid pointers are there
	int *d_hashCompactifiedCounter;    //atomic counter to add compactified entries atomically
	Voxel *d_SDFBlocks;                //sub-blocks that contain 8x8x8 voxels (linearized); are allocated by heap
	int *d_hashBucketMutex;            //binary flag per hash bucket; used for allocation to atomically lock a bucket

	bool m_bIsOnGPU;                    //the class be be used on both cpu and gpu
} HashDataStruct;


void resetCUDA(HashDataStruct& hashData, const HashParams& hashParams);
void resetHashBucketMutexCUDA(HashDataStruct& hashData, const HashParams& hashParams);
void allocCUDA(HashDataStruct& hashData, const HashParams& hashParams, const CUDAFrame& frame);
void fillDecisionArrayCUDA(HashDataStruct& hashData, const HashParams& hashParams, const CUDAFrame &frame);
void compactifyHashCUDA(HashDataStruct& hashData, const HashParams& hashParams);
unsigned int compactifyHashAllInOneCUDA(HashDataStruct& hashData, const HashParams& hashParams, const CUDAFrame &frame);
void integrateDepthMapCUDA(HashDataStruct& hashData, const HashParams& hashParams, const CUDAFrame& frame);
void deIntegrateDepthMapCUDA(HashDataStruct& hashData, const HashParams& hashParams, const CUDAFrame& frame);
void bindInputDepthColorTextures(const CUDAFrame& frame);

void starveVoxelsKernelCUDA(HashDataStruct& hashData, const HashParams& hashParams);
void garbageCollectIdentifyCUDA(HashDataStruct& hashData, const HashParams& hashParams);
void garbageCollectFreeCUDA(HashDataStruct& hashData, const HashParams& hashParams);

class CUDASceneRepHashSDF
{
public:

	CUDASceneRepHashSDF(const HashParams& params) {
		create(params);
	}

	~CUDASceneRepHashSDF() {
		destroy();
	}

	void bindDepthCameraTextures(const CUDAFrame & frame) {
		bindInputDepthColorTextures(frame);
	}

	void integrate(const mat4f & lastRigidTransform, const CUDAFrame& frame) {
		
		bindDepthCameraTextures(frame);

		setLastRigidTransform(lastRigidTransform);

		//allocate all hash blocks which are corresponding to depth map entries
		alloc(frame);

		//generate a linear hash array with only occupied entries
		compactifyHashEntries(frame);

		//volumetrically integrate the depth data into the depth SDFBlocks
		integrateDepthMap(frame);

		//garbageCollect();

		m_numIntegratedFrames++;
	}

	void deIntegrate(const mat4f & lastRigidTransform, const CUDAFrame& frame) {

		bindDepthCameraTextures(frame);


		setLastRigidTransform(lastRigidTransform);

		//generate a linear hash array with only occupied entries
		compactifyHashEntries(frame);

		//volumetrically integrate the depth data into the depth SDFBlocks
		deIntegrateDepthMap(frame);

		m_numIntegratedFrames--;
	}

	void garbageCollect() {
		//only perform if enabled by global app state
        if (m_hashParams.m_numOccupiedBlocks > 0) {
            garbageCollectIdentifyCUDA(m_hashData, m_hashParams);
            resetHashBucketMutexCUDA(m_hashData, m_hashParams);	//needed if linked lists are enabled -> for memeory deletion
            garbageCollectFreeCUDA(m_hashData, m_hashParams);
        }
	}

	void setLastRigidTransform(const mat4f& lastRigidTransform) {
		m_hashParams.m_rigidTransform = lastRigidTransform;
		m_hashParams.m_rigidTransformInverse = m_hashParams.m_rigidTransform.getInverse();

		//make the rigid transform available on the GPU
		m_hashData.updateParams(m_hashParams);
	}

	//! resets the hash to the initial state (i.e., clears all data)
	void reset() {
		m_numIntegratedFrames = 0;

		m_hashParams.m_rigidTransform.setIdentity();
		m_hashParams.m_rigidTransformInverse.setIdentity();
		m_hashParams.m_numOccupiedBlocks = 0;
		m_hashData.updateParams(m_hashParams);
		resetCUDA(m_hashData, m_hashParams);
	}


	HashDataStruct& getHashData() {
		return m_hashData;
	} 

	const HashParams& getHashParams() const {
		return m_hashParams;
	}


	//! debug only!
	unsigned int getHeapFreeCount() {
		unsigned int count;
		cutilSafeCall(cudaMemcpy(&count, m_hashData.d_heapCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		return count+1;	//there is one more free than the address suggests (0 would be also a valid address)
	}

	unsigned int getNumIntegratedFrames() const {
		return m_numIntegratedFrames;
	}

private:

	void create(const HashParams& params) {
		m_hashParams = params;
		m_hashData.allocate(m_hashParams);

		reset();
	}

	void destroy() {
		m_hashData.free();
	}

	void alloc(const CUDAFrame& frame) {
		 
		unsigned int prevFree = getHeapFreeCount();
		while (1) {
			resetHashBucketMutexCUDA(m_hashData, m_hashParams);
			allocCUDA(m_hashData, m_hashParams, frame);

			unsigned int currFree = getHeapFreeCount();

			if (prevFree != currFree) {
				prevFree = currFree;
			}
			else {
				break;
			}
		}

	}


	void compactifyHashEntries(const CUDAFrame& frame) {
		m_hashParams.m_numOccupiedBlocks = compactifyHashAllInOneCUDA(m_hashData, m_hashParams, frame);
		m_hashData.updateParams(m_hashParams);	//make sure numOccupiedBlocks is updated on the GPU

	}

	void integrateDepthMap(const CUDAFrame& frame) {
		integrateDepthMapCUDA(m_hashData, m_hashParams, frame);
	}

	void deIntegrateDepthMap(const CUDAFrame& frame) {
		deIntegrateDepthMapCUDA(m_hashData, m_hashParams, frame);
	}



	HashParams		m_hashParams;
	HashDataStruct		m_hashData;

	unsigned int	m_numIntegratedFrames;	//used for garbage collect
};

__align__(16)	//has to be aligned to 16 bytes
typedef struct RayCastParams {
    float4x4 m_viewMatrix;
    float4x4 m_viewMatrixInverse;
    float mx, my, fx, fy; //raycast intrinsics

    unsigned int m_width;
    unsigned int m_height;

    unsigned int m_numOccupiedSDFBlocks;
    unsigned int m_maxNumVertices;
    int m_splatMinimum;

    float m_minDepth;
    float m_maxDepth;
    float m_rayIncrement;
    float m_thresSampleDist;
    float m_thresDist;
    bool  m_useGradients;

    uint dummy0;
} RayCastParams;

typedef struct RayCastSample
{
	float sdf;
	float alpha;
	uint weight;
	//uint3 color;
} RayCastSample;

static __constant__ RayCastParams c_rayCastParams;
extern "C" void updateConstantRayCastParams(const RayCastParams& params);

typedef struct RayCastData {

    ///////////////
    // Host part //
    ///////////////

    __device__ __host__
    RayCastData();

    void allocate(const RayCastParams &params);

    __host__
    void free();

    /////////////////
    // Device part //
    /////////////////

    __device__
		const RayCastParams& params() const;

	void updateParams(const RayCastParams &params);

	__device__
	float frac(float val) const;
	__device__
	float3 frac(const float3& val) const;

	__device__
	bool trilinearInterpolationSimpleFastFast(const HashDataStruct& hash, const float3& pos, float& dist, uchar3& color) const;


	__device__
	float findIntersectionLinear(float tNear, float tFar, float dNear, float dFar) const;

	static const unsigned int nIterationsBisection = 3;

	// d0 near, d1 far
	__device__
		bool findIntersectionBisection(const HashDataStruct& hash, const float3& worldCamPos, const float3& worldDir, float d0, float r0, float d1, float r1, float& alpha, uchar3& color) const;


	__device__
	float3 gradientForPoint(const HashDataStruct& hash, const float3& pos) const;

	static __inline__ __device__
	float depthProjToCameraZ(float z)	{
		return z * (c_rayCastParams.m_maxDepth - c_rayCastParams.m_minDepth) + c_rayCastParams.m_minDepth;
	}
	static __inline__ __device__
	float3 depthToCamera(unsigned int ux, unsigned int uy, float depth)
	{
		const float x = ((float)ux-c_rayCastParams.mx) / c_rayCastParams.fx;
		const float y = ((float)uy-c_rayCastParams.my) / c_rayCastParams.fy;
		return make_float3(depth*x, depth*y, depth);
	}
	static __inline__ __device__
	float3 cameraToDepthProj(const float3& pos)	{
		float2 proj = make_float2(
			pos.x*c_rayCastParams.fx/pos.z + c_rayCastParams.mx,
			pos.y*c_rayCastParams.fy/pos.z + c_rayCastParams.my);

		float3 pImage = make_float3(proj.x, proj.y, pos.z);

		pImage.x = (2.0f*pImage.x - (c_rayCastParams.m_width- 1.0f))/(c_rayCastParams.m_width- 1.0f);
		//pImage.y = (2.0f*pImage.y - (c_rayCastParams.m_height-1.0f))/(c_rayCastParams.m_height-1.0f);
		pImage.y = ((c_rayCastParams.m_height-1.0f) - 2.0f*pImage.y)/(c_rayCastParams.m_height-1.0f);
		pImage.z = (pImage.z - c_rayCastParams.m_minDepth)/(c_rayCastParams.m_maxDepth - c_rayCastParams.m_minDepth);

		return pImage;
	}

	__device__ inline bool isInCameraFrustumApprox(const float4x4& viewMatrixInverse, const float3& pos) {
		float3 pCamera = viewMatrixInverse * pos;
		float3 pProj = cameraToDepthProj(pCamera);
		//pProj *= 1.5f;	//TODO THIS IS A HACK FIX IT :)
		pProj *= 0.95;
		return !(pProj.x < -1.0f || pProj.x > 1.0f || pProj.y < -1.0f || pProj.y > 1.0f || pProj.z < 0.0f || pProj.z > 1.0f);
	}

	__device__
	void traverseCoarseGridSimpleSampleAll(const HashDataStruct& hash, const float3& worldCamPos, const float3& worldDir, const float3& camDir, const int3& dTid, float minInterval, float maxInterval) const;

	float*  d_depth;
	float4* d_depth4;
	float4* d_normals;
	float4* d_colors;

	float4* d_vertexBuffer; // ray interval splatting triangles, mapped from directx (memory lives there)

	float* d_rayIntervalSplatMinArray;
	float* d_rayIntervalSplatMaxArray;
} RayCastData;

class CUDARayCastSDF
{
public:
    CUDARayCastSDF(const RayCastParams& params) {
        create(params);
    }

    ~CUDARayCastSDF(void) {
        destroy();
    }

    void render(const HashDataStruct& hashData, const HashParams& hashParams, const mat4f& lastRigidTransform);

    const RayCastData& getRayCastData(void) {
        return m_data;
    }
    const RayCastParams& getRayCastParams() const {
        return m_params;
    }

    mat4f getIntrinsicsInv() const { return m_rayCastIntrinsicsInverse; }
    mat4f getIntrinsics() const { return m_rayCastIntrinsics; }

    //! the actual raycast calls the gpu update
    void updateRayCastMinMax(float depthMin, float depthMax) {
        m_params.m_minDepth = depthMin;
        m_params.m_maxDepth = depthMax;
    }
    //! the actual raycast calls the gpu update
    void setRayCastIntrinsics(unsigned int width, unsigned int height, const mat4f& intrinsics, const mat4f& intrinsicsInverse) {
		m_params.m_width = width;
		m_params.m_height = height;
		m_params.fx = intrinsics(0, 0);
		m_params.fy = intrinsics(1, 1);
		m_params.mx = intrinsics(0, 2);
		m_params.my = intrinsics(1, 2);
		m_rayCastIntrinsics = intrinsics;
		m_rayCastIntrinsicsInverse = intrinsicsInverse;
    }

private:

    void create(const RayCastParams& params);
    void destroy(void);

	void rayIntervalSplatting(const HashDataStruct& hashData, const HashParams& hashParams, const mat4f& lastRigidTransform); // rasterize

	RayCastParams m_params;
    RayCastData m_data;
    mat4f m_rayCastIntrinsics;
    mat4f m_rayCastIntrinsicsInverse;
};


typedef struct MarchingCubesParams {
	bool m_boxEnabled;
	float3 m_minCorner;

	unsigned int m_maxNumTriangles;
	float3 m_maxCorner;

	unsigned int m_sdfBlockSize;
	unsigned int m_hashNumBuckets;
	unsigned int m_hashBucketSize;
	float m_threshMarchingCubes;
	float m_threshMarchingCubes2;
} MarchingCubesParams;



typedef struct MarchingCubesData {

	///////////////
	// Host part //
	///////////////

	struct Vertex
	{
		float3 p;
		float3 c;
	};

	struct Triangle
	{
		Vertex v0;
		Vertex v1;
		Vertex v2;
	};

	__device__ __host__
	MarchingCubesData();

	__host__
	void allocate(const MarchingCubesParams& params, bool dataOnGPU = true);

	__host__
	void updateParams(const MarchingCubesParams& params);

	__host__
	void free();

	__host__
	MarchingCubesData copyToCPU() const;

	/////////////////
	// Device part //
	/////////////////
	__device__
	void extractIsoSurfaceAtPosition(const float3& worldPos, const HashDataStruct& hashData, const RayCastData& rayCastData);

	__device__
	Vertex vertexInterp(float isolevel, const float3& p1, const float3& p2, float d1, float d2, const uchar4& c1, const uchar4& c2) const;

	__device__
	bool isInBoxAA(const float3& minCorner, const float3& maxCorner, const float3& pos) const;
	__device__
	uint append();

	__device__
	void appendTriangle(const Triangle& t);

	MarchingCubesParams*	d_params;

	uint*			d_numTriangles;
	Triangle*		d_triangles;

	bool			m_bIsOnGPU;				// the class be be used on both cpu and gpu
} MarchingCubesData;

using Mesh = pcl::PolygonMesh;
using Vertices = pcl::Vertices;

class CUDAMarchingCubesHashSDF
{
public:
	CUDAMarchingCubesHashSDF(const MarchingCubesParams& params) {
		create(params);
	}

	~CUDAMarchingCubesHashSDF(void) {
		destroy();
	}

	void clearMeshBuffer(void) {
		if(m_meshData) {
			delete m_meshData;
			m_meshData = nullptr;
		}
		cloud.clear();
	}

	//! copies the intermediate result of extract isoSurfaceCUDA to the CPU and merges it with meshData
	void copyTrianglesToCPU();
	void saveMesh(const std::string& filename);
	void extractIsoSurface(const HashDataStruct& hashData, const HashParams& hashParams, const RayCastData& rayCastData, const vec3f& minCorner = {0.0f, 0.0f, 0.0f}, const vec3f& maxCorner = {0.0f, 0.0f, 0.0f}, bool boxEnabled = false);
	Mesh * getMeshData();
private:

	void create(const MarchingCubesParams& params);
	void destroy(void);

	MarchingCubesParams m_params;
	MarchingCubesData	m_data;

	Mesh * m_meshData = nullptr;

	pcl::PointCloud<pcl::PointXYZRGB> cloud;
};

#endif