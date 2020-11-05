/**
 * this file is based on https://github.com/niessner/BundleFusion.git
 */

#include "cuda_scene_rep.h"
#include <pcl/Vertices.h>
#include <pcl/conversions.h>


/** raycast */
__host__
void RayCastData::allocate(const RayCastParams &params) {
    cutilSafeCall(cudaMalloc(&d_depth, sizeof(float) * params.m_width * params.m_height));
    cutilSafeCall(cudaMalloc(&d_depth4, sizeof(float4) * params.m_width * params.m_height));
    cutilSafeCall(cudaMalloc(&d_normals, sizeof(float4) * params.m_width * params.m_height));
    cutilSafeCall(cudaMalloc(&d_colors, sizeof(float4) * params.m_width * params.m_height));

    cutilSafeCall(cudaMalloc(&d_rayIntervalSplatMinArray,  sizeof(float) * params.m_width * params.m_height));
    cutilSafeCall(cudaMalloc(&d_rayIntervalSplatMaxArray, sizeof(float) * params.m_width * params.m_height));

}

__host__
void RayCastData::free() {
    cutilSafeCall(cudaFree(d_depth));
    cutilSafeCall(cudaFree(d_depth4));
    cutilSafeCall(cudaFree(d_normals));
    cutilSafeCall(cudaFree(d_colors));
    cutilSafeCall(cudaFree(d_rayIntervalSplatMinArray));
    cutilSafeCall(cudaFree(d_rayIntervalSplatMaxArray));
}

extern "C" void renderCS(const HashDataStruct& hashData, const RayCastData &rayCastData, const RayCastParams &rayCastParams);

extern "C" void computeNormals(float4* d_output, float4* d_input, unsigned int width, unsigned int height);

extern "C" void resetRayIntervalSplatCUDA(RayCastData& data, const RayCastParams& params);
extern "C" void rayIntervalSplatCUDA(const HashDataStruct& hashData, const RayCastData &rayCastData, const RayCastParams &rayCastParams);

void CUDARayCastSDF::create(const RayCastParams& params)
{
    m_params = params;
    m_data.allocate(m_params);

    m_rayCastIntrinsics = mat4f(
            params.fx, 0.0f, params.mx, 0.0f,
            0.0f, params.fy, params.my, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f);
    m_rayCastIntrinsicsInverse = m_rayCastIntrinsics.getInverse();
}

void CUDARayCastSDF::destroy(void)
{
    m_data.free();
}

void CUDARayCastSDF::render(const HashDataStruct& hashData, const HashParams& hashParams, const mat4f& lastRigidTransform)
{
    cutilSafeCall(cudaMemset(m_data.d_rayIntervalSplatMinArray, 0, sizeof(float) * m_params.m_width*m_params.m_height));
    cutilSafeCall(cudaMemset(m_data.d_rayIntervalSplatMaxArray, 0, sizeof(float) * m_params.m_width*m_params.m_height));

    rayIntervalSplatting(hashData, hashParams, lastRigidTransform);

    renderCS(hashData, m_data, m_params);

    //convertToCameraSpace(cameraData);
    if (!m_params.m_useGradients)
    {
        computeNormals(m_data.d_normals, m_data.d_depth4, m_params.m_width, m_params.m_height);
    }

}

void CUDARayCastSDF::rayIntervalSplatting(const HashDataStruct& hashData, const HashParams& hashParams, const mat4f& lastRigidTransform)
{
    if (hashParams.m_numOccupiedBlocks == 0)	return;

    if (m_params.m_maxNumVertices <= 6*hashParams.m_numOccupiedBlocks) { // 6 verts (2 triangles) per block
        throw "not enough space for vertex buffer for ray interval splatting";
    }

    m_params.m_numOccupiedSDFBlocks = hashParams.m_numOccupiedBlocks;
    m_params.m_viewMatrix = lastRigidTransform.getInverse();
    m_params.m_viewMatrixInverse = lastRigidTransform;

    //m_data.updateParams(m_params); // !!! debugging
    // splat minimum
    m_params.m_splatMinimum = 1;
    m_data.updateParams(m_params);
    rayIntervalSplatCUDA(hashData, m_data, m_params);

    // splat maximum
    m_params.m_splatMinimum = 0;
    m_data.updateParams(m_params);
    rayIntervalSplatCUDA(hashData, m_data, m_params);

}


__host__
void MarchingCubesData::allocate(const MarchingCubesParams &params, bool dataOnGPU) {
    m_bIsOnGPU = dataOnGPU;
    if (m_bIsOnGPU) {
        cutilSafeCall(cudaMalloc(&d_params, sizeof(MarchingCubesParams)));
        cutilSafeCall(cudaMalloc(&d_triangles, sizeof(Triangle) * params.m_maxNumTriangles));
        cutilSafeCall(cudaMalloc(&d_numTriangles, sizeof(uint)));
    } else {
        d_params = new MarchingCubesParams;
        d_triangles = new Triangle[params.m_maxNumTriangles];
        d_numTriangles = new uint;
    }
}

__host__
void MarchingCubesData::updateParams(const MarchingCubesParams &params) {
    if (m_bIsOnGPU) {
        cutilSafeCall(cudaMemcpy(d_params, &params, sizeof(MarchingCubesParams), cudaMemcpyHostToDevice));
    } else {
        *d_params = params;
    }
}

__host__
void MarchingCubesData::free() {
    if (m_bIsOnGPU) {
        cutilSafeCall(cudaFree(d_params));
        cutilSafeCall(cudaFree(d_triangles));
        cutilSafeCall(cudaFree(d_numTriangles));
    } else {
        if (d_params) delete d_params;
        if (d_triangles) delete[] d_triangles;
        if (d_numTriangles) delete d_numTriangles;
    }

    d_params = NULL;
    d_triangles = NULL;
    d_numTriangles = NULL;
}

__host__
MarchingCubesData MarchingCubesData::copyToCPU() const {
    MarchingCubesParams params;
    cutilSafeCall(cudaMemcpy(&params, d_params, sizeof(MarchingCubesParams), cudaMemcpyDeviceToHost));

    MarchingCubesData data;
    data.allocate(params, false);    // allocate the data on the CPU
    cutilSafeCall(cudaMemcpy(data.d_params, d_params, sizeof(MarchingCubesParams), cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(data.d_numTriangles, d_numTriangles, sizeof(uint), cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(data.d_triangles, d_triangles, sizeof(Triangle) * (params.m_maxNumTriangles),
                             cudaMemcpyDeviceToHost));
    return data;    //TODO MATTHIAS look at this (i.e,. when does memory get destroyed ; if it's in the destructor it would kill everything here
}

void CUDAMarchingCubesHashSDF::destroy(void) {
    m_data.free();
}

void CUDAMarchingCubesHashSDF::copyTrianglesToCPU() {

    MarchingCubesData cpuData = m_data.copyToCPU();

    unsigned int nTriangles = *cpuData.d_numTriangles;

    std::cout << "Marching Cubes: #triangles = " << nTriangles << std::endl;

    if (nTriangles != 0) {
        uint baseIndex = cloud.size();
        cloud.resize(baseIndex + 3 * nTriangles);

        vec3f *vc = (vec3f *) cpuData.d_triangles;
        for (unsigned int i = 0; i < 3 * nTriangles; i++) {
            cloud.points[baseIndex+i].x = vc[2 * i + 0].x;
            cloud.points[baseIndex+i].y = vc[2 * i + 0].y;
            cloud.points[baseIndex+i].z = vc[2 * i + 0].z;

            cloud.points[baseIndex+i].r = vc[2 * i + 1].x * 255;
            cloud.points[baseIndex+i].g = vc[2 * i + 1].y * 255;
            cloud.points[baseIndex+i].b = vc[2 * i + 1].z * 255;
        }
        if(m_meshData == nullptr) {
            m_meshData = new Mesh();
        }
        pcl::toPCLPointCloud2(cloud, m_meshData->cloud);

        for (unsigned int i = 0; i < nTriangles; i++) {
            pcl::Vertices v;
            v.vertices = {baseIndex + 3 * i + 0, baseIndex + 3 * i + 1, baseIndex + 3 * i + 2};
            m_meshData->polygons.emplace_back(v);
        }
    }
    cpuData.free();
}

void CUDAMarchingCubesHashSDF::saveMesh(const std::string &filename) {
    std::string actualFilename = filename;
    std::cout << "saving mesh (" << actualFilename << ") ...";
    rtf::PointUtil::savePLYMesh(actualFilename, *m_meshData);
    std::cout << "done!" << std::endl;

    clearMeshBuffer();
}


