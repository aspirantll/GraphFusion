//
// Created by liulei on 2020/7/14.
//

#ifndef GraphFusion_CUDA_TYPES_H
#define GraphFusion_CUDA_TYPES_H

#include <glog/logging.h>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_texture_types.h>

#include "base_types.h"

using namespace std;
extern thread_local cudaStream_t stream;

#define BLOCK_X 8
#define BLOCK_Y 8

#define CUDA_CHECKED_CALL(cuda_call) \
    do{                              \
        cudaError error = (cuda_call);                                  \
        if (cudaSuccess != error) {                                     \
          LOG(FATAL) << "Cuda Error: " << cudaGetErrorString(error);    \
        }                                                                \
    }while (false)


#define CUDA_CHECKED_NO_ERROR() \
        CUDA_CHECKED_CALL(cudaGetLastError())

#define CUDA_SYNCHRONIZE() \
        CUDA_CHECKED_CALL(cudaStreamSynchronize(stream))


#define CUDA_LINE_BLOCK(n) \
    long blockSize = n > BLOCK_X * BLOCK_Y? BLOCK_X * BLOCK_Y: n;\
    dim3 block(blockSize);\
    dim3 grid((n+blockSize-1)/blockSize)

#define CUDA_MAT_BLOCK(m, n) \
    long blockSize = BLOCK_X * BLOCK_Y;\
    dim3 block(BLOCK_X, BLOCK_Y);\
    dim3 grid((m+BLOCK_X-1)/BLOCK_X, (n+BLOCK_Y-1)/BLOCK_Y)

#define CUDA_IMG_BLOCK(width, height) \
    dim3 block(BLOCK_X, BLOCK_Y);\
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y)

namespace rtf {

    template <class DataT> class CUDAArray {
    public:
        DataT * data = nullptr;
        DataT * uploadData = nullptr;
        ulong size;
        bool uploaded = false;
        bool copied = false;


        CUDAArray(ulong size) {
            this->size = size;
            CUDA_CHECKED_CALL(cudaMalloc(&data, size*sizeof(DataT)));
            copied = false;
        }

        CUDAArray(const CUDAArray<DataT> &other):data(other.data), size(other.size), uploaded(other.uploaded) {
            copied = true;
        }

        CUDAArray(const vector<DataT> &other) {
            this->size = other.size();
            CUDA_CHECKED_CALL(cudaMalloc(&data, size*sizeof(DataT)));
            copied = false;
            upload(other.data(), 0, other.size());
        }


        CUDAArray<DataT> &operator=(const CUDAArray<DataT> &other) {
            this->data = other.data;
            this->size = other.size;
            this->uploaded = other.uploaded;
            this->copied = true;
            return *this;
        }

        void memset(DataT t) {
            CUDA_CHECKED_CALL(cudaMemset(data, t, size));
        }

        ~CUDAArray() {
            free();
        }

        void free() {
            if(!copied&&data) {
                cudaFree(data);
            }

            if(uploadData) {
                cudaFreeHost(uploadData);
            }
        }

        __host__ __device__ void mallocHost(const DataT * src, long uploadSize) {
            if(uploadData) {
                cudaFreeHost(uploadData);
            }
            CUDA_CHECKED_CALL(cudaMallocHost(&uploadData, sizeof(DataT) * uploadSize));
            memcpy(uploadData, src, sizeof(DataT) * uploadSize);
        }

        __host__ __device__ DataT operator[](int index) {
            if(!uploaded) {
                printf("there is no data!\n");
            }

            if(index >= size) {
                printf("index is not in ranges: %ld\n", index);
            }

            // regard col as prime order
            return data[index];
        }

        __host__ __device__ void set(long index, DataT e) {
            data[index] = e;
        }

        __host__ __device__ long getSizes() {
            return size;
        }


        /**
         *
         * @param src
         * @param offset double count
         * @param uploadSize double count
         */
        void upload(const DataT * src, long offset, long uploadSize) {
            LOG_ASSERT(offset+uploadSize<=size) << "the size of data array must be less than total size";
            mallocHost(src, uploadSize);
            CUDA_CHECKED_CALL(cudaMemcpyAsync(data+offset, uploadData, uploadSize * sizeof(DataT), cudaMemcpyHostToDevice, stream));
            uploaded = true;
        }

        /**
        *
        * @param src
        * @param offset double count
        * @param uploadSize double count
        */
        void copyDevice(DataT * src, long offset, long uploadSize) {
            LOG_ASSERT(offset+uploadSize<=size) << "the size of data array must be less than total size";
            CUDA_CHECKED_CALL(cudaMemcpyAsync(data+offset, src, uploadSize*sizeof(DataT), cudaMemcpyDeviceToDevice, stream));
            uploaded = true;
        }

        /**
         *
         * @param dst
         * @param offset double count
         * @param downloadSize double count
         */
        void download(DataT* dst, long offset=0, long downloadSize=-1) {
            if(downloadSize==-1) {
                downloadSize = size;
            }

            LOG_ASSERT(offset+downloadSize<=size) << "the size of data array must be less than total size";
            CUDA_SYNCHRONIZE();
            CUDA_CHECKED_CALL(cudaMemcpyAsync(dst, data+offset, downloadSize*sizeof(DataT), cudaMemcpyDeviceToHost, stream));
        }

        void download(VectorX & vector) {
            vector.resize(size);
            CUDAArray<DataT>::download(vector.data(), 0, size);
        }

        void download(vector<DataT> &other) {
            other.resize(size);
            CUDAArray<DataT>::download(other.data(), 0, size);
        }

    };

    typedef CUDAArray<Scalar> CUDAArrays;
    typedef CUDAArray<float> CUDAArrayf;
    typedef CUDAArray<unsigned int> CUDAArrayu;


    template <class DataT> class CUDAMatrix: public CUDAArray<DataT> {
    protected:
        long rows, cols;
        bool colMajor = true;

        CUDAMatrix(DataT *data, ulong size, long rows, long cols) {
            this->data = data;
            this->size = size;
            this->rows = rows;
            this->cols = cols;
            this->copied = true;
        }
    public:
        typedef Eigen::Matrix<DataT, Eigen::Dynamic, Eigen::Dynamic> EigenMatrix;
        typedef Eigen::Matrix<DataT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrixRowMajor;

        CUDAMatrix(long rows, long cols): rows(rows), cols(cols), CUDAArray<DataT>(rows*cols) {

        }

        CUDAMatrix(EigenMatrixRowMajor& mat): CUDAMatrix(mat.rows(), mat.cols()){
            upload(mat);
        }

        CUDAMatrix(EigenMatrix mat): CUDAMatrix(mat.rows(), mat.cols()) {
            upload(mat);
        }

        CUDAMatrix(vector<DataT> vec): CUDAMatrix(vec.size(), 1) {
            upload(vec);
        }

        CUDAMatrix(CUDAArray<DataT> array, long rows, long cols): rows(rows), cols(cols), CUDAArray<DataT>(array) {
            CHECK_EQ(rows*cols, array.size);
        }

         ~CUDAMatrix() {

        }

        __host__ __device__ long convert1DIndex(long i, long j) {
            return colMajor?j*rows+i:i*cols+j;
        }

        __host__ __device__ void convert2DIndex(long index, long *i, long *j) {
            if(colMajor) {
                *i = index%rows;
                *j = index/rows;
            }else {
                *i = index/cols;
                *j = index%cols;
            }
        }

        __host__ __device__ DataT operator()(int i, int j) {
            if(i<0||i>=rows||j<0||j>=cols) {
                printf("i and j is not in ranges, the rows is %d, the cols is %ld!\n", rows, cols);
            }

            return (*this)[convert1DIndex(i,j)];
        }

        __host__ __device__ void resize(long rows, long cols) {
            this->rows = rows;
            this->cols = cols;
        }

        __host__ __device__ long getRows() {
            return rows;
        }

        __host__ __device__ long getCols() {
            return cols;
        }

        __host__ __device__ bool isColMajor() {
            return colMajor;
        }

        using CUDAArray<DataT>::upload;

        void upload(EigenMatrixRowMajor& eigenMat) {
            rows = eigenMat.rows();
            cols = eigenMat.cols();
            colMajor = false;
            CUDAArray<DataT>::upload(eigenMat.data(), 0, rows*cols);
        }

        void upload(EigenMatrix eigenMat) {
            rows = eigenMat.rows();
            cols = eigenMat.cols();
            CUDAArray<DataT>::upload(eigenMat.data(), 0, rows*cols);
        }

        void upload(vector<DataT> vec) {
            rows = vec.size();
            cols = 1;
            CUDAArray<DataT>::upload(vec.data(), 0, rows*cols);
        }

        using CUDAArray<DataT>::download;

        void download(EigenMatrix& eigenMat) {
            eigenMat.resize(rows, cols);
            CUDAArray<DataT>::download(eigenMat.data(), 0, rows*cols);
        }
    };

    typedef CUDAMatrix<Scalar> CUDAMatrixs;
    typedef CUDAMatrix<float> CUDAMatrixf;
    typedef CUDAMatrix<long> CUDAMatrixl;
    typedef CUDAMatrix<uchar> CUDAMatrixc;


    template <class DataT> class CUDAPtr {
    public:
        DataT * data;
        long rows, cols;
        bool colMajor = true;

        __host__ __device__ CUDAPtr() {

        }

        __host__ __device__ CUDAPtr(CUDAMatrix<DataT> mat) {
            data = mat.data;
            rows = mat.getRows();
            cols = mat.getCols();
            colMajor = mat.isColMajor();
        }

        __host__ __device__ CUDAPtr(CUDAArray<DataT> array) {
            data = array.data;
            rows = array.getSizes();
            cols = 1;
        }

        __host__ __device__ CUDAPtr(DataT* data, long rows, long cols): data(data), rows(rows), cols(cols) {

        }

        __device__ long convert1DIndex(long i, long j) {
            return colMajor?j*rows+i:i*cols+j;
        }

        __device__ void convert2DIndex(long index, long *i, long *j) {
            if(colMajor) {
                *i = index%rows;
                *j = index/rows;
            }else {
                *i = index/cols;
                *j = index%cols;
            }
        }

        __device__ DataT operator()(long i, long j) {
            if(i<0||i>=rows||j<0||j>=cols) {
                printf("i and j is not in ranges, the rows is %d, the cols is %ld!\n", rows, cols);
            }

            return data[convert1DIndex(i,j)];
        }

        __device__ DataT operator[](long index) {
            if(index >= rows*cols) {
                printf("index is not in ranges: %ld\n", index);
            }

            // regard col as prime order
            return data[index];
        }

        __host__ __device__ void set(long row, long col, DataT e) {
            long index = col*rows+row;
            if(!colMajor)
                index = row*cols+col;
            data[index] = e;
        }

        __host__ __device__ void setIndex(long index, DataT e) {
            if(index >= rows*cols) {
                printf("index is not in ranges: %ld\n", index);
            }
            data[index] = e;
        }

        __host__ __device__ long getRows() {
            return rows;
        }

        __host__ __device__ long getCols() {
            return cols;
        }

        __device__ void free() {
            delete data;
        }

        __device__ long size() {
            return rows*cols;
        }
    };

    typedef CUDAPtr<Scalar> CUDAPtrs;
    typedef CUDAPtr<float> CUDAPtrf;
    typedef CUDAPtr<long> CUDAPtrl;
    typedef CUDAPtr<unsigned int> CUDAPtru;
    typedef CUDAPtr<uchar> CUDAPtrc;


    template <class DataT> class CUDARow {
    protected:
        CUDAPtr<DataT> mat;
        long row;
    public:
        __device__ CUDARow(const CUDAPtr<DataT> &mat, long row) : mat(mat), row(row) {}
        __device__ DataT operator[](long col) {
            return mat(row, col);
        }

        __device__ void setElement(long col, DataT e) {
            mat.set(row, col, e);
        }
    };

    template <class DataT> class CUDACol {
    protected:
        CUDAPtr<DataT> mat;
        long col;
    public:
        __device__ CUDACol(const CUDAPtr<DataT> &mat, long col) : mat(mat), col(col) {}
        __device__ DataT operator[](long row) {
            return mat(row, col);
        }

        __device__ void setElement(long row, DataT e) {
            mat.set(row, col, e);
        }
    };


    template <class T> class CUDAPtrArray {
    public:
        T * ptrs;
        long num;

        __host__ __device__ CUDAPtrArray(T* ptrs, long num) {
            this->ptrs = ptrs;
            this->num = num;
        }

        __host__ __device__ T operator[](long index) {
            if(index >= num) {
                printf("index is not in ranges: %ld\n", index);
            }

            // regard col as prime order
            return ptrs[index];
        }

        __host__ __device__ long getNum() {
            return num;
        }
    };

    template <class T> class CUDAVector {
    protected:
        vector<T> cudaPtrVec;
        T * ptrs = nullptr;
    public:

        void addItem(T t) {
            cudaPtrVec.emplace_back(t);
        }

        void removeItem(long index) {
            cudaPtrVec.erase(cudaPtrVec.begin()+index);
        }

        T& operator[] (long index) {
            return cudaPtrVec[index];
        }

        int getNum() {
            return cudaPtrVec.size();
        }

        CUDAPtrArray<T> uploadToCUDA() {
            if(!ptrs) CUDA_CHECKED_CALL(cudaFree(ptrs));
            long num = cudaPtrVec.size();
            CUDA_CHECKED_CALL(cudaMalloc(&ptrs, sizeof(T)*num));

            CUDA_CHECKED_CALL(cudaMemcpyAsync(ptrs, cudaPtrVec.data(), sizeof(T)*num, cudaMemcpyHostToDevice, stream));

            return {ptrs, num};
        }

        void clear() {
            cudaPtrVec.clear();
        }

        ~CUDAVector() {
            if(!ptrs) CUDA_CHECKED_CALL(cudaFree(ptrs));
        }
    };
}


#endif //GraphFusion_CUDA_TYPES_H
