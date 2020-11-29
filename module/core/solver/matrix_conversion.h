#pragma once
#include "cuda_matrix.h"
#include <Eigen/Core>
namespace MatrixConversion
{
	static mat3f toCUDA(const Matrix3& m) {
		const Eigen::Matrix3f mf = m.cast<CudaScalar>();
		return float3x3(mf.data()).getTranspose();
	}
	static mat4f toCUDA(const Matrix4 mat) {
		const Eigen::Matrix4f mf = mat.cast<CudaScalar>();
		return float4x4(mf.data()).getTranspose();
	}
}