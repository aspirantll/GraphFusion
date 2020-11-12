#pragma once
#include "cuda_matrix.h"
#include <Eigen/Core>
namespace MatrixConversion
{
	static mat3f toCUDA(const Matrix3& m) {
		return float3x3(m.data()).getTranspose();
	}
	static mat4f toCUDA(const Matrix4 mat) {
		return float4x4(mat.data()).getTranspose();
	}
}