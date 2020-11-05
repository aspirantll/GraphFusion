#pragma once
#include "cuda_matrix.h"
#include <Eigen/Core>
namespace MatrixConversion
{
	static mat3f toCUDA(const Eigen::Matrix3f& m) {
		return float3x3(m.data()).getTranspose();
	}
	static mat4f toCUDA(const Eigen::Matrix4f mat) {
		return float4x4(mat.data()).getTranspose();
	}

	static Eigen::Vector4f VecH(const Eigen::Vector3f& v)
	{
		return Eigen::Vector4f(v[0], v[1], v[2], 1.0);
	}

	static Eigen::Vector3f VecDH(const Eigen::Vector4f& v)
	{
		return Eigen::Vector3f(v[0] / v[3], v[1] / v[3], v[2] / v[3]);
	}
}