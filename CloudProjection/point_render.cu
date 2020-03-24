/*
@author:  Minye Wu
@contact: wuminye.x@gmail.com
*/


#include "point_render.cuh"
#include <stdio.h>

#include "helper_math.h"


struct Matrix4x4
{
public:
	float4 col[4];
	__device__ __forceinline__
		Matrix4x4()
	{
		col[0] = col[1] = col[2] = col[3] = make_float4(0, 0, 0, 0);
	}
	__device__ __forceinline__
		Matrix4x4(float3 a, float3 b, float3 c, float3 d)
	{
		col[0].x = a.x;
		col[0].y = a.y;
		col[0].z = a.z;
		col[0].w = 0;

		col[1].x = b.x;
		col[1].y = b.y;
		col[1].z = b.z;
		col[1].w = 0;

		col[2].x = c.x;
		col[2].y = c.y;
		col[2].z = c.z;
		col[2].w = 0;

		col[3].x = d.x;
		col[3].y = d.y;
		col[3].z = d.z;
		col[3].w = 1;
	}

	__device__ __forceinline__
		Matrix4x4 transpose() const
	{
		Matrix4x4 res;

		res.col[0].x = col[0].x;
		res.col[0].y = col[1].x;
		res.col[0].z = col[2].x;
		res.col[0].w = col[3].x;

		res.col[1].x = col[0].y;
		res.col[1].y = col[1].y;
		res.col[1].z = col[2].y;
		res.col[1].w = col[3].y;

		res.col[2].x = col[0].z;
		res.col[2].y = col[1].z;
		res.col[2].z = col[2].z;
		res.col[2].w = col[3].z;

		res.col[3].x = 0;
		res.col[3].y = 0;
		res.col[3].z = 0;
		res.col[3].w = 1;
		return res;

	}
	__device__ __forceinline__
		Matrix4x4 inv() const
	{
		Matrix4x4 res;
		res.col[0].x = col[0].x;
		res.col[0].y = col[1].x;
		res.col[0].z = col[2].x;
		res.col[0].w = 0;

		res.col[1].x = col[0].y;
		res.col[1].y = col[1].y;
		res.col[1].z = col[2].y;
		res.col[1].w = 0;

		res.col[2].x = col[0].z;
		res.col[2].y = col[1].z;
		res.col[2].z = col[2].z;
		res.col[2].w = 0;

		res.col[3].x = -dot(col[0], col[3]);
		res.col[3].y = -dot(col[1], col[3]);
		res.col[3].z = -dot(col[2], col[3]);
		res.col[3].w = 1;
		return res;
	}

	__device__ __forceinline__
		static	Matrix4x4 RotateX(float rad)
	{
		Matrix4x4 res;
		res.col[0].x = 1;
		res.col[0].y = 0;
		res.col[0].z = 0;
		res.col[0].w = 0;

		res.col[1].x = 0;
		res.col[1].y = cos(rad);
		res.col[1].z = sin(rad);
		res.col[1].w = 0;

		res.col[2].x = 0;
		res.col[2].y = -sin(rad);
		res.col[2].z = cos(rad);
		res.col[2].w = 0;

		res.col[3].x = 0;
		res.col[3].y = 0;
		res.col[3].z = 0;
		res.col[3].w = 1;
		return res;
	}
};



typedef struct CamPoseNode
{
	float3 norm, Xaxis, Yaxis, offset;
	__device__ __forceinline__
		Matrix4x4 getRT() const
	{
		return Matrix4x4(Xaxis, Yaxis, norm, offset);
	}

}CamPose;



typedef struct CamIntrinsic
{
	float3 r[3];

	__device__ __forceinline__
		Matrix4x4 getMatrix(float scale = 1.0) const
	{
		Matrix4x4 res;
		res.col[0].x = r[0].x * scale;
		res.col[0].y = r[1].x * scale;
		res.col[0].z = r[2].x * scale;
		res.col[0].w = 0;

		res.col[1].x = r[0].y * scale;
		res.col[1].y = r[1].y * scale;
		res.col[1].z = r[2].y * scale;
		res.col[1].w = 0;

		res.col[2].x = r[0].z * scale;
		res.col[2].y = r[1].z * scale;
		res.col[2].z = r[2].z;
		res.col[2].w = 0;

		res.col[3].x = 0;
		res.col[3].y = 0;
		res.col[3].z = 0;
		res.col[3].w = 1;
		return res;
	}
	__device__ __forceinline__
		float4 PointInverse(float x, float y, float scale = 1.0)
	{
		float xx = (x - r[0].z * scale) / (r[0].x * scale);
		float yy = (y - r[1].z * scale) / (r[1].y * scale);
		return make_float4(xx, yy, 1, 1);
	}

};


namespace math
{
	__device__ __forceinline__
	float4 MatrixMul(const Matrix4x4& mat, float4& x)
	{
		Matrix4x4 res = mat.transpose();
		float4 ans;
		ans.x = dot(res.col[0], x);
		ans.y = dot(res.col[1], x);
		ans.z = dot(res.col[2], x);
		ans.w = dot(res.col[3], x);

		ans = ans / ans.w;
		return ans;
	}
}


__global__
void DepthProject(float3 * point_clouds, int num_points,
	CamIntrinsic* tar_intrinsic, CamPose* tar_Pose, int tar_width, int tar_height,
	int * mutex_map, float near, float far, float max_splatting_size,
	float* out_depth, int* out_index)
{
	int ids = blockDim.x * blockIdx.x + threadIdx.x; //  index of point


	if (ids > num_points) 
		return;


	// Cache camera parameters
	 CamPose _tarcamPose = *tar_Pose;
	 CamIntrinsic _tarcamIntrinsic = *tar_intrinsic;


	float4 p = make_float4(point_clouds[ids], 1.0);

	Matrix4x4 camT = _tarcamPose.getRT();
	camT = camT.inv();
	float4 camp = math::MatrixMul(camT, p);



	float tdepth = camp.z;

	if (tdepth < 0)
		return;
	camp = math::MatrixMul(_tarcamIntrinsic.getMatrix(), camp);

	camp = camp / camp.w;
	camp = camp / camp.z;



	// splatting radius

	float rate = (tdepth - near) / (far - near);
	rate = 1.0 - rate;
	rate = max(rate, 0.0);
	rate = min(rate, 1.0);
	

	float radius = max_splatting_size * rate;
	

	// splatting
	for (int xx = round(camp.x - radius ); xx <= round(camp.x + radius ); ++xx)
	{
		for (int yy = round(camp.y - radius ); yy <= round(camp.y + radius ); ++yy)
		{
			if (xx < 0 || xx >= tar_width || yy < 0 || yy >= tar_height)
				return;

			int ind = yy * tar_width + xx ;



			bool next = true;
			while(next)
			{
				__threadfence();
				int v = atomicCAS(&mutex_map[ind], 0, 1);
				if (v == 0)
				{
					if (out_depth[ind] > tdepth || out_depth[ind]==0)
					{
						out_depth[ind] = tdepth;
						out_index[ind] = ids + 1; // 0 denote empty
					}
					atomicExch(&mutex_map[ind], 0);
					next = false;
				}  
			}


		}
	}

}

void GPU_PCPR(
	torch::Tensor in_points, //(num_points,3)
	torch::Tensor tar_intrinsic, torch::Tensor tar_Pose, 
	float near, float far, float max_splatting_size,
	torch::Tensor out_depth, torch::Tensor out_index) // (tar_height ,tar_width)
{
	const auto num_points = in_points.size(0);

	dim3 dimBlock(256,1);
	dim3 dimGrid(num_points / dimBlock.x + 1, 1);

	int tar_height = out_depth.size(0);
	int tar_width = out_depth.size(1);

	int *mutex_map;
	cudaMalloc(&mutex_map, sizeof(int) * tar_width *tar_height);
	cudaMemset(mutex_map, 0, tar_width * tar_height * sizeof(int));


	DepthProject << <dimGrid, dimBlock >> > (
		(float3*)in_points.data<float>(), num_points,
		(CamIntrinsic*)tar_intrinsic.data<float>(),(CamPose*)tar_Pose.data<float>(), tar_width, tar_height,
		mutex_map, near, far, max_splatting_size,
		out_depth.data<float>(), out_index.data<int>() );

	cudaFree(mutex_map);
}



__global__
void RGB_Project(
	CamIntrinsic* tar_intrinsic, CamPose* tar_Pose, int tar_width, int tar_height,
	CamIntrinsic* frame_intrinsic, CamPose* frame_Pose, int * width_height, int cam_num, 
	uchar3 * src_imgs, int src_width, int src_height,
	float* in_depth, uchar3 * out_rgb )
{
	int x = blockDim.x * blockIdx.x + threadIdx.x; // width
	int y = blockDim.y * blockIdx.y + threadIdx.y; // height

	if ((y ) >= tar_height || (x ) >= tar_width)
		return;

	

	// Cache camera parameters
	CamPose _tarcamPose = *tar_Pose;
	CamIntrinsic _tarcamIntrinsic = *tar_intrinsic;



	// shared cameraparameters 


	CamPose *  _frame_camPose = frame_Pose;
	CamIntrinsic * _frame_Intrinsic = frame_intrinsic;

/*
	// Cache cameraparameters
	if (threadIdx.x < cam_num && threadIdx.y == 0) {
		_frame_camPose[threadIdx.x] = *(tar_Pose + threadIdx.x);
		_frame_Intrinsic[threadIdx.x] = *(tar_intrinsic + threadIdx.x);
	}
	__syncthreads();
	*/
	//-------------------------------------------------------------------

	int ind_stride_1 = y*tar_width + x;
	int ind_stride_3 = ind_stride_1 * 3;

	
	if (in_depth[ind_stride_1]>0)
	{
	


		float4 p = _tarcamIntrinsic.PointInverse(x, y);
		p = in_depth[ind_stride_1] * p;
		p.w = 1;
		p = math::MatrixMul(tar_Pose->getRT(), p);

		p = p/p.w;

		float3 p_offset = make_float3(p.x,p.y,p.z);

		float3 dir = normalize(p_offset - tar_Pose->offset);

		float max_cos = 0;

		for (int i = 0; i < cam_num; ++i)
		{
			Matrix4x4 camT = _frame_camPose[i].getRT();
			camT = camT.inv();
			float4 camp = math::MatrixMul(camT, p);
			unsigned short tdepth = camp.z;
			camp = math::MatrixMul(_frame_Intrinsic[i].getMatrix(), camp);

			camp = camp / camp.w;
			camp = camp / camp.z;

			camp.x = round(camp.x);
			camp.y = round(camp.y);

			if (camp.x < 0 || camp.x >= width_height[i*2] || camp.y < 0 || camp.y >= width_height[i*2+1])
				continue;

			int xx = round(camp.x);
			int yy = round(camp.y);

			int offset = src_width*src_height * i;

			float3 cam_dir = normalize(p_offset - _frame_camPose[i].offset);

			float ccos = dot(dir, cam_dir);

			if (ccos > max_cos)
			{
				max_cos = ccos;
				out_rgb[ind_stride_1] = src_imgs[yy*src_width + xx + offset];
			}
		}
		

	}
	
	//out_rgb[y*tar_width + x] = make_uchar3(255,0,0);
	

}

void GPU_PCPR_RGB(
	torch::Tensor tar_intrinsic, torch::Tensor tar_Pose, 
	torch::Tensor frame_intrinsic, torch::Tensor frame_Pose, torch::Tensor width_height, int cam_num,
	torch::Tensor src_imgs,
	torch::Tensor in_depth, torch::Tensor out_rgb) 
{

	




	int tar_height = in_depth.size(0);
	int tar_width = in_depth.size(1);


	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid(tar_width  / dimBlock.x + 1, tar_height  / dimBlock.y + 1, 1);


	cudaMemset(out_rgb.data<unsigned char>(), 0, tar_width * tar_height * 3);


	RGB_Project << <dimGrid, dimBlock, 0 >> > (
		(CamIntrinsic*)tar_intrinsic.data<float>(),(CamPose*)tar_Pose.data<float>(), tar_width, tar_height,
		(CamIntrinsic*)frame_intrinsic.data<float>(),(CamPose*)frame_Pose.data<float>(), (int*)width_height.data<int>(),cam_num, 
		(uchar3*)src_imgs.data<unsigned char>(), src_imgs.size(2), src_imgs.size(1),
		in_depth.data<float>(), (uchar3*)out_rgb.data<unsigned char>() );


}




__global__
void PCPR_backward(float* grad_feature_image, int* index, int* num_points,
	float* out_grad_feature_points, float* out_grad_default_feature,
	int feature_dim, int num_batch, int width, int height, int total_sum)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x; // width
	int y = blockDim.y * blockIdx.y + threadIdx.y; // height


	if (y >= height || x >= width)
		return;

	__shared__ int _num_points[16];


	if (threadIdx.x < num_batch && threadIdx.y == 0) {
		_num_points[threadIdx.x] = *(num_points + threadIdx.x);
	}
	__syncthreads();


	int beg = 0;
	for (int i = 0; i < num_batch; ++i)
	{
		float* grad_feature_subimage = grad_feature_image + feature_dim * width * height * i
			+ y * width + x;

		int subindex = index[width * height * i + y * width + x];


		int num_points_sub = _num_points[i];

		int point_index = beg + subindex;

		float* out_grad_feature_points_sub = out_grad_feature_points + point_index;

		if (subindex == _num_points[i])
		{ // default feature
			for (int j = 0; j < feature_dim; ++j)
			{
				atomicAdd(out_grad_default_feature + j, grad_feature_subimage[j * width * height]);
			}
		}
		else
		{ // accumulate point gradient
			for (int j = 0; j < feature_dim; ++j)
			{
				atomicAdd(out_grad_feature_points_sub + j * total_sum, grad_feature_subimage[j * width * height]);
			}
		}

		beg += _num_points[i];
	}

}



void GPU_PCPR_backward(
    torch::Tensor grad_feature_image, //(batch, dim, height, width)
    torch::Tensor index,        //(batch, height, width)
    torch::Tensor num_points,     // (batch)
    torch::Tensor out_grad_feature_points, // (dim, total points)
	torch::Tensor out_grad_default_feature, // (dim, 1)
	int total_num
    )
{
	int num_batch = num_points.size(0);
	int feature_dim = out_grad_feature_points.size(0);

	cudaMemset(out_grad_feature_points.data<float>(), 0, sizeof(float)*feature_dim*out_grad_feature_points.size(1));
	cudaMemset(out_grad_default_feature.data<float>(), 0, sizeof(float)*feature_dim*out_grad_default_feature.size(1));

	int height = index.size(1);
	int width = index.size(2);

	dim3 dimBlock(32,32,1);
	dim3 dimGrid(height / dimBlock.x + 1, width / dimBlock.y + 1,1);


	PCPR_backward<< <dimGrid, dimBlock >> >(grad_feature_image.data<float>(), index.data<int>(), num_points.data<int>(),
		  out_grad_feature_points.data<float>(), out_grad_default_feature.data<float>(),
		  feature_dim, num_batch, width, height, total_num);


}


__global__
void RGB_INDEX_Project(
	CamIntrinsic* tar_intrinsic, CamPose* tar_Pose, int tar_width, int tar_height,
	CamIntrinsic* frame_intrinsic, CamPose* frame_Pose, int * width_height, int cam_num, 
	int* black_list, int black_list_num,
	float* in_depth, short3 * out_rgb )
{
	int x = blockDim.x * blockIdx.x + threadIdx.x; // width
	int y = blockDim.y * blockIdx.y + threadIdx.y; // height

	if ((y ) >= tar_height || (x ) >= tar_width)
		return;

	

	// Cache camera parameters
	CamPose _tarcamPose = *tar_Pose;
	CamIntrinsic _tarcamIntrinsic = *tar_intrinsic;



	// shared cameraparameters 


	CamPose *  _frame_camPose = frame_Pose;
	CamIntrinsic * _frame_Intrinsic = frame_intrinsic;

/*
	// Cache cameraparameters
	if (threadIdx.x < cam_num && threadIdx.y == 0) {
		_frame_camPose[threadIdx.x] = *(tar_Pose + threadIdx.x);
		_frame_Intrinsic[threadIdx.x] = *(tar_intrinsic + threadIdx.x);
	}
	__syncthreads();
	*/
	//-------------------------------------------------------------------

	int ind_stride_1 = y*tar_width + x;
	int ind_stride_3 = ind_stride_1 * 3;

	
	if (in_depth[ind_stride_1]>0)
	{
	


		float4 p = _tarcamIntrinsic.PointInverse(x, y);
		p = in_depth[ind_stride_1] * p;
		p.w = 1;
		p = math::MatrixMul(tar_Pose->getRT(), p);

		p = p/p.w;

		float3 p_offset = make_float3(p.x,p.y,p.z);

		float3 dir = normalize(p_offset - tar_Pose->offset);

		float max_cos = 0;

		for (int i = 0; i < cam_num; ++i)
		{
			bool flag = false;
			for (int j =0;j<black_list_num;++j)
			{
				if (i==black_list[j])
				flag = true;
				continue;
			}
			if (flag) continue;

			Matrix4x4 camT = _frame_camPose[i].getRT();
			camT = camT.inv();
			float4 camp = math::MatrixMul(camT, p);
			unsigned short tdepth = camp.z;
			camp = math::MatrixMul(_frame_Intrinsic[i].getMatrix(), camp);

			camp = camp / camp.w;
			camp = camp / camp.z;

			camp.x = round(camp.x);
			camp.y = round(camp.y);

			if (camp.x < 0 || camp.x >= width_height[i*2] || camp.y < 0 || camp.y >= width_height[i*2+1])
				continue;

			int xx = round(camp.x);
			int yy = round(camp.y);

			int offset = tar_width*tar_height * i;

			float3 cam_dir = normalize(p_offset - _frame_camPose[i].offset);

			float ccos = dot(dir, cam_dir);

			if (ccos > max_cos)
			{
				max_cos = ccos;
				out_rgb[ind_stride_1] = make_short3(i+1,xx,yy);
			}
		}
		

	}
	
	//out_rgb[y*tar_width + x] = make_uchar3(255,0,0);
	

}

void GPU_PCPR_RGB_INDEX(
	torch::Tensor tar_intrinsic, torch::Tensor tar_Pose, 
	torch::Tensor frame_intrinsic, torch::Tensor frame_Pose, torch::Tensor width_height, int cam_num,
	torch::Tensor black_list,
	torch::Tensor in_depth, torch::Tensor out_rgb) 
{
	int tar_height = in_depth.size(0);
	int tar_width = in_depth.size(1);


	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid(tar_width  / dimBlock.x + 1, tar_height  / dimBlock.y + 1, 1);


	cudaMemset(out_rgb.data<short>(), 0, tar_width * tar_height * sizeof(short) * 3);


	RGB_INDEX_Project << <dimGrid, dimBlock, 0 >> > (
		(CamIntrinsic*)tar_intrinsic.data<float>(),(CamPose*)tar_Pose.data<float>(), tar_width, tar_height,
		(CamIntrinsic*)frame_intrinsic.data<float>(),(CamPose*)frame_Pose.data<float>(), (int*)width_height.data<int>(),cam_num,
		(int*)black_list.data<int>(), black_list.size(0),
		in_depth.data<float>(), (short3*)out_rgb.data<short>() );


}