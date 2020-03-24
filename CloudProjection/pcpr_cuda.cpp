/*
@author:  Minye Wu
@contact: wuminye.x@gmail.com
*/
#include <torch/extension.h>
#include <vector>
#include "point_render.cuh"
#include <iostream>


// CUDA forward declarations

std::vector<torch::Tensor> pcpr_cuda_forward(
    torch::Tensor in_points, //(num_points,3)
    torch::Tensor tar_intrinsic, torch::Tensor tar_Pose,
    torch::Tensor out_depth, torch::Tensor out_index, // (tar_heigh ,tar_width)
    float near, float far, float max_splatting_size
    );

std::vector<torch::Tensor> pcpr_cuda_backward(
    torch::Tensor grad_feature_image, //(batch, dim, heigh, width)
    torch::Tensor index,        //(batch, heigh, width)
    torch::Tensor num_points,     // (batch) - GPU
    torch::Tensor out_grad_feature_points, // (dim, total points)
    torch::Tensor out_grad_default_feature, // (dim, 1)
    int total_num
    );

std::vector<torch::Tensor> pcpr_rgb_index_cuda_forward(
	    torch::Tensor tar_intrinsic, torch::Tensor tar_Pose, 
	    torch::Tensor frame_intrinsic, torch::Tensor frame_Pose, torch::Tensor width_height,
      torch::Tensor black_list,
	    torch::Tensor in_depth,  torch::Tensor out_rgb
    );


std::vector<torch::Tensor> pcpr_rgb_cuda_forward(
	    torch::Tensor tar_intrinsic, torch::Tensor tar_Pose, 
	    torch::Tensor frame_intrinsic, torch::Tensor frame_Pose, torch::Tensor width_height,
	    torch::Tensor src_imgs,
	    torch::Tensor in_depth,  torch::Tensor out_rgb
    );

// C++ interface

#define CHECK_CPU(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) AT_ASSERTM(x.type().scalarType()==torch::ScalarType::Float, #x " must be a float tensor")
#define CHECK_INT(x) AT_ASSERTM(x.type().scalarType()==torch::ScalarType::Int, #x " must be a Int tensor")
#define CHECK_SHORT(x) AT_ASSERTM(x.type().scalarType()==torch::ScalarType::Short, #x " must be a Int tensor")
#define CHECK_UCHAR(x) AT_ASSERTM(x.type().scalarType()==torch::ScalarType::Byte, #x " must be a Int tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

std::vector<torch::Tensor> pcpr_cuda_forward(
    torch::Tensor in_points, //(num_points,3)
    torch::Tensor tar_intrinsic, torch::Tensor tar_Pose,
    torch::Tensor out_depth, torch::Tensor out_index, // (tar_heigh ,tar_width)
    float near, float far, float max_splatting_size
    ) 
{
  CHECK_INPUT(in_points); CHECK_FLOAT(in_points);
  CHECK_INPUT(tar_intrinsic); CHECK_FLOAT(tar_intrinsic);
  CHECK_INPUT(tar_Pose); CHECK_FLOAT(tar_Pose);
  CHECK_INPUT(out_depth); CHECK_FLOAT(out_depth);
  CHECK_INPUT(out_index); CHECK_INT(out_index);

  AT_ASSERTM(out_depth.size(0)== out_index.size(0), "out_depth and out_index must be the same size");
  AT_ASSERTM(out_depth.size(1)== out_index.size(1), "out_depth and out_index must be the same size");

  GPU_PCPR(
	in_points, //(num_points,3)
	tar_intrinsic, tar_Pose, 
	near, far, max_splatting_size,
	out_depth, out_index);
 

  return {out_depth, out_index};
}

std::vector<torch::Tensor> pcpr_rgb_cuda_forward(
	    torch::Tensor tar_intrinsic, torch::Tensor tar_Pose, 
	    torch::Tensor frame_intrinsic, torch::Tensor frame_Pose, torch::Tensor width_height,
	    torch::Tensor src_imgs,
	    torch::Tensor in_depth,  torch::Tensor out_rgb
    ) 
{
  CHECK_INPUT(tar_intrinsic); CHECK_FLOAT(tar_intrinsic);
  CHECK_INPUT(tar_Pose); CHECK_FLOAT(tar_Pose);
  CHECK_INPUT(in_depth); CHECK_FLOAT(in_depth);

  CHECK_INPUT(frame_intrinsic); CHECK_FLOAT(frame_intrinsic);
  CHECK_INPUT(frame_Pose); CHECK_FLOAT(frame_Pose);
  CHECK_INPUT(width_height); CHECK_INT(width_height);

  CHECK_INPUT(src_imgs); CHECK_UCHAR(src_imgs);
  CHECK_INPUT(out_rgb); CHECK_UCHAR(out_rgb);

  AT_ASSERTM(in_depth.size(0)== out_rgb.size(0), "in_depth and out_rgb must be the same size");
  AT_ASSERTM(in_depth.size(1)== out_rgb.size(1), "in_depth and out_rgb must be the same size");
  AT_ASSERTM(src_imgs.size(3)== 3, "src_imgs must be (num, height, width, 3)");

  int cam_num = frame_intrinsic.size(0);

  GPU_PCPR_RGB(
	tar_intrinsic, tar_Pose, 
	frame_intrinsic, frame_Pose, width_height, cam_num,
  src_imgs,
	in_depth,  out_rgb);
 

  return {out_rgb};
}



std::vector<torch::Tensor> pcpr_rgb_index_cuda_forward(
	    torch::Tensor tar_intrinsic, torch::Tensor tar_Pose, 
	    torch::Tensor frame_intrinsic, torch::Tensor frame_Pose, torch::Tensor width_height,
      torch::Tensor black_list,
	    torch::Tensor in_depth,  torch::Tensor out_rgb
    ) 
{
  CHECK_INPUT(tar_intrinsic); CHECK_FLOAT(tar_intrinsic);
  CHECK_INPUT(tar_Pose); CHECK_FLOAT(tar_Pose);
  CHECK_INPUT(in_depth); CHECK_FLOAT(in_depth);

  CHECK_INPUT(frame_intrinsic); CHECK_FLOAT(frame_intrinsic);
  CHECK_INPUT(frame_Pose); CHECK_FLOAT(frame_Pose);
  CHECK_INPUT(width_height); CHECK_INT(width_height);
  CHECK_INPUT(black_list); CHECK_INT(black_list);

  CHECK_INPUT(out_rgb); CHECK_SHORT(out_rgb);

  AT_ASSERTM(in_depth.size(0)== out_rgb.size(0), "in_depth and out_rgb must be the same size");
  AT_ASSERTM(in_depth.size(1)== out_rgb.size(1), "in_depth and out_rgb must be the same size");

  int cam_num = frame_intrinsic.size(0);

  GPU_PCPR_RGB_INDEX(
	tar_intrinsic, tar_Pose, 
	frame_intrinsic, frame_Pose, width_height, cam_num,
  black_list,
	in_depth,  out_rgb);
 

  return {out_rgb};
}


std::vector<torch::Tensor> pcpr_cuda_backward(
    torch::Tensor grad_feature_image, //(batch, dim, heigh, width)
    torch::Tensor index,        //(batch, heigh, width)
    torch::Tensor num_points,     // (batch) - GPU
    torch::Tensor out_grad_feature_points, // (dim, total points)
    torch::Tensor out_grad_default_feature, // (dim, 1)
    int total_num
    )
{
  CHECK_INPUT(grad_feature_image); CHECK_FLOAT(grad_feature_image);
  CHECK_INPUT(index); CHECK_INT(index);
  CHECK_INPUT(num_points); CHECK_INT(num_points);
  CHECK_INPUT(out_grad_feature_points); CHECK_FLOAT(out_grad_feature_points);
  CHECK_INPUT(out_grad_default_feature); CHECK_FLOAT(out_grad_default_feature);

  AT_ASSERTM(grad_feature_image.size(0)== index.size(0), "grad_feature_image and index must be the same batch size");
  AT_ASSERTM(index.size(0)== num_points.size(0), "grad_feature_image and num_points must be the same batch size");


  GPU_PCPR_backward(
    grad_feature_image, //(batch, dim, heigh, width)
    index,        //(batch, heigh, width)
    num_points,     // (batch)
    out_grad_feature_points, // (dim, total points)
	  out_grad_default_feature, // (dim, 1)
    total_num
    );


  return {out_grad_feature_points, out_grad_default_feature};
}

torch::Tensor pcpr_img_index_render(torch::Tensor rgb_index, torch::Tensor src_imgs,
                  torch::Tensor out_rgb)
{
  CHECK_CPU(rgb_index); CHECK_CONTIGUOUS(rgb_index); CHECK_SHORT(rgb_index);
  CHECK_CPU(src_imgs); CHECK_CONTIGUOUS(src_imgs); CHECK_UCHAR(src_imgs);
  CHECK_CPU(out_rgb); CHECK_CONTIGUOUS(out_rgb); CHECK_UCHAR(out_rgb);


  AT_ASSERTM(src_imgs.size(1)== out_rgb.size(0), "src_imgs and out_rgb must be the same image size");
  AT_ASSERTM(src_imgs.size(2)== out_rgb.size(1), "src_imgs and out_rgb must be the same image size");
  AT_ASSERTM(src_imgs.size(3)== 3, "src_imgs must be (num, height, width, 3)");
  AT_ASSERTM(out_rgb.size(2)== 3, "out_rgb must be (height, width, 3)");
  AT_ASSERTM(rgb_index.size(2)== 3, "rgb_index must be (height, width, 3)");


  int width = out_rgb.size(1);
  int height = out_rgb.size(0);
  int num = src_imgs.size(0);

  int src_width = src_imgs.size(2);
  int src_height = src_imgs.size(1);


  short3 * _rgb_index = (short3*)rgb_index.data<short>();

  uchar3 * _src_imgs = (uchar3*)src_imgs.data<unsigned char>();
  uchar3 * _out_rgb = (uchar3*)out_rgb.data<unsigned char>();




  for (int y=0;y<height;++y)
  {
    for (int x=0;x<width;++x)
    {
      int ind_stride_1 = y*width + x;

      if (_rgb_index[ind_stride_1].x == 0)
        continue;

      _out_rgb[ind_stride_1] = _src_imgs[ _rgb_index[ind_stride_1].z*src_width + _rgb_index[ind_stride_1].y + src_height*src_width*(_rgb_index[ind_stride_1].x-1)];
    }
  }
  

  return out_rgb;
 
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &pcpr_cuda_forward, "PCPR forward (CUDA)");
  m.def("rgb_render", &pcpr_rgb_cuda_forward, "PCPR RGB forward (CUDA)");
  m.def("rgb_index_calc", &pcpr_rgb_index_cuda_forward, "PCPR RGB INDEX forward (CUDA)");
  m.def("rgb_index_render", &pcpr_img_index_render, "RGB Render forward (CPU)");
  m.def("backward", &pcpr_cuda_backward, "PCPR backward (CUDA)");
}

