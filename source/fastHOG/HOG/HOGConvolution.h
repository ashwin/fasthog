#ifndef __HOG_CONVOLUTION__
#define __HOG_CONVOLUTION__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include <cuda_gl_interop.h>
#include <cuda.h>

#include "HOGDefines.h"

__host__ void InitConvolution(int width, int height, bool useGrayscale);
__host__ void SetConvolutionSize(int width, int height);
__host__ void CloseConvolution();

__host__ void ComputeColorGradients1to2(float1* inputImage, float2* outputImage);
__host__ void ComputeColorGradients4to2(float4* inputImage, float2* outputImage);

__global__ void convolutionRowGPU1(float1 *d_Result, float1 *d_Data, int dataW, int dataH);
__global__ void convolutionRowGPU4(float4 *d_Result, float4 *d_Data, int dataW, int dataH);

__global__ void convolutionColumnGPU1to2 ( float1 *d_Result, float1 *d_Data, float1 *d_DataRow, int dataW, int dataH, int smemStride, int gmemStride);
__global__ void convolutionColumnGPU4to2 ( float2 *d_Result, float4 *d_Data, float4 *d_DataRow, int dataW, int dataH, int smemStride, int gmemStride);

#endif
