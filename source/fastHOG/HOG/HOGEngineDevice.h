#ifndef __CUDA_HOG__
#define __CUDA_HOG__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include <cuda_gl_interop.h>
#include <cutil_inline.h>
#include <cuda.h>

#include "HOGDefines.h"

extern "C" __host__ void InitHOG(int width, int height,
								 int avSizeX, int avSizeY,
								 int marginX, int marginY,
								 int cellSizeX, int cellSizeY,
								 int blockSizeX, int blockSizeY,
								 int windowSizeX, int windowSizeY,
								 int noOfHistogramBins, float wtscale,
								 float svmBias, float* svmWeights, int svmWeightsCount,
								 bool useGrayscale);

extern "C" __host__ void CloseHOG();

extern "C" __host__ void BeginHOGProcessing(unsigned char* hostImage, int minx, int miny, int maxx, int maxy, float minScale, float maxScale);
extern "C" __host__ float* EndHOGProcessing();

extern "C"  __host__ void GetHOGParameters(float *cStartScale, float *cEndScale, float *cScaleRatio, int *cScaleCount,
										   int *cPaddingSizeX, int *cPaddingSizeY, int *cPaddedWidth, int *cPaddedHeight,
										   int *cNoOfCellsX, int *cNoOfCellsY, int *cNoOfBlocksX, int *cNoOfBlocksY,
										   int *cNumberOfWindowsX, int *cNumberOfWindowsY,
										   int *cNumberOfBlockPerWindowX, int *cNumberOfBlockPerWindowY);

extern "C" __host__ void GetProcessedImage(unsigned char* hostImage, int imageType);

extern "C" __host__ float3* CUDAImageRescale(float3* src, int width, int height, int &rWidth, int &rHeight, float scale);

__host__ void InitCUDAHOG(int cellSizeX, int cellSizeY,
						  int blockSizeX, int blockSizeY,
						  int windowSizeX, int windowSizeY,
						  int noOfHistogramBins, float wtscale,
						  float svmBias, float* svmWeights, int svmWeightsCount,
						  bool useGrayscale);
__host__ void CloseCUDAHOG();

#endif
