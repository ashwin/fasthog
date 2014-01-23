#ifndef __HOG_HISTOGRAM__
#define __HOG_HISTOGRAM__

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

__host__ void InitHistograms(int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY, int noHistogramBins, float wtscale);
__host__ void CloseHistogram();

__host__ void ComputeBlockHistogramsWithGauss(float2* inputImage, float1* blockHistograms, int noHistogramBins,
											  int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY,
											  int windowSizeX, int windowSizeY,
											  int width, int height);
__host__ void NormalizeBlockHistograms(float1* blockHistograms, int noHistogramBins,
									  int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY,
									  int width, int height);

__global__ void computeBlockHistogramsWithGauss(float2* inputImage, float1* blockHistograms, int noHistogramBins,
												int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY,
												int leftoverX, int leftoverY, int width, int height);

__global__ void normalizeBlockHistograms(float1 *blockHistograms, int noHistogramBins,
										int rNoOfHOGBlocksX, int rNoOfHOGBlocksY,
										int blockSizeX, int blockSizeY,
										int alignedBlockDimX, int alignedBlockDimY, int alignedBlockDimZ,
										int width, int height);

#endif
