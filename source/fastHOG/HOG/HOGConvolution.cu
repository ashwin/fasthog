#include "HOGConvolution.h"
#include "HOGUtils.h"
#include "cutil.h"

dim3 blockGridRows;
dim3 blockGridColumns;
dim3 threadBlockRows;
dim3 threadBlockColumns;

#define convKernelRadius 1
#define convKernelWidth (2 * convKernelRadius + 1)
__device__ __constant__ float d_Kernel[convKernelWidth];
float *h_Kernel;

#define convRowTileWidth 128
#define convKernelRadiusAligned 16

#define convColumnTileWidth 16
#define convColumnTileHeight 48

float4 *convBuffer4;
float1 *convBuffer1;

int convWidth;
int convHeight;
const int convKernelSize = convKernelWidth * sizeof(float);

bool convUseGrayscale;

template<int i> __device__ float1 convolutionRow(float1 *data) {
	float1 val = data[convKernelRadius-i];
	val.x *= d_Kernel[i];
	val.x += convolutionRow<i-1>(data).x;
	return val;
}
template<> __device__ float1 convolutionRow<-1>(float1 *data){float1 zero; zero.x = 0; return zero;}
template<int i> __device__ float1 convolutionColumn(float1 *data) {
	float1 val = data[(convKernelRadius-i)*convColumnTileWidth];
	val.x *= d_Kernel[i];
	val.x += convolutionColumn<i-1>(data).x;
	return val;
}
template<> __device__ float1 convolutionColumn<-1>(float1 *data){float1 zero; zero.x = 0; return zero;}

template<int i> __device__ float4 convolutionRow(float4 *data) {
	float4 val = data[convKernelRadius-i];
	val.x *= d_Kernel[i]; val.y *= d_Kernel[i];
	val.z *= d_Kernel[i]; val.w *= d_Kernel[i];
	float4 val2 = convolutionRow<i-1>(data);
	val.x += val2.x; val.y += val2.y;
	val.z += val2.z; val.w += val2.w;
	return val;
}
template<> __device__ float4 convolutionRow<-1>(float4 *data) {
	float4 zero; zero.x = 0; zero.y = 0; zero.z = 0; zero.w = 0;
	return zero;
}
template<int i> __device__ float4 convolutionColumn(float4 *data) {
	float4 val = data[(convKernelRadius-i)*convColumnTileWidth];
	val.x *= d_Kernel[i]; val.y *= d_Kernel[i];
	val.z *= d_Kernel[i]; val.w *= d_Kernel[i];
	float4 val2 = convolutionColumn<i-1>(data);
	val.x += val2.x; val.y += val2.y;
	val.z += val2.z; val.w += val2.w;
	return val;
}
template<> __device__ float4 convolutionColumn<-1>(float4 *data) {
	float4 zero; zero.x = 0; zero.y = 0; zero.z = 0; zero.w = 0;
	return zero;
}

__global__ void convolutionRowGPU1(float1 *d_Result, float1 *d_Data, int dataW, int dataH)
{
	float1 zero; zero.x = 0;

	const int rowStart = IMUL(blockIdx.y, dataW);

	__shared__ float1 data[convKernelRadius + convRowTileWidth + convKernelRadius];

	const int tileStart = IMUL(blockIdx.x, convRowTileWidth);
	const int tileEnd = tileStart + convRowTileWidth - 1;
	const int apronStart = tileStart - convKernelRadius;
	const int apronEnd = tileEnd + convKernelRadius;

	const int tileEndClamped = min(tileEnd, dataW - 1);
	const int apronStartClamped = max(apronStart, 0);
	const int apronEndClamped = min(apronEnd, dataW - 1);

	const int apronStartAligned = tileStart - convKernelRadiusAligned;

	const int loadPos = apronStartAligned + threadIdx.x;

	if(loadPos >= apronStart)
	{
		const int smemPos = loadPos - apronStart;
		data[smemPos] = ((loadPos >= apronStartClamped) && (loadPos <= apronEndClamped)) ? d_Data[rowStart + loadPos] : zero;
	}

	__syncthreads();
	const int writePos = tileStart + threadIdx.x;

	if(writePos <= tileEndClamped)
	{
		const int smemPos = writePos - apronStart;
		float1 sum = convolutionRow<2 * convKernelRadius>(data + smemPos);
		d_Result[rowStart + writePos] = sum;
	}
}
__global__ void convolutionRowGPU4(float4 *d_Result, float4 *d_Data, int dataW, int dataH)
{
	float4 zero; zero.x = 0; zero.y = 0; zero.z = 0; zero.w = 0;

	const int rowStart = IMUL(blockIdx.y, dataW);

	__shared__ float4 data[convKernelRadius + convRowTileWidth + convKernelRadius];

	const int tileStart = IMUL(blockIdx.x, convRowTileWidth);
	const int tileEnd = tileStart + convRowTileWidth - 1;
	const int apronStart = tileStart - convKernelRadius;
	const int apronEnd = tileEnd + convKernelRadius;

	const int tileEndClamped = min(tileEnd, dataW - 1);
	const int apronStartClamped = max(apronStart, 0);
	const int apronEndClamped = min(apronEnd, dataW - 1);

	const int apronStartAligned = tileStart - convKernelRadiusAligned;

	const int loadPos = apronStartAligned + threadIdx.x;

	if(loadPos >= apronStart)
	{
		const int smemPos = loadPos - apronStart;
		data[smemPos] = ((loadPos >= apronStartClamped) && (loadPos <= apronEndClamped)) ? d_Data[rowStart + loadPos] : zero;
	}

	__syncthreads();
	const int writePos = tileStart + threadIdx.x;

	if(writePos <= tileEndClamped)
	{
		const int smemPos = writePos - apronStart;
		float4 sum = convolutionRow<2 * convKernelRadius>(data + smemPos);
		d_Result[rowStart + writePos] = sum;
	}
}
__global__ void convolutionColumnGPU1to2 ( float2 *d_Result, float1 *d_Data, float1 *d_DataRow, int dataW, int dataH, int smemStride, int gmemStride)
{
	float1 rowValue;
	float1 zero; zero.x = 0;
	float2 result;

	const int columnStart = IMUL(blockIdx.x, convColumnTileWidth) + threadIdx.x;

	__shared__ float1 data[convColumnTileWidth * (convKernelRadius + convColumnTileHeight + convKernelRadius)];

	const int tileStart = IMUL(blockIdx.y, convColumnTileHeight);
	const int tileEnd = tileStart + convColumnTileHeight - 1;
	const int apronStart = tileStart - convKernelRadius;
	const int apronEnd = tileEnd   + convKernelRadius;

	const int tileEndClamped = min(tileEnd, dataH - 1);
	const int apronStartClamped = max(apronStart, 0);
	const int apronEndClamped = min(apronEnd, dataH - 1);

	int smemPos = IMUL(threadIdx.y, convColumnTileWidth) + threadIdx.x;
	int gmemPos = IMUL(apronStart + threadIdx.y, dataW) + columnStart;

	for(int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y)
	{
		data[smemPos] = ((y >= apronStartClamped) && (y <= apronEndClamped)) ?  d_Data[gmemPos] : zero;
		smemPos += smemStride;
		gmemPos += gmemStride;
	}

	__syncthreads();

	smemPos = IMUL(threadIdx.y + convKernelRadius, convColumnTileWidth) + threadIdx.x;
	gmemPos = IMUL(tileStart + threadIdx.y , dataW) + columnStart;

	for(int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y)
	{
		float1 sum = convolutionColumn<2 * convKernelRadius>(data + smemPos);
		rowValue = d_DataRow[gmemPos];

		result.x = sqrtf(sum.x * sum.x + rowValue.x * rowValue.x);
		result.y = atan2f(sum.x, rowValue.x) * RADTODEG;

		d_Result[gmemPos] = result;
		smemPos += smemStride;
		gmemPos += gmemStride;
	}
}

__global__ void convolutionColumnGPU4to2 ( float2 *d_Result, float4 *d_Data, float4 *d_DataRow, int dataW, int dataH, int smemStride, int gmemStride)
{
	//float3 max12, mag4;
	float3 mag1, mag2, mag3;
	float3 max34, magMax;
	float2 result;
	float4 rowValue;
	float4 zero; zero.x = 0; zero.y = 0; zero.z = 0; zero.w = 0;

	const int columnStart = IMUL(blockIdx.x, convColumnTileWidth) + threadIdx.x;

	__shared__ float4 data[convColumnTileWidth * (convKernelRadius + convColumnTileHeight + convKernelRadius)];

	const int tileStart = IMUL(blockIdx.y, convColumnTileHeight);
	const int tileEnd = tileStart + convColumnTileHeight - 1;
	const int apronStart = tileStart - convKernelRadius;
	const int apronEnd = tileEnd   + convKernelRadius;

	const int tileEndClamped = min(tileEnd, dataH - 1);
	const int apronStartClamped = max(apronStart, 0);
	const int apronEndClamped = min(apronEnd, dataH - 1);

	int smemPos = IMUL(threadIdx.y, convColumnTileWidth) + threadIdx.x;
	int gmemPos = IMUL(apronStart + threadIdx.y, dataW) + columnStart;

	for(int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y)
	{
		data[smemPos] = ((y >= apronStartClamped) && (y <= apronEndClamped)) ?  d_Data[gmemPos] : zero;
		smemPos += smemStride;
		gmemPos += gmemStride;
	}

	__syncthreads();

	smemPos = IMUL(threadIdx.y + convKernelRadius, convColumnTileWidth) + threadIdx.x;
	gmemPos = IMUL(tileStart + threadIdx.y , dataW) + columnStart;

	for(int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y)
	{
		float4 sum = convolutionColumn<2 * convKernelRadius>(data + smemPos);
		rowValue = d_DataRow[gmemPos];

		mag1.x = sqrtf(sum.x * sum.x + rowValue.x * rowValue.x); mag1.y = sum.x; mag1.z = rowValue.x;
		mag2.x = sqrtf(sum.y * sum.y + rowValue.y * rowValue.y); mag2.y = sum.y; mag2.z = rowValue.y;
		mag3.x = sqrtf(sum.z * sum.z + rowValue.z * rowValue.z); mag3.y = sum.z; mag3.z = rowValue.z;

		max34 = (mag2.x > mag3.x) ? mag2 : mag3;
		magMax = (mag1.x > max34.x) ? mag1 : max34;

		result.x = magMax.x;
		result.y = atan2f(magMax.y, magMax.z);
		result.y = result.y * 180 / PI + 180;
		result.y = int(result.y) % 180; //TODO-> if semicerc

		d_Result[gmemPos] = result;
		smemPos += smemStride;
		gmemPos += gmemStride;
	}
}
__host__ void InitConvolution(int width, int height, bool useGrayscale)
{
	convUseGrayscale = useGrayscale;

	h_Kernel = (float *)malloc(convKernelSize);
	h_Kernel[0] = 1.0f; h_Kernel[1] = 0;  h_Kernel[2] = -1.0f;

	cutilSafeCall( cudaMemcpyToSymbol(d_Kernel, h_Kernel, convKernelSize) );

	if (useGrayscale)
		cutilSafeCall(cudaMalloc((void**) &convBuffer1, sizeof(float1) * width * height));
	else
		cutilSafeCall(cudaMalloc((void**) &convBuffer4, sizeof(float4) * width * height));
}

__host__ void SetConvolutionSize(int width, int height)
{
	convWidth = width;
	convHeight = height;

	blockGridRows = dim3(iDivUp(convWidth, convRowTileWidth), convHeight);
	blockGridColumns = dim3(iDivUp(convWidth, convColumnTileWidth), iDivUp(convHeight, convColumnTileHeight));
	threadBlockRows = dim3(convKernelRadiusAligned + convRowTileWidth + convKernelRadius);
	threadBlockColumns = dim3(convColumnTileWidth, 8);
}
__host__ void CloseConvolution()
{
	if (convUseGrayscale)
		cutilSafeCall(cudaFree(convBuffer1));
	else
		cutilSafeCall(cudaFree(convBuffer4));

	free(h_Kernel);
}
__host__ void ComputeColorGradients1to2(float1* inputImage, float2* outputImage)
{
	convolutionRowGPU1<<<blockGridRows, threadBlockRows>>>(convBuffer1, inputImage, convWidth, convHeight);
	convolutionColumnGPU1to2<<<blockGridColumns, threadBlockColumns>>>(outputImage, inputImage, convBuffer1, convWidth, convHeight,
		convColumnTileWidth * threadBlockColumns.y, convWidth * threadBlockColumns.y);
}

__host__ void ComputeColorGradients4to2(float4* inputImage, float2* outputImage)
{
	convolutionRowGPU4<<<blockGridRows, threadBlockRows>>>(convBuffer4, inputImage, convWidth, convHeight);
	convolutionColumnGPU4to2<<<blockGridColumns, threadBlockColumns>>>(outputImage, inputImage, convBuffer4, convWidth, convHeight,
		convColumnTileWidth * threadBlockColumns.y, convWidth * threadBlockColumns.y);
}
