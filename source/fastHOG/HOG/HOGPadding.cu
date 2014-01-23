#include "HOGPadding.h"
#include "HOGUtils.h"
#include "cutil.h"

extern int hWidthROI, hHeightROI;
extern int hPaddedWidth, hPaddedHeight;
extern int hWidth, hHeight;
extern int hPaddingSizeX, hPaddingSizeY;

extern int avSizeX, avSizeY, marginX, marginY;

uchar4* paddedRegisteredImageU4;

__host__ void InitPadding(int hPaddedWidth, int hPaddedHeight)
{
	cutilSafeCall(cudaMalloc((void**) &paddedRegisteredImageU4, sizeof(uchar4) * hPaddedWidth * hPaddedHeight));
}

__host__ void ClosePadding()
{
	cutilSafeCall(cudaFree(paddedRegisteredImageU4));
}

__host__ void PadHostImage(uchar4* registeredImage, float4 *paddedRegisteredImage,
		int minx, int miny, int maxx, int maxy)
{
	hWidthROI = maxx - minx;
	hHeightROI = maxy - miny;

	int toaddxx = 0, toaddxy = 0, toaddyx = 0, toaddyy = 0;

	if (avSizeX) { toaddxx = hWidthROI * marginX / avSizeX; toaddxy = hHeightROI * marginY / avSizeX; }
	if (avSizeY) { toaddyx = hWidthROI * marginX / avSizeY; toaddyy = hHeightROI * marginY / avSizeY; }

	hPaddingSizeX = max(toaddxx, toaddyx); hPaddingSizeY = max(toaddxy, toaddyy);

	hPaddedWidth = hWidthROI + hPaddingSizeX*2;
	hPaddedHeight = hHeightROI + hPaddingSizeY*2;

	cutilSafeCall(cudaMemset(paddedRegisteredImageU4, 0, sizeof(uchar4) * hPaddedWidth * hPaddedHeight));

	cutilSafeCall(cudaMemcpy2D(paddedRegisteredImageU4 + hPaddingSizeX + hPaddingSizeY * hPaddedWidth,
			hPaddedWidth * sizeof(uchar4), registeredImage + minx + miny * hWidth,
			hWidth * sizeof(uchar4), hWidthROI * sizeof(uchar4),
			hHeightROI, cudaMemcpyHostToDevice));

	Uchar4ToFloat4(paddedRegisteredImageU4, paddedRegisteredImage, hPaddedWidth, hPaddedHeight);
}
