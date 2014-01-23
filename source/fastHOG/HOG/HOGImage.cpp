/*
 * HOGImage.cpp
 *
 *  Created on: May 14, 2009
 *      Author: viprad
 */

#include "HOGImage.h"

#include <stdlib.h>
#include <string.h>

#include <stdio.h>

#include <FreeImage.h>

using namespace HOG;

HOGImage::HOGImage(int width, int height)
{
	this->width = width;
	this->height = height;

	isLoaded = false;
	this->pixels = (unsigned char*) malloc(sizeof(unsigned char) * 4 * width * height);
	memset(this->pixels, 0, sizeof(unsigned char) * 4 * width * height);
}

HOGImage::HOGImage(int width, int height, unsigned char* pixels)
{
	this->width = width;
	this->height = height;

	this->pixels = (unsigned char*) malloc(sizeof(unsigned char) * 4 * width * height);
	memcpy(this->pixels, pixels, sizeof(unsigned char) * 4 * width * height);

	isLoaded = true;
}

HOGImage::HOGImage(char* fileName)
{
	bool bLoaded = false;
	int bpp;
	FIBITMAP *bmp = 0;
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	fif = FreeImage_GetFileType(fileName);
	if (fif == FIF_UNKNOWN)
	{
		fif = FreeImage_GetFIFFromFilename(fileName);
	}

	if (fif != FIF_UNKNOWN && FreeImage_FIFSupportsReading(fif))
	{
		bmp = FreeImage_Load(fif, fileName, 0);
		bLoaded = true;
		if (bmp == NULL)
			bLoaded = false;
	}

	if (bLoaded)
	{
		width = FreeImage_GetWidth(bmp);
		height = FreeImage_GetHeight(bmp);

		bpp = FreeImage_GetBPP(bmp);
		switch (bpp)
		{
		case 32:
			break;
		default:
			FIBITMAP *bmpTemp = FreeImage_ConvertTo32Bits(bmp);
			if (bmp != NULL) FreeImage_Unload(bmp);
			bmp = bmpTemp;
			bpp = FreeImage_GetBPP(bmp);
			break;
		}

		this->pixels = (unsigned char*) malloc(sizeof(unsigned char) * 4 * width * height);
		FreeImage_ConvertToRawBits(this->pixels, bmp, width * 4, bpp, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, true);

		isLoaded = true;
	}
	else
		isLoaded = false;
}

HOGImage::~HOGImage()
{
	free(pixels);
}
