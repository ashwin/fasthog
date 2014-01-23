#include "HOGEngine.h"
#include "HOGNMS.h"

#include "HOGDefines.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cstdio>

using namespace HOG;

HOGEngine* HOGEngine::instance;

extern "C" void InitHOG(int width, int height, int avSizeX, int avSizeY,
								 int marginX, int marginY, int cellSizeX, int cellSizeY,
								 int blockSizeX, int blockSizeY, int windowSizeX, int windowSizeY,
								 int noOfHistogramBins, float wtscale, float svmBias, float* svmWeights,
								 int svmWeightsCount, bool useGrayscale);

extern "C" void CloseHOG();

extern "C" void BeginHOGProcessing(unsigned char* hostImage, int minx, int miny, int maxx, int maxy, float minScale, float maxScale);
extern "C" float* EndHOGProcessing();

extern "C" void GetProcessedImage(unsigned char* hostImage, int imageType);
extern "C" void GetHOGParameters(float *cStartScale, float *cEndScale, float *cScaleRatio, int *cScaleCount,
										   int *cPaddingSizeX, int *cPaddingSizeY, int *cPaddedWidth, int *cPaddedHeight,
										   int *cNoOfCellsX, int *cNoOfCellsY, int *cNoOfBlocksX, int *cNoOfBlocksY,
										   int *cNumberOfWindowsX, int *cNumberOfWindowsY,
										   int *cNumberOfBlockPerWindowX, int *cNumberOfBlockPerWindowY);

int HOGEngine::iDivUpF(int a, float b) { return (a % int(b) != 0) ? int(a / b + 1) : int(a / b);}

void HOGEngine::InitializeHOG(int iw, int ih, std::string fileName)
{
	this->imageWidth = iw;
	this->imageHeight = ih;

	this->avSizeX = 0;
	this->avSizeY = 0;
	this->marginX = 0;
	this->marginY = 0;

	this->hCellSizeX = 4; // 8
	this->hCellSizeY = 4; // 8
	this->hBlockSizeX = 2;
	this->hBlockSizeY = 2;
	this->hWindowSizeX = 24; //64
	this->hWindowSizeY = 24; //128
	this->hNoOfHistogramBins = 9;

	this->wtScale = 2.0f;

	this->useGrayscale = false;

	this->readSVMFromFile(fileName);

	this->formattedResultsAvailable = false;

	nmsProcessor = new HOGNMS();

	InitHOG(iw, ih, avSizeX, avSizeY, marginX, marginY, hCellSizeX, hCellSizeY, hBlockSizeX, hBlockSizeY,
		hWindowSizeX, hWindowSizeY, hNoOfHistogramBins, wtScale, svmBias, svmWeights, svmWeightsCount, useGrayscale);
}

void HOGEngine::InitializeHOG(int iw, int ih, float svmBias, float* svmWeights, int svmWeightsCount)
{
	this->imageWidth = iw;
	this->imageHeight = ih;

	this->avSizeX = 48; //48
	this->avSizeY = 96; //96
	this->marginX = 4; // 4
	this->marginY = 4; // 4

	this->hCellSizeX = 8;
	this->hCellSizeY = 8;
	this->hBlockSizeX = 2;
	this->hBlockSizeY = 2;
	this->hWindowSizeX = 64;
	this->hWindowSizeY = 128;
	this->hNoOfHistogramBins = 9;

	this->svmWeightsCount = svmWeightsCount;
	this->svmBias = svmBias;
	this->svmWeights = svmWeights;

	this->wtScale = 2.0f;

	this->useGrayscale = false;

	this->formattedResultsAvailable = false;

	nmsProcessor = new HOGNMS();

	InitHOG(iw, ih, avSizeX, avSizeY, marginX, marginY, hCellSizeX, hCellSizeY, hBlockSizeX, hBlockSizeY,
		hWindowSizeX, hWindowSizeY, hNoOfHistogramBins, wtScale, svmBias, svmWeights, svmWeightsCount, useGrayscale);
}


void HOGEngine::readSVMFromFile(std::string modelfile)
{
	double linearbias_, *linearwt_;

    FILE *modelfl;
#ifdef _WIN32
    if ((fopen_s (&modelfl, modelfile.c_str(), "rb")) != 0)
    { printf("File not found!\n"); exit(1); }
#else
    if ((modelfl = fopen (modelfile.c_str(), "rb")) == NULL)
    { printf("File not found!\n"); exit(1); }
#endif
    char version_buffer[10];
    if (!fread (&version_buffer,sizeof(char),10,modelfl))
    { printf("Wrong file version!\n"); exit(1); }

    if(strcmp(version_buffer,"V6.01")) {
    	printf("Wrong file version!\n"); exit(1);
    }
    /* read version number */
    int version = 0;
    if (!fread (&version,sizeof(int),1,modelfl))
    { printf("Wrong file version!\n"); exit(1); }
    if (version < 200)
    { printf("Wrong file version!\n"); exit(1); }

    long long kernel_type;
    fread(&(kernel_type),sizeof(long long),1,modelfl);

    {// ignore these
        long long poly_degree;
        fread(&(poly_degree),sizeof(long long),1,modelfl);

        double rbf_gamma;
        fread(&(rbf_gamma),sizeof(double),1,modelfl);

        double  coef_lin;
        fread(&(coef_lin),sizeof(double),1,modelfl);
        double coef_const;
        fread(&(coef_const),sizeof(double),1,modelfl);

        long long l;
        fread(&l,sizeof(long long),1,modelfl);
        char* custom = new char[(unsigned int)l];
        fread(custom,sizeof(char),(size_t)l,modelfl);
        delete[] custom;
    }

    long long totwords;
    fread(&(totwords),sizeof(long long),1,modelfl);

    {// ignore these
        long long totdoc;
        fread(&(totdoc),sizeof(long long),1,modelfl);

        long long sv_num;
        fread(&(sv_num), sizeof(long long),1,modelfl);
    }

    fread(&linearbias_, sizeof(double),1,modelfl);

    if(kernel_type == 0) { /* linear kernel */
        /* save linear wts also */
        linearwt_ = new double[(unsigned int)totwords+1];
		svmWeightsCount = (int) totwords;
        fread(linearwt_, sizeof(double),(size_t)totwords+1,modelfl);
    } else {
        exit(1);
    }

	svmWeights = new float[svmWeightsCount+1];
	for (int i=0; i<svmWeightsCount; i++)
		svmWeights[i] = (float) linearwt_[i];

	svmBias = (float)linearbias_;

	fclose(modelfl);

	delete linearwt_;
}

void HOGEngine::FinalizeHOG()
{
	delete nmsProcessor;

	CloseHOG();
}

void HOGEngine::BeginProcess(HOGImage* hostImage,
		int _minx, int _miny, int _maxx, int _maxy, float minScale, float maxScale)
{
	minX = _minx, minY = _miny, maxX = _maxx, maxY = _maxy;

	if (minY == -1 && minY == -1 && maxX == -1 && maxY == -1)
	{
		minX = 0;
		minY = 0;
		maxX = imageWidth;
		maxY = imageHeight;
	}

	BeginHOGProcessing(hostImage->pixels, minX, minY, maxX, maxY, minScale, maxScale);
}

void HOGEngine::EndProcess()
{
	cppResult = EndHOGProcessing();

	GetHOGParameters(&startScale, &endScale, &scaleRatio, &scaleCount,
		&hPaddingSizeX, &hPaddingSizeY, &hPaddedWidth, &hPaddedHeight,
		&hNoOfCellsX, &hNoOfCellsY, &hNoOfBlocksX, &hNoOfBlocksY, &hNumberOfWindowsX,
		&hNumberOfWindowsY, &hNumberOfBlockPerWindowX, &hNumberOfBlockPerWindowY);

	ComputeFormattedResults();

	nmsResults = nmsProcessor->ComputeNMSResults(formattedResults, formattedResultsCount, &nmsResultsAvailable, &nmsResultsCount,
		hWindowSizeX, hWindowSizeY);
}

void HOGEngine::GetImage(HOGImage *imageCUDA, ImageType imageType)
{
	switch (imageType)
	{
	case IMAGE_RESIZED:
		GetProcessedImage(imageCUDA->pixels, 0);
		break;
	case IMAGE_COLOR_GRADIENTS:
		GetProcessedImage(imageCUDA->pixels, 1);
		break;
	case IMAGE_GRADIENT_ORIENTATIONS:
		GetProcessedImage(imageCUDA->pixels, 2);
		break;
	case IMAGE_PADDED:
		GetProcessedImage(imageCUDA->pixels, 3);
		break;
	case IMAGE_ROI:
		GetProcessedImage(imageCUDA->pixels, 4);
		break;
	}
}

void HOGEngine::SaveResultsToDisk(char* fileName)
{
	FILE* f; 
#ifdef _WIN32
	fopen_s(&f, fileName, "w+");
#else
	f = fopen(fileName, "w+");
#endif
	fprintf(f, "%d\n", formattedResultsCount);
	for (int i=0; i<formattedResultsCount; i++)
	{
		fprintf(f, "%f %f %d %d %d %d %d %d\n",
			formattedResults[i].scale, formattedResults[i].score,
			formattedResults[i].width, formattedResults[i].height,
			formattedResults[i].x, formattedResults[i].y,
			formattedResults[i].origX, formattedResults[i].origY);
	}
	fclose(f);
}

void HOGEngine::ComputeFormattedResults()
{
	int i, j, k, resultId;
	int leftoverX, leftoverY, currentWidth, currentHeight, rNumberOfWindowsX, rNumberOfWindowsY;

	resultId = 0;
	formattedResultsCount = 0;

	float* currentScaleWOffset;
	float currentScale = startScale;

	for (i=0; i<scaleCount; i++)
	{
		currentScaleWOffset = cppResult + i * hNumberOfWindowsX * hNumberOfWindowsY;

		for (j = 0; j < hNumberOfWindowsY; j++)
		{
			for (k = 0; k < hNumberOfWindowsX; k++)
			{
				float score = currentScaleWOffset[k + j * hNumberOfWindowsX];
				if (score > 0)
					formattedResultsCount++;
			}
		}
	}

	if (formattedResultsAvailable) delete formattedResults;
	formattedResults = new HOGResult[formattedResultsCount];

	for (i=0; i<scaleCount; i++)
	{
		currentScaleWOffset = cppResult + i * hNumberOfWindowsX * hNumberOfWindowsY;

		for (j=0; j<hNumberOfWindowsY; j++)
		{
			for (k=0; k<hNumberOfWindowsX; k++)
			{
				float score = currentScaleWOffset[k + j * hNumberOfWindowsX];
				if (score > 0)
				{
					HOGResult hogResult;

					currentWidth = iDivUpF(hPaddedWidth, currentScale);
					currentHeight = iDivUpF(hPaddedHeight, currentScale);

					rNumberOfWindowsX = (currentWidth - hWindowSizeX) / hCellSizeX + 1;
					rNumberOfWindowsY = (currentHeight - hWindowSizeY) / hCellSizeY + 1;

					leftoverX = (currentWidth - hWindowSizeX - hCellSizeX * (rNumberOfWindowsX - 1)) / 2;
					leftoverY = (currentHeight - hWindowSizeY - hCellSizeY * (rNumberOfWindowsY - 1)) / 2;

					hogResult.origX = k * hCellSizeX + leftoverX;
					hogResult.origY = j * hCellSizeY + leftoverY;

					hogResult.width = (int)floorf((float)hWindowSizeX * currentScale);
					hogResult.height = (int)floorf((float)hWindowSizeY * currentScale);

					hogResult.x = (int)ceilf(currentScale * (hogResult.origX + hWindowSizeX / 2) - (float) hWindowSizeX * currentScale / 2) - hPaddingSizeX + minX;
					hogResult.y = (int)ceilf(currentScale * (hogResult.origY + hWindowSizeY / 2) - (float) hWindowSizeY * currentScale / 2) - hPaddingSizeY + minY;

					hogResult.scale = currentScale;
					hogResult.score = score;

					formattedResults[resultId] = hogResult;
					resultId++;
				}
			}
		}

		currentScale = currentScale * scaleRatio;
	}
}
