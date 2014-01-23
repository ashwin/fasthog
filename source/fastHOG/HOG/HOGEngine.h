#ifndef __HOG_ENGINE__
#define __HOG_ENGINE__

#include "HOGResult.h"
#include "HOGNMS.h"
#include "HOGImage.h"

#include <string>

using namespace std;

namespace HOG
{
	class HOGEngine
	{
	private:
		static HOGEngine* instance;

		int iDivUpF(int a, float b);

		HOGNMS* nmsProcessor;
		void readSVMFromFile(std::string fileName);

	public:
		int imageWidth, imageHeight;

		int avSizeX, avSizeY, marginX, marginY;

		int scaleCount;
		int hCellSizeX, hCellSizeY;
		int hBlockSizeX, hBlockSizeY;
		int hWindowSizeX, hWindowSizeY;
		int hNoOfHistogramBins;
		int hPaddedWidth, hPaddedHeight;
		int hPaddingSizeX, hPaddingSizeY;

		int minX, minY, maxX, maxY;

		float wtScale;

		float startScale, endScale, scaleRatio;

		int svmWeightsCount;
		float svmBias, *svmWeights;

		int hNoOfCellsX, hNoOfCellsY;
		int hNoOfBlocksX, hNoOfBlocksY;
		int hNumberOfWindowsX, hNumberOfWindowsY;
		int hNumberOfBlockPerWindowX, hNumberOfBlockPerWindowY;

		bool useGrayscale;

		float* cppResult;

		HOGResult* formattedResults;
		HOGResult* nmsResults;

		bool formattedResultsAvailable;
		int formattedResultsCount;

		bool nmsResultsAvailable;
		int nmsResultsCount;

		enum ImageType
		{
			IMAGE_RESIZED,
			IMAGE_COLOR_GRADIENTS,
			IMAGE_GRADIENT_ORIENTATIONS,
			IMAGE_PADDED,
			IMAGE_ROI
		};

		static HOGEngine* Instance(void) {
			if (instance == NULL) instance = new HOGEngine();
			return instance;
		}

		void InitializeHOG(int iw, int ih, float svmBias, float* svmWeights, int svmWeightsCount);
		void InitializeHOG(int iw, int ih, std::string fileName);

		void FinalizeHOG();

		void BeginProcess(HOGImage* hostImage, int _minx = -1, int _miny = -1, int _maxx = -1, int _maxy = -1,
			float minScale = -1.0f, float maxScale = -1.0f);
		void EndProcess();
		void GetImage(HOGImage *imageCUDA, ImageType imageType);

		void ComputeFormattedResults();

		void SaveResultsToDisk(char* fileName);

		HOGEngine(void) { }
		~HOGEngine(void) { }
	};
}

#endif
