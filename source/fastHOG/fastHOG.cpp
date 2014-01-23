/*
 * fastHog.cpp
 *
 *  Created on: May 14, 2009
 *      Author: viprad
 */

#include <stdio.h>

#include <boost/thread/thread.hpp>
#include <fltk/run.h>

#include "HOG/HOGEngine.h"
#include "HOG/HOGImage.h"

#include "Utils/ImageWindow.h"
#include "Utils/Timer.h"

#include "Others/persondetectorwt.tcc"

using namespace HOG;

ImageWindow* fastHOGWindow;
HOGImage* image;
HOGImage* imageCUDA;

void doStuffHere()
{
	HOGEngine::Instance()->InitializeHOG(image->width, image->height,
			PERSON_LINEAR_BIAS, PERSON_WEIGHT_VEC, PERSON_WEIGHT_VEC_LENGTH);

	//HOGEngine::Instance()->InitializeHOG(image->width, image->height,
	//		"Files//SVM//head_W24x24_C4x4_N2x2_G4x4_HeadSize16x16.alt");

	Timer t;
	t.restart();
	HOGEngine::Instance()->BeginProcess(image);
	HOGEngine::Instance()->EndProcess();
	t.stop(); t.check("Processing time");

	printf("Found %d positive results.\n", HOGEngine::Instance()->formattedResultsCount);

	HOGEngine::Instance()->GetImage(imageCUDA, HOGEngine::IMAGE_ROI);
	fastHOGWindow->setImage(imageCUDA);

	for (int i=0; i<HOGEngine::Instance()->nmsResultsCount; i++)
	{
		printf("%1.5f %1.5f %4d %4d %4d %4d %4d %4d\n",
				HOGEngine::Instance()->nmsResults[i].scale,
				HOGEngine::Instance()->nmsResults[i].score,
				HOGEngine::Instance()->nmsResults[i].origX,
				HOGEngine::Instance()->nmsResults[i].origY,
				HOGEngine::Instance()->nmsResults[i].x,
				HOGEngine::Instance()->nmsResults[i].y,
				HOGEngine::Instance()->nmsResults[i].width,
				HOGEngine::Instance()->nmsResults[i].height);
				fastHOGWindow->drawRect(HOGEngine::Instance()->nmsResults[i].x,
						HOGEngine::Instance()->nmsResults[i].y,
						HOGEngine::Instance()->nmsResults[i].width,
						HOGEngine::Instance()->nmsResults[i].height);
	}

	printf("Drawn %d positive results.\n", HOGEngine::Instance()->nmsResultsCount);

	HOGEngine::Instance()->FinalizeHOG();
}

int main(void)
{
	image = new HOGImage("Files//Images//testImage.bmp");
	imageCUDA = new HOGImage(image->width,image->height);

	fastHOGWindow = new ImageWindow(image, "fastHOG");
	fastHOGWindow->doStuff = &doStuffHere;
	fastHOGWindow->show();

	fltk::run();

	delete image;
	delete imageCUDA;

	return 0;
}
