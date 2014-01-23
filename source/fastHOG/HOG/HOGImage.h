/*
* HOGImage.h
*
*  Created on: May 14, 2009
*      Author: viprad
*/

#ifndef __HOGIMAGE_H__
#define __HOGIMAGE_H__

namespace HOG
{
	class HOGImage
	{
	public:
		//must me uchar4
		bool isLoaded;

		int width, height;
		unsigned char* pixels;

		HOGImage(char* fileName);
		HOGImage(int width, int height);
		HOGImage(int width, int height, unsigned char *pixels);

		virtual ~HOGImage();
	};
}

#endif /* HOGIMAGE_H_ */
