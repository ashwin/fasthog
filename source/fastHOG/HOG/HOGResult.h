#ifndef __HOG_RESUL__
#define __HOG_RESUL__

namespace HOG
{
	class HOGResult
	{
	public:
		float score;
		float scale;

		int width, height;
		int origX, origY;
		int x, y;

		HOGResult()
		{
			width = 0;
			height = 0;
			origX = 0;
			origY = 0;
			x = 0;
			y = 0;
		}
	};
}

#endif

