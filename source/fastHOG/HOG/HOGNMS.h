#ifndef __HOG_NMS__
#define __HOG_NMS__

#include "HOGPoint3.h"
#include "HOGResult.h"

namespace HOG
{
	class HOGNMS
	{
	private:
		HOGPoint3 *at, *ms, *tomode, *nmsToMode;

		HOGResult *nmsResults;

		float* wt;

		float center, scale;
		float nonmaxSigma[3];
		float nsigma[3];
		float modeEpsilon;
		float epsFinalDist;

		int maxIterations;

		bool isAllocated;

		float sigmoid(float score) { return (score > center) ? scale * (score - center) : 0.0f; }
		void nvalue(HOGPoint3* ms, HOGPoint3* at, float* wt, int length);
		void nvalue(HOGPoint3* ms, HOGPoint3* msnext, HOGPoint3* at, float* wt, int length);
		void fvalue(HOGPoint3* modes, HOGResult* results, int lengthModes, HOGPoint3* at, float* wt, int length);
		void shiftToMode(HOGPoint3* ms, HOGPoint3* at, float* wt, HOGPoint3 *tomode, int length);
		float distqt(HOGPoint3 *p1, HOGPoint3 *p2);

	public:
		HOGResult* ComputeNMSResults(HOGResult* formattedResults, int formattedResultsCount, bool *nmsResultsAvailable, int *nmsResultsCount,
			int hWindowSizeX, int hWindowSizeY);

		HOGNMS();
		~HOGNMS(void);
	};
}
#endif
