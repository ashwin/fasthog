#include "HOGNMS.h"

#include <math.h>

using namespace HOG;

HOGNMS::HOGNMS()
{
	center = 0.0f; scale = 1.0f;
	nonmaxSigma[0] = 8.0f; nonmaxSigma[1] = 16.0f; nonmaxSigma[2] = 1.3f;
	maxIterations = 100;
	modeEpsilon = (float)1e-5;
	epsFinalDist = 1.0f;

	nsigma[0] = nonmaxSigma[0]; nsigma[1] = nonmaxSigma[1]; nsigma[2] = logf(nonmaxSigma[2]);

	isAllocated = false;
}

HOGNMS::~HOGNMS()
{
	if (isAllocated)
	{
		delete tomode;
		delete wt;
		delete ms;
		delete at;
		delete nmsResults;
		delete nmsToMode;
	}
}

void HOGNMS::nvalue(HOGPoint3* ms, HOGPoint3* at, float* wt, int length)
{
	int i, j;
	float dotxmr, w;
	HOGPoint3 x, r, ns, numer, denum;

	for (i=0; i<length; i++)
	{
		numer.x = 0; numer.y = 0; numer.z = 0;
		denum.x = 0; denum.y = 0; denum.z = 0;

		for (j=0; j<length; j++)
		{
			ns.x = nsigma[0] * expf(at[j].z); ns.y =  nsigma[1] * expf(at[j].z); ns.z = nsigma[2];
			x.x = at[j].x / ns.x; x.y = at[j].y / ns.y; x.z = at[j].z / ns.z;
			r.x = at[i].x / ns.x; r.y = at[i].y / ns.y; r.z = at[i].z / ns.z;

			dotxmr = (x.x - r.x) * (x.x - r.x) + (x.y - r.y) * (x.y - r.y) + (x.z - r.z) * (x.z - r.z);
			w = wt[j] * expf(-dotxmr/2.0f)/sqrtf(ns.x * ns.y * ns.z);

			numer.x += w * x.x; numer.y += w * x.y; numer.z += w * x.z;
			denum.x += w / ns.x; denum.y += w / ns.y; denum.z += w / ns.z;
		}

		ms[i].x = numer.x / denum.x; ms[i].y = numer.y / denum.y; ms[i].z = numer.z / denum.z;
	}
}

void HOGNMS::nvalue(HOGPoint3 *ms, HOGPoint3* msnext, HOGPoint3* at, float* wt, int length)
{
	int j;
	float dotxmr, w;
	HOGPoint3 x, r, ns, numer, denum, toReturn;

	for (j=0; j<length; j++)
	{
		ns.x = nsigma[0] * expf(at[j].z); ns.y =  nsigma[1] * expf(at[j].z); ns.z = nsigma[2];
		x.x = at[j].x / ns.x; x.y = at[j].y / ns.y; x.z = at[j].z / ns.z;
		r.x = ms->x / ns.x; r.y = ms->y / ns.y; r.z = ms->z / ns.z;

		dotxmr = (x.x - r.x) * (x.x - r.x) + (x.y - r.y) * (x.y - r.y) + (x.z - r.z) * (x.z - r.z);
		w = wt[j] * expf(-dotxmr/2.0f)/sqrtf(ns.x * ns.y * ns.z);

		numer.x += w * x.x; numer.y += w * x.y; numer.z += w * x.z;
		denum.x += w / ns.x; denum.y += w / ns.y; denum.z += w / ns.z;
	}

	msnext->x = numer.x / denum.x; msnext->y = numer.y / denum.y; msnext->z = numer.z / denum.z;
}

void HOGNMS::fvalue(HOGPoint3* modes, HOGResult* results, int lengthModes, HOGPoint3* at, float* wt, int length)
{
	int i, j;
	float no, dotxx;
	HOGPoint3 x, ns;
	for (i=0; i<lengthModes; i++)
	{
		no = 0;
		for (j=0; j<length; j++)
		{
			ns.x = nsigma[0] * expf(at[j].z); ns.y =  nsigma[1] * expf(at[j].z); ns.z = nsigma[2];
			x.x = (at[j].x - modes[i].x) / ns.x;
			x.y = (at[j].y - modes[i].y) / ns.y;
			x.z = (at[j].z - modes[i].z) / ns.z;

			dotxx = x.x * x.x + x.y * x.y + x.z * x.z;

			no += wt[j] * expf(-dotxx/2)/sqrtf(ns.x * ns.y * ns.z);
		}
		results[i].score = no;
	}
}

float HOGNMS::distqt(HOGPoint3 *p1, HOGPoint3 *p2)
{
	HOGPoint3 ns, b;
	ns.x = nsigma[0] * expf(p2->z); ns.y = nsigma[1] * expf(p2->z); ns.z = nsigma[2];
	b.x = p2->x - p1->x; b.y = p2->y - p1->y; b.z = p2->z - p1->z;
	b.x /= ns.x; b.y /= ns.y; b.z /= ns.z;
	return b.x * b.x + b.y * b.y + b.z * b.z;
}

void HOGNMS::shiftToMode(HOGPoint3* ms, HOGPoint3* at, float* wt, HOGPoint3 *tomode, int length)
{
	int i, count;
	HOGPoint3 ii,II;
	for (i=0; i<length; i++)
	{
		II = ms[i];;
		count = 0;

		do
		{
			ii = II;
			nvalue(&ii, &II, at, wt, length);
			++count;
		} while ( count < maxIterations && distqt(&ii,&II) > modeEpsilon );

		tomode[i].x = II.x; tomode[i].y = II.y; tomode[i].z = II.z;
	}
}

HOGResult* HOGNMS::ComputeNMSResults(HOGResult* formattedResults, int formattedResultsCount, bool *nmsResultsAvailable, int *nmsResultsCount,
									 int hWindowSizeX, int hWindowSizeY)
{
	if (!isAllocated)
	{
		wt = new float[hWindowSizeX * hWindowSizeX];
		at = new HOGPoint3[hWindowSizeX * hWindowSizeX];
		ms = new HOGPoint3[hWindowSizeX * hWindowSizeX];
		tomode = new HOGPoint3[hWindowSizeX * hWindowSizeX];
		nmsToMode = new HOGPoint3[hWindowSizeX * hWindowSizeX];
		nmsResults = new HOGResult[hWindowSizeX * hWindowSizeX];
		isAllocated = true;
	}

	int i, j;
	float cenx, ceny, nmsOK;

	*nmsResultsCount = 0;
	nmsResultsAvailable = false;

	for (i=0; i<formattedResultsCount; i++)
	{
		wt[i] = this->sigmoid(formattedResults[i].score);
		cenx = formattedResults[i].x + formattedResults[i].width / 2.0f;
		ceny = formattedResults[i].y + formattedResults[i].height / 2.0f;
		at[i] = HOGPoint3(cenx, ceny, logf(formattedResults[i].scale));
	}

	nvalue(ms, at, wt, formattedResultsCount);
	shiftToMode(ms, at, wt, tomode, formattedResultsCount);

	for (i=0; i<formattedResultsCount; i++)
	{
		nmsOK = true;
		for (j=0; j<*nmsResultsCount; j++)
		{
			if (distqt(&nmsToMode[j], &tomode[i]) < epsFinalDist)
			{
				nmsOK = false;
				break;
			}
		}

		if (nmsOK)
		{
			nmsResults[*nmsResultsCount].scale = expf(tomode[i].z);

			nmsResults[*nmsResultsCount].width = (int)floorf((float)hWindowSizeX * nmsResults[*nmsResultsCount].scale);
			nmsResults[*nmsResultsCount].height = (int)floorf((float)hWindowSizeY * nmsResults[*nmsResultsCount].scale);

			nmsResults[*nmsResultsCount].x = (int)ceilf(tomode[i].x - (float) hWindowSizeX * nmsResults[*nmsResultsCount].scale / 2);
			nmsResults[*nmsResultsCount].y = (int)ceilf(tomode[i].y - (float) hWindowSizeY * nmsResults[*nmsResultsCount].scale / 2);
	
			nmsToMode[*nmsResultsCount] = tomode[i];

			(*nmsResultsCount)++;
		}
	}

	fvalue(nmsToMode, nmsResults, *nmsResultsCount, at, wt, formattedResultsCount);

	return nmsResults;
}
