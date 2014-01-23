#ifndef __HOG_VECTOR_3D__
#define __HOG_VECTOR_3D__

namespace HOG
{
	class HOGPoint3
	{
	public:
		float x,y,z;

		HOGPoint3(float x, float y, float z) { this->x = x; this->y = y; this->z = z; }
		HOGPoint3() { this->x = 0; this->y = 0; this->z = 0; }
	};
}

#endif

