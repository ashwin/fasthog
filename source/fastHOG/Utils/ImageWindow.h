#ifndef __IMAGE_WINDOW_H__
#define __IMAGE_WINDOW_H__

#include <fltk/Window.h>
#include <fltk/draw.h>
#include <fltk/Rectangle.h>
#include <fltk/Widget.h>
#include <fltk/events.h>

#include "../HOG/HOGImage.h"

#include <vector>
#include <cstddef>

using namespace HOG;

class ImageWidget: public fltk::Widget
{
	struct rect
	{
		int x, y, w, h;
		rect(int _x, int _y, int _w, int _h) { x = _x; y = _y; w = _w; h = _h; }
	};

public:
	std::vector<rect> rects;

	unsigned char* pixels;
	fltk::Rectangle* rectangle;

	ImageWidget(int x, int y, int w, int h) :
		fltk::Widget(x, y, w, h)
	{
		rectangle = new fltk::Rectangle(0, 0, w, h);
		this->box(fltk::BORDER_BOX);
		this->buttonbox(fltk::FLAT_BOX);
	}

	ImageWidget(int x, int y, int w, int h, unsigned char* pixels) :
		fltk::Widget(x, y, w, h)
	{
		this->pixels = pixels;
		rectangle = new fltk::Rectangle(0, 0, w, h);
		this->box(fltk::BORDER_BOX);
		this->buttonbox(fltk::FLAT_BOX);
	}

	void draw()
	{
		fltk::drawimage((unsigned char*) pixels, fltk::RGB32, *rectangle);
		fltk::setcolor(fltk::RED);
		for (size_t i=0; i<rects.size(); i++)
			fltk::strokerect(rects[i].x, rects[i].y, rects[i].w, rects[i].h);
		this->redraw();
	}

	void setImage(unsigned char* pixelsNew)
	{
		this->pixels = pixelsNew;
	}

	void drawRect(int x, int y, int w, int h)
	{
		rects.push_back(rect(x,y,w,h));
		this->redraw();
	}
};

class ImageWindow: public fltk::Window
{
	bool colorImage;

	int width, height;

	ImageWidget* imageWidget;
	fltk::Window *otherWindow;

public:

	void (*doStuff)();

	ImageWindow(int width, int height, char* title);
	ImageWindow(HOGImage* image, char* title);

	void setImage(HOGImage* image);

	void show(int x = -1, int y = -1);
	void drawRect(int x, int y, int w, int h);

	int handle(int);

	void Close();

	~ImageWindow(void);
};

#endif
