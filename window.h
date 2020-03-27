#ifndef WINDOW_H
#define WINDOW_H

#include <iostream>
#include <SDL/SDL.h>

class rgbColor
{
public:
	unsigned char r = 0, g = 0, b = 0, a = 0;

	rgbColor(unsigned char r = 0, unsigned char g = 0, unsigned char b = 0, unsigned char a = 255) : r{ r }, g{ g }, b{ b }, a{ a }
	{}

	rgbColor operator=(rgbColor const &c)
	{
		r = c.r;
		g = c.g;
		b = c.b;
		a = c.a;
		return *this;
	}
	

};

class window {
/// Variables
public:


private:
	SDL_Window* sdlWindow = 0;
	SDL_Renderer* sdlRenderer = 0;

	int windowWidth, windowHeight;

	rgbColor bgColor;

	bool isRunning = 0;


/// Functions & Methods
public:
	// Constructor & Deconstructor
	window(const char title[], int windowWidth, int windowHeight);
	~window();

	// Getters
	bool checkIfRunning() const;
	SDL_Renderer* getRenderer();

	// Setters
	void setBackgroundColor(const rgbColor c);
	void stopWindow();

	// Refresh Window
	void renderWindow();

private:



};

#endif
