#ifndef GFX_HELPER_H
#define GFX_HELPER_H

/*
	I wrote this header-only helper to be used in most of my SDL projects
		This is cut down and comments are limited
*/

#include <SDL/SDL.h>

#define PI  3.14159265359

// Don't use this standalone
void gfxDrawCircle(SDL_Renderer* renderer, int cx, int cy, int x, int y)
{
	SDL_RenderDrawPoint(renderer, cx + x, cy + y);
	SDL_RenderDrawPoint(renderer, cx - x, cy + y);
	SDL_RenderDrawPoint(renderer, cx + x, cy - y);
	SDL_RenderDrawPoint(renderer, cx - x, cy - y);
	SDL_RenderDrawPoint(renderer, cx + y, cy + x);
	SDL_RenderDrawPoint(renderer, cx - y, cy + x);
	SDL_RenderDrawPoint(renderer, cx + y, cy - x);
	SDL_RenderDrawPoint(renderer, cx - y, cy - x);
}

// Don't use this standalone
void gfxDrawFilledCircle(SDL_Renderer* renderer, int cx, int cy, int x, int y)
{
	for (int i = -x; i < x; i++)
	{
		SDL_RenderDrawPoint(renderer, cx + i, cy + y);
		SDL_RenderDrawPoint(renderer, cx + i, cy - y);
	}

	for (int j = -y; j < y; j++)
	{
		SDL_RenderDrawPoint(renderer, cx + j, cy + x);
		SDL_RenderDrawPoint(renderer, cx + j, cy - x);
	}
}

/*
	Draws a circle using Bresenham's Circle Algorithm
		@param renderer Render target
		@param cx		The x-axis component of the circle's center
		@param cy		The y-axis component of the circle's center
		@param radius	The radius of the circle
		@param filled	Should the circle be filled in or outline only
*/
void gfxDrawBrenCircle(SDL_Renderer* renderer, int cx, int cy, int radius, bool filled)
{
	if (radius <= 1)
	{
		SDL_RenderDrawPoint(renderer, cx, cy);
		return;
	}

	int x = 0, y = radius, d = 3 - (2 * radius);

	if (!filled)
		gfxDrawCircle(renderer, cx, cy, x, y);
	else
		gfxDrawFilledCircle(renderer, cx, cy, x, y);

	while (x <= y)
	{
		x++;
		
		if (d < 0)
			d = d + (4 * x) + 6;
		else
		{
			d = d + 4 * (x - y) + 6;
			y--;
		}


		if (!filled)
			gfxDrawCircle(renderer, cx, cy, x, y);
		else
			gfxDrawFilledCircle(renderer, cx, cy, x, y);
	}
}

/* 
	Draws a horizontal line
	@param renderer Render target
	@param cx		The x-axis component of the line's center
	@param cy		The y-axis component of the line's center
	@param width	The distance left and right it should go, in pixels
*/
void gfxDrawHorizontalLine(SDL_Renderer* renderer, int x, int y, int width)
{
	SDL_RenderDrawLine(renderer, x - width, y, x + width, y);
}

/* 
	Draws a square
	@param renderer Render target
	@param cx		The x-axis component of the square's center
	@param cy		The y-axis component of the square's center
	@param width	The distance from the center to the top, bottom, left, and right mid-points 
*/
void gfxDrawSquare(SDL_Renderer* renderer, int cx, int cy, int width)
{
	for (int j = -width; j < width; j++)
		gfxDrawHorizontalLine(renderer, cx, cy + j, width);
}

#endif