#include <stdio.h>
#include <iostream>
#include "window.h"
#include "gfxHelper.h"
#include "solver.cuh"

window* gameWindow;
float iTime = 0;

solver* sphSolver;

void inputs();
void update();
void draw();

int main(int args, char* argv[])
{
	gameWindow = new window("SPH CUDA Flowmap - Hunter Werenskjold", WINDOW_WIDTH, WINDOW_HEIGHT);
	
	SDL_Event e;
	while (gameWindow->checkIfRunning())
	{
		while (SDL_PollEvent(&e))
		{
			if ((e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) || e.type == SDL_QUIT)
			{
				gameWindow->stopWindow();
				break;
			}

			

		}

		// If the game window is no longer running, then we break from the game loop
		if (!gameWindow->checkIfRunning())
			break;

		inputs();

		update();

		draw();

		iTime += DELTA_TIME;
	}

	// Delete the gameWindow object and Grid object before closing the application
	delete gameWindow;

	return 0;
}

void inputs()
{}

void update()
{
	
}

void draw()
{
	SDL_Renderer* renderer = gameWindow->getRenderer();

	//SDL_SetRenderDrawColor(renderer, 200, 100, 100, 255);
	//for (int i = 0; i < 40; i++) {
	//	SDL_RenderDrawPoint(renderer, 20 * i + 200 + 10.f * sinf(iTime), 20 * i + 200 + 10.f * cosf(iTime));
	//}

	gameWindow->renderWindow();
}