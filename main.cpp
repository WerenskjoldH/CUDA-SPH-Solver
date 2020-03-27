#include <stdio.h>
#include <iostream>
#include "window.h"
#include "gfxHelper.h"
#include "solver.cuh"

window* gameWindow;
float iTime = 0;

solver* sphSolver;

void initialize();
void inputs();
void update();
void draw();

int main(int args, char* argv[])
{
	gameWindow = new window("SPH CUDA Flowmap - Hunter Werenskjold", WINDOW_WIDTH, WINDOW_HEIGHT);

	initialize();
	
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

void initialize()
{
	sphSolver = new solver();
	sphSolver->initialize();

	for (int i = 0; i < 30; i++)
		for(int j = 0; j < 28; j++)
			sphSolver->addParticle(vector2f(0.1 + (i * 0.03), 0.1 + (j * 0.03)));

	std::cout << sphSolver->currentParticles << std::endl;
	
}

void inputs()
{}

void update()
{
	sphSolver->update();
}

void draw()
{
	SDL_Renderer* renderer = gameWindow->getRenderer();

	for (int x = 0; x < sphSolver->currentParticles; x++)
	{
		// Draw particles
		if (sphSolver->particles[x].identifier == -1)
			continue;

		SDL_SetRenderDrawColor(renderer, 220, 220, 255, 255);

		SDL_RenderDrawPoint(renderer, sphSolver->particles[x].position.x * WINDOW_WIDTH, sphSolver->particles[x].position.y * WINDOW_HEIGHT);
		//gfxDrawBrenCircle(renderer, sphSolver->particles[x].position.x * WINDOW_WIDTH, sphSolver->particles[x].position.y * WINDOW_HEIGHT, 10, true);
	}

	gameWindow->renderWindow();
}