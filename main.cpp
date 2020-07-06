#define TEST_PARTICLES true // Defines if we should have our Dam Break starting particles, for testing

#include <stdio.h>
#include <iostream>
#include <memory>

#include "window.h"
#include "gfxHelper.h"
#include "solver.cuh"

std::shared_ptr<window> gameWindow;

std::unique_ptr<solver> sphSolver;

int mouseX, mouseY;

void initialize();
void update();
void draw();
void addParticleSquare(float x, float y, int numberOfParticles, float spacing, float particleMass = DEFAULT_MASS);

// Program Starting Point
int main(int args, char* argv[])
{
	initialize();

	SDL_Event* e;
	while (gameWindow->checkIfRunning())
	{
		while (SDL_PollEvent(e))
		{
			if ((e->type == SDL_KEYDOWN && e->key.keysym.sym == SDLK_ESCAPE) || e->type == SDL_QUIT)
			{
				gameWindow->stopWindow();
				break;
			}


			if (e->type == SDL_KEYDOWN && e->key.keysym.sym == SDLK_r)
				sphSolver->clearParticles();

			if (e->type == SDL_MOUSEMOTION || e->type == SDL_MOUSEBUTTONDOWN)
			{
				SDL_GetMouseState(&mouseX, &mouseY);

				if (e->type == SDL_MOUSEBUTTONDOWN)
				{
					if(e->button.button == SDL_BUTTON_LEFT)
						addParticleSquare(mouseX, mouseY, 4, DEFAULT_MASS*.5);
					if(e->button.button == SDL_BUTTON_RIGHT)
						addParticleSquare(mouseX, mouseY, 4, DEFAULT_MASS, 5 * DEFAULT_MASS);
				}
			}

		}

		// If the game window is no longer running, then we break from the game loop
		if (!gameWindow->checkIfRunning())
			break;

		update();

		draw();
	}

	return 0;
}

// Handle all initialization logic
void initialize()
{
	// Create and get a reference to a window object
	gameWindow = std::make_shared<window>("SPH CUDA Flowmap - Hunter Werenskjold", WINDOW_WIDTH, WINDOW_HEIGHT);

	// Do the same as above for a solver object
	sphSolver = std::make_unique<solver>();

#if TEST_PARTICLES
	// This is to add all the starting particles for our simulation
	for (int i = 0; i < 30; i++)
		for (int j = 0; j < 28; j++)
			sphSolver->addParticle(vector2f(0.1 + (i * 0.03), 0.1 + (j * 0.03)));
#endif

	std::cout << "Particles Added: " << sphSolver->currentParticles << std::endl;
}

// Handle all update logic
void update()
{
	sphSolver->update();
}


// Handle all draw step logic
void draw()
{
	// Gets a temporary pointer to the current SDL_Renderer
	SDL_Renderer* renderer = gameWindow->getRenderer();

	for (int x = 0; x < sphSolver->currentParticles; x++)
	{
		// Draw particles
		if (sphSolver->particles[x].identifier == -1)
			continue;

		// Set the current color to draw with
		SDL_SetRenderDrawColor(renderer, 220, 220, 255, 255);

		// Draw a single point/pixel at the given coordinates
		SDL_RenderDrawPoint(renderer, sphSolver->particles[x].position.x * WINDOW_WIDTH, sphSolver->particles[x].position.y * WINDOW_HEIGHT);
	}

	// Rasterize and draw the window
	gameWindow->renderWindow();
}

/*
	Adds a "solid" square of particles centered at the the given x, y coordinates
	@param x X-axis coordinate
	@param y Y-axis coordinate
	@param numberOfParticles The total number of particles on one axis of the square, this give numberOfParticles^2 particles total
	@param spacing The spacing between each particle in the square
	@param particleMass The mass each particle will have
*/
void addParticleSquare(float x, float y, int numberOfParticles, float spacing, float particleMass)
{
	// Convert from screen to world space
	float worldCenterX = x / WINDOW_WIDTH;
	float worldCenterY = y / WINDOW_HEIGHT;
	int loopBounds = numberOfParticles / 2;

	for(int i = -loopBounds; i < loopBounds; i++)
		for (int j = -loopBounds; j < loopBounds; j++)
		{
			float offsetX = i * spacing;
			float offsetY = j * spacing;
			sphSolver->addParticle(particleMass, vector2f(worldCenterX + offsetX, worldCenterY + offsetY));
		}
}