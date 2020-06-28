#include <stdio.h>
#include <iostream>
#include <memory>

#include "window.h"
#include "gfxHelper.h"
#include "solver.cuh"

std::shared_ptr<window> gameWindow;
float iTime = 0;

std::unique_ptr<solver> sphSolver;

int mouseX, mouseY;

void initialize();
void update();
void draw();
void addParticleSquare(float x, float y, int numberOfParticles, float spacing, float particleMass = DEFAULT_MASS);

int main(int args, char* argv[])
{
	gameWindow = std::make_shared<window>("SPH CUDA Flowmap - Hunter Werenskjold", WINDOW_WIDTH, WINDOW_HEIGHT);

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

		iTime += DELTA_TIME;
	}

	return 0;
}

void initialize()
{
	sphSolver = std::make_unique<solver>();
	sphSolver->initialize();

	for (int i = 0; i < 30; i++)
		for (int j = 0; j < 28; j++)
			sphSolver->addParticle(vector2f(0.1 + (i * 0.03), 0.1 + (j * 0.03)));

	std::cout << "Particles Added: " << sphSolver->currentParticles << std::endl;

}

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
	}

	gameWindow->renderWindow();
}

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