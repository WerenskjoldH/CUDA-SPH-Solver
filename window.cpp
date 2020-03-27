#include "window.h"

window::window(const char title[], int windowWidth, int windowHeight) : windowWidth{ windowWidth }, windowHeight{ windowHeight }
{
	this->windowWidth = windowWidth;
	this->windowHeight = windowHeight;

	// Create Window and Begin
	if (SDL_Init(SDL_INIT_EVERYTHING) == 0)
		std::cout << "Subsystems Initialized" << std::endl;
	else
	{
		std::cout << "ERROR::FATAL::Subsystems FAILED to initalize" << std::endl;
		return;
	}

	sdlWindow = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, windowWidth, windowHeight, SDL_WINDOW_SHOWN);
	if (sdlWindow)
		std::cout << "Window Created" << std::endl;
	else
	{
		std::cout << "ERROR::FATAL::Window FAILED to be created" << std::endl;
		return;
	}

	sdlRenderer = SDL_CreateRenderer(sdlWindow, -1, 0);

	if (sdlRenderer)
		std::cout << "Renderer Created" << std::endl;
	else
	{
		std::cout << "ERROR::FATAL::Subsystems FAILED to be created" << std::endl;
		return;
	}

	SDL_SetRenderDrawColor(sdlRenderer, bgColor.r, bgColor.g, bgColor.b, bgColor.a);
	SDL_RenderClear(sdlRenderer);

	isRunning = 1;
}

window::~window()
{
	SDL_DestroyWindow(sdlWindow);
	SDL_DestroyRenderer(sdlRenderer);
	SDL_Quit();
}

bool 
window::checkIfRunning() const
{
	return isRunning;
}

SDL_Renderer* 
window::getRenderer()
{
	return sdlRenderer;
}

void 
window::setBackgroundColor(const rgbColor c)
{
	// Maybe add boundary checks to this?
	bgColor = c;
}

void 
window::stopWindow()
{
	isRunning = false;
}

void 
window::renderWindow()
{
	SDL_RenderPresent(sdlRenderer);
	SDL_SetRenderDrawColor(sdlRenderer, bgColor.r, bgColor.g, bgColor.b, bgColor.a);
	SDL_RenderClear(sdlRenderer);
}
