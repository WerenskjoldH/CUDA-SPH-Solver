#ifndef SOLVER_H
#define SOLVER_H

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector2f.h"
#include "definitions.h"

struct particle
{
	// Use for dynamic particles
	particle(vector2f position, vector2f velocity, int id);

	// Use for static particles
	particle(vector2f position, int id);

	// The delete operation is overriden since particles are shared between Host and Device code
	// This means the memory must be synchronized before being freed making it safer to simply override the delete operator
	void operator delete(void* ptr);

	int identifier;

	float mass;
	float density;
	float pressure;

	vector2f position;
	vector2f velocity;
	vector2f evelocity;
	vector2f acceleration;
};

// Cells were implemented alongside everything else for use in exporting flowmaps and future optimizations ( I.E. Quad Trees )
// Cells (can) store regional/discrete data in the scene but also sum average velocity as a side purpose
struct cell
{
	cell(vector2f position)
	{
		this->position = position;
		velocity = 0.f;
	}

	vector2f position;
	vector2f velocity;
};

// The Solver handles all logic related to the Smoothed-Particle Hydrodynamics(SPH) simulation
// This ONLY handles the simulation and does not handle any rendering or output
class solver
{
public:
	// Constructor & Deconstructor //
	solver();
	~solver();

	/// Methods ///

	// Procedures //

	/* All initialization logic, this is called internally at the end of the constructor */
	void initialize();

	/* All simulation integration and per-frame logic is handled here */
	void update();

	/**
		Creates a particle at the given position with default mass
		@param pos Position to place the particle in simulation space coordinates
		@param vel Starting velocity of the particle
	*/ 
	void addParticle(vector2f pos, vector2f vel = vector2f(0.0f, 0.0f));

	/**
		Creates a particle at the given position with given mass
		@param mass The mass of the particle
		@param pos Position to place the particle in simulation space coordinates
		@param vel Starting velocity of the particle
	*/
	void addParticle(float mass, vector2f pos, vector2f vel = vector2f(0.0f, 0.0f));

	/* Sets the current number of particles to 0, do keep in mind this does not free/clear memory */
	void clearParticles();

	/* Create a wall particle -- NOT IMPLEMENTED */
	void addWall(vector2f pos);

	/* Create an outlet particle -- NOT IMPLEMENTED */
	void addOutlet(vector2f pos);

	// Functions //

	/** Gets a cell from simulation space coordinates 
		@param pos Simulation space coordinates
		@return Pointer to cell at given simulation space coordinates
	*/
	cell* getCell(vector2f pos);

	/// Variables ///

	particle* particles;
	cell* grid[GRID_SIZE];

	float worldSize_width;
	float worldSize_height;

	int currentParticles = 0;

	// This is calculated for us during update and relates to the number of blocks of threads CUDA will use
	int blocks = 1;

private:
	/// Methods ///

	// Procedures //
	void integration();


	/// Variables ///
	vector2f gravity;

};

#endif SOLVER_H