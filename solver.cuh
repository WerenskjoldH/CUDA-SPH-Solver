#ifndef SOLVER_H
#define SOLVER_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector2f.h"
#include "definitions.h"

#include <stdio.h>

struct particle
{
	// Use for dynamic particles
	particle(vector2f position, vector2f velocity, int id);

	// Use for static - Currently Not Re-implemented
	particle(vector2f position, int id);

	int identifier;

	float mass;
	float density;
	float pressure;

	vector2f position;
	vector2f velocity;
	vector2f evelocity;
	vector2f acceleration;
};


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

class solver
{
public:
	// Constructor & Deconstructor //
	solver();
	~solver();

	/// Methods ///
	// Procedures
	void initialize();
	void update();
	void addParticle(vector2f pos, vector2f vel = vector2f(0.0f, 0.0f));
	void addParticle(float mass, vector2f pos, vector2f vel = vector2f(0.0f, 0.0f));
	void clearParticles();

	void addWall(vector2f pos);

	void addOutlet(vector2f pos);

	// Functions
	cell* getCell(vector2f pos);

	/// Variables ///
	particle* particles;
	cell* grid[GRID_SIZE];

	float worldSize_width;
	float worldSize_height;

	int currentParticles = 0;
	int blocks = 1;

private:
	/// Methods ///
	// Procedures
	void integration();

	// Functions


	/// Variables ///
	vector2f gravity;

};

#endif SOLVER_H