#include "solver.cuh"

/// Kernel Functions
__global__ void calculateDensityPressure(particle* pArray, int currentParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (pArray[i].identifier == -1)
		return;

	if (i < currentParticles)
	{

		pArray[i].density = 0.0f;

		for (int j = 0; j < currentParticles; j++)
		{
			if (pArray[j].identifier == -1)
				continue;

			float rx = pArray[i].position.x - pArray[j].position.x;
			float ry = pArray[i].position.y - pArray[j].position.y;
			float r2 = rx * rx + ry * ry;

			if (r2 >= HSQ || r2 < 1e-12)
				continue;

			pArray[i].density += pArray[j].mass * POLY6 * pow(HSQ - r2, 3);
		}

		pArray[i].density += pArray[i].mass * SELF_DENSITY_WO_MASS;
		pArray[i].pressure = (pow(pArray[i].density / REST_DENSITY, 7) - 1) * GAS_CONSTANT;
	}
}

__global__ void calculateForces(particle* pArray, int currentParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (pArray[i].identifier == -1)
		return;

	if (i < currentParticles)
	{
		pArray[i].acceleration.x = 0.f;
		pArray[i].acceleration.y = 0.f;


		for (int j = 0; j < currentParticles; j++)
		{
			if (i == j || pArray[j].identifier == -1)
				continue;

			float r0x = pArray[i].position.x - pArray[j].position.x;
			float r0y = pArray[i].position.y - pArray[j].position.y;

			float r2 = r0x * r0x + r0y * r0y;

			if (r2 < HSQ && r2 > 1e-12)
			{
				// Removal of particles - Currently In Testing Branch
				//if (pArray[j].pType == OUTLET)
				//{
				//	pArray[i].identifier = -1;
				//	continue;
				//}

				float r = sqrt(r2);

				float V = pArray[j].mass / pArray[j].density / 2;
				float Kr = KERNEL - r;
				float Kp = SPIKY_GRAD * Kr * Kr;

				float tempForce = V * (pArray[i].pressure + pArray[j].pressure) * Kp;
				pArray[i].acceleration.x -= r0x * tempForce / r;
				pArray[i].acceleration.y -= r0y * tempForce / r;

				float rVx = pArray[j].evelocity.x - pArray[i].evelocity.x;
				float rVy = pArray[j].evelocity.y - pArray[i].evelocity.y;

				float Kv = VISC_LAP * (KERNEL - r);
				tempForce = V * VISCOSITY * Kv;
				pArray[i].acceleration.x += rVx * tempForce;
				pArray[i].acceleration.y += rVy * tempForce;
			}
		}
		pArray[i].acceleration.x = pArray[i].acceleration.x / pArray[i].density;
		pArray[i].acceleration.y = pArray[i].acceleration.y / pArray[i].density;

	}
}

/// Class Function Definitions

particle::particle(vector2f position, vector2f velocity, int id) : position{ position }, velocity{ velocity }, identifier{ id }
{
	mass = DEFAULT_MASS;
	density = REST_DENSITY;
	pressure = 0.f;

	acceleration = 0.f;
}

// Use for static particles - Currently Not Re-implemented
particle::particle(vector2f position, int id) : position{ position }, identifier{ id }
{
	mass = DEFAULT_MASS;
	density = REST_DENSITY;
	pressure = 0.f;

	acceleration = 0.f;
}

void particle::operator delete(void* ptr)
{
	cudaDeviceSynchronize();
	cudaFree(ptr);
}

solver::solver()
{
	cudaMallocManaged(&particles, MAX_PARTICLES * sizeof(particle));

	for (int x = 0; x < MAX_PARTICLES; x++)
	{
		particles[currentParticles].position = 0;
		particles[currentParticles].velocity = 0;
		particles[currentParticles].mass = 0;
		particles[currentParticles].identifier = -1;
	}

	initialize();
}

solver::~solver()
{
	for (int i = 0; i < MAX_PARTICLES; i++) {
		delete& particles[i];
	}
}

void solver::initialize()
{
	worldSize_width = 1.0f;
	worldSize_height = 1.0f;

	for (int x = 0; x < GRID_WIDTH; x++)
		for (int y = 0; y < GRID_WIDTH; y++)
			grid[GRID_XY(x, y)] = new cell(vector2f(x, y));

	gravity.x = GRAVITY_X;
	gravity.y = GRAVITY_Y;
}

// This is where we apply all steps of integration and update the simulation each step
void solver::update()
{
	int blocks;
	
	blocks = (currentParticles + (MAX_THREADS - 1)) / MAX_THREADS;

	calculateDensityPressure <<<blocks, MAX_THREADS>>>(particles, currentParticles);

	calculateForces <<<blocks, MAX_THREADS>>>(particles, currentParticles);

	cudaDeviceSynchronize();

	integration();
}


void solver::addParticle(vector2f pos, vector2f vel)
{
	if (currentParticles < MAX_PARTICLES)
	{
		particles[currentParticles] = particle(pos, vel, currentParticles);
		particles[currentParticles].mass = DEFAULT_MASS;
		currentParticles++;
	}
	else
		for (int i = 0; i < currentParticles; i++)
			if (particles[i].identifier == -1)
			{
				particles[i] = particle(pos, vel, currentParticles);
				particles[i].mass = DEFAULT_MASS;
				break;
			}

}

// Add particles of varying mass 
// Parameter ordering is strange, but that's because velocity has a default value of <0,0>
void solver::addParticle(float mass, vector2f pos, vector2f vel)
{
	if (currentParticles < MAX_PARTICLES)
	{
		particles[currentParticles] = particle(pos, vel, currentParticles);
		particles[currentParticles].mass = mass;
		currentParticles++;
	}
	else
		for (int i = 0; i < currentParticles; i++)
			if (particles[i].identifier == -1)
			{
				particles[currentParticles] = particle(pos, vel, currentParticles);
				particles[currentParticles].mass = mass;
				break;
			}
}


void solver::clearParticles()
{
	currentParticles = 0;
}

// Add a wall - Currently Not Re-implemented
void solver::addWall(vector2f pos)
{
}

// Add an outlet - Currently Not Re-implemented
void solver::addOutlet(vector2f pos)
{
}

cell* solver::getCell(vector2f pos)
{
	return grid[GRID_XY((int)(pos.x * GRID_WIDTH), (int)(pos.y * GRID_WIDTH))];
}

void solver::integration()
{
	for (int i = 0; i < currentParticles; i++)
	{
		if (particles[i].identifier == -1)
			continue;

		particles[i].velocity += particles[i].acceleration * DT + gravity * DT;
		particles[i].position += particles[i].velocity * DT;

		if (particles[i].position.x >= worldSize_width)
		{
			particles[i].velocity.x = particles[i].velocity.x * BOUND_DAMPINING;
			particles[i].position.x = worldSize_width - BOUNDARY;
		}

		if (particles[i].position.x < 0.f)
		{
			particles[i].velocity.x = particles[i].velocity.x * BOUND_DAMPINING;
			particles[i].position.x = BOUNDARY;
		}

		if (particles[i].position.y >= worldSize_height)
		{
			particles[i].velocity.y = particles[i].velocity.y * BOUND_DAMPINING;
			particles[i].position.y = worldSize_height - BOUNDARY;
		}

		if (particles[i].position.y < 0.f)
		{
			particles[i].velocity.y = particles[i].velocity.y * BOUND_DAMPINING;
			particles[i].position.y = BOUNDARY;
		}


		if (particles[i].position.x > 0.f && particles[i].position.x <= worldSize_width && particles[i].position.y > 0.f && particles[i].position.y <= worldSize_height)
		{
			cell* c = getCell(particles[i].position);
			c->velocity = (c->velocity + particles[i].velocity) / 2.f;
		}
		particles[i].evelocity = (particles[i].evelocity + particles[i].velocity) / 2.f;
	}
}