#include "solver.cuh"

/// Kernel Functions -- Device Code

// This calculates the discrete density field
__global__ void calculateDensityPressure(particle* pArray, int currentParticles)
{
	// The array index this thread is operating on
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	// If the particle being addressed is unused, don't evaluate
	if (pArray[i].identifier == -1)
		return;

	if (i < currentParticles)
	{
		// Density will need to be recalculated for each particle once a pass/frame
		pArray[i].density = 0.0f;
	
		// Density is calculated as a summation of all surrounding particles including self
		for (int j = 0; j < currentParticles; j++)
		{
			// we can't break since we might have destroyed a particle at any point in the array
			if (pArray[j].identifier == -1)
				continue;
			
			// Get the distance to current observed neighbor particle
			float rx = pArray[i].position.x - pArray[j].position.x;
			float ry = pArray[i].position.y - pArray[j].position.y;
			float r2 = rx * rx + ry * ry;

			// If the other particle is too far or far too close, we continue
			if (r2 >= HSQ || r2 < 1e-12)
				continue;

			// We use the constant POLY6 to help build a smoothing kernel
			pArray[i].density += pArray[j].mass * POLY6 * pow(HSQ - r2, 3);
		}

		pArray[i].density += pArray[i].mass * SELF_DENSITY_WO_MASS;
		pArray[i].pressure = (pow(pArray[i].density / REST_DENSITY, 7) - 1) * GAS_CONSTANT;
	}
}

// This calculates the forces on each particle by all surrounding neighbors
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
			// Skip calculating against self
			if (i == j || pArray[j].identifier == -1)
				continue;

			float r0x = pArray[i].position.x - pArray[j].position.x;
			float r0y = pArray[i].position.y - pArray[j].position.y;

			float r2 = r0x * r0x + r0y * r0y;

			// If the other particle is within distance
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
	// It's best to synchronize the data before you free the memory
	cudaDeviceSynchronize();
	cudaFree(ptr);
}

solver::solver()
{
	// We allocate the memory on the host and device as managed memory, this allows us to simply synchronize the two
	cudaMallocManaged(&particles, MAX_PARTICLES * sizeof(particle));

	// We fill in empty values for all particles
	for (int x = 0; x < MAX_PARTICLES; x++)
	{
		particles[currentParticles].position = 0;
		particles[currentParticles].velocity = 0;
		particles[currentParticles].mass = 0;
		particles[currentParticles].identifier = -1;
	}

	cudaDeviceSynchronize();

	initialize();
}

solver::~solver()
{
	// We must free the memory of all particle objects on deletion of a solver class instance
	for (int i = 0; i < MAX_PARTICLES; i++) {
		delete& particles[i];
	}
}

void solver::initialize()
{
	// We will use 0->1 to indicate the position along each axis in the simulation space
	// This simplification makes scaling far easier as well as keeping all screen space to simulation space calculations far more readable
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
	
	// We must calculate the number of blocks each update depending on the number of current particles
	// This will let us allocate the proper amount of blocks of cores to use in our device code
	blocks = (currentParticles + (MAX_THREADS - 1)) / MAX_THREADS;

	// Functions in this format are handled by the NVCC
	// <<<# of blocks, # of threads>>>
	calculateDensityPressure <<<blocks, MAX_THREADS>>>(particles, currentParticles);

	calculateForces <<<blocks, MAX_THREADS>>>(particles, currentParticles);

	// This synchronizes "managed"/"shared" memory between the GPU and RAM
	// This will also prevent race conditions
	// Do note we only need to do this once and not between the two CUDA functions since the memory is not accessed on the host side until integration
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
	for (int i = 0; i < currentParticles; i++)
		particles->identifier = -1;

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

		// Integrate: Calculate new velocity and then position
		particles[i].velocity += particles[i].acceleration * DT + gravity * DT;
		particles[i].position += particles[i].velocity * DT;

		// Particle Boundary Conditions //
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

		// Update cell information
		if (particles[i].position.x > 0.f && particles[i].position.x <= worldSize_width && particles[i].position.y > 0.f && particles[i].position.y <= worldSize_height)
		{
			cell* c = getCell(particles[i].position);
			c->velocity = (c->velocity + particles[i].velocity) / 2.f;
		}

		// This is required for calculating stable and near accurate forces during the calculateForces(...) step
		particles[i].evelocity = (particles[i].evelocity + particles[i].velocity) / 2.f;
	}
}