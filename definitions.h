#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define DELTA_TIME 0.003

// Must be a power of 2
// 2 ^ 15 
// 1024 blocks
// 32 threads 
#define MAX_PARTICLES 32768

// Constants
#define KERNEL 0.06				// 0.04 (default)			
#define DEFAULT_MASS 0.05		// 0.01 (default)
#define REST_DENSITY 1000.0		// 1000.0 (default)
#define GAS_CONSTANT 1.0		// 1.0 (default)
#define HSQ KERNEL * KERNEL		
#define VISCOSITY 6.5			// 6.5 (default)
#define DT 0.003				// 0.003 (default)

#define GRAVITY_X 0.0			// I often use -9.8 here to corner the water and have to use fewer particles to see results
#define GRAVITY_Y 4.5			// -9.8 (default)

// SPH Kernels
#define POLY6 315.0/(64.0 * PI * (KERNEL*KERNEL*KERNEL*KERNEL*KERNEL*KERNEL*KERNEL*KERNEL*KERNEL))
#define SPIKY_GRAD -45.0 / (PI*(KERNEL*KERNEL*KERNEL*KERNEL*KERNEL*KERNEL))
#define VISC_LAP 45.0 / (PI*(KERNEL*KERNEL*KERNEL*KERNEL*KERNEL*KERNEL))

#define SELF_DENSITY_WO_MASS POLY6 * (KERNEL*KERNEL*KERNEL*KERNEL*KERNEL*KERNEL)

// Simulation
#define BOUNDARY 0.0005			// 0.005 (default)
#define BOUND_DAMPINING -0.8	// -0.5 (default)

// Grid 
#define GRID_XY(X,Y) (Y*GRID_WIDTH) + X
#define GRID_WIDTH 10				
#define GRID_SIZE GRID_WIDTH * GRID_WIDTH 	// Number of cells
#define GRID_TRANSFORM 1.0/GRID_WIDTH		// This is a basis, so multiply by cell number to get the location of the cell

#define MAX_THREADS 32

#endif