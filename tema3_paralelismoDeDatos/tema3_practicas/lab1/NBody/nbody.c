#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "nbody.h"
#include "gtime.h"
#include "memory.h"
#include "nbody_routines.h"

/*
double solutionPos(body *p, int n)
{
	float pos_private = 0.0f;
	float pos_global  = 0.0f;

	
	for (int i = 0 ; i < n; i++) {
		pos_private = sqrtf(p->x[i]*p->x[i]+p->y[i]*p->y[i]+p->z[i]*p->z[i]);
		pos_global+=pos_private;
	}
	return(pos_global);
}


 int main(const int argc, const char** argv) {
	int nBodies = 1000;
	if (argc > 1) nBodies = atoi(argv[1]);

	const float dt = 0.01f; // time step
	const int nIters = 100;  // simulation iterations
	body *p = get_memory(nBodies);

	randomizeBodies(p, nBodies); // Init pos / vel data
	double t0 = get_time();
	for (int iter = 1; iter <= nIters; iter++) {
		bodyForce(p, dt, nBodies); // compute interbody forces
		integrate(p, dt, nBodies); // integrate position
	}

	double totalTime = get_time()-t0;
	double solPos = solutionPos(p, nBodies);
	printf("%d Bodies with %d iterations: %0.3f Millions Interactions/second\n", nBodies, nIters, 1e-6 * nBodies * nBodies / totalTime);
	printf("pos=%e\n", solPos);

	free_memory(p);
}
 */



	//PRUEBA
//*****************************************************************************************

int main(const int argc, const char** argv) {
	int nBodies = 1000;
	int alignment=32;
	if (argc > 1) nBodies = atoi(argv[1]);

	const float dt = 0.01f; // time step
	const int nIters = 100;  // simulation iterations
 	body p;
	p.m = (float*)_mm_malloc(sizeof(float)*nBodies, alignment);
	p.x = (float*)_mm_malloc(sizeof(float)*nBodies, alignment);
	p.y = (float*)_mm_malloc(sizeof(float)*nBodies, alignment);
	p.z = (float*)_mm_malloc(sizeof(float)*nBodies, alignment);
	p.vx = (float*)_mm_malloc(sizeof(float)*nBodies, alignment);
	p.vy = (float*)_mm_malloc(sizeof(float)*nBodies, alignment);
	p.vz = (float*)_mm_malloc(sizeof(float)*nBodies, alignment);

	for (int i = 0; i < nBodies; i++) {
		p.m[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		p.x[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		p.y[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		p.z[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		p.vx[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		p.vy[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		p.vz[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
	}


	float softeningSquared = 1e-3;
	float G = 6.674e-11;
	double t0 = get_time();
	for (int iter = 1; iter <= nIters; iter++) {

		for (int i = 0; i < nBodies; i++) { 
			float Fx = 0.0f;
			float Fy = 0.0f;
			float Fz = 0.0f;
			float tempX = p.x[i];
			float tempY = p.y[i];
			float tempZ = p.z[i];
			float tempM = p.m[i];

			//#pragma vector aligned
			#pragma loop_count max(1000)
			#pragma omp simd
			for (int j = 0; j < nBodies; j++) {
				if (j!=i) {
					float dx = p.x[j] - tempX;
					float dy = p.y[j] - tempY;
					float dz = p.z[j] - tempZ;

					float distSqr = dx*dx + dy*dy + dz*dz + softeningSquared;
					float invDist = 1.0f / sqrtf(distSqr);
					float invDist3 = invDist * invDist * invDist;
					float g_masses = G * tempM * tempM *invDist3;

					Fx += g_masses * dx; 
					Fy += g_masses * dy; 
					Fz += g_masses * dz;
				}
			}
			p.vx[i] += dt*Fx/tempM;
			p.vy[i] += dt*Fy/tempM;
			p.vz[i] += dt*Fz/tempM;
		}

		for (int i = 0 ; i < nBodies; i++) {
			p.x[i] += p.vx[i]*dt;
			p.y[i] += p.vy[i]*dt;
			p.z[i] += p.vz[i]*dt;
		}
	}

	double totalTime = get_time()-t0;
	float pos_private = 0.0f;
	float pos_global  = 0.0f;

	for (int i = 0 ; i < nBodies; i++) {
		pos_private = sqrtf(p.x[i]*p.x[i]+p.y[i]*p.y[i]+p.z[i]*p.z[i]);
		pos_global+=pos_private;
	}


	printf("%d Bodies with %d iterations: %0.3f Millions Interactions/second\n", nBodies, nIters, 1e-6 * nBodies * nBodies / totalTime);
	printf("pos=%e\n", pos_global);

 	_mm_free(p.m);
	_mm_free(p.x);
	_mm_free(p.y);
	_mm_free(p.z);
	_mm_free(p.vx);
	_mm_free(p.vy);
	_mm_free(p.vz);
}



