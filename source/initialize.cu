#include "cudust.cuh"

// =========================================================================================================================

__global__ 
void particle_init (swarm *dev_particle, real *dev_prof_azi, real *dev_prof_rad, real *dev_prof_col)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= 0 && idx < NUM_PAR)
    {
        // dev_particle[idx].position.x = dev_prof_azi[idx];
        // dev_particle[idx].position.y = dev_prof_rad[idx];
        // dev_particle[idx].position.z = dev_prof_col[idx];

        dev_particle[idx].position.x = 0.25*M_PI;
        dev_particle[idx].position.y = 1.0;
        dev_particle[idx].position.z = 0.25*M_PI;
    }
}

// =========================================================================================================================

__global__ 
void dynamics_init (swarm *dev_particle, real *dev_optdepth)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if(idx >= 0 && idx < NUM_PAR)
    {
        dev_particle[idx].dynamics.x = sqrt(G*M_REF*dev_particle[idx].position.y)*sin(dev_particle[idx].position.z);
        dev_particle[idx].dynamics.y = 0.0;
        dev_particle[idx].dynamics.z = 0.0;
    }
}