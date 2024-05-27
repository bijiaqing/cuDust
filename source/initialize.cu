#include "cudust.cuh"

// =========================================================================================================================

__global__ 
void particle_init (swarm *dev_particle, real *dev_prof_azi, real *dev_prof_rad, real *dev_prof_col)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= 0 && idx < NUM_PAR)
    {
        real azi = dev_prof_azi[idx];
        real rad = dev_prof_rad[idx];
        real col = dev_prof_col[idx];
        
        dev_particle[idx].position.x = azi;
        dev_particle[idx].position.y = rad;
        dev_particle[idx].position.z = col;
    }
}

// =========================================================================================================================

__global__ 
void velocity_init (swarm *dev_particle, real *dev_optdepth)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if(idx >= 0 && idx < NUM_PAR)
    {
        dev_particle[idx].velocity.x = sqrt(G*M_STAR*dev_particle[idx].position.y);
        dev_particle[idx].velocity.y = 0.0;
        dev_particle[idx].velocity.z = 0.0;
    }
}
