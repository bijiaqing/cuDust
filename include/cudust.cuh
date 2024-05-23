#ifndef FUNCLIB_CUH 
#define FUNCLIB_CUH

#include <fstream>  // for std::ofstream, std::ifstream
#include <random>   // for std::mt19937

#include "const.cuh"
#include "curand_kernel.h"

// =========================================================================================================================
// mesh calculation

__global__ void optdepth_init (real *dev_optdepth);
__global__ void optdepth_enum (real *dev_optdepth, swarm *dev_particle);
__global__ void optdepth_calc (real *dev_optdepth);
__global__ void optdepth_rint (real *dev_optdepth);
__global__ void optdepth_mean (real *dev_optdepth);

__global__ void dustdens_init (real *dev_dustdens);
__global__ void dustdens_enum (real *dev_dustdens, swarm *dev_particle);
__global__ void dustdens_calc (real *dev_dustdens);

// __global__ void histsize_init (real *dev_histsize);
// __global__ void histsize_enum (real *dev_histsize, swarm *dev_particle);

// =========================================================================================================================
// initialization 

__global__ void particle_init (swarm *dev_particle, real *dev_prof_azi, real *dev_prof_rad, real *dev_prof_col, real *dev_prof_size);
__global__ void velocity_init (swarm *dev_particle, real *dev_optdepth);

// =========================================================================================================================
// collision related

__global__ void rngstate_init (curandState *dev_rngstate, int seed);
__global__ void treenode_init (swarm *dev_particle, tree *dev_treenode);
__global__ void collrate_calc (swarm *dev_particle, swarm_tmp *dev_tmp_info, tree *dev_treenode, int  *dev_collrate, 
    const cukd::box_t<float3> *dev_boundbox);
__global__ void dustcoag_calc (swarm *dev_particle, swarm_tmp *dev_tmp_info, tree *dev_treenode, real *dev_timestep, 
    const cukd::box_t<float3> *dev_boundbox, curandState *dev_rngstate);
__global__ void dustsize_updt (swarm *dev_particle, swarm_tmp *dev_tmp_info);

// =========================================================================================================================
// integrator

__global__ void ssa_substep_1 (swarm *dev_particle, real *dev_timestep);
__global__ void ssa_substep_2 (swarm *dev_particle, real *dev_optdepth, real *dev_timestep);

// =========================================================================================================================
// interpolation

__device__ interp linear_interp_cent (real par_azi, real par_rad, real par_col);
__device__ interp linear_interp_stag (real par_azi, real par_rad, real par_col);

__device__ real get_optdepth (real *dev_optdepth, real par_azi, real par_rad, real par_col);

// =========================================================================================================================
// files open and save

__host__ std::string frame_num (int number, std::size_t length = 5);

__host__ void open_txt_file (std::ofstream &txt_file, std::string file_name);
__host__ void save_variable (std::ofstream &txt_file);

__host__ void open_bin_file (std::ofstream &bin_file, std::string file_name);
__host__ void save_bin_file (std::ofstream &bin_file, swarm *data, int number);
__host__ void save_bin_file (std::ofstream &bin_file, float *data, int number);
__host__ void save_bin_file (std::ofstream &bin_file, real  *data, int number);

__host__ void load_bin_file (std::ifstream &bin_file, std::string file_name);
__host__ void read_bin_file (std::ifstream &bin_file, swarm *data, int number);

// =========================================================================================================================
// profile generators

extern std::mt19937 rand_generator;

__host__ void rand_uniform  (real *profile, int number, real p_min, real p_max);
__host__ void rand_gaussian (real *profile, int number, real p_min, real p_max, real p_0, real sigma);
__host__ void rand_powerlaw (real *profile, int number, real p_min, real p_max, real idx_pow);
__host__ void rand_conv_pow (real *profile, int number, real p_min, real p_max, real idx_pow, real smooth, int bins);

// =========================================================================================================================

#endif
