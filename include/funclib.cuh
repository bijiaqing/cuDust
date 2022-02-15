#ifndef FUNCLIB_CUH 
#define FUNCLIB_CUH

#include "const.h"

//====================================================================================================================================================================
// Density & optical depth

__global__ void optdept_init (double *dev_optdept);
__global__ void optdept_enum (double *dev_optdept, par *dev_grain);
__global__ void optdept_calc (double *dev_optdept);
__global__ void optdept_cumu (double *dev_optdept);
__global__ void optdept_mean (double *dev_optdept);

__global__ void density_init (double *dev_density, double *dev_lStokes, double *dev_hStokes);
__global__ void density_enum (double *dev_density, double *dev_lStokes, double *dev_hStokes, par *dev_grain);
__global__ void density_calc (double *dev_density, double *dev_lStokes, double *dev_hStokes);

__global__ void stepnum_calc (double *dev_stepnum, par *dev_grain);

//====================================================================================================================================================================
// Initialization 

__global__ void position_init (par *dev_grain, double *dev_prof_azi, double *dev_prof_rad, double *dev_prof_col, double *dev_prof_siz);
__global__ void velocity_init (par *dev_grain, double *dev_stepnum);

//====================================================================================================================================================================
// Integrator

__global__ void parmove_act1 (par *dev_grain);
__global__ void parmove_act2 (par *dev_grain, double *dev_optdept);

//====================================================================================================================================================================
// Device functions

__device__ interp linear_interp_cent (double par_azi, double par_rad, double par_col);
__device__ interp linear_interp_stag (double par_azi, double par_rad, double par_col);

__device__ double get_optdept (double par_azi, double par_rad, double par_col, double *dev_optdept);

//====================================================================================================================================================================

#endif
