#include "const.h"
#include "funclib.cuh"

//====================================================================================================================================================================
//====================================================================================================================================================================

__global__
void optdept_init (double *dev_optdept)
{
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    	
    if (index >= 0 && index < RES_AZI*RES_RAD*RES_COL)
    {
        dev_optdept[index] = 0.0;
    }
}

__global__
void density_init (double *dev_density, double *dev_lStokes, double *dev_hStokes)
{
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    	
    if (index >= 0 && index < RES_AZI*RES_RAD*RES_COL)
    {
        dev_density[index] = 0.0;
        dev_lStokes[index] = 0.0;
        dev_hStokes[index] = 0.0;
    }
}

//====================================================================================================================================================================
//====================================================================================================================================================================

__global__ void optdept_enum (double *dev_optdept, par *dev_grain)
{
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    
    if (index >= 0 && index < PAR_NUM)
    {	
        int    next_azi, next_rad, next_col;
        int    idx_azi, idx_rad, idx_col, idx_cell;
        double par_azi, frac_azi;
        double par_rad, frac_rad;
        double par_col, frac_col;
        double weight = 0.0;

        interp interp_result;

      //weight = pow(dev_grain[index].siz / 1.0e-4, -1.0);
        weight = 1.0;
        
        par_azi = (dev_grain[index].azi - AZI_MIN) / ((AZI_MAX - AZI_MIN) / (double)RES_AZI);
        idx_azi = floor(par_azi); 
    
        par_rad = log(dev_grain[index].rad / RAD_MIN) / log(pow(RAD_MAX / RAD_MIN, 1.0 / (double)RES_RAD));
        idx_rad = floor(par_rad); 
    
        par_col = (dev_grain[index].col - COL_MIN) / ((COL_MAX - COL_MIN) / (double)RES_COL);
        idx_col = floor(par_col); 

        idx_cell = idx_col*RES_RAD*RES_AZI + idx_rad*RES_AZI + idx_azi;
        
        // this filters the grains that are inside the domain
        if (par_azi >= 0.0 && par_azi < RES_AZI && par_rad >= 0.0 && par_rad < RES_RAD && par_col >= 0.0 && par_col < RES_COL)
        {		
            interp_result = linear_interp_stag(par_azi, par_rad, par_col);
            
            next_azi = interp_result.next_azi;
            next_rad = interp_result.next_rad;
            next_col = interp_result.next_col;
            frac_azi = interp_result.frac_azi;
            frac_rad = interp_result.frac_rad;
            frac_col = interp_result.frac_col;
            
            atomicAdd(&dev_optdept[idx_cell                                 ], (1.0-frac_azi)*(1.0-frac_rad)*(1.0-frac_col)*weight);
            atomicAdd(&dev_optdept[idx_cell + next_azi                      ],      frac_azi *(1.0-frac_rad)*(1.0-frac_col)*weight);
            atomicAdd(&dev_optdept[idx_cell            + next_rad           ], (1.0-frac_azi)*     frac_rad *(1.0-frac_col)*weight);
            atomicAdd(&dev_optdept[idx_cell + next_azi + next_rad           ],      frac_azi *     frac_rad *(1.0-frac_col)*weight);
            atomicAdd(&dev_optdept[idx_cell                       + next_col], (1.0-frac_azi)*(1.0-frac_rad)*     frac_col *weight);
            atomicAdd(&dev_optdept[idx_cell + next_azi            + next_col],      frac_azi *(1.0-frac_rad)*     frac_col *weight);
            atomicAdd(&dev_optdept[idx_cell            + next_rad + next_col], (1.0-frac_azi)*     frac_rad *     frac_col *weight);
            atomicAdd(&dev_optdept[idx_cell + next_azi + next_rad + next_col],      frac_azi *     frac_rad *     frac_col *weight);
        }   
    } 
}

__global__ void density_enum (double *dev_density, double *dev_lStokes, double *dev_hStokes, par *dev_grain)
{
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    
    if (index >= 0 && index < PAR_NUM)
    {	
        int    next_azi, next_rad, next_col;
        int    idx_azi, idx_rad, idx_col, idx_cell;
        double par_azi, frac_azi;
        double par_rad, frac_rad;
        double par_col, frac_col;
        double weight = 0.0;

        interp interp_result;
        
        // in the case of superparticles having equal surface densities,
        // the mass of each superparticle goes proportional to the grain size.
        weight = dev_grain[index].siz / 1.0e-4;
        
        par_azi = (dev_grain[index].azi - AZI_MIN) / ((AZI_MAX - AZI_MIN) / (double)RES_AZI);
        idx_azi = floor(par_azi); 
    
        par_rad = log(dev_grain[index].rad / RAD_MIN) / log(pow(RAD_MAX / RAD_MIN, 1.0 / (double)RES_RAD));
        idx_rad = floor(par_rad); 
    
        par_col = (dev_grain[index].col - COL_MIN) / ((COL_MAX - COL_MIN) / (double)RES_COL);
        idx_col = floor(par_col); 

        idx_cell = idx_col*RES_RAD*RES_AZI + idx_rad*RES_AZI + idx_azi;
        
        // this filters the grains that are inside the domain
        if (par_azi >= 0.0 && par_azi < RES_AZI && par_rad >= 0.0 && par_rad < RES_RAD && par_col >= 0.0 && par_col < RES_COL)
        {		
            interp_result = linear_interp_cent(par_azi, par_rad, par_col);
            
            next_azi = interp_result.next_azi;
            next_rad = interp_result.next_rad;
            next_col = interp_result.next_col;
            frac_azi = interp_result.frac_azi;
            frac_rad = interp_result.frac_rad;
            frac_col = interp_result.frac_col;
            
            atomicAdd(&dev_density[idx_cell                                 ], (1.0-frac_azi)*(1.0-frac_rad)*(1.0-frac_col)*weight);
            atomicAdd(&dev_density[idx_cell + next_azi                      ],      frac_azi *(1.0-frac_rad)*(1.0-frac_col)*weight);
            atomicAdd(&dev_density[idx_cell            + next_rad           ], (1.0-frac_azi)*     frac_rad *(1.0-frac_col)*weight);
            atomicAdd(&dev_density[idx_cell + next_azi + next_rad           ],      frac_azi *     frac_rad *(1.0-frac_col)*weight);
            atomicAdd(&dev_density[idx_cell                       + next_col], (1.0-frac_azi)*(1.0-frac_rad)*     frac_col *weight);
            atomicAdd(&dev_density[idx_cell + next_azi            + next_col],      frac_azi *(1.0-frac_rad)*     frac_col *weight);
            atomicAdd(&dev_density[idx_cell            + next_rad + next_col], (1.0-frac_azi)*     frac_rad *     frac_col *weight);
            atomicAdd(&dev_density[idx_cell + next_azi + next_rad + next_col],      frac_azi *     frac_rad *     frac_col *weight);
            
            if (dev_grain[index].siz <= CRITICAL_SIZ)
            {
                atomicAdd(&dev_lStokes[idx_cell                                 ], (1.0-frac_azi)*(1.0-frac_rad)*(1.0-frac_col)*weight);
                atomicAdd(&dev_lStokes[idx_cell + next_azi                      ],      frac_azi *(1.0-frac_rad)*(1.0-frac_col)*weight);
                atomicAdd(&dev_lStokes[idx_cell            + next_rad           ], (1.0-frac_azi)*     frac_rad *(1.0-frac_col)*weight);
                atomicAdd(&dev_lStokes[idx_cell + next_azi + next_rad           ],      frac_azi *     frac_rad *(1.0-frac_col)*weight);
                atomicAdd(&dev_lStokes[idx_cell                       + next_col], (1.0-frac_azi)*(1.0-frac_rad)*     frac_col *weight);
                atomicAdd(&dev_lStokes[idx_cell + next_azi            + next_col],      frac_azi *(1.0-frac_rad)*     frac_col *weight);
                atomicAdd(&dev_lStokes[idx_cell            + next_rad + next_col], (1.0-frac_azi)*     frac_rad *     frac_col *weight);
                atomicAdd(&dev_lStokes[idx_cell + next_azi + next_rad + next_col],      frac_azi *     frac_rad *     frac_col *weight);
            }
            else
            {
                atomicAdd(&dev_hStokes[idx_cell                                 ], (1.0-frac_azi)*(1.0-frac_rad)*(1.0-frac_col)*weight);
                atomicAdd(&dev_hStokes[idx_cell + next_azi                      ],      frac_azi *(1.0-frac_rad)*(1.0-frac_col)*weight);
                atomicAdd(&dev_hStokes[idx_cell            + next_rad           ], (1.0-frac_azi)*     frac_rad *(1.0-frac_col)*weight);
                atomicAdd(&dev_hStokes[idx_cell + next_azi + next_rad           ],      frac_azi *     frac_rad *(1.0-frac_col)*weight);
                atomicAdd(&dev_hStokes[idx_cell                       + next_col], (1.0-frac_azi)*(1.0-frac_rad)*     frac_col *weight);
                atomicAdd(&dev_hStokes[idx_cell + next_azi            + next_col],      frac_azi *(1.0-frac_rad)*     frac_col *weight);
                atomicAdd(&dev_hStokes[idx_cell            + next_rad + next_col], (1.0-frac_azi)*     frac_rad *     frac_col *weight);
                atomicAdd(&dev_hStokes[idx_cell + next_azi + next_rad + next_col],      frac_azi *     frac_rad *     frac_col *weight);
            }
        }   
    } 
}

//====================================================================================================================================================================
//====================================================================================================================================================================

__global__ void optdept_calc (double *dev_optdept)
{
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    
    if (index >= 0 && index < RES_AZI*RES_RAD*RES_COL)
    {	
        int idx_azi, idx_rad, idx_col;
        
        idx_azi = index % RES_AZI;
        idx_col = (int)floor((double)index / (double)(RES_AZI*RES_RAD));
        idx_rad = (index - idx_azi - idx_col*RES_AZI*RES_RAD) / RES_AZI;
        
        double d_azi  = (AZI_MAX - AZI_MIN) / (double)RES_AZI;
        double d_col  = (COL_MAX - COL_MIN) / (double)RES_COL;
        double d_rad  = pow(RAD_MAX / RAD_MIN, 1.0 / (double)RES_RAD);
        double rad_in = RAD_MIN*pow(d_rad, (double)idx_rad);
        double col_in = COL_MIN+idx_col*d_col;
        double volume = (4.0 / 3.0)*M_PI*rad_in*rad_in*rad_in*(d_rad*d_rad*d_rad - 1.0)*(d_azi / (2.0*M_PI))*((cos(col_in) - cos(col_in + d_col)) / 2.0);
        
        dev_optdept[index] /= volume;
        dev_optdept[index] *= (d_rad - 1.0)*rad_in*OPACITY;
    }
}

__global__ void density_calc (double *dev_density, double *dev_lStokes, double *dev_hStokes)
{
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    
    if (index >= 0 && index < RES_AZI*RES_RAD*RES_COL)
    {	
        int idx_azi, idx_rad, idx_col;
        
        idx_azi = index % RES_AZI;
        idx_col = (int)floor((double)index / (double)(RES_AZI*RES_RAD));
        idx_rad = (index - idx_azi - idx_col*RES_AZI*RES_RAD) / RES_AZI;
        
        double d_azi  = (AZI_MAX - AZI_MIN) / (double)RES_AZI;
        double d_col  = (COL_MAX - COL_MIN) / (double)RES_COL;
        double d_rad  = pow(RAD_MAX / RAD_MIN, 1.0 / (double)RES_RAD);
        double rad_in = RAD_MIN*pow(d_rad, (double)idx_rad);
        double col_in = COL_MIN+idx_col*d_col;
        double volume = (4.0 / 3.0)*M_PI*rad_in*rad_in*rad_in*(d_rad*d_rad*d_rad - 1.0)*(d_azi / (2.0*M_PI))*((cos(col_in) - cos(col_in + d_col)) / 2.0);
        
        dev_density[index] /= volume;
        dev_lStokes[index] /= volume;
        dev_hStokes[index] /= volume;
    }
}

//====================================================================================================================================================================
//====================================================================================================================================================================

__global__ void optdept_cumu (double *dev_optdept)
{
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    
    if (index >= 0 && index < RES_AZI*RES_COL)
    {	
        int idx_azi, idx_col, idx_layer;
        
        idx_azi = index % RES_AZI;
        idx_col = (index - idx_azi) / RES_AZI;
        
        for (int i = 1; i < RES_RAD; i++)
        {
            idx_layer = idx_col*RES_RAD*RES_AZI + i*RES_AZI + idx_azi;
            dev_optdept[idx_layer] += dev_optdept[idx_layer - RES_AZI];
        }
    }
}


__global__ void optdept_mean (double *dev_optdept)
{
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    
    if (index >= 0 && index < RES_RAD*RES_COL)
    {	
        int idx_rad, idx_col, idx_layer;
        
        double optdept_sum = 0.0;
        
        idx_rad = index % RES_RAD;
        idx_col = (index - idx_rad) / RES_RAD;
        
        for (int i = 0; i < RES_AZI; i++)
        {
            idx_layer = idx_col*RES_RAD*RES_AZI + idx_rad*RES_AZI + i;
            optdept_sum += dev_optdept[idx_layer];
        }
        
        for (int j = 0; j < RES_AZI; j++)
        {
            idx_layer = idx_col*RES_RAD*RES_AZI + idx_rad*RES_AZI + j;
            dev_optdept[idx_layer] = optdept_sum / RES_AZI;
        }
    }
}

//====================================================================================================================================================================
//====================================================================================================================================================================
