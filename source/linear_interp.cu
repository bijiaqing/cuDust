#include <cmath>
#include <cfloat>

#include "const.h"
#include "funclib.cuh"

__device__ interp linear_interp_cent (double par_azi, double par_rad, double par_col)
{
    double dec_azi, mid_azi, frac_azi;
    double dec_rad, mid_rad, frac_rad;
    double dec_col, mid_col, frac_col;
    
    int    next_azi, next_rad, next_col;
    bool   interior_azi, interior_rad, interior_col;

    double d_rad = pow(RAD_MAX / RAD_MIN, 1.0 / (double)RES_RAD); 
    
    interp interp_result;
    
    dec_azi = par_azi - floor(par_azi);
    dec_rad = par_rad - floor(par_rad);
    dec_col = par_col - floor(par_col);

    mid_azi = 0.5;
    mid_col = 0.5;
    mid_rad = log(0.5*(1.0 + exp(1.0)));
    
    interior_azi = par_azi >= mid_azi && par_azi <= RES_AZI + mid_azi - 1.0;
    interior_rad = par_rad >= mid_rad && par_rad <= RES_RAD + mid_rad - 1.0;
    interior_col = par_col >= mid_col && par_col <= RES_COL + mid_col - 1.0;
    
    if (interior_azi)
    {
        if (dec_azi >= mid_azi)
        {
            frac_azi = dec_azi - mid_azi;
            next_azi = 1;
        }
        else
        {
            frac_azi = mid_azi - dec_azi;
            next_azi = -1;
        }
    }
    else
    {
        if (dec_azi >= mid_azi)
        {
            frac_azi = dec_azi - mid_azi;
            next_azi = 1 - RES_AZI;
        }
        else
        {
            frac_azi = mid_azi - dec_azi;
            next_azi = RES_AZI - 1;
        }
    }
    
    if (interior_rad)
    {
        if (dec_rad >= mid_rad)
        {
            frac_rad = (pow(d_rad, 3.0*(dec_rad - mid_rad)) - 1.0) / (pow(d_rad, 3.0) - 1.0);
            next_rad = RES_AZI;
        }
        else
        {
            frac_rad = (pow(d_rad, 3.0) - pow(d_rad, 3.0*(dec_rad - mid_rad + 1.0))) / (pow(d_rad, 3.0) - 1.0);
            next_rad = -RES_AZI;
        }
    }
    else
    {
        if (dec_rad >= mid_rad)
        {
            frac_rad = (pow(d_rad, 3.0*(dec_rad - mid_rad)) - 1.0) / (pow(d_rad, 3.0) - 1.0);
            next_rad = 0;
        }
        else
        {
            frac_rad = (pow(d_rad, 3.0) - pow(d_rad, 3.0*(dec_rad - mid_rad + 1.0))) / (pow(d_rad, 3.0) - 1.0);
            next_rad = 0;
        }
    }
    
    if (interior_col)
    {
        if (dec_col >= mid_col)
        {
            frac_col = dec_col - mid_col;
            next_col = RES_RAD*RES_AZI;
        }
        else
        {
            frac_col = mid_col - dec_col;
            next_col = -RES_RAD*RES_AZI;
        }
    }
    else
    {
        if (dec_col >= mid_col)
        {
            frac_col = dec_col - mid_col;
            next_col = 0;
        }
        else
        {
            frac_col = mid_col - dec_col;
            next_col = 0;
        }
    }
    
    interp_result.next_azi = next_azi;
    interp_result.next_rad = next_rad;
    interp_result.next_col = next_col;
    interp_result.frac_azi = frac_azi;
    interp_result.frac_rad = frac_rad;
    interp_result.frac_col = frac_col;
    
    return interp_result;
}

//====================================================================================================================================================================
//====================================================================================================================================================================

__device__ interp linear_interp_stag (double par_azi, double par_rad, double par_col)
{
    double dec_azi, mid_azi, frac_azi;
    double dec_rad, mid_rad, frac_rad;
    double dec_col, mid_col, frac_col;
    
    int    next_azi, next_rad, next_col;
    bool   interior_azi, interior_rad, interior_col;

    double d_rad = pow(RAD_MAX / RAD_MIN, 1.0 / (double)RES_RAD); 
    
    interp interp_result;
    
    dec_azi = par_azi - floor(par_azi);
    dec_rad = par_rad - floor(par_rad);
    dec_col = par_col - floor(par_col);

    mid_azi = 0.5;
    mid_col = 0.5;
    mid_rad = 1.0;
    
    interior_azi = par_azi >= mid_azi && par_azi <= RES_AZI + mid_azi - 1.0;
    interior_rad = par_rad >= mid_rad && par_rad <= RES_RAD + mid_rad - 1.0;
    interior_col = par_col >= mid_col && par_col <= RES_COL + mid_col - 1.0;
    
    if (interior_azi)
    {
        if (dec_azi >= mid_azi)
        {
            frac_azi = dec_azi - mid_azi;
            next_azi = 1;
        }
        else
        {
            frac_azi = mid_azi - dec_azi;
            next_azi = -1;
        }
    }
    else
    {
        if (dec_azi >= mid_azi)
        {
            frac_azi = dec_azi - mid_azi;
            next_azi = 1 - RES_AZI;
        }
        else
        {
            frac_azi = mid_azi - dec_azi;
            next_azi = RES_AZI - 1;
        }
    }
    
    if (interior_rad)
    {
        frac_rad = (pow(d_rad, 3.0) - pow(d_rad, 3.0*dec_rad)) / (pow(d_rad, 3.0) - 1.0);
        next_rad = -RES_AZI;
    }
    else
    {
        frac_rad = (pow(d_rad, 3.0) - pow(d_rad, 3.0*dec_rad)) / (pow(d_rad, 3.0) - 1.0);
        next_rad = 0;
    }
    
    if (interior_col)
    {
        if (dec_col >= mid_col)
        {
            frac_col = dec_col - mid_col;
            next_col = RES_RAD*RES_AZI;
        }
        else
        {
            frac_col = mid_col - dec_col;
            next_col = -RES_RAD*RES_AZI;
        }
    }
    else
    {
        if (dec_col >= mid_col)
        {
            frac_col = dec_col - mid_col;
            next_col = 0;
        }
        else
        {
            frac_col = mid_col - dec_col;
            next_col = 0;
        }
    }
    
    interp_result.next_azi = next_azi;
    interp_result.next_rad = next_rad;
    interp_result.next_col = next_col;
    interp_result.frac_azi = frac_azi;
    interp_result.frac_rad = frac_rad;
    interp_result.frac_col = frac_col;
    
    return interp_result;
}

//====================================================================================================================================================================
//====================================================================================================================================================================

__device__ double get_optdept (double par_azi, double par_rad, double par_col, double *dev_optdept)
{
    double optdept  = 0.0;
    
    if (par_azi >= 0.0 && par_azi < RES_AZI && par_rad >= 0.0 && par_rad < RES_RAD && par_col >= 0.0 && par_col < RES_COL)
    {		
        int    idx_azi,  idx_rad,  idx_col;
        int    next_azi, next_rad, next_col;
        double frac_azi, frac_rad, frac_col;
        
        idx_azi = floor(par_azi);
        idx_rad = floor(par_rad);
        idx_col = floor(par_col);
        
        int    idx_cell = idx_col*RES_RAD*RES_AZI + idx_rad*RES_AZI + idx_azi; 
        interp interp_result = linear_interp_stag (par_azi, par_rad, par_col);
            
        next_azi = interp_result.next_azi;
        next_rad = interp_result.next_rad;
        next_col = interp_result.next_col;
        frac_azi = interp_result.frac_azi;
        frac_rad = interp_result.frac_rad;
        frac_col = interp_result.frac_col;
        
        optdept += dev_optdept[idx_cell                                 ]*(1.0-frac_azi)*(1.0-frac_rad)*(1.0-frac_col);
        optdept += dev_optdept[idx_cell + next_azi                      ]*     frac_azi *(1.0-frac_rad)*(1.0-frac_col);
        optdept += dev_optdept[idx_cell            + next_rad           ]*(1.0-frac_azi)*     frac_rad *(1.0-frac_col);
        optdept += dev_optdept[idx_cell + next_azi + next_rad           ]*     frac_azi *     frac_rad *(1.0-frac_col);
        optdept += dev_optdept[idx_cell                       + next_col]*(1.0-frac_azi)*(1.0-frac_rad)*     frac_col ;
        optdept += dev_optdept[idx_cell + next_azi            + next_col]*     frac_azi *(1.0-frac_rad)*     frac_col ;
        optdept += dev_optdept[idx_cell            + next_rad + next_col]*(1.0-frac_azi)*     frac_rad *     frac_col ;
        optdept += dev_optdept[idx_cell + next_azi + next_rad + next_col]*     frac_azi *     frac_rad *     frac_col ;
    } 
    else if (par_rad >= RES_RAD)
    {
        optdept = DBL_MAX;
    }
    
    return optdept;
}

//====================================================================================================================================================================
//====================================================================================================================================================================

