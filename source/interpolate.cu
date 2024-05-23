#include "cudust.cuh"

// =========================================================================================================================

__device__ 
interp linear_interp_cent (real par_azi, real par_rad, real par_col)
{
    real deci_azi, deci_rad, deci_col;
    real cent_azi, cent_rad, cent_col;
    real frac_azi, frac_rad, frac_col;
    int  next_azi, next_rad, next_col;

    real d_rad = pow(RAD_MAX / RAD_MIN, 1.0 / static_cast<real>(RES_RAD));

    deci_azi = par_azi - floor(par_azi);
    deci_rad = par_rad - floor(par_rad);
    deci_col = par_col - floor(par_col);

    cent_azi = 0.5;
    cent_rad = log(0.5*(1.0 + d_rad)) / log(d_rad); // deci_rad for the midpoint of the cell
    cent_col = 0.5;

    // "cent_rad = log(0.5*(1.0 + d_rad)) / log(d_rad)" is slightly larger than 0.5
    // it means that values are defined at the (radial) geometric center of each cell
    // therefore, linear interpolation at radial edges of cells will always bias to the inner one
    // if "cent_rad = 0.5", then particles exactly at the radial edges will be equally split into the two cells

    bool inside_azi = par_azi >= cent_azi && par_azi < RES_AZI + cent_azi - 1.0;
    bool inside_rad = par_rad >= cent_rad && par_rad < RES_RAD + cent_rad - 1.0;
    bool inside_col = par_col >= cent_col && par_col < RES_COL + cent_col - 1.0;

    if (inside_azi)
    {
        if (deci_azi >= cent_azi)
        {
            frac_azi = deci_azi - cent_azi;
            next_azi = 1;
        }
        else
        {
            frac_azi = cent_azi - deci_azi;
            next_azi = -1;
        }
    }
    else
    {
        if (deci_azi >= cent_azi)
        {
            frac_azi = deci_azi - cent_azi;
            next_azi = 1 - RES_AZI;
        }
        else
        {
            frac_azi = cent_azi - deci_azi;
            next_azi = RES_AZI - 1;
        }
    }

    if (inside_rad)
    {
        if (deci_rad >= cent_rad)
        {
            frac_rad = (pow(d_rad, 3.0*(deci_rad - cent_rad)) - 1.0) / (pow(d_rad, 3.0) - 1.0);
            next_rad = RES_AZI;
        }
        else
        {
            frac_rad = (pow(d_rad, 3.0) - pow(d_rad, 3.0*(deci_rad - cent_rad + 1.0))) / (pow(d_rad, 3.0) - 1.0);
            next_rad = -RES_AZI;
        }
    }
    else
    {
        if (deci_rad >= cent_rad)
        {
            frac_rad = (pow(d_rad, 3.0*(deci_rad - cent_rad)) - 1.0) / (pow(d_rad, 3.0) - 1.0);
            next_rad = 0;
        }
        else
        {
            frac_rad = (pow(d_rad, 3.0) - pow(d_rad, 3.0*(deci_rad - cent_rad + 1.0))) / (pow(d_rad, 3.0) - 1.0);
            next_rad = 0;
        }
    }

    if (inside_col)
    {
        if (deci_col >= cent_col)
        {
            frac_col = deci_col - cent_col;
            next_col = NUM_COL;
        }
        else
        {
            frac_col = cent_col - deci_col;
            next_col = -NUM_COL;
        }
    }
    else
    {
        if (deci_col >= cent_col)
        {
            frac_col = deci_col - cent_col;
            next_col = 0;
        }
        else
        {
            frac_col = cent_col - deci_col;
            next_col = 0;
        }
    }

    interp result;

    result.next_azi = next_azi;
    result.next_rad = next_rad;
    result.next_col = next_col;
    result.frac_azi = frac_azi;
    result.frac_rad = frac_rad;
    result.frac_col = frac_col;

    return result;
}

// =========================================================================================================================

__device__ 
interp linear_interp_stag (real par_azi, real par_rad, real par_col)
{
    real deci_azi, deci_rad, deci_col;
    real cent_azi, cent_rad, cent_col;
    real frac_azi, frac_rad, frac_col;
    int  next_azi, next_rad, next_col;

    real d_rad = pow(RAD_MAX / RAD_MIN, 1.0 / static_cast<real>(RES_RAD));

    deci_azi = par_azi - floor(par_azi);
    deci_rad = par_rad - floor(par_rad);
    deci_col = par_col - floor(par_col);

    cent_azi = 0.5;
    cent_rad = 1.0;
    cent_col = 0.5;

    // "cent_rad = 1.0" means that values are defined at the outer radial edge of the cell
    // however, this means particles at radial edge cells will get 100% self-shadowing effect

    bool inside_azi = par_azi >= cent_azi && par_azi < RES_AZI + cent_azi - 1.0;
    bool inside_rad = par_rad >= cent_rad && par_rad < RES_RAD + cent_rad - 1.0;
    bool inside_col = par_col >= cent_col && par_col < RES_COL + cent_col - 1.0;

    if (inside_azi)
    {
        if (deci_azi >= cent_azi)
        {
            frac_azi = deci_azi - cent_azi;
            next_azi = 1;
        }
        else
        {
            frac_azi = cent_azi - deci_azi;
            next_azi = -1;
        }
    }
    else
    {
        if (deci_azi >= cent_azi)
        {
            frac_azi = deci_azi - cent_azi;
            next_azi = 1 - RES_AZI;
        }
        else
        {
            frac_azi = cent_azi - deci_azi;
            next_azi = RES_AZI - 1;
        }
    }

    if (inside_rad)
    {
        frac_rad = (pow(d_rad, 3.0) - pow(d_rad, 3.0*deci_rad)) / (pow(d_rad, 3.0) - 1.0);
        next_rad = -RES_AZI;
    }
    else
    {
        frac_rad = (pow(d_rad, 3.0) - pow(d_rad, 3.0*deci_rad)) / (pow(d_rad, 3.0) - 1.0);
        next_rad = 0;
    }
    
    if (inside_col)
    {
        if (deci_col >= cent_col)
        {
            frac_col = deci_col - cent_col;
            next_col = NUM_COL;
        }
        else
        {
            frac_col = cent_col - deci_col;
            next_col = -NUM_COL;
        }
    }
    else
    {
        if (deci_col >= cent_col)
        {
            frac_col = deci_col - cent_col;
            next_col = 0;
        }
        else
        {
            frac_col = cent_col - deci_col;
            next_col = 0;
        }
    }

    interp result;

    result.next_azi = next_azi;
    result.next_rad = next_rad;
    result.next_col = next_col;
    result.frac_azi = frac_azi;
    result.frac_rad = frac_rad;
    result.frac_col = frac_col;

    return result;
}
