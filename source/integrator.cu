#include <cmath>

#include "const.h"
#include "funclib.cuh"

//====================================================================================================================================================================
//====================================================================================================================================================================

__global__
void stepnum_calc (double *dev_stepnum, par *dev_grain)
{
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    
    if(index >= 0 && index < PAR_NUM)
    {
        double      l_azi, time_azi;
        double rad, v_rad, time_rad;
        double      l_col, time_col;
        double d_rad, par_rad, time_min, step_num;
        
        rad = dev_grain[index].rad;
        l_azi = dev_grain[index].l_azi;
        v_rad = dev_grain[index].v_rad;
        l_col = dev_grain[index].l_col;
        
        d_rad   = pow(RAD_MAX / RAD_MIN, 1.0 / (double)RES_RAD);
        par_rad = log(rad / RAD_MIN) / log(d_rad);
        
        time_azi = ((AZI_MAX - AZI_MIN) / (double)RES_AZI) / (abs(l_azi / rad) + 1.0e-10);
        time_col = ((COL_MAX - COL_MIN) / (double)RES_COL) / (abs(l_col / rad) + 1.0e-10);
        time_rad = (d_rad - 1.0)*RAD_MIN*pow(d_rad, floor(par_rad)) / (abs(v_rad) + 1.0e-10);
        time_min = min(time_azi, min(time_rad, time_col));
        step_num = time_min / TIME_STEP;
        
        if (step_num >= 1.0) dev_stepnum[index] = 1.0;
        else                 dev_stepnum[index] = step_num;
    }
}

//====================================================================================================================================================================
//====================================================================================================================================================================

__global__
void parmove_act1 (par *dev_grain)
{
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    
    if(index >= 0 && index < PAR_NUM)
    {
        double azi, l_azi, azi_temp;
        double rad, v_rad, rad_temp;
        double col, l_col, col_temp;
        
        azi = dev_grain[index].azi;
        rad = dev_grain[index].rad;
        col = dev_grain[index].col;
        l_azi = dev_grain[index].l_azi;
        v_rad = dev_grain[index].v_rad;
        l_col = dev_grain[index].l_col;
        
        rad_temp = rad + 0.5*v_rad*TIME_STEP;
        col_temp = col + 0.5*l_col*TIME_STEP / rad / rad_temp;
        azi_temp = azi + 0.5*l_azi*TIME_STEP / rad / rad_temp / sin(col) / sin(col_temp);

        while (azi_temp >= AZI_MAX) azi_temp -= AZI_MAX;
        while (azi_temp <  AZI_MIN) azi_temp += AZI_MAX;
        
        if (rad_temp < RAD_MIN) 
        {
            rad_temp = RAD_MAX;
            dev_grain[index].v_rad = 0.0;
            dev_grain[index].l_azi = sqrt(G*M*rad_temp);
        }

        dev_grain[index].azi = azi_temp;
        dev_grain[index].rad = rad_temp;
        dev_grain[index].col = col_temp;
    }
}

//====================================================================================================================================================================
//====================================================================================================================================================================

__global__
void parmove_act2 (par *dev_grain, double *dev_optdept)
{
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    
    if(index >= 0 && index < PAR_NUM)
    {
        double azi_temp, azi_next, l_azi, l_azi_temp, l_azi_next, torq_azi, lg_azi, par_azi;
        double rad_temp, rad_next, v_rad, v_rad_temp, v_rad_next, forc_rad, vg_rad, par_rad;
        double col_temp, col_next, l_col, l_col_temp, l_col_next, torq_col, lg_col, par_col;
        double cent_rad, cent_col, T_stop, tau, beta_eff, optdept;
    
        azi_temp = dev_grain[index].azi;
        rad_temp = dev_grain[index].rad;
        col_temp = dev_grain[index].col;
        l_azi    = dev_grain[index].l_azi;
        v_rad    = dev_grain[index].v_rad;
        l_col    = dev_grain[index].l_col;
    
        par_azi = (azi_temp - AZI_MIN) / ((AZI_MAX - AZI_MIN) / (double)RES_AZI);
        par_col = (col_temp - COL_MIN) / ((COL_MAX - COL_MIN) / (double)RES_COL);
        par_rad = log(rad_temp / RAD_MIN) / log(pow(RAD_MAX / RAD_MIN, 1.0 / (double)RES_RAD));
        optdept = get_optdept(par_azi, par_rad, par_col, dev_optdept);
    
        // equilibrium solutions of gas kinematics (independent from velocities)
        lg_azi = sqrt(G*M*rad_temp*sin(col_temp))*sqrt(1.0 + ASPECT_RATIO*ASPECT_RATIO*(SIGMA_INDEX + TEMPE_INDEX));
        vg_rad = 0.0;
        lg_col = 0.0;
    
        // note that Omega_0 = 1 is omitted here    
        T_stop  = St_0*(dev_grain[index].siz / 1.0e-4)*pow(rad_temp*sin(col_temp), -SIGMA_INDEX - 0.5*TEMPE_INDEX);
        tau     = TIME_STEP / T_stop;

        // using updated positions but outdated velocities
        beta_eff = 1.0 - BETA*pow((dev_grain[index].siz / 1.0e-4), -1.0)*exp(-optdept);
        forc_rad =-beta_eff*G*M / rad_temp / rad_temp;
        torq_azi = 0.0;
        torq_col = 0.0;
    
        cent_rad = l_azi*l_azi / rad_temp / rad_temp / rad_temp / sin(col_temp) / sin(col_temp) + l_col*l_col / rad_temp / rad_temp / rad_temp;
        cent_col = l_azi*l_azi*cos(col_temp) / rad_temp / rad_temp / sin(col_temp) / sin(col_temp) / sin(col_temp);
    
        v_rad_temp = v_rad + ((forc_rad + cent_rad)*T_stop + vg_rad - v_rad)*(1.0 - exp(-0.5*tau));
        //v_rad_temp = 0.0;
        l_col_temp = l_col + ((torq_col + cent_col)*T_stop + lg_col - l_col)*(1.0 - exp(-0.5*tau));
        l_azi_temp = l_azi + ((torq_azi           )*T_stop + lg_azi - l_azi)*(1.0 - exp(-0.5*tau));
        
        // using updated positions and velocities
        cent_rad = l_azi_temp*l_azi_temp / rad_temp / rad_temp / rad_temp / sin(col_temp) / sin(col_temp) + l_col_temp*l_col_temp / rad_temp / rad_temp / rad_temp;
        cent_col = l_azi_temp*l_azi_temp*cos(col_temp) / rad_temp / rad_temp / sin(col_temp) / sin(col_temp) / sin(col_temp);
        
        v_rad_next = v_rad + ((forc_rad + cent_rad)*T_stop + vg_rad - v_rad)*(1.0 - exp(-tau));
        //v_rad_next = 0.0;
        l_col_next = l_col + ((torq_col + cent_col)*T_stop + lg_col - l_col)*(1.0 - exp(-tau));
        l_azi_next = l_azi + ((torq_azi           )*T_stop + lg_azi - l_azi)*(1.0 - exp(-tau));
        
        rad_next = rad_temp + 0.5*v_rad_next*TIME_STEP;
        col_next = col_temp + 0.5*l_col_next*TIME_STEP / rad_temp / rad_next;
        azi_next = azi_temp + 0.5*l_azi_next*TIME_STEP / rad_temp / rad_next / sin(col_temp) / sin(col_next);
        
        while (azi_next >= AZI_MAX) azi_next -= AZI_MAX;
        while (azi_next <  AZI_MIN) azi_next += AZI_MAX;
        
        if (rad_next < RAD_MIN) 
        {
            rad_next   = RAD_MAX;
            v_rad_next = 0.0;
            l_azi_next = sqrt(G*M*rad_next);
        }
    
        dev_grain[index].azi = azi_next;
        dev_grain[index].rad = rad_next;
        dev_grain[index].col = col_next;
        dev_grain[index].l_azi = l_azi_next;
        dev_grain[index].v_rad = v_rad_next;
        dev_grain[index].l_col = l_col_next;
    }
}
