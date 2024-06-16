#include "cudust.cuh"

// =========================================================================================================================

__global__
void ssa_substep_1 (swarm *dev_particle, real *dev_timestep)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if(idx >= 0 && idx < NUM_PAR)
    {
        real dt = *dev_timestep;

        real azi_i   = dev_particle[idx].position.x;
        real rad_i   = dev_particle[idx].position.y;
        real col_i   = dev_particle[idx].position.z;
        real l_azi_i = dev_particle[idx].dynamics.x;
        real v_rad_i = dev_particle[idx].dynamics.y;
        real l_col_i = dev_particle[idx].dynamics.z;

        real rad_1 = rad_i + 0.5*v_rad_i*dt;
        real col_1 = col_i + 0.5*l_col_i*dt / rad_i / rad_1;
        real azi_1 = azi_i + 0.5*l_azi_i*dt / rad_i / rad_1 / sin(col_i) / sin(col_1);

        while (azi_1 >= AZI_MAX) azi_1 -= 2.0*M_PI;
        while (azi_1 <  AZI_MIN) azi_1 += 2.0*M_PI;

        if (rad_1 < RAD_MIN)
        {
            rad_1 = RAD_MAX;
            col_1 = 0.5*M_PI;
            dev_particle[idx].dynamics.x = sqrt(G*M*RAD_MAX);
            dev_particle[idx].dynamics.y = 0.0;
            dev_particle[idx].dynamics.z = 0.0;
        }

        dev_particle[idx].position.x = azi_1;
        dev_particle[idx].position.y = rad_1;
        dev_particle[idx].position.z = col_1;
    }
}

// =========================================================================================================================

__global__
void ssa_substep_2 (swarm *dev_particle, real *dev_timestep, real *dev_optdepth)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if(idx >= 0 && idx < NUM_PAR)
    {
        real dt = *dev_timestep;
        
        real azi_1   = dev_particle[idx].position.x;
        real rad_1   = dev_particle[idx].position.y;
        real col_1   = dev_particle[idx].position.z;
        real l_azi_i = dev_particle[idx].dynamics.x;
        real v_rad_i = dev_particle[idx].dynamics.y;
        real l_col_i = dev_particle[idx].dynamics.z;

        real bigR_1 = rad_1*sin(col_1);

        // get the velocities of gas in the hydrostatic equilibrium state
        real eta_1    = (IDX_TEMP + IDX_SIGMAG - 1.0)*H_REF*H_REF*pow(bigR_1 / R_REF, IDX_TEMP + 1.0) + IDX_TEMP*(1.0 - bigR_1 / rad_1);
        real vg_rad_1 = 0.0;
        real lg_col_1 = 0.0;
        real lg_azi_1 = sqrt(G*M*bigR_1)*sqrt(1.0 + eta_1);

        // calculate the stopping time and the dimensionless time  
        real ts_1 = ST_REF / sqrt(G*M / R_REF / R_REF / R_REF);
        ts_1 *= pow(bigR_1 / R_REF, 1.0 - IDX_SIGMAG - 0.5*IDX_TEMP); // radial gas density and sound speed 
        ts_1 /= exp(-cos(col_1)*cos(col_1) / (2.0*H_REF*H_REF*sin(col_1)*sin(col_1)*pow(bigR_1 / R_REF, IDX_TEMP + 1.0))); // vertical gas density
        real tau_1 = dt / ts_1;

        // get the updated positions of the stars, assuming col_star is always half-PI
        real rad_star_1 = D / (1.0 + Q);                // for the secondary; for the primary: rad_star_1 * Q
        real azi_star_1 = 0.5*sqrt(G*M / D / D / D)*dt; // for the secondary; for the primary: azi_star_1 + M_PI
        real dist_pri_1 = sqrt(rad_1*rad_1 + Q*Q*rad_star_1*rad_star_1 + 2.0*Q*rad_1*rad_star_1*sin(col_1)*cos(azi_1 - azi_star_1));
        real dist_sec_1 = sqrt(rad_1*rad_1 +     rad_star_1*rad_star_1 - 2.0*  rad_1*rad_star_1*sin(col_1)*cos(azi_1 - azi_star_1));

        // retrieve the optical depth of the particle
        real optdepth = get_optdepth(dev_optdepth, azi_1, rad_1, col_1);

        // calculate the gravity components in the spherical coordinates
        real forces_pri_rad_1 = -G*M   / (1.0 + Q) / dist_pri_1 / dist_pri_1 / dist_pri_1 * (rad_1 + Q*rad_star_1*sin(col_1)*cos(azi_star_1 - azi_1));
        real forces_pri_col_1 = -G*M   / (1.0 + Q) / dist_pri_1 / dist_pri_1 / dist_pri_1 * (      + Q*rad_star_1*cos(col_1)*cos(azi_star_1 - azi_1));
        real forces_pri_azi_1 = -G*M   / (1.0 + Q) / dist_pri_1 / dist_pri_1 / dist_pri_1 * (      + Q*rad_star_1           *sin(azi_star_1 - azi_1));

        real forces_sec_rad_1 = -G*M*Q / (1.0 + Q) / dist_sec_1 / dist_sec_1 / dist_sec_1 * (rad_1 -   rad_star_1*sin(col_1)*cos(azi_star_1 - azi_1));
        real forces_sec_col_1 = -G*M*Q / (1.0 + Q) / dist_sec_1 / dist_sec_1 / dist_sec_1 * (      -   rad_star_1*cos(col_1)*cos(azi_star_1 - azi_1));
        real forces_sec_azi_1 = -G*M*Q / (1.0 + Q) / dist_sec_1 / dist_sec_1 / dist_sec_1 * (      -   rad_star_1           *sin(azi_star_1 - azi_1));

        // calculate the external forces and torques (using the updated positions but outdated velocities)
        real ext_forces_rad_1 = (1.0 - BETA_REF*exp(-optdepth))*forces_pri_rad_1                  + forces_sec_rad_1;
        real ext_torque_col_1 = (1.0 - BETA_REF*exp(-optdepth))*forces_pri_col_1*rad_1            + forces_sec_col_1*rad_1;
        real ext_torque_azi_1 = (1.0 - BETA_REF*exp(-optdepth))*forces_pri_azi_1*rad_1*sin(col_1) + forces_sec_azi_1*rad_1*sin(col_1);

        // calculate the extra terms in spherical coordinates (using the updated positions but outdated velocities)
        real extra_rad_1 = l_azi_i*l_azi_i / bigR_1 / bigR_1 / rad_1 + l_col_i*l_col_i / rad_1 / rad_1 / rad_1;
        real extra_col_1 = l_azi_i*l_azi_i / bigR_1 / bigR_1 / sin(col_1) * cos(col_1);

        // calculate the updated velocities
        real v_rad_1 = v_rad_i + ((ext_forces_rad_1 + extra_rad_1)*ts_1 + vg_rad_1 - v_rad_i)*(1.0 - exp(-0.5*tau_1));
        real l_col_1 = l_col_i + ((ext_torque_col_1 + extra_col_1)*ts_1 + lg_col_1 - l_col_i)*(1.0 - exp(-0.5*tau_1));
        real l_azi_1 = l_azi_i + ((ext_torque_azi_1              )*ts_1 + lg_azi_1 - l_azi_i)*(1.0 - exp(-0.5*tau_1));

        // calculate the external forces and torques (using the updated positions and velocities)
        real ext_forces_rad_2 = ext_forces_rad_1; // because here the external forces are indipendent from velocity
        real ext_torque_col_2 = ext_torque_col_1; // because here the external forces are indipendent from velocity
        real ext_torque_azi_2 = ext_torque_azi_1; // because here the external forces are indipendent from velocity

        // calculate the extra terms in spherical coordinates (using the updated positions and velocities)
        real extra_rad_2 = l_azi_1*l_azi_1 / bigR_1 / bigR_1 / rad_1 + l_col_1*l_col_1 / rad_1 / rad_1 / rad_1;
        real extra_col_2 = l_azi_1*l_azi_1 / bigR_1 / bigR_1 / sin(col_1) * cos(col_1);

        // calculate the next-step velocities
        real v_rad_j = v_rad_i + ((ext_forces_rad_2 + extra_rad_2)*ts_1 + vg_rad_1 - v_rad_i)*(1.0 - exp(-tau_1));
        real l_col_j = l_col_i + ((ext_torque_col_2 + extra_col_2)*ts_1 + lg_col_1 - l_col_i)*(1.0 - exp(-tau_1));
        real l_azi_j = l_azi_i + ((ext_torque_azi_2              )*ts_1 + lg_azi_1 - l_azi_i)*(1.0 - exp(-tau_1));

        // calculate the next-step positions
        real rad_j = rad_1 + 0.5*v_rad_j*dt;
        real col_j = col_1 + 0.5*l_col_j*dt / rad_1 / rad_j;
        real azi_j = azi_1 + 0.5*l_azi_j*dt / rad_1 / rad_j / sin(col_1) / sin(col_j);

        // co-rotating with the stars
        azi_j -= sqrt(G*M / D / D / D)*dt;

        while (azi_j >= AZI_MAX) azi_j -= 2.0*M_PI;
        while (azi_j <  AZI_MIN) azi_j += 2.0*M_PI;

        if (rad_j < RAD_MIN)
        {
            rad_j   = RAD_MAX;
            col_j   = 0.5*M_PI;
            l_azi_j = sqrt(G*M*RAD_MAX);
            v_rad_j = 0.0;
            l_col_j = 0.0;
        }

        dev_particle[idx].position.x = azi_j;
        dev_particle[idx].position.y = rad_j;
        dev_particle[idx].position.z = col_j;

        dev_particle[idx].dynamics.x = l_azi_j;
        dev_particle[idx].dynamics.y = v_rad_j;
        dev_particle[idx].dynamics.z = l_col_j;
    }
}