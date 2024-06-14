#ifndef CONST_CUH
#define CONST_CUH

#include <cmath>    // for M_PI
#include <string>   // for std::string

using real  = double;
using real3 = double3;

// =========================================================================================================================
// resolutions

const int NUM_PAR = 1;
const int RES_AZI = 1024;
const int RES_RAD = 1024;
const int RES_COL = 3;

// =========================================================================================================================
// code units

const real G     = 1.0;
const real M_REF = 1.0;
const real R_REF = 1.0;

// =========================================================================================================================
// disk paramters

const real H_REF      =  0.05;
const real IDX_TEMP   = -3.0/7.0;
const real IDX_SIGMAG = -1.5;

// =========================================================================================================================
// dust parameters

const real ST_REF    = 1.0e+06;
const real BETA_REF  = 0.0; // 1.0e+01;
const real KAPPA_REF = 1.0e+07*R_REF*R_REF/M_REF;  // [kappa] = [R^2]/[M], ~1cm2/g, Miyake & Nakagawa 1993

const real IDX_SIZE  = -3.5;

// =========================================================================================================================
// companion parameters

const real M_COMP   = 0.5*M_REF;
const real RAD_COMP = 0.5*R_REF;
const real SIZE_CSD = 0.1*RAD_COMP;

// =========================================================================================================================
// disk size and dust init

const real AZI_INIT_MIN  = 0.0*M_PI;
const real AZI_INIT_MAX  = 2.0*M_PI;
const real AZI_MIN       = AZI_INIT_MIN;
const real AZI_MAX       = AZI_INIT_MAX;

const real SMOOTH_RAD    = 0.02*R_REF;
const real RAD_INIT_MIN  = 1.0*R_REF;
const real RAD_INIT_MAX  = 1.5*R_REF;
const real RAD_MIN       = RAD_INIT_MIN - 2.5*SMOOTH_RAD;
const real RAD_MAX       = RAD_INIT_MAX + 2.5*SMOOTH_RAD;

const real ARCTAN_3H     = 0.1488899476095;
const real COL_INIT_MIN  = 0.5*M_PI;
const real COL_INIT_MAX  = 0.5*M_PI;
const real COL_MIN       = 0.5*M_PI - 0.005*ARCTAN_3H;
const real COL_MAX       = 0.5*M_PI + 0.005*ARCTAN_3H;

// =========================================================================================================================
// cuda numerical parameters

const int THREADS_PER_BLOCK = 32;

const int NUM_AZI = RES_RAD*RES_COL;
const int NUM_RAD = RES_AZI*RES_COL;
const int NUM_COL = RES_AZI*RES_RAD;
const int NUM_DIM = RES_AZI*RES_RAD*RES_COL;

const int BLOCKNUM_PAR = NUM_PAR / THREADS_PER_BLOCK + 1;
const int BLOCKNUM_AZI = NUM_AZI / THREADS_PER_BLOCK + 1;
const int BLOCKNUM_RAD = NUM_RAD / THREADS_PER_BLOCK + 1;
const int BLOCKNUM_DIM = NUM_DIM / THREADS_PER_BLOCK + 1;

// =========================================================================================================================
// time step and output parameters

const int  OUTPUT_NUM = 100;
const int  OUTPUT_PAR = 1;
const real OUTPUT_INT = 0.02*M_PI;
const real DT_MAX     = 0.001*M_PI;

// const real DT_MAX     = 2.0*M_PI/static_cast<real>(RES_AZI);

const std::string OUTPUT_PATH = "outputs/";

// =========================================================================================================================
// structures

struct swarm
{
    real3 position;     // x = azi, y = rad, z = col
    real3 dynamics;     // x = specific AM in azi, y = radial velocity, z = specific AM in col
};

struct interp
{
    int  next_azi, next_rad, next_col;
    real frac_azi, frac_rad, frac_col;
};

// =========================================================================================================================

#endif
