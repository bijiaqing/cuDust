#ifndef CONST_CUH
#define CONST_CUH

#include <cmath>    // for M_PI
#include <string>   // for std::string

#include "cukd/builder.h"   // for cukd functions

using real  = double;
using real2 = double2;
using real3 = double3;

// =========================================================================================================================
// resolutions

const int NUM_PAR = 1e+07;
const int RES_AZI = 1024;
const int RES_RAD = 1024;
const int RES_COL = 3;

// =========================================================================================================================
// code units

const real G       = 1.0;
const real M_STAR  = 1.0;
const real RAD_REF = 1.0;

// =========================================================================================================================
// disk paramters

const real ASP_REF  = 0.05;
const real IDX_TEMP =-3.0/7.0;
const real IDX_SIGG =-1.5;

// =========================================================================================================================
// dust parameters

const real M_DUST    = 3.0e-06*M_STAR;
const real RHO_DUST  = 1.0e+06*M_STAR/RAD_REF/RAD_REF/RAD_REF;  // 1g/cm3 when R = 1au, M = 1Msun

const real ST_REF    = 1.0e-04;
const real BETA_REF  = 1.0e+01;
const real SIZE_REF  = 1.0e-18*RAD_REF;                 // 0.1um when R = 1au
const real KAPPA_REF = 1.0e+07*RAD_REF*RAD_REF/M_STAR;  // [kappa] = [R^2]/[M], ~1cm2/g, Miyake & Nakagawa 1993

const real V_FRAG    = 1.0;
const real IDX_SIZE  =-3.5;

const int  KNN_SIZE = 20;
const real MAX_QUERY_DIST = 0.01*RAD_REF;
const real COLLRATE_NORM = 1.0e-07;

// =========================================================================================================================
// disk size and dust init

const real AZI_INIT_MIN  = 0.0*M_PI;
const real AZI_INIT_MAX  = 2.0*M_PI;
const real AZI_MIN       = AZI_INIT_MIN;
const real AZI_MAX       = AZI_INIT_MAX;

const real SMOOTH_RAD    = 0.02*RAD_REF;
const real RAD_INIT_MIN  = 1.0*RAD_REF;
const real RAD_INIT_MAX  = 1.5*RAD_REF;
const real RAD_MIN       = RAD_INIT_MIN - 2.5*SMOOTH_RAD;
const real RAD_MAX       = RAD_INIT_MAX + 2.5*SMOOTH_RAD;

const real ARCTAN_3H     = 0.1488899476095;
const real COL_INIT_MIN  = 0.5*M_PI;
const real COL_INIT_MAX  = 0.5*M_PI;
const real COL_MIN       = 0.5*M_PI - 0.005*ARCTAN_3H;
const real COL_MAX       = 0.5*M_PI + 0.005*ARCTAN_3H;

const real SIZE_INIT_MIN = 1.0e+00*SIZE_REF;
const real SIZE_INIT_MAX = 1.0e+05*SIZE_REF;

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

const int  OUTPUT_NUM = 10;
const int  OUTPUT_PAR = 1;
// const real OUTPUT_INT = 0.2*M_PI;
// const real DT_MAX     = 2.0*M_PI/static_cast<real>(RES_AZI);

const real OUTPUT_INT = 1.0;
const real DT_MAX     = 0.1;

const std::string OUTPUT_PATH = "outputs/";

// =========================================================================================================================
// structures

struct swarm
{
    real3 position;     // x = azi, y = rad, z = col
    real3 velocity;     // velocity for rad but specific angular momentum for azi and col
    real  dustsize;     // size of the individual grain
    real  numgrain;     // number of grains in the swarm
};

struct swarm_tmp
{
    real  dustsize;     // size of the individual grain
    real  numgrain;     // number of grains in the swarm
    real  collrate_i;   // total collision rate for particle i
};

struct tree
{
    float3 xyz;         // xyz position of a tree node (particle), no higher than single precision allowed
    int index_old;      // the index of partile before being shuffled by the tree builder
    int split_dim;      // an 1-byte param for the k-d tree build
};

struct tree_traits
{
    using point_t = float3;
    enum { has_explicit_dim = true };
    
    static inline __host__ __device__ const point_t &get_point (const tree &node) { return node.xyz; }
    static inline __host__ __device__ float get_coord (const tree &node, int dim) { return cukd::get_coord(node.xyz, dim); }
    static inline __host__ __device__ int get_dim (const tree &node) { return node.split_dim; }
    static inline __host__ __device__ void set_dim (tree &node, int dim) { node.split_dim = dim; }
};

struct interp
{
    int  next_azi, next_rad, next_col;
    real frac_azi, frac_rad, frac_col;
};

// =========================================================================================================================

#endif
