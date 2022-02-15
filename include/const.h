#ifndef CONST_H
#define CONST_H

#include <cmath>
#include <string>
#include <cstdlib>

const int PAR_NUM = 1.0e+07;
const int RES_AZI =    4096;
const int RES_RAD =    1024;
const int RES_COL =       3;

const int THREADSPERBLOCK = 32;

const double G = 1.0;
const double M = 1.0;

const double SIGMA_INDEX = -1.5;
const double TEMPE_INDEX = -1.0;
const double GSIZE_INDEX = -1.5;    // corresponds to collisional equilibrium
                                    // when superparticles have equal surface areas

const double BETA = 1.0e+01;        // radiation pressure over gravity for 1 micron grains
const double St_0 = 1.0e-04;        // Stokes number at 1au for 1 micron grains
const double OPACITY = 0.3/(double)PAR_NUM; //fiducial: 0.3/(double)PAR_NUM
const double ASPECT_RATIO = 0.05;

#define ARCTAN_3H 0.1488899476095;

const double AZI_INIT_MIN = 0.0*M_PI;
const double AZI_INIT_MAX = 2.0*M_PI;
const double AZI_MIN = AZI_INIT_MIN;
const double AZI_MAX = AZI_INIT_MAX;

const double SMOOTH  = 0.02;
const double RAD_INIT_MIN = 1.0e+00;
const double RAD_INIT_MAX = 1.5e+00;
const double RAD_MIN = RAD_INIT_MIN - 2.5*SMOOTH;
const double RAD_MAX = RAD_INIT_MAX + 2.5*SMOOTH;

const double COL_INIT_MIN = 0.5*M_PI;
const double COL_INIT_MAX = 0.5*M_PI;
const double COL_MIN = 0.5*M_PI - 0.005*ARCTAN_3H;
const double COL_MAX = 0.5*M_PI + 0.005*ARCTAN_3H;

const double SIZ_INIT_MIN = 1.0e-04;    // 1 um
const double SIZ_INIT_MAX = 1.0e-04;    // 0.3 mm
const double CRITICAL_SIZ = 1.0e-03;    // 0.1 mm (used in l/h Stokes)

const int    OUTPUT_NUM = 500;
const int    OGRAIN_INT = 500;
const double OUTPUT_INT = 2.0*M_PI;
const double TIME_STEP  = 2.0*M_PI/(double)RES_AZI;

const std::string PATH = "outputs/";

struct par
{
    double azi, rad, col, l_azi, v_rad, l_col, siz;
};

struct interp
{
    int    next_azi, next_rad, next_col;
    double frac_azi, frac_rad, frac_col;
};

#endif
