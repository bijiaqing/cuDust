#include <cmath>
#include <cstdlib>

#include "funclib.h"

using namespace std;

//====================================================================================================================================================================
//====================================================================================================================================================================

double erf_invsq (double input) 
{
    // used in 'errfunc_generator'
    // this gives the shape of the radial prof
    // normalized as the global maximum is 0.0571418 at x ~ 3.51612
    // both input and output should range from 0 to 1

    double x = 30.0*input;
    double y = (erf(x - 2.5) + 1.0) / (x*x + 20.0);

    double output = y / 0.0571418;
    
    return output;
}

//====================================================================================================================================================================

void sigmoid_generator (double *prof, double p_min, double p_max) 
{
    int count = 0;
    double rand_x, rand_y;
    
    while (count < PAR_NUM) 
    {
        rand_x = (double)rand() / double(RAND_MAX);
        rand_y = (double)rand() / double(RAND_MAX);
        
        if (rand_y <= erf_invsq(rand_x))
        {	
            prof[count] = p_min + (p_max - p_min)*rand_x;
            count++;
        }
    }
}

//====================================================================================================================================================================
//====================================================================================================================================================================

double pow_law (double m, double m_min, double m_max, double exponent)
{
    double output;

    if (m >= m_min && m <= m_max) output = pow(m / m_min, exponent); 
    else                          output = 0.0;
    
    return output;
}

//====================================================================================================================================================================

double kernel (double n, double mu, double sigma)
{
    return exp(-(n - mu)*(n - mu) / (2.0*sigma*sigma));
}

//====================================================================================================================================================================

void convpow_generator (double *prof, double p_min, double p_max)
{
    int count = 0;
    int x_res = RES_RAD;
    
    double x_idx = 0.0;
    double fracx = 0.0;
    double x_min = p_min - 3.0*SMOOTH;
    double x_max = p_max + 3.0*SMOOTH;
    double rand_x, rand_y, y_max;
    double x_axis[x_res], y_axis[x_res];
    
    for (int i = 0; i < x_res; i++)
    {
        x_axis[i] = x_min + ((double)i / (double)(x_res - 1))*(x_max - x_min);
        y_axis[i] = 0.0;
    }
    
    for (int j = 0; j < x_res; j++)
    {
        for (int k = 0; k < x_res; k++)
            y_axis[k] += pow_law(x_axis[j], p_min, p_max, SIGMA_INDEX + 1.0)*kernel(x_axis[k], x_axis[j], SMOOTH);
    }
    
    y_max = 0.0;
    
    // find the peak of the rand_r profile
    for (int l = 0; l < x_res; l++)
    {
        if (y_axis[l] > y_max) y_max = y_axis[l];
    }
    
    //printf("%f\n", y_max);
    //for (int fuck = 0; fuck < x_res; fuck++)
    //    printf("%f, %f\n", x_axis[fuck], y_axis[fuck]);
    
    while (count < PAR_NUM) 
    {
        rand_x = x_min + (x_max - x_min)*(double)rand() / double(RAND_MAX);
        rand_y = y_max                  *(double)rand() / double(RAND_MAX);
        
        x_idx = (rand_x - x_min) / ((x_max - x_min) / (double)(x_res - 1));
        fracx = x_idx - floor(x_idx);
        
        if (rand_y <= (1.0 - fracx)*y_axis[(int)floor(x_idx)] + fracx*y_axis[(int)floor(x_idx) + 1])
        {	
            prof[count] = rand_x;
            count++;
        }
    }
}

//====================================================================================================================================================================
//====================================================================================================================================================================

double gaussian (double input)
{
    double sigma = atan(ASPECT_RATIO);
    double x = -(input - 0.5*M_PI)*(input - 0.5*M_PI) / (2.0*sigma*sigma);
    
    return exp(x);
}

//====================================================================================================================================================================

void gaussym_generator (double *prof, double p_min, double p_max)
{
    int count = 0;
    double rand_x, rand_y;
    
    while (count < PAR_NUM) 
    {
        rand_x = p_min + (p_max - p_min)*(double)rand() / double(RAND_MAX);
        rand_y =                         (double)rand() / double(RAND_MAX);
        
        if (rand_y <= gaussian(rand_x))
        {	
            prof[count] = rand_x;
            count++;
        }
    }
}

//====================================================================================================================================================================
//====================================================================================================================================================================

void uniform_generator (double *prof, double p_min, double p_max)
{
    double random;
    
    for (int i = 0; i < PAR_NUM; i++)
    {
        random  = (double)rand() / double(RAND_MAX);
        prof[i] = p_min + (p_max - p_min)*random;
    }
}

//====================================================================================================================================================================
//====================================================================================================================================================================

void pow_law_generator (double *prof, double p_min, double p_max, double index)
{
    double pow_min, pow_max, random;
    
    pow_min = pow(p_min, index + 1.0);
    pow_max = pow(p_max, index + 1.0);
    
    // check https://mathworld.wolfram.com/RandomNumber.html
    for (int i = 0; i < PAR_NUM; i++)
    {
        random  = (double)rand() / double(RAND_MAX);
        prof[i] = pow((pow_max - pow_min)*random + pow_min, 1.0 / (index + 1.0));
    }
}
