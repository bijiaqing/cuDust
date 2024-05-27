#include "cudust.cuh"

// =========================================================================================================================

__host__
void rand_uniform (real *profile, int number, real p_min, real p_max)
{
    std::uniform_real_distribution <real> random(0.0, 1.0);

    for (int i = 0; i < number; i++)
    {
        profile[i] = p_min + (p_max - p_min)*random(rand_generator);
    }
}

// =========================================================================================================================

__host__
void rand_gaussian (real *profile, int number, real p_min, real p_max, real p_0, real sigma)
{
    int i = 0;
    real random_x, random_y;
    std::uniform_real_distribution <real> random(0.0, 1.0);

    while (i < number)
    {
        random_x = p_min + (p_max - p_min)*random(rand_generator);
        random_y =                         random(rand_generator);
        
        if (random_y <= std::exp(-(random_x - p_0)*(random_x - p_0)/(2.0*sigma*sigma)))
        {	
            profile[i] = random_x;
            i++;
        }
    }
}

// =========================================================================================================================

__host__
void rand_powerlaw (real *profile, int number, real p_min, real p_max, real idx_pow)
{
    std::uniform_real_distribution <real> random(0.0, 1.0);

    // check https://mathworld.wolfram.com/RandomNumber.html for derivations
    // NOTE: the above website is for distribution of dN(x) ~ x^n*dx, which is N(x) ~ x^(n+1)
    //       all (n+1) terms in the provided solution have been substituted by n here
    
    real tmp_min = std::pow(p_min, idx_pow);
    real tmp_max = std::pow(p_max, idx_pow);

    for (int i = 0; i < number; i++)
    {
        profile[i] = std::pow((tmp_max - tmp_min)*random(rand_generator) + tmp_min, 1.0/idx_pow);
    }
}

// =========================================================================================================================

__host__
real pow_law (real x, real x_min, real x_max, real idx_pow)
{
    real output;

    if (x >= x_min && x <= x_max)
    {
        output = std::pow(x/x_min, idx_pow); 
    }
    else
    {
        output = 0.0;
    }

    return output;
}

__host__
real gauss_kernel (real x, real x_0, real sigma)
{
    return std::exp(-(x - x_0)*(x - x_0)/(2.0*sigma*sigma));
}

__host__
void rand_conv_pow (real *profile, int number, real p_min, real p_max, real idx_pow, real smooth, int bins)
{
    int n = 0;

    real x_min = p_min - 2.0*smooth;
    real x_max = p_max + 2.0*smooth;
    real par_x = 0.0;
    real dec_x = 0.0;
    real y_max = 0.0;

    real rand_x, x_axis[bins];
    real rand_y, y_axis[bins];

    std::uniform_real_distribution <real> random(0.0, 1.0);

    for (int i = 0; i < bins; i++)
    {
        x_axis[i] = x_min + (static_cast<real>(i) / static_cast<real>(bins - 1))*(x_max - x_min);
        y_axis[i] = 0.0;
    }

    for (int j = 0; j < bins; j++)
    {
        for (int k = 0; k < bins; k++)
        {
            y_axis[k] += pow_law(x_axis[j], p_min, p_max, idx_pow)*gauss_kernel(x_axis[k], x_axis[j], smooth);
        }
    }

    // find the peak of the rand_r profile
    for (int m = 0; m < bins; m++)
    {
        if (y_axis[m] > y_max)
        {
            y_max = y_axis[m];
        }
    }

    while (n < number) 
    {
        rand_x = x_min + (x_max - x_min)*random(rand_generator);
        rand_y = y_max                  *random(rand_generator);

        par_x = (rand_x - x_min) / ((x_max - x_min) / static_cast<real>(bins - 1));
        dec_x = par_x - std::floor(par_x);

        if (rand_y <= (1.0 - dec_x)*y_axis[static_cast<int>(par_x)] + dec_x*y_axis[static_cast<int>(par_x) + 1])
        {	
            profile[n] = rand_x;
            n++;
        }
    }
}
