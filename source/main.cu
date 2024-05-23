#include <ctime>            // for std::time_t, std::time, std::ctime
#include <chrono>           // for std::chrono::system_clock
#include <iomanip>          // for std::setw, std::setfill
#include <sstream>          // for std::stringstream
#include <iostream>         // for std::cout, std::endl
#include <sys/stat.h>       // for mkdir

#include "cudust.cuh"
#include "curand_kernel.h"

std::mt19937 rand_generator;

int main (int argc, char **argv)
{
    int resume;
    real output_timer = 0.0;
    
    std::string fname;
    std::ofstream ofile;
    std::uniform_real_distribution <real> random(0.0, 1.0); // distribution in [0, 1)

    swarm *particle, *dev_particle;

    cudaMallocHost((void**)&particle, sizeof(swarm)*NUM_PAR);
    cudaMalloc((void**)&dev_particle, sizeof(swarm)*NUM_PAR);
    
    real *dustdens, *dev_dustdens;
    real *optdepth, *dev_optdepth;

    cudaMallocHost((void**)&dustdens, sizeof(real)*NUM_DIM);
    cudaMalloc((void**)&dev_dustdens, sizeof(real)*NUM_DIM);
    cudaMallocHost((void**)&optdepth, sizeof(real)*NUM_DIM);
    cudaMalloc((void**)&dev_optdepth, sizeof(real)*NUM_DIM);

    tree *dev_treenode;

    cudaMalloc((void**)&dev_treenode, sizeof(tree)*NUM_PAR);

    real *timestep, *dev_timestep;

    cudaMallocHost((void**)&timestep, sizeof(real));
    cudaMalloc((void**)&dev_timestep, sizeof(real));

    int *collrate, *dev_collrate;

    cudaMallocHost((void**)&collrate, sizeof(int));
    cudaMalloc((void**)&dev_collrate, sizeof(int));

    curandState *dev_rngstate;

    cudaMalloc((void**)&dev_rngstate, sizeof(curandState)*NUM_PAR);

    cukd::box_t<float3> *dev_boundbox;

    cudaMalloc((void**)&dev_boundbox, sizeof(cukd::box_t<float3>));

    swarm_tmp *dev_tmp_info;

    cudaMalloc((void**)&dev_tmp_info, sizeof(swarm_tmp)*NUM_PAR);

    if (argc <= 1) // no flag, start from the initial condition
	{
        resume = 0;

        real *profile_azi,  *dev_prof_azi;
        real *profile_rad,  *dev_prof_rad;
        real *profile_col,  *dev_prof_col;
        real *profile_size, *dev_prof_size;

        cudaMallocHost((void**)&profile_azi,  sizeof(real)*NUM_PAR);
        cudaMalloc((void**)&dev_prof_azi,     sizeof(real)*NUM_PAR);
        cudaMallocHost((void**)&profile_rad,  sizeof(real)*NUM_PAR);
        cudaMalloc((void**)&dev_prof_rad,     sizeof(real)*NUM_PAR);
        cudaMallocHost((void**)&profile_col,  sizeof(real)*NUM_PAR);
        cudaMalloc((void**)&dev_prof_col,     sizeof(real)*NUM_PAR);
        cudaMallocHost((void**)&profile_size, sizeof(real)*NUM_PAR);
        cudaMalloc((void**)&dev_prof_size,    sizeof(real)*NUM_PAR);

        rand_generator.seed(0); // or use rand_generator.seed(std::time(NULL));

        // rand_uniform  (profile_azi,  NUM_PAR, AZI_INIT_MIN,   AZI_INIT_MAX);
        // rand_conv_pow (profile_rad,  NUM_PAR, RAD_INIT_MIN,   RAD_INIT_MAX, IDX_SIGG - 1.0, SMOOTH_RAD, RES_RAD);
        // rand_uniform  (profile_col,  NUM_PAR, COL_INIT_MIN,   COL_INIT_MAX);
        // rand_powerlaw (profile_size, NUM_PAR, SIZE_INIT_MIN,  SIZE_INIT_MAX, -1.5);

        rand_uniform (profile_azi,  NUM_PAR, 0.0, 1.0);
        rand_uniform (profile_rad,  NUM_PAR, 0.0, 1.0);
        rand_uniform (profile_col,  NUM_PAR, 0.0, 1.0);
        rand_uniform (profile_size, NUM_PAR, 1.0e-6, 1.0e-6);

        cudaMemcpy(dev_prof_azi,  profile_azi,  sizeof(real)*NUM_PAR, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_prof_rad,  profile_rad,  sizeof(real)*NUM_PAR, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_prof_col,  profile_col,  sizeof(real)*NUM_PAR, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_prof_size, profile_size, sizeof(real)*NUM_PAR, cudaMemcpyHostToDevice);

        particle_init <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_particle, dev_prof_azi, dev_prof_rad, dev_prof_col, dev_prof_size);

        cudaFreeHost(profile_azi);  cudaFree(dev_prof_azi);
        cudaFreeHost(profile_rad);  cudaFree(dev_prof_rad);
        cudaFreeHost(profile_col);  cudaFree(dev_prof_col);
        cudaFreeHost(profile_size); cudaFree(dev_prof_size);

        // auto start = std::chrono::system_clock::now();
        // cudaDeviceSynchronize();
        // auto end = std::chrono::system_clock::now();
        // std::chrono::duration<double> elapsed_seconds = end - start;
        // std::cout << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

        // optdepth_init <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_optdepth);
        // optdepth_enum <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_optdepth, dev_particle);
        // optdepth_calc <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_optdepth);
        // optdepth_rint <<<BLOCKNUM_RAD, THREADS_PER_BLOCK>>> (dev_optdepth);
        // optdepth_mean <<<BLOCKNUM_AZI, THREADS_PER_BLOCK>>> (dev_optdepth);
        
        // dustdens_init <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_dustdens);
        // dustdens_enum <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_dustdens, dev_particle);
        // dustdens_calc <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_dustdens);
        
        // velocity_init <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_particle, dev_optdepth);

        // mkdir(OUTPUT_PATH.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        // open_txt_file(ofile, OUTPUT_PATH + "variables.txt");
        // save_variable(ofile);

        // cudaMemcpy(dustdens, dev_dustdens, sizeof(real)*NUM_DIM, cudaMemcpyDeviceToHost);
        // fname = OUTPUT_PATH + "dustdens_" + frame_num(resume) + ".bin";
        // open_bin_file(ofile, fname);
        // save_bin_file(ofile, dustdens, NUM_DIM);

        // cudaMemcpy(optdepth, dev_optdepth, sizeof(real)*NUM_DIM, cudaMemcpyDeviceToHost);
        // fname = OUTPUT_PATH + "optdepth_" + frame_num(resume) + ".bin";
        // open_bin_file(ofile, fname);
        // save_bin_file(ofile, optdepth, NUM_DIM);

        cudaMemcpy(particle, dev_particle, sizeof(swarm)*NUM_PAR, cudaMemcpyDeviceToHost);
        fname = OUTPUT_PATH + "particle_" + frame_num(resume) + ".par";
        open_bin_file(ofile, fname);
        save_bin_file(ofile, particle, NUM_PAR);
    }
    // else
    // {
    //     std::stringstream convert{argv[1]};     // set up a stringstream variable named convert, initialized with the input from argv[1]
    //     if (!(convert >> resume)) resume = -1;  // do the conversion, if conversion fails, set resume to a default value

    //     std::ifstream ifile;
    //     fname = OUTPUT_PATH + "particle_" + frame_num(resume) + ".bin";
    //     load_bin_file(ifile, fname);
    //     read_bin_file(ifile, particle, NUM_PAR);
    //     cudaMemcpy(dev_particle, particle, sizeof(swarm)*NUM_PAR,  cudaMemcpyHostToDevice);

    //     optdepth_init <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_optdepth);
    //     optdepth_enum <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_optdepth, dev_particle);
    //     optdepth_calc <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_optdepth);
    //     optdepth_rint <<<BLOCKNUM_RAD, THREADS_PER_BLOCK>>> (dev_optdepth);
    // }

    for (int i = 1 + resume; i <= OUTPUT_NUM; i++)
    {
        while (output_timer < OUTPUT_INT)
        {
            treenode_init <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_particle, dev_treenode);
            cukd::buildTree <tree, tree_traits> (dev_treenode, NUM_PAR, dev_boundbox);                                                  // 250 ms!!
            collrate_calc <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_particle, dev_tmp_info, dev_treenode, dev_collrate, dev_boundbox); //  80 ms!!
            cudaMemcpy(collrate, dev_collrate, sizeof(int), cudaMemcpyDeviceToHost);
            *timestep = -std::log(1.0 - random(rand_generator)) / static_cast<real>(*collrate);
            if (*timestep > DT_MAX) *timestep = DT_MAX;
            if (*timestep > OUTPUT_INT - output_timer) *timestep = OUTPUT_INT - output_timer;
            cudaMemcpy(dev_timestep, timestep, sizeof(real), cudaMemcpyHostToDevice);

            // ssa_substep_1 <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_particle, dev_timestep);
            // optdepth_init <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_optdepth);
            // optdepth_enum <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_optdepth, dev_particle);
            // optdepth_calc <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_optdepth);
            // optdepth_rint <<<BLOCKNUM_RAD, THREADS_PER_BLOCK>>> (dev_optdepth);
            // ssa_substep_2 <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_particle, dev_optdepth, dev_timestep);

            dustcoag_calc <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_particle, dev_tmp_info, dev_treenode, dev_timestep, dev_boundbox, dev_rngstate);
            dustsize_updt <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_particle, dev_tmp_info);

            // cudaMemcpy(timestep, dev_timestep, sizeof(real), cudaMemcpyDeviceToHost);
            output_timer += *timestep;

            std::cout << *timestep << ' ' << output_timer << std::endl;
        }
    
        output_timer = 0.0;
    
        // calculate dustdens grids for each output
        // dustdens_init <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_dustdens);
        // dustdens_enum <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_dustdens, dev_particle);
        // dustdens_calc <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_dustdens);

        // cudaMemcpy(dustdens, dev_dustdens, sizeof(real)*NUM_DIM, cudaMemcpyDeviceToHost);
        // fname = OUTPUT_PATH + "dustdens_" + frame_num(i) + ".bin";
        // open_bin_file(ofile, fname);
        // save_bin_file(ofile, dustdens, NUM_DIM);

        // calculate optical depth grids for each output
        // optdepth_init <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_optdepth);
        // optdepth_enum <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_optdepth, dev_particle);
        // optdepth_calc <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_optdepth);
        // optdepth_rint <<<BLOCKNUM_RAD, THREADS_PER_BLOCK>>> (dev_optdepth);

        // cudaMemcpy(optdepth, dev_optdepth, sizeof(real)*NUM_DIM, cudaMemcpyDeviceToHost);
        // fname = OUTPUT_PATH + "optdepth_" + frame_num(i) + ".bin";
        // open_bin_file(ofile, fname);
        // save_bin_file(ofile, optdepth, NUM_DIM);

        if (i % OUTPUT_PAR == 0)
        {
            cudaMemcpy(particle, dev_particle, sizeof(swarm)*NUM_PAR, cudaMemcpyDeviceToHost);
            fname = OUTPUT_PATH + "particle_" + frame_num(i) + ".par";
            open_bin_file(ofile, fname);
            save_bin_file(ofile, particle, NUM_PAR);
        }

        std::time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << std::setw(3) << std::setfill('0') << i << "/" << std::setw(3) << std::setfill('0') << OUTPUT_NUM << " finished on " << std::ctime(&end_time);
    }
 
    return 0;

}
