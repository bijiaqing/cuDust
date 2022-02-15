#include <ctime>
#include <chrono>
#include <sstream>  // for std::stringstream
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>

#include "const.h"
#include "funclib.h"
#include "funclib.cuh"

using namespace std; 

int main (int argc, char *argv[])
{
    int    resume;
    char   *fname;
    string sfname;
    
    par    *grain,    *dev_grain;
    double *density,  *dev_density;
    double *optdept,  *dev_optdept;
    double *stepnum,  *dev_stepnum;
    double *lStokes,  *dev_lStokes; // low  Stokes number (size <= 0.1 mm)
    double *hStokes,  *dev_hStokes; // high Stokes number (size >  0.1 mm)
        
    const int AZI_NUM = RES_RAD*RES_COL;
    const int RAD_NUM = RES_AZI*RES_COL;
    const int DIM_NUM = RES_AZI*RES_RAD*RES_COL;
        
    const int BLOCKNUM_PAR = PAR_NUM / THREADSPERBLOCK + 1;
    const int BLOCKNUM_DIM = DIM_NUM / THREADSPERBLOCK + 1;
    const int BLOCKNUM_AZI = AZI_NUM / THREADSPERBLOCK + 1;
    const int BLOCKNUM_RAD = RAD_NUM / THREADSPERBLOCK + 1;

    cudaMallocHost((void**)&grain,   sizeof(par)   *PAR_NUM);
    cudaMalloc((void**)&dev_grain,   sizeof(par)   *PAR_NUM);
    cudaMallocHost((void**)&density, sizeof(double)*DIM_NUM);
    cudaMalloc((void**)&dev_density, sizeof(double)*DIM_NUM);
    cudaMallocHost((void**)&optdept, sizeof(double)*DIM_NUM);
    cudaMalloc((void**)&dev_optdept, sizeof(double)*DIM_NUM);
    cudaMallocHost((void**)&stepnum, sizeof(double)*PAR_NUM);
    cudaMalloc((void**)&dev_stepnum, sizeof(double)*PAR_NUM);
    cudaMallocHost((void**)&lStokes, sizeof(double)*DIM_NUM);
    cudaMalloc((void**)&dev_lStokes, sizeof(double)*DIM_NUM);
    cudaMallocHost((void**)&hStokes, sizeof(double)*DIM_NUM);
    cudaMalloc((void**)&dev_hStokes, sizeof(double)*DIM_NUM);


    if (argc <= 1) // no flag, start from the initial condition
	{
        resume = 0;
        
        double *prof_azi, *dev_prof_azi;
        double *prof_rad, *dev_prof_rad;
        double *prof_col, *dev_prof_col;
        double *prof_siz, *dev_prof_siz;

        cudaMallocHost((void**)&prof_azi, sizeof(double)*PAR_NUM);
        cudaMalloc((void**)&dev_prof_azi, sizeof(double)*PAR_NUM);
        cudaMallocHost((void**)&prof_rad, sizeof(double)*PAR_NUM);
        cudaMalloc((void**)&dev_prof_rad, sizeof(double)*PAR_NUM);
        cudaMallocHost((void**)&prof_col, sizeof(double)*PAR_NUM);
        cudaMalloc((void**)&dev_prof_col, sizeof(double)*PAR_NUM);
        cudaMallocHost((void**)&prof_siz, sizeof(double)*PAR_NUM);
        cudaMalloc((void**)&dev_prof_siz, sizeof(double)*PAR_NUM);
        
      //srand(time(NULL));
        srand(0);
        
        uniform_generator (prof_azi, AZI_INIT_MIN, AZI_INIT_MAX);
        convpow_generator (prof_rad, RAD_INIT_MIN, RAD_INIT_MAX);
        gaussym_generator (prof_col, COL_INIT_MIN, COL_INIT_MAX);    
        pow_law_generator (prof_siz, SIZ_INIT_MIN, SIZ_INIT_MAX, GSIZE_INDEX);
        
        cudaMemcpy(dev_prof_azi, prof_azi, sizeof(double)*PAR_NUM, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_prof_rad, prof_rad, sizeof(double)*PAR_NUM, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_prof_col, prof_col, sizeof(double)*PAR_NUM, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_prof_siz, prof_siz, sizeof(double)*PAR_NUM, cudaMemcpyHostToDevice);
        
        position_init <<<BLOCKNUM_PAR, THREADSPERBLOCK>>> (dev_grain, dev_prof_azi, dev_prof_rad, dev_prof_col, dev_prof_siz);
        
        density_init  <<<BLOCKNUM_DIM, THREADSPERBLOCK>>> (dev_density, dev_lStokes, dev_hStokes);
        density_enum  <<<BLOCKNUM_PAR, THREADSPERBLOCK>>> (dev_density, dev_lStokes, dev_hStokes, dev_grain);
        density_calc  <<<BLOCKNUM_DIM, THREADSPERBLOCK>>> (dev_density, dev_lStokes, dev_hStokes);
        
        optdept_init  <<<BLOCKNUM_DIM, THREADSPERBLOCK>>> (dev_optdept);
        optdept_enum  <<<BLOCKNUM_PAR, THREADSPERBLOCK>>> (dev_optdept, dev_grain);
        optdept_calc  <<<BLOCKNUM_DIM, THREADSPERBLOCK>>> (dev_optdept);
        optdept_cumu  <<<BLOCKNUM_RAD, THREADSPERBLOCK>>> (dev_optdept);
        optdept_mean  <<<BLOCKNUM_AZI, THREADSPERBLOCK>>> (dev_optdept);
        
        velocity_init <<<BLOCKNUM_PAR, THREADSPERBLOCK>>> (dev_grain, dev_stepnum);
        
        mkdir(PATH.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        
        ofstream ofile_params;
        open_txt_file(ofile_params, PATH + "parameters.ini");
        save_variable(ofile_params);
        
        cudaMemcpy(density, dev_density, sizeof(double)*DIM_NUM, cudaMemcpyDeviceToHost);
        
        ofstream ofile_dens;
        open_bin_file(ofile_dens, PATH + "density_00000.bin");
        save_bin_file(ofile_dens, density, DIM_NUM);
        /*
        cudaMemcpy(lStokes, dev_lStokes, sizeof(double)*DIM_NUM, cudaMemcpyDeviceToHost);
        
        ofstream ofile_loSt;
        open_bin_file(ofile_loSt, PATH + "lStokes_00000.bin");
        save_bin_file(ofile_loSt, lStokes, DIM_NUM);
        
        cudaMemcpy(hStokes, dev_hStokes, sizeof(double)*DIM_NUM, cudaMemcpyDeviceToHost);
        
        ofstream ofile_hiSt;
        open_bin_file(ofile_hiSt, PATH + "hStokes_00000.bin");
        save_bin_file(ofile_hiSt, hStokes, DIM_NUM);
        */
        cudaMemcpy(optdept, dev_optdept, sizeof(double)*DIM_NUM, cudaMemcpyDeviceToHost);
        
        ofstream ofile_odep;
        open_bin_file(ofile_odep, PATH + "optdepth_00000.bin");
        save_bin_file(ofile_odep, optdept, DIM_NUM);
        
        cudaMemcpy(grain, dev_grain, sizeof(par)*PAR_NUM, cudaMemcpyDeviceToHost);
            
        ofstream ofile_dust;
        open_bin_file(ofile_dust, PATH + "grain_00000.bin");
        save_bin_file(ofile_dust, grain, PAR_NUM);
    }
        
    else
    {
        stringstream convert{argv[1]}; // set up a stringstream variable named convert, initialized with the input from argv[1]
 
	    if (!(convert >> resume)) // do the conversion
		    resume = 0;           // if conversion fails, set myint to a default value
		    
		sfname = PATH + "grain_" + frame_num(resume) + ".bin";
        fname  = new char[sfname.size() + 1];
        strcpy(fname, sfname.c_str());
		    
		ifstream ifile_dust;
        load_bin_file(ifile_dust, fname);
        read_bin_file(ifile_dust, grain, PAR_NUM);
        
        cudaMemcpy(dev_grain, grain, sizeof(par)*PAR_NUM, cudaMemcpyHostToDevice);

        optdept_init <<<BLOCKNUM_DIM, THREADSPERBLOCK>>> (dev_optdept);
        optdept_enum <<<BLOCKNUM_PAR, THREADSPERBLOCK>>> (dev_optdept, dev_grain);
        optdept_calc <<<BLOCKNUM_DIM, THREADSPERBLOCK>>> (dev_optdept);
        optdept_cumu <<<BLOCKNUM_RAD, THREADSPERBLOCK>>> (dev_optdept);
      //optdept_mean <<<BLOCKNUM_AZI, THREADSPERBLOCK>>> (dev_optdept);
    }
    
    for (int i = 1 + resume; i <= OUTPUT_NUM; i++)
    {
        for (int j = 0; j < OUTPUT_INT / TIME_STEP; j++)
        {
            parmove_act1 <<<BLOCKNUM_PAR, THREADSPERBLOCK>>> (dev_grain);
            optdept_init <<<BLOCKNUM_DIM, THREADSPERBLOCK>>> (dev_optdept);
            optdept_enum <<<BLOCKNUM_PAR, THREADSPERBLOCK>>> (dev_optdept, dev_grain);
            optdept_calc <<<BLOCKNUM_DIM, THREADSPERBLOCK>>> (dev_optdept);
            optdept_cumu <<<BLOCKNUM_RAD, THREADSPERBLOCK>>> (dev_optdept);
          //optdept_mean <<<BLOCKNUM_AZI, THREADSPERBLOCK>>> (dev_optdept);
            parmove_act2 <<<BLOCKNUM_PAR, THREADSPERBLOCK>>> (dev_grain, dev_optdept);
        }
        
        // calculate density grids for each output
        density_init <<<BLOCKNUM_DIM, THREADSPERBLOCK>>> (dev_density, dev_lStokes, dev_hStokes);
        density_enum <<<BLOCKNUM_PAR, THREADSPERBLOCK>>> (dev_density, dev_lStokes, dev_hStokes, dev_grain);
        density_calc <<<BLOCKNUM_DIM, THREADSPERBLOCK>>> (dev_density, dev_lStokes, dev_hStokes);
        
        // calculate optical depth grids for each output
        optdept_init <<<BLOCKNUM_DIM, THREADSPERBLOCK>>> (dev_optdept);
        optdept_enum <<<BLOCKNUM_PAR, THREADSPERBLOCK>>> (dev_optdept, dev_grain);
        optdept_calc <<<BLOCKNUM_DIM, THREADSPERBLOCK>>> (dev_optdept);
        optdept_cumu <<<BLOCKNUM_RAD, THREADSPERBLOCK>>> (dev_optdept);
        
        cudaMemcpy(density, dev_density, sizeof(double)*DIM_NUM, cudaMemcpyDeviceToHost);

        sfname = PATH + "density_" + frame_num(i) + ".bin";
        fname  = new char[sfname.size() + 1];
        strcpy(fname, sfname.c_str());
        
        ofstream ofile_dens;
        open_bin_file(ofile_dens, fname);
        save_bin_file(ofile_dens, density, DIM_NUM);
        /*
        cudaMemcpy(lStokes, dev_lStokes, sizeof(double)*DIM_NUM, cudaMemcpyDeviceToHost);
        
        sfname = PATH + "lStokes_" + frame_num(i) + ".bin";
        fname  = new char[sfname.size() + 1];
        strcpy(fname, sfname.c_str());
        
        ofstream ofile_loSt;
        open_bin_file(ofile_loSt, fname);
        save_bin_file(ofile_loSt, lStokes, DIM_NUM);
        
        cudaMemcpy(hStokes, dev_hStokes, sizeof(double)*DIM_NUM, cudaMemcpyDeviceToHost);
        
        sfname = PATH + "hStokes_" + frame_num(i) + ".bin";
        fname  = new char[sfname.size() + 1];
        strcpy(fname, sfname.c_str());
        
        ofstream ofile_hiSt;
        open_bin_file(ofile_hiSt, fname);
        save_bin_file(ofile_hiSt, hStokes, DIM_NUM);
        */
        cudaMemcpy(optdept, dev_optdept, sizeof(double)*DIM_NUM, cudaMemcpyDeviceToHost);
            
        sfname = PATH + "optdepth_" + frame_num(i) + ".bin";
        fname  = new char[sfname.size() + 1];
        strcpy(fname, sfname.c_str());
            
        ofstream ofile_odep;
        open_bin_file(ofile_odep, fname);
        save_bin_file(ofile_odep, optdept, DIM_NUM);
        
        if (i % OGRAIN_INT == 0) 
        {
            cudaMemcpy(grain, dev_grain, sizeof(par)*PAR_NUM, cudaMemcpyDeviceToHost);
        
            sfname = PATH + "grain_" + frame_num(i) + ".bin";
            fname  = new char[sfname.size() + 1];
            strcpy(fname, sfname.c_str());
        
            ofstream ofile_dust;
            open_bin_file(ofile_dust, fname);
            save_bin_file(ofile_dust, grain, PAR_NUM);
            
            /*stepnum_calc <<<BLOCKNUM_PAR, THREADSPERBLOCK>>> (dev_stepnum, dev_grain);
            
            cudaMemcpy(stepnum, dev_stepnum, sizeof(double)*PAR_NUM, cudaMemcpyDeviceToHost);

            sfname = PATH + "stepnum_" + frame_num(i) + ".bin";
            fname  = new char[sfname.size() + 1];
            strcpy(fname, sfname.c_str());
        
            ofstream ofile_stepnum;
            open_bin_file(ofile_stepnum, fname);
            save_bin_file(ofile_stepnum, stepnum, PAR_NUM);*/
        }
        
        time_t end_time = chrono::system_clock::to_time_t(chrono::system_clock::now());
        cout << setw(3) << setfill('0') << i << "/" << setw(3) << setfill('0') << OUTPUT_NUM << " finished on " << ctime(&end_time);
    }
 
    return 0;

}
