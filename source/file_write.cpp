#include <sstream>  // for stringstream
#include <fstream>  // for ofstream 
#include <iomanip>  // for setprecision, setw
#include <iostream>

#include "const.h"
#include "funclib.h"

using namespace std;

//====================================================================================================================================================================
//====================================================================================================================================================================

string int_to_string (int number)
{
    stringstream A;
    A << number;
    
    return (A.str());
}

string frame_num (int number) 
{
    string B;
    string C = int_to_string(number);
    
    if      (C.length() == 5) B =          C;
    else if (C.length() == 4) B =    "0" + C;
    else if (C.length() == 3) B =   "00" + C;
    else if (C.length() == 2) B =  "000" + C;
    else if (C.length() == 1) B = "0000" + C;
    
    return (B);
}

//====================================================================================================================================================================
//====================================================================================================================================================================

void open_bin_file (ofstream &bin_file, string file_name) 
{
    bin_file.open(file_name.c_str(), ios::out | ios::binary);
}

void save_bin_file (ofstream &bin_file, par *data, int number) 
{
    bin_file.write((char*)data, sizeof(par)*number);
    bin_file.close();
}

void save_bin_file (ofstream &bin_file, int *data, int number) 
{
    bin_file.write((char*)data, sizeof(int)*number);
    bin_file.close();
}

void save_bin_file (ofstream &bin_file, double *data, int number) 
{
    bin_file.write((char*)data, sizeof(double)*number);
    bin_file.close();
}

void load_bin_file (ifstream &bin_file, string file_name) 
{
    bin_file.open(file_name.c_str(), ios::in | ios::binary);
}

void read_bin_file (ifstream &bin_file, par *data, int number) 
{
    bin_file.read((char*)data, sizeof(par)*number);
    bin_file.close();
}

//====================================================================================================================================================================
//====================================================================================================================================================================

void open_txt_file (ofstream &txt_file, string file_name) 
{
    txt_file.open(file_name.c_str(), ios::out);
}

void aped_txt_file (ofstream &txt_file, string file_name)
{
    txt_file.open(file_name.c_str(), ios::app);
}

void save_txt_file (ofstream &txt_file, par *data, int number) 
{
    double *array = (double*)(&data[0].azi);
    
    for (int i = 0; i < number*7; i++)
    {
        if ((i + 1) % 7 == 0)
        {
            txt_file << scientific << setprecision(17) << setw(24) << setfill(' ') << array[i] << " " << endl;
        }
        else
        {
            txt_file << scientific << setprecision(17) << setw(24) << setfill(' ') << array[i] << " ";
        }
    }
    
    txt_file.close();
}

void save_txt_file (ofstream &txt_file, double *data, int number) 
{
    for (int i = 0; i < number; i++)
    {
        if ((i + 1) % RES_AZI == 0)
        {
            txt_file << data[i] << " " << endl;
        }
        else
        {
            txt_file << data[i] << " ";
        }
    }

    txt_file.close();
}

void save_txt_file (ofstream &txt_file, double data) 
{
    txt_file << data << endl;

    txt_file.close();
}

//====================================================================================================================================================================
//====================================================================================================================================================================

void save_variable (ofstream &txt_file)
{
    txt_file << "[PARAMETERS]"                                                                                              << endl;
    txt_file << "PAR_NUM = \t"          << scientific << setprecision(15) << setw(24) << setfill(' ') << PAR_NUM            << endl;
    txt_file << "RES_AZI = \t"          << scientific << setprecision(15) << setw(24) << setfill(' ') << RES_AZI            << endl;
    txt_file << "RES_RAD = \t"          << scientific << setprecision(15) << setw(24) << setfill(' ') << RES_RAD            << endl;
    txt_file << "RES_COL = \t"          << scientific << setprecision(15) << setw(24) << setfill(' ') << RES_COL            << endl;
    txt_file << "AZI_MIN = \t"          << scientific << setprecision(15) << setw(24) << setfill(' ') << AZI_MIN            << endl;
    txt_file << "AZI_MAX = \t"          << scientific << setprecision(15) << setw(24) << setfill(' ') << AZI_MAX            << endl;
    txt_file << "RAD_MIN = \t"          << scientific << setprecision(15) << setw(24) << setfill(' ') << RAD_MIN            << endl;
    txt_file << "RAD_MAX = \t"          << scientific << setprecision(15) << setw(24) << setfill(' ') << RAD_MAX            << endl;
    txt_file << "COL_MIN = \t"          << scientific << setprecision(15) << setw(24) << setfill(' ') << COL_MIN            << endl;
    txt_file << "COL_MAX = \t"          << scientific << setprecision(15) << setw(24) << setfill(' ') << COL_MAX            << endl;
    txt_file << "SIZE_MIN = \t"         << scientific << setprecision(15) << setw(24) << setfill(' ') << SIZ_INIT_MIN       << endl;
    txt_file << "SIZE_MAX = \t"         << scientific << setprecision(15) << setw(24) << setfill(' ') << SIZ_INIT_MAX       << endl;
    txt_file << "CRIT_SIZ = \t"         << scientific << setprecision(15) << setw(24) << setfill(' ') << CRITICAL_SIZ       << endl;
    txt_file << "THREADSPERBLOCK = \t"  << scientific << setprecision(15) << setw(16) << setfill(' ') << THREADSPERBLOCK    << endl;
    txt_file << "BETA =    \t"          << scientific << setprecision(15) << setw(24) << setfill(' ') << BETA               << endl;
    txt_file << "St_0 =    \t"          << scientific << setprecision(15) << setw(24) << setfill(' ') << St_0               << endl;
    txt_file << "OPACITY = \t"          << scientific << setprecision(15) << setw(24) << setfill(' ') << OPACITY            << endl;
    txt_file << "SIGMA_INDEX = \t"      << scientific << setprecision(15) << setw(24) << setfill(' ') << SIGMA_INDEX        << endl;
    txt_file << "TEMPE_INDEX = \t"      << scientific << setprecision(15) << setw(24) << setfill(' ') << TEMPE_INDEX        << endl;
    txt_file << "ASPECT_RATIO = \t"     << scientific << setprecision(15) << setw(24) << setfill(' ') << ASPECT_RATIO       << endl;
    txt_file << "OUTPUT_NUM = \t"       << scientific << setprecision(15) << setw(24) << setfill(' ') << OUTPUT_NUM         << endl;
    txt_file << "OUTPUT_INT = \t"       << scientific << setprecision(15) << setw(24) << setfill(' ') << OUTPUT_INT         << endl;
    txt_file << "TIME_STEP = \t"        << scientific << setprecision(15) << setw(24) << setfill(' ') << TIME_STEP          << endl;
    txt_file << "OUTPUT_PATH = \t"      << scientific << setprecision(15) << setw(24) << setfill(' ') << PATH               << endl;
    
    txt_file.close();
}

//====================================================================================================================================================================
//====================================================================================================================================================================
