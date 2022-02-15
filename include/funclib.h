#ifndef FUNCLIB_H 
#define FUNCLIB_H

#include <fstream>

#include "const.h"

using namespace std;

//====================================================================================================================================================================
// Files open and save

string frame_num (int number);

void open_bin_file (ofstream &bin_file, string file_name);
void load_bin_file (ifstream &bin_file, string file_name);
void save_bin_file (ofstream &bin_file, par    *data, int number);
void save_bin_file (ofstream &bin_file, int    *data, int number);
void save_bin_file (ofstream &bin_file, double *data, int number);
void read_bin_file (ifstream &bin_file, par *data,    int number);

void open_txt_file (ofstream &txt_file, string file_name);
void aped_txt_file (ofstream &txt_file, string file_name);
void save_txt_file (ofstream &txt_file, par    *data, int number);
void save_txt_file (ofstream &txt_file, double *data, int number);
void save_txt_file (ofstream &txt_file, double  data);

void save_variable (ofstream &txt_file);

//====================================================================================================================================================================
// Profile generators

void sigmoid_generator (double *prof, double p_min, double p_max);
void convpow_generator (double *prof, double p_min, double p_max);
void uniform_generator (double *prof, double p_min, double p_max);
void gaussym_generator (double *prof, double p_min, double p_max);
void pow_law_generator (double *prof, double p_min, double p_max, double index);

//====================================================================================================================================================================
#endif
