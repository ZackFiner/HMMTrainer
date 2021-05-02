#pragma once
#include <string>

class HMM;
void pickle_hmm(HMM* hmm, std::string fpath);
void initialize_hmm(HMM* hmm, std::string fpath);
HMM load_hmm(std::string fpath);