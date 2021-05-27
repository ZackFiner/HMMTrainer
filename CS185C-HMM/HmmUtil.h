#pragma once
#include <string>

class HMM;
void pickleHmm(HMM* hmm, std::string fpath);
void initializeHmm(HMM* hmm, std::string fpath);
HMM loadHmm(std::string fpath);