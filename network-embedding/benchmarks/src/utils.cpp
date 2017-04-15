#include "utils.hpp"


int negative_sample(std::unordered_set<int>& existing)//,  std::uniform_int_distribution<>& sampler ) {
//
//	std::random_device rd;
//	std::mt19937 gen(rd);
//  auto sample_num = sampler(gen);
//  while(existing.find(sample_num) != existing.end())
//    sample_num = sampler(gen);
// return sample_num;
}

double getDistanceGrad(int l_idx, double start, double end) {
  double rev = 2*(end-start);
  // l1 norm
  if (l_idx == 1) {
    return rev >0? 1:-1;
  }
  else
    return rev;
}

