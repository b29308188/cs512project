#include "utils.hpp"


int negative_sample(std::unordered_set<int>& existing,  std::uniform_int_distribution<>& sampler ) {

	auto sample_num = sampler(Utils::generator);
	while(existing.find(sample_num) != existing.end())
		sample_num = sampler(gen);
	return sample_num;
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

void normalize(std::vector<double>& vec) {
  auto squared_sum = std::inner_product( vec.begin(), vec.end(), vec.begin(), 0);
  std::transform(vec.begin(), vec.end(), vec.begin(), [](double elem) { 
                                            return elem/squared_sum; 
                                            });
}



double computeDist(int norm_flag, const std::vector<double>& vec1, const std::vector<double>& vec2) {
  double sum = 0.0;
  auto diffVec = vec1-vec2;
  
  if ( norm_flag == 1) 
    sum = std::accumulate(diffVec.begin(), diffVec.end(), 0.0, [](double elem){
          return fabs(elem);
        };)
  else if (norm_flag == 2)
    sum = std::accumulate(diffVec.begin(), diffVec.end(), 0.0, [](double elem) {
          return elem*elem;
        };)

  return sum; 
}
