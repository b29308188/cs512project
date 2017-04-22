#include "utils.hpp"
int negative_sample(const std::unordered_set<size_t>& existing,  std::uniform_int_distribution<>& sampler ) {

	auto sample_num = sampler(Utils::generator);
	while(existing.find(sample_num) != existing.end())
		sample_num = sampler(Utils::generator);
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
  std::transform(vec.begin(), vec.end(), vec.begin(), [&](double elem) { 
                                            return elem/squared_sum; 
                                            });
}


double computeDist(int norm_flag, const std::vector<double>& vec1, const std::vector<double>& vec2) {
  double sum = 0.0;
  auto diffVec = vec1-vec2;
  
  if ( norm_flag == 1) 
    sum = std::accumulate(diffVec.begin(), diffVec.end(), 0.0, [](double curr, double elem){
          return curr+fabs(elem);
        });
  else if (norm_flag == 2)
    sum = std::accumulate(diffVec.begin(), diffVec.end(), 0.0, [](double curr, double elem) {
          return curr+elem*elem;
        });

  return sum; 
}

size_t countLine(std::istream& is) {
    if(is.bad())
        throw std::runtime_error("counting line. Input stream is bad");
    // keep the old sate 
    std::istream::iostate oState = is.rdstate();
    // clear the state
    is.clear();
    std::istream::streampos oPos = is.tellg();
    // count the line
    auto numLines = std::count(std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>(), '\n');
    // see if the file end with \n, if not, add one line
    is.unget();
    if( is.get()!= '\n') numLines ++;
    // reset all state
    is.clear();
    is.seekg(oPos);
    is.setstate(oState);
    return numLines;
}
