#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <random>
#include <iomanip>
#include <algorithm>
#include <stdexcept>

namespace Utils {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> coin(1,2);
}

/*
 * Reshape a matrix
 */
template <typename T>
void reshape(std::vector<std::vector<T>>& mat, int row, int col){
	mat.resize(row);
	for(auto& each_row : mat)
		each_row.resize(col);
}

/* 
 * Arithmetic operation for vector
 */
template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
    if(a.size() != b.size())
      throw std::domain_error("adding vectors with different dimensions");

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(), 
                   std::back_inserter(result), std::plus<T>());
    return result;
}

template <typename T>
std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b)
{
    if(a.size() != b.size())
      throw std::domain_error("subtracting vectors with different dimensions");

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(), 
                   std::back_inserter(result), std::minus<T>());
    return result;
}
/* 
 * This function will sample in [0, range) for an integer not in existing 
 */
int negative_sample(const std::unordered_set<size_t>& existing,  std::uniform_int_distribution<>& sampler ) {

	auto sample_num = sampler(Utils::generator);
	while(existing.find(sample_num) != existing.end())
		sample_num = sampler(Utils::generator);
	return sample_num;
}
/*
 * Compute the distance of two vector
 */
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
/*
 * Compute the graident of a norm
 */
double getDistanceGrad(int l_idx, double start, double end) {
  double rev = 2*(end-start);
  // l1 norm
  if (l_idx == 1) {
    return rev >0? 1:-1;
  }
  else
    return rev;
}

/*
 * Normalize a vector
 */
void normalize(std::vector<double>& vec) {
  auto squared_sum = std::inner_product( vec.begin(), vec.end(), vec.begin(), 0);
  std::transform(vec.begin(), vec.end(), vec.begin(), [&](double elem) { 
                                            return elem/squared_sum; 
                                            });
}



/*************************/
/*    File processing    */
/*************************/
/*
 * Count number of lines of a file
 */
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
