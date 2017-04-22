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
 * This function will sample in [0, range) for an integer not in existing 
 */
int negative_sample(const std::unordered_set<size_t>& existing,  std::uniform_int_distribution<int>& sampler ); 

/*
 * Compute the distance of two vector
 */
double computeDist(int norm_flag, const std::vector<double>& vec1, const std::vector<double>& vec2);
/*
 * Compute the graident of a norm
 */
double getDistanceGrad(int l_idx, double start, double end);

/*
 * Normalize a vector
 */
void normalize(std::vector<double>& vec);

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


/*************************/
/*    File processing    */
/*************************/
/*
 * Count number of lines of a file
 */
size_t countLine(std::ifstream& ifs);
