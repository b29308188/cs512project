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
int negative_sample(std::unordered_set<int>& existing,  std::uniform_int_distribution<>& sampler ); 

/*
 * Compute the distance of two vector
 */

/*
 * Compute the graident of a norm
 */
double getDistanceGrad(int l_idx, double start, double end);

/*
 * Normalize a vector
 */
void normalize(vector<double>& vec);

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

