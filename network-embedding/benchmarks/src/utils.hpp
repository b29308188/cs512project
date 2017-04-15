#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <random>

/* 
 * This function will sample in [0, range) for an integer not in existing 
 */
int negative_sample(std::unordered_set<int>& existing);//,  std::uniform_int_distribution<>& sampler ); 

/*
 * Compute the graident of a norm
 */
double getDistanceGrad(int l_idx, double start, double end);
