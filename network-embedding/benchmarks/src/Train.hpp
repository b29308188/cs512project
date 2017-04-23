#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <random>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <exception>

using namespace std;


enum class SamplingMethod {
  UniformDistribution,
  BernouliDistribution,
  GaussianDistribution
};


// (head, tail, relationship)
using triplet_t = tuple<size_t, size_t, size_t>;
using features_t = std::vector<double>;

class Train {
public:
	Train(size_t e_dimension, size_t r_dimension, 
			double learning_rate, double margin,SamplingMethod s): e_dimension_(e_dimension), r_dimension_(r_dimension), learning_rate_(learning_rate), margin_(margin), sampling_m_(s) {}
  /*
   * Read pre-processed data
   */
  void readData(const string& relation_file_name, const string& entity_file_name, const string& network_file_name);

  /*
   * Traverse all relationship and update   
   */
  void run();

  /*
   * Write to file
   */
  void writeData(const string& relation_file, const string& entity_file);
/*
 * private functions
 */
private:
  // read weight vectors from a given file
  void readWeights(ifstream& ifs, std::vector<features_t>& dataMat, const size_t& dimension);
  // read network data
  void readNetwork(ifstream& ifs);
  // update a mini batch
  void batchUpdate();
  // update weight
  void weightUpdate(size_t heada, size_t taila, size_t rela, size_t headb, size_t tailb, size_t relb);

  void updateGradient(features_t& head_vec, features_t& tail_vec, features_t& relation_vec);

/*
 * Data
 */
private:

  size_t e_dimension_ = 0;
  size_t r_dimension_ = 0;
  double learning_rate_ = 0.1;
  double margin_ = 1;

  size_t num_batches_ = 100;
  size_t num_epoch_ = 1000;
  SamplingMethod sampling_m_ = SamplingMethod::UniformDistribution;
  
  vector<triplet_t> triplets_; 
  vector<vector<int>> feature_;
  vector<features_t> relation_mat_, entity_mat_;
  vector<features_t> relation_tmp_, entity_tmp_;

  string relation_file_name = "relation_mat";  
  string entity_file_name = "entity_vec";

  uniform_int_distribution<int> entity_sampler_;
  uniform_int_distribution<int> triplet_sampler_;

  double loss_ = 0.0;
  int norm_flag_ = 2;

  using uset_t = unordered_set<size_t>;
  using table_t = unordered_map<size_t, uset_t>;

  unordered_map<size_t, table_t> relation_id_table;
  unordered_map<size_t, table_t> inverse_relation_id_table;
};

