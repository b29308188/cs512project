#pragma once
#include "utils.hpp"
using namespace std;


enum class SamplingMethod {
  UniformDistribution,
  BernouliDistribution,
  GaussianDistribution
};


// (head, tail, relationship)
using triplet_t = tuple<size_t, size_t, size_t>;

class Train {
public:
  Train(size_t e_dimension, size_t r_dimension, double learning_rate, double margin,SamplingMethod s);
  /*
   * Read pre-processed data
   */
  void readData();

  /*
   * Traverse all relationship and update   
   */
  void run();

  /*
   * Write to file
   */
  void writeData();
/*
 * private functions
 */
private:
  // update a mini batch
  void batchUpdate();
  // update weight
  void weightUpdate();

  void computeGraident();

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
  vector<vector<int> > feature_;
  vector<vector<double> > relation_mat_, entity_mat_;
  vector<vector<double> > relation_tmp_, entity_tmp_;

  string relation_file_name = "relation_mat";  
  string entity_file_name = "entity_vec";

  uniform_int_distribution<> entity_sampler_;
  uniform_int_distribution<> triplet_sampler_;

  double loss_ = 0.0;

  int norm_flag_ = 2;
};

