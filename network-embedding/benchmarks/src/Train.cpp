#include "Train.hpp"

Train::Train(size_t e_dimension, size_t r_dimension, double learning_rate, double margin,SamplingMethod s):
  e_dimension_(e_dimension), r_dimension_(r_dimension), learning_rate_(learning_rate), margin_(margin), sampling_m_(s)
{
}

void Train::run() {
  // prepare random device
  entity_sampler_ = uniform_int_distribution<>(0, entity_mat_.size());
  triplet_sampler_ = uniform_int_distribution<>(0, triplets_.size());

  const int batchsize = triplets_.size();
  loss_= 0.0;

  // for each epoch
  for ( size_t epoch_idx = 0; epoch_idx < num_epoch_; epoch_idx++ ){

    loss_= 0.0;
    // for each batch
    for ( size_t batch_idx = 0; batch_idx < num_batches_; batch_idx ++) {
      // keep the current value of realtion vectors and entity vectors
      relation_tmp_ = relation_mat_;
      entity_tmp_ = entity_mat_;
      batchUpdate();
    } //end for each batch 
    // update the embedding after each batch update
    relation_mat_ = relation_tmp_;
    entity_mat_ = entity_tmp_;

    cout << "Epoch: " << epoch_idx << " " << loss_<< endl;
  } // end for each epoch
}


void Train::batchUpdate(){

  for ( size_t idx; idx < batchsize; idx ++ ) {
    // sample a triplet
    auto triplet_idx = triplet_sampler(gen);
    // get the head and the tail
    auto head_id = std::get<0> triplets_[triplet_idx];
    auto tail_id = std::get<1> triplets_[triplet_idx];
    auto relation_id = std::get<2> tripletes_[triplet_idx];

    // throw a coin to decide to remove head_id or remove tail_id
    auto coin_num = Utils::coin(Utils::generator);
    const auto& exist_entities = coin_num? relation_id_table[ make_pair(head_id, relation_id)]:inverse_relation_id_table[make_pair(tail_id, relation_id)]
    size_t replace_id = negative_sample(exist_entities, entity_sampler);
    // replace tail
    if (coin_num == 0) 
      weightUpdate(head_id, tail_id, relation_id, head_id, replace_id, relation_id);
    // replace head
    else 
      weightUpdate(head_id, tail_id, relation_id, replace_id, tail_id, relation_id);

    normalize(relation_tmp_[relation_id]);
    normalize(entity_tmp_[head_id]);
    normalize(entity_tmp_[tail_id]);
    normalize(entity_tmp_[replace_id]);
  }
}


void Train::weightUpdate(size_t head_id, size_t tail_id, size_t relation_id, 
                  size_t comp_head_id, size_t comp_tail_id, size_t comp_relation_id) {
  auto& head_vec = entity_mat_[head_id];
  auto& tail_vec = entity_mat_[tail_id];
  auto& relation_vec = relation_mat_[relation_id];

  auto& comp_head = entity_mat_[comp_head_id];
  auto& comp_tail = entity_mat_[comp_tail_id];
  auto& comp_relation = relation_mat_[comp_head_id];

  auto sum1 = computeDist( 2, tail_vec, head_vec + relation_vec);
  auto sum2 = computeDist( 2, comp_tail, comp_head + compe_relation);
  if ( sum1 + margin_ > sum2) {
    loss_+= margin_ + sum1 - sum2;
    computeGradient( head_id, tail_id, relation_id, comp_head_id, comp_tail_id, comp_relation_id);
  }
}

void  Train::computeGraident(size_t head_id, size_t tail_id, size_t relation_id, 
                      size_t comp_head_id, size_t comp_tail_id, size_t comp_relation_id) {
  auto& head_vec = entity_mat_[head_id];
  auto& tail_vec = entity_mat_[tail_id];
  auto& relation_vec = relation_mat_[relation_id];

  auto& comp_head = entity_mat_[comp_head_id];
  auto& comp_tail = entity_mat_[comp_tail_id];
  auto& comp_relation = relation_mat_[comp_head_id];

  for ( int idx = 0 ; idx < e_dimension_ ; idx++ ) {
    double distGrad = getDistanceGrad(norm_flag_, tail_vec[idx], head_vec[idx] + relation_vec[idx]);
    double diff = distGrad*-1*learning_rate_;

    relation_tmp_[relation_id][idx] -= diff;
    entity_tmp_[head_id][idx] -= diff;
    entity_tmp_[tail_id][idx] += diff;

    double comp_distGrad = getDistanceGrad(norm_flag_, comp_tail[idx], comp_head[idx] + comp_relation[idx]);

    double comp_diff = comp_distGrad*-1*learning_rate_;

    relation_tmp_[comp_relation_id][idx] -= comp_diff;
    entity_tmp_[comp_head_id][idx] -= comp_diff;
    entity_tmp_[comp_tail_id][idx] += comp_diff;
  }
}


void Train::writeData() {
  // open file for writing
  ofstream relation_file, entity_file;
  relation_file.open(relation_file_name, ios::out|ios::trunc);
  entity_file.open(relation_file_name, ios::out|ios::trunc);
  // print out each realtion vector
  for( auto relation_vec : relation_mat_) {
    for ( auto element : relation_vec)
      relation_file << setprecision(7) << element << '\t' ;
    relation_file << endl;
  }
  // print out each entity vector
  for ( auto entity_vec : entity_mat_ ) {
    for ( auto element : entity_vec) 
      entity_file << setprecision(7) << element << '\t';
    entity_file << endl;
  }
}
