#include "Train.hpp"
#include "utils.hpp"

void Train::readData(const std::string& relation_file_name, const std::string& entity_file_name, const std::string& network_file_name) {
  ifstream relation_file_handler, entity_file_handler, network_file_handler;
  // read networks
  network_file_handler.open(network_file_name, std::ifstream::in);
  readNetwork(network_file_handler);
  // relation reading
  relation_file_handler.open(relation_file_name, std::ifstream::in);
  readWeights(relation_file_handler, relation_mat_, r_dimension_);
  relation_file_handler.close();
  // entity reading
  entity_file_handler.open(entity_file_name, std::ifstream::in);
  readWeights(entity_file_handler, entity_mat_, e_dimension_);
  entity_file_handler.close();
}

void Train::readNetwork(ifstream& fileHandler) {
	auto numLines = countLine(fileHandler);
	triplets_.reserve(numLines);
	std::string eachLine;
	while(std::getline(fileHandler, eachLine)) {
		std::istringstream inputStream(eachLine); 
		size_t headID, tailID, relationID;
		triplets_.emplace_back( std::make_tuple(headID, tailID, relationID));
	}
	cout << "length is " << triplets_.size() << endl;
}

void Train::readWeights(ifstream& fileHandler, std::vector<features_t>& dataMat, const size_t& dimension) {
    std::string eachLine;
    // count the total number of line
    // recall that countLine will recover state for you, dont worry
    auto numLines = countLine(fileHandler);
    dataMat.reserve(numLines);
    while (std::getline(fileHandler, eachLine)) {
        std::istringstream inputStream(eachLine);
        features_t newFeatureVec(dimension, 0.0);
        for (size_t i = 0 ; i < dimension ; ++i ) {
            inputStream >> newFeatureVec[i];
        }
        dataMat.push_back(newFeatureVec);
    }
}

void Train::run() {
  // prepare random device
  entity_sampler_ = uniform_int_distribution<>(0, entity_mat_.size());
  triplet_sampler_ = uniform_int_distribution<>(0, triplets_.size());

  loss_= 0.0;

  // for each epoch
  for ( size_t epoch_idx = 0; epoch_idx < num_epoch_; epoch_idx++ ){
    loss_= 0.0;
    // for each batch
    for ( size_t batch_idx = 0; batch_idx < num_batches_; batch_idx ++) {
      // keep the current value of relation std::vectors and entity vectors
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

  auto batchsize = triplets_.size()/num_batches_;

  for ( size_t idx = 0; idx < batchsize; idx ++ ) {
    // sample a triplet
    auto triplet_idx = triplet_sampler_(Utils::generator);
    // get the head and the tail
    size_t head_id, tail_id, relation_id;
    std::tie(head_id, tail_id, relation_id) = triplets_[triplet_idx];

      // throw a coin to decide to remove head_id or remove tail_id
    auto coin_num = Utils::coin(Utils::generator);

    const auto& exist_entities = 
      coin_num? relation_id_table[head_id][relation_id]:inverse_relation_id_table[tail_id][relation_id];

    size_t replace_id = negative_sample(exist_entities, entity_sampler_);
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

  auto sum1 = computeDist( norm_flag_, tail_vec, head_vec + relation_vec);
  auto sum2 = computeDist( norm_flag_, comp_tail, comp_head + comp_relation);

  // if too far, compute gradient and do update 
  if ( sum1 + margin_ > sum2) {
    loss_+= margin_ + sum1 - sum2;
    updateGradient( head_vec, tail_vec, relation_vec);
    updateGradient( comp_head, comp_tail, comp_relation);
  }
}

void Train::updateGradient(features_t& head_vec, features_t& tail_vec, features_t& relation_vec) {
	// First, compute difference
	auto gradVec = tail_vec - ( head_vec + relation_vec);
	// Transform from difference to gradient
	std::transform(gradVec.begin(), gradVec.end(), gradVec.begin(), 
			[this] (double& val) {
				if(norm_flag_ == 1)
					return (val>0) ? 1.0:-1.0;
				else
					return 2*val;
			}
			);	

	// Do update on each std::vector
	// For each element, minus gradient
	auto minusFunc = [this](double& grad, double& elem) {
		auto update = -1*grad*learning_rate_;
		return elem - update;
	};
	// For each element, plus graident
	auto plusFunc = [this](double& grad, double& elem) {
		auto update = -1*grad*learning_rate_;
		return elem + update;
	};

	std::transform(gradVec.begin(), gradVec.end(), 
				relation_vec.begin(), relation_vec.begin(), minusFunc);

	std::transform(gradVec.begin(), gradVec.end(), 
				head_vec.begin(), head_vec.begin(), minusFunc);

	std::transform(gradVec.begin(), gradVec.end(),
			tail_vec.begin(), tail_vec.begin(), plusFunc);
}


void Train::writeData(const string& relation_file_name, const string& entity_file_name) {
  // open file for writing
  ofstream relation_file, entity_file;
  relation_file.open(relation_file_name, ios::out|ios::trunc);
  entity_file.open(relation_file_name, ios::out|ios::trunc);
  // print out each relation std::vector
  for( auto relation_vec : relation_mat_) {
    for ( auto element : relation_vec)
      relation_file << setprecision(7) << element << '\t' ;
    relation_file << endl;
  }
  // print out each entity std::vector
  for ( auto entity_vec : entity_mat_ ) {
    for ( auto element : entity_vec) 
      entity_file << setprecision(7) << element << '\t';
    entity_file << endl;
  }
}
