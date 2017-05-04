#include "Train.hpp"
#include "utils.hpp"
#include <cmath>

void containsNaN(vector<vector<vector<double>>> tocheck) {
	for (auto& eachMat : tocheck)
		for (auto& eachvec : eachMat)
			for(auto& elem : eachvec)
				if (isnan(elem) || isinf(elem)) {
					cout << "Element is nan " << endl;
					abort();
				}
}

void Train::readData(const std::string& relation_file_name, const std::string& entity_file_name, const std::string& network_file_name) {
  cout << "start reading networks" << endl;
  ifstream relation_file_handler(network_file_name);
  ifstream entity_file_handler(entity_file_name);
  ifstream network_file_handler(network_file_name);
  // read networks
  if (!network_file_handler)
    throw runtime_error("file doesnt exist");
  cout << "reding networks ... " << endl;
  readNetwork(network_file_handler);
  // relation reading
  if (!relation_file_handler)
    throw runtime_error("file doesnt exist");
  cout << "reding weights from " << relation_file_name << " ... " << endl;
  readWeights(relation_file_handler, relation_mat_, r_dimension_);
  cout << "relation data size: " << relation_mat_.size() << " x " << relation_mat_[0].size() <<endl;
  relation_file_handler.close();
  // entity reading
  if (!entity_file_handler)
    throw runtime_error("file doesnt exist");
  cout << "reding weights from " << entity_file_name << " ... " << endl;
  readWeights(entity_file_handler, entity_mat_, e_dimension_);
  cout << "entity data size: " << entity_mat_.size() << " x " << relation_mat_[0].size() <<endl;
  entity_file_handler.close();
}

void Train::readNetwork(ifstream& fileHandler) {
	auto numLines = countLine(fileHandler);
	triplets_.reserve(numLines);
	std::string eachLine;
	while(std::getline(fileHandler, eachLine)) {
		std::istringstream inputStream(eachLine); 
		size_t headID, tailID, relationID;
		inputStream >> headID >> tailID >> relationID;
		triplets_.emplace_back( std::make_tuple(headID, tailID, relationID));
	}
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
  cout << "running ... " <<endl;
  // prepare random device
  entity_sampler_ = uniform_int_distribution<>(0, entity_mat_.size()-1);
  triplet_sampler_ = uniform_int_distribution<>(0, triplets_.size()-1);

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
	  relation_mat_ = relation_tmp_;
	  entity_mat_ = entity_tmp_;
    } //end for each batch 
    // update the embedding after each batch update

    cout << "epoch: " << epoch_idx << " loss: " << loss_<< endl;
	writeSnapshot();
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


char sp = ' ';
void Train::weightUpdate(size_t head_id, size_t tail_id, size_t relation_id, 
                  size_t comp_head_id, size_t comp_tail_id, size_t comp_relation_id) {
  auto& head_vec = entity_mat_[head_id];
  auto& tail_vec = entity_mat_[tail_id];
  auto& relation_vec = relation_mat_[relation_id];

  auto& head_tmp = entity_tmp_[head_id];
  auto& tail_tmp = entity_tmp_[tail_id];
  auto& relation_tmp = relation_tmp_[relation_id];

  auto& comp_head = entity_mat_[comp_head_id];
  auto& comp_tail = entity_mat_[comp_tail_id];
  auto& comp_relation = relation_mat_[comp_relation_id];

  auto& comp_head_tmp = entity_tmp_[comp_head_id];
  auto& comp_tail_tmp = entity_tmp_[comp_tail_id];
  auto& comp_relation_tmp = relation_tmp_[comp_relation_id];

  if(head_vec.size()*tail_vec.size()*relation_vec.size() == 0) {
	  cerr << "incorrect dimension" <<endl;
    throw std::runtime_error("incorrect dimension");
  }
  if(comp_head.size()*comp_tail.size()*comp_relation.size() == 0) {
	  cerr << "incorrect dimension" <<endl;
    throw std::runtime_error("incorrect dimension");
  }

  auto sum1 = computeDist( norm_flag_, tail_vec, head_vec + relation_vec);
  auto sum2 = computeDist( norm_flag_, comp_tail, comp_head + comp_relation);

  // if too far, compute gradient and do update 
  if ( sum1 + margin_ > sum2) {
    loss_+= margin_ + sum1 - sum2;
    updateGradient( head_vec, tail_vec, relation_vec, head_tmp, tail_tmp, relation_tmp);
    updateGradient( comp_head, comp_tail, comp_relation, comp_head_tmp, comp_tail_tmp, comp_relation_tmp);
  }
}

void Train::updateGradient(features_t& head_vec, features_t& tail_vec, features_t& relation_vec, features_t& head_tmp, features_t& tail_tmp, features_t& relation_tmp) {
	for ( size_t i = 0 ; i < head_vec.size(); i++ ) {
		double x = 2*(tail_vec[i] - head_vec[i] - relation_vec[i]);
		if ( x > 0)
			x = 1;
		else
			x = -1;
		
		relation_tmp[i] -= -1*learning_rate_*x;
		head_tmp[i] -= -1*learning_rate_*x;
		tail_tmp[i] += -1*learning_rate_*x;
		if( isnan(relation_tmp[i]) || isinf(relation_tmp[i]) ){
			cout << "NaN appears " << learning_rate_ << " " << x << endl;
			abort();
			}
		if( isnan(head_tmp[i]) || isinf(head_tmp[i]) ){
			cout << "NaN appears " << learning_rate_ << " " << x << endl;
			abort();
			}
		if( isnan(tail_tmp[i]) || isinf(tail_tmp[i]) ){
			cout << "NaN appears " << learning_rate_ << " " << x << endl;
			abort();
			}
	}
}

//void backup(features_t& head_vec, features_t& tail_vec, features_t& relation_vec) {
//
//	// First, compute difference
//	auto gradVec = tail_vec - ( head_vec + relation_vec);
//	// Transform from difference to gradient
//	std::transform(gradVec.begin(), gradVec.end(), gradVec.begin(), 
//			[this] (double& val) {
//				if(norm_flag_ == 1)
//					return (val>0) ? 1.0:-1.0;
//				else
//					return 2*val;
//			}
//			);	
//
//	// Do update on each std::vector
//	// For each element, minus gradient
//	auto minusFunc = [this](double& grad, double& elem) {
//		auto update = -1*grad*learning_rate_;
//		return elem - update;
//	};
//	// For each element, plus graident
//	auto plusFunc = [this](double& grad, double& elem) {
//		auto update = -1*grad*learning_rate_;
//		return elem + update;
//	};
//
//	std::transform(gradVec.begin(), gradVec.end(), 
//				relation_vec.begin(), relation_vec.begin(), minusFunc);
//
//	std::transform(gradVec.begin(), gradVec.end(), 
//				head_vec.begin(), head_vec.begin(), minusFunc);
//
//	std::transform(gradVec.begin(), gradVec.end(),
//			tail_vec.begin(), tail_vec.begin(), plusFunc);
//}

void Train::writeSnapshot() {
	relation_file_name = "relation-snap-shot.data";
	entity_file_name = "entity_snap-shot.data";
	writeData(relation_file_name, entity_file_name);
}
void Train::writeData(const string& relation_file_name, const string& entity_file_name) {
  std::cout << "writing data ... " << std::endl;
  // open file for writing
  ofstream relation_file, entity_file;
  relation_file.open(relation_file_name, ios::out|ios::trunc);
  entity_file.open(relation_file_name, ios::out|ios::trunc);
  std::cout << "writing relation to " << relation_file_name << std::endl;
  // print out each relation std::vector
  for( auto relation_vec : relation_mat_) {
    for ( auto element : relation_vec)
      relation_file << setprecision(7) << element << '\t' ;
    relation_file << endl;
  }
  std::cout << "writing entity to " << entity_file_name << std::endl;
  // print out each entity std::vector
  for ( auto entity_vec : entity_mat_ ) {
    for ( auto element : entity_vec) 
      entity_file << setprecision(7) << element << '\t';
    entity_file << endl;
  }
}
