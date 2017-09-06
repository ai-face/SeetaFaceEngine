//
// Created by tom on 17-9-6.
//

#include "SearchActor.h"
#include "FeaFile.h"
#include "falconn/eigen_wrapper.h"
#include "falconn/lsh_nn_table.h"
#include <cassert>

class SearchActor::Table {
public:
    std::vector<falconn::DenseVector<float>> data;
    falconn::LSHConstructionParameters params_cp;
    std::unique_ptr<falconn::LSHNearestNeighborTable<falconn::DenseVector<float>>> cptable;
    size_t dim;
    size_t num;
};

void SearchActor::finalize() {
    table.reset();
    names.clear();
    feats.clear();
}

bool SearchActor::initialize(const std::vector<std::string> & userdata,
                const std::vector<FaceEngine::FeatVec> & feats){
    if( table )
        return false;

    table.reset(new Table);
    _do_init(userdata, feats);
    return true;
}

bool SearchActor::load(const std::string & path, int hint) {
    if( table )
        return false;

    FeaFile::loadFeaturesFilePair(this->names, this->feats, path + "/" + SearchActorDbName);
    _do_init();
    return true;
}

void SearchActor::_do_init(const std::vector<std::string> & names_, const std::vector<std::vector<float>> & feats_){
    this->names = names_;
    this->feats = feats_;
    this->_do_init();
}

void SearchActor::_do_init() {
    table.reset(new Table);
    assert(names.size());
    assert(feats.size() == names.size());
    assert(feats.at(0).size());

    table->num = names.size();
    table->dim = feats.at(0).size();

    uint64_t seed = 119417657;
    int num_tables = 8;
    int num_setup_threads = 0;

    // 转换数据类型
    for (int ii = 0; ii < table->num; ++ii) {
        falconn::DenseVector<float> v = Eigen::VectorXf::Map(&feats[ii][0], table->dim);
        v.normalize(); // L2归一化
        table->data.push_back(v);
    }

    table->params_cp.dimension = table->dim;
    table->params_cp.lsh_family = falconn::LSHFamily::CrossPolytope;
    table->params_cp.distance_function = falconn::DistanceFunction::NegativeInnerProduct;
    table->params_cp.storage_hash_table = falconn::StorageHashTable::FlatHashTable;
    table->params_cp.k = 2; // 每个哈希表的哈希函数数目
    table->params_cp.l = num_tables; // 哈希表数目
    table->params_cp.last_cp_dimension = 2;
    table->params_cp.num_rotations = 2;
    table->params_cp.num_setup_threads = num_setup_threads;
    table->params_cp.seed = seed ^ 833840234;
    table->cptable = std::unique_ptr<falconn::LSHNearestNeighborTable<falconn::DenseVector<float>>>(
            std::move(falconn::construct_table<falconn::DenseVector<float>>(table->data, table->params_cp)));
    table->cptable->set_num_probes(400);
}


std::vector<std::pair<std::string, float>> SearchActor::toUserData(
        const std::vector<std::pair<size_t, float>> & indexs) const{
    std::vector<std::pair<std::string, float>> ns;
    for(auto & index : indexs){
        ns.push_back(std::make_pair(names[index.first],index.second));
    }
    return ns;
}

std::vector<std::pair<std::size_t, float>> SearchActor::search(
        float * feature, size_t dim, int numKNN) const {
    assert( dim == table->dim);

    falconn::DenseVector<float> q = Eigen::VectorXf::Map(feature, dim);
    q.normalize();

    std::vector<int32_t> idxCandidate;
    table->cptable->find_k_nearest_neighbors(q, numKNN, &idxCandidate);

    std::vector<std::pair<size_t, float>> ids;
    for(auto & i : idxCandidate) {
        ids.push_back(std::make_pair(i, q.dot(table->data[i])));
    }
    return ids;
}