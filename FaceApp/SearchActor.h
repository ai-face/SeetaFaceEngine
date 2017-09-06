//
// Created by tom on 17-9-6.
//

#ifndef FACESPYPRJ_SEARCHACTOR_H
#define FACESPYPRJ_SEARCHACTOR_H


#include <string>
#include <vector>
#include <memory>
#include <map>
#include "FaceEngine.h"

const std::string SearchActorDbName("namesFeats.bin");

class SearchActor {
public:
    bool load(const std::string & path, int hint=-1);

    /**
     * create crop db in path;
     * @param dest, crop dir
     * @param engine
     * @param src face pic dir
     */
    static bool createDb(const std::string & dest, FaceEngine & engine, const std::string & src);

    bool initialize(const std::vector<std::string> & userdata,
                    const std::vector<FaceEngine::FeatVec> & feats);

    //todo reload
    bool reload(const std::string & path, int hint = -1);
    //void save(const std::string & path);

    //
    void finalize();

    std::vector<std::pair<std::size_t, float>> search( float * feature, size_t dim, int numKNN) const;

    std::vector<std::pair<std::string, float>> toUserData(const std::vector<std::pair<size_t, float>> & indexs) const;

protected:

    void _do_init(const std::vector<std::string> & names_, const std::vector<std::vector<float>> & feats_);
    void _do_init();
    class Table ;
    std::vector<std::string> names;
    std::vector<std::vector<float>> feats;
    std::shared_ptr<Table> table = nullptr;
};


#endif //FACESPYPRJ_SEARCHACTOR_H
