//
// Created by tom on 17-9-6.
//

#ifndef FACESPYPRJ_FEAFILE_H
#define FACESPYPRJ_FEAFILE_H

#include <map>
#include <vector>
#include <string>


class FeaFile {

public :
    static void saveFeaturesFilePair(
            const std::vector<std::string> & names,
            const std::vector<std::vector<float> >  & features,
            const std::string &filename);
    static void loadFeaturesFilePair(
            std::vector<std::string> & names,
            std::vector<std::vector<float> > &features,
            const std::string &filename);

};


#endif //FACESPYPRJ_FEAFILE_H
