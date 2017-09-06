//
// Created by tom on 17-9-6.
//

#include "FeaFile.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/list.hpp>
#include <fstream>
#include <sstream>

using namespace std;

void FeaFile::saveFeaturesFilePair(
        const std::vector<std::string> & names,
        const std::vector<std::vector<float> >  & features,
        const string &filename) {
    ofstream out(filename.c_str());
    stringstream ss;
    boost::archive::binary_oarchive oa(ss);
    oa << names << features;
    out << ss.str();
    out.close();
}
void FeaFile::loadFeaturesFilePair(std::vector<std::string> & names,
                                   std::vector<std::vector<float> > &features,
                                   const string &filename) {
    names.clear();
    features.clear();
    ifstream in(filename.c_str());
    boost::archive::binary_iarchive ia(in);
    ia >> names >> features;
    in.close();
}