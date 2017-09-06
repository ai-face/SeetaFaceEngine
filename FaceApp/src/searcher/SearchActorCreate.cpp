//
// Created by tom on 17-9-6.
//


#include <QDir>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "SearchActor.h"
#include "FeaFile.h"

bool SearchActor::createDb(const std::string & dest, FaceEngine & engine, const std::string & src_path) {
    std::vector<std::string>        names;
    std::vector<std::vector<float>> feats;

    QDir src(QString::fromStdString(src_path));
    QDir cropsdir(QString::fromStdString(dest));
    if( !cropsdir.exists()) return false;

    auto fl = src.entryInfoList(QDir::Files | QDir::NoDotAndDotDot);
    for(auto & finfo : fl){
        cv::Mat img_color = cv::imread(finfo.absoluteFilePath().toStdString());
        auto result = engine.extractFeat(img_color,1);
        if( result.size() ) {
            cv::imwrite(cropsdir.absoluteFilePath(finfo.fileName()).toStdString(), result[0].first);
            feats.push_back(result[0].second);
            names.push_back(finfo.fileName().toStdString());
        }
    }
    FeaFile::saveFeaturesFilePair(names, feats, cropsdir.absoluteFilePath(QString::fromStdString(SearchActorDbName)).toStdString());
    return true;
}