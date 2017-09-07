//
// Created by tom on 17-9-6.
//

#include <QString>
#include <searcher/SearchActor.h>
#include <searcher/FaceEngine.h>
#include <QTime>
#include <QDir>
#include <iostream>
#include <fstream>

int main(int argc, char *argv []) {

    QString model_path;
    if( argc < 2)
        model_path = "/home/tom/seeta_face/model";
    else
        model_path = argv[1];

    QString crop_path;
    if( argc < 3)
        crop_path = "/home/tom/seeta_face/test_4_data_400_db/";
    else
        crop_path = argv[2];

    QString test_data;
    if( argc < 4)
        test_data = "/home/tom/seeta_face/test_4_test_400/";
    else
        test_data = argv[3];

    QString result_data;
    if( argc < 5 )
        result_data = "/home/tom/seeta_face/test_4_test_400.csv";
    else
        result_data = argv[4];

    FaceEngine engine;
    engine.load(model_path.toStdString());

    // mkdb to crop_path
    SearchActor actor;
    actor.load(crop_path.toStdString());


    QDir testdir(test_data);
    if( !testdir.exists()) return false;

    std::fstream fs;
    fs.open(result_data.toStdString(), std::ios_base::out);
    if(!fs.is_open())
        return -1;

    fs<<"target,result,similarity"<<std::endl;
    std::cout<<"target,result,similarity"<<std::endl;

    auto fl = testdir.entryInfoList(QDir::Files | QDir::NoDotAndDotDot);
    QTime time = QTime::currentTime();
    int count = 0;
    for(auto & finfo : fl){
        std::cout<<finfo.fileName().toStdString()<<", ";
        auto feat = engine.extractFeat(finfo.absoluteFilePath().toStdString(),1);
        std::vector<std::pair<std::string, float>> result;
        if( feat.size())
            result = actor.toUserData(actor.search(&feat[0].second[0], feat[0].second.size(), 1));

        fs<<finfo.fileName().toStdString()<<", ";
        if( result.size()) {
            std::cout << result[0].first << ", " << result[0].second << std::endl;
            fs<<result[0].first << ", " << result[0].second << std::endl;
        } else {
            std::cout << "-1" << ", " << 0  << std::endl;
            fs<<"-1" << ", " << 0 << std::endl;
        }
        count++;
//        if(count<0)
//            break;
    }
    std::cout<<"elapsed="<< time.elapsed()/1000 << std::endl;
    std::cout<<"count="<<count<<std::endl;

    return 0;
}