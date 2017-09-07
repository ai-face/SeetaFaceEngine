//
// Created by tom on 17-9-6.
//

#include <string>
#include <QDir>
//#include <QString>

#include <searcher/SearchActor.h>
#include <searcher/FaceEngine.h>
#include <QTime>
#include <iostream>

int main(int argc, char * argv[])
{

    QString model_path;
    if( argc < 2)
        model_path = "/home/tom/seeta_face/model";
    else
        model_path = argv[1];

    QString image_path;
    if( argc < 3)
        image_path = "/home/tom/seeta_face/test_4_data_400/";
    else
        image_path = argv[2];

    QString crop_path;
    if( argc < 4)
        crop_path = "/home/tom/seeta_face/test_4_data_400_db/";
    else
        crop_path = argv[3];

    // load model
    FaceEngine engine;
    engine.load(model_path.toStdString());

    // mkdb to crop_path
    QTime time = QTime::currentTime();
    SearchActor::createDb(
            crop_path.toStdString(),
            engine,
            image_path.toStdString()
    );
    std::cout<<"elapsed="<< time.elapsed()/1000 << std::endl;

    return 0;
}