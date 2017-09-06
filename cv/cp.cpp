﻿
#include <QDir>
#include <QFile>
#include <QCoreApplication>
#include <QFileInfo>
#include <random>
#include <iostream>
int main(int argc, char ** argv){
    QCoreApplication app(argc, argv);

    QDir topdir("/home/tom/seeta_face/");

    QDir origdir(topdir.absoluteFilePath("Face"));
    if( !origdir.exists()) return -1;

    QDir datadir(topdir.absoluteFilePath("Face_data"));
    if( !datadir.exists())
        topdir.mkdir("Face_data");

    QDir testdir(topdir.absoluteFilePath("Face_test"));
    if( !testdir.exists())
        topdir.mkdir("Face_test");
    std::cout << "list subdirs" << std::endl;
    auto subdirs = origdir.entryInfoList(
                QDir::NoDotAndDotDot | QDir::AllDirs);
    for(auto sub : subdirs) {
        std::cout<<sub.absoluteFilePath().toStdString()<<std::endl;
        QDir subdir(sub.absoluteFilePath());
        auto files = subdir.entryInfoList(
                    QDir::NoDotAndDotDot | QDir::Files);
        auto r = std::rand() % 5;
        auto str = QString::number(r);
        for(auto f : files) {
            QFile file(f.absoluteFilePath());
            auto basename = f.baseName();
            if(basename.endsWith(str))
                 QFile::copy(f.absoluteFilePath(), testdir.absoluteFilePath(f.fileName()));
            else
                 QFile::copy(f.absoluteFilePath(), datadir.absoluteFilePath(f.fileName()));
        }
    }

    return 0;
}