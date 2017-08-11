#-------------------------------------------------
#
# Project created by QtCreator 2016-05-15T08:52:04
#
#-------------------------------------------------

QT       += core gui

CONFIG += c++11

QMAKE_CXXFLAGS += -msse -msse2 -msse3 -mssse3

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = CocsoFace
TEMPLATE = app

SOURCES += main.cpp\
        mainwindow.cpp \
    videohandler.cpp \
    helper.cpp \
    src/extractFeats.cpp

HEADERS  += mainwindow.h \
    videohandler.h \
    helper.h \
    src/extractFeats.h

FORMS    += \
    mainwindow.ui

QMAKE_MACOSX_DEPLOYMENT_TARGET = 10.10

# OpenCV include
INCLUDEPATH += /usr/local/include/opencv
INCLUDEPATH += /usr/local/include/opencv2

# Boost include
INCLUDEPATH += /usr/local/include

# Face include

SeetaFaceRoot = /home/tom/z.git.github.beautycpp.ai/SeetaFaceEngine/build.install
INCLUDEPATH += $$SeetaFaceRoot/include/SeetaFace/align
INCLUDEPATH += $$SeetaFaceRoot/include/SeetaFace/detec
INCLUDEPATH += $$SeetaFaceRoot/include/SeetaFace/ident
LIBS += -L$$SeetaFaceRoot/lib -lseeta_fa_lib -lseeta_facedet_lib -lviplnet

# Falconn include
FalconnRoot = /home/tom/z.git.github.beautycpp.ai/FALCONN/
INCLUDEPATH += $$FalconnRoot/src/include

# Eigen include
INCLUDEPATH += $$FalconnRoot/external/eigen

LIBS += -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video -lopencv_photo -lboost_iostreams -lboost_serialization


RESOURCES += design.qrc

DISTFILES += \
    icon.iconset/1473021726_eye.png \
    icon.iconset/1473021730_eye.png \
    icon.iconset/1473021733_eye.png \
    icon.iconset/1473021736_eye.png \
    icon.iconset/1473021739_eye.png \
    icon.iconset/1473021741_eye.png \
    icon.iconset/1473021744_eye.png \
    icon.iconset/icon_12x12.png \
    icon.iconset/icon_256x256.png \
    icon.icns \
    qss/blue.css \
    qss/brown.css \
    qss/dev.css \
    qss/lightgray.css \
    qss/MetroDark.css \
    qss/silvery.css
