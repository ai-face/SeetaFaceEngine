#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QProgressBar>
#include <QTimer>
#include <QDebug>
#include <QDir>
#include "videohandler.h"
#include <qlistwidget.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "face_identification.h"
#include "recognizer.h"
#include "math_functions.h"
#include "face_detection.h"
#include "face_alignment.h"

//#include "falconn/eigen_wrapper.h"
#include "falconn/lsh_nn_table.h"

#include <qfilesystemmodel.h>
#include <vector>
#include <string>

#include "src/extractFeats.h"

#include<QMenu>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    static const int PROCESS_TIMEOUT = 50;

private:
    Ui::MainWindow *ui;
    void initForm();

    seeta::FaceDetection *face_detector;
    seeta::FaceAlignment *point_detector;
    seeta::FaceIdentification *face_recognizer;

    int MinFaceSize;
    float ImagePyramidScaleFactor;

    unsigned int numKNN;
    unsigned int numReranking;


    std::pair<std::vector<string>, std::vector<std::vector<float> >>  namesFeats;

    extractFeats *featExtractor;

    std::vector<falconn::DenseVector<float>> data;
    falconn::LSHConstructionParameters params_cp;
    unique_ptr<falconn::LSHNearestNeighborTable<falconn::DenseVector<float>>> cptable;


    cv::Mat dst_img;


    QListWidget *imgs_listeWidget;



private slots:
    void on_listView_clicked(const QModelIndex &index);
    void on_listView_doubleClicked(const QModelIndex &index);
    void on_faceDetectionButton_clicked(bool checked);
    void on_queryButton_clicked();
    void on_ImgsOpenButton_clicked();

protected:

    bool eventFilter(QObject *watched, QEvent *event) override;
    void dropEvent(QDropEvent * event) override;

public:

    std::vector<int32_t> do_LSH_search(cv::Mat &img_color);

    //======================
    QDir rootdir;
    QDir modeldir;
    QDir cropsdir;
    QDir srcdir;
    void setCropsDir(const QString & dir);
    void clearDb();
    void loadDb();
    void createDb(const QString & srcdir);
    void searchSimilearImgs(const QString & target);
};

#endif // MAINWINDOW_H
