#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "helper.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <QDesktopWidget>
#include <QStringListModel>
#include <QListWidget>
#include <QMessageBox>
#include <QFileDialog>
#include <QDir>
#include <ctime>
#include <iostream>

#include "SearchActor.h"
#include "FeaFile.h"


const QString namefeats("namesFeats.bin");

MainWindow::MainWindow(QWidget *parent) :QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // start the windows at center
    QRect desktopRect = QApplication::desktop()->availableGeometry(this);
    QPoint center = desktopRect.center();
    move(center.x()-width()*0.5, center.y()-height()*0.5);

    ui->scrlArea->setContentsMargins(0, 0, 0, 0);
    ui->scrlArea->setAlignment(Qt::AlignCenter);
    ui->scrlArea->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui->scrlArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui->scrlArea->setFrameStyle(0);
    ui->scrlArea->setWidgetResizable(true);

    imgs_listeWidget = new  QListWidget;
    imgs_listeWidget->setViewMode(QListWidget::IconMode);
    imgs_listeWidget->setIconSize(QSize(105, 120));
    imgs_listeWidget->setResizeMode(QListWidget::Adjust);
    imgs_listeWidget->setMovement(QListView::Static);

    this->initForm();
    this->loadDb();

    std::vector<QWidget*> dest;
    dest.push_back(ui->previewImg);
    dest.push_back(imgs_listeWidget);
    dest.push_back(ui->cropImgLabel);
    for(auto wdg : dest) {
        wdg->setAcceptDrops(true);
        wdg->installEventFilter(this);
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}

// 初始化主窗体
void MainWindow::initForm()
{
    setWindowIcon(QIcon(QStringLiteral(":icons/eyeMain.png")));
    this->MinFaceSize = 20; // 最小人脸尺寸

    this->ImagePyramidScaleFactor = 10; // 采样率

    this->numKNN = 10;
    this->numReranking = 10;

    QString rootname("seeta_face");
    QString modelname("model");
    QString cropsname("crops");
    rootdir = QDir(QDir::home().absoluteFilePath(rootname));
    if(!rootdir.exists()) QDir::home().mkdir(rootname);
    modeldir= QDir(rootdir.absoluteFilePath(modelname));
    if(!modeldir.exists()) rootdir.mkdir(modelname);
    cropsdir= QDir(rootdir.absoluteFilePath(cropsname));
    if(!cropsdir.exists()) rootdir.mkdir(cropsname);

    engine.load(modeldir.absolutePath().toStdString());
}



void MainWindow::on_ImgsOpenButton_clicked()
{
    QString src_dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                  "/home", QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if ( !src_dir.isEmpty() )
    {
        srcdir = QDir(src_dir);
        // qDebug() << dir;
        QDir export_folder(src_dir);
        QStringList sl ; sl<<"*.bmp"<<"*.png"<<"*.jpg";
        export_folder.setNameFilters(sl);

        QStringList imgNamesQString = export_folder.entryList();
        QStringListModel *model = new QStringListModel(this);
        // Populate our model
        model->setStringList(imgNamesQString);
        // Glue model and view together
        ui->listView->setModel(model);
    }
}





// 提取特征
void MainWindow::on_faceDetectionButton_clicked(bool )
{
    createDb(srcdir.absolutePath());
    return;
}


void MainWindow::on_queryButton_clicked()
{
    //idxCandidate.clear();
    imgs_listeWidget->clear();

    QString path_queryImg = QFileDialog::getOpenFileName(this, "选择图像",
                 QDir::currentPath(), "Document files (*.jpg *.png *.bmp);;All files(*.*)");
    if(!path_queryImg.isEmpty())
        searchSimilearImgs(path_queryImg);
}

//========================================================
// listView
void MainWindow::on_listView_clicked(const QModelIndex &index)
{
    auto target =  cropsdir.absoluteFilePath(index.data(Qt::DisplayRole).toString());
    searchSimilearImgs(target);
}

void MainWindow::on_listView_doubleClicked(const QModelIndex &index)
{
    auto target =  cropsdir.absoluteFilePath(index.data(Qt::DisplayRole).toString());
    searchSimilearImgs(target);
}

#include <QDropEvent>
#include <QMimeData>
void MainWindow::dropEvent(QDropEvent * event) {
    const QMimeData * mimeData = event->mimeData();
    if( mimeData->hasUrls()) {
        auto ul = mimeData->urls();
        if(ul.size())
            searchSimilearImgs(ul.at(0).toLocalFile());
    }
    event->acceptProposedAction();
}

bool MainWindow::eventFilter(QObject *watched, QEvent *event) {
    if( watched == this->ui->previewImg && event->type() == QEvent::Drop) {
        this->dropEvent(dynamic_cast<QDropEvent*>(event));
        return true;
    } else if(watched == this->ui->previewImg && event->type() == QEvent::DragEnter) {
        dynamic_cast<QDragEnterEvent*>(event)->acceptProposedAction();
        return true;
    } else if(watched == this->ui->previewImg && event->type() == QEvent::DragMove) {
        dynamic_cast<QDragMoveEvent*> (event)->acceptProposedAction();
        return true;
    } else if(watched == this->ui->previewImg && event->type() == QEvent::DragLeave) {
        event->accept();
        return true;
    }

    return false;
}

// 搜索模块
//std::vector<int32_t> MainWindow::do_LSH_search(cv::Mat &img_color){
//
//    // Calculate cosine distance between query and data base faces
//    //float query_feat[2048];
//    auto r = engine.extractFeat(img_color, 1);
//    //featExtractor->extractFeat(face_detector, point_detector, face_recognizer, img_color, dst_img, query_feat);
//
//    this->loadDb();
//    std::vector<std::pair<std::size_t, float>> result = actor.search(&(r[0].second[0]), 2048, numKNN);
//
//    std::vector<int32_t> idxCandidate;
//    for(int i = 0; i < result.size() && i < numReranking; i++)
//        idxCandidate.push_back(result[i].first);
//    return idxCandidate;
//}

//======================================================
void MainWindow::setCropsDir(const QString & dir) {
    QDir destdir(dir);
    if(!destdir.exists())
        return;
    clearDb();
    cropsdir = destdir;
}

void MainWindow::clearDb(){
//    namesFeats.first.clear();
//    namesFeats.second.clear();
    cropsdir = QDir();
}

void MainWindow::createDb(const QString & srcdir) {
    QDir src(srcdir);
    if(!src.exists())
        return;

//    cv::Mat dst = cv::Mat((int)face_recognizer->crop_height(),
//                      (int)face_recognizer->crop_width(),
//                      CV_8UC((int)face_recognizer->crop_channels()));

//    std::vector<float> feat(2048, 0.0);
    if(true){
        SearchActor::createDb(cropsdir.absolutePath().toStdString(), engine, src.absolutePath().toStdString());
//        namesFeats.first.clear();
//        namesFeats.second.clear();
//        auto fl = src.entryInfoList(QDir::Files | QDir::NoDotAndDotDot);
//        for(auto finfo : fl){
//            cv::Mat img_color = cv::imread(finfo.absoluteFilePath().toStdString());
//            auto result = engine.extractFeat(img_color,1);
//            //featExtractor->extractFeat(face_detector ,point_detector, face_recognizer,img_color, dst, &feat[0]);
//            if( result.size() ) {
//                cv::imwrite(cropsdir.absoluteFilePath(finfo.fileName()).toStdString(), result[0].first);
//                namesFeats.second.push_back(result[0].second);
//                namesFeats.first.push_back(finfo.fileName().toStdString());
//            }
//        }
//        // Save image names and features
////        featExtractor->saveFeaturesFilePair(namesFeats, cropsdir.absoluteFilePath(namefeats).toStdString());
//        FeaFile::saveFeaturesFilePair(namesFeats.first, namesFeats.second, cropsdir.absoluteFilePath(namefeats).toStdString());
    }
}

void MainWindow::loadDb() {
    QFile db(cropsdir.absoluteFilePath(namefeats));
    if(!db.exists())
        return;

//    if(namesFeats.first.empty()){
        actor.load(cropsdir.absolutePath().toStdString());
//    }
}


void MainWindow::searchSimilearImgs(const QString & target) {
    std::cout<<"search = "<<target.toStdString()<<std::endl;
    QFile file(target);
    if(!file.exists())
        return;


    cv::Mat image_color = cv::imread(target.toStdString());

    ui->previewImg->setPixmap(QPixmap::fromImage(
                                  Helper::mat2qimage(image_color)));

    auto r = engine.extractFeat(image_color, 1);
    //featExtractor->extractFeat(face_detector, point_detector, face_recognizer, img_color, dst_img, query_feat);

    this->loadDb();
    std::vector<std::pair<std::size_t, float>> result = actor.search(&(r[0].second[0]), 2048, numKNN);

    auto ns = actor.toUserData(result);


    ui->scrlArea->takeWidget();
    imgs_listeWidget->clear();
    for(size_t  i= 0; i < ns.size() && i < 10; ++i){
        QString filename = QString::fromStdString(ns[i].first);
        QString tempImgName(cropsdir.absoluteFilePath(filename));
        imgs_listeWidget->addItem(new QListWidgetItem(QIcon(tempImgName),
                                                      filename));
    }
    ui->scrlArea->setWidget(imgs_listeWidget);
}

