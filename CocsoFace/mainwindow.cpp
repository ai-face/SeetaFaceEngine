#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "helper.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "face_identification.h"
#include "recognizer.h"
#include "math_functions.h"
#include "face_detection.h"
#include "face_alignment.h"

//#include "falconn/eigen_wrapper.h"
#include "falconn/lsh_nn_table.h"

#include <QDesktopWidget>
#include <QStringListModel>
#include <QListWidget>
#include <QMessageBox>
#include <QDir>
#include <ctime>
#include <iostream>

#include "src/extractFeats.h"

using namespace seeta;

using falconn::construct_table;
using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborTable;
using falconn::QueryStatistics;
using falconn::StorageHashTable;

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
    ui->frame->hide();
    this->loadDb();
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
    ui->ScaleFactorQSlider->setMinimum(1);
    ui->ScaleFactorQSlider->setMaximum(10);
    ui->culrrentScaleFactorQLabel->setText(QString::number(1.0));

    this->numKNN = 50;
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

    // Initialize face detection model
    face_detector = new seeta::FaceDetection(
                modeldir.absoluteFilePath("seeta_fd_frontal_v1.0.bin").toStdString().c_str());
    face_detector->SetMinFaceSize(this->MinFaceSize);
    face_detector->SetScoreThresh(2.f);
    face_detector->SetImagePyramidScaleFactor(this->ImagePyramidScaleFactor);
    face_detector->SetWindowStep(4, 4);

    // Initialize face alignment model
    point_detector = new seeta::FaceAlignment(
                modeldir.absoluteFilePath("seeta_fa_v1.1.bin").toStdString().c_str());

    std::cout<<modeldir.absoluteFilePath("seeta_fr_v1.0.bin").toStdString()<<std::endl;
    // Initialize face Identification model
    face_recognizer = new seeta::FaceIdentification(
                modeldir.absoluteFilePath("seeta_fr_v1.0.bin").toStdString().c_str());

    path_namesFeats = cropsdir.absoluteFilePath(namefeats).toStdString();

    dst_img = cv::Mat((int)face_recognizer->crop_height(),
                      (int)face_recognizer->crop_width(),
                      CV_8UC((int)face_recognizer->crop_channels()));
}


// -----------------------------render loop-----------------------------------------------
void MainWindow::process()
{

}

// general slots
void MainWindow::exitClicked()
{
    qDebug("exit");
    QApplication::quit();
}

void MainWindow::aboutClicked()
{
    qDebug("about");
}

// ----------------------------------------------------------------------------
void MainWindow::on_startButton_toggled(bool checked)
{

}

void MainWindow::on_rMaxHorizontalSlider_valueChanged(int )
{
    this->MinFaceSize = ui->rMaxHorizontalSlider->value();
    //qDebug() << "rMax: " << this->Radius_Max;
    ui->culrrentRadius->setText(QString::number(this->MinFaceSize));
}


void MainWindow::on_ScaleFactorQSlider_valueChanged(int )
{
    this->ImagePyramidScaleFactor = ui->ScaleFactorQSlider->value();
    ui->culrrentScaleFactorQLabel->setText(QString::number(0.1*this->ImagePyramidScaleFactor));

}


void MainWindow::on_testButton_toggled(bool checked)
{
}

void MainWindow::on_ImgsOpenButton_clicked()
{
    src_dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                  "/home", QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if ( !src_dir.isEmpty() )
    {
        srcdir = QDir(src_dir);
        // qDebug() << dir;
        QDir export_folder(src_dir);
        QStringList sl ; sl<<"*.bmp"<<"*.png"<<"*.jpg";
        export_folder.setNameFilters(sl);
        imgNamesQString = export_folder.entryList();
        // for ( QStringList::Iterator it = imgNamesQString.begin(); it != imgNamesQString.end(); ++it )
        // qDebug() << "Processed command: " << *it;
        QStringListModel *model = new QStringListModel(this);
        // Populate our model
        model->setStringList(imgNamesQString);
        // Glue model and view together
        ui->listView->setModel(model);
    }
}

// 右键菜单添加搜索功能
void MainWindow::searchSimilarImgs()
{
//    imgs_listeWidget->clear();

//    cv::Mat img_color = cv::imread(imgNameSelected.toStdString());
//    idxCandidate = do_LSH_search(img_color);

//    for (size_t i = 0 ; i != idxCandidate.size() ; i++) {
//        QString tmpImgName = src_dir + '/' + namesFeats.first.at(idxCandidate[i]).c_str();
//        imgs_listeWidget->addItem(new QListWidgetItem(QIcon(tmpImgName), QString::fromStdString(namesFeats.first.at(idxCandidate[i]).c_str())));
//    }
//    ui->previewImg->setPixmap(QPixmap::fromImage(Helper::mat2qimage(img_color)));
//    ui->cropImgLabel->setPixmap(QPixmap::fromImage(Helper::mat2qimage(dst_img)));

//    ui->scrlArea->setWidget(imgs_listeWidget);

//    idxCandidate.clear();
}

// todo: 右键菜单添加搜索功能
void MainWindow::deleteImg(){
    qDebug()<<"hello world";
}

// 创建动作
void MainWindow::createActions()
{
}



// 提取特征
void MainWindow::on_faceDetectionButton_clicked(bool checked)
{
    createDb(src_dir);
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
//    imgNameSelected = target;
    searchSimilearImgs(target);
    //QPixmap img(target);
    //ui->previewImg->setPixmap(img);
    //ui->cropImgLabel->setPixmap(img);
}

void MainWindow::on_listView_doubleClicked(const QModelIndex &index)
{
    auto target =  cropsdir.absoluteFilePath(index.data(Qt::DisplayRole).toString());
    searchSimilearImgs(target);
}

// 搜索模块
std::vector<int32_t> MainWindow::do_LSH_search(cv::Mat &img_color){

    // Calculate cosine distance between query and data base faces
    float query_feat[2048];
    featExtractor->extractFeat(face_detector, point_detector, face_recognizer, img_color, dst_img, query_feat);

    falconn::DenseVector<float> q = Eigen::VectorXf::Map(&query_feat[0], 2048);
    q.normalize();

    loadDb();

    std::vector<int32_t> idxCandidate;
    cptable->find_k_nearest_neighbors(q, numKNN, &idxCandidate); // LSH find the K nearest neighbors
    std::cout<<"idxCandidate size="<<idxCandidate.size()<<std::endl;
    // do reranking
    std::vector<std::pair<float, size_t> > dists_idxs;
    for (unsigned int i = 0 ; i < numReranking && i < idxCandidate.size() ; i++) {
        float tmp_cosine_dist = q.dot(data[idxCandidate[i]]);
        dists_idxs.push_back(std::make_pair(tmp_cosine_dist, idxCandidate[i]));
    }

    std::sort(dists_idxs.begin(), dists_idxs.end());
    std::reverse(dists_idxs.begin(), dists_idxs.end());

    for(unsigned int i = 0 ; i < numReranking && i < idxCandidate.size() ; i++){
        idxCandidate.at(i) = (int32_t)dists_idxs[i].second;
    }
    return idxCandidate;
}

//======================================================
void MainWindow::setCropsDir(const QString & dir) {
    QDir destdir(dir);
    if(!destdir.exists())
        return;
    clearDb();
    cropsdir = destdir;
}

void MainWindow::clearDb(){
    namesFeats.first.clear();
    namesFeats.second.clear();
    cropsdir = QDir();
}

void MainWindow::createDb(const QString & srcdir) {
    QDir src(srcdir);
    if(!src.exists())
        return;

    cv::Mat dst = cv::Mat((int)face_recognizer->crop_height(),
                      (int)face_recognizer->crop_width(),
                      CV_8UC((int)face_recognizer->crop_channels()));

    std::vector<float> feat(2048, 0.0);
    if(true){
        namesFeats.first.clear();
        namesFeats.second.clear();
        auto fl = src.entryInfoList(QDir::Files | QDir::NoDotAndDotDot);
        for(auto finfo : fl){
            cv::Mat img_color = cv::imread(finfo.absoluteFilePath().toStdString());
            featExtractor->extractFeat(face_detector ,point_detector, face_recognizer,img_color, dst, &feat[0]);
            cv::imwrite(cropsdir.absoluteFilePath(finfo.fileName()).toStdString(), dst);
            namesFeats.second.push_back(feat);
            namesFeats.first.push_back(finfo.fileName().toStdString());
        }
        // Save image names and features
        featExtractor->saveFeaturesFilePair(namesFeats, cropsdir.absoluteFilePath(namefeats).toStdString());
    }
}

void MainWindow::loadDb() {
    QFile db(cropsdir.absoluteFilePath(namefeats));
    if(!db.exists())
        return;

    if(namesFeats.first.empty()){
        featExtractor->loadFeaturesFilePair(namesFeats, cropsdir.absoluteFilePath(namefeats).toStdString());
        qDebug()<<"first loaded";

        // LSH搜索方案
        int numFeats = (int)namesFeats.first.size();
        int dim = (int)namesFeats.second[0].size();

        // Data set parameters
        uint64_t seed = 119417657;

        // Common LSH parameters
        int num_tables = 8;
        int num_setup_threads = 0;
        StorageHashTable storage_hash_table = StorageHashTable::FlatHashTable;
        DistanceFunction distance_function = DistanceFunction::NegativeInnerProduct;

        // 转换数据类型
        qDebug() << "Generating data set ...";
        for (int ii = 0; ii < numFeats; ++ii) {
            falconn::DenseVector<float> v = Eigen::VectorXf::Map(&namesFeats.second[ii][0], dim);
            v.normalize(); // L2归一化
            data.push_back(v);
        }

        // Cross polytope hashing
        params_cp.dimension = dim;
        params_cp.lsh_family = LSHFamily::CrossPolytope;
        params_cp.distance_function = distance_function;
        params_cp.storage_hash_table = storage_hash_table;
        params_cp.k = 2; // 每个哈希表的哈希函数数目
        params_cp.l = num_tables; // 哈希表数目
        params_cp.last_cp_dimension = 2;
        params_cp.num_rotations = 2;
        params_cp.num_setup_threads = num_setup_threads;
        params_cp.seed = seed ^ 833840234;
        cptable = unique_ptr<falconn::LSHNearestNeighborTable<falconn::DenseVector<float>>>(std::move(construct_table<falconn::DenseVector<float>>(data, params_cp)));
        cptable->set_num_probes(896);
        qDebug() << "index build finished ...";

    }
}


void MainWindow::searchSimilearImgs(const QString & target) {
    std::cout<<"search = "<<target.toStdString()<<std::endl;
    QFile file(target);
    if(!file.exists())
        return;


    cv::Mat image_color = cv::imread(target.toStdString());

    ui->previewImg->setPixmap(QPixmap::fromImage(
                                  Helper::mat2qimage(image_color)));


    auto candidates = do_LSH_search(image_color);

    ui->cropImgLabel->setPixmap(QPixmap::fromImage(
                                  Helper::mat2qimage(dst_img)));


    ui->scrlArea->takeWidget();
    imgs_listeWidget->clear();
    for(size_t  i= 0; i < candidates.size() && i < 10; ++i){
        QString filename = QString::fromStdString(namesFeats.first.at(candidates[i]));
        QString tempImgName(cropsdir.absoluteFilePath(filename));
        imgs_listeWidget->addItem(new QListWidgetItem(QIcon(tempImgName),
                                                      filename));
    }
    ui->scrlArea->setWidget(imgs_listeWidget);
}

