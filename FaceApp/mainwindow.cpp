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

    dst_img = cv::Mat((int)face_recognizer->crop_height(),
                      (int)face_recognizer->crop_width(),
                      CV_8UC((int)face_recognizer->crop_channels()));
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

