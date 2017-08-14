


#include <opencv/cv.h>
#include <QPixmap>

void io()
{
    std::string inpath;
    std::string outpath;

    cv::Mat img_color = cv::imread(inpath);

    //featExtractor->extractFeat(face_detector, point_detector, face_recognizer, img_color, dst_img, feat);

    //std::vector<float> featVector(std::begin(feat), std::end(feat));
    //feats.push_back(featVector);

    //imgNames.push_back(imgNameQString.toStdString());
    //++img_idx;
    //qDebug()<<imgNameQString<< "extracted, "<<img_idx<<"/"<< (int)imgNamesQString.size()<<"face images";

    cv::imwrite(outpath, img_color);

    auto pixmap = QPixmap::fromImage(Helper::mat2qimage(img_color)).scaled(240, 240);
}