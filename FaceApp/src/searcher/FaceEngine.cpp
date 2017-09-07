//
// Created by tom on 17-9-6.
//

#include "FaceEngine.h"
#include "face_identification.h"
#include "recognizer.h"
#include "face_detection.h"
#include "face_alignment.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cassert>
//#include <common.h> // from seetaFaceEngine

class FaceEngine::Engine {
public:
    std::shared_ptr<seeta::FaceDetection> face_detector = nullptr;
    std::shared_ptr<seeta::FaceAlignment> point_detector= nullptr;
    std::shared_ptr<seeta::FaceIdentification>face_recognizer = nullptr;
};
bool FaceEngine::load(const std::string & model_dir) {
    assert( !engine ) ;

    engine.reset(new Engine);

    // Initialize face detection model
    engine->face_detector = std::make_shared<seeta::FaceDetection>((model_dir + "/" +"seeta_fd_frontal_v1.0.bin").c_str());
    engine->face_detector->SetMinFaceSize(20);  //todo
    engine->face_detector->SetScoreThresh(2.f); //todo
    engine->face_detector->SetImagePyramidScaleFactor(10); //todo
    engine->face_detector->SetWindowStep(4, 4); //todo

    // Initialize face alignment model
    engine->point_detector.reset(new seeta::FaceAlignment((model_dir +"/" + "seeta_fa_v1.1.bin").c_str()));

    // Initialize face Identification model
    engine->face_recognizer.reset(new seeta::FaceIdentification((model_dir+ "/" + "seeta_fr_v1.0.bin").c_str()));
}

std::vector<std::pair<cv::Mat, FaceEngine::FeatVec> > FaceEngine::extractFeat(const std::string & color_img, size_t max  ) {
    cv::Mat m = cv::imread(color_img);
    return extractFeat(m, max);
};

std::vector<std::pair<cv::Mat, FaceEngine::FeatVec> > FaceEngine::extractFeat(const cv::Mat & img_color, size_t max) {
    cv::Mat img_gray;
    cv::cvtColor(img_color, img_gray, CV_BGR2GRAY);

    seeta::ImageData img_data_color(img_color.cols, img_color.rows, img_color.channels());
    img_data_color.data = img_color.data;
    seeta::ImageData img_data_gray(img_gray.cols, img_gray.rows, img_gray.channels());
    img_data_gray.data = img_gray.data;

    // Detect faces
    std::vector<seeta::FaceInfo> img_faces = engine->face_detector->Detect(img_data_gray);

    std::vector<std::pair<cv::Mat, FaceEngine::FeatVec> > result;
    if(!img_faces.size() || max <= 0)
        return result;

    for(int i = 0; i < img_faces.size() && i < max; i++) {
        cv::Mat dst_img = cv::Mat((int)engine->face_recognizer->crop_height(),
                              (int)engine->face_recognizer->crop_width(),
                              CV_8UC((int)engine->face_recognizer->crop_channels()));

        // Detect 5 facial landmarks
        seeta::FacialLandmark five_points[5];
        engine->point_detector->PointDetectLandmarks(img_data_gray, img_faces[i], five_points);

        // Create a image to store crop face.
        seeta::ImageData dst_img_data(dst_img.cols, dst_img.rows, dst_img.channels());
        dst_img_data.data = dst_img.data;
        /* Crop Face */
        engine->face_recognizer->CropFace(img_data_color, five_points, dst_img_data);

        // Extract face identity feature
        FeatVec feat(2048, 0.0);
        engine->face_recognizer->ExtractFeatureWithCrop(img_data_color, five_points, &feat[0]);
        result.push_back(std::make_pair(std::move(dst_img), std::move(feat)));
    }

    return result;
}