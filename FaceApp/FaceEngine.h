//
// Created by tom on 17-9-6.
//

#ifndef FACESPYPRJ_FACEENGINE_H
#define FACESPYPRJ_FACEENGINE_H

#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <map>
#include <memory>

class FaceEngine {

public:

    typedef std::vector<float> FeatVec;

    bool load(const std::string & model_dir);

    std::vector<std::pair<cv::Mat, FeatVec> > extractFeat(const cv::Mat & color_img, size_t max = 1 );

protected:
    class Engine;
    std::shared_ptr<Engine> engine = nullptr;
};

#endif //FACESPYPRJ_FACEENGINE_H
