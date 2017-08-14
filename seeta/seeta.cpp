

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

    // Initialize face Identification model
    face_recognizer = new seeta::FaceIdentification(
                modeldir.absoluteFilePath("seeta_fr_v1.0.bin").toStdString().c_str());