//
// Created by gopizza on 2024-06-12.
//

#ifndef WINTRT_POSTPROCESS_DET_H
#define WINTRT_POSTPROCESS_DET_H

#include "postprocess.h"

class PostProcessOBB : public PostProcess {
public:
    void normalize(float* data, float conf_threshold=0.35f) override;
    void denormalize(std::vector<std::vector<float>> _input_box, std::vector<std::vector<float>>* _output_box) override;
};

class PostProcessBase : public PostProcess {
public:
    void normalize(float* data, float conf_threshold=0.35f) override;
    void denormalize(std::vector<std::vector<float>> _input_box, std::vector<std::vector<float>>* _output_box) override;
};



#endif //AISMARTTOPPINGTABLE2_POSTPROCESS_DET_H
