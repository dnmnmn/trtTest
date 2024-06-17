//
// Created by gopizza on 2024-06-12.
//

#include "postprocess_seg.h"

void PostProcessSeg::normalize(float *output, float conf_threshold)
{
    //GetInverseMatrix(output, output_height, output_width);
    memcpy(inverse_data, output, output_width * sizeof(float) * 6);
    int local_width = output_height;
    int local_height = output_width;
    for(int y_index = 0; y_index < local_height; y_index++)
    {
        int y_area = 6 * y_index;
        //auto pmax_index = std::max_element(inverse_data + 4 + y_area , inverse_data + class_num + y_area);
        //int max_index = std::distance(inverse_data, pmax_index);
        if(inverse_data[y_area + 4] > conf_threshold)
        {
            int center_x = inverse_data[y_area];
            int center_y = inverse_data[y_area + 1];
            boxes.push_back(std::vector<float>(7, 0));
            boxes[box_count][0] = (center_x - (inverse_data[y_area + 2] / 2)) / input_width;
            boxes[box_count][0] = std::clamp(boxes[box_count][0], 0.0f, 1.0f);
            boxes[box_count][1] = (center_y - (inverse_data[y_area + 3] / 2)) / input_height;
            boxes[box_count][1] = std::clamp(boxes[box_count][1], 0.0f, 1.0f);
            boxes[box_count][2] = (center_x + (inverse_data[y_area + 2] / 2)) / input_width;
            boxes[box_count][2] = std::clamp(boxes[box_count][2], 0.0f, 1.0f);
            boxes[box_count][3] = (center_y + (inverse_data[y_area + 3] / 2)) / input_height;
            boxes[box_count][3] = std::clamp(boxes[box_count][3], 0.0f, 1.0f);
            boxes[box_count][4] = inverse_data[y_area + 4];
            boxes[box_count][5] = inverse_data[y_area + 5];//max_index - y_area - 4; //class_id
            if (++box_count >= MAX_NMS_) break;
        }
    }
    // std::sort(boxes.begin(), boxes.end(), [](const std::vector<float>& a, const std::vector<float>& b) { return a[4] > b[4]; });
    // if(box_count == 0) return false;
}

void PostProcessSeg::denormalize(std::vector<std::vector<float>> _input_box, std::vector<std::vector<float>>* _output_box)
{
    // [x1, y1, x2, y2, conf, class]
    int size = MAX_NMS_ > _input_box.size() ? _input_box.size() : MAX_NMS_;
    _output_box->clear();
    _output_box->resize(size, std::vector<float>(7, 0));
    for(int i = 0; i < size; i++)
    {
        (*_output_box)[i] = _input_box[i];
        (*_output_box)[i][0] *= input_width;
        (*_output_box)[i][1] *= input_height;
        (*_output_box)[i][2] *= input_width;
        (*_output_box)[i][3] *= input_height;
    }
    return;
}