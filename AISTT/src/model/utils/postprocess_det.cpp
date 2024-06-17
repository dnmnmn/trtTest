//
// Created by gopizza on 2024-06-12.
//

#include "postprocess_det.h"

void PostProcessBase::normalize(float *output, float conf_threshold)
{
    GetInverseMatrix(output, output_height, output_width);
    int local_width = output_height;
    int local_height = output_width;
    for(int y_index = 0; y_index < local_height; y_index++)
    {
        int y_area = local_width * y_index;
        auto pmax_index = std::max_element(inverse_data + 4 + y_area , inverse_data + local_width + y_area);
        int max_index = std::distance(inverse_data, pmax_index);
        if(inverse_data[max_index] > conf_threshold)
        {
            int center_x = inverse_data[y_area];
            int center_y = inverse_data[y_area + 1];
            boxes.push_back(std::vector<float>(7, 0));
            boxes[box_count][0] = (center_x - (inverse_data[y_area + 2] / 2)) / input_width;
            boxes[box_count][0] = boxes[box_count][0] < 0 ? 0 : boxes[box_count][0];
            boxes[box_count][0] = boxes[box_count][0] > 1 ? 1 : boxes[box_count][0];
            boxes[box_count][1] = (center_y - (inverse_data[y_area + 3] / 2)) / input_height;
            boxes[box_count][1] = boxes[box_count][1] < 0 ? 0 : boxes[box_count][1];
            boxes[box_count][1] = boxes[box_count][1] > 1 ? 1 : boxes[box_count][1];
            boxes[box_count][2] = (center_x + (inverse_data[y_area + 2] / 2)) / input_width;
            boxes[box_count][2] = boxes[box_count][2] < 0 ? 0 : boxes[box_count][2];
            boxes[box_count][2] = boxes[box_count][2] > 1 ? 1 : boxes[box_count][2];
            boxes[box_count][3] = (center_y + (inverse_data[y_area + 3] / 2)) / input_height;
            boxes[box_count][3] = boxes[box_count][3] < 0 ? 0 : boxes[box_count][3];
            boxes[box_count][3] = boxes[box_count][3] > 1 ? 1 : boxes[box_count][3];
            boxes[box_count][4] = inverse_data[max_index];
            boxes[box_count][5] = max_index - y_area - 4; //class_id
            if (++box_count >= MAX_NMS_) break;
        }
    }
    // std::sort(boxes.begin(), boxes.end(), [](const std::vector<float>& a, const std::vector<float>& b) { return a[4] > b[4]; });
    // if(box_count == 0) return false;
}

void PostProcessBase::denormalize(std::vector<std::vector<float>> _input_box, std::vector<std::vector<float>>* _output_box)
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

void PostProcessOBB::normalize(float *output, float conf_threshold)
{
    // [x, y, w, h, conf, class, angle]
    for(int i = 0; i < output_width; i++)
    {
        if(output[output_width * 4 + i] > conf_threshold)
        {
            int ind_x = i;
            int ind_y = output_width + i;
            int ind_w = output_width * 2 + i;
            int ind_h = output_width * 3 + i;
            int ind_conf  = output_width * 4 + i;
            int ind_class = output_width * 5 + i;
            int ind_rot = output_width * 6 + i;


            float center_x = output[ind_x];
            float center_y = output[ind_y];

            // xywh -> xyxy
            boxes.push_back(std::vector<float>(7, 0));
            boxes[box_count][0] = (center_x - (output[ind_w] / 2)) / input_width;
            boxes[box_count][0] = boxes[box_count][0] < 0 ? 0 : boxes[box_count][0];
            boxes[box_count][0] = boxes[box_count][0] > 1 ? 1 : boxes[box_count][0];
            boxes[box_count][1] = (center_y - (output[ind_h] / 2)) / input_height;
            boxes[box_count][1] = boxes[box_count][1] < 0 ? 0 : boxes[box_count][1];
            boxes[box_count][1] = boxes[box_count][1] > 1 ? 1 : boxes[box_count][1];
            boxes[box_count][2] = (center_x + (output[ind_w] / 2)) / input_width;
            boxes[box_count][2] = boxes[box_count][2] < 0 ? 0 : boxes[box_count][2];
            boxes[box_count][2] = boxes[box_count][2] > 1 ? 1 : boxes[box_count][2];
            boxes[box_count][3] = (center_y + (output[ind_h] / 2)) / input_height;
            boxes[box_count][3] = boxes[box_count][3] < 0 ? 0 : boxes[box_count][3];
            boxes[box_count][3] = boxes[box_count][3] > 1 ? 1 : boxes[box_count][3];
            boxes[box_count][4] = output[ind_conf];
            boxes[box_count][5] = round(output[ind_class]); //class_id
            boxes[box_count][6] = output[ind_rot];
            box_count++;
            if (box_count >= MAX_NMS_) break;
        }
    }
    std::sort(boxes.begin(), boxes.end(), [](const std::vector<float>& a, const std::vector<float>& b) { return a[4] > b[4]; });
}

void PostProcessOBB::denormalize(std::vector<std::vector<float>> _input_box, std::vector<std::vector<float>>* _output_box)
{
    // [x1, y1, x2, y2, conf, class, angle]
    // [x1, y1, x2, y2, x3, y3, x4, y4, conf, class]]
    int size = MAX_NMS_ > _input_box.size() ? _input_box.size() : MAX_NMS_;
    // std::vector<std::vector<float>> box(size, std::vector<float>(10, 0));
    _output_box->clear();
    _output_box->resize(size, std::vector<float>(10, 0));
    for(int i = 0; i < _input_box.size(); i++)
    {
        auto b = _input_box[i];
        cv::RotatedRect rRect = cv::RotatedRect(cv::Point2f((b[0]+b[2])/2 * input_width,(b[1]+b[3])/2 * input_height), cv::Size2f((b[2] - b[1]) * input_width,(b[3]-b[1]) * input_height), b[6]);
        cv::Point2f vertices[4];
        rRect.points(vertices);
        (*_output_box)[i][0] = vertices[0].x;
        (*_output_box)[i][1] = vertices[0].y;
        (*_output_box)[i][2] = vertices[1].x;
        (*_output_box)[i][3] = vertices[1].y;
        (*_output_box)[i][4] = vertices[2].x;
        (*_output_box)[i][5] = vertices[2].y;
        (*_output_box)[i][6] = vertices[3].x;
        (*_output_box)[i][7] = vertices[3].y;
        (*_output_box)[i][8] = b[4];
        (*_output_box)[i][9] = b[5];
    }
    return;
}