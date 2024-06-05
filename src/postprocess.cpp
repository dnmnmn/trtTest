//
// Created by gopizza on 2024-05-23.
//
#include "postprocess.h"

#include <iostream>
#include <fstream>
#include <algorithm>


void PostProcess::initialize(std::shared_ptr<nvinfer1::Dims32> _input_dim,
                             std::shared_ptr<std::vector<nvinfer1::Dims32>> _output_dims) {
    input_dim_ = _input_dim;
    output_dims_ = _output_dims;
    input_height = _input_dim->d[2];
    input_width = _input_dim->d[3];
    output_channel = 1;
    output_height = (*_output_dims)[0].d[1];
    output_width = (*_output_dims)[0].d[2];

    boxes.resize(100, std::vector<float>(7, 0));
    std::cout << "PostProcess::initialize()" << std::endl;
    printf("input_width: %d, input_height: %d, output_width: %d, output_height: %d, output_channel: %d\n", input_width, input_height, output_width, output_height, output_channel);
}

void PostProcess::release() {
    std::cout << "PostProcess::release()" << std::endl;
}

std::vector<std::vector<float>> PostProcess::nms(const float iou_threshold)
{
    int neighbors = 0;
    // [x1, y1, x2, y2, conf, class]
    std::vector<std::vector<float>> resRects;
    resRects.clear();

    const size_t size = boxes.size();
    if (!size) return resRects;

    // Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
    std::multimap<int, size_t> idxs;
    for (size_t i = 0; i < size; ++i)
    {
        idxs.emplace(boxes[i][3], i);
        // idxs.emplace(srcRects[i].br().y, i);
    }

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0)
    {
        // grab the last rectangle
        auto lastElem = --std::end(idxs);
        std::vector<float> rect1 = boxes[lastElem->second];
        // const cv::Rect& rect1 = srcRects[lastElem->second];

        int neigborsCount = 0;

        idxs.erase(lastElem);

        for (auto pos = std::begin(idxs); pos != std::end(idxs); )
        {
            // grab the current rectangle
            std::vector<float> rect2 = boxes[pos->second];
            //const cv::Rect& rect2 = srcRects[pos->second];

            float intArea = get_overlap_area(rect1, rect2);
            float unionArea = get_box_area(rect1) + get_box_area(rect2) - intArea;
            float overlap = intArea / unionArea;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > iou_threshold)
            {
                pos = idxs.erase(pos);
                ++neigborsCount;
            }
            else
            {
                ++pos;
            }
        }
        if (neigborsCount >= neighbors)
            resRects.push_back(rect1);
    }
    std::sort(resRects.begin(), resRects.end(), [](const std::vector<float>& a, const std::vector<float>& b) { return a[4] > b[4]; });
    for(int i = 0; i < resRects.size(); i++)
    {
        if(resRects[i][4] == 0)
        {
            resRects.resize(i);
            break;
        }
    }
    return resRects;
}

std::vector<std::vector<float>> PostProcess::weighted_boxes_fusion(const float iou_threshold) {
    //std::cout << "PostProcess::wbf()" << std::endl;
    std::vector<std::vector<float>> overall_boxes;
    std::vector<std::vector<std::vector<float>>> new_box;
    std::vector<std::vector<float>> weighted_boxes;

    for(const auto& box : boxes)
    {
        auto [index, best_iou] = find_matching_box_fast(weighted_boxes, box, iou_threshold, box_count);
        if(index != -1)
        {
            new_box[index].push_back(box);
            weighted_boxes[index] = get_weighted_box(new_box[index], 1, box_count);
        }
        else
        {
            new_box.push_back(std::vector<std::vector<float>>{box});
            weighted_boxes.push_back(box);
        }
    }
    std::sort(weighted_boxes.begin(), weighted_boxes.end(), [](const std::vector<float>& a, const std::vector<float>& b) { return a[4] > b[4]; });
    for(int i = 0; i < weighted_boxes.size(); i++)
    {
        if(weighted_boxes[i][4] == 0)
        {
            weighted_boxes.resize(i);
            break;
        }
    }
    return weighted_boxes;
}

void PostProcess::clear()
{
    boxes.clear();
    boxes.resize(100, std::vector<float>(7, 0));
    box_count = 0;
}

void PostProcess::debug(float* data) {
    std::cout << "PostProcess::debug()" << std::endl;
    std::string wPath = "output.txt";

    std::ofstream writeFile(wPath);
    if (writeFile.is_open())
    {
        int output_tensor_size = output_height * output_width;
        for(int i = 0; i < output_tensor_size; i++)
        {
            int ln = i % output_height;
            if (ln == 0) writeFile << "\n";
            int index = (ln * output_width) + (i / output_height);
            float val = data[index];
            writeFile << val << " ";

        }
        writeFile.close();
    }
}

std::pair<int, float> PostProcess::find_matching_box_fast(const std::vector<std::vector<float>>& boxes, const std::vector<float>& new_box, const float iou_threshold, int box_count)
{
    // boxes : [x1, y1, x2, y2, conf, class]
    if(boxes.size() == 0) return std::make_pair(-1, iou_threshold);
    std::vector<float> ious = bb_iou_array(boxes, new_box);
    auto max_iter = max_element(ious.begin(), ious.end());
    int best_idx = distance(ious.begin(), max_iter);
    float best_iou = *max_iter;
    if (best_iou <= iou_threshold) {
        best_iou = iou_threshold;
        best_idx = -1;
    }
    return std::make_pair(best_idx, best_iou);
}

std::vector<float> PostProcess::bb_iou_array(const std::vector<std::vector<float>> boxes, const std::vector<float> new_box)
{
    std::vector<float> ious(boxes.size(), 0);
    for(size_t i = 0; i < boxes.size(); i++)
    {
        if(boxes[i][5] != new_box[5]) ious[i] = -1;
        float x1 = std::max(boxes[i][0], new_box[0]);
        float y1 = std::max(boxes[i][1], new_box[1]);
        float x2 = std::min(boxes[i][2], new_box[2]);
        float y2 = std::min(boxes[i][3], new_box[3]);

        float w = std::max(0.0f, x2 - x1);
        float h = std::max(0.0f, y2 - y1);

        float inter = w * h;
        float area1 = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]);
        float area2 = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1]);
        ious[i] = inter / (area1 + area2 - inter);
    }
    return ious;
}

std::vector<float> PostProcess::get_weighted_box(const std::vector<std::vector<float>> boxes, int conf_type, int box_count)
{
    // boxes : [x1, y1, x2, y2, conf, class]
    // scores : [conf]
    // conf_type ['allow overflow' : 0, 'avg' : 1, 'max' : 2, 'box_and_model_avg' : 3, 'absent_model_aware_avg' : 4]:
    std::vector<float> weighted_box(7 ,0);

    float conf = 0;
    std::vector<float> conf_list;
    float weight = 0;

    for(int t = 0; t < boxes.size(); t++)
    {
        weighted_box[0] = boxes[t][0] * boxes[t][4];
        weighted_box[1] = boxes[t][1] * boxes[t][4];
        weighted_box[2] = boxes[t][2] * boxes[t][4];
        weighted_box[3] = boxes[t][3] * boxes[t][4];
        conf = boxes[t][4];
        conf_list.push_back(conf);
        weight += 1;
    }
    weighted_box[5] = boxes[0][5];
    if(conf_type == 1 || conf_type == 3 || conf_type == 4)
    {
        weighted_box[4] = conf/ box_count;
    }
    if(conf_type == 2)
    {
        weighted_box[4] = *max_element(conf_list.begin(), conf_list.end());
    }
    weighted_box[0] /= conf;
    weighted_box[1] /= conf;
    weighted_box[2] /= conf;
    weighted_box[3] /= conf;
    return weighted_box;
}

float PostProcess::get_box_area(const std::vector<float>& box)
{
    // (right - left) * (bottom - top)
    return (box[2] - box[0]) * (box[3] - box[1]);
}

float PostProcess::get_overlap_area(const std::vector<float> &box1, const std::vector<float> &box2)
{
    // [x1, y1, x2, y2]
    float x1 = std::max(box1[0], box2[0]);
    float y1 = std::max(box1[1], box2[1]);
    float x2 = std::min(box1[2], box2[2]);
    float y2 = std::min(box1[3], box2[3]);

    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);

    return w * h;
}

void PostProcessBase::normalize(float *output, float conf_threshold)
{
    bool stop_flag = false;
    //std::cout << "PostProcess::normalize()" << std::endl;
    // [x, y, w, h, conf, class, angle]
    for(int i = 0; i < output_width; i++)
    {
        for(int c = 4; c < output_height; c++)
        {
            if(output[output_width * c + i] > conf_threshold)
            {
                int ind_x = i;
                int ind_y = output_width + i;
                int ind_w = output_width * 2 + i;
                int ind_h = output_width * 3 + i;
                int ind_conf = output_width * c + i;

                float center_x = output[ind_x];
                float center_y = output[ind_y];

                // xywh -> xyxy
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
                boxes[box_count][5] = c - 4; //class_id
                box_count++;
                if(box_count >= 100)
                {
                    stop_flag = true;
                    break;
                }
            }
        }
        if(stop_flag) break;
    }
    std::sort(boxes.begin(), boxes.end(), [](const std::vector<float>& a, const std::vector<float>& b) { return a[4] > b[4]; });
    // if(box_count == 0) return false;
}

void PostProcessBase::denormalize(std::vector<std::vector<float>> _input_box, int _size, std::vector<std::vector<float>>* _output_box)
{
    // [x1, y1, x2, y2, conf, class]
    int size = _size > _input_box.size() ? _input_box.size() : _size;
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
            if (box_count >= 100) break;
        }
    }
    std::sort(boxes.begin(), boxes.end(), [](const std::vector<float>& a, const std::vector<float>& b) { return a[4] > b[4]; });
}

void PostProcessOBB::denormalize(std::vector<std::vector<float>> _input_box, int _size, std::vector<std::vector<float>>* _output_box)
{
    // [x1, y1, x2, y2, conf, class, angle]
    // [x1, y1, x2, y2, x3, y3, x4, y4, conf, class]]
    int size = _size > _input_box.size() ? _input_box.size() : _size;
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