//
// Created by gopizza on 2024-05-30.
//

#include "src/aistt.h"
using namespace nvinfer1;

int main(int argc, char** argv)
{

    AISTT aistt = AISTT();
    aistt.Initialize();
    aistt.Run();
    aistt.Release();
    return 0;
}

