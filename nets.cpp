//#include "alexnet.h"
#include "nets.hpp"
#include "clfc.hpp"
#include "torch/torch.h"

#include <iostream>

using namespace torch;
typedef unsigned long int uli;

//const alnet_t POOLS_NUM   = 3;

/*const alnet_t STRIDE_SIZE    = 1;
const alnet_t PADDING_SIZE_1 = 2;
const alnet_t PADDING_SIZE_2 = 1;*/

/*const alnet_t OUR_OUT_NEURONS_CONV_1 = 64;
const alnet_t OUR_OUT_NEURONS_CONV_2 = 128;
const alnet_t OUR_OUT_NEURONS_CONV_3 = 256;
const alnet_t OUR_OUT_NEURONS_CONV_4 = OUT_NEURONS_CONV_3 * 2 / 3;
const alnet_t OUR_OUT_NEURONS_CONV_5 = OUT_NEURONS_CONV_4;  */

//const alnet_t OUT_PIC_SIZE = OUT_NEURONS_CONV_5 * PICTURE_SIZE / pow(4, POOLS_NUM); // 256 * 6 * 6

//const alnet_t FC_NEURONS_NUM = 4096;

const std::string FORWARD_ERROR_STR = "Forward failed: ";

Tensor Net::forward(Tensor x)
{
    return x;
}

OurNet::OurNet()
{
    conv1_1 = register_module("conv1_1", nn::Conv2d(nn::Conv2dOptions(CHANELS_NUM, 64, 3).stride(1).padding(1)));
    conv1_2 = register_module("conv1_2", nn::Conv2d(nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)));

    conv2_1 = register_module("conv2_1", nn::Conv2d(nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)));
    conv2_2 = register_module("conv2_2", nn::Conv2d(nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)));

    conv3_1 = register_module("conv3_1", nn::Conv2d(nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)));
    conv3_2 = register_module("conv3_2", nn::Conv2d(nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)));

    fc1 = register_module("fc1", nn::Linear(256 * 6 * 6, 32));
    fc2 = register_module("fc2", nn::Linear(32, NUM_CLASSES));
}

Tensor OurNet::forward(Tensor x)
{
    x = x.view({x.size(0), 1, PICTURE_HEIGHT, PICTURE_WIDTH});

    x = relu(conv1_1->forward(x));
    x = relu(conv1_2->forward(x));
    x = nn::BatchNorm2d(64)->forward(x);
    x = max_pool2d(x, 2, 2);

    x = relu(conv2_1->forward(x));
    x = relu(conv2_2->forward(x));
    x = nn::BatchNorm2d(128)->forward(x);
    x = max_pool2d(x, 2, 2);

    x = relu(conv3_1->forward(x));
    x = relu(conv3_2->forward(x));
    x = nn::BatchNorm2d(256)->forward(x);
    x = max_pool2d(x, 2, 2);

    x = x.view({-1, 256 * 6 * 6});

    try {
        x = fc1->forward(x);
    }
    catch (c10::IndexError er){
        //std::cout << er << std::endl;
    }
    x = nn::BatchNorm1d(32)->forward(x);
    x = relu(x);
    x = dropout(x, 0.2, is_training());
    x = fc2->forward(x);

    return x;
}


AlexNet::AlexNet()
{
    conv1 = register_module("conv1", nn::Conv2d(nn::Conv2dOptions(CHANELS_NUM, 64, 5).stride(1).padding(2)));
    conv2 = register_module("conv2", nn::Conv2d(nn::Conv2dOptions(64, 192, 5).stride(1).padding(2)));
    conv3 = register_module("conv3", nn::Conv2d(nn::Conv2dOptions(192, 384, 3).stride(1).padding(1)));
    conv4 = register_module("conv4", nn::Conv2d(nn::Conv2dOptions(384, 256, 3).stride(1).padding(1)));
    conv5 = register_module("conv5", nn::Conv2d(nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)));

    fc1 = register_module("fc1", nn::Linear(256 * 6 * 6, 4096));
    fc2 = register_module("fc2", nn::Linear(4096, 4096));
    fc3 = register_module("fc3", nn::Linear(4096, NUM_CLASSES));

    //std::cout << "Alex" << std::endl;
}

    // Implement the Net's algorithm.
Tensor AlexNet::forward(Tensor x)
{
    // Use one of many tensor manipulation functions.
    //std::cout << "type " << (typeid(x) == typeid(torch::Tensor)) << std::endl;
    x = x.view({x.size(0), 1, PICTURE_HEIGHT, PICTURE_WIDTH}); //reshape

    x = relu(conv1->forward(x));
    x = max_pool2d(x, 2, 2);
    x = relu(conv2->forward(x));
    x = max_pool2d(x, 2);

    x = relu(conv3->forward(x));
    x = relu(conv4->forward(x));
    x = relu(conv5->forward(x));
    x = max_pool2d(x, 2);

    x = x.view({x.size(0), -1});

    x = dropout(x, /*p=*/0.7, /*train=*/is_training());
    x = relu(fc1->forward(x));
    x = nn::BatchNorm1d(4096)->forward(x);
    x = dropout(x, /*p=*/0.7, /*train=*/is_training());
    x = relu(fc2->forward(x));
    x = nn::BatchNorm1d(4096)->forward(x);
    x = fc3->forward(x);
    //x = torch::nn::functional::log_softmax(x, /*dim=*/1);
    return x;
}

SmallNet::SmallNet()
{
    // Construct and register two Linear submodules.
    //fc1  = register_module("f1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 5).padding(1)));

    conv1 = register_module("conv1", nn::Conv2d(torch::nn::Conv2dOptions(CHANELS_NUM , 64, 5).stride(1).padding(2)));
    conv2 = register_module("conv2", nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)));

    fc1 = register_module("fc1", nn::Linear(12 * 12 * 128, 256));
    fc2 = register_module("fc2", nn::Linear(256, NUM_CLASSES));

    //std::cout << "Small" << std::endl;
}

Tensor SmallNet::forward(Tensor x)
{
    // Use one of many tensor manipulation functions.
    //std::cout << "type " << (typeid(x) == typeid(torch::Tensor)) << std::endl;
    x = x.view({x.size(0), 1, PICTURE_HEIGHT, PICTURE_WIDTH}); //reshape

    x = dropout(x, 0.2, is_training());
    x = relu(conv1->forward(x));
    x = nn::BatchNorm2d(64)->forward(x);
    x = max_pool2d(x, 2, 2);

    x = torch::dropout(x, 0.5, is_training());
    x = torch::relu(conv2->forward(x));
    x = torch::nn::BatchNorm2d(128)->forward(x);
    x = torch::max_pool2d(x, 2, 2);

    x = x.view({x.size(0), -1});

    x = dropout(x, 0.5, is_training());
    x = relu(fc1->forward(x));
    x = nn::BatchNorm1d(256)->forward(x);
    x = dropout(x, 0.2, is_training());
    x = fc2->forward(x);
    //x = torch::nn::functional::log_softmax(x, /*dim=*/1);
    return x;
}

VGG16::VGG16()
{
    // Construct and register two Linear submodules.
    //fc1  = register_module("f1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 5).padding(1)));

    conv1 = register_module("conv1", nn::Conv2d(nn::Conv2dOptions(CHANELS_NUM, 64, 3).stride(1).padding(1)));
    conv2 = register_module("conv2", nn::Conv2d(nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)));
    conv3 = register_module("conv3", nn::Conv2d(nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)));
    conv4 = register_module("conv4", nn::Conv2d(nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)));
    conv5 = register_module("conv5", nn::Conv2d(nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)));
    conv6 = register_module("conv6", nn::Conv2d(nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)));
    conv7 = register_module("conv7", nn::Conv2d(nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)));
    conv8 = register_module("conv8", nn::Conv2d(nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)));

    fc1 = register_module("fc1", nn::Linear(512 * 3 * 3, 4096));
    fc2 = register_module("fc2", nn::Linear(4096, 4096));
    fc3 = register_module("fc3", nn::Linear(4096, NUM_CLASSES));

    //std::cout << "VGG16" << std::endl;
}

Tensor VGG16::forward(Tensor x)
{
    // Use one of many tensor manipulation functions.
    //std::cout << "type " << (typeid(x) == typeid(torch::Tensor)) << std::endl;
    x = x.view({x.size(0), 1, PICTURE_HEIGHT, PICTURE_WIDTH}); //reshape

    x = conv1->forward(x);
    x = conv2->forward(x);
    x = max_pool2d(x, 2, 2);

    x = conv3->forward(x);
    x = conv4->forward(x);
    x = max_pool2d(x, 2, 2);

    x = conv5->forward(x);
    x = conv6->forward(x);
    x = conv6->forward(x);
    x = max_pool2d(x, 2, 2);

    x = conv7->forward(x);
    x = conv8->forward(x);
    x = conv8->forward(x);
    x = max_pool2d(x, 2, 2);

    x = x.view({x.size(0), -1});

    x = fc1->forward(x);
    x = fc2->forward(x);
    x = fc3->forward(x);
    //x = torch::nn::functional::log_softmax(x, /*dim=*/1);
    return x;
}

nn::Conv2d conv3x3(alnet_t in_planes, alnet_t out_planes, alnet_t stride = 1)
{
    //nn::Conv2d conv = nn::Module::register_module("conv3x3", nn::Conv2d(nn::Conv2dOptions(in_planes, out_planes, 3).stride(stride).padding(1)));
    return nn::Conv2d(nn::Conv2dOptions(in_planes, out_planes, 3).stride(stride).padding(1));
}

BasicBlock::BasicBlock(alnet_t inplanes, alnet_t planes, alnet_t str, nn::Sequential down)
{
    conv1 = conv3x3(inplanes, planes, str);
    conv2 = conv3x3(planes, planes);
    expansion = 1;
    this->planes = planes;

    downsample = down;
    stride = str;
}

Tensor BasicBlock::forward(Tensor x)
{
    Tensor residual;
    if (downsample)
        residual = downsample->forward(x);
    else
        residual = x;
    x = relu(conv1->forward(x));
    x = nn::BatchNorm2d(planes)->forward(x);
    x = conv2->forward(x);
    x = nn::BatchNorm2d(planes)->forward(x);

    x += residual;
    x = relu(x);
    return x;
}

ResNet18::ResNet18()
{
    inplanes = 64;
    convv1 = nn::Conv2d(nn::Conv2dOptions(CHANELS_NUM, 64, 2).stride(2).padding(3));
    layer1 = make_layer(64, 1);
    layer2 = make_layer(64, 2);
    layer3 = make_layer(64, 2);
    layer4 = make_layer(64, 2);
    fc = register_module("fc", nn::Linear(64, 4096));
}

nn::Sequential ResNet18::make_layer(alnet_t planes, alnet_t stride)
{
    nn::Sequential downsample = nullptr;
    if (stride != 1 || inplanes != planes)
        downsample = nn::Sequential(
                    nn::Conv2d(nn::Conv2dOptions(inplanes, planes, 1).stride(stride).padding(0)),
                    nn::BatchNorm2d(planes)
                    );
    return nn::Sequential(BasicBlock(inplanes, planes, stride, downsample),
                          BasicBlock(inplanes, planes, 1, nullptr),
                          BasicBlock(inplanes, planes, 1, nullptr));
}

Tensor ResNet18::forward(Tensor x)
{
    x = x.view({x.size(0), 1, PICTURE_HEIGHT, PICTURE_WIDTH});
    x = relu(convv1->forward(x));
    x = nn::BatchNorm2d(64)->forward(x);
    x = max_pool2d(x, 2, 2);

    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);

    x = avg_pool2d(x, 2, 1);
    x = x.view({x.size(0), -1});
    x = fc->forward(x);

    return x;
}

