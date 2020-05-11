#include "nets.hpp"
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>

/*extern const ssize_t PICTURE_HEIGHT;
extern const ssize_t PICTURE_WIDTH;
extern const ssize_t PICTURE_SIZE;*/

const ssize_t PICTURE_HEIGHT = 48;
const ssize_t PICTURE_WIDTH = 48;
const ssize_t PICTURE_SIZE = PICTURE_HEIGHT * PICTURE_WIDTH;

torch::Tensor Net::forward(torch::Tensor x)
{
    return x;
}

OurNet::OurNet()
{
    conv1_1 = register_module("conv1_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 3).stride(1).padding(1)));
    conv1_2 = register_module("conv1_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)));

    conv2_1 = register_module("conv2_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)));
    conv2_2 = register_module("conv2_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)));

    conv3_1 = register_module("conv3_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)));
    conv3_2 = register_module("conv3_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)));

    fc1 = register_module("fc1", torch::nn::Linear(256 * 6 * 6, 32));
    fc2 = register_module("fc2", torch::nn::Linear(32, 7));

    //std::cout << "Our" << std::endl;
}

torch::Tensor OurNet::forward(torch::Tensor x)
{
    x = x.view({x.size(0), 1, PICTURE_HEIGHT, PICTURE_WIDTH});

    x = torch::relu(conv1_1->forward(x));
    x = torch::relu(conv1_2->forward(x));
    x = torch::nn::BatchNorm2d(64)->forward(x);
    x = torch::max_pool2d(x, 2, 2);

    x = torch::relu(conv2_1->forward(x));
    x = torch::relu(conv2_2->forward(x));
    x = torch::nn::BatchNorm2d(128)->forward(x);
    x = torch::max_pool2d(x, 2, 2);

    x = torch::relu(conv3_1->forward(x));
    x = torch::relu(conv3_2->forward(x));
    x = torch::nn::BatchNorm2d(256)->forward(x);
    x = torch::max_pool2d(x, 2, 2);

    x = x.view({-1, 256 * 6 * 6});

    try {
        x = fc1->forward(x);
    }
    catch (c10::IndexError er){
        //std::cout << er << std::endl;
    }
    x = torch::nn::BatchNorm1d(32)->forward(x);
    x = torch::relu(x);
    x = torch::dropout(x, 0.2, is_training());
    x = fc2->forward(x);

    return x;
}


AlexNet::AlexNet()
{
    // Construct and register two Linear submodules.
    //fc1  = register_module("f1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 5).padding(1)));

    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 5).stride(1).padding(2)));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 5).stride(1).padding(2)));
    conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).stride(1).padding(1)));
    conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).stride(1).padding(1)));
    conv5 = register_module("conv5", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)));

    fc1 = register_module("fc1", torch::nn::Linear(256 * 6 * 6, 4096));
    fc2 = register_module("fc2", torch::nn::Linear(4096, 4096));
    fc3 = register_module("fc3", torch::nn::Linear(4096, 7));

    //std::cout << "Alex" << std::endl;
}

    // Implement the Net's algorithm.
torch::Tensor AlexNet::forward(torch::Tensor x)
{
    // Use one of many tensor manipulation functions.
    //std::cout << "type " << (typeid(x) == typeid(torch::Tensor)) << std::endl;
    x = x.view({x.size(0), 1, PICTURE_HEIGHT, PICTURE_WIDTH}); //reshape

    x = torch::relu(conv1->forward(x));
    x = torch::max_pool2d(x, 2, 2);
    x = torch::relu(conv2->forward(x));
    x = torch::max_pool2d(x, 2);

    x = torch::relu(conv3->forward(x));
    x = torch::relu(conv4->forward(x));
    x = torch::relu(conv5->forward(x));
    x = torch::max_pool2d(x, 2);

    x = x.view({x.size(0), 256 * 6 * 6});

    x = torch::dropout(x, /*p=*/0.7, /*train=*/is_training());
    x = torch::relu(fc1->forward(x));
    x = torch::nn::BatchNorm1d(4096)->forward(x);
    x = torch::dropout(x, /*p=*/0.7, /*train=*/is_training());
    x = torch::relu(fc2->forward(x));
    x = torch::nn::BatchNorm1d(4096)->forward(x);
    x = fc3->forward(x);
    //x = torch::nn::functional::log_softmax(x, /*dim=*/1);
    return x;
}

SmallNet::SmallNet()
{
    // Construct and register two Linear submodules.
    //fc1  = register_module("f1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 5).padding(1)));

    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 5).stride(1).padding(2)));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)));

    fc1 = register_module("fc1", torch::nn::Linear(12 * 12 * 128, 256));
    fc2 = register_module("fc2", torch::nn::Linear(256, 7));

    //std::cout << "Small" << std::endl;
}

torch::Tensor SmallNet::forward(torch::Tensor x)
{
    // Use one of many tensor manipulation functions.
    //std::cout << "type " << (typeid(x) == typeid(torch::Tensor)) << std::endl;
    x = x.view({x.size(0), 1, PICTURE_HEIGHT, PICTURE_WIDTH}); //reshape

    x = torch::dropout(x, 0.2, is_training());
    x = torch::relu(conv1->forward(x));
    x = torch::nn::BatchNorm2d(64)->forward(x);
    x = torch::max_pool2d(x, 2, 2);

    x = torch::dropout(x, 0.5, is_training());
    x = torch::relu(conv2->forward(x));
    x = torch::nn::BatchNorm2d(128)->forward(x);
    x = torch::max_pool2d(x, 2, 2);

    x = x.view({x.size(0), 12 * 12 * 128});

    x = torch::dropout(x, 0.5, is_training());
    x = torch::relu(fc1->forward(x));
    x = torch::nn::BatchNorm1d(256)->forward(x);
    x = torch::dropout(x, 0.2, is_training());
    x = fc2->forward(x);
    //x = torch::nn::functional::log_softmax(x, /*dim=*/1);
    return x;
}

VGG16::VGG16()
{
    // Construct and register two Linear submodules.
    //fc1  = register_module("f1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 5).padding(1)));

    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 3).stride(1).padding(1)));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)));
    conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)));
    conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)));
    conv5 = register_module("conv5", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)));
    conv6 = register_module("conv6", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)));
    conv7 = register_module("conv7", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)));
    conv8 = register_module("conv8", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)));

    fc1 = register_module("fc1", torch::nn::Linear(512 * 3 * 3, 4096));
    fc2 = register_module("fc2", torch::nn::Linear(4096, 4096));
    fc3 = register_module("fc3", torch::nn::Linear(4096, 7));

    //std::cout << "VGG16" << std::endl;
}

torch::Tensor VGG16::forward(torch::Tensor x)
{
    // Use one of many tensor manipulation functions.
    //std::cout << "type " << (typeid(x) == typeid(torch::Tensor)) << std::endl;
    x = x.view({x.size(0), 1, PICTURE_HEIGHT, PICTURE_WIDTH}); //reshape

    x = conv1->forward(x);
    x = conv2->forward(x);
    x = torch::max_pool2d(x, 2, 2);

    x = conv3->forward(x);
    x = conv4->forward(x);
    x = torch::max_pool2d(x, 2, 2);

    x = conv5->forward(x);
    x = conv6->forward(x);
    x = conv6->forward(x);
    x = torch::max_pool2d(x, 2, 2);

    x = conv7->forward(x);
    x = conv8->forward(x);
    x = conv8->forward(x);
    x = torch::max_pool2d(x, 2, 2);

    x = x.view({x.size(0), 512 * 3 * 3});

    x = fc1->forward(x);
    x = fc2->forward(x);
    x = fc3->forward(x);
    //x = torch::nn::functional::log_softmax(x, /*dim=*/1);
    return x;
}

