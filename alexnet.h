#ifndef ALEXNET_H
#define ALEXNET_H

#include "torch/torch.h"

using namespace torch;

typedef size_t alnet_t;

extern const alnet_t CHANELS_NUM;
extern const alnet_t POOLS_NUM;

extern const alnet_t STRIDE_SIZE;
extern const alnet_t PADDING_SIZE_1;
extern const alnet_t PADDING_SIZE_2;

extern const alnet_t OUT_NEURONS_CONV_1; // 64
extern const alnet_t OUT_NEURONS_CONV_2; // 192
extern const alnet_t OUT_NEURONS_CONV_3; // 384
extern const alnet_t OUT_NEURONS_CONV_4; // 256
extern const alnet_t OUT_NEURONS_CONV_5; // 256

extern const std::string FORWARD_ERROR_STR;

struct Net : nn::Module {
    nn::Linear fc1;
    nn::Linear fc2;
    nn::Linear fc3;

    nn::Conv2d conv1;
    nn::Conv2d conv2;
    nn::Conv2d conv3;
    nn::Conv2d conv4;
    nn::Conv2d conv5;

    Net(int num_classes);

    Tensor forward(Tensor x);
};

#endif // ALEXNET_H
