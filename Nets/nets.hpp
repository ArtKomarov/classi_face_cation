
//#ifndef ALEXNET_H
//#define ALEXNET_H

#include "torch/torch.h"

using namespace torch;

typedef size_t alnet_t;

extern const alnet_t CHANELS_NUM;

/*extern const alnet_t STRIDE_SIZE;
extern const alnet_t PADDING_SIZE_1;
extern const alnet_t PADDING_SIZE_2;

extern const alnet_t OUT_NEURONS_CONV_1; // 64
extern const alnet_t OUT_NEURONS_CONV_2; // 192
extern const alnet_t OUT_NEURONS_CONV_3; // 384
extern const alnet_t OUT_NEURONS_CONV_4; // 256
extern const alnet_t OUT_NEURONS_CONV_5; // 256*/

extern const std::string FORWARD_ERROR_STR;

class Net: public nn::Module
{
public:
    virtual Tensor forward(torch::Tensor x);
};

class OurNet: public Net//public torch::nn::Module
{
public:
    nn::Conv2d conv1_1{nullptr};
    nn::Conv2d conv1_2{nullptr};
    nn::Conv2d conv2_1{nullptr};
    nn::Conv2d conv2_2{nullptr};
    nn::Conv2d conv3_1{nullptr};
    nn::Conv2d conv3_2{nullptr};

    nn::Linear fc1{nullptr}, fc2{nullptr};

    OurNet();
    Tensor forward(Tensor x);
};

class AlexNet : public Net//public torch::nn::Module
{
public:
    nn::Conv2d conv1 = nullptr;
    nn::Conv2d conv2 = nullptr;
    nn::Conv2d conv3 = nullptr;
    nn::Conv2d conv4 = nullptr;
    nn::Conv2d conv5 = nullptr;

    nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    AlexNet();
    Tensor forward(Tensor x);
};

class SmallNet : public Net//public torch::nn::Module
{
public:
    nn::Conv2d conv1{nullptr};
    nn::Conv2d conv2{nullptr};

    nn::Linear fc1{nullptr}, fc2{nullptr};

    SmallNet();
    Tensor forward(Tensor x);
};

class VGG16 : public Net //public torch::nn::Module
{
public:
    nn::Conv2d conv1 = nullptr;
    nn::Conv2d conv2 = nullptr;
    nn::Conv2d conv3 = nullptr;
    nn::Conv2d conv4 = nullptr;
    nn::Conv2d conv5 = nullptr;
    nn::Conv2d conv6 = nullptr;
    nn::Conv2d conv7 = nullptr;
    nn::Conv2d conv8 = nullptr;

    nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    VGG16();
    Tensor forward(Tensor x);
};

class BasicBlock : public Net //public torch::nn::Module
{
public:
    alnet_t expansion;
    alnet_t planes;

    //nn::BatchNorm2d bn1 = nullptr;
    nn::Conv2d conv1 = nullptr;
    //nn::BatchNorm2d bn2 = nullptr;
    nn::Conv2d conv2 = nullptr;
    nn::Sequential downsample = nullptr;
    alnet_t stride;

    BasicBlock(alnet_t inplanes, alnet_t planes, alnet_t stride, nn::Sequential downsample);
    Tensor forward(Tensor x);
};

class ResNet18 : public Net
{
public:
    alnet_t inplanes;
    nn::Conv2d convv1 = nullptr;

    nn::Sequential layer1 = nullptr;
    nn::Sequential layer2 = nullptr;
    nn::Sequential layer3 = nullptr;
    nn::Sequential layer4 = nullptr;

    nn::Linear fc = nullptr;

    ResNet18();
    nn::Sequential make_layer(alnet_t planes, alnet_t stride = 1);
    Tensor forward(Tensor x);
};

//#endif // NETS_H
