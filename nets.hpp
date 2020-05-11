#ifndef NETS_H
#define NETS_H
#include <torch/torch.h>
#include <ATen/ATen.h>

class Net: public torch::nn::Module
{
public:
    virtual torch::Tensor forward(torch::Tensor x);
};

class OurNet: public Net//public torch::nn::Module
{
public:
    torch::nn::Conv2d conv1_1{nullptr};
    torch::nn::Conv2d conv1_2{nullptr};
    torch::nn::Conv2d conv2_1{nullptr};
    torch::nn::Conv2d conv2_2{nullptr};
    torch::nn::Conv2d conv3_1{nullptr};
    torch::nn::Conv2d conv3_2{nullptr};

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    OurNet();
    torch::Tensor forward(torch::Tensor x);
};

class AlexNet : public Net//public torch::nn::Module
{
public:
    torch::nn::Conv2d conv1 = nullptr;
    torch::nn::Conv2d conv2 = nullptr;
    torch::nn::Conv2d conv3 = nullptr;
    torch::nn::Conv2d conv4 = nullptr;
    torch::nn::Conv2d conv5 = nullptr;

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    AlexNet();
    torch::Tensor forward(torch::Tensor x);
};

class SmallNet : public Net//public torch::nn::Module
{
public:
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    SmallNet();
    torch::Tensor forward(torch::Tensor x);
};

class VGG16 : public Net //public torch::nn::Module
{
public:
    torch::nn::Conv2d conv1 = nullptr;
    torch::nn::Conv2d conv2 = nullptr;
    torch::nn::Conv2d conv3 = nullptr;
    torch::nn::Conv2d conv4 = nullptr;
    torch::nn::Conv2d conv5 = nullptr;
    torch::nn::Conv2d conv6 = nullptr;
    torch::nn::Conv2d conv7 = nullptr;
    torch::nn::Conv2d conv8 = nullptr;

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    VGG16();
    torch::Tensor forward(torch::Tensor x);
};

#endif // NETS_H
