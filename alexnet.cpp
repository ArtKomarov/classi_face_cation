#include "alexnet.h"
#include "clfc.hpp"
#include "torch/torch.h"

#include <iostream>

using namespace torch;

const alnet_t CHANELS_NUM = 1;
const alnet_t POOLS_NUM   = 3;

const alnet_t STRIDE_SIZE    = 1;
const alnet_t PADDING_SIZE_1 = 2;
const alnet_t PADDING_SIZE_2 = 1;

const alnet_t OUT_NEURONS_CONV_1 = 64;
const alnet_t OUT_NEURONS_CONV_2 = OUT_NEURONS_CONV_1 * 3;     // 192
const alnet_t OUT_NEURONS_CONV_3 = OUT_NEURONS_CONV_2 * 2;     // 384
const alnet_t OUT_NEURONS_CONV_4 = OUT_NEURONS_CONV_3 * 2 / 3; // 256
const alnet_t OUT_NEURONS_CONV_5 = OUT_NEURONS_CONV_4;         // 256

const alnet_t OUT_PIC_SIZE = OUT_NEURONS_CONV_5 * PICTURE_SIZE / pow(4, POOLS_NUM); // 256 * 6 * 6

const alnet_t FC_NEURONS_NUM = 4096;

const std::string FORWARD_ERROR_STR = "Forward failed: ";


// Build network
Net::Net(int num_classes) :
    // Full cover layers
    fc1   (register_module("fc1",   nn::Linear (OUT_PIC_SIZE,   FC_NEURONS_NUM))),
    fc2   (register_module("fc2",   nn::Linear (FC_NEURONS_NUM, FC_NEURONS_NUM))),
    fc3   (register_module("fc3",   nn::Linear (FC_NEURONS_NUM, num_classes))),
    // Convolution layers
    conv1 (register_module("conv1", nn::Conv2d (nn::Conv2dOptions (CHANELS_NUM,        OUT_NEURONS_CONV_1, 5).stride(STRIDE_SIZE).padding(PADDING_SIZE_1)))),
    conv2 (register_module("conv2", nn::Conv2d (nn::Conv2dOptions (OUT_NEURONS_CONV_1, OUT_NEURONS_CONV_2, 5).stride(STRIDE_SIZE).padding(PADDING_SIZE_1)))),
    conv3 (register_module("conv3", nn::Conv2d (nn::Conv2dOptions (OUT_NEURONS_CONV_2, OUT_NEURONS_CONV_3, 3).stride(STRIDE_SIZE).padding(PADDING_SIZE_2)))),
    conv4 (register_module("conv4", nn::Conv2d (nn::Conv2dOptions (OUT_NEURONS_CONV_3, OUT_NEURONS_CONV_4, 3).stride(STRIDE_SIZE).padding(PADDING_SIZE_2)))),
    conv5 (register_module("conv5", nn::Conv2d (nn::Conv2dOptions (OUT_NEURONS_CONV_4, OUT_NEURONS_CONV_5, 3).stride(STRIDE_SIZE).padding(PADDING_SIZE_2)))) {
}

// Implement the Net's algorithm.
Tensor Net::forward(Tensor x) {
    try {
    x = x.view({x.size(0), 1, PICTURE_HEIGHT, PICTURE_WIDTH}); //reshape

    x = relu(conv1->forward(x));
    x = max_pool2d(x, 2, 2);

    x = relu(conv2->forward(x));
    x = max_pool2d(x, 2);

    x = relu(conv3->forward(x));
    x = relu(conv4->forward(x));
    x = relu(conv5->forward(x));
    x = max_pool2d(x, 2);

    x = x.view({x.size(0), 2 + OUT_PIC_SIZE});

    x = dropout(x, /*p=*/0.6, /*train=*/is_training());
    x = relu(fc1->forward(x));
    x = nn::BatchNorm1d(4096)->forward(x);

    x = dropout(x, /*p=*/0.6, /*train=*/is_training());
    x = relu(fc2->forward(x));
    x = nn::BatchNorm1d(4096)->forward(x);

    x = fc3->forward(x);
    //x = torch::nn::functional::log_softmax(x, /*dim=*/1);
    } catch (const c10::ValueError& ve) {
        std::cerr << FORWARD_ERROR_STR << ve.what() << std::endl;
        std::cerr << ve.msg() << std::endl;
    } catch (const c10::IndexError& ie) {
        std::cerr << FORWARD_ERROR_STR << ie.what() << std::endl;
        std::cerr << ie.msg() << std::endl;
    } catch (const c10::EnforceFiniteError& efe) {
        std::cerr << FORWARD_ERROR_STR << efe.what() << std::endl;
        std::cerr << efe.msg() << std::endl;
    } catch (c10::Error oe) {
        std::cerr << FORWARD_ERROR_STR << oe.what() << std::endl;
        std::cerr << oe.msg() << std::endl;
    } catch (...) {
        std::cerr << FORWARD_ERROR_STR << std::endl;
    }

    return x;
}

// TORCH_MODULE(Net); //for NetImpl
