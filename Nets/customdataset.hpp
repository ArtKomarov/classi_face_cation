#ifndef CUSTOMDATASET_HPP
#define CUSTOMDATASET_HPP

#include "torch/torch.h"

extern const torch::TensorOptions options;
extern const torch::TensorOptions lbl_options;

extern const size_t      DOTS_PERIOD;
extern const std::string IMG_PROC_FAIL_MSG;

class CustomDataset : public torch::data::Dataset<CustomDataset> {
    std::vector <torch::Tensor> images_,      labels_;
//    std::vector <torch::Tensor> test_images_, test_labels_;

public:
    CustomDataset(std::string dataset_name, float test_size);

    // Get one sample (image and label)
    torch::data::Example<> get(size_t index) override;

    // Get number of samples
    torch::optional<size_t> size() const override;
};

#endif // CUSTOMDATASET_HPP
