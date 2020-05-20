#ifndef CUSTOMDATASET_HPP
#define CUSTOMDATASET_HPP

#include "torch/torch.h"

extern const torch::TensorOptions options;
extern const torch::TensorOptions lbl_options;

extern const size_t      DOTS_PERIOD;
extern const std::string IMG_PROC_FAIL_MSG;

class CustomDataset : public torch::data::Dataset<CustomDataset> {
    std::vector <torch::Tensor> images_, labels_;
//    std::vector <torch::Tensor> test_images_, test_labels_;

public:
    CustomDataset(std::vector<torch::Tensor> &images, std::vector<torch::Tensor> &labels);

    // Get one sample (image and label)
    torch::data::Example<> get(size_t index) override;

    // Get number of samples
    torch::optional<size_t> size() const override;
};

typedef torch::disable_if_t<false, std::unique_ptr<torch::data::StatelessDataLoader\
<torch::data::datasets::MapDataset<CustomDataset, torch::data::transforms::Stack<torch::data::Example<> > >,\
torch::data::samplers::SequentialSampler>, std::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<CustomDataset,\
torch::data::transforms::Stack<torch::data::Example<> > >, torch::data::samplers::SequentialSampler> > > > data_loader_t;

struct TrainTestData {
private:
    bool good_;
public:
    data_loader_t train_;
    data_loader_t test_;
    TrainTestData(std::string dataset_name, float test_size);
    bool good() const;
};


#define STD_STOI_FAIL_CASE(database, sup, str_label) \
try {                                                \
    std::getline(database, sup); /* Skip row*/       \
                                                            \
    std::getline(database, str_label, ','); /* Get label */ \
} catch (const std::ifstream::failure &e) {                             \
    good_ = false;                                                      \
    std::cerr << IMG_PROC_FAIL_MSG << e.what() << std::endl;            \
    std::cerr << "failbit: "  << database.fail()                        \
              << "\neofbit: " << database.eof()                         \
              << "\nbadbit: " << database.bad() << std::endl;           \
    return;                                                             \
} catch (...) {                                                         \
    good_ = false;                                                      \
    std::cerr << IMG_PROC_FAIL_MSG << "unknown exception" << std::endl; \
    return;                                                             \
}


#endif // CUSTOMDATASET_HPP
