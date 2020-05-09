#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <ATen/ATen.h>
#include <vector>
#include <string>
#include <ctime>

typedef unsigned long int uli;

const uli DATABASE_SIZE = (uli)pow(2, 27); //с запасом

const size_t PICTURE_HEIGHT = 48;
const size_t PICTURE_WIDTH = 48;
const size_t PICTURE_SIZE = PICTURE_HEIGHT * PICTURE_WIDTH;

// options
const torch::TensorOptions options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        //    .device(torch::kCUDA, 1)
        .requires_grad(true);

const torch::TensorOptions lbl_options = torch::TensorOptions()
        .dtype(torch::kLong)
        .layout(torch::kStrided);
        //    .device(torch::kCUDA, 1)
        //.requires_grad(true);

int batch_size = 20;

#if 0 // Может пригодится

std::vector<torch::Tensor> convert_labels(std::vector<int> lbls) {
    std::vector<torch::Tensor> labels;
    for(const auto &i : lbls) {
        labels.push_back(torch::full({1}, i, options));
    }
    return labels;
    //return torch::from_blob(lbls.data(), {lbls.size()}, options);
}

int convert_images(std::vector<std::string> str_imgs, std::vector<torch::Tensor> &tens_img) {
    //std::vector<torch::Tensor> images;
    std::stringstream ss;
    std::string str_num;
    float pix[PICTURE_SIZE];
    std::cout << "str_imgs vector size : " << str_imgs.size() << std::endl;
    int breakp = 0;
    for(const auto &img_iter : str_imgs) {
        ss.str(img_iter);
        size_t i = 0;
        while(!ss.eof()) {
            std::getline(ss, str_num, ' ');
            pix[i++] = std::stof(str_num);
            //i++;
        }
        tens_img.push_back(torch::from_blob(std::move(pix), {PICTURE_SIZE}, options));
        //std::cout << "images.data : " << images[breakp] << std::endl;
        ss.clear();

        ++breakp;
    }
    std::cout << "BREAKP : " << breakp << std::endl;
    std::cout << tens_img.at(35887 - 20) << std::endl;
    return 0;
    //return torch::from_blob(lbls.data(), {lbls.size()}, options);
}

#endif

class CustomDataset : public torch::data::Dataset<CustomDataset> {
    std::vector <torch::Tensor> images_, labels_;

public:
    CustomDataset(std::string dataset_name) {
        std::ifstream database(dataset_name);
        if(!database.good()) {
            std::cerr << "Can't open database file" << std::endl;
            return;
        }

        std::cout << "Start processing images" << std::endl;

        std::stringstream str_image;
        std::string str_label;
        std::string sup;

        int label_int;
        float pix;
        float image_float[PICTURE_SIZE];

        // Read first row with column names
        std::getline(database, sup);
        std::getline(database, str_label, ',');

        unsigned int start_time =  clock();

        uli el_num = 0;
        while(database.good() && !database.eof()) {
            label_int = std::stoi(str_label);
            labels_.push_back(torch::full({1}, label_int, lbl_options));

            std::getline(database, sup, ',');

            str_image.clear();
            str_image.str(sup);

            for(el_num = 0; el_num < PICTURE_SIZE && !str_image.eof(); el_num++) {
                str_image >> pix;
                image_float[el_num] = pix / 255; //normalize
            }

            images_.push_back(torch::from_blob(image_float, {PICTURE_HEIGHT, PICTURE_WIDTH}, options).clone());

            std::getline(database, sup);
            std::getline(database, str_label, ',');
        }

//        // Check, 35887 - number of rows
//        std::cout << "SUCCESS!!" << std::endl;
//        std::cout << images_.at(35887 - 24) << std::endl;
//        std::cout << "Label = " << labels_.at(35887 - 24) << std::endl;
//        std::cout << "Label = " << labels_.at(35887 - 25) << std::endl;
//        std::cout << "Label = " << labels_.at(35887 - 26) << std::endl;
//        std::cout << "Label = " << labels_.at(35887 - 27) << std::endl;

        std::cout << "End processing images" << std::endl;
        std::cout << "EXECUTED TIME:" << (clock() - start_time) / CLOCKS_PER_SEC <<  "sec" << std::endl;
    }

    torch::data::Example<> get(size_t index) override {
        torch::Tensor sample_img = images_.at(index);
        torch::Tensor sample_lbl = labels_.at(index);
        return {sample_img.clone(), sample_lbl.clone()};
    }

    torch::optional<size_t> size() const override {
        return labels_.size();
    }

};

// Define a new Module.
struct Net : torch::nn::Module {
    torch::nn::Conv2d conv1 = nullptr;
    torch::nn::Conv2d conv2 = nullptr;
    torch::nn::Conv2d conv3 = nullptr;
    torch::nn::Conv2d conv4 = nullptr;
    torch::nn::Conv2d conv5 = nullptr;


    Net(int num_classes) {
        // Construct and register two Linear submodules.
        //fc1  = register_module("f1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 5).padding(1)));

        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 5).stride(1).padding(2)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 5).stride(1).padding(2)));
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).stride(1).padding(1)));
        conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).stride(1).padding(1)));
        conv5 = register_module("conv5", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)));

        fc1 = register_module("fc1", torch::nn::Linear(256 * 6 * 6, 4096));
        fc2 = register_module("fc2", torch::nn::Linear(4096, 4096));
        fc3 = register_module("fc3", torch::nn::Linear(4096, num_classes));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x) {
        // Use one of many tensor manipulation functions.
        //std::cout << "type " << (typeid(x) == typeid(torch::Tensor)) << std::endl;
        x = x.view({x.size(0), 1, PICTURE_HEIGHT, PICTURE_WIDTH}); //reshape

        x = torch::relu(conv1->forward(x));
        x = torch::max_pool2d(x, 2);
        x = torch::relu(conv2->forward(x));
        x = torch::max_pool2d(x, 2);

        x = torch::relu(conv3->forward(x));
        x = torch::relu(conv4->forward(x));
        x = torch::relu(conv5->forward(x));
        x = torch::max_pool2d(x, 2);

        x = x.view({batch_size, 256 * 6 * 6});

        x = torch::dropout(x, /*p=*/0.7, /*train=*/is_training());
        x = torch::relu(fc1->forward(x));
        //x = torch::batch_norm(x);
        x = torch::dropout(x, /*p=*/0.7, /*train=*/is_training());
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        //x = torch::nn::functional::log_softmax(x, /*dim=*/1);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    // Use one of many "standard library" modules.
};


int main() {
    auto custom_dataset = CustomDataset("fer2013.csv").map(torch::data::transforms::Stack<>());

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      std::move(custom_dataset),
      batch_size
    );

//    for(auto& batch: *data_loader) {
//      auto data = batch.data;
//      auto target = batch.target.squeeze();
//    }

    auto net = std::make_shared<Net>(7);
    torch::optim::Adam optimizer(net->parameters(),
                                 torch::optim::AdamOptions(1e-3));
    auto loss_class = torch::nn::CrossEntropyLoss();

    for(size_t epoch=1; epoch<=10; ++epoch) {
        size_t batch_index = 0;
        // Iterate data loader to yield batches from the dataset
        for (auto& batch : *data_loader) {
            // Reset gradients
            optimizer.zero_grad();
            // Execute the model
            torch::Tensor prediction = net->forward(batch.data.clone());
            // Compute loss value
            std::cout << batch.target << std::endl;
            //prediction.squeeze_();
            std::cout << prediction << std::endl;
            torch::Tensor loss = loss_class(prediction, batch.target.squeeze());
            // Compute gradients
            loss.backward();
            // Update the parameters
            optimizer.step();

            if (++batch_index % 2 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                          << " | Loss: " << loss.item<float>() << std::endl;
            }
        }
    }

    return 0;

    //    // Create a new Net.
    //    auto net = std::make_shared<Net>();

    //    // Create a multi-threaded data loader for the MNIST dataset.
    //    auto data_loader = torch::data::make_data_loader(
    //        torch::data::datasets::MNIST("./data").map(
    //            torch::data::transforms::Stack<>()), /*batch_size=*/64);

    //    // Instantiate an Adam optimization algorithm to update our Net's parameters.
    //    torch::optim::Adam optimizer(net->parameters(), /*lr=*/0.001);

    //    for (size_t epoch = 1; epoch <= 10; ++epoch) {
    //        size_t batch_index = 0;
    //        // Iterate the data loader to yield batches from the dataset.
    //        for (auto& batch : *data_loader) {
    //            // Reset gradients.
    //            optimizer.zero_grad();
    //            // Execute the model on the input data.
    //            torch::Tensor prediction = net->forward(batch.data);
    //            // Compute a loss value to judge the prediction of our model.
    //            torch::Tensor loss = torch::nll_loss(prediction, batch.target);
    //            // Compute gradients of the loss w.r.t. the parameters of our model.
    //            loss.backward();
    //            // Update the parameters based on the calculated gradients.
    //            optimizer.step();
    //            // Output the loss and checkpoint every 100 batches.
    //            if (++batch_index % 100 == 0) {
    //                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index << " | Loss: " << loss.item<float>() << std::endl;
    //                // Serialize your model periodically as a checkpoint.
    //                torch::save(net, "net.pt");
    //            }
    //        }
    //    }
}

//    std::cout << "X_str.size = " << X_str.size() << std::endl;
//    std::ifstream database("fer2013.csv");
//    //database.good()
//    std::stringstream ss;
//    std::string s;
//    int label_i;
//    float pix;

//    //unsigned long long int num_of_el = 0;

//    // Read first row with column names
//    std::getline(database, s);
//    std::getline(database, s, ',');
//    //label_i = std::stoi(s);

//    std::cerr << DATABASE_SIZE << std::endl;


//    unsigned int start_time =  clock();
//    uli el_num = 0;
//    while(database.good() && !database.eof()) {
//        //        std::getline(database, s, ',');
//        //        std::cout << "s = " << s << std::endl;
//        label_i = std::stoi(s);
//        y.push_back(torch::full({1}, label_i, options));

//        std::getline(database, s, ',');
//        //std::cout << "s = " << s << std::endl;
//        ss.clear();
//        //X_str.push_back(s);
//        ss.str(s);
//        while(!ss.eof()) {
//            ss >> pix;
//            X_float[el_num] = pix;
//            //std::cout << X_float[el_num] << " ";
//            el_num++;
//        }

//        X.push_back(torch::from_blob(X_float, {PICTURE_SIZE}, options).clone());
//        el_num = 0;

//        std::getline(database, s);
//        std::getline(database, s, ',');
//        //std::cout << "s = " << s << std::endl;
//        //std::getline(database, s, ',');
//        //i++;
//        //        std::cout << "-----------------------------------------------------" << std::endl;
//        //        std::cout << "X_float[" << num_of_el - 2 << "] = " << X_float[num_of_el-2] << std::endl;
//        //        std::cout << num_of_el << std::endl;
//        //        std::cout << "-----------------------------------------------------" << std::endl;
//    }

//    printf("EXECUTED TIME:%d sec\n", (clock() - start_time) / CLOCKS_PER_SEC);
//    std::cout << "X_str.size = " << X_str.size() << std::endl;
//    std::cout << "y_int.size = " << y_int.size() << std::endl;
//    std::cout << "X = " << X << std::endl;
//    std::cout << "y = " << y[35887 - 20] << std::endl;
