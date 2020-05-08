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
            labels_.push_back(torch::full({1}, label_int, options));

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

//// Define a new Module.
//struct Net : torch::nn::Module {
//    Net() {
//        // Construct and register two Linear submodules.
//        fc1 = register_module("fc1", torch::nn::Linear(784, 64));
//        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
//        fc3 = register_module("fc3", torch::nn::Linear(32, 10));
//    }

//    // Implement the Net's algorithm.
//    torch::Tensor forward(torch::Tensor x) {
//        // Use one of many tensor manipulation functions.
//        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
//        x = torch::dropout(x, /*p=*/0.7, /*train=*/is_training());
//        x = torch::relu(fc2->forward(x));
//        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
//        return x;
//    }

//    // Use one of many "standard library" modules.
//    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
//};


int main() {
    CustomDataset custom_dataset("fer2013.csv");
    int batch_size = 100;

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      std::move(custom_dataset),
      batch_size
    );

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
