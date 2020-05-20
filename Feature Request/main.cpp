#include "torch/torch.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <ATen/ATen.h>
#include <vector>
#include <string>
#include <ctime>
#include <cerrno>
#include <unistd.h>

#include "customdataset.hpp"
#include "nets.hpp"

using namespace torch;

const size_t batch_size = 2;

int main(int argc, char **argv) { //
    std::string name; // добавлено
    if (argc < 2) //
    { 
        std::cout << "Enter the dataset name:" << std::endl; //
        std::cin >> name; //
    }
    else //
        name = string(argv[1]); //
    auto custom_dataset = CustomDataset(name, 0.3).map(data::transforms::Stack<>()); // изменено

    auto data_loader = data::make_data_loader<data::samplers::SequentialSampler>(
                std::move(custom_dataset),
                batch_size
                );

//    std::string sup;
//    std::cout << "Do you want to continue? (y/n)" << std::endl;
//    std::cin >> sup;
//    if(sup == "n")
//        return 0;

    int num = 0;
    auto net = std::make_shared<Net>();
    std::cout << "Enter the number:" << std::endl;
    std::cout << "1 - AlexNet" << std::endl;
    std::cout << "2 - OurNet" << std::endl;
    std::cout << "3 - SmallNet" << std::endl;
    std::cout << "4 - VGG16" << std::endl;
    std::cout << "5 - ResNet18" << std::endl;
    while (!num) // добавлено
    {
        std::cin >> num;
        switch (num)
        {
        case 1:
            net = std::make_shared<AlexNet>();
            break;
        case 2:
            net = std::make_shared<OurNet>();
            break;
        case 3:
            net = std::make_shared<SmallNet>();
            break;
        case 4:
            net = std::make_shared<VGG16>();
            break;
        case 5:
            net = std::make_shared<ResNet18>();
            break;
        default: // добавлено
            num = 0; //
            std::cout << "The number should be from 1 to 5" << std::endl; //
            break; //
        }
    }


    optim::Adam optimizer(net->parameters(),
                                 optim::AdamOptions(1e-3));
    //torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(1e-3));

    // Set loss function class
    auto loss_class = nn::CrossEntropyLoss();

    //Start training
    for(size_t epoch=1; epoch<=2; ++epoch) {
        size_t batch_index = 0;
        // Iterate data loader to yield batches from the dataset
        for (auto& batch : *data_loader) {
            // Reset gradients
            optimizer.zero_grad();
            // Execute the model
            Tensor prediction = net->forward(batch.data);
            // Compute loss value
            //prediction.squeeze_();
            std::cout << prediction << std::endl;
            // Calculate loss
            Tensor loss = loss_class(prediction, batch.target.squeeze()); //squeeze reshape tensor by remove dimetions with size 1

            // Compute gradients
            loss.backward();

            //std:: cout << "grad = " << optimizer.parameters() << std::endl; //can show, how it is BIG
            // Update the parameters
            optimizer.step();

            prediction.detach_(); //free memory (batch.data.clone())

            break;
            if (++batch_index % 2 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                          << " | Loss: " << loss.item<float>() << std::endl;
            }
        }

        // Save our model
        save(net, "net3.pt");
    }

    //torch::data::Example<> a = *((*data_loader).begin()); // Вывод - если ссылка, то память очищается (похоже, что происходит move assigment с последующим удалением)
    //std::cout << a.target << std::endl;

//    int stop = 0;
    for (auto& batch : *data_loader) { // А тут норм WTF???
        Tensor prediction = net->forward(batch.data); //BUUUUM утечка только из-за этого
        std::cout << loss_class(prediction, batch.target.squeeze()) << std::endl;

        //prediction.detach_(); //free memory (batch.data.clone())
        std::cout << batch.data;
//        if(++stop == 2)
//            break;
    }
    return 0;
}
