#include "customdataset.hpp"
#include "torch/torch.h"
#include "clfc.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>

// options
const torch::TensorOptions options = torch::TensorOptions(torch::kFloat32)
        .layout(torch::kStrided);
//    .device(torch::kCUDA, 1)
//.requires_grad(true);

const torch::TensorOptions lbl_options = torch::TensorOptions(torch::kLong)
        .layout(torch::kStrided);
//    .device(torch::kCUDA, 1)
//.requires_grad(true);

const size_t DOTS_PERIOD    = 800;
const std::string IMG_PROC_FAIL_MSG = "Processing images failed: ";

class TrainTestData {
    data_loader_t train_;
    std::vector<torch::Tensor> test_img_;
    std::vector<torch::Tensor> test_lbl_;
public:
    TrainTestData(std::string dataset_name, float test_size = 0) {
        std::vector<torch::Tensor> images;
        std::vector<torch::Tensor> labels;
        if(test_size > 1 || test_size < 0) {
            std::cerr << "Error: test size must be in interval [0, 1])!" << std::endl;
            return;
        }

        std::ifstream database(dataset_name);
        if(!database) {
            std::cerr << "Error: Can't open database file properly!" << std::endl;
            return;
        }

        std::cout << "Start processing images" << std::endl;

        std::stringstream str_image; // for reading strings with pixeles
        std::string str_label;       // for reading strings with label
        std::string sup;             // for reading other strings

        int label_int;
        float pix;
        float image_float[PICTURE_SIZE];

        unsigned int start_time = clock(); // To measure the running time

        try {
            // Read first row with column names
            std::getline(database, sup);
            std::getline(database, str_label, ','); // Get label
        } catch(const std::ifstream::failure& e) {
            std::cerr << "Data procession failed at the beginning,"
                         " so program terminated. Error:" << std::endl;
            std::cerr << IMG_PROC_FAIL_MSG << e.what() << std::endl;
            std::cerr << "failbit: "  << database.fail()
                      << "\neofbit: " << database.eof()
                      << "\nbadbit: " << database.bad() << std::endl;
            exit(-1);
        }

        bool lbl_greater = false; // Shows, that vector with labels has bigger size


        uli dots = 0; // for showing progress

        while(!database.eof()) { //database.good()
            dots++;
            if(dots % DOTS_PERIOD == 0)
                std::cout << "." << std::flush;

            //label_int = std::stoi(str_label); зачем нам тут лишние exception?
            //char *end; // Checker
            //errno = 0; // Set for future checking
            //label_int = std::strtol(str_label.c_str(), &end, 0);

            //if(str_label.c_str() == end || errno == ERANGE) {

            try {
                label_int = std::stoi(str_label); // Хотите c++ будет вам catch в catch(
            } catch(std::invalid_argument& e){    // If fall, current row will be skiped
                    // if no conversion could be performed
                    std::cerr << "Some label has invalid format!" << std::endl;
                    STD_STOI_FAIL_CASE(database, sup, str_label)
                    continue;
            } catch(std::out_of_range& e){
                // if the converted value would fall out of the range of the result type
                std::cerr << "Some label is out of range!" << std::endl;
                STD_STOI_FAIL_CASE(database, sup, str_label)
                continue;
            }

            try {
                labels.push_back(torch::full({1}, label_int, lbl_options)); // Create Tensor with label and push it to labels_
                lbl_greater = true;
                std::getline(database, sup, ','); // Get string with pixels

                //Create string stream
                str_image.clear();
                str_image.str(sup);

                // string with pixels to array with pixels
                for(uli el_num = 0; el_num < PICTURE_SIZE && !str_image.eof(); el_num++) {
                    str_image >> pix;                // Get pixel
                    image_float[el_num] = pix / 255; // Normalize
                }

                // Create Tensor with image and push it to images_
                images.push_back(torch::from_blob(image_float, {PICTURE_HEIGHT, PICTURE_WIDTH}, options).clone());
                lbl_greater = false;


                std::getline(database, sup);            // Skip last column
                std::getline(database, str_label, ','); // Get label

            } catch (const std::ifstream::failure &e) {
                if(lbl_greater)
                    labels_.pop_back();

                std::cerr << IMG_PROC_FAIL_MSG << e.what() << std::endl;
                std::cerr << "failbit: "  << database.fail()
                          << "\neofbit: " << database.eof()
                          << "\nbadbit: " << database.bad() << std::endl;
                return;
            } catch (const std::bad_alloc &ba) {
                std::cerr << IMG_PROC_FAIL_MSG << ba.what() << std::endl;
                if(lbl_greater)
                    labels.pop_back();
                return;
            }
            catch (...) {
                if(lbl_greater)
                    labels.pop_back();

                std::cerr << IMG_PROC_FAIL_MSG << "unknown exception" << std::endl;
                return;
            }
        }


        std::cout << std::endl; // Dots end

        //Split on train/test data
        if(test_size != 0) {
            auto img_begin = images_.begin();
            uli train_size = uli((1 - test_size) * labels.size());

            test_img_.assign( img_begin + train_size + 1, images.end() );
            test_lbl_.assign( labels.begin() + train_size + 1, labels.end() );

            images.assign( img_begin, img_begin + train_size );
            labels.assign( labels.begin(), labels.begin() + train_size );
        }

        auto custom_dataset = CustomDataset(images, labels).map(data::transforms::Stack<>());

        train_ = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                    std::move(custom_dataset),
                    batch_size
                    );

        // Information for user
        std::cout << "End processing images" << std::endl;
        std::cout << "Train data size: "     << labels_.size()      << std::endl;
        //std::cout << "Test data size: "      << test_labels_.size() << std::endl;
        std::cout << "Executed time:"        << (clock() - start_time) / CLOCKS_PER_SEC <<  "sec" << std::endl;

    }

};


CustomDataset::CustomDataset(std::string dataset_name, float test_size = 0) {
    if(test_size > 1 || test_size < 0) {
        std::cerr << "Error: test size must be in interval [0, 1])!" << std::endl;
        return;
    }

    std::ifstream database(dataset_name);
    if(!database) {
        std::cerr << "Error: Can't open database file properly!" << std::endl;
        return;
    }

    std::cout << "Start processing images" << std::endl;

    std::stringstream str_image; // for reading strings with pixeles
    std::string str_label;       // for reading strings with label
    std::string sup;             // for reading other strings

    int label_int;
    float pix;
    float image_float[PICTURE_SIZE];

    unsigned int start_time = clock(); // To measure the running time

    try {
        // Read first row with column names
        std::getline(database, sup);
        std::getline(database, str_label, ','); // Get label
    } catch(const std::ifstream::failure& e) {
        std::cerr << "Data procession failed at the beginning,"
                     " so program terminated. Error:" << std::endl;
        std::cerr << IMG_PROC_FAIL_MSG << e.what() << std::endl;
        std::cerr << "failbit: "  << database.fail()
                  << "\neofbit: " << database.eof()
                  << "\nbadbit: " << database.bad() << std::endl;
        exit(-1);
    }

    bool lbl_greater = false; // Shows, that vector with labels has bigger size


    uli dots = 0; // for showing progress

    while(!database.eof()) { //database.good()
        dots++;
        if(dots % DOTS_PERIOD == 0)
            std::cout << "." << std::flush;

        //label_int = std::stoi(str_label); зачем нам тут лишние exception?
        //char *end; // Checker
        //errno = 0; // Set for future checking
        //label_int = std::strtol(str_label.c_str(), &end, 0);

        //if(str_label.c_str() == end || errno == ERANGE) {

        try {
            label_int = std::stoi(str_label); // Хотите c++ будет вам catch в catch(
        } catch(std::invalid_argument& e){    // If fall, current row will be skiped
                // if no conversion could be performed
                std::cerr << "Some label has invalid format!" << std::endl;
                STD_STOI_FAIL_CASE(database, sup, str_label)
                continue;
        } catch(std::out_of_range& e){
            // if the converted value would fall out of the range of the result type
            std::cerr << "Some label is out of range!" << std::endl;
            STD_STOI_FAIL_CASE(database, sup, str_label)
            continue;
        }

        try {
            labels_.push_back(torch::full({1}, label_int, lbl_options)); // Create Tensor with label and push it to labels_
            lbl_greater = true;
            std::getline(database, sup, ','); // Get string with pixels

            //Create string stream
            str_image.clear();
            str_image.str(sup);

            // string with pixels to array with pixels
            for(uli el_num = 0; el_num < PICTURE_SIZE && !str_image.eof(); el_num++) {
                str_image >> pix;                // Get pixel
                image_float[el_num] = pix / 255; // Normalize
            }

            // Create Tensor with image and push it to images_
            images_.push_back(torch::from_blob(image_float, {PICTURE_HEIGHT, PICTURE_WIDTH}, options).clone());
            lbl_greater = false;


            std::getline(database, sup);            // Skip last column
            std::getline(database, str_label, ','); // Get label

        } catch (const std::ifstream::failure &e) {
            if(lbl_greater)
                labels_.pop_back();

            std::cerr << IMG_PROC_FAIL_MSG << e.what() << std::endl;
            std::cerr << "failbit: "  << database.fail()
                      << "\neofbit: " << database.eof()
                      << "\nbadbit: " << database.bad() << std::endl;
            return;
        } catch (const std::bad_alloc &ba) {
            std::cerr << IMG_PROC_FAIL_MSG << ba.what() << std::endl;
            if(lbl_greater)
                labels_.pop_back();
            return;
        }
        catch (...) {
            if(lbl_greater)
                labels_.pop_back();

            std::cerr << IMG_PROC_FAIL_MSG << "unknown exception" << std::endl;
            return;
        }
    }


    std::cout << std::endl; // Dots end

    //            //Split on train/test data
    //            if(test_size != 0) {
    //                auto img_begin = images_.begin();
    //                uli train_size = uli((1 - test_size) * labels_.size());

    //                test_images_.assign( img_begin + train_size + 1, images_.end() );
    //                test_labels_.assign( labels_.begin() + train_size + 1, labels_.end() );

    //                images_.assign( img_begin, img_begin + train_size );
    //                labels_.assign( labels_.begin(), labels_.begin() + train_size );
    //            }

    // Information for user
    std::cout << "End processing images" << std::endl;
    std::cout << "Train data size: "     << labels_.size()      << std::endl;
    //std::cout << "Test data size: "      << test_labels_.size() << std::endl;
    std::cout << "Executed time:"        << (clock() - start_time) / CLOCKS_PER_SEC <<  "sec" << std::endl;

}

// Get one sample (image and label)
torch::data::Example<> CustomDataset::get(size_t index) {
    try {
        torch::Tensor sample_img = images_.at(index);
        torch::Tensor sample_lbl = labels_.at(index);
        return {sample_img, sample_lbl}; //.clone()
    }
    catch(const std::out_of_range &oof) {
        std::cerr << oof.what() << std::endl;
        std::cerr << "The random value returned" << std::endl;
        size_t id = rand() % static_cast<int>(this->size().value());
        try {
            torch::Tensor sample_img = images_.at(id);
            torch::Tensor sample_lbl = labels_.at(id);
            return {sample_img, sample_lbl}; //.clone()
        }
        catch(const std::out_of_range &oof) {
            std::cerr << "WTF! Your vector is irreversibly broken!" << std::endl;
            std::cerr << oof.what() << std::endl;
            exit(0);
        }
    }
}


// Get number of samples
torch::optional<size_t> CustomDataset::size() const {
    return labels_.size(); //never throws exceptions
}
