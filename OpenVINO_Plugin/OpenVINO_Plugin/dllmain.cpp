// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
using namespace InferenceEngine;

#define DLLExport __declspec (dllexport)

extern "C" {
    std::string currentDevice = "";
    std::string allDevices = "";
    
    cv::Mat texture;

    Core ie;
    std::vector<std::string> availableDevices;

    CNNNetwork network;

    std::string firstOutputName;

    ExecutableNetwork executable_network;
    InferRequest infer_request;

    DLLExport const std::string* GetAvailableDevices() {
        return &allDevices;
    }

    DLLExport void SetDeviceCache() {
        std::regex e("(GPU)(.*)");
        for (auto&& device : availableDevices) {
            if (std::regex_match(device, e)) {
                ie.SetConfig({ {CONFIG_KEY(CACHE_DIR), "cache"} }, device);
            }
        }
    }

    DLLExport void PrepareOutputBlobs() {
        OutputsDataMap outputInfo(network.getOutputsInfo());

        for (auto& item : outputInfo) {
            if (firstOutputName.empty()) {
                firstOutputName = item.first;
            }

            item.second->setPrecision(Precision::FP32);
        }
    }

    DLLExport void InitializeOpenVINO(char* modelPath) {
        
        // Read network file
        network = ie.ReadNetwork(modelPath);
        // Set batch size to one image
        network.setBatchSize(1);
        // Get the output name and set the output precision
        PrepareOutputBlobs();
        // Get a list of the available compute devices
        availableDevices = ie.GetAvailableDevices();
        std::reverse(availableDevices.begin(), availableDevices.end());
        for (auto&& device : availableDevices) {
            allDevices += device;
            allDevices += ((device == availableDevices[availableDevices.size() - 1]) ? "" : ",");
        }
        // Specify the cache directory for GPU inference
        SetDeviceCache();
    }

    DLLExport bool SetInputDims(int width, int height) {
        bool success = true;

        // ------------- 1. Collect the map of input names and shapes from IR---------------
        auto input_shapes = network.getInputShapes();
        // ---------------------------------------------------------------------------------

         // ------------- 2. Set new input shapes -------------------------------------------
        std::string input_name;
        InferenceEngine::SizeVector input_shape;
        std::tie(input_name, input_shape) = *input_shapes.begin(); // let's consider first input only
        input_shape[0] = 1; // set batch size to the first input dimension
        input_shape[2] = height; // changes input height to the image one
        input_shape[3] = width; // changes input width to the image one
        input_shapes[input_name] = input_shape;
        // ---------------------------------------------------------------------------------

        // ------------- 3. Call reshape ---------------------------------------------------
        try {
            network.reshape(input_shapes);
            texture = cv::Mat(height, width, CV_8UC4);
        }
        catch (...) {
            success = false;
        }
        // ---------------------------------------------------------------------------------
        return success;
    }

    
    DLLExport std::string* UploadModelToDevice(int deviceNum) {

        bool success = true;        
        std::string devices;
        
        currentDevice = availableDevices[deviceNum];
        
        try {
            executable_network = ie.LoadNetwork(network, currentDevice);
        }
        catch (...) {
            success = false;
        }
        return &currentDevice;
    }

    DLLExport void CreateInferenceRequest() {
        infer_request = executable_network.CreateInferRequest();
    }

    DLLExport void PerformInference(uchar* inputData, int width, int height) {

        texture.data = inputData;
        cv::cvtColor(texture, texture, cv::COLOR_RGBA2RGB);
        

        InputsDataMap inputInfo(network.getInputsInfo());
        for (const auto& item : inputInfo) {
            MemoryBlob::Ptr minput = as<MemoryBlob>(infer_request.GetBlob(item.first));

            // locked memory holder should be alive all time while access to its buffer happens
            auto ilmHolder = minput->wmap();

            /** Filling input tensor with images. First b channel, then g and r channels **/
            size_t num_channels = minput->getTensorDesc().getDims()[1];
            size_t image_size = minput->getTensorDesc().getDims()[3] * minput->getTensorDesc().getDims()[2];

            auto data = ilmHolder.as<PrecisionTrait<Precision::FP32>::value_type*>();
            if (data == nullptr)
                throw std::runtime_error("Input blob has not allocated buffer");

            /** Iterate over all pixel in image (b,g,r) **/
            for (size_t pid = 0; pid < image_size; pid++) {
                /** Iterate over all channels **/
                for (size_t ch = 0; ch < num_channels; ++ch) {
                    /**          [images stride + channels stride + pixel id ] all in bytes            **/
                    data[ch * image_size + pid] =  texture.data[pid * num_channels + ch];
                }
            }

            /** Iterate over all input images **/
            //for (size_t image_id = 0; image_id < 1; ++image_id) {
            //    /** Iterate over all pixel in image (b,g,r) **/
            //    for (size_t pid = 0; pid < image_size; pid++) {
            //        /** Iterate over all channels **/
            //        for (size_t ch = 0; ch < num_channels; ++ch) {
            //            /**          [images stride + channels stride + pixel id ] all in bytes            **/
            //            data[image_id * image_size * num_channels + ch * image_size + pid] =
            //                texture.data[pid * num_channels + ch];
            //        }
            //    }
            //}
        }

        bool success = true;
        try {
            infer_request.Infer();

            MemoryBlob::CPtr moutput = as<MemoryBlob>(infer_request.GetBlob(firstOutputName));
            
            // locked memory holder should be alive all time while access to its buffer happens
            auto lmoHolder = moutput->rmap();
            const auto output_data = lmoHolder.as<const PrecisionTrait<Precision::FP32>::value_type*>();

            size_t num_images = moutput->getTensorDesc().getDims()[0];
            size_t num_channels = moutput->getTensorDesc().getDims()[1];
            size_t H = moutput->getTensorDesc().getDims()[2];
            size_t W = moutput->getTensorDesc().getDims()[3];
            size_t nPixels = W * H;

            std::vector<float> data_img(nPixels * num_channels);

            for (size_t n = 0; n < num_images; n++) {
                for (size_t i = 0; i < nPixels; i++) {
                    data_img[i * num_channels] = static_cast<float>(output_data[i + n * nPixels * num_channels]);
                    data_img[i * num_channels + 1] = static_cast<float>(
                        output_data[(i + nPixels) + n * nPixels * num_channels]);
                    data_img[i * num_channels + 2] = static_cast<float>(
                        output_data[(i + 2 * nPixels) + n * nPixels * num_channels]);

                    float temp = data_img[i * num_channels];
                    data_img[i * num_channels] = data_img[i * num_channels + 2];
                    data_img[i * num_channels + 2] = temp;

                    if (data_img[i * num_channels] < 0) data_img[i * num_channels] = 0;
                    if (data_img[i * num_channels] > 255) data_img[i * num_channels] = 255;

                    if (data_img[i * num_channels + 1] < 0) data_img[i * num_channels + 1] = 0;
                    if (data_img[i * num_channels + 1] > 255) data_img[i * num_channels + 1] = 255;

                    if (data_img[i * num_channels + 2] < 0) data_img[i * num_channels + 2] = 0;
                    if (data_img[i * num_channels + 2] > 255) data_img[i * num_channels + 2] = 255;

                    texture.data[i * num_channels] = data_img[i * num_channels];
                    texture.data[i * num_channels + 1] = data_img[i * num_channels + 1];
                    texture.data[i * num_channels + 2] = data_img[i * num_channels + 2];
                }

                cv::cvtColor(texture, texture, cv::COLOR_BGR2RGBA);
                std::memcpy(inputData, texture.data, texture.total() * 4);
            }
        }
        catch (...) {
            success = false;
        }

        
    }
}