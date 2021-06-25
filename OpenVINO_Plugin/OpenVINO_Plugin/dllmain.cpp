// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"

using namespace InferenceEngine;

// Create a macro to quickly mark a function for export
#define DLLExport __declspec (dllexport)

// Wrap code to prevent name-mangling issues
extern "C" {

    // List of available compute devices
    std::vector<std::string> availableDevices;
    // An unparsed list of available compute devices
    std::string allDevices = "";
    // The name of the input layer of Neural Network "input.1"
    std::string firstInputName;
    // The name of the output layer of Neural Network "140"
    std::string firstOutputName;

    // Stores the pixel data for model input image and output image
    cv::Mat texture;

    // Inference engine instance
    Core ie;
    // Contains all the information about the Neural Network topology and related constant values for the model
    CNNNetwork network;
    // Provides an interface for an executable network on the compute device
    ExecutableNetwork executable_network;
    // Provides an interface for an asynchronous inference request
    InferRequest infer_request;

    // A poiner to the input tensor for the model
    MemoryBlob::Ptr minput;
    // A poiner to the output tensor for the model
    MemoryBlob::CPtr moutput;

    // The number of color channels 
    size_t num_channels;
    // The number of pixels in the input image
    size_t nPixels;

    // A vector for processing the raw model output
    std::vector<float> data_img;

    // Returns an unparsed list of available compute devices
    DLLExport const std::string* GetAvailableDevices() {
        // Add all available compute devices to a single string
        for (auto&& device : availableDevices) {
            allDevices += device;
            allDevices += ((device == availableDevices[availableDevices.size() - 1]) ? "" : ",");
        }
        return &allDevices;
    }

    // Configure the cache directory for GPU compute devices
    void SetDeviceCache() {
        std::regex e("(GPU)(.*)");
        // Iterate through the available compute devices
        for (auto&& device : availableDevices) {
            // Only configure the cache directory for GPUs
            if (std::regex_match(device, e)) {
                ie.SetConfig({ {CONFIG_KEY(CACHE_DIR), "cache"} }, device);
            }
        }
    }

    // Get the names of the input and output layers and set the precision
    DLLExport void PrepareBlobs() {
        // Get information about the network input
        InputsDataMap inputInfo(network.getInputsInfo());
        firstInputName = inputInfo.begin()->first;
        inputInfo.begin()->second->setPrecision(Precision::U8);

        // Get information about the network output
        OutputsDataMap outputInfo(network.getOutputsInfo());
        // Get the name of the output layer
        firstOutputName = outputInfo.begin()->first;
        // Set the output precision
        outputInfo.begin()->second->setPrecision(Precision::FP32);
    }

    // Set up OpenVINO inference engine
    DLLExport void InitializeOpenVINO(char* modelPath) {
        
        // Read network file
        network = ie.ReadNetwork(modelPath);
        // Set batch size to one image
        network.setBatchSize(1);
        // Get the output name and set the output precision
        PrepareBlobs();
        // Get a list of the available compute devices
        availableDevices = ie.GetAvailableDevices();
        // Reverse the order of the list
        std::reverse(availableDevices.begin(), availableDevices.end());
        // Specify the cache directory for GPU inference
        SetDeviceCache();
    }

    // Manually set the input resolution for the model
    DLLExport void SetInputDims(int width, int height) {

        // Collect the map of input names and shapes from IR
        auto input_shapes = network.getInputShapes();

        // Set new input shapes
        std::string input_name;
        InferenceEngine::SizeVector input_shape;
        // create a tuple for accessing the input dimensions
        std::tie(input_name, input_shape) = *input_shapes.begin();
        // set batch size to the first input dimension
        input_shape[0] = 1;
        // changes input height to the image one
        input_shape[2] = height;
        // changes input width to the image one
        input_shape[3] = width;
        input_shapes[input_name] = input_shape;

        // Call reshape
        // Perform shape inference with the new input dimensions
        network.reshape(input_shapes);
        // Initialize the texture variable with the new dimensions
        texture = cv::Mat(height, width, CV_8UC4);
    }

    // Create an executable network for the target compute device
    DLLExport std::string* UploadModelToDevice(int deviceNum) {

        // Create executable network
        executable_network = ie.LoadNetwork(network, availableDevices[deviceNum]);
        // Create an inference request object
        infer_request = executable_network.CreateInferRequest();
        
        // Get a poiner to the input tensor for the model
        minput = as<MemoryBlob>(infer_request.GetBlob(firstInputName));
        // Get a poiner to the ouptut tensor for the model
        moutput = as<MemoryBlob>(infer_request.GetBlob(firstOutputName));

        // Get the number of color channels 
        num_channels = minput->getTensorDesc().getDims()[1];
        // Get the number of pixels in the input image
        size_t H = minput->getTensorDesc().getDims()[2];
        size_t W = minput->getTensorDesc().getDims()[3];
        nPixels = W * H;

        // Filling input tensor with image data
        data_img = std::vector<float>(nPixels * num_channels);
        
        // Return the name of the current compute device
        return &availableDevices[deviceNum];;
    }
       
    // Perform inference with the provided texture data
    DLLExport void PerformInference(uchar* inputData) {

        // Assign the inputData to the OpenCV Mat
        texture.data = inputData;
        // Remove the alpha channel
        cv::cvtColor(texture, texture, cv::COLOR_RGBA2RGB);
        
        // locked memory holder should be alive all time while access to its buffer happens
        LockedMemory<void> ilmHolder = minput->wmap();

        // Filling input tensor with image data
        auto input_data = ilmHolder.as<PrecisionTrait<Precision::U8>::value_type*>();

        // Iterate over each pixel in image
        for (size_t p = 0; p < nPixels; p++) {
            // Iterate over each color channel for each pixel in image
            for (size_t ch = 0; ch < num_channels; ++ch) {
                input_data[ch * nPixels + p] = texture.data[p * num_channels + ch];
            }
        }

        // Perform inference
        infer_request.Infer();

        // locked memory holder should be alive all time while access to its buffer happens
        LockedMemory<const void> lmoHolder = moutput->rmap();
        const auto output_data = lmoHolder.as<const PrecisionTrait<Precision::FP32>::value_type*>();

        // Iterate through each pixel in the model output
        for (size_t p = 0; p < nPixels; p++) {
            // Iterate through each color channel for each pixel in image
            for (size_t ch = 0; ch < num_channels; ++ch) {
                // Get values from the model output
                data_img[p * num_channels + ch] = static_cast<float>(output_data[ch * nPixels + p]);

                // Clamp color values to the range [0, 255]
                if (data_img[p * num_channels + ch] < 0) data_img[p * num_channels + ch] = 0;
                if (data_img[p * num_channels + ch] > 255) data_img[p * num_channels + ch] = 255;

                // Copy the processed output to the OpenCV Mat
                texture.data[p * num_channels + ch] = data_img[p * num_channels + ch];
            }
        }

        // Add alpha channel
        cv::cvtColor(texture, texture, cv::COLOR_RGB2RGBA);
        // Copy values form the OpenCV Mat back to inputData
        std::memcpy(inputData, texture.data, texture.total() * texture.channels());
    }
}