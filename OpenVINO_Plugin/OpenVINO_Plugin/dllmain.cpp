// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"

using namespace InferenceEngine;

// Create a macro to quickly mark a function for export
#define DLLExport __declspec (dllexport)

// Wrap code to prevent name-mangling issues
extern "C" {

	// The width of the source input image
	int img_w;
	// The height of the source input image
	int img_h;
	// The input width for the model
	int input_w = 960;
	// The input height for the model
	int input_h = 540;

	// List of available compute devices
	std::vector<std::string> available_devices;

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


	// Returns an unparsed list of available compute devices
	DLLExport int FindAvailableDevices(std::string* device_list) {

		// Get a list of the available compute devices
		//available_devices = ie.GetAvailableDevices();

		available_devices.clear();

		for (auto&& device : ie.GetAvailableDevices()) {
			if (device.find("GNA") != std::string::npos) continue;

			available_devices.push_back(device);
		}

		// Reverse the order of the list
		std::reverse(available_devices.begin(), available_devices.end());

		// Configure the cache directory for GPU compute devices
		ie.SetConfig({ {CONFIG_KEY(CACHE_DIR), "cache"} }, "GPU");

		return available_devices.size();
	}

	DLLExport std::string* GetDeviceName(int index) {
		return &available_devices[index];
	}

	// Manually set the input resolution for the model
	void SetInputDims(int width, int height) {

		img_w = width;
		img_h = height;

		input_w = (int)(8 * std::roundf(img_w / 8));
		input_h = (int)(8 * std::roundf(img_h / 8));

		// Collect the map of input names and shapes from IR
		auto input_shapes = network.getInputShapes();

		// Set new input shapes
		std::string input_name_1;
		InferenceEngine::SizeVector input_shape;
		// Create a tuple for accessing the input dimensions
		std::tie(input_name_1, input_shape) = *input_shapes.begin();
		// Set batch size to the first input dimension
		input_shape[0] = 1;
		// Update the height for the input dimensions
		input_shape[2] = input_h;
		// Update the width for the input dimensions
		input_shape[3] = input_w;
		input_shapes[input_name_1] = input_shape;

		// Perform shape inference with the new input dimensions
		network.reshape(input_shapes);
	}

	// Create an executable network for the target compute device
	std::string* UploadModelToDevice(int deviceNum) {

		// Create executable network
		executable_network = ie.LoadNetwork(network, available_devices[deviceNum]);
		// Create an inference request object
		infer_request = executable_network.CreateInferRequest();

		// Get the name of the input layer
		std::string input_name = network.getInputsInfo().begin()->first;
		// Get a poiner to the input tensor for the model
		minput = as<MemoryBlob>(infer_request.GetBlob(input_name));

		// Get the name of the output layer
		std::string output_name = network.getOutputsInfo().begin()->first;
		// Get a poiner to the ouptut tensor for the model
		moutput = as<MemoryBlob>(infer_request.GetBlob(output_name));

		// Return the name of the current compute device
		return &available_devices[deviceNum];;
	}

	// Set up OpenVINO inference engine
	DLLExport std::string* InitOpenVINO(char* modelPath, int width, int height, int deviceNum) {

		//// Read network file
		network = ie.ReadNetwork(modelPath);

		SetInputDims(width, height);

		return UploadModelToDevice(deviceNum);
	}


	// Perform inference with the provided texture data
	DLLExport void PerformInference(uchar* inputData) {

		// Store the pixel data for the source input image
		cv::Mat texture = cv::Mat(img_h, img_w, CV_8UC4);

		// Assign the inputData to the OpenCV Mat
		texture.data = inputData;
		// Remove the alpha channel
		cv::cvtColor(texture, texture, cv::COLOR_RGBA2RGB);

		// The number of color channels 
		int num_channels = texture.channels();
		// Get the number of pixels in the input image
		int H = minput->getTensorDesc().getDims()[2];
		int W = minput->getTensorDesc().getDims()[3];
		int nPixels = W * H;

		// locked memory holder should be alive all time while access to its buffer happens
		LockedMemory<void> ilmHolder = minput->wmap();

		// Filling input tensor with image data
		float* input_data = ilmHolder.as<float*>();

		// Iterate over each pixel in image
		for (int p = 0; p < nPixels; p++) {
			// Iterate over each color channel for each pixel in image
			for (int ch = 0; ch < num_channels; ++ch) {
				input_data[ch * nPixels + p] = texture.data[p * num_channels + ch];
			}
		}

		// Perform inference
		infer_request.Infer();

		// locked memory holder should be alive all time while access to its buffer happens
		LockedMemory<const void> moutputHolder = moutput->rmap();
		const float* net_pred = moutputHolder.as<const PrecisionTrait<Precision::FP32>::value_type*>();

		// Iterate through each pixel in the model output
		for (size_t p = 0; p < nPixels; p++) {
			// Iterate through each color channel for each pixel in image
			for (size_t ch = 0; ch < num_channels; ++ch) {
				// Get values from the model output
				float pixel_val = static_cast<float>(net_pred[ch * nPixels + p]);

				// Clamp color values to the range [0, 255]
				pixel_val = std::max(pixel_val, (float)0);
				pixel_val = std::min(pixel_val, (float)255);

				// Copy the processed output to the OpenCV Mat
				texture.data[p * num_channels + ch] = pixel_val;
			}
		}

		// Add alpha channel
		cv::cvtColor(texture, texture, cv::COLOR_RGB2RGBA);
		// Copy values form the OpenCV Mat back to inputData
		std::memcpy(inputData, texture.data, texture.total() * texture.channels());
	}

	DLLExport void FreeResources() {
		available_devices.clear();
	}
}