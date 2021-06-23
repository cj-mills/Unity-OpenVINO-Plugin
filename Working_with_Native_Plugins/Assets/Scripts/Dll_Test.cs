using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Runtime.InteropServices;
using UnityEngine.UI;
using Unity.Barracuda;
using UnityEngine.Rendering;

public class Dll_Test : MonoBehaviour
{
    [Tooltip("Performs the preprocessing and postprocessing steps")]
    public ComputeShader styleTransferShader;

    [Tooltip("Toggle between OpenVINO and Barracuda")]
    public TMPro.TMP_Dropdown inferenceEngineDropdown;

    [Tooltip("Switch between the available compute devices for OpenVINO")]
    public TMPro.TMP_Dropdown deviceDropdown;

    [Tooltip("Switch between the available OpenVINO models")]
    public TMPro.TMP_Dropdown modelDropdown;

    [Tooltip("Turn stylization on and off")]
    public Toggle stylize;

    [Tooltip("Turn AsyncGPUReadback on and off")]
    public Toggle useAsync;

    [Tooltip("Text box for the input width")]
    public TMPro.TMP_InputField widthText;
    [Tooltip("Text box for the input height")]
    public TMPro.TMP_InputField heightText;

    [Tooltip("The model asset file that will be used when performing inference")]
    public NNModel[] modelAssets;

    [Tooltip("The backend used when performing inference")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;

    [Tooltip("Text area to display console output")]
    public Text consoleText;

    // Name of the DLL file
    const string dll = "OpenVINO_Plugin";

    [DllImport(dll)]
    private static extern IntPtr GetAvailableDevices();

    [DllImport(dll)]
    private static extern void InitializeOpenVINO(string modelPath);

    [DllImport(dll)]
    private static extern bool SetInputDims(int width, int height);

    [DllImport(dll)]
    private static extern void PrepareBlobs();

    [DllImport(dll)]
    private static extern IntPtr UploadModelToDevice(int deviceNum = 0);

    [DllImport(dll)]
    private static extern void PerformInference(IntPtr inputData);

    // The compiled model used for performing inference
    private Model[] m_RuntimeModels;

    // The interface used to execute the neural network
    private IWorker[] engines;

    // Contains the resized input texture
    private RenderTexture tempTex;
    // Contains the input texture that will be sent to the OpenVINO inference engine
    private Texture2D inputTex;
    // Stores the raw pixel data for inputTex
    private byte[] inputData;

    // Input image width
    private int width = 960;
    // Input image height
    private int height = 540;

    // Unparsed list of available compute devices for OpenVINO
    private string openvinoDevices;
    // Current compute device for OpenVINO
    private string currentDevice;
    // Parsed list of compute devices for OpenVINO
    private List<string> deviceList = new List<string>();

    // File paths for the OpenVINO IR models
    private List<string> openVINOPaths = new List<string>();
    // Names of the OpenVINO IR model
    private List<string> openvinoModels = new List<string>();
    // Names of the ONNX models
    private List<string> onnxModels = new List<string>();

    // Start is called before the first frame update
    void Start()
    {
        string processorType = SystemInfo.processorType.ToString();
        Debug.Log($"Processor Type: {processorType}");
        string graphicsDeviceName = SystemInfo.graphicsDeviceName.ToString();
        Debug.Log($"Graphics Device Name: {graphicsDeviceName}");

        string[] openVINOFiles = System.IO.Directory.GetFiles("models");
        Debug.Log("Available OpenVINO Models");
        foreach (string file in openVINOFiles)
        {
            if (file.EndsWith(".xml"))
            {
                Debug.Log(file);
                openVINOPaths.Add(file);
                string modelName = file.Split('\\')[1];
                openvinoModels.Add(modelName.Substring(0, modelName.Length-4));
            }
        }

        // Remove default dropdown options
        modelDropdown.ClearOptions();
        // Add OpenVINO models to menu
        modelDropdown.AddOptions(openvinoModels);
        // Select the first option in the dropdown
        modelDropdown.SetValueWithoutNotify(0);

        // Remove default dropdown options
        deviceDropdown.ClearOptions();

        // Check if either the CPU of GPU is made by Intel
        if (processorType.Contains("Intel") || graphicsDeviceName.Contains("Intel"))
        {
            Debug.Log("Initializing OpenVINO");
            InitializeOpenVINO(openVINOPaths[0]);
            Debug.Log($"Setting Input Dims to W: {width} x H: {height}");
            SetInputDims(width, height);
            Debug.Log("Uploading IR Model to Compute Device");
            currentDevice = Marshal.PtrToStringAnsi(UploadModelToDevice());
            Debug.Log($"OpenVINO using: {currentDevice}");

            // Get an unparsed list of available 
            openvinoDevices = Marshal.PtrToStringAnsi(GetAvailableDevices());
            
            Debug.Log($"Available Devices:");
            // Parse list of available compute devices
            foreach (string device in openvinoDevices.Split(','))
            {
                // Add device name to list
                deviceList.Add(device);
                Debug.Log(device);
            }

            // Add OpenVINO compute devices to dropdown
            deviceDropdown.AddOptions(deviceList);
            // Set the value for the dropdown to the current compute device
            deviceDropdown.SetValueWithoutNotify(deviceList.IndexOf(currentDevice));
        }
        else
        {
            // Use Barracuda inference engine if not on Intel hardware
            inferenceEngineDropdown.SetValueWithoutNotify(1);
        }
        
        // Initialize textures with default input resolution
        tempTex = RenderTexture.GetTemporary(width, height, 24, RenderTextureFormat.ARGB32);
        inputTex = new Texture2D(width, height, TextureFormat.RGBA32, false);

        // Initialize list of Barracuda models
        m_RuntimeModels = new Model[modelAssets.Length];
        // Initialize list of Barracuda inference engines
        engines = new IWorker[modelAssets.Length];
        for(int i = 0; i < modelAssets.Length; i++)
        {
            // Add names of available ONNX models to list
            onnxModels.Add(modelAssets[i].name);
            Debug.Log($"ModelAsset Name: {modelAssets[i].name}");
            // Compile the model asset into an object oriented representation
            m_RuntimeModels[i] = ModelLoader.Load(modelAssets[i]);
            // Create a worker that will execute the model with the selected backend
            engines[i] = WorkerFactory.CreateWorker(workerType, m_RuntimeModels[i]);
        }
    }

    // Called when the MonoBehaviour will be destroyed
    private void OnDestroy()
    {
        Destroy(inputTex);
        Destroy(tempTex);
        RenderTexture.ReleaseTemporary(tempTex);

        // Release the resources allocated for the inference engines
        foreach (IWorker engine in engines)
        {
            engine.Dispose();
        }
        
        Application.logMessageReceived -= Log;
    }


    //Update is called once per frame
    private void Update()
    {

    }

    /// <summary>
    /// Perform a flip operation of the GPU
    /// </summary>
    /// <param name="image">The image to be flipped</param>
    /// <param name="tempTex">Stores the flipped image</param>
    /// <param name="functionName">The name of the function to execute in the compute shader</param>
    private void FlipImage(RenderTexture image, string functionName)
    {
        // Specify the number of threads on the GPU
        int numthreads = 4;
        // Get the index for the PreprocessResNet function in the ComputeShader
        int kernelHandle = styleTransferShader.FindKernel(functionName);

        /// Allocate a temporary RenderTexture
        RenderTexture result = RenderTexture.GetTemporary(image.width, image.height, 24, image.format);
        // Enable random write access
        result.enableRandomWrite = true;
        // Create the RenderTexture
        result.Create();

        // Set the value for the Result variable in the ComputeShader
        styleTransferShader.SetTexture(kernelHandle, "Result", result);
        // Set the value for the InputImage variable in the ComputeShader
        styleTransferShader.SetTexture(kernelHandle, "InputImage", image);
        // Set the value for the height variable in the ComputeShader
        styleTransferShader.SetInt("height", image.height);
        // Set the value for the width variable in the ComputeShader
        styleTransferShader.SetInt("width", image.width);

        // Execute the ComputeShader
        styleTransferShader.Dispatch(kernelHandle, image.width / numthreads, image.height / numthreads, 1);

        // Copy the flipped image to tempTex
        Graphics.Blit(result, image);

        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(result);
    }

    /// <summary>
    /// Process the provided image using the specified function on the GPU
    /// </summary>
    /// <param name="image"></param>
    /// <param name="functionName"></param>
    /// <returns>The processed image</returns>
    private void ProcessImage(RenderTexture image, string functionName)
    {
        // Specify the number of threads on the GPU
        int numthreads = 4;
        // Get the index for the specified function in the ComputeShader
        int kernelHandle = styleTransferShader.FindKernel(functionName);
        // Define a temporary HDR RenderTexture
        RenderTexture result = RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
        // Enable random write access
        result.enableRandomWrite = true;
        // Create the HDR RenderTexture
        result.Create();

        // Set the value for the Result variable in the ComputeShader
        styleTransferShader.SetTexture(kernelHandle, "Result", result);
        // Set the value for the InputImage variable in the ComputeShader
        styleTransferShader.SetTexture(kernelHandle, "InputImage", image);

        // Execute the ComputeShader
        styleTransferShader.Dispatch(kernelHandle, result.width / numthreads, result.height / numthreads, 1);

        // Copy the result into the source RenderTexture
        Graphics.Blit(result, image);

        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(result);
    }

    /// <summary>
    /// Stylize the provided image
    /// </summary>
    /// <param name="src"></param>
    /// <returns></returns>
    private void StylizeImage(RenderTexture src)
    {
        // Create a new RenderTexture variable
        RenderTexture rTex;

        // Check if the target display is larger than the height
        // and make sure the height is at least 4
        if (src.height > height && height >= 4)
        {
            // Adjust the target image dimensions to be multiples of 4
            int targetHeight = height - (height % 4);
            int targetWidth = width - (width % 4);

            // Assign a temporary RenderTexture with the new dimensions
            rTex = RenderTexture.GetTemporary(targetWidth, targetHeight, 24, src.format);
        }
        else
        {
            // Assign a temporary RenderTexture with the src dimensions
            rTex = RenderTexture.GetTemporary(src.width, src.height, 24, src.format);
        }

        // Copy the src RenderTexture to the new rTex RenderTexture
        Graphics.Blit(src, rTex);

        // Apply preprocessing steps
        ProcessImage(rTex, "ProcessInput");

        // Create a Tensor of shape [1, rTex.height, rTex.width, 3]
        Tensor input = new Tensor(rTex, channels: 3);

        // Execute neural network with the provided input
        //engine.Execute(input);
        engines[modelDropdown.value].Execute(input);

        // Get the raw model output
        Tensor prediction = engines[modelDropdown.value].PeekOutput();

        // Release GPU resources allocated for the Tensor
        input.Dispose();

        // Make sure rTex is not the active RenderTexture
        RenderTexture.active = null;
        // Copy the model output to rTex
        prediction.ToRenderTexture(rTex);
        // Release GPU resources allocated for the Tensor
        prediction.Dispose();

        // Apply postprocessing steps
        ProcessImage(rTex, "ProcessOutput");
        // Copy rTex into src
        Graphics.Blit(rTex, src);

        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(rTex);
    }


    /// <summary>
    /// Called when an inference engine option is selected from dropdown
    /// </summary>
    public void SetInferenceEngine()
    {
        // Get the list of models for the selected inference engine
        List<string> currentList = inferenceEngineDropdown.value == 0 ? openvinoModels : onnxModels;

        // Remove current dropdown options
        modelDropdown.ClearOptions();
        // Add current inference engine models to menu
        modelDropdown.AddOptions(currentList);
        // Select the first option in the dropdown
        modelDropdown.SetValueWithoutNotify(0);
        
        // Get index for current model selection
        int index = modelDropdown.value;
        // Get the name for the previously selected inference engine model
        string previousModel = inferenceEngineDropdown.value == 0 ? onnxModels[index] : openvinoModels[index];
        // Check if the model for the previous inference engine is available for the current engine
        if (currentList.Contains(previousModel))
        {
            modelDropdown.SetValueWithoutNotify(openvinoModels.IndexOf(previousModel));
        }
    }

    /// <summary>
    /// Called when a model option is selected from the dropdown
    /// </summary>
    public void UpdateModel()
    {
        Debug.Log($"Selecte Model: {modelDropdown.value}");
        // Initialize the selected OpenVINO model
        if (inferenceEngineDropdown.value == 0)
        {
            InitializeOpenVINO(openVINOPaths[modelDropdown.value]);
            UpdateInputDims();
        }
    }

    /// <summary>
    /// Called when the input resolution is updated
    /// </summary>
    public void UpdateInputDims()
    {
        // Get the integer value from the width input
        int.TryParse(widthText.text, out width);
        // Get the integer value from the height input
        int.TryParse(heightText.text, out height);

        // Update tempTex with the new dimensions
        tempTex = RenderTexture.GetTemporary(width, height, 24, RenderTextureFormat.ARGB32);
        // Update inputTex with the new dimensions
        inputTex = new Texture2D(width, height, TextureFormat.RGBA32, false);
        

        // Set input resolution width x height for the OpenVINO model
        Debug.Log($"Setting Input Dims to W: {width} x H: {height}");
        SetInputDims(width, height);
        SetDevice();

        // Preparing Output Blobs
        PrepareBlobs();
    }

    /// <summary>
    /// Called when a compute device is selected from dropdown
    /// </summary>
    public void SetDevice()
    {
        // Uploading model to device
        currentDevice = Marshal.PtrToStringAnsi(UploadModelToDevice(deviceDropdown.value));
    }

    /// <summary>
    /// Pin memory for the input data and send it to OpenVINO for inference
    /// </summary>
    /// <param name="inputData"></param>
    public unsafe void UpdateTexture(byte[] inputData)
    {
        //Pin Memory
        fixed (byte* p = inputData)
        {
            // Perform inference with OpenVINO
            PerformInference((IntPtr)p);
        }
    }

    /// <summary>
    /// Called once AsyncGPUReadback has been completed
    /// </summary>
    /// <param name="request"></param>
    void OnCompleteReadback(AsyncGPUReadbackRequest request)
    {
        if (request.hasError)
        {
            Debug.Log("GPU readback error detected.");
            return;
        }

        // Fill Texture2D with raw data from the AsyncGPUReadbackRequest
        inputTex.LoadRawTextureData(request.GetData<uint>());
        // Apply changes to Textur2D
        inputTex.Apply();
    }


    /// <summary>
    /// OnRenderImage is called after the Camera had finished rendering 
    /// </summary>
    /// <param name="src">Input from the Camera</param>
    /// <param name="dest">The texture for the targer display</param>
    void OnRenderImage(RenderTexture src, RenderTexture dest)
    {
        // Only stylize current framne if Stylize toggle is on
        if (stylize.isOn)
        {
            // Check which inference engine to use
            if (inferenceEngineDropdown.value == 0)
            {
                // Copy current frame to smaller temporary texture
                Graphics.Blit(src, tempTex);
                // Flip image before sending to DLL
                FlipImage(tempTex, "FlipXAxis");

                if (useAsync.isOn)
                {
                    AsyncGPUReadback.Request(tempTex, 0, TextureFormat.RGBA32, OnCompleteReadback);
                }
                else
                {
                    RenderTexture.active = tempTex;
                    inputTex.ReadPixels(new Rect(0, 0, tempTex.width, tempTex.height), 0, 0);
                    inputTex.Apply();
                }

                // Get raw data from Texture2D
                inputData = inputTex.GetRawTextureData();
                // Send reference to inputData to DLL
                UpdateTexture(inputData);
                // Load the new image data from the DLL to the texture
                inputTex.LoadRawTextureData(inputData);
                // Apply the changes to the texture
                inputTex.Apply();
                // Copy output image to temporary texture
                Graphics.Blit(inputTex, tempTex);
                // Flip output image from DLL
                FlipImage(tempTex, "FlipXAxis");
                // Copy the temporary texture to the source resolution texture
                Graphics.Blit(tempTex, src);
            }
            else
            {
                StylizeImage(src);
            }
        }
        
        Graphics.Blit(src, dest);
    }


    // Called when the object becomes enabled and active
    void OnEnable()
    {
        Application.logMessageReceived += Log;
    }

    /// <summary>
    /// Updates onscreen console text
    /// </summary>
    /// <param name="logString"></param>
    /// <param name="stackTrace"></param>
    /// <param name="type"></param>
    public void Log(string logString, string stackTrace, LogType type)
    {
        consoleText.text = consoleText.text + "\n " + logString;
    }

    /// <summary>
    /// Called when the Quit button is clicked.
    /// </summary>
    public void Quit()
    {
        // Causes the application to exit
        Application.Quit();
    }
}
