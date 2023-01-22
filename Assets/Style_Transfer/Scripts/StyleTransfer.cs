using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Runtime.InteropServices;
using UnityEngine.UI;
using Unity.Barracuda;
using UnityEngine.Rendering;
using System.IO;

public class StyleTransfer : MonoBehaviour
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


    // Name of the DLL file
    const string dll = "OpenVINO_Plugin";

    [DllImport(dll)]
    private static extern int FindAvailableDevices();

    [DllImport(dll)]
    private static extern IntPtr GetDeviceName(int index);

    [DllImport(dll)]
    private static extern IntPtr InitOpenVINO(string model, int width, int height, int device);

    [DllImport(dll)]
    private static extern void PerformInference(IntPtr inputData);

    [DllImport(dll)]
    private static extern void FreeResources();


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



    public void Awake()
    {
        #if UNITY_EDITOR_WIN
                return;
        #else

        Debug.Log("Checking for plugins.xml file");

            string sourcePath = $"{Application.streamingAssetsPath}/plugins.xml";
            string targetPath = $"{Application.dataPath}/Plugins/x86_64/plugins.xml";

            if (File.Exists(targetPath))
            {
                Debug.Log("plugins.xml already in folder");
            }
            else
            {
                Debug.Log("Moving plugins.xml file from StreamingAssets to Plugins folder.");
                File.Move(sourcePath, targetPath);
            }

        #endif
    }

    /// <summary>
    /// Initialize the options for the dropdown menus
    /// </summary>
    private void InitializeDropdowns()
    {
        // Remove default dropdown options
        deviceDropdown.ClearOptions();
        // Add OpenVINO compute devices to dropdown
        deviceDropdown.AddOptions(deviceList);
        // Set the value for the dropdown to the current compute device
        deviceDropdown.SetValueWithoutNotify(deviceList.IndexOf(currentDevice));

        // Remove default dropdown options
        modelDropdown.ClearOptions();
        // Add OpenVINO models to menu
        modelDropdown.AddOptions(openvinoModels);
        // Select the first option in the dropdown
        modelDropdown.SetValueWithoutNotify(0);
    }


    /// <summary>
    /// Called when a model option is selected from the dropdown
    /// </summary>
    public void InitializeOpenVINO()
    {
        // Only initialize OpenVINO when performing inference
        if (stylize.isOn == false) return;

        Debug.Log("Initializing OpenVINO");
        Debug.Log($"Selected Model: {openvinoModels[modelDropdown.value]}");
        Debug.Log($"Selected Model Path: {openVINOPaths[modelDropdown.value]}");
        Debug.Log($"Setting Input Dims to W: {width} x H: {height}");
        Debug.Log("Uploading IR Model to Compute Device");

        // Set up the neural network for the OpenVINO inference engine
        currentDevice = Marshal.PtrToStringAnsi(InitOpenVINO(
            openVINOPaths[modelDropdown.value],
            inputTex.width,
            inputTex.height,
            deviceDropdown.value));

        Debug.Log($"OpenVINO using: {currentDevice}");
    }


    /// <summary>
    /// Get the list of available OpenVINO models
    /// </summary>
    private void GetOpenVINOModels()
    {
        // Get the model files in each subdirectory
        List<string> openVINOFiles = new List<string>();
        openVINOFiles.AddRange(System.IO.Directory.GetFiles(Application.streamingAssetsPath + "/models"));

        // Get the paths for the .xml files for each model
        Debug.Log("Available OpenVINO Models:");
        foreach (string file in openVINOFiles)
        {
            if (file.EndsWith(".xml"))
            {
                openVINOPaths.Add(file);
                string modelName = file.Split('\\')[1];
                openvinoModels.Add(modelName.Substring(0, modelName.Length));

                Debug.Log($"Model Name: {modelName}");
                Debug.Log($"File Path: {file}");
            }
        }
        Debug.Log("");
    }


    // Start is called before the first frame update
    void Start()
    {
        string processorType = SystemInfo.processorType.ToString();
        Debug.Log($"Processor Type: {processorType}");
        string graphicsDeviceName = SystemInfo.graphicsDeviceName.ToString();
        Debug.Log($"Graphics Device Name: {graphicsDeviceName}");

        // Check if either the CPU of GPU is made by Intel
        if (processorType.Contains("Intel") || graphicsDeviceName.Contains("Intel"))
        {
            // Get the list of available models
            GetOpenVINOModels();

            Debug.Log("Available Devices:");
            int deviceCount = FindAvailableDevices();
            for (int i = 0; i < deviceCount; i++)
            {
                deviceList.Add(Marshal.PtrToStringAnsi(GetDeviceName(i)));
                Debug.Log(deviceList[i]);
            }
        }
        else
        {
            inferenceEngineDropdown.value = 1;
            Debug.Log("No Intel hardware detected using Barracuda");
        }

        // Initialize textures with default input resolution
        width = (int)(8 * (int)(width / 8));
        height = (int)(8 * (int)(height / 8));
        widthText.text = $"{width}";
        heightText.text = $"{height}";
        tempTex = RenderTexture.GetTemporary(width, height, 24, RenderTextureFormat.ARGB32);
        inputTex = new Texture2D(width, height, TextureFormat.RGBA32, false);

        // Initialize the dropdown menus
        InitializeDropdowns();
        // Set up the neural network for the OpenVINO inference engine
        InitializeOpenVINO();

        // Initialize list of Barracuda models
        m_RuntimeModels = new Model[modelAssets.Length];
        // Initialize list of Barracuda inference engines
        engines = new IWorker[modelAssets.Length];
        for (int i = 0; i < modelAssets.Length; i++)
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
        
        // Set the value for the height variable in the ComputeShader
        styleTransferShader.SetInt("height", image.height);
        // Set the value for the width variable in the ComputeShader
        styleTransferShader.SetInt("width", image.width);
        styleTransferShader.Dispatch(kernelHandle, result.width / numthreads, result.height / numthreads, 1);
        styleTransferShader.SetTexture(kernelHandle, "BaseImage", result);
        // Debug.Log(functionName);
        /*if (functionName == "Mix")
        {
            int Mixer = styleTransferShader.FindKernel("Mix");
            // Set the value for the Result variable in the ComputeShader
            styleTransferShader.SetTexture(kernelHandle, "Result", result);
            // Set the value for the InputImage variable in the ComputeShader
            styleTransferShader.SetTexture(kernelHandle, "InputImage", result);
            styleTransferShader.SetTexture(kernelHandle, "BaseImage",image );
            // Set the value for the height variable in the ComputeShader
            styleTransferShader.SetInt("height", image.height);
            // Set the value for the width variable in the ComputeShader
            styleTransferShader.SetInt("width", image.width);
            // Execute the ComputeShader
            styleTransferShader.Dispatch(Mixer, result.width / numthreads, result.height / numthreads, 1);
        }*/
        // Copy the result into the source RenderTexture
        Graphics.Blit(result, image);

        //ProcessImage(image, "Mix");
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
        Tensor prediction = engines[0].PeekOutput();

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
        //ProcessImage(rTex, "Mix");
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
    /// Called when the input resolution is updated
    /// </summary>
    public void UpdateInputDims()
    {
        // Get the integer value from the width input
        int.TryParse(widthText.text, out width);
        // Get the integer value from the height input
        int.TryParse(heightText.text, out height);

        width = (int)(8 * (int)(width / 8));
        height = (int)(8 * (int)(height / 8));
        widthText.text = $"{width}";
        heightText.text = $"{height}";

        // Update tempTex with the new dimensions
        tempTex = RenderTexture.GetTemporary(width, height, 24, RenderTextureFormat.ARGB32);
        // Update inputTex with the new dimensions
        inputTex = new Texture2D(width, height, TextureFormat.RGBA32, false);


        // Set input resolution width x height for the OpenVINO model
        Debug.Log($"Setting Input Dims to W: {width} x H: {height}");
        if (inferenceEngineDropdown.value == 0)
        {
            InitializeOpenVINO();
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
            InitializeOpenVINO();
            UpdateInputDims();
        }
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
                ProcessImage(tempTex, "FlipXAxis");

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
                ProcessImage(tempTex, "FlipXAxis");
                //ProcessImage(tempTex, "FlipXAxis");
                // Debug.Log("error time");
                int Mixer = styleTransferShader.FindKernel("Mix");
                RenderTexture resultalt = RenderTexture.GetTemporary(src.width, src.height , 24, RenderTextureFormat.ARGBHalf);
                resultalt.enableRandomWrite = true;
                // Create the HDR RenderTexture
                resultalt.Create();
                RenderTexture tempTexw = RenderTexture.GetTemporary(src.width, src.height , 24, RenderTextureFormat.ARGB32);
                tempTexw.enableRandomWrite = true;
                // Create the HDR RenderTexture

                tempTexw.Create();
                // Set the value for the Result variable in the ComputeShader
                styleTransferShader.SetTexture(Mixer, "Result", resultalt);
                // Set the value for the InputImage variable in the ComputeShader
               // styleTransferShader.SetTexture(Mixer, "InputImage", inputTex);
                styleTransferShader.SetTexture(Mixer, "InputImage", src);
                Graphics.Blit(tempTex, tempTexw);
                //Graphics.Blit(tempTex, tempTexw,new Vector2(0.5f,0.5f),new Vector2(0,0),24,24);
                //tempTexw.Release();
                //tempTex.width = src.width;
                //tempTex.height = src.height;
                styleTransferShader.SetTexture(Mixer, "BaseImage", tempTexw);
                //styleTransferShader.SetTexture(Mixer, "BaseImage", inputTex);

                // Set the value for the height variable in the ComputeShader
                styleTransferShader.SetInt("height", resultalt.height);
                // Set the value for the width variable in the ComputeShader
                styleTransferShader.SetInt("width", resultalt.width);
                // Execute the ComputeShader
                int numthreads = 4;
                styleTransferShader.Dispatch(Mixer, src.width / numthreads, src.height / numthreads, 1);

                //ProcessImage(src, "Mix");
                // Copy the temporary texture to the source resolution texture
                Graphics.Blit(resultalt, src);
                tempTexw.Release();
                tempTex.Release();
                resultalt.Release();
            }
            else
            {
                //Debug.Log("error time");
                StylizeImage(src);
            }
        }

        Graphics.Blit(src, dest);
    }

    private void OnDisable()
    {
        FreeResources();
    }

    // Called when the MonoBehaviour will be destroyed
    private void OnDestroy()
    {
        Destroy(inputTex);
        RenderTexture.ReleaseTemporary(tempTex);

        // Release the resources allocated for the inference engines
        foreach (IWorker engine in engines)
        {
            engine.Dispose();
        }
    }

    /// <summary>
    /// Called when the Quit button is clicked.
    /// </summary>
    public void Quit()
    {
        // Causes the application to exit
        Application.Quit();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
