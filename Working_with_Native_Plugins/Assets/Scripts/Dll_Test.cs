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


    const string dll = "OpenVINO_Plugin";

    [DllImport(dll)]
    private static extern IntPtr GetAvailableDevices();

    [DllImport(dll)]
    private static extern void InitializeOpenVINO(string modelPath);

    [DllImport(dll)]
    private static extern bool SetInputDims(int width, int height);

    [DllImport(dll)]
    private static extern void PrepareOutputBlobs();

    [DllImport(dll)]
    private static extern IntPtr UploadModelToDevice(int deviceNum = 0);

    [DllImport(dll)]
    private static extern void PerformInference(IntPtr inputData, int width, int height);

    // The compiled model used for performing inference
    private Model[] m_RuntimeModels;

    // The interface used to execute the neural network
    private IWorker[] engines;

    // Contains the resized input texture
    private RenderTexture tempTex;
    // Contains the input texture that will be sent to the OpenVINO inference engine
    private Texture2D inputTex;
    
    // Input image width
    private int width = 960;
    // Input image height
    private int height = 540;

    // Unparsed list of available compute devices for OpenVINO
    private string openvinoDevices;
    // Current compute device for OpenVINO
    private string currentDevice;
    // Parsed list of compute devices for OpenVINO
    private List<string> deviceList;


    private List<string> openVINOPaths;

    private List<string> modelPaths;

    private List<string> openvinoModels;
    private List<string> onnxModels;

    private List<string> models;

    // Start is called before the first frame update
    void Start()
    {
        string processorType = SystemInfo.processorType.ToString();
        Debug.Log($"Processor Type: {processorType}");
        string graphicsDeviceName = SystemInfo.graphicsDeviceName.ToString();
        Debug.Log($"Graphics Device Name: {graphicsDeviceName}");

        string[] openVINOFiles = System.IO.Directory.GetFiles("models");
        openVINOPaths = new List<string>();
        openvinoModels = new List<string>();
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
        modelDropdown.AddOptions(openvinoModels);
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

            openvinoDevices = Marshal.PtrToStringAnsi(GetAvailableDevices());
            deviceList = new List<string>();

            Debug.Log($"Available Devices:");
            foreach (string device in openvinoDevices.Split(','))
            {
                deviceList.Add(device);
                Debug.Log(device);
            }

            deviceDropdown.AddOptions(deviceList);
            deviceDropdown.SetValueWithoutNotify(deviceList.IndexOf(currentDevice));
        }
        else
        {
            inferenceEngineDropdown.SetValueWithoutNotify(1);
        }
        
        // Set the initial input resolution
        tempTex = RenderTexture.GetTemporary(width, height, 24, RenderTextureFormat.ARGB32);
        //inputTex = new Texture2D(width, height, TextureFormat.RGBA32, false);
        inputTex = new Texture2D(width, height, TextureFormat.RGBA32, false);

        onnxModels = new List<string>();
        m_RuntimeModels = new Model[modelAssets.Length];
        engines = new IWorker[modelAssets.Length];
        int index = 0;
        foreach(NNModel m in modelAssets)
        {
            onnxModels.Add(m.name);
            Debug.Log($"ModelAsset Name: {m.name}");
            // Compile the model asset into an object oriented representation
            m_RuntimeModels[index] = ModelLoader.Load(m);
            // Create a worker that will execute the model with the selected backend
            engines[index] = WorkerFactory.CreateWorker(workerType, m_RuntimeModels[index]);
            index++;
        }
    }

    private void OnDestroy()
    {
        Destroy(inputTex);
        Destroy(tempTex);
        RenderTexture.ReleaseTemporary(tempTex);

        // Release the resources allocated for the inference engine
        foreach (IWorker engine in engines)
        {
            engine.Dispose();
        }
        //engine.Dispose();
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
        // and make sure the height is at least 8
        if (src.height > height && height >= 4)
        {
            // Calculate the scale value for reducing the size of the input image
            float scale = src.height / height;
            // Calcualte the new image width
            int targetWidth = (int)(src.width / scale);

            // Adjust the target image dimensions to be multiples of 4
            height -= (height % 4);
            targetWidth -= (targetWidth % 4);

            // Assign a temporary RenderTexture with the new dimensions
            rTex = RenderTexture.GetTemporary(targetWidth, height, 24, src.format);
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
        //Tensor prediction = engine.PeekOutput();
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


    public void SetInferenceEngine()
    {
        //inferenceEngineDropdown.
        Debug.Log("Here");
        string currentModel;
        modelDropdown.ClearOptions();

        if (inferenceEngineDropdown.value == 0)
        {
            currentModel = onnxModels[modelDropdown.value];
            // Remove default dropdown options
            //modelDropdown.ClearOptions();
            modelDropdown.AddOptions(openvinoModels);
            modelDropdown.SetValueWithoutNotify(0);

            if (openvinoModels.Contains(currentModel))
            {
                modelDropdown.SetValueWithoutNotify(openvinoModels.IndexOf(currentModel));
            }
        }
        else
        {
            currentModel = openvinoModels[modelDropdown.value];
            //Debug.Log("Here");

            // Remove default dropdown options
            //modelDropdown.ClearOptions();
            foreach (string model in onnxModels)
            {
                Debug.Log(model);
            }
            modelDropdown.AddOptions(onnxModels);
            modelDropdown.SetValueWithoutNotify(0);

            if (onnxModels.Contains(currentModel))
            {
                modelDropdown.SetValueWithoutNotify(onnxModels.IndexOf(currentModel));
            }
        }
    }


    public void UpdateModel()
    {
        Debug.Log(modelDropdown.value);
        if (inferenceEngineDropdown.value == 0)
        {
            InitializeOpenVINO(openVINOPaths[modelDropdown.value]);
            UpdateInputDims();
        }
    }


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
        PrepareOutputBlobs();
    }

    public void SetDevice()
    {
        // Uploading model to device
        currentDevice = Marshal.PtrToStringAnsi(UploadModelToDevice(deviceDropdown.value));
    }

    public unsafe void UpdateTexture(Color32[] outArray, int width, int height)
    {
        //Pin Memory
        fixed (Color32* p = outArray)
        {
            PerformInference((IntPtr)p, width, height);
        }
    }

            
    void OnCompleteReadback(AsyncGPUReadbackRequest request)
    {
        if (request.hasError)
        {
            Debug.Log("GPU readback error detected.");
            return;
        }

        inputTex.LoadRawTextureData(request.GetData<uint>());
        inputTex.Apply();
    }


    /// <summary>
    /// OnRenderImage is called after the Camera had finished rendering 
    /// </summary>
    /// <param name="src">Input from the Camera</param>
    /// <param name="dest">The texture for the targer display</param>
    void OnRenderImage(RenderTexture src, RenderTexture dest)
    {
        if (stylize.isOn)
        {
            if (inferenceEngineDropdown.value == 0)
            {
                Graphics.Blit(src, tempTex);
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

                Color32[] inputPixel32 = inputTex.GetPixels32();
                UpdateTexture(inputPixel32, inputTex.width, inputTex.height);
                inputTex.SetPixels32(inputPixel32);
                inputTex.Apply();

                Graphics.Blit(inputTex, tempTex);

                FlipImage(tempTex, "FlipXAxis");
                Graphics.Blit(tempTex, src);
            }
            else
            {
                StylizeImage(src);
            }
        }

        Graphics.Blit(src, dest);
    }


    void OnEnable()
    {
        Application.logMessageReceived += Log;
    }

    public void Log(string logString, string stackTrace, LogType type)
    {
        consoleText.text = consoleText.text + "\n " + logString;
    }

    public void Quit()
    {
        Application.Quit();
    }
}
