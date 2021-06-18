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
    private static extern IntPtr UploadModelToDevice(int deviceNum=-1);

    [DllImport(dll)]
    private static extern void CreateInferenceRequest();

    [DllImport(dll)]
    private static extern void PerformInference(IntPtr inputData, int width, int height);

    [Tooltip("Performs the preprocessing and postprocessing steps")]
    public ComputeShader styleTransferShader;

    [Tooltip("Toggle between OpenVINO and Barracuda")]
    public TMPro.TMP_Dropdown inferenceEngineDropdown;

    [Tooltip("Toggle between OpenVINO and Barracuda")]
    public TMPro.TMP_Dropdown deviceDropdown;

    public Toggle stylize;

    public TMPro.TMP_InputField widthText;
    public TMPro.TMP_InputField heightText;

    [Tooltip("The height of the image being fed to the model")]
    public int targetHeight = 540;

    [Tooltip("The model asset file that will be used when performing inference")]
    public NNModel modelAsset;

    [Tooltip("The backend used when performing inference")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;

    public GameObject videoScreen;

    
    // The compiled model used for performing inference
    private Model m_RuntimeModel;

    // The interface used to execute the neural network
    private IWorker engine;

    //public ComputeShader flipShader;

    public RenderTexture videoTexture;

    static string myLog = "";
    private string output;
    private string stack;

    public Text fpsText;
    public float deltaTime;

    private Texture2D inputTex;

    RenderTexture tempTex;

    private int width;
    private int height;

    string openvinoDevices;
    string currentDevice;
    List<string> deviceList;

    // Start is called before the first frame update
    void Start()
    {
        Application.targetFrameRate = 300;
        //Debug.Log(SystemInfo.graphicsDeviceType);
        width = 960;
        height = 540;
        
        targetHeight = height;

        Debug.Log("Reading Network File");
        InitializeOpenVINO("final_fp16.xml");

        Debug.Log("Setting Input Dims");
        SetInputDims(width, height);

        Debug.Log("Uploading Model to GPU");
        currentDevice = Marshal.PtrToStringAnsi(UploadModelToDevice(-1));
        Debug.Log($"OpenVINO using: {currentDevice}");
        Debug.Log("Creating Inference Request");
        CreateInferenceRequest();



        openvinoDevices = Marshal.PtrToStringAnsi(GetAvailableDevices());
        deviceList = new List<string>();

        foreach (string device in openvinoDevices.Split(','))
        {
            deviceList.Add(device);
            Debug.Log(device);
        }

        deviceDropdown.ClearOptions();
        deviceDropdown.AddOptions(deviceList);
        deviceDropdown.SetValueWithoutNotify(deviceList.IndexOf(currentDevice));
       

        //inputTex = new Texture2D(videoTexture.width, videoTexture.height, TextureFormat.RGBA32, false);
        inputTex = new Texture2D(width, height, TextureFormat.RGBA32, false);

        //tempTex = RenderTexture.GetTemporary(videoTexture.width, videoTexture.height, 24, videoTexture.format);
        tempTex = RenderTexture.GetTemporary(width, height, 24, videoTexture.format);

        // Compile the model asset into an object oriented representation
        m_RuntimeModel = ModelLoader.Load(modelAsset);

        // Create a worker that will execute the model with the selected backend
        engine = WorkerFactory.CreateWorker(workerType, m_RuntimeModel);
    }

    private void OnDestroy()
    {
        Destroy(inputTex);
        Destroy(tempTex);
        RenderTexture.ReleaseTemporary(tempTex);

        // Release the resources allocated for the inference engine
        engine.Dispose();
        Application.logMessageReceived -= Log;
    }


    //Update is called once per frame
    private void Update()
    {
        deltaTime += (Time.deltaTime - deltaTime) * 0.1f;
        float fps = 1.0f / deltaTime;
        fpsText.text = $"FPS: {Mathf.Ceil(fps)}";
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

        // Check if the target display is larger than the targetHeight
        // and make sure the targetHeight is at least 8
        if (src.height > targetHeight && targetHeight >= 4)
        {
            // Calculate the scale value for reducing the size of the input image
            float scale = src.height / targetHeight;
            // Calcualte the new image width
            int targetWidth = (int)(src.width / scale);

            // Adjust the target image dimensions to be multiples of 4
            targetHeight -= (targetHeight % 4);
            targetWidth -= (targetWidth % 4);

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
        engine.Execute(input);


        // Get the raw model output
        Tensor prediction = engine.PeekOutput();
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


    public void UpdateInputDims()
    {
        int.TryParse(widthText.text, out width);
        int.TryParse(heightText.text, out height);

        inputTex = new Texture2D(width, height, TextureFormat.RGBA32, false);
        tempTex = RenderTexture.GetTemporary(width, height, 24, videoTexture.format);
        SetInputDims(width, height);
        Debug.Log($"Input Dims: {width}x{height}");
        SetDevice();

        Debug.Log("Preparing Output Blobs");
        PrepareOutputBlobs();

        targetHeight = height;
    }

    public void SetDevice()
    {
        Debug.Log("Uploading Model to GPU");
        Debug.Log(deviceDropdown.value);
        //Debug.Log(deviceDropdown.);
        currentDevice = Marshal.PtrToStringAnsi(UploadModelToDevice(deviceDropdown.value));
        Debug.Log($"OpenVINO using: {currentDevice}");
        Debug.Log("Creating Inference Request");
        CreateInferenceRequest();
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

                RenderTexture.active = tempTex;
                inputTex.ReadPixels(new Rect(0, 0, tempTex.width, tempTex.height), 0, 0);
                inputTex.Apply();

                
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
        output = logString;
        stack = stackTrace;
        myLog = output + "\n" + myLog;
        if (myLog.Length > 5000)
        {
            myLog = myLog.Substring(0, 4000);
        }
    }

    void OnGUI()
    {
        myLog = GUI.TextArea(new Rect(10, Screen.height - (Screen.height * 0.25f) - 10,
                                      Screen.width * 0.25f, Screen.height * 0.25f), myLog);
    }


    public void Quit()
    {
        Application.Quit();
    }
}
