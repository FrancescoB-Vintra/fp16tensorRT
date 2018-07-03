#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvUtils.h"
#include "cuda_runtime_api.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <type_traits>
#include "fp16.h"

#define CHECK(status)									\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}


// stuff we know about the network and the input/output blobs
static const int INPUT_H = 256;
static const int INPUT_W = 384;
static const int INPUT_C = 3;

static const int OUTPUT_TEST_H = 256;
static const int OUTPUT_TEST_W = 384;
static const int OUTPUT_TEST_C = 64;
static const int OUTPUT_TEST_SIZE = OUTPUT_TEST_H * OUTPUT_TEST_W * OUTPUT_TEST_C;

static const int BATCH_SIZE = 4;
static const int MAX_BATCH_SIZE = 4;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "out";

const int N_BINDINGS = 2;
static void* buffers[N_BINDINGS];
static cudaStream_t stream;
static int inputIndex, outputIndexTest;

using namespace nvinfer1;


static const DataType DATATYPE = DataType::kHALF;
const char* WEIGHTS_FILENAME = "weights_demo16.wts";
//static const DataType DATATYPE = DataType::kFLOAT;
//const char* WEIGHTS_FILENAME = "weights_demo32.wts";


// Logger for GIE info/warning/errors
class Logger : public nvinfer1::ILogger			
{
    public:
	void log(nvinfer1::ILogger::Severity severity, const char* msg) override
	{
		// suppress info-level messages
        if (severity == Severity::kINFO) return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
	}
};


static Logger gLogger;

union hf
{
    uint32_t data_u;
    float data_f;
} converter;

// Our weight files are in a very simple space delimited format.
// [type] [size] <data x size in hex> 
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::map<std::string, Weights> weightMap;
	std::ifstream input(file);
	assert(input.is_open() && "Unable to load weight file.");
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    while(count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t type, size;
        std::string name;
        input >> name >> std::dec >> type >> size;
        wt.type = static_cast<DataType>(type);
        
        if (wt.type == DataType::kFLOAT)
        {
            uint32_t *val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        }
        else if (wt.type == DataType::kHALF)
        {
            __half *val = reinterpret_cast<__half*>(malloc(sizeof(__half) * size));
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> converter.data_u;		
                val[x] = fp16::__float2half(converter.data_f);
            }
            wt.values = val;
        }
        wt.count = size;
        weightMap[name] = wt;
    }
    return weightMap;
}



std::string locateFile(const std::string& input, const std::vector<std::string> & directories)
{
    std::string file;
	const int MAX_DEPTH{10};
    bool found{false};
    for (auto &dir : directories)
    {
        file = dir + input;
        for (int i = 0; i < MAX_DEPTH && !found; i++)
        {
            std::ifstream checkFile(file);
            found = checkFile.is_open();
            if (found) break;
            file = "../" + file;
        }
        if (found) break;
        file.clear();
    }

    assert(!file.empty() && "Could not find a file due to it not existing in the data directory.");
    return file;
}

// We have the data files located in a specific directory. This 
// searches for that directory format from the current directory.
std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"./data/"};
    return locateFile(input, dirs);
}

// print tensor dimensions
void printDims(ITensor* data)
{
    Dims dims = data->getDimensions();
    int nbDims = dims.nbDims;
    for (int d = 0; d < nbDims; d++)
        std::cout << dims.d[d] << " ";// << dims.d[1] << " " << dims.d[2] << " " << dims.d[3] << std::endl;
    std::string sss;    
    if (data->getType() == DataType::kHALF)
        sss = "float16";
    if (data->getType() == DataType::kFLOAT)
        sss = "float32";
    std::cout << sss << " ";
    std::cout << std::endl;
}


static void setAllLayerOutputsToHalf(INetworkDefinition* network)
{
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        nvinfer1::ILayer* layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); j++)
        {
            if (layer->getOutput(j)->isNetworkOutput())
                layer->getOutput(j)->setType(DataType::kHALF);
        }
    }
}

void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream)
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);


///////////////////////////////////////////////////////////
    INetworkDefinition* network = builder->createNetwork();
    
    // load weights values from disk
    std::map<std::string, Weights> weightMap = loadWeights(locateFile(WEIGHTS_FILENAME));

	// define input
	auto data = network->addInput(INPUT_BLOB_NAME, DATATYPE, DimsCHW{INPUT_C, INPUT_H, INPUT_W});
	assert(data != nullptr);
    std::cout << "input" << std::endl;
    printDims(data);


    // add layer
    // 1 ////////////////////////////////////
    auto conv1 = network->addConvolution(*data, 64, DimsHW{3, 3}, weightMap["conv1_w"], weightMap["conv1_b"]);
	assert(conv1 != nullptr);
	conv1->setStride(DimsHW{1, 1});
    conv1->setPadding(DimsHW{1, 1});

    // set output
    conv1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
	network->markOutput(*conv1->getOutput(0));



///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1e6);
    if (DATATYPE == DataType::kHALF)
    {
    	builder->setHalf2Mode(true);
        setAllLayerOutputsToHalf(network);
	}
    std::cout << "building the engine..." << std::endl;

	auto engine = builder->buildCudaEngine(*network);
         assert(engine != nullptr);

    std::cout << "engine built!" << std::endl;

	// serialize the engine, then close everything down
	(*modelStream) = engine->serialize();

	// Once we have built the cuda engine, we can release all of our held memory.
	for (auto &mem : weightMap)
    {
        free((void*)(mem.second.values));
    }
///////////////////////////////////////////////////////////

    network->destroy();
	engine->destroy();
	builder->destroy();
}

template <typename T>
void setUpDevice(IExecutionContext& context, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == N_BINDINGS);

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    outputIndexTest = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * INPUT_C * sizeof(T)));
    CHECK(cudaMalloc(&buffers[outputIndexTest], batchSize * OUTPUT_TEST_SIZE * sizeof(T)));

    // create cuda stream
    CHECK(cudaStreamCreate(&stream));
}

void cleanUp()
{
  	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndexTest]));
}

void doInference(IExecutionContext& context, __half* input, __half* output, int batchSize)
{
	// DMA the input to the GPU, execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * INPUT_C * sizeof(__half), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndexTest], batchSize * OUTPUT_TEST_SIZE * sizeof(__half), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	// DMA the input to the GPU, execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * INPUT_C * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndexTest], batchSize * OUTPUT_TEST_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);
}


// rearrange image data to [N, C, H, W] order
void prepareDataBatch(__half* data, std::vector<cv::Mat> &frames)
{
     assert(data && !frames.empty());
     unsigned int volChl = INPUT_H * INPUT_W;
     unsigned int volImg = INPUT_H * INPUT_W * INPUT_C;
     
     for (int b = 0; b < BATCH_SIZE; b++)
         for (int c = 0; c < INPUT_C; c++)
         {
              // the color image to input should be in BGR order
              for (unsigned j = 0; j < volChl; j++)
                   data[b * volImg + c * volChl + j] = fp16::__float2half( (frames[b].data[j * INPUT_C + c]) / 255.0);
         }
     
     return;
}

void prepareDataBatch(float* data, std::vector<cv::Mat> &frames)
{
     assert(data && !frames.empty());
     unsigned int volChl = INPUT_H * INPUT_W;
     unsigned int volImg = INPUT_H * INPUT_W * INPUT_C;
     
     for (int b = 0; b < BATCH_SIZE; b++)
         for (int c = 0; c < INPUT_C; c++)
         {
              // the color image to input should be in BGR order
              for (unsigned j = 0; j < volChl; j++)
                   data[b * volImg + c * volChl + j] = (frames[b].data[j * INPUT_C + c]) / 255.0;
         }
     
     return;
}



void printOutput(float *out, const int batch_size, const int output_c,  const int output_h,  const int output_w)
{
    int output_size(output_c * output_h * output_w);

    std::cout << "================="<< std::endl;   
    std::cout << "================="<< std::endl;
    std::cout << "-----------------"<< std::endl; 
    for (int b = 0; b < batch_size; b++)
        {
        for (int c = 0; c < output_c; c++)
            {
                for (int h = 0; h < 5; h++)//output_h; h++)
                {
                    for (int w = 0; w < 5; w++)//output_w; w++)
                        std::cout << out[b * output_size + c * output_h * output_w + h * output_w + w] << " ";
                    std::cout << std::endl;
                }
            std::cout << "-----------------"<< std::endl; 
            }
        std::cout << "================="<< std::endl;   
        std::cout << "================="<< std::endl;
        }

    return;
}



void castOutput(__half* data_h, float* data_f, int output_size)
{
    for (int i = 0; i < output_size; i++)
         data_f[i] = fp16::__half2float(data_h[i]);
    return;
}

void castOutput(float* data_h, float* data_f, int output_size)
{
    for (int i = 0; i < output_size; i++)
         data_f[i] = data_h[i];
    return;
}


typedef std::conditional<DATATYPE == DataType::kHALF, __half, float>::type FloatPrecision;


int main(int argc, char** argv)
{
    std::cout << sizeof(float) << std::endl;

    //read input, convert, resize
    std::vector<std::string> image_paths;
    image_paths.push_back("./images/2.jpg");
    image_paths.push_back("./images/9.jpg");
    image_paths.push_back("./images/18.jpg");
    image_paths.push_back("./images/20.jpg");

    std::vector<cv::Mat> images(BATCH_SIZE);

    for (int b = 0; b < BATCH_SIZE; b++)
    {
        std::cout << image_paths[b] << std::endl;
        cv::Mat image_bgr = cv::imread(image_paths[b]);
        cv::Mat image_rgb, image;
        cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);
        cv::resize(image_rgb, image_rgb, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_LINEAR);
        images[b] = image_rgb;
    }

    // allocate CPU memory for input and output
    int inputSize = BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W;
    int outputSizeTest = BATCH_SIZE * OUTPUT_TEST_SIZE;
    FloatPrecision* data = new FloatPrecision[inputSize];
    FloatPrecision* outtest = new FloatPrecision[outputSizeTest];
    float* outtest_f = new float[outputSizeTest];

	// init model stream variables
    IHostMemory *modelStream{nullptr};
	IRuntime* runtime = createInferRuntime(gLogger);

	// create a model using the API directly and serialize it to a stream
    APIToModel(MAX_BATCH_SIZE, &modelStream);
    ICudaEngine* engine = runtime->deserializeCudaEngine(modelStream->data(), modelStream->size(), nullptr);

    // create execution context
	IExecutionContext *context = engine->createExecutionContext();

    // allocate memory on device
    if (DATATYPE == DataType::kHALF)
        setUpDevice<__half>(*context, BATCH_SIZE);
    else
        setUpDevice<float>(*context, BATCH_SIZE);

    // flatten image Mat, convert to float, do TF-style whitening
    prepareDataBatch(data, images);

    // run inference
    doInference(*context, data, outtest, BATCH_SIZE);

	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();

    // clean-up device
    cleanUp();

    // print out
    castOutput(outtest, outtest_f, outputSizeTest);
    printOutput(outtest_f, BATCH_SIZE, OUTPUT_TEST_C, OUTPUT_TEST_H, OUTPUT_TEST_W);

    //free mem
    delete[] data;
    delete[] outtest;
    delete[] outtest_f;

    std::cout << "done!" << std::endl;

    return 0;
}

