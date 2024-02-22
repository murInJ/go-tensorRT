#pragma once
#include "TensorRTBuffer.h"
#include <opencv2/imgcodecs.hpp>
#include<opencv2/core/core.hpp>
#define F64 0
#define F32 1
#define I32 2
typedef struct Tensor {
	int dim_len;
	int* _dims;
	int _type;
	float* data;
}Tensor;

class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override {
		if (severity <= Severity::kWARNING)
			std::cout << msg << std::endl;
	}
};

static Logger logger;

class CtensorRT
{
public:
	CtensorRT();
	~CtensorRT();

private:
	
	std::shared_ptr<nvinfer1::IRuntime> _runtime;
	std::shared_ptr<nvinfer1::ICudaEngine> _engine;
	std::shared_ptr<nvinfer1::IExecutionContext> _context;
	

	nvinfer1::Dims _inputDims;
	nvinfer1::Dims _outputDims;
public:
	void cudaCheck(cudaError_t ret, std::ostream& err = std::cerr);

	bool loadOnnxModel(const std::string& filepath, const std::string* input_names,
                              const int* const* minSizes, const int* const* optSizes,
                              const int* const* maxSizes, int numInputs, int numDims);
	

	bool loadEngineModel(const std::string& filepath);
	

	void ONNX2TensorRT(const char* ONNX_file, std::string save_ngine,const std::string* input_names,
                              const int* const* minSizes, const int* const* optSizes,
                              const int* const* maxSizes, int numInputs, int numDims);
	
	uint32_t getElementSize(nvinfer1::DataType t) noexcept;


	int64_t volume(const nvinfer1::Dims& d);


	bool infer(Tensor* input, int batch, int real_input_size, Tensor* output);
	
};


