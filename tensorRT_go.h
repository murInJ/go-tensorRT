#ifndef INFERENCE_WRAPPER_H
#define INFERENCE_WRAPPER_H
#include <stdlib.h>

typedef struct TensorrtContext{
    void* tensorRT;
    int numDims;
    void* input_names;
    void* minSizes;
    void* optSizes;
    void* maxSizes;
}TensorrtContext;

typedef struct Tensor {
	int dim_len;
	int* _dims;
	int _type;
	float* data;
}Tensor;

#ifdef __cplusplus
extern "C" {
#endif


TensorrtContext* InitTensorrtContext();

// int cudaCheck(cudaError_t ret, std::ostream& err = std::cerr);

void addDynamicInput(TensorrtContext* ctx,const char* input_name,int* minSize,int* optSize,int* maxSize,int numDims);

int loadOnnxModel(TensorrtContext* ctx,  const char* filepath);


int loadEngineModel(TensorrtContext* ctx,const char* filepath);


void ONNX2TensorRT(TensorrtContext* ctx,const char* ONNX_file, const char* save_ngine);

// uint32_t getElementSize(nvinfer1::DataType t) noexcept;


// int64_t volume(const nvinfer1::Dims& d);


Tensor* infer(TensorrtContext* ctx, Tensor* input);


#ifdef __cplusplus
}
#endif

#endif // INFERENCE_WRAPPER_H
