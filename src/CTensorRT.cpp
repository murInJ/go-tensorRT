#include "CTensorRT.h"
#include<NvOnnxParser.h>
//CTensorRT.cpp
using namespace nvinfer1;
nvinfer1::Dims getDims(const int* sizes, int nbDims) {
    nvinfer1::Dims dims;
    dims.nbDims = nbDims;
    for (int i = 0; i < nbDims; ++i) {
        dims.d[i] = sizes[i];
    }
    return dims;
}


CtensorRT::CtensorRT() {}
CtensorRT::~CtensorRT() {}

void CtensorRT::cudaCheck(cudaError_t ret, std::ostream& err)
{
    if (ret != cudaSuccess)
    {
        err << "Cuda failure: " << cudaGetErrorString(ret) << std::endl;
        abort();
    }
}

nvinfer1::IHostMemory* loadONNX(const std::string& filepath,const std::string* input_names,
                              const int* const* minSizes, const int* const* optSizes,
                              const int* const* maxSizes, int numInputs, int numDims){
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder)
    {
        return nullptr;
    }
    
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return nullptr;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return nullptr;
    }
    
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser)
    {
        return nullptr;
    }

    parser->parseFromFile(filepath.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
    
    config->setMaxWorkspaceSize(1 << 30); // 设置工作空间大小为1GB

    for (int i = 0; i < numInputs; ++i) {
        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
        profile->setDimensions(input_names[i].c_str(), nvinfer1::OptProfileSelector::kMIN,
                            getDims(minSizes[i], numDims));
        profile->setDimensions(input_names[i].c_str(), nvinfer1::OptProfileSelector::kOPT,
                            getDims(optSizes[i], numDims));
        profile->setDimensions(input_names[i].c_str(), nvinfer1::OptProfileSelector::kMAX,
                            getDims(maxSizes[i], numDims));

        config->addOptimizationProfile(profile);
    }

    //2
    // 加载模型文件
    auto plan = builder->buildSerializedNetwork(*network, *config);
    return plan;
}

bool CtensorRT::loadOnnxModel(const std::string& filepath, const std::string* input_names,
                              const int* const* minSizes, const int* const* optSizes,
                              const int* const* maxSizes, int numInputs, int numDims)
{
    // 1
    this->_runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!this->_runtime)
    {
        std::cout << "runtime create failed" << std::endl;
        return false;
    }
    
 
    //2
    // 加载模型文件
    // auto plan = builder->buildSerializedNetwork(*network, *config);
    auto plan = loadONNX(filepath,input_names,minSizes,optSizes,maxSizes,numInputs,numDims);
    
    // 反序列化生成engine
    this->_engine = std::shared_ptr<nvinfer1::ICudaEngine>(this->_runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!this->_engine)
    {
        return false;
    }
 
    // 3
    this->_context = std::unique_ptr<nvinfer1::IExecutionContext>(this->_engine->createExecutionContext());
    if (!this->_context)
    {
        std::cout << "context create failed" << std::endl;
        return false;
    }
    int nbBindings = this->_engine->getNbBindings();
    // assert(nbBindings == 2); // 输入和输出，一共是2个

    for (int i = 0; i < nbBindings; i++)
    {
        if (_engine->bindingIsInput(i))
            this->_inputDims = this->_engine->getBindingDimensions(i);    // (1,3,64,64)
        else
            this->_outputDims = this->_engine->getBindingDimensions(i);
    }
    return true;
}

bool CtensorRT::loadEngineModel(const std::string& filepath)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.good())
    {
        return false;
    }

    std::vector<char> data;
    try
    {
        file.seekg(0, file.end);
        const auto size = file.tellg();
        file.seekg(0, file.beg);

        data.resize(size);
        file.read(data.data(), size);
    }
    catch (const std::exception& e)
    {
        file.close();
        return false;
    }
    file.close();

    this->_runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!this->_runtime)
    {
        std::cout << "runtime create failed" << std::endl;
        return false;
    }

    this->_engine = std::shared_ptr<nvinfer1::ICudaEngine>(this->_runtime->deserializeCudaEngine(data.data(), data.size()));
    if (!this->_engine)
    {
        return false;
    }

    this->_context = std::shared_ptr<nvinfer1::IExecutionContext>(this->_engine->createExecutionContext());
    if (!this->_context)
    {
        return false;
    }

    int nbBindings = this->_engine->getNbBindings();
    // assert(nbBindings == 2); // 输入和输出，一共是2个

    // 为输入和输出创建空间
    for (int i = 0; i < nbBindings; i++)
    {
        if (_engine->bindingIsInput(i))
            this->_inputDims = this->_engine->getBindingDimensions(i);    //得到输入结构
        else
            this->_outputDims = this->_engine->getBindingDimensions(i);//得到输出结构
    }
    return true;
}

void CtensorRT::ONNX2TensorRT(const char* ONNX_file, std::string save_ngine,const std::string* input_names,
                              const int* const* minSizes, const int* const* optSizes,
                              const int* const* maxSizes, int numInputs, int numDims)
{
    // 7.指定配置后，构建引擎
    nvinfer1::IHostMemory* serializedModel = loadONNX(ONNX_file,input_names,minSizes,optSizes,maxSizes,numInputs,numDims);
    // 8.保存TensorRT模型
    std::ofstream p(save_ngine, std::ios::binary);
    p.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());

    // 10.将引擎保存到磁盘，并且可以删除它被序列化到的缓冲区
    delete serializedModel;
}

uint32_t CtensorRT::getElementSize(nvinfer1::DataType t) noexcept
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8: return 1;
    }
    return 0;
}

int64_t CtensorRT::volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

bool CtensorRT::infer(Tensor* input,int batch, int real_input_size, Tensor* output)
{
    _context->setBindingDimensions(0, nvinfer1::Dims4({ input->_dims[0],input->_dims[1],input->_dims[2],input->_dims[3] }));
    tensor_custom::BufferManager buffer(this->_engine, batch, _context.get());
    cudaStream_t stream;
    cudaStreamCreate(&stream); // 创建异步cuda流

    int binds = _engine->getNbBindings();
    for (int i = 0; i < binds; i++)
    {
        if (_engine->bindingIsInput(i))
        {
            size_t input_size;
            float* host_buf = static_cast<float*>(buffer.getHostBuffer(_engine->getBindingName(i)));
            memcpy(host_buf, input->data, real_input_size);
            break;
        }
    }

    // 将输入传递到GPU
    buffer.copyInputToDeviceAsync(stream);
    // 异步执行
    bool status = _context->enqueueV2(buffer.getDeviceBindings().data(), stream, nullptr);
    if (!status)
        return false;
    buffer.copyOutputToHostAsync(stream);
    for (int i = 0; i < binds; i++)
    {
        if (!_engine->bindingIsInput(i))
        {
            output->dim_len = _engine->getTensorShape(_engine->getBindingName(i)).nbDims;
            output->_dims = new int[output->dim_len];
            output->_type = F32;

            size_t output_size;
            int total = input->_dims[0];
            //float* tmp_out = static_cast<float*>(buffer.getHostBuffer(_engine->getBindingName(i)));
            for (size_t j = 1; j < _engine->getTensorShape(_engine->getBindingName(i)).nbDims; j++)
            {
                total *= _engine->getTensorShape(_engine->getBindingName(i)).d[j];
                output->_dims[j]= _engine->getTensorShape(_engine->getBindingName(i)).d[j];
                //std::cout << _engine->getTensorShape(_engine->getBindingName(i)).d[j] << std::endl;
            }
            output->data = new float[total];
            float* tmp_out= static_cast<float*>(buffer.getHostBuffer(_engine->getBindingName(i)));
            memcpy(output->data, tmp_out, total*sizeof(float));

            // 这里做一些处理
            break;
        }
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    return true;
}


