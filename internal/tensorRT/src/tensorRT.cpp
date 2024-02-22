#include "CTensorRT.h"
// #include "tensorRT.h"
#include "tensorRT_c.h"
cv::Mat* tensorToMat(const Tensor* tensor) {
    if (tensor == nullptr) {
        std::cerr << "Invalid tensor pointer" << std::endl;
        return nullptr;
    }

    int depth = CV_8U; // Assume unsigned char by default

    switch(tensor->_type) {
        case F64: // F64
            depth = CV_64F;
            break;
        case F32: // F32
            depth = CV_32F;
            break;
        case I32: // I32
            depth = CV_32S;
            break;
        default:
            std::cerr << "Unsupported tensor type" << std::endl;
            return nullptr;
    }


    // Constructing the cv::Mat from the tensor data
    // cv::Mat* mat = new cv::Mat(tensor->_dims[0], tensor->_dims[1], CV_MAKETYPE(depth, channels), tensor->data);
    //cv::Mat* mat = new cv::Mat(tensor->dim_len, tensor->_dims, depth, tensor->data);
    cv::Mat* mat = new cv::Mat(tensor->_dims[2], tensor->_dims[3], CV_32FC3, tensor->data);

    //std::cout << "Mat:" << std::endl << *mat << std::endl;

     return new cv::Mat(mat->clone()); // Cloning to make sure it's detached from original data
}

Tensor* matToTensor(const cv::Mat* mat) {
     if (mat == nullptr) {
         std::cerr << "Invalid mat pointer" << std::endl;
         return nullptr;
     }
    
     Tensor* tensor = new Tensor();
     tensor->_dims = new int[2];
     tensor->_dims[0] = mat->rows;
     tensor->_dims[1] = mat->cols;
     tensor->_type = mat->depth();

     int dataSize = mat->total() * mat->elemSize();
     tensor->data = new float[dataSize];
     std::memcpy(tensor->data, mat->data, dataSize);

     return tensor;
}


TensorrtContext* InitTensorrtContext(){
    TensorrtContext* ctx = new TensorrtContext;
    CtensorRT* cTensor = new CtensorRT;
    std::vector<std::string>* input_names = new std::vector<std::string>;
    std::vector<int*>* minSizes = new std::vector<int*>;
    std::vector<int*>* optSizes = new std::vector<int*>;
    std::vector<int*>* maxSizes = new std::vector<int*>;
    ctx->tensorRT =  static_cast<void*>(cTensor);
    ctx->input_names = static_cast<void*>(input_names);
    ctx->minSizes = static_cast<void*>(minSizes);
    ctx->optSizes = static_cast<void*>(optSizes);
    ctx->maxSizes = static_cast<void*>(maxSizes);
    return ctx;
}

//for debug
// void printArray(const std::vector<std::string>& input_names,
//                 const std::vector<int*>& minSizes,
//                 const std::vector<int*>& optSizes,
//                 const std::vector<int*>& maxSizes,
//                 int numDims) {
//     // Print input_names
//     std::cout << "Input Names: ";
//     for (const auto& name : input_names) {
//         std::cout << name << " ";
//     }
//     std::cout << std::endl;
//
//     // Print minSizes
//     std::cout << "Min Sizes: ";
//     for (int i = 0; i < minSizes.size(); ++i) {
//         std::cout << "Input " << i << ": ";
//         for (int j = 0; j < numDims; ++j) {
//             std::cout << minSizes[i][j] << " ";
//         }
//         std::cout << "| ";
//     }
//     std::cout << std::endl;
//
//     // Print optSizes
//     std::cout << "Opt Sizes: ";
//     for (int i = 0; i < optSizes.size(); ++i) {
//         std::cout << "Input " << i << ": ";
//         for (int j = 0; j < numDims; ++j) {
//             std::cout << optSizes[i][j] << " ";
//         }
//         std::cout << "| ";
//     }
//     std::cout << std::endl;
//
//     // Print maxSizes
//     std::cout << "Max Sizes: ";
//     for (int i = 0; i < maxSizes.size(); ++i) {
//         std::cout << "Input " << i << ": ";
//         for (int j = 0; j < numDims; ++j) {
//             std::cout << maxSizes[i][j] << " ";
//         }
//         std::cout << "| ";
//     }
//     std::cout << std::endl;
// }

void addDynamicInput(TensorrtContext* ctx,const char* input_name,int* minSize,int* optSize,int* maxSize,int numDims){
   
    CtensorRT* cTensor = reinterpret_cast<CtensorRT*>(ctx->tensorRT);
   
     // 为输入名称创建字符串，并添加到 input_names 向量中
    std::string inputNameStr=input_name;
    std::vector<std::string>* inputNamesVec = reinterpret_cast<std::vector<std::string>*>(ctx->input_names);
    inputNamesVec->push_back(inputNameStr);

    // 为每个数组分配内存，并将数据复制到该内存中
    int* t = new int[10];
    int* minSizeCopy = new int[numDims];
    std::copy(minSize, minSize + numDims, minSizeCopy);
    std::vector<int*>* minSizesVec = reinterpret_cast<std::vector<int*>*>(ctx->minSizes);
    minSizesVec->push_back(minSizeCopy);

    int* optSizeCopy = new int[numDims];
    std::copy(optSize, optSize + numDims, optSizeCopy);
    std::vector<int*>* optSizesVec = reinterpret_cast<std::vector<int*>*>(ctx->optSizes);
    optSizesVec->push_back(optSizeCopy);

    int* maxSizeCopy = new int[numDims];
    std::copy(maxSize, maxSize + numDims, maxSizeCopy);
    std::vector<int*>* maxSizesVec = reinterpret_cast<std::vector<int*>*>(ctx->maxSizes);
    maxSizesVec->push_back(maxSizeCopy);

    ctx->numDims = numDims;

    // printArray(*inputNamesVec,*minSizesVec,*optSizesVec,*maxSizesVec,ctx->numDims);
}

int loadOnnxModel(TensorrtContext* ctx,  const char* filepath) {
    std::vector<std::string>* input_names = reinterpret_cast<std::vector<std::string>*>(ctx->input_names);
    std::vector<int*>* minSizes = reinterpret_cast<std::vector<int*>*>(ctx->minSizes);
    std::vector<int*>* optSizes = reinterpret_cast<std::vector<int*>*>(ctx->optSizes);
    std::vector<int*>* maxSizes = reinterpret_cast<std::vector<int*>*>(ctx->maxSizes);
   
 
    const std::string* inputNamesPtr = input_names->data();
    const int *const *minSizesPtr = reinterpret_cast<const int *const *>(minSizes->data());
    const int *const *optSizesPtr = reinterpret_cast<const int *const *>(optSizes->data());
    const int *const *maxSizesPtr = reinterpret_cast<const int *const *>(maxSizes->data());
    CtensorRT* cTensor = reinterpret_cast<CtensorRT*>(ctx->tensorRT);
    return cTensor->loadOnnxModel(filepath, inputNamesPtr, minSizesPtr, optSizesPtr, maxSizesPtr, minSizes->size(),ctx->numDims) ? 1 : 0;
}

int loadEngineModel(TensorrtContext* ctx,const char* filepath){
    CtensorRT* cTensor = reinterpret_cast<CtensorRT*>(ctx->tensorRT);
    return cTensor->loadEngineModel(filepath) ? 1 : 0;
}

void ONNX2TensorRT(TensorrtContext* ctx,const char* ONNX_file, const char* save_ngine){
    std::vector<std::string>* input_names = reinterpret_cast<std::vector<std::string>*>(ctx->input_names);
    std::vector<int*>* minSizes = reinterpret_cast<std::vector<int*>*>(ctx->minSizes);
    std::vector<int*>* optSizes = reinterpret_cast<std::vector<int*>*>(ctx->optSizes);
    std::vector<int*>* maxSizes = reinterpret_cast<std::vector<int*>*>(ctx->maxSizes);
   
 
    const std::string* inputNamesPtr = input_names->data();
    const int *const *minSizesPtr = reinterpret_cast<const int *const *>(minSizes->data());
    const int *const *optSizesPtr = reinterpret_cast<const int *const *>(optSizes->data());
    const int *const *maxSizesPtr = reinterpret_cast<const int *const *>(maxSizes->data());
    CtensorRT* cTensor = reinterpret_cast<CtensorRT*>(ctx->tensorRT);
    cTensor->ONNX2TensorRT(ONNX_file,save_ngine,inputNamesPtr,minSizesPtr,optSizesPtr,maxSizesPtr,1,ctx->numDims);
    //cTensor->ONNX2TensorRT(ONNX_file,save_ngine,inputNamesPtr,minSizesPtr,optSizesPtr,maxSizesPtr,minSizes->size(),ctx->numDims);
}

Tensor* infer(TensorrtContext* ctx, Tensor* input) {
    if (ctx == nullptr || input == nullptr) {
        std::cerr << "Invalid context or input tensor pointer" << std::endl;
        return nullptr;
    }

    CtensorRT* cTensor = reinterpret_cast<CtensorRT*>(ctx->tensorRT);
    cv::Mat* inputMat = tensorToMat(input); // Assuming tensorToMat function converts Tensor to cv::Mat
    if (inputMat == nullptr) {
        std::cerr << "Failed to convert input tensor to cv::Mat" << std::endl;
        return nullptr;
    }

     Tensor *output=new Tensor;
     int input_numel = input->_dims[0]* input->_dims[1] * input->_dims[2] * input->_dims[3]*sizeof(float);
     bool success = cTensor->infer(input, input->_dims[0], input_numel, output);
   
     delete inputMat; // Free the allocated cv::Mat pointer
     if (success) {
         output->_dims[0] = input->_dims[0];
         return output;
     } else {
         std::cerr << "Inference failed" << std::endl;
         return nullptr;
     }
}


//用于c++测试
//int main() {
//
//    auto ctx = InitTensorrtContext();
//    //std::string onnxFilePath = "resnet18.onnx";
//    //ONNX2TensorRT(ctx, onnxFilePath.c_str(), engineFilePath.c_str());
//    //std::string engineFilePath = "resnet18.engine";
//    std::string engineFilePath = "myresnet18.engine";
//    auto ret = loadEngineModel(ctx,engineFilePath.c_str());
//    Tensor *t=new Tensor();
//    t->_dims = new int[4];
//    t->dim_len = 4;
//    t->_type = 1;
//    t->_dims[0] = 2;
//    t->_dims[1] = 3;
//    t->_dims[2] = 224;
//    t->_dims[3] = 224;
//    int input_numel = t->_dims[1] * t->_dims[2] * t->_dims[3];
//    cv::Mat mat(224, 224, CV_32FC3, cv::Scalar(0.5, 0.2, 0.3));
//    t->data = new float[t->_dims[0] * input_numel];
//    for (size_t i = 0; i < t->_dims[0]; i++)
//    {
//        memcpy(t->data+ input_numel*i, mat.data, sizeof(float) * input_numel);
//    }
//    //t->data = (float*)mat.data;
//    auto out=infer(ctx,t);
//    std::cout <<"succuss?"<< ret << std::endl;
//    //std::cout<<out->_dims[0]<<std::endl;
//    //std::cout<<out->_dims[1]<<std::endl;
//    //std::cout<<out->_type<<std::endl;
//    return 0;
//}