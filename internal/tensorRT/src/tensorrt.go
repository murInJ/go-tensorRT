package internal

/*
#cgo LDFLAGS: -L. -lTensorRT
#include "tensorRT.h"
*/
import "C"
import "fmt"
import "unsafe"
import "reflect"

const (
	F64 = 0
	F32 = 1
	I32 = 2
)

func getDims(arr interface{}) []int {
	val := reflect.ValueOf(arr)
	dims := make([]int, 0)
	for val.Kind() == reflect.Slice {
		dims = append(dims, val.Len())
		val = val.Index(0)
	}
	return dims
}

func getInnerElemType(sliceType reflect.Type) reflect.Type {
	if sliceType.Kind() != reflect.Slice {
		return sliceType
	}
	return getInnerElemType(sliceType.Elem())
}

func getTypeCode(arr any) int {
	// 检查 arr 是否是切片类型
	sliceType := reflect.TypeOf(arr)
	if sliceType.Kind() != reflect.Slice {
		return -1 // 不是切片类型，返回错误代码
	}

	// 获取 arr 中元素的类型
	elemType := getInnerElemType(sliceType)
	switch elemType.Kind() {
	case reflect.Float64:
		return F64
	case reflect.Float32:
		return F32
	case reflect.Int32:
		return I32
	default:
		// 不支持的类型
		return -1
	}
}

func flattenArray(data any) []float32 {
	var result []float32

	val := reflect.ValueOf(data)

	// 检查数据类型是否是数组或切片
	if val.Kind() != reflect.Array && val.Kind() != reflect.Slice {
		fmt.Println("传入的值不是数组或切片")
		return result
	}

	// 获取数组长度
	length := val.Len()

	// 遍历数组
	for i := 0; i < length; i++ {
		element := val.Index(i)

		// 如果数组元素是数组或切片，则递归展开
		if element.Kind() == reflect.Array || element.Kind() == reflect.Slice {
			result = append(result, flattenArray(element.Interface())...)
		} else {
			// 如果数组元素是基本类型，则将其添加到结果数组中
			result = append(result, float32(element.Float()))
		}
	}

	return result
}

type Tensor struct {
	_dims []int
	_type int
	data  []float32
}

func (t *Tensor) GetData() []float32 {
	return t.data
}

// 将 C.Tensor 转换为 Tensor
func cTensorToTensor(cTensor *C.Tensor) *Tensor {
	if cTensor == nil {
		return nil
	}

	// 获取维度信息
	dims := make([]int, cTensor.dim_len)
	dimsPtr := (*[1 << 30]C.int)(unsafe.Pointer(cTensor._dims))
	for i := 0; i < int(cTensor.dim_len); i++ {
		dims[i] = int(dimsPtr[i])
	}

	// 获取数据长度
	dataSize := 1
	for _, dim := range dims {
		dataSize *= dim
	}

	// 将 C 数组中的数据复制到 Go 的 float32 数组中
	var data []float32
	dataPtr := (*[1 << 30]C.float)(unsafe.Pointer(cTensor.data))
	for i := 0; i < dataSize; i++ {
		data = append(data, float32(dataPtr[i]))
	}

	// 创建 tensor 对象并返回
	return &Tensor{
		_dims: dims,
		_type: int(cTensor._type),
		data:  data,
	}
}

// 将 Tensor 转换为 C.Tensor
func tensorToCTensor(t *Tensor) *C.Tensor {
	if t == nil {
		fmt.Println("Invalid input tensor")
		return nil
	}

	// 分配 C.Tensor 的内存
	cTensor := (*C.Tensor)(C.malloc(C.size_t(unsafe.Sizeof(C.Tensor{}))))
	if cTensor == nil {
		fmt.Println("Failed to allocate memory for C.Tensor")
		return nil
	}

	// 分配维度数据的内存
	cDims := make([]C.int, len(t._dims))
	for i, v := range t._dims {
		cDims[i] = C.int(v)
	}

	cData := make([]C.float, len(t.data))
	for i, v := range t.data {
		cData[i] = C.float(v)
	}

	// 设置 C.Tensor 的属性
	cTensor._dims = (*C.int)(unsafe.Pointer(&cDims[0]))
	cTensor._type = C.int(t._type)
	cTensor.data = (*C.float)(unsafe.Pointer(&cData[0]))
	cTensor.dim_len = C.int(len(t._dims))

	return cTensor
}

func ArrayToTensor(arr any) (bool, *Tensor) {
	// 检查arr是否为空
	if arr == nil {
		return false, nil
	}

	// 获取arr的类型
	arrType := getTypeCode(arr)

	// 获取arr的维度
	dims := getDims(arr)

	data := flattenArray(arr)

	return true, &Tensor{
		_dims: dims,
		_type: arrType,
		data:  data,
	}
}

// TensorrtContext represents a TensorRT context
type TensorrtContext struct {
	ptr *C.TensorrtContext
}

// InitTensorrtContext initializes a new TensorRT context
func InitTensorrtContext() *TensorrtContext {
	ctx := C.InitTensorrtContext()
	return &TensorrtContext{ptr: ctx}
}

func (ctx *TensorrtContext) AddDynamicInput(inputName string, minSize, optSize, maxSize []int, numDims int) {
	cInputName := C.CString(inputName)
	defer C.free(unsafe.Pointer(cInputName))

	// 创建C数组，并确保其长度与切片相同
	cMinSize := (*C.int)(C.malloc(C.size_t(len(minSize)) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(cMinSize))
	cOptSize := (*C.int)(C.malloc(C.size_t(len(optSize)) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(cOptSize))
	cMaxSize := (*C.int)(C.malloc(C.size_t(len(maxSize)) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(cMaxSize))

	// 将切片复制到C数组中
	for i, v := range minSize {
		*(*C.int)(unsafe.Pointer(uintptr(unsafe.Pointer(cMinSize)) + uintptr(i)*unsafe.Sizeof(C.int(0)))) = C.int(v)
	}
	for i, v := range optSize {
		*(*C.int)(unsafe.Pointer(uintptr(unsafe.Pointer(cOptSize)) + uintptr(i)*unsafe.Sizeof(C.int(0)))) = C.int(v)
	}
	for i, v := range maxSize {
		*(*C.int)(unsafe.Pointer(uintptr(unsafe.Pointer(cMaxSize)) + uintptr(i)*unsafe.Sizeof(C.int(0)))) = C.int(v)
	}

	C.addDynamicInput(ctx.ptr, cInputName, cMinSize, cOptSize, cMaxSize, C.int(numDims))
}

// LoadOnnxModel loads an ONNX model
func (ctx *TensorrtContext) LoadOnnxModel(filepath string) int {

	cFilepath := C.CString(filepath)
	defer C.free(unsafe.Pointer(cFilepath))

	return int(C.loadOnnxModel(ctx.ptr, cFilepath))
}

// LoadEngineModel loads an engine model
func (ctx *TensorrtContext) LoadEngineModel(filepath string) int {
	cFilepath := C.CString(filepath)
	defer C.free(unsafe.Pointer(cFilepath))
	return int(C.loadEngineModel(ctx.ptr, cFilepath))
}

// ONNX2TensorRT converts ONNX to TensorRT
func (ctx *TensorrtContext) ONNX2TensorRT(ONNXFile, engineFile string) {
	cONNXFile := C.CString(ONNXFile)
	cEngineFile := C.CString(engineFile)
	defer C.free(unsafe.Pointer(cONNXFile))
	defer C.free(unsafe.Pointer(cEngineFile))
	C.ONNX2TensorRT(ctx.ptr, cONNXFile, cEngineFile)
}

// Infer performs inference
func (ctx *TensorrtContext) Infer(input *Tensor) *Tensor {
	cTensor := tensorToCTensor(input)

	output := C.infer(ctx.ptr, cTensor)
	if output == nil {
		fmt.Println("Inference failed")
		return nil
	}
	outTensor := cTensorToTensor(output)
	return outTensor
}

//func main() {
//	// onnxFilePath := "../myresnet18.onnx"
//	engineFilePath := "../../../models/tensorrt/resnet18.engine"
//	// Example usage
//	// Initialize TensorRT context
//	ctx := InitTensorrtContext()
//	// Load ONNX model
//	ctx.AddDynamicInput("input.1", []int{1, 3, 224, 224}, []int{64, 3, 224, 224}, []int{64, 3, 224, 224}, 4)
//
//	// status := ctx.LoadOnnxModel(onnxFilePath)
//	// if status == 0 {
//	//     fmt.Println("Failed to load ONNX model")
//	//     return
//	// }
//	// Convert ONNX to TensorRT
//	// ctx.ONNX2TensorRT(onnxFilePath, engineFilePath)
//
//	// Load engine model
//	status := ctx.LoadEngineModel(engineFilePath)
//	if status == 0 {
//		fmt.Println("Failed to load engine model")
//		return
//	}
//	fmt.Println(status)
//
//	// 创建一个大小为 2x3x64x64 的输入数据
//	inputRows := 224
//	inputCols := 224
//	inputChannels := 3
//	batchSize := 2
//
//	// 创建一个大小为 2 的批次
//	inputData := make([][][][]float32, batchSize)
//
//	// 填充输入数据
//	for b := 0; b < batchSize; b++ {
//		inputData[b] = make([][][]float32, inputChannels)
//		for i := 0; i < inputChannels; i++ {
//			inputData[b][i] = make([][]float32, inputRows)
//			for j := 0; j < inputRows; j++ {
//				inputData[b][i][j] = make([]float32, inputCols)
//				for c := 0; c < inputCols; c++ {
//					// 填充随机数据，这里只是示例，你可以使用实际的数据
//					inputData[b][i][j][c] = float32(1)
//				}
//			}
//		}
//	}
//	fmt.Println(6)
//	ok, inputTensor := ArrayToTensor(inputData)
//	if ok {
//		fmt.Println(7)
//	} else {
//		fmt.Println(inputTensor)
//	}
//	// Perform inference
//
//	// fmt.Println("go tensor value(in):")
//	// fmt.Println("dims:",inputTensor._dims)
//	// fmt.Println("type:",inputTensor._type)
//
//	output := ctx.Infer(inputTensor)
//	fmt.Println("Inference output:", &output)
//
//	fmt.Println(output.GetData())
//}
