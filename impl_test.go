package go_tensorRT

import (
	"testing"
)

func create_data() (bool, *Tensor) {
	inputRows := 224
	inputCols := 224
	inputChannels := 3
	batchSize := 2

	inputData := make([][][][]float32, batchSize)

	for b := 0; b < batchSize; b++ {
		inputData[b] = make([][][]float32, inputChannels)
		for i := 0; i < inputChannels; i++ {
			inputData[b][i] = make([][]float32, inputRows)
			for j := 0; j < inputRows; j++ {
				inputData[b][i][j] = make([]float32, inputCols)
				for c := 0; c < inputCols; c++ {
					inputData[b][i][j][c] = float32(1)
				}
			}
		}
	}

	return ArrayToTensor(inputData)
}

func Test_LoadFromONNX(t *testing.T) {
	onnxFilePath := "models/onnx/resnet18.onnx"
	ctx := InitTensorrtContext()
	ctx.AddDynamicInput("input.1", []int{1, 3, 224, 224}, []int{64, 3, 224, 224}, []int{64, 3, 224, 224}, 4)

	status := ctx.LoadOnnxModel(onnxFilePath)
	if status == 0 {
		t.Errorf("Failed to load ONNX model")
	}

	ok, inputTensor := create_data()
	if !ok {
		t.Errorf("create data failed")
	}

	output := ctx.Infer(inputTensor)

	t.Log(&output)
}
func Test_ONNXTOEngine(t *testing.T) {
	onnxFilePath := "models/onnx/resnet18.onnx"
	engineFilePath := "models/tensorrt/resnet18.engine"

	ctx := InitTensorrtContext()

	ctx.AddDynamicInput("input.1", []int{1, 3, 224, 224}, []int{64, 3, 224, 224}, []int{64, 3, 224, 224}, 4)

	ctx.ONNX2TensorRT(onnxFilePath, engineFilePath)
}
func Test_LoadFromEngine(t *testing.T) {
	engineFilePath := "models/tensorrt/resnet18.engine"

	ctx := InitTensorrtContext()

	ctx.AddDynamicInput("input.1", []int{1, 3, 224, 224}, []int{64, 3, 224, 224}, []int{64, 3, 224, 224}, 4)

	status := ctx.LoadEngineModel(engineFilePath)
	if status == 0 {
		t.Errorf("Failed to load engine model")
	}
	ok, inputTensor := create_data()
	if !ok {
		t.Errorf("create data failed")
	}

	output := ctx.Infer(inputTensor)
	t.Log(&output)
}
