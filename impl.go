package go_tensorRT

import internal "github.com/murInJ/go-tensorRT/internal/tensorRT/src"

func InitTensorrtContext() *internal.TensorrtContext {
	return internal.InitTensorrtContext()
}

func ArrayToTensor(arr any) (bool, *internal.Tensor) {
	return internal.ArrayToTensor(arr)
}
