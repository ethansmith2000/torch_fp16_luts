// exp_lut_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Declare the LUT as a device pointer
__device__ __constant__ half exp_lut_device[65536];

// Kernel to perform LUT-based exp
__global__ void exp_lut_kernel(const half* input, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Reinterpret the input half as uint16 to use as index
        unsigned short input_bits = *((unsigned short*)&input[idx]);
        half result = exp_lut_device[input_bits];
        output[idx] = result;
    }
}

// Host function to launch the kernel
void exp_lut_cuda(torch::Tensor input, torch::Tensor output, torch::Tensor lut) {
    // Ensure inputs are on CUDA
    const half* input_ptr = input.data_ptr<half>();
    half* output_ptr = output.data_ptr<half>();
    const half* lut_ptr = lut.data_ptr<half>();

    // Copy the LUT to constant memory
    cudaMemcpyToSymbol(exp_lut_device, lut_ptr, sizeof(half) * 65536);

    // Define CUDA grid and block dimensions
    int threads = 256;
    int blocks = (input.numel() + threads - 1) / threads;

    // Launch the kernel
    exp_lut_kernel<<<blocks, threads>>>(input_ptr, output_ptr, input.numel());

    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}

// Binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("exp_lut_cuda", &exp_lut_cuda, "FP16 Exp using LUT on CUDA");
}
