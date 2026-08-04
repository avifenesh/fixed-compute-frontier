#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_sum(float value, float* warp_sums) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_sum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < kWarps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    value = warp_sum(value);
  }
  return value;
}

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

template <typename scalar_t, int Length, int Mode>
__global__ void fused_rank_one_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ scale,
    const scalar_t* __restrict__ packed_coupling,
    scalar_t* __restrict__ output,
    int dimension) {
  const int batch = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t x_base = static_cast<int64_t>(batch) * dimension;
  const int64_t expert_base = static_cast<int64_t>(batch) * Length * dimension;
  __shared__ float preactivation[Length];
  __shared__ float coefficient[Length];
  __shared__ float warp_sums[kWarps];

  if constexpr (Mode != 2) {
    // Assign independent x dot u_i reductions to separate warps.  This is the
    // concurrency the unfused PyTorch prototype could not preserve.
    for (int i = warp; i < Length; i += kWarps) {
      float partial = 0.0f;
      const int64_t row = expert_base + static_cast<int64_t>(i) * dimension;
      for (int d = lane; d < dimension; d += 32) {
        partial += static_cast<float>(x[x_base + d]) *
                   static_cast<float>(u[row + d]);
      }
      partial = warp_sum(partial);
      if (lane == 0) {
        preactivation[i] = partial +
                           static_cast<float>(bias[batch * Length + i]);
      }
    }
    __syncthreads();

    if constexpr (Mode == 0) {
      for (int i = threadIdx.x; i < Length; i += kThreads) {
        coefficient[i] =
            static_cast<float>(scale[batch * Length + i]) *
            silu(preactivation[i]);
      }
    } else {
      if (threadIdx.x == 0) {
        const int64_t coupling_base =
            static_cast<int64_t>(batch) * Length * (Length - 1) / 2;
#pragma unroll
        for (int i = 0; i < Length; ++i) {
          float value = preactivation[i];
          const int start = i * (i - 1) / 2;
#pragma unroll
          for (int j = 0; j < i; ++j) {
            value += static_cast<float>(packed_coupling[coupling_base + start + j]) *
                     coefficient[j];
          }
          coefficient[i] =
              static_cast<float>(scale[batch * Length + i]) * silu(value);
        }
      }
    }
    __syncthreads();

    for (int d = threadIdx.x; d < dimension; d += kThreads) {
      float value = static_cast<float>(x[x_base + d]);
#pragma unroll
      for (int i = 0; i < Length; ++i) {
        const int64_t row = expert_base + static_cast<int64_t>(i) * dimension;
        value += coefficient[i] * static_cast<float>(v[row + d]);
      }
      output[x_base + d] = static_cast<scalar_t>(value);
    }
  } else {
    // Strong fused sequential control.  It pays no compiled Gram table, but
    // it must serialize Length block-wide reductions and vector updates.
    extern __shared__ float state[];
    for (int d = threadIdx.x; d < dimension; d += kThreads) {
      state[d] = static_cast<float>(x[x_base + d]);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < Length; ++i) {
      float partial = 0.0f;
      const int64_t row = expert_base + static_cast<int64_t>(i) * dimension;
      for (int d = threadIdx.x; d < dimension; d += kThreads) {
        partial += state[d] * static_cast<float>(u[row + d]);
      }
      const float total = block_sum(partial, warp_sums);
      if (threadIdx.x == 0) {
        coefficient[0] =
            static_cast<float>(scale[batch * Length + i]) *
            silu(total + static_cast<float>(bias[batch * Length + i]));
      }
      __syncthreads();
      const float c = coefficient[0];
      for (int d = threadIdx.x; d < dimension; d += kThreads) {
        state[d] += c * static_cast<float>(v[row + d]);
      }
      __syncthreads();
    }
    for (int d = threadIdx.x; d < dimension; d += kThreads) {
      output[x_base + d] = static_cast<scalar_t>(state[d]);
    }
  }
}

template <typename scalar_t, int Length>
void launch(
    const torch::Tensor& x,
    const torch::Tensor& u,
    const torch::Tensor& v,
    const torch::Tensor& bias,
    const torch::Tensor& scale,
    const torch::Tensor& coupling,
    torch::Tensor& output,
    int mode,
    cudaStream_t stream) {
  const int batches = x.size(0);
  const int dimension = x.size(1);
  const size_t direct_shared = static_cast<size_t>(dimension) * sizeof(float);
  if (mode == 0) {
    fused_rank_one_kernel<scalar_t, Length, 0>
        <<<batches, kThreads, 0, stream>>>(
            x.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(), scale.data_ptr<scalar_t>(),
            coupling.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), dimension);
  } else if (mode == 1) {
    fused_rank_one_kernel<scalar_t, Length, 1>
        <<<batches, kThreads, 0, stream>>>(
            x.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(), scale.data_ptr<scalar_t>(),
            coupling.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), dimension);
  } else {
    fused_rank_one_kernel<scalar_t, Length, 2>
        <<<batches, kThreads, direct_shared, stream>>>(
            x.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(), scale.data_ptr<scalar_t>(),
            coupling.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), dimension);
  }
}

}  // namespace

torch::Tensor fused_rank_one_cuda(
    torch::Tensor x,
    torch::Tensor u,
    torch::Tensor v,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor packed_coupling,
    int64_t mode) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto output = torch::empty_like(x);
  const int64_t length = u.size(1);
  TORCH_CHECK(length == 4 || length == 8 || length == 16 || length == 32,
              "length must be one of 4, 8, 16, 32");
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "fused_rank_one_cuda", [&] {
        switch (length) {
          case 4:
            launch<scalar_t, 4>(x, u, v, bias, scale, packed_coupling,
                                output, mode, stream);
            break;
          case 8:
            launch<scalar_t, 8>(x, u, v, bias, scale, packed_coupling,
                                output, mode, stream);
            break;
          case 16:
            launch<scalar_t, 16>(x, u, v, bias, scale, packed_coupling,
                                 output, mode, stream);
            break;
          case 32:
            launch<scalar_t, 32>(x, u, v, bias, scale, packed_coupling,
                                 output, mode, stream);
            break;
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
