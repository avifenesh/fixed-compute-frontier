#include <torch/extension.h>

torch::Tensor fused_rank_one_cuda(
    torch::Tensor x,
    torch::Tensor u,
    torch::Tensor v,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor packed_coupling,
    int64_t mode);

torch::Tensor fused_rank_one(
    torch::Tensor x,
    torch::Tensor u,
    torch::Tensor v,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor packed_coupling,
    int64_t mode) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(u.is_cuda() && v.is_cuda(), "u and v must be CUDA");
  TORCH_CHECK(bias.is_cuda() && scale.is_cuda(), "metadata must be CUDA");
  TORCH_CHECK(packed_coupling.is_cuda(), "coupling must be CUDA");
  TORCH_CHECK(x.is_contiguous() && u.is_contiguous() && v.is_contiguous(),
              "x, u, and v must be contiguous");
  TORCH_CHECK(bias.is_contiguous() && scale.is_contiguous() &&
                  packed_coupling.is_contiguous(),
              "metadata must be contiguous");
  TORCH_CHECK(x.dim() == 2 && u.dim() == 3 && v.dim() == 3,
              "expected x[B,D], u/v[B,L,D]");
  TORCH_CHECK(u.sizes() == v.sizes(), "u and v shapes differ");
  TORCH_CHECK(x.size(0) == u.size(0) && x.size(1) == u.size(2),
              "x and expert shapes differ");
  TORCH_CHECK(x.scalar_type() == u.scalar_type() &&
                  x.scalar_type() == v.scalar_type() &&
                  x.scalar_type() == bias.scalar_type() &&
                  x.scalar_type() == scale.scalar_type() &&
                  x.scalar_type() == packed_coupling.scalar_type(),
              "all inputs must have the same dtype");
  TORCH_CHECK(mode >= 0 && mode <= 2,
              "mode must be 0=additive, 1=compiled, or 2=direct");
  return fused_rank_one_cuda(x, u, v, bias, scale, packed_coupling, mode);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fused_rank_one, "Fused rank-one executor");
}
