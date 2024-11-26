#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

// Kernel declarations
void kernel_double(Kokkos::View<float*> data);
void kernel_float(Kokkos::View<float*> data);
void kernel_half(Kokkos::View<Kokkos::Experimental::half_t*> data);
void kernel_long(Kokkos::View<float*> data);
void kernel_int(Kokkos::View<float*> data);
void kernel_short(Kokkos::View<float*> data);
void kernel_char(Kokkos::View<float*> data);
void kernel_coalesced_write(Kokkos::View<float*> data);
void kernel_coalesced_read(Kokkos::View<float*> data, Kokkos::View<float*> result);
void kernel_misaligned_write(Kokkos::View<float*> data);
void kernel_misaligned_read(Kokkos::View<float*> data, Kokkos::View<float*> result);

#if defined(Kokkkos_ENABLE_CUDA)
#include <cuda_fp16.h>

// Custom fma implementation for half_t
__device__ inline
Kokkos::Experimental::half_t custom_fma(Kokkos::Experimental::half_t a, Kokkos::Experimental::half_t b, Kokkos::Experimental::half_t c) {
    return __hfma(static_cast<__half>(a), static_cast<__half>(b), static_cast<__half>(c));
}

#else
// Fallback for non-CUDA builds
KOKKOS_INLINE_FUNCTION
Kokkos::Experimental::half_t custom_fma(Kokkos::Experimental::half_t a, Kokkos::Experimental::half_t b, Kokkos::Experimental::half_t c) {
    return static_cast<Kokkos::Experimental::half_t>(static_cast<float>(a) * static_cast<float>(b) + static_cast<float>(c));
}
#endif

#endif // KERNELS_HPP
