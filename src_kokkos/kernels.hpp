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

#endif // KERNELS_HPP
