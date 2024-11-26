#include "kernels.hpp"

// Kernel implementations

void kernel_double(Kokkos::View<float*> data) {
   Kokkos::parallel_for("kernel_double", data.extent(0), KOKKOS_LAMBDA(const int i) {
      double x = static_cast<double>(i);
      double y = static_cast<double>(i % 64);  // Simulating get_local_id(0)
      for (uint j = 0u; j < 128u; j++) {
         x = fma(y, x, y);
         y = fma(x, y, x);
      }
      data(i) = static_cast<float>(y);
   });
}

void kernel_float(Kokkos::View<float*> data) {
   Kokkos::parallel_for("kernel_float", data.extent(0), KOKKOS_LAMBDA(const int i) {
      float x = static_cast<float>(i);
      float y = static_cast<float>(i % 64);  // Simulating get_local_id(0)
      for (uint j = 0u; j < 512u; j++) {
         x = fma(y, x, y);
         y = fma(x, y, x);
      }
      data(i) = y;
   });
}

void kernel_half(Kokkos::View<Kokkos::Experimental::half_t*> data) {
   Kokkos::parallel_for("kernel_half", data.extent(0), KOKKOS_LAMBDA(const int i) {
      Kokkos::Experimental::half_t x = static_cast<Kokkos::Experimental::half_t>(i);
      Kokkos::Experimental::half_t y = static_cast<Kokkos::Experimental::half_t>(i % 64);
      for (uint j = 0u; j < 1024u; j++) {
         x = fma(y, x, y);
         y = fma(x, y, x);
      }
      data(i) = y;
   });
}

void kernel_long(Kokkos::View<float*> data) {
   Kokkos::parallel_for("kernel_long", data.extent(0), KOKKOS_LAMBDA(const int i) {
      long x = static_cast<long>(i);
      long y = static_cast<long>(i % 64);
      for (uint j = 0u; j < 256u; j++) {
         x = x ^ y;
         y = y ^ x;
         x = x ^ y;
      }
      data(i) = static_cast<float>(x + y);
   });
}

void kernel_int(Kokkos::View<float*> data) {
   Kokkos::parallel_for("kernel_int", data.extent(0), KOKKOS_LAMBDA(const int i) {
      int x = static_cast<int>(i);
      int y = static_cast<int>(i % 64);
      for (uint j = 0u; j < 512u; j++) {
         x = (x + y) * j;
         y = (y + x) * j;
      }
      data(i) = static_cast<float>(x + y);
   });
}

void kernel_short(Kokkos::View<float*> data) {
   Kokkos::parallel_for("kernel_short", data.extent(0), KOKKOS_LAMBDA(const int i) {
      short x = static_cast<short>(i);
      short y = static_cast<short>(i % 64);
      for (uint j = 0u; j < 128u; j++) {
         x = static_cast<short>((x + y) % 256);
         y = static_cast<short>((y + x) % 256);
      }
      data(i) = static_cast<float>(x + y);
   });
}

void kernel_char(Kokkos::View<float*> data) {
   Kokkos::parallel_for("kernel_char", data.extent(0), KOKKOS_LAMBDA(const int i) {
      char x = static_cast<char>(i);
      char y = static_cast<char>(i % 64);
      for (uint j = 0u; j < 64u; j++) {
         x = static_cast<char>((x + y) % 128);
         y = static_cast<char>((y + x) % 128);
      }
      data(i) = static_cast<float>(x + y);
   });
}

void kernel_coalesced_write(Kokkos::View<float*> data) {
   Kokkos::parallel_for("kernel_coalesced_write", data.extent(0), KOKKOS_LAMBDA(const int i) {
      data(i) = static_cast<float>(i);
   });
}

void kernel_coalesced_read(Kokkos::View<float*> data, Kokkos::View<float*> result) {
   Kokkos::parallel_for("kernel_coalesced_read", data.extent(0), KOKKOS_LAMBDA(const int i) {
      result(i) = data(i);
   });
}

void kernel_misaligned_write(Kokkos::View<float*> data) {
   Kokkos::parallel_for("kernel_misaligned_write", data.extent(0), KOKKOS_LAMBDA(const int i) {
      if (i + 1 < data.extent(0)) {
         data(i + 1) = static_cast<float>(i);
      }
   });
}

void kernel_misaligned_read(Kokkos::View<float*> data, Kokkos::View<float*> result) {
   Kokkos::parallel_for("kernel_misaligned_read", data.extent(0), KOKKOS_LAMBDA(const int i) {
      if (i + 1 < data.extent(0)) {
         result(i) = data(i + 1);
      } else {
         result(i) = 0.0f;
      }
   });
}
