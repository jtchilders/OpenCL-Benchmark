#include <Kokkos_Core.hpp>
#include "kernels.hpp"
#include <iostream>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;

void benchmark_kernels() {
   const uint N = 4096u * 4096u;  // Kernel range: 1GB memory allocation
   const uint N_kernel = 256u;   // Iterations for kernel calls

   // Allocate data on the device
   Kokkos::View<float*> data("data", N);
   Kokkos::View<float*> result("result", N);
   Kokkos::View<Kokkos::Experimental::half_t*> half_data("half_data", N);

   std::cout << "Starting kernel benchmarks...\n";

   // Measure execution time for each kernel
   auto measure_time = [](const std::string& name, const auto& kernel) {
      auto start = Clock::now();
      kernel();
      Kokkos::fence();  // Ensure all operations complete
      auto end = Clock::now();
      std::chrono::duration<double> elapsed = end - start;
      std::cout << name << " completed in " << elapsed.count() << " seconds.\n";
   };

   // Benchmark each kernel
   measure_time("kernel_double", [&]() { kernel_double(data); });
   measure_time("kernel_float", [&]() { kernel_float(data); });
   measure_time("kernel_half", [&]() { kernel_half(half_data); });
   measure_time("kernel_long", [&]() { kernel_long(data); });
   measure_time("kernel_int", [&]() { kernel_int(data); });
   measure_time("kernel_short", [&]() { kernel_short(data); });
   measure_time("kernel_char", [&]() { kernel_char(data); });
   measure_time("kernel_coalesced_write", [&]() { kernel_coalesced_write(data); });
   measure_time("kernel_coalesced_read", [&]() { kernel_coalesced_read(data, result); });
   measure_time("kernel_misaligned_write", [&]() { kernel_misaligned_write(data); });
   measure_time("kernel_misaligned_read", [&]() { kernel_misaligned_read(data, result); });

   std::cout << "All benchmarks completed.\n";
}

int main(int argc, char* argv[]) {
   Kokkos::initialize(argc, argv);  // Initialize Kokkos runtime
   {
      benchmark_kernels();          // Run benchmarks
   }
   Kokkos::finalize();              // Finalize Kokkos runtime
   return 0;
}
