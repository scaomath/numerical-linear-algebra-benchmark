'''
modified from https://github.com/HPAC/LinearAlgebra-Awareness-Benchmark
'''
import torch
import os
import time

class bcolors:
    WARNING = '\033[93m'
    ENDC = '\033[0m'

#Check if MKL is enabled
print(bcolors.WARNING + "MKL Enabled : ", torch.backends.mkl.is_available(), bcolors.ENDC)


#Sets the number of threads used for intraop parallelism on CPU.
torch.set_num_threads(1)

#Problem size
n = 3000
reps = 10
DTYPE = torch.float32


@torch.jit.script
def mc_cse_non_optimized(A,B):
    ret = torch.t(torch.t(A)@B)@(torch.t(A)@B)    
    return ret

@torch.jit.script
def mc_cse_optimized(A,B):
    tmp = torch.t(A)@B
    ret = torch.t(tmp)@tmp
    return ret


A = torch.randn([n, n], dtype=DTYPE)
B = torch.randn([n, n], dtype=DTYPE)

non_optimized_time = 0
optimized_time = 0

for _ in range(reps):
   start = time.perf_counter()
   _ = mc_cse_non_optimized(A,B)
   non_optimized_time += time.perf_counter() - start

   start = time.perf_counter()
   _ = mc_cse_optimized(A,B)
   optimized_time += time.perf_counter() - start

print(f"Non Optimized : {non_optimized_time/reps:.5f}s ")
print(f"Optimized : {optimized_time/reps:.5f}s ")
