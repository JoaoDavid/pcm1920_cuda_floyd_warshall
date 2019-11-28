There are multiple limits. All must be satisfied.

1. The maximum number of threads in the block is limited to 1024. This is the product of whatever your threadblock dimensions are (x*y*z). For example (32,32,1) creates a block of 1024 threads. (33,32,1) is not legal, since 33*32*1 > 1024.

2. The maximum x-dimension is 1024. (1024,1,1) is legal. (1025,1,1) is not legal.

3. The maximum y-dimension is 1024. (1,1024,1) is legal. (1,1025,1) is not legal.

4. The maximum z-dimension is 64. (1,1,64) is legal. (2,2,64) is also legal. (1,1,65) is not legal.

Also, threadblock dimensions of 0 in any position are not legal.

Your choice of threadblock dimensions (x,y,z) must satisfy *each* of the rules 1-4 above.

You should also do proper cuda error checking. Not sure what that is? Google "proper cuda error checking" and take the first hit.

Also run your codes with cuda-memcheck.

Do these steps *before* asking others for help. Even if you don't understand the error output, it will be useful to others trying to help you.

https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability