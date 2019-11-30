#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define GRAPH_SIZE 2000

#define EDGE_COST(graph, graph_size, a, b) graph[a * graph_size + b]
#define D(a, b) EDGE_COST(output, graph_size, a, b)

#define INF 0x1fffffff

void generate_random_graph(int *output, int graph_size) {
  int i, j;
  int counter = 0;
  srand(0xdadadada);

  for (i = 0; i < graph_size; i++) {
    for (j = 0; j < graph_size; j++) {
      if (i == j) {
        D(i, j) = 0;
      } else {
        int r;
        r = rand() % 40;
        if (r > 20) {
          //r = INF;
        }

        D(i, j) = r;
        if(r == 0){
          counter++;
          D(i, j) = 1;
        }
      }
    }
  }
  printf("counter:%d\n", counter);
}

int gcd(int a, int b) { 
    if (b == 0) {
      return a; 
    }        
    return gcd(b, a % b);  
} 

__global__ void gpu_calculate(int k, int graph_size, int *output, int threads) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < GRAPH_SIZE && j < GRAPH_SIZE){
      extern __shared__ int shared[];
      int* frozenZoneHoriz = &shared[0];
      int* frozenZoneVert = &shared[threads];
      /*if(threadIdx.y == 0){
        frozenZoneHoriz[threadIdx.x] = D(i, k);
      }
      if(threadIdx.x == 0){
        frozenZoneVert[threadIdx.y] = D(k, j);
      }*/
      if(threadIdx.x == threadIdx.y){
        frozenZoneHoriz[threadIdx.x] = D(i, k);
        frozenZoneVert[threadIdx.y] = D(k, j);
      }
      
      __syncthreads();

      if (frozenZoneHoriz[threadIdx.x] + frozenZoneVert[threadIdx.y] < D(i, j)) {
        D(i, j) = frozenZoneHoriz[threadIdx.x] + frozenZoneVert[threadIdx.y];
      }
    }
}

void floyd_warshall_gpu(const int *graph, int graph_size, int *output) {
  int threads = gcd(GRAPH_SIZE,32);
  /*if(threads == 1){
    double aux = GRAPH_SIZE / 16;
    threads = ceil(aux);
  }*/
  threads = 16;
  printf("threads per block %d x %d\n",threads,threads);
  dim3 threadsPerBlock(threads, threads);
  dim3 numBlocks(GRAPH_SIZE / threadsPerBlock.x, GRAPH_SIZE / threadsPerBlock.y); 
  int *dev;
  int size = sizeof(int) * graph_size * graph_size;
  cudaMalloc(&dev, size);
  cudaMemcpy(dev, graph, size, cudaMemcpyHostToDevice);
  for (int k = 0; k < graph_size; k++) {
    gpu_calculate<<<numBlocks, threadsPerBlock, sizeof(int) * threads * 2>>>(k, graph_size, dev, threads);
  }
  cudaMemcpy(output, dev, size, cudaMemcpyDeviceToHost);
  cudaFree(dev);  
}

void floyd_warshall_cpu(const int *graph, int graph_size, int *output) {
  int i, j, k;

  memcpy(output, graph, sizeof(int) * graph_size * graph_size);

  for (k = 0; k < graph_size; k++) {
    for (i = 0; i < graph_size; i++) {
      for (j = 0; j < graph_size; j++) {
        if (D(i, k) + D(k, j) < D(i, j)) {
          D(i, j) = D(i, k) + D(k, j);
        }
      }
    }
  }
}

int main(int argc, char **argv) {
#define TIMER_START() gettimeofday(&tv1, NULL)
#define TIMER_STOP()                                                           \
  gettimeofday(&tv2, NULL);                                                    \
  timersub(&tv2, &tv1, &tv);                                                   \
  time_delta = (float)tv.tv_sec + tv.tv_usec / 1000000.0

  struct timeval tv1, tv2, tv;
  float time_delta;

  int *graph, *output_cpu, *output_gpu;
  int size;

  size = sizeof(int) * GRAPH_SIZE * GRAPH_SIZE;

  graph = (int *)malloc(size);
  assert(graph);

  output_cpu = (int *)malloc(size);
  assert(output_cpu);
  memset(output_cpu, 0, size);

  output_gpu = (int *)malloc(size);
  assert(output_gpu);

  generate_random_graph(graph, GRAPH_SIZE);

  fprintf(stderr, "running on cpu...\n");
  TIMER_START();
  floyd_warshall_cpu(graph, GRAPH_SIZE, output_cpu);
  TIMER_STOP();
  fprintf(stderr, "%f secs\n", time_delta);

  fprintf(stderr, "running on gpu...\n");
  TIMER_START();
  floyd_warshall_gpu(graph, GRAPH_SIZE, output_gpu);
  TIMER_STOP();
  fprintf(stderr, "%f secs\n", time_delta);

  if (memcmp(output_cpu, output_gpu, size) != 0) {
    fprintf(stderr, "FAIL!\n");
  } else {
    /*for (int k = 500; k < 550; k++) {
      printf("cpu:%d gpu:%d origin:%d\n", output_cpu[k], output_gpu[k], graph[k]);
    }*/
    printf("OK\n");
  }

  return 0;
}
