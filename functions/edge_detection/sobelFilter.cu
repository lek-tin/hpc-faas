#include <thread>
#include <chrono>
#include <time.h>
#include <iostream>
#include <math.h>
#include "load_image.cpp"

#define GRIDVAL 20.0

/************************************************************************************************
 * void sobel_gpu(const byte*, byte*, uint, uint);
 * - This function runs on the GPU, it works on a 2D grid giving the current x, y pair being worked
 * - on, the const byte* is the original image being processed and the second byte* is the image
 * - being created using the sobel filter. This function runs through a given x, y pair and uses 
 * - a sobel filter to find whether or not the current pixel is an edge, the more of an edge it is
 * - the higher the value returned will be
 * 
 * Inputs: const byte* orig : the original image being evaluated
 *                byte* cpu : the image being created using the sobel filter
 *               uint width : the width of the image
 *              uint height : the height of the image
 * 
 ***********************************************************************************************/
__global__ void sobel_gpu(const byte *orig, byte *cpu, const unsigned int width, const unsigned int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float dx, dy;

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {

        dx = (-1 * orig[(y - 1) * width + (x - 1)]) +
             (-2 * orig[y * width + (x - 1)]) +
             (-1 * orig[(y + 1) * width + (x - 1)]) +
             (orig[(y - 1) * width + (x + 1)]) +
             (2 * orig[y * width + (x + 1)]) +
             (orig[(y + 1) * width + (x + 1)]);

        dy = (orig[(y - 1) * width + (x - 1)]) +
             (2 * orig[(y - 1) * width + x]) +
             (orig[(y - 1) * width + (x + 1)]) +
             (-1 * orig[(y + 1) * width + (x - 1)]) +
             (-2 * orig[(y + 1) * width + x]) +
             (-1 * orig[(y + 1) * width + (x + 1)]);

        cpu[y * width + x] = sqrt((dx * dx) + (dy * dy));
    }
}

//__global__ void gray_gpu(const byte* orig, byte* cpu, const unsigned int width, const unsigned int height) {
//    int col = threadIdx.x + blockIdx.x * blockDim.x;
//    int row = threadIdx.y + blockIdx.y * blockDim.y;
//
////    int offset_out = row * width;  // 1 color per pixel
//    int offset = row * width;   // 3 colors per pixel
//    const byte *pixel = &orig[offset + col];
//    cpu[offset + col] = pixel[0] * 0.0722f + // B
//    pixel[1] * 0.7152f + // G
//    pixel[2] * 0.2126f;  // R
//}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        printf("%s: Invalid number of command line arguments. Exiting program\n", argv[0]);
        printf("Usage: %s [image.png]", argv[0]);
        return 1;
    }

    /** Load our img and allocate space for our modified images **/
    image_data origImg = img_load(argv[1]);
    image_data gpuImg(new byte[origImg.width * origImg.height], origImg.width, origImg.height);
//    image_data grayImg(new byte[origImg.width * origImg.height], origImg.width, origImg.height); //change


    /** Allocate space in the GPU for our original img, new img, and dimensions **/
    byte *gpu_orig, *gpu_sobel, *gpu_gray; //change
    cudaMalloc((void **) &gpu_orig, (origImg.width * origImg.height));
    cudaMalloc((void **) &gpu_sobel, (origImg.width * origImg.height));
//    cudaMalloc((void **) &gpu_gray, (origImg.width * origImg.height)); //change

    /** Transfer over the memory from host to device and memset the sobel array to 0s **/
    cudaMemcpy(gpu_orig, origImg.pixels, (origImg.width * origImg.height), cudaMemcpyHostToDevice);
    cudaMemset(gpu_sobel, 0, (origImg.width * origImg.height));
//    cudaMemset(gpu_gray, 0, (origImg.width * origImg.height)); //change

    /** set up the dim3's for the gpu to use as arguments (threads per block & num of blocks)**/
    dim3 threadsPerBlock(GRIDVAL, GRIDVAL, 1);
    dim3 numBlocks(ceil(origImg.width / GRIDVAL), ceil(origImg.height / GRIDVAL), 1);

    /** Run the sobel filter using the GPU **/
    auto c = std::chrono::system_clock::now();
    sobel_gpu << < numBlocks, threadsPerBlock >> > (gpu_orig, gpu_sobel, origImg.width, origImg.height);
//    gray_gpu << < numBlocks, threadsPerBlock >> > (gpu_orig, gpu_gray, origImg.width, origImg.height); //change

    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code

    if (cudaerror != cudaSuccess)
        fprintf(stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName(cudaerror)); // if error, output error

        std::chrono::duration<double> time_gpu = std::chrono::system_clock::now() - c;

    /** Copy data back to CPU from GPU **/
    cudaMemcpy(gpuImg.pixels, gpu_sobel, (origImg.width * origImg.height), cudaMemcpyDeviceToHost);
//    cudaMemcpy(grayImg.pixels, gpu_gray, (origImg.width * origImg.height), cudaMemcpyDeviceToHost); //change

    /** Output runtimes of each method of sobel filtering **/
    printf("\nProcessing %s: %d rows x %d columns\n", argv[1], origImg.height, origImg.width);
    printf("CUDA execution time   = %*.1f msec\n", 5, 1000 * time_gpu.count());
    printf("\n");

    /** Output the images of each sobel filter with an appropriate string appended to the original image name **/
    image_write(argv[1], "gpu", gpuImg);
//    image_write(argv[1], "gpugray", grayImg); //change

    /** Free any memory leftover.. gpuImig, cpuImg, and ompImg get their pixels free'd while writing **/
    cudaFree(gpu_orig);
    cudaFree(gpu_sobel);
    return 0;
}