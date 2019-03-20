#include "load_image.cpp"

#define TILE_SIZE 16

// This function does edge detection on the GPU
// it works on .png images that have been loaded in with the other file "lodepng.cpp"
// it uses the sobel filtering method of edge detection
__global__ void edge_detect_gpu(const byte *source_img, byte *tar_img, const unsigned int w, const unsigned int h) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float dx;
    float dy;

    if (x > 0 && y > 0 && x < w - 1 && y < h - 1) {

        dx = (-1 * source_img[(y - 1) * w + (x - 1)]) +
             (-2 * source_img[y * w + (x - 1)]) +
             (-1 * source_img[(y + 1) * w + (x - 1)]) +
             (source_img[(y - 1) * w + (x + 1)]) +
             (2 * source_img[y * w + (x + 1)]) +
             (source_img[(y + 1) * w + (x + 1)]);

        dy = (source_img[(y - 1) * w + (x - 1)]) +
             (2 * source_img[(y - 1) * w + x]) +
             (source_img[(y - 1) * w + (x + 1)]) +
             (-1 * source_img[(y + 1) * w + (x - 1)]) +
             (-2 * source_img[(y + 1) * w + x]) +
             (-1 * source_img[(y + 1) * w + (x + 1)]);

        tar_img[y * w + x] = sqrt((dx * dx) + (dy * dy));
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("%s: Invalid number of command line arguments, must have 2 arguments.\n", argv[0]);
        return 1;
    }

    // Loading .png image
    image_data original_img = img_load(argv[1]);
    // allocating space for the new images we will create
    image_data gpu_image(new byte[original_img.width * original_img.height], original_img.width, original_img.height);

    // allocating space for the images in the GPU (original_img image and the one we will create)
    byte *gpu_source;
    byte *gpu_edge;
    cudaMalloc((void **) &gpu_source, (original_img.width * original_img.height));
    cudaMalloc((void **) &gpu_edge, (original_img.width * original_img.height));

    // Memcpy to transfer mem from host to device
    cudaMemcpy(gpu_source, original_img.pixels, (original_img.width * original_img.height), cudaMemcpyHostToDevice);
    cudaMemset(gpu_edge, 0, (original_img.width * original_img.height));

    // setting up the dimensions for block and grid sizes
    dim3 dim_block(TILE_SIZE, TILE_SIZE);
    dim3 dim_grid(original_img.width - 1 / TILE_SIZE + 1, original_img.height - 1 / TILE_SIZE + 1);

    // running edge detection program on the GPU
    edge_detect_gpu << < dim_grid, dim_block >> > (gpu_source, gpu_edge, original_img.width, original_img.height);

    // synchronize threads on GPU
    cudaDeviceSynchronize();

    // use Memcpy to copy data back to the CPU
    cudaMemcpy(gpu_image.pixels, gpu_edge, (original_img.width * original_img.height), cudaMemcpyDeviceToHost);

    // print image info
    printf("\nNow doing edge detection on GPU.\n");
    printf("Image info: [ %s ] %d rows x %d columns\n\n", argv[1], original_img.height, original_img.width);

    // create the image and change name
    image_write(argv[1], "gpu_edge", gpu_image);

    // free memory on the GPU with cuda free
    cudaFree(gpu_source);
    cudaFree(gpu_edge);
    return 0;
}