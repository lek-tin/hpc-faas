#include "lodepng.h"
#include <string>

typedef unsigned char byte;

struct image_data {
    image_data(byte *pix = nullptr, unsigned int w = 0, unsigned int h = 0) : pixels(pix), width(w), height(h) {
    };
    byte *pixels;
    unsigned int width;
    unsigned int height;
};

void image_write(char *filename, std::string appendTxt, image_data img);

image_data img_load(char *filename) {
    unsigned int width, height;
    byte *rgb;
    unsigned error = lodepng_decode_file(&rgb, &width, &height, filename, LCT_RGBA, 8);
    if (error) {
        printf("LodePNG had an error during file processing. Exiting program.\n");
        printf("Error code: %u: %s\n", error, lodepng_error_text(error));
        exit(2);
    }
    byte *grayscale = new byte[width * height];
    byte *img = rgb;
    for (int i = 0; i < width * height; ++i) {
        int r = *img++;
        int g = *img++;
        int b = *img++;
        int a = *img++;
        grayscale[i] = 0.3 * r + 0.6 * g + 0.1 * b + 0.5;
    }
    free(rgb);
    image_data gray = image_data(grayscale, width, height);
    image_write(filename, "grayscale", gray);
    return gray;//image_data(grayscale, width, height);
}

/************************************************************************************************
 * void image_write(char*, std::string, image_data)
 * - This function takes a filename as a char array, a string of text, and a structure containing
 * - the image's pixel info, width, and height. The function will take the original filename,
 * - remove the .png ending and append text before re-adding the .png extension, then lodepng is
 * - called to encode the pixel data into a png, before the function leaves the pixel data from the
 * - structure is freed as it is not needed anymore.
 * Inputs:        char* filename : the filename of the original image file
 *         std::string appendTxt : the text to append after the original image filename
 *                   image_data img : the structure containing the image's pixel and dimensions
 ***********************************************************************************************/
void image_write(char *filename, std::string appendTxt, image_data img) {
    std::string newName = filename;
    newName = newName.substr(0, newName.rfind("."));
    newName.append("_").append(appendTxt).append(".png");
    unsigned error = lodepng_encode_file(newName.c_str(), img.pixels, img.width, img.height, LCT_GREY, 8);
    if (error) {
        printf("LodePNG had an error during file writing. Exiting program.\n");
        printf("Error code: %u: %s\n", error, lodepng_error_text(error));
        exit(3);
    }
    delete[] img.pixels;
}
