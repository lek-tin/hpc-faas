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

void image_write(char *file_name, std::string text, image_data image);

// loads a .png image, and makes it gray_scale so we can start the edge detection
// this is all on cpu side
image_data img_load(char *file_name) {
    unsigned int width, height;
    byte *rgb;
    lodepng_decode_file(&rgb, &width, &height, file_name, LCT_RGBA, 8);

    byte *gray_scale = new byte[width * height];
    byte *image = rgb;

    for (int i = 0; i < width * height; ++i) {
        int r = *image++;
        int g = *image++;
        int b = *image++;
        int a = *image++;
        gray_scale[i] = 0.3 * r + 0.6 * g + 0.1 * b + 0.5;
    }

    free(rgb);
    image_data gray = image_data(gray_scale, width, height);
    image_write(file_name, "grayscale", gray);
    return gray;
}

// writes the new image as a png image
// gets info from the gpu and uses it to write the edge-detected image
void image_write(char *file_name, std::string text, image_data image) {
    std::string out_file_name = file_name;
    out_file_name = out_file_name.substr(0, out_file_name.rfind("."));
    out_file_name.append("_").append(text).append(".png");
    lodepng_encode_file(out_file_name.c_str(), image.pixels, image.width, image.height, LCT_GREY, 8);
    delete[] image.pixels;
}
