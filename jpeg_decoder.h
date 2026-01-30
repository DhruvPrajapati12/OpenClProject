#include <jpeglib.h>
#include <stdlib.h>
#include <stdio.h>

unsigned char *load_jpeg_rgba(const char *filename,
                              int *width, int *height)
{
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *fp = fopen(filename, "rb");

    if (!fp) return NULL;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, fp);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    *width  = cinfo.output_width;
    *height = cinfo.output_height;

    unsigned char *rgb =
        malloc((*width) * (*height) * 3);

    unsigned char *rgba =
        malloc((*width) * (*height) * 4);

    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char *row =
            rgb + cinfo.output_scanline * (*width) * 3;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(fp);

    /* RGB → RGBA */
    for (int i = 0; i < (*width) * (*height); i++) {
        rgba[i*4+0] = rgb[i*3+0];
        rgba[i*4+1] = rgb[i*3+1];
        rgba[i*4+2] = rgb[i*3+2];
        rgba[i*4+3] = 255;
    }

    free(rgb);
    return rgba;
}

void save_ppm(const char *filename,
              unsigned char *rgba,
              int width, int height)
{
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P6\n%d %d\n255\n", width, height);

    for (int i = 0; i < width * height; i++) {
        fwrite(&rgba[i*4], 1, 3, fp); // RGB only
    }

    fclose(fp);
}
