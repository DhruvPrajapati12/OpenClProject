__kernel void invert_image(
    __global uchar4* input,
    __global uchar4* output,
    int width,
    int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int idx = y * width + x;

    uchar4 pixel = input[idx];

    pixel.x = 255 - pixel.x;
    pixel.y = 255 - pixel.y;
    pixel.z = 255 - pixel.z;

    output[idx] = pixel;
}