__kernel void devide_by_two(__global uchar4 *img,
                            int width,
                            int total_pixels)
{
    int id = get_global_id(0);

    if (id >= total_pixels)
        return;

    int x = id % width;

    if (x < width / 2) {
        img[id].x >>= 1;
        img[id].y >>= 1;
        img[id].z >>= 1;
    }
}

__kernel void increase_brightness(__global uchar4 *img,
                                  int total_pixels,
                                  uchar value)
{
    int id = get_global_id(0);

    if (id >= total_pixels) {
        return;
    }

    uchar4 p = img[id];

    p.x = min((int)p.x + value, 255);
    p.y = min((int)p.y + value, 255);
    p.z = min((int)p.z + value, 255);

    img[id] = p;
}

__kernel void grayscale(__global uchar4 *img,
                         int total_pixels)
{
    int id = get_global_id(0);

    if (id >= total_pixels)
        return;

    uchar4 p = img[id];

    uchar gray = (uchar)(
        0.299f * p.x +
        0.587f * p.y +
        0.114f * p.z
    );

    p.x = gray;
    p.y = gray;
    p.z = gray;

    img[id] = p;
}

__kernel void invert_colors(__global uchar4 *img,
                             int total_pixels)
{
    int id = get_global_id(0);

    if (id >= total_pixels)
        return;

    img[id].x = 255 - img[id].x;
    img[id].y = 255 - img[id].y;
    img[id].z = 255 - img[id].z;
}


__kernel void left_half_grayscale(__global uchar4 *img,
                                   int width,
                                   int total_pixels)
{
    int id = get_global_id(0);

    if (id >= total_pixels)
        return;

    int x = id % width;

    if (x < width / 2) {
        uchar4 p = img[id];
        uchar gray = (p.x + p.y + p.z) / 3;

        p.x = gray;
        p.y = gray;
        p.z = gray;

        img[id] = p;
    }
}