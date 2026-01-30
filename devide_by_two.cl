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