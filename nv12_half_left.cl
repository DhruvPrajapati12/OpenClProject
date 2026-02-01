__kernel void nv12_half_left(__global uchar *y,
                             int width,
                             int height,
                             int stride)
{
    int x = get_global_id(0);
    int yid = get_global_id(1);

    if (x >= width || yid >= height)
        return;

    int off = yid * stride + x;

    if (x < width / 2)
        y[off] = y[off] >> 1;
}

__kernel void nv12_invert_left(__global uchar *y,
                              int width,
                              int height,
                              int stride)
{
    int x   = get_global_id(0);
    int yid = get_global_id(1);

    if (x >= width || yid >= height)
        return;

    int off = yid * stride + x;

    if (x < width / 2)
        y[off] = (uchar)(255 - y[off]);
}

__kernel void nv12_bright_left(__global uchar *y,
                               int width,
                               int height,
                               int stride)
{
    int x   = get_global_id(0);
    int yid = get_global_id(1);

    if (x >= width || yid >= height)
        return;

    int off = yid * stride + x;

    if (x < width / 2) {
        int val = y[off] + 40;
        y[off] = (uchar)(val > 255 ? 255 : val);
    }
}
