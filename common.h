__constant float2 OPTICAL_CENTER = {U0, V0};
__constant float2 FOCAL_LENGTH = {FU, FV};
__constant float2 MAX_SRC_COORDINATE = {(float)(SRC_COLS - 1), (float)(SRC_ROWS - 1)};

inline float3 unproject(const float4 params, const float2 p_pix) {
  const float2 m = (p_pix - /*(U0, V0) = */ params.xy) / /*(FU, FV)*/ params.zw;
  return (float3)(m, 1.0f);
}

#ifdef DOUBLE_SPHERE
inline float d2_(const float d1, float3 pcam) {
  pcam.z += XI * d1;
  return length(pcam);
}

inline float kappa_(const float3 pcam) {
  const float d1 = length(pcam);
  // return ALPHA * d2_(d1, pcam) + (1.0f - ALPHA) * (XI * d1 + pcam.z);
  return ALPHA * d2_(d1, pcam) + ONE_MINUS_ALPHA_TIMES_XI * d1 + ONE_MINUS_ALPHA * pcam.z;
}
#elif defined EXTENDED_UNIFIED
inline float d_(float3 pcam) {
  pcam.xy *= SQRT_BETA;
  return length(pcam);
}

inline float kappa_(const float3 pcam) { return ALPHA * d_(pcam) + ONE_MINUS_ALPHA * pcam.z; }
#endif

inline float2 project(const float3 pcam) { return OPTICAL_CENTER + FOCAL_LENGTH * pcam.xy / kappa_(pcam); }
#ifdef YUV
// Returns UVYW (UVY - warped image, W - warped weight)
inline float4 interpolate(__global const uchar* restrict image, __global const float* restrict weights_in, float2 xy) {
  // Clamp coordinate to edge.
#ifndef BORDER_CHECK
  xy = clamp(xy, (float2)0.0f, MAX_SRC_COORDINATE);
#endif

  float2 xy_floor = floor(xy);
  int2 xy_floor_i = convert_int2(xy_floor);
  int offset_t = xy_floor_i.y * SRC_COLS + xy_floor_i.x;
  int offset_b = offset_t + SRC_COLS;

  float2 alpha_beta = xy - xy_floor;

  // Load top and bottom values (2 bytes per pixel!).
  uchar4 tu = vload4(0, image + offset_t * 2);
  uchar4 bu = vload4(0, image + offset_b * 2);

  // Load top and bottom values.
  float2 tw = vload2(0, weights_in + offset_t);
  float2 bw = vload2(0, weights_in + offset_b);

  // Cast to float.
  float4 top = convert_float4(tu);
  float4 bottom = convert_float4(bu);

  float2 t = top.s13;
  float2 b = bottom.s13;

  float2 uv = top.s02 + (bottom.s02 - top.s02) * alpha_beta.y;
  // Re-order u and v to always have u first.
  uv = ((xy_floor_i.x & 1) == 0) ? uv : uv.yx;

  float alpha_times_beta = alpha_beta.y * alpha_beta.x;
  float4 ret_val = {
    uv,
    alpha_times_beta * (b.y - b.x - t.y + t.x) + alpha_beta.y * (b.x - t.x) +
        alpha_beta.x * (t.y - t.x) + t.x,
    alpha_times_beta * (bw.y - bw.x - tw.y + tw.x) +
        alpha_beta.y * (bw.x - tw.x) + alpha_beta.x * (tw.y - tw.x) + tw.x
  };

  return ret_val;
}
#else  // GRAYSCALE
// Returns IW (I - warped image, W - warped weight)
inline float2 interpolate(__global uchar* image, __global float* weights_in, float2 xy) {
  // Clamp coordinate to edge.
#ifndef BORDER_CHECK
  xy = clamp(xy, (float2)0.0f, MAX_SRC_COORDINATE);
#endif

  const float2 xy_floor = floor(xy);
  const int2 xy_floor_i = {(int)xy_floor.x, (int)xy_floor.y};

  const float2 alpha_beta = xy - xy_floor;

  // Load top and bottom values (2 at once).
  const uchar2 tu = vload2(0, image + xy_floor_i.y * SRC_COLS + xy_floor_i.x);
  const uchar2 bu = vload2(0, image + (xy_floor_i.y + 1) * SRC_COLS + xy_floor_i.x);

  const float2 tw = vload2(0, weights_in + offset_t);
  const float2 bw = vload2(0, weights_in + offset_b);

  // Cast to float.
  const float2 t = {(float)tu.x, (float)tu.y};
  const float2 b = {(float)bu.x, (float)bu.y};

  const float alpha_times_beta = alpha_beta.y * alpha_beta.x;
  const float2 ret_val = {
    alpha_times_beta * (b.y - b.x - t.y + t.x) + alpha_beta.y * (b.x - t.x) +
        alpha_beta.x * (t.y - t.x) + t.x,
    alpha_times_beta * (bw.y - bw.x - tw.y + tw.x) + alpha_beta.y * (bw.x - tw.x) +
        alpha_beta.x * (tw.y - tw.x) + tw.x;
  return ret_val;
}
#endif

