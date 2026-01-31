#pragma once
#include <gst/video/gstvideofilter.h>
#include <CL/cl.h>

G_BEGIN_DECLS

#define GST_TYPE_OPENCL_FILTER (gst_opencl_filter_get_type())
G_DECLARE_FINAL_TYPE(GstOpenCLFilter,
                     gst_opencl_filter,
                     GST, OPENCL_FILTER,
                     GstVideoFilter)

struct _GstOpenCLFilter {
    GstVideoFilter parent;

    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
};

G_END_DECLS
