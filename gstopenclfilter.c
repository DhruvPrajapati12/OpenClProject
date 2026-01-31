#ifndef PACKAGE
#define PACKAGE "gstopenclfilter"
#endif

#ifndef VERSION
#define VERSION "1.0"
#endif

#include <gst/gst.h>
#include <gst/video/gstvideofilter.h>
#include <gst/video/video.h>
#include <CL/cl.h>
#include <stdio.h>
#include <string.h>

/* ================= DEBUG ================= */
GST_DEBUG_CATEGORY_STATIC(gst_opencl_filter_debug);
#define GST_CAT_DEFAULT gst_opencl_filter_debug

/* ================= OBJECT ================= */
typedef struct _GstOpenCLFilter {
    GstVideoFilter parent;

    cl_platform_id platform;
    cl_device_id   device;
    cl_context     context;
    cl_command_queue queue;
    cl_program     program;
    cl_kernel      kernel;

    gboolean cl_ready;
    guint64 frame_count;
} GstOpenCLFilter;

typedef struct _GstOpenCLFilterClass {
    GstVideoFilterClass parent_class;
} GstOpenCLFilterClass;

#define GST_TYPE_OPENCL_FILTER (gst_opencl_filter_get_type())
G_DEFINE_TYPE(GstOpenCLFilter, gst_opencl_filter, GST_TYPE_VIDEO_FILTER)

/* ================= CAPS ================= */
static GstStaticPadTemplate sink_template =
GST_STATIC_PAD_TEMPLATE(
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw, format=(string)NV12")
);

static GstStaticPadTemplate src_template =
GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw, format=(string)NV12")
);

/* ================= HELPERS ================= */
static char *load_file(const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    rewind(fp);

    char *buf = g_malloc(size + 1);
    fread(buf, 1, size, fp);
    buf[size] = 0;
    fclose(fp);
    return buf;
}

#define CHECK_CL(err, msg) \
    if ((err) != CL_SUCCESS) { \
        GST_ERROR_OBJECT(self, "%s failed: %d", msg, err); \
        goto cl_error; \
    }

/* ================= OPENCL INIT ================= */
static gboolean
gst_opencl_filter_set_info(GstVideoFilter *filter,
                           GstCaps *incaps, GstVideoInfo *ininfo,
                           GstCaps *outcaps, GstVideoInfo *outinfo)
{
    GstOpenCLFilter *self = (GstOpenCLFilter *)filter;
    cl_int err;

    GST_INFO_OBJECT(self, "Initializing OpenCL");

    err = clGetPlatformIDs(1, &self->platform, NULL);
    CHECK_CL(err, "clGetPlatformIDs");

    err = clGetDeviceIDs(self->platform, CL_DEVICE_TYPE_GPU, 1,
                          &self->device, NULL);
    CHECK_CL(err, "clGetDeviceIDs");

    self->context = clCreateContext(NULL, 1,
                                    &self->device,
                                    NULL, NULL, &err);
    CHECK_CL(err, "clCreateContext");

    self->queue =
        clCreateCommandQueueWithProperties(self->context,
                                           self->device,
                                           NULL, &err);
    CHECK_CL(err, "clCreateCommandQueueWithProperties");

    char *src = load_file("/home/kyoto/dhruv/Oscar/OpenCL/OpenCLProject/nv12_half_left.cl");
    if (!src) {
        GST_ERROR_OBJECT(self, "Failed to load OpenCL kernel file");
        return FALSE;
    }

    self->program =
        clCreateProgramWithSource(self->context, 1,
                                  (const char **)&src,
                                  NULL, &err);
    g_free(src);
    CHECK_CL(err, "clCreateProgramWithSource");

    err = clBuildProgram(self->program, 1,
                         &self->device,
                         NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(self->program, self->device,
                              CL_PROGRAM_BUILD_LOG,
                              sizeof(log), log, NULL);
        GST_ERROR_OBJECT(self, "OpenCL build error:\n%s", log);
        return FALSE;
    }

    self->kernel =
        clCreateKernel(self->program,
                       "nv12_half_left", &err);
    CHECK_CL(err, "clCreateKernel");

    self->cl_ready = TRUE;
    GST_INFO_OBJECT(self, "OpenCL initialized successfully");
    return TRUE;

cl_error:
    GST_ERROR_OBJECT(self, "OpenCL initialization failed");
    return FALSE;
}

/* ================= FRAME PROCESS ================= */
static GstFlowReturn
gst_opencl_filter_transform_frame(GstVideoFilter *filter,
                                  GstVideoFrame *in,
                                  GstVideoFrame *out)
{
    GstOpenCLFilter *self = (GstOpenCLFilter *)filter;
    cl_int err;

    self->frame_count++;

    GST_LOG_OBJECT(self,
        "transform_frame(): frame=%" G_GUINT64_FORMAT
        " pts=%" GST_TIME_FORMAT,
        self->frame_count,
        GST_TIME_ARGS(GST_BUFFER_PTS(in->buffer)));

    /* Always start with a valid frame */
    gst_video_frame_copy(out, in);

    if (!self->cl_ready) {
        GST_WARNING_OBJECT(self, "OpenCL not ready, bypassing");
        return GST_FLOW_OK;
    }

    guint8 *y = GST_VIDEO_FRAME_PLANE_DATA(out, 0);
    int width  = GST_VIDEO_FRAME_WIDTH(out);
    int height = GST_VIDEO_FRAME_HEIGHT(out);
    int stride = GST_VIDEO_FRAME_PLANE_STRIDE(out, 0);

    size_t size = stride * height;

    GST_DEBUG_OBJECT(self,
        "Frame info: %dx%d stride=%d",
        width, height, stride);

    guint8 before = y[0];

    cl_mem ybuf =
        clCreateBuffer(self->context,
                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        size, y, &err);
    CHECK_CL(err, "clCreateBuffer");

    err = clSetKernelArg(self->kernel, 0,
                          sizeof(cl_mem), &ybuf);
    CHECK_CL(err, "clSetKernelArg(0)");

    err = clSetKernelArg(self->kernel, 1,
                          sizeof(int), &width);
    CHECK_CL(err, "clSetKernelArg(1)");

    err = clSetKernelArg(self->kernel, 2,
                          sizeof(int), &height);
    CHECK_CL(err, "clSetKernelArg(2)");

    err = clSetKernelArg(self->kernel, 3,
                          sizeof(int), &stride);
    CHECK_CL(err, "clSetKernelArg(3)");

    size_t global[2] = {
        (size_t)width,
        (size_t)height
    };

    GST_DEBUG_OBJECT(self,
        "Enqueue kernel global=(%zu x %zu)",
        global[0], global[1]);

    err = clEnqueueNDRangeKernel(self->queue,
                                 self->kernel,
                                 2, NULL,
                                 global, NULL,
                                 0, NULL, NULL);
    CHECK_CL(err, "clEnqueueNDRangeKernel");

    err = clFinish(self->queue);
    CHECK_CL(err, "clFinish");

    err = clEnqueueReadBuffer(self->queue,
                              ybuf, CL_TRUE,
                              0, size, y,
                              0, NULL, NULL);
    CHECK_CL(err, "clEnqueueReadBuffer");

    guint8 after = y[0];

    GST_LOG_OBJECT(self,
        "Y[0] before=%u after=%u",
        before, after);

    clReleaseMemObject(ybuf);
    return GST_FLOW_OK;

cl_error:
    GST_ERROR_OBJECT(self, "OpenCL execution failed");
    return GST_FLOW_ERROR;
}

/* ================= FINALIZE ================= */
static void
gst_opencl_filter_finalize(GObject *object)
{
    GstOpenCLFilter *self = (GstOpenCLFilter *)object;

    GST_INFO_OBJECT(self, "Finalizing OpenCL");

    if (self->kernel)  clReleaseKernel(self->kernel);
    if (self->program) clReleaseProgram(self->program);
    if (self->queue)   clReleaseCommandQueue(self->queue);
    if (self->context) clReleaseContext(self->context);

    G_OBJECT_CLASS(gst_opencl_filter_parent_class)->finalize(object);
}

/* ================= INIT ================= */
static void
gst_opencl_filter_init(GstOpenCLFilter *self)
{
    self->cl_ready = FALSE;
    self->frame_count = 0;
}

static void
gst_opencl_filter_class_init(GstOpenCLFilterClass *klass)
{
    GstElementClass *eclass = GST_ELEMENT_CLASS(klass);
    GstVideoFilterClass *vclass = GST_VIDEO_FILTER_CLASS(klass);
    GObjectClass *gclass = G_OBJECT_CLASS(klass);

    GST_DEBUG_CATEGORY_INIT(gst_opencl_filter_debug,
                            "openclfilter", 0,
                            "OpenCL NV12 video filter");

    gclass->finalize = gst_opencl_filter_finalize;
    vclass->set_info =
        GST_DEBUG_FUNCPTR(gst_opencl_filter_set_info);
    vclass->transform_frame =
        GST_DEBUG_FUNCPTR(gst_opencl_filter_transform_frame);

    gst_element_class_add_pad_template(eclass,
        gst_static_pad_template_get(&sink_template));
    gst_element_class_add_pad_template(eclass,
        gst_static_pad_template_get(&src_template));

    gst_element_class_set_static_metadata(
        eclass,
        "OpenCL NV12 Filter",
        "Filter/Video",
        "Applies OpenCL processing on NV12 video",
        "Custom");
}

/* ================= PLUGIN ================= */
static gboolean
plugin_init(GstPlugin *plugin)
{
    return gst_element_register(plugin,
                                "openclfilter",
                                GST_RANK_NONE,
                                GST_TYPE_OPENCL_FILTER);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    openclfilter,
    "OpenCL NV12 filter",
    plugin_init,
    VERSION,
    "LGPL",
    PACKAGE,
    PACKAGE
)