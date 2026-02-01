#ifndef PACKAGE
#define PACKAGE "gstopenclfilter"
#endif

#ifndef VERSION
#define VERSION "1.0"
#endif

#define CL_TARGET_OPENCL_VERSION 300 // Targets OpenCL 3.0

#include <gst/gst.h>
#include <gst/video/gstvideofilter.h>
#include <gst/video/video.h>
#include <CL/cl.h>
#include <string.h>
#include <stdio.h>


/* ================= DEBUG ================= */
GST_DEBUG_CATEGORY_STATIC(gst_opencl_filter_debug);
#define GST_CAT_DEFAULT gst_opencl_filter_debug

/* ================= OBJECT ================= */
typedef struct _GstOpenCLFilter {
    GstVideoFilter parent;

    /* OpenCL */
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    cl_mem ybuf;
    size_t buf_size;

    /* Video info */
    gint width;
    gint height;
    gint stride;

    gboolean cl_ready;
    guint64 frame_count;

    /* Property */
    gchar *kernel_file;

} GstOpenCLFilter;

typedef struct _GstOpenCLFilterClass {
    GstVideoFilterClass parent_class;
} GstOpenCLFilterClass;

enum {
    PROP_0,
    PROP_KERNEL_FILE,
};

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
        GST_ERROR_OBJECT(self, "%s failed (%d)", msg, err); \
        goto error; \
    }

static gchar *
load_kernel_file(const gchar *path)
{
    gchar *data = NULL;
    gsize size = 0;

    if (!g_file_get_contents(path, &data, &size, NULL))
        return NULL;

    return data;
}

/* ================= OPENCL INIT ================= */
static gboolean
gst_opencl_filter_set_info(GstVideoFilter *filter,
                           GstCaps *incaps, GstVideoInfo *ininfo,
                           GstCaps *outcaps, GstVideoInfo *outinfo)
{
    GstOpenCLFilter *self = (GstOpenCLFilter *)filter;

    if (!self->kernel_file || !g_file_test(self->kernel_file, G_FILE_TEST_EXISTS)) {
        GST_INFO_OBJECT(self,
            "No kernel-file provided, running in bypass mode");
        self->cl_ready = FALSE;
        return TRUE;
    }

    cl_int err;

    GST_INFO_OBJECT(self, "Initializing OpenCL");

    err = clGetPlatformIDs(1, &self->platform, NULL);
    CHECK_CL(err, "clGetPlatformIDs");

    err = clGetDeviceIDs(self->platform, CL_DEVICE_TYPE_GPU,
                          1, &self->device, NULL);
    CHECK_CL(err, "clGetDeviceIDs");

    self->context = clCreateContext(NULL, 1,
                                    &self->device,
                                    NULL, NULL, &err);
    CHECK_CL(err, "clCreateContext");

    self->queue = clCreateCommandQueueWithProperties(
                    self->context, self->device, NULL, &err);
    CHECK_CL(err, "clCreateCommandQueueWithProperties");

    // char *kernel_src = load_file("/home/kyoto/dhruv/Oscar/OpenCL/OpenCLProject/nv12_half_left.cl");
    // if (!kernel_src)
    // {
    //     GST_ERROR_OBJECT(self, "Failed to load OpenCL kernel file");
    //     return FALSE;
    // }

    gchar *kernel_src = load_kernel_file(self->kernel_file);
    if (!kernel_src) {
        GST_ERROR_OBJECT(self,
            "Failed to load kernel file: %s", self->kernel_file);
        self->cl_ready = FALSE;
        return TRUE; /* bypass */
    }

    self->program = clCreateProgramWithSource(
                        self->context, 1,
                        (const char **)&kernel_src, NULL, &err);
    CHECK_CL(err, "clCreateProgramWithSource");

    g_free(kernel_src);

    err = clBuildProgram(self->program, 1,
                         &self->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(self->program, self->device,
                              CL_PROGRAM_BUILD_LOG,
                              sizeof(log), log, NULL);
        GST_ERROR_OBJECT(self, "OpenCL build error:\n%s", log);
        goto error;
    }

    self->kernel = clCreateKernel(
                        self->program, "nv12_half_left", &err);
    CHECK_CL(err, "clCreateKernel");

    self->cl_ready = TRUE;
    return TRUE;

error:
    GST_WARNING_OBJECT(self,
        "OpenCL unavailable, running in bypass mode");
    self->cl_ready = FALSE;
    return TRUE; /* Allow pipeline to continue */
}

/* ================= FRAME PROCESS ================= */
static GstFlowReturn
gst_opencl_filter_transform_frame(GstVideoFilter *filter,
                                  GstVideoFrame *in,
                                  GstVideoFrame *out)
{
    GstOpenCLFilter *self = (GstOpenCLFilter *)filter;
    cl_int err;

    gst_video_frame_copy(out, in);
    self->frame_count++;

    GST_LOG_OBJECT(self, "transform_frame(): frame=%" G_GUINT64_FORMAT " pts=%" GST_TIME_FORMAT, self->frame_count, GST_TIME_ARGS(GST_BUFFER_PTS(in->buffer)));

    if (!self->cl_ready) {
        GST_WARNING_OBJECT(self, "OpenCL not ready, bypassing");
        return GST_FLOW_OK;
    }
    guint8 *y = GST_VIDEO_FRAME_PLANE_DATA(out, 0);
    gint width  = GST_VIDEO_FRAME_WIDTH(out);
    gint height = GST_VIDEO_FRAME_HEIGHT(out);
    gint stride = GST_VIDEO_FRAME_PLANE_STRIDE(out, 0);

    size_t size = stride * height;

    GST_DEBUG_OBJECT(self, "Frame info: %dx%d stride=%d", width, height, stride);
    guint8 before = y[0];

    /* Reallocate buffer if format changed */
    if (!self->ybuf || size != self->buf_size) {
        if (self->ybuf)
            clReleaseMemObject(self->ybuf);

        self->ybuf = clCreateBuffer(self->context,
                                    CL_MEM_READ_WRITE,
                                    size, NULL, &err);
        CHECK_CL(err, "clCreateBuffer");

        self->buf_size = size;
        self->width = width;
        self->height = height;
        self->stride = stride;
    }

    err = clEnqueueWriteBuffer(self->queue,
                               self->ybuf, CL_TRUE,
                               0, size, y,
                               0, NULL, NULL);
    CHECK_CL(err, "clEnqueueWriteBuffer");

    err = clSetKernelArg(self->kernel, 0, sizeof(cl_mem), &self->ybuf);
    CHECK_CL(err, "clSetKernelArg(0)");
    err = clSetKernelArg(self->kernel, 1, sizeof(int), &width);
    CHECK_CL(err, "clSetKernelArg(1)");
    err = clSetKernelArg(self->kernel, 2, sizeof(int), &height);
    CHECK_CL(err, "clSetKernelArg(2)");
    err = clSetKernelArg(self->kernel, 3, sizeof(int), &stride);
    CHECK_CL(err, "clSetKernelArg(3)");

    // clSetKernelArg(self->kernel, 4, sizeof(char), "10");
    // CHECK_CL(err, "clSetKernelArg(4)");

    size_t global[2] = { width, height };

    GST_DEBUG_OBJECT(self, "Enqueue kernel global=(%zu x %zu)", global[0], global[1]);

    err = clEnqueueNDRangeKernel(self->queue,
                                 self->kernel,
                                 2, NULL,
                                 global, NULL,
                                 0, NULL, NULL);
    CHECK_CL(err, "clEnqueueNDRangeKernel");

    err = clFinish(self->queue);
    CHECK_CL(err, "clFinish");

    err = clEnqueueReadBuffer(self->queue,
                              self->ybuf, CL_TRUE,
                              0, size, y,
                              0, NULL, NULL);
    CHECK_CL(err, "clEnqueueReadBuffer");

    guint8 after = y[0];
    GST_LOG_OBJECT(self, "Y[0] before=%u after=%u", before, after);

    return GST_FLOW_OK;
error:
    GST_ERROR_OBJECT(self, "OpenCL execution failed");
    return GST_FLOW_ERROR;
}

static void
gst_opencl_filter_set_property(GObject *object,
                               guint prop_id,
                               const GValue *value,
                               GParamSpec *pspec)
{
    GstOpenCLFilter *self = (GstOpenCLFilter *)object;

    switch (prop_id) {
    case PROP_KERNEL_FILE:
        g_free(self->kernel_file);
        self->kernel_file = g_value_dup_string(value);

        GST_INFO_OBJECT(self,
            "kernel-file set to: %s",
            self->kernel_file ? self->kernel_file : "(null)");

        /* Force re-init on next set_info */
        self->cl_ready = FALSE;
        break;

    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void
gst_opencl_filter_get_property(GObject *object,
                               guint prop_id,
                               GValue *value,
                               GParamSpec *pspec)
{
    GstOpenCLFilter *self = (GstOpenCLFilter *)object;

    switch (prop_id) {
    case PROP_KERNEL_FILE:
        g_value_set_string(value, self->kernel_file);
        break;

    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

/* ================= FINALIZE ================= */
static void
gst_opencl_filter_finalize(GObject *object)
{
    GstOpenCLFilter *self = (GstOpenCLFilter *)object;

    if (self->ybuf)    clReleaseMemObject(self->ybuf);
    if (self->kernel)  clReleaseKernel(self->kernel);
    if (self->program) clReleaseProgram(self->program);
    if (self->queue)   clReleaseCommandQueue(self->queue);
    if (self->context) clReleaseContext(self->context);

    g_clear_pointer(&self->kernel_file, g_free);
    G_OBJECT_CLASS(gst_opencl_filter_parent_class)->finalize(object);
}

/* ================= INIT ================= */
static void
gst_opencl_filter_init(GstOpenCLFilter *self)
{
    self->cl_ready = FALSE;
    self->frame_count = 0;
    self->ybuf = NULL;
    self->buf_size = 0;
    self->kernel_file = NULL;
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
    vclass->set_info = GST_DEBUG_FUNCPTR(gst_opencl_filter_set_info);
    vclass->transform_frame =
        GST_DEBUG_FUNCPTR(gst_opencl_filter_transform_frame);

    gst_element_class_add_pad_template(
        eclass, gst_static_pad_template_get(&sink_template));
    gst_element_class_add_pad_template(
        eclass, gst_static_pad_template_get(&src_template));

    gst_element_class_set_static_metadata(
        eclass,
        "OpenCL NV12 Filter",
        "Filter/Video",
        "Applies OpenCL processing on NV12 video",
        "Dhruv Prajapati");

    gclass->set_property = gst_opencl_filter_set_property;
    gclass->get_property = gst_opencl_filter_get_property;

    g_object_class_install_property(
        gclass,
        PROP_KERNEL_FILE,
        g_param_spec_string(
            "kernel-file",
            "OpenCL kernel file",
            "Path to OpenCL kernel file (.cl). "
            "If not set, filter runs in bypass mode.",
            NULL,  /* default */
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

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

