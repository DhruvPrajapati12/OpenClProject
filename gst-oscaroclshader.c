/*
 * gst-oscaroclshader.c
 *
 * A minimal GstBaseTransform video filter
 * Accepts NV12 video/x-raw
 *
 */

#define CL_TARGET_OPENCL_VERSION 300 // Targets OpenCL 3.0

#include <gst/gst.h>
#include <gst/video/gstvideofilter.h>
#include <gst/video/video.h>
#include <CL/cl.h>
#include <stdio.h>

#ifndef PACKAGE
#define PACKAGE "oscaroclshader"
#endif

#ifndef VERSION
#define VERSION "1.0"
#endif

#define NUM_BUFFERS 2

/* Debug category for GstOCLShader logging. */
GST_DEBUG_CATEGORY_STATIC(gst_ocl_shader_debug);
#define GST_CAT_DEFAULT gst_ocl_shader_debug

/* ================= OBJECT ================= */
typedef struct _GstOCLShader {
    GstVideoFilter parent;

    /* OpenCL */
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    /* Double-buffered GPU memory */
    cl_mem ybuf[NUM_BUFFERS];
    size_t buf_size;

    /* Events per buffer */
    cl_event write_evt[NUM_BUFFERS];
    cl_event kernel_evt[NUM_BUFFERS];
    cl_event read_evt[NUM_BUFFERS];

    /* Video info */
    gint width;
    gint height;
    gint stride;

    gboolean cl_ready;
    guint64 frame_count;

    /* OpenCL Property */
    gchar *kernel_file;
    gchar *kernel_func;

} GstOCLShader;

/* Class definition for the GstOCLShader element. */
typedef struct _GstOCLShaderClass {
    GstVideoFilterClass parent_class;
} GstOCLShaderClass;

/* Property identifiers used to configure the OpenCL kernel file and function. */
enum {
    PROP_0,
    PROP_KERNEL_FILE,
    PROP_KERNEL_FUNC,
};

/* GObject type macro for the GstOCLShader element. */
#define GST_TYPE_OCL_SHADER (gst_ocl_shader_get_type())
/* Register GstOCLShader as a GstVideoFilter subclass with the type system. */
G_DEFINE_TYPE(GstOCLShader, gst_ocl_shader, GST_TYPE_VIDEO_FILTER)

/* Sink Pad supports NV12 */
static GstStaticPadTemplate sink_template =
GST_STATIC_PAD_TEMPLATE(
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        "video/x-raw, "
        "format = (string) NV12, "
        "width = (int) [ 1, MAX ], "
        "height = (int) [ 1, MAX ], "
        "framerate = (fraction) [ 0/1, MAX ]"
    )
);

/* Source Pad supports NV12 */
static GstStaticPadTemplate src_template =
GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        "video/x-raw, "
        "format = (string) NV12, "
        "width = (int) [ 1, MAX ], "
        "height = (int) [ 1, MAX ], "
        "framerate = (fraction) [ 0/1, MAX ]"
    )
);

/* ================= HELPERS ================= */

#define CHECK_CL(err, msg) \
    if ((err) != CL_SUCCESS) { \
        GST_ERROR_OBJECT(self, "%s failed (%d)", msg, err); \
        goto error; \
    }

/* Load entire OpenCL kernel source file into a memory buffer. */
static gchar *
load_kernel_file(const gchar *path)
{
    gchar *data = NULL;
    gsize size = 0;

    if (!g_file_get_contents(path, &data, &size, NULL))
        return NULL;

    return data;
}

/* OpenCL queue with profiling enabled. */
cl_queue_properties props[] = {
    CL_QUEUE_PROPERTIES,
    CL_QUEUE_PROFILING_ENABLE,
    0
};

/* ================= OPENCL INITIALIZATION =================*/
static gboolean
gst_ocl_shader_set_info(GstVideoFilter *filter,
                           GstCaps *incaps, GstVideoInfo *ininfo,
                           GstCaps *outcaps, GstVideoInfo *outinfo)
{
    GstOCLShader *self = (GstOCLShader *)filter;

    if (!self->kernel_file || !g_file_test(self->kernel_file, G_FILE_TEST_EXISTS) || !self->kernel_func) {
        GST_ERROR_OBJECT(self,
            "kernel-file or kernel-func not set, running in bypass mode");
        goto error;
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
                    self->context, self->device, props, &err);
    CHECK_CL(err, "clCreateCommandQueueWithProperties");

    gchar *kernel_src = load_kernel_file(self->kernel_file);
    if (!kernel_src) {
        GST_ERROR_OBJECT(self,
            "Failed to load kernel file: %s", self->kernel_file);
        goto error;
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
                        self->program, self->kernel_func, &err);
    CHECK_CL(err, "clCreateKernel");

    self->cl_ready = TRUE;
    return TRUE;

error:
    GST_ERROR_OBJECT(self,
        "OpenCL unavailable, running in bypass mode");
    self->cl_ready = FALSE;
    return TRUE; /* Allow pipeline to continue */
}

/* ================= FRAME PROCESS ================= */
static GstFlowReturn
gst_ocl_shader_transform_frame(GstVideoFilter *filter,
                                  GstVideoFrame *in,
                                  GstVideoFrame *out)
{
    GstOCLShader *self = (GstOCLShader *)filter;
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
    int idx = self->frame_count % NUM_BUFFERS;

    /* Reallocate buffers on caps change */
    if (size != self->buf_size) {
        for (int i = 0; i < NUM_BUFFERS; i++) {
            if (self->ybuf[i]) {
                clReleaseMemObject(self->ybuf[i]);
                self->ybuf[i] = NULL;
            }
        }

        for (int i = 0; i < NUM_BUFFERS; i++) {
            self->ybuf[i] = clCreateBuffer(
                self->context,
                CL_MEM_READ_WRITE,
                size, NULL, &err);
            CHECK_CL(err, "clCreateBuffer");
        }

        self->buf_size = size;
        self->width = width;
        self->height = height;
        self->stride = stride;
    }

    /* Cleanup old events */
    if (self->write_evt[idx])  clReleaseEvent(self->write_evt[idx]);
    if (self->kernel_evt[idx]) clReleaseEvent(self->kernel_evt[idx]);
    if (self->read_evt[idx])   clReleaseEvent(self->read_evt[idx]);

    /* Non-blocking write */
    err = clEnqueueWriteBuffer(self->queue,
                               self->ybuf[idx], CL_FALSE,
                               0, size, y,
                               0, NULL, &self->write_evt[idx]);
    CHECK_CL(err, "clEnqueueWriteBuffer");

    /* OpenCL Kernel Argument, Modify this code if you are adding new arguments into OpenCL kernel function. */
    err = clSetKernelArg(self->kernel, 0, sizeof(cl_mem), &self->ybuf[idx]);
    CHECK_CL(err, "clSetKernelArg(0)");
    err = clSetKernelArg(self->kernel, 1, sizeof(int), &width);
    CHECK_CL(err, "clSetKernelArg(1)");
    err = clSetKernelArg(self->kernel, 2, sizeof(int), &height);
    CHECK_CL(err, "clSetKernelArg(2)");
    err = clSetKernelArg(self->kernel, 3, sizeof(int), &stride);
    CHECK_CL(err, "clSetKernelArg(3)");

    size_t global[2] = { width, height };

    GST_DEBUG_OBJECT(self, "Enqueue kernel global=(%zu x %zu)", global[0], global[1]);

    /* Kernel depends on write */
    err = clEnqueueNDRangeKernel(self->queue,
                                 self->kernel,
                                 2, NULL,
                                 global, NULL,
                                 1, &self->write_evt[idx],
                                 &self->kernel_evt[idx]);
    CHECK_CL(err, "clEnqueueNDRangeKernel");

    err = clFinish(self->queue);
    CHECK_CL(err, "clFinish");

    /* Read depends on kernel */
    err = clEnqueueReadBuffer(self->queue,
                              self->ybuf[idx], CL_FALSE,
                              0, size, y,
                              1, &self->kernel_evt[idx],
                                &self->read_evt[idx]);
    CHECK_CL(err, "clEnqueueReadBuffer");

    /* Allow GPU to run asynchronously */
    clFlush(self->queue);

    return GST_FLOW_OK;
error:
    GST_ERROR_OBJECT(self, "OpenCL execution failed");
    return GST_FLOW_ERROR;
}

static void
gst_ocl_shader_set_property(GObject *object,
                               guint prop_id,
                               const GValue *value,
                               GParamSpec *pspec)
{
    GstOCLShader *self = (GstOCLShader *)object;

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

        case PROP_KERNEL_FUNC:
            g_free(self->kernel_func);
            self->kernel_func = g_value_dup_string(value);

            GST_INFO_OBJECT(self,
                "kernel-func set to: %s",
                self->kernel_func ? self->kernel_func : "(null)");

            /* Force re-init on next set_info */
            self->cl_ready = FALSE;
            break;

        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void
gst_ocl_shader_get_property(GObject *object,
                               guint prop_id,
                               GValue *value,
                               GParamSpec *pspec)
{
    GstOCLShader *self = (GstOCLShader *)object;

    switch (prop_id) {
    case PROP_KERNEL_FILE:
        g_value_set_string(value, self->kernel_file);
        break;

    case PROP_KERNEL_FUNC:
        g_value_set_string(value, self->kernel_func);
        break;

    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void
gst_ocl_shader_finalize(GObject *object)
{
    GstOCLShader *self = (GstOCLShader *)object;

    GST_DEBUG_OBJECT(self, "Finalizing OpenCL filter");

    /* Release OpenCL events */
    for (int i = 0; i < NUM_BUFFERS; i++) {
        if (self->write_evt[i]) {
            clReleaseEvent(self->write_evt[i]);
            self->write_evt[i] = NULL;
        }
        if (self->kernel_evt[i]) {
            clReleaseEvent(self->kernel_evt[i]);
            self->kernel_evt[i] = NULL;
        }
        if (self->read_evt[i]) {
            clReleaseEvent(self->read_evt[i]);
            self->read_evt[i] = NULL;
        }
    }

    /* Release OpenCL buffers */
    for (int i = 0; i < NUM_BUFFERS; i++) {
        if (self->ybuf[i]) {
            clReleaseMemObject(self->ybuf[i]);
            self->ybuf[i] = NULL;
        }
    }

    /* Release OpenCL kernel/program/queue/context */
    if (self->kernel) {
        clReleaseKernel(self->kernel);
        self->kernel = NULL;
    }

    if (self->program) {
        clReleaseProgram(self->program);
        self->program = NULL;
    }

    if (self->queue) {
        clReleaseCommandQueue(self->queue);
        self->queue = NULL;
    }

    if (self->context) {
        clReleaseContext(self->context);
        self->context = NULL;
    }

    /* Free GObject properties */
    g_clear_pointer(&self->kernel_file, g_free);
    g_clear_pointer(&self->kernel_func, g_free);

    /* Chain up to parent class */
    G_OBJECT_CLASS(gst_ocl_shader_parent_class)->finalize(object);
}


/* Instance initialization */
static void
gst_ocl_shader_init(GstOCLShader *self)
{
    self->cl_ready = FALSE;
    self->frame_count = 0;
    self->buf_size = 0;
    self->kernel_file = NULL;
    self->kernel_func = NULL;

    for (int i = 0; i < NUM_BUFFERS; i++) {
        self->ybuf[i] = NULL;
        self->write_evt[i] = NULL;
        self->kernel_evt[i] = NULL;
        self->read_evt[i] = NULL;
    }
}

/* Class initialization */
static void
gst_ocl_shader_class_init(GstOCLShaderClass *klass)
{
    GstElementClass *eclass = GST_ELEMENT_CLASS(klass);
    GstVideoFilterClass *vclass = GST_VIDEO_FILTER_CLASS(klass);
    GObjectClass *gclass = G_OBJECT_CLASS(klass);

    GST_DEBUG_CATEGORY_INIT(gst_ocl_shader_debug,
                            "oscaroclshader", 0,
                            "OpenCL NV12 Shader");

    /* BaseTransform virtual functions */
    gclass->finalize = gst_ocl_shader_finalize;
    vclass->set_info = GST_DEBUG_FUNCPTR(gst_ocl_shader_set_info);
    vclass->transform_frame =
        GST_DEBUG_FUNCPTR(gst_ocl_shader_transform_frame);

    /* Add pad templates */
    gst_element_class_add_pad_template(
        eclass, gst_static_pad_template_get(&sink_template));
    gst_element_class_add_pad_template(
        eclass, gst_static_pad_template_get(&src_template));

    /* Element metadata */
    gst_element_class_set_static_metadata(
        eclass,
        "OpenCL NV12 Shader",
        "Filter/Video",
        "Applies OpenCL processing on NV12 video",
        "eInfochips-Leica");

    gclass->set_property = gst_ocl_shader_set_property;
    gclass->get_property = gst_ocl_shader_get_property;

    g_object_class_install_property(
        gclass,
        PROP_KERNEL_FILE,
        g_param_spec_string(
            "kernel-file",
            "OpenCL kernel file",
            "Path to OpenCL kernel file (.cl). "
            "If not set, shader runs in bypass mode.",
            NULL,  /* default */
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(
        gclass,
        PROP_KERNEL_FUNC,
        g_param_spec_string(
            "kernel-func",
            "OpenCL kernel function",
            "Kernel function name inside the OpenCL program. "
            "If not set, shader runs in bypass mode.",
            NULL, /* default */
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

}

/* Plugin entry point */
static gboolean
plugin_init(GstPlugin *plugin)
{
    return gst_element_register(plugin,
                                "oscaroclshader",
                                GST_RANK_NONE,
                                GST_TYPE_OCL_SHADER);
}

/* Define plugin */
GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    oscaroclshader,
    "OpenCL NV12 shader",
    plugin_init,
    VERSION,
    "LGPL",
    PACKAGE,
    PACKAGE
)
