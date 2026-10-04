#include "libmpv_gpu_next.h"
#include <stddef.h>             // for NULL
#include <sys/stat.h>           // for stat (cache)
#include <time.h>               // for time, difftime (cache)
#include <libplacebo/cache.h>   // for pl_cache
#include "common/common.h"      // for MP_TARRAY_APPEND
#include "misc/io_utils.h"      // for mp_save_to_file
#include "options/m_config.h"   // for mp_get_config_group
#include "options/path.h"       // for mp_get_user_path, mp_path_join
#include "osdep/io.h"           // for opendir/stat/unlink wrappers on Windows
#include "osdep/timer.h"        // for mp_time_ns
#include "stream/stream.h"      // for stream_read_file
#include "video/out/gpu/video.h"      // for gl_video_conf (gpu-shader-cache*)
#include "video/out/placebo/utils.h"  // for mppl_log_create
#include "common/msg.h"         // for mp_log_new, MP_ERR
#include "config.h"             // for HAVE_GL
#include "libplacebo/config.h"  // for PL_HAVE_OPENGL
#include "libplacebo/gpu.h"     // for pl_tex, pl_tex_params, pl_tex_t
#include "mpv/client.h"         // for mpv_error
#include "mpv/render.h"         // for mpv_render_param, mpv_render_param_type
#include "ra.h"                 // for ra_next_tex_destroy
#include "stdbool.h"            // for bool, false
#include "string.h"             // for strcmp
#include "ta/ta_talloc.h"       // for talloc_free, talloc_zero
#include "video.h"              // for pl_video_check_format, pl_video_init
#include "video/hwdec.h"        // for hwdec_devices_create, hwdec_devices_d...
#include "video/out/libmpv.h"   // for render_backend, get_mpv_render_param
#include "video/out/vo.h"       // for vo_frame (ptr only), voctrl_screenshot

/*
 * Structure for the image parameters.
 */
struct mp_image_params;
struct mp_osd_res;
struct mp_rect;

/*
 * Private data for the GPU next render backend.
 */
/*
 * On-disk shader cache. libplacebo bakes size-dependent constants (e.g. the
 * polar scaler's compute work group / shmem layout) into shader source, so
 * every new output size compiles fresh shaders. On D3D11 that goes through
 * GLSL -> SPIR-V -> HLSL -> FXC on the render thread and stalls the first
 * frame at a new size (first maximize / fullscreen goes black for ~0.5 s).
 * Persisting compiled shaders makes that a once-per-size-ever cost.
 *
 * Copied from vo_gpu_next.c (struct cache / cache_*); kept identical so
 * upstream fixes there can be carried over.
 */
struct shader_cache {
    struct mp_log *log;
    struct mpv_global *global;
    char *dir;
    const char *name;
    size_t size_limit;
    pl_cache cache;
};

struct priv {
    struct libmpv_gpu_next_context *context; // Manages the API (e.g., OpenGL)
    struct pl_video *video_engine;           // Manages synchronous libplacebo rendering
    pl_log cache_log;
    struct shader_cache shader_cache;
};

static char *cache_filepath(void *ta_ctx, char *dir, const char *prefix, uint64_t key)
{
    bstr filename = {0};
    bstr_xappend_asprintf(ta_ctx, &filename, "%s_%016" PRIx64, prefix, key);
    return mp_path_join_bstr(ta_ctx, bstr0(dir), filename);
}

static pl_cache_obj cache_load_obj(void *p, uint64_t key)
{
    struct shader_cache *c = p;
    void *ta_ctx = talloc_new(NULL);
    pl_cache_obj obj = {0};

    if (!c->dir)
        goto done;

    char *filepath = cache_filepath(ta_ctx, c->dir, c->name, key);
    if (!filepath)
        goto done;

    if (stat(filepath, &(struct stat){0}))
        goto done;

    int64_t load_start = mp_time_ns();
    struct bstr data = stream_read_file(filepath, ta_ctx, c->global, STREAM_MAX_READ_SIZE);
    int64_t load_end = mp_time_ns();
    MP_DBG(c, "%s: key(%" PRIx64 "), size(%zu), load time(%.3f ms)\n",
           __func__, key, data.len,
           MP_TIME_NS_TO_MS(load_end - load_start));

    obj = (pl_cache_obj){
        .key = key,
        .data = talloc_steal(NULL, data.start),
        .size = data.len,
        .free = talloc_free,
    };

done:
    talloc_free(ta_ctx);
    return obj;
}

static void cache_save_obj(void *p, pl_cache_obj obj)
{
    const struct shader_cache *c = p;
    void *ta_ctx = talloc_new(NULL);

    if (!c->dir)
        goto done;

    char *filepath = cache_filepath(ta_ctx, c->dir, c->name, obj.key);
    if (!filepath)
        goto done;

    if (!obj.data || !obj.size) {
        unlink(filepath);
        goto done;
    }

    // Don't save if already exists
    struct stat st;
    if (!stat(filepath, &st) && st.st_size == obj.size) {
        MP_DBG(c, "%s: key(%"PRIx64"), size(%zu)\n", __func__, obj.key, obj.size);
        goto done;
    }

    int64_t save_start = mp_time_ns();
    mp_save_to_file(filepath, obj.data, obj.size);
    int64_t save_end = mp_time_ns();
    MP_DBG(c, "%s: key(%" PRIx64 "), size(%zu), save time(%.3f ms)\n",
           __func__, obj.key, obj.size,
           MP_TIME_NS_TO_MS(save_end - save_start));

done:
    talloc_free(ta_ctx);
}

static void cache_init(struct render_backend *ctx, struct shader_cache *cache,
                       const char *dir_opt)
{
    struct priv *p = ctx->priv;

    char *dir;
    if (dir_opt && dir_opt[0]) {
        dir = mp_get_user_path(p, ctx->global, dir_opt);
    } else {
        dir = mp_find_user_file(p, ctx->global, "cache", "");
    }
    if (!dir || !dir[0]) {
        MP_VERBOSE(ctx, "No shader cache directory, shaders are recompiled "
                   "on every launch (set gpu-shader-cache-dir).\n");
        return;
    }

    mp_mkdirp(dir);
    *cache = (struct shader_cache){
        .log        = ctx->log,
        .global     = ctx->global,
        .dir        = dir,
        .name       = "shader",
        .size_limit = 128 << 20,
        .cache = pl_cache_create(pl_cache_params(
            .log = p->cache_log,
            .get = cache_load_obj,
            .set = cache_save_obj,
            .priv = cache
        )),
    };
}

struct file_entry {
    char *filepath;
    size_t size;
    time_t atime;
};

static int compare_atime(const void *a, const void *b)
{
    return (((struct file_entry *)b)->atime - ((struct file_entry *)a)->atime);
}

static void cache_uninit(struct render_backend *ctx, struct shader_cache *cache)
{
    if (!cache->cache)
        return;

    void *ta_ctx = talloc_new(NULL);
    struct file_entry *files = NULL;
    size_t num_files = 0;
    mp_assert(cache->dir);
    mp_assert(cache->name);

    DIR *d = opendir(cache->dir);
    if (!d)
        goto done;

    struct dirent *dir;
    while ((dir = readdir(d)) != NULL) {
        char *filepath = mp_path_join(ta_ctx, cache->dir, dir->d_name);
        if (!filepath)
            continue;
        struct stat filestat;
        if (stat(filepath, &filestat))
            continue;
        if (!S_ISREG(filestat.st_mode))
            continue;
        bstr fname = bstr0(dir->d_name);
        if (!bstr_eatstart0(&fname, cache->name))
            continue;
        if (!bstr_eatstart0(&fname, "_"))
            continue;
        if (fname.len != 16) // %016x
            continue;
        MP_TARRAY_APPEND(ta_ctx, files, num_files,
                         (struct file_entry){
                             .filepath = filepath,
                             .size     = filestat.st_size,
                             .atime    = filestat.st_atime,
                         });
    }
    closedir(d);

    if (!num_files)
        goto done;

    qsort(files, num_files, sizeof(struct file_entry), compare_atime);

    time_t t = time(NULL);
    size_t cache_size = 0;
    size_t cache_limit = cache->size_limit ? cache->size_limit : SIZE_MAX;
    for (int i = 0; i < num_files; i++) {
        // Remove files that exceed the size limit but are older than one day.
        cache_size += files[i].size;
        double rel_use = difftime(t, files[i].atime);
        if (cache_size > cache_limit && rel_use > 60 * 60 * 24) {
            MP_VERBOSE(ctx, "Removing %s | size: %9zu bytes | last used: %9d seconds ago\n",
                       files[i].filepath, files[i].size, (int)rel_use);
            unlink(files[i].filepath);
        }
    }

done:
    talloc_free(ta_ctx);
    pl_cache_destroy(&cache->cache);
}

/*
* List of available API context implementations (e.g., GL, Vulkan - currently only OpenGL)
*/
static const struct libmpv_gpu_next_context_fns *context_backends[] = {
#if HAVE_GL && defined(PL_HAVE_OPENGL)
    &libmpv_gpu_next_context_gl,
#endif
#if HAVE_D3D11 && defined(PL_HAVE_D3D11)
    &libmpv_gpu_next_context_d3d11,
#endif
#if HAVE_VULKAN && defined(PL_HAVE_VULKAN)
    &libmpv_gpu_next_context_vulkan,
#endif
    NULL
};

/*
 * @brief Initializes the render_backend layer.
 * @param ctx The render_backend context.
 * @param params The render parameters.
 * @return 0 on success, negative error code on failure.
 */
static int init(struct render_backend *ctx, mpv_render_param *params)
{
    ctx->priv = talloc_zero(NULL, struct priv);
    struct priv *p = ctx->priv;

    // Get the API type from the render parameters.
    char *api = get_mpv_render_param(params, MPV_RENDER_PARAM_API_TYPE, NULL);
    if (!api) {
        MP_ERR(ctx, "API type not specified.\n");
        return MPV_ERROR_INVALID_PARAMETER;
    }

    // Find and initialize the requested API context (e.g., _gl.c).
    // This will create the pl_gpu and the `ra` (libplacebo render abstraction).
    for (int n = 0; context_backends[n]; n++) {
        const struct libmpv_gpu_next_context_fns *backend = context_backends[n];
        if (strcmp(backend->api_name, api) == 0) {
            p->context = talloc_zero(p, struct libmpv_gpu_next_context);
            *p->context = (struct libmpv_gpu_next_context){
                .global = ctx->global,
                .log = mp_log_new(p, ctx->log, "gpu-next-ctx"),
                .fns = backend,
            };
            break;
        }
    }
    if (!p->context) {
        MP_ERR(ctx, "Requested API type '%s' is not supported.\n", api);
        return MPV_ERROR_NOT_IMPLEMENTED;
    }
    int err = p->context->fns->init(p->context, params);
    if (err < 0) {
        talloc_free(p->context);
        p->context = NULL;
        return err;
    }

    // Attach the on-disk shader cache before anything compiles a shader.
    struct gl_video_opts *gl_opts = mp_get_config_group(p, ctx->global, &gl_video_conf);
    p->cache_log = mppl_log_create(p, ctx->log);
    if (gl_opts->shader_cache)
        cache_init(ctx, &p->shader_cache, gl_opts->shader_cache_dir);
    pl_gpu_set_cache(p->context->gpu, p->shader_cache.cache);

    // Initialize our synchronous libplacebo rendering engine.
    p->video_engine = pl_video_init(ctx->global, ctx->log, p->context->ra);
    if (!p->video_engine) {
        pl_gpu_set_cache(p->context->gpu, NULL);
        cache_uninit(ctx, &p->shader_cache);
        pl_log_destroy(&p->cache_log);
        p->context->fns->destroy(p->context);
        talloc_free(p->context);
        return MPV_ERROR_VO_INIT_FAILED;
    }

    // Create hardware decoder devices.
    ctx->hwdec_devs = hwdec_devices_create();
    ctx->driver_caps = VO_CAP_ROTATE90 | VO_CAP_VFLIP;
    return 0;
}

/*
 * @brief Destroys the render_backend layer.
 * @param ctx The render_backend context.
 */
static void destroy(struct render_backend *ctx)
{
    struct priv *p = ctx->priv;
    if (!p) return;

    hwdec_devices_destroy(ctx->hwdec_devs);
    pl_video_uninit(&p->video_engine);
    if (p->context && p->context->gpu)
        pl_gpu_set_cache(p->context->gpu, NULL);
    cache_uninit(ctx, &p->shader_cache);
    pl_log_destroy(&p->cache_log);
    if (p->context) {
        p->context->fns->destroy(p->context); // This destroys the RA
        talloc_free(p->context);
    }
    talloc_free(p);
    ctx->priv = NULL;
}

/*
 * @brief Renders a video frame.
 * @param ctx The render_backend context.
 * @param params The render parameters.
 * @param frame The video frame to render.
 * @return 0 on success, negative error code on failure.
 */
static int render(struct render_backend *ctx, mpv_render_param *params,
                  struct vo_frame *frame)
{
    struct priv *p = ctx->priv;
    if (!p->video_engine) return MPV_ERROR_UNINITIALIZED;

    // Wrap the framebuffer object (FBO) for rendering.
    pl_tex target_tex = NULL;
    int err = p->context->fns->wrap_fbo(p->context, params, &target_tex);
    if (err < 0) return err;
    if (!target_tex) return MPV_ERROR_GENERIC;

    // Render the video frame.
    pl_video_render(p->video_engine, frame, target_tex);

    // done_frame fires before texture destruction — Vulkan backends need
    // this ordering to hold the image back before the wrapper is freed.
    if (p->context->fns->done_frame)
        p->context->fns->done_frame(p->context);

    // Backends with persistent_target_tex own target_tex across frames (see
    // libmpv_gpu_next.h) and free it themselves in destroy(); destroying it
    // here would race whatever GPU work this frame's render just submitted.
    if (!p->context->fns->persistent_target_tex)
        ra_next_tex_destroy(p->context->ra, &target_tex);

    return 0;
}

/*
 * @brief Reconfigures the video engine with new image parameters.
 * @param ctx The render_backend context.
 * @param params The new image parameters.
 */
static void reconfig(struct render_backend *ctx, struct mp_image_params *params)
{
    struct priv *p = ctx->priv;
    if (p->video_engine)
        pl_video_reconfig(p->video_engine, params);
}

/*
 * @brief Resizes the video output.
 * @param ctx The render_backend context.
 * @param src The source rectangle.
 * @param dst The destination rectangle.
 * @param osd The OSD rectangle.
 */
static void resize(struct render_backend *ctx, struct mp_rect *src,
                   struct mp_rect *dst, struct mp_osd_res *osd)
{
    struct priv *p = ctx->priv;
    if (p->video_engine)
        pl_video_resize(p->video_engine, dst, osd);
}

/*
 * @brief Updates the external state of the render_backend.
 * @param ctx The render_backend context.
 * @param vo The video output context.
 */
static void update_external(struct render_backend *ctx, struct vo *vo)
{
    struct priv *p = ctx->priv;
    if (p->video_engine)
        pl_video_update_osd(p->video_engine, vo ? vo->osd : NULL);
}

/*
 * @brief Resets the video engine.
 * @param ctx The render_backend context.
 */
static void reset(struct render_backend *ctx)
{
    struct priv *p = ctx->priv;
    if (p->video_engine)
        pl_video_reset(p->video_engine);
}

/*
 * @brief Checks if the given image format is supported.
 * @param ctx The render_backend context.
 * @param imgfmt The image format to check.
 * @return True if the format is supported, false otherwise.
 */
static bool check_format(struct render_backend *ctx, int imgfmt)
{
    struct priv *p = ctx->priv;
    return p->video_engine ? pl_video_check_format(p->video_engine, imgfmt) : false;
}

/*
 * @brief Gets the target size for rendering.
 * @param ctx The render_backend context.
 * @param params The render parameters.
 * @param out_w Pointer to the output width.
 * @param out_h Pointer to the output height.
 * @return 0 on success, negative error code on failure.
 */
static int get_target_size(struct render_backend *ctx, mpv_render_param *params, int *out_w, int *out_h)
{
    struct priv *p = ctx->priv;
    if (!p->context || !p->context->fns || !p->context->ra) return MPV_ERROR_UNINITIALIZED;
    pl_tex tex = NULL;
    int err = p->context->fns->wrap_fbo(p->context, params, &tex);
    if (err < 0) return err;
    if (!tex) return MPV_ERROR_GENERIC;
    *out_w = tex->params.w;
    *out_h = tex->params.h;
    // See render(): persistent_target_tex backends own this across frames.
    if (!p->context->fns->persistent_target_tex)
        ra_next_tex_destroy(p->context->ra, &tex);
    return 0;
}

/*
 * @brief Takes a screenshot of the current video frame.
 * @param ctx The render_backend context.
 * @param frame The video frame to capture.
 * @param args The screenshot arguments.
 */
static void screenshot(struct render_backend *ctx, struct vo_frame *frame,
                       struct voctrl_screenshot *args)
{
    struct priv *p = ctx->priv;
    args->res = NULL;
    if (!p || !p->video_engine)
        return;

    /* Let the pl_video engine perform the screenshot (uploads, tone-mapping,
     * render to an sRGB temporary, download). Returns an mp_image* or NULL. */
    struct mp_image *img = pl_video_screenshot(p->video_engine, frame);
    if (img)
        args->res = img;
}

/*
 * @brief Sets a render parameter.
 * @param ctx The render_backend context.
 * @param param The render parameter to set.
 * @return 0 on success, negative error code on failure.
 */
static int set_parameter(struct render_backend *ctx, mpv_render_param param)
{
    return MPV_ERROR_NOT_IMPLEMENTED;
}

/*
 * @brief Gets an image for rendering.
 * @param ctx The render_backend context.
 * @param imgfmt The image format.
 * @param w The width of the image.
 * @param h The height of the image.
 * @param stride_align The stride alignment.
 * @param flags The image flags.
 * @return A pointer to the image, or NULL on failure.
 */
static struct mp_image *get_image(struct render_backend *ctx, int imgfmt,
                                  int w, int h, int stride_align, int flags)
{
    return NULL;
}

/*
 * @brief Collects performance data from the render_backend.
 * @param ctx The render_backend context.
 * @param out The output structure to fill with performance data.
 */
static void perfdata(struct render_backend *ctx,
                     struct voctrl_performance_data *out)
{
    // Pass timing collection is not implemented for this backend, but the
    // caller (mp_property_vo_passes) hands us an UNINITIALIZED struct and
    // trusts whatever we leave in it — vo_libmpv reports VO_TRUE whenever
    // this hook exists. Leaving it unwritten means garbage pass counts and
    // an out-of-bounds crash the moment the stats script queries vo-passes.
    *out = (struct voctrl_performance_data){0};
}

const struct render_backend_fns render_backend_gpu_next = {
    .init = init,
    .destroy = destroy,
    .render = render,
    .check_format = check_format,
    .set_parameter = set_parameter,
    .reconfig = reconfig,
    .reset = reset,
    .update_external = update_external,
    .resize = resize,
    .get_target_size = get_target_size,
    .get_image = get_image,
    .screenshot = screenshot,
    .perfdata = perfdata,
};
