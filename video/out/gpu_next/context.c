/*
 * This file is part of mpv.
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <libplacebo/config.h>

#ifdef PL_HAVE_D3D11
#include <libplacebo/d3d11.h>
#endif

#ifdef PL_HAVE_VULKAN
#include <libplacebo/vulkan.h>
#include "mpv/render_vulkan.h"
#endif

#ifdef PL_HAVE_OPENGL
#include <libplacebo/opengl.h>
#include "mpv/render_gl.h"
#include "video/out/gpu_next/libmpv_gpu_next.h"
#include "video/out/gpu_next/ra.h"
#endif

#include "context.h"
#include "config.h"
#include "common/common.h"
#include "options/m_config.h"
#include "video/out/libmpv.h"
#include "video/out/placebo/utils.h"
#include "video/out/gpu/video.h"

#if HAVE_D3D11
#include "osdep/windows_utils.h"
#include "video/out/d3d11/ra_d3d11.h"
#include "video/out/d3d11/context.h"
#endif

#if HAVE_GL
#include "video/out/opengl/context.h"
#include "video/out/opengl/ra_gl.h"
# if HAVE_EGL
#include <EGL/egl.h>
# endif
#endif

#if HAVE_VULKAN
#include "video/out/vulkan/context.h"
#endif

#if HAVE_D3D11
static bool d3d11_pl_init(struct vo *vo, struct gpu_ctx *ctx,
                          struct ra_ctx_opts *ctx_opts)
{
#if !defined(PL_HAVE_D3D11)
    MP_MSG(ctx, vo->probing ? MSGL_V : MSGL_ERR,
           "libplacebo was built without D3D11 support.\n");
    return false;
#else // defined(PL_HAVE_D3D11)
    bool success = false;

    ID3D11Device   *device    = ra_d3d11_get_device(ctx->ra_ctx->ra);
    IDXGISwapChain *swapchain = ra_d3d11_ctx_get_swapchain(ctx->ra_ctx);
    if (!device || !swapchain) {
        mp_err(ctx->log,
               "Failed to receive required components from the mpv d3d11 "
               "context! (device: %s, swap chain: %s)\n",
               device    ? "OK" : "failed",
               swapchain ? "OK" : "failed");
        goto err_out;
    }

    pl_d3d11 d3d11 = pl_d3d11_create(ctx->pllog,
        pl_d3d11_params(
            .device = device,
        )
    );
    if (!d3d11) {
        mp_err(ctx->log, "Failed to acquire a d3d11 libplacebo context!\n");
        goto err_out;
    }
    ctx->gpu = d3d11->gpu;

    mppl_log_set_probing(ctx->pllog, false);

    struct pl_d3d11_swapchain_params *params = pl_d3d11_swapchain_params(
         .swapchain = swapchain,
    );
    ra_d3d11_ctx_set_swapchain_params(ctx->ra_ctx, params);
    ctx->swapchain = pl_d3d11_create_swapchain(d3d11, params);
    if (!ctx->swapchain) {
        mp_err(ctx->log, "Failed to acquire a d3d11 libplacebo swap chain!\n");
        goto err_out;
    }

    success = true;

err_out:
    SAFE_RELEASE(swapchain);
    SAFE_RELEASE(device);

    return success;
#endif // defined(PL_HAVE_D3D11)
}
#endif // HAVE_D3D11

struct gpu_ctx *gpu_ctx_create(struct vo *vo, struct ra_ctx_opts *ctx_opts)
{
    struct gpu_ctx *ctx = talloc_zero(NULL, struct gpu_ctx);
    ctx->log = vo->log;
    ctx->ra_ctx = ra_ctx_create(vo, *ctx_opts);
    if (!ctx->ra_ctx)
        goto err_out;

#if HAVE_VULKAN
    struct mpvk_ctx *vkctx = ra_vk_ctx_get(ctx->ra_ctx);
    if (vkctx) {
        ctx->pllog = vkctx->pllog;
        ctx->gpu = vkctx->gpu;
        ctx->swapchain = vkctx->swapchain;
        return ctx;
    }
#endif

    ctx->pllog = mppl_log_create(ctx, ctx->log);
    if (!ctx->pllog)
        goto err_out;

    mppl_log_set_probing(ctx->pllog, vo->probing);

#if HAVE_D3D11
    if (ra_is_d3d11(ctx->ra_ctx->ra)) {
        if (!d3d11_pl_init(vo, ctx, ctx_opts))
            goto err_out;

        return ctx;
    }
#endif

#if HAVE_GL && defined(PL_HAVE_OPENGL)
    if (ra_is_gl(ctx->ra_ctx->ra)) {
        struct GL *gl = ra_gl_get(ctx->ra_ctx->ra);
        struct pl_opengl_params params = *pl_opengl_params(
            .debug = ctx_opts->debug,
            .allow_software = ctx_opts->allow_sw,
            .get_proc_addr_ex = (void *) gl->get_fn,
            .proc_ctx = gl->fn_ctx,
        );
# if HAVE_EGL
        params.egl_display = eglGetCurrentDisplay();
        params.egl_context = eglGetCurrentContext();
# endif
        pl_opengl opengl = pl_opengl_create(ctx->pllog, &params);
        if (!opengl)
            goto err_out;
        ctx->gpu = opengl->gpu;

        mppl_log_set_probing(ctx->pllog, false);

        ctx->swapchain = pl_opengl_create_swapchain(opengl, pl_opengl_swapchain_params(
            .max_swapchain_depth = vo->opts->swapchain_depth,
            .framebuffer.flipped = gl->flipped,
        ));
        if (!ctx->swapchain)
            goto err_out;

        return ctx;
    }
#elif HAVE_GL
    if (ra_is_gl(ctx->ra_ctx->ra)) {
        MP_MSG(ctx, vo->probing ? MSGL_V : MSGL_ERR,
            "libplacebo was built without OpenGL support.\n");
    }
#endif

err_out:
    gpu_ctx_destroy(&ctx);
    return NULL;
}

bool gpu_ctx_resize(struct gpu_ctx *ctx, int w, int h)
{
#if HAVE_VULKAN
    if (ra_vk_ctx_get(ctx->ra_ctx))
        // vulkan RA handles this by itself
        return true;
#endif

    return pl_swapchain_resize(ctx->swapchain, &w, &h);
}

void gpu_ctx_destroy(struct gpu_ctx **ctxp)
{
    struct gpu_ctx *ctx = *ctxp;
    if (!ctx)
        return;
    if (!ctx->ra_ctx)
        goto skip_common_pl_cleanup;

#if HAVE_VULKAN
    if (ra_vk_ctx_get(ctx->ra_ctx))
        // vulkan RA context handles pl cleanup by itself,
        // skip common local clean-up.
        goto skip_common_pl_cleanup;
#endif

    if (ctx->swapchain)
        pl_swapchain_destroy(&ctx->swapchain);

    if (ctx->gpu) {
#if HAVE_GL && defined(PL_HAVE_OPENGL)
        if (ra_is_gl(ctx->ra_ctx->ra)) {
            pl_opengl opengl = pl_opengl_get(ctx->gpu);
            pl_opengl_destroy(&opengl);
        }
#endif

#if HAVE_D3D11 && defined(PL_HAVE_D3D11)
        if (ra_is_d3d11(ctx->ra_ctx->ra)) {
            pl_d3d11 d3d11 = pl_d3d11_get(ctx->gpu);
            pl_d3d11_destroy(&d3d11);
        }
#endif
    }

    if (ctx->pllog)
        pl_log_destroy(&ctx->pllog);

skip_common_pl_cleanup:
    ra_ctx_destroy(&ctx->ra_ctx);

    talloc_free(ctx);
    *ctxp = NULL;
}

#if HAVE_GL && defined(PL_HAVE_OPENGL)

struct priv {
    pl_log pl_log;
    pl_opengl gl;
    pl_gpu gpu;
    struct ra_next *ra;
    mpv_opengl_init_params gl_params;
};

static bool pl_callback_makecurrent_gl(void *priv)
{
    mpv_opengl_init_params *gl_params = priv;
    if (gl_params && gl_params->get_proc_address) {
        gl_params->get_proc_address(gl_params->get_proc_address_ctx, "glGetString");
        return true;
    }
    return false;
}

static void pl_callback_releasecurrent_gl(void *priv)
{
}

static void pl_log_cb(void *log_priv, enum pl_log_level level, const char *msg)
{
    struct mp_log *log = log_priv;
    mp_msg(log, MSGL_WARN, "[gpu-next:pl] %s\n", msg);
}

static int libmpv_gpu_next_init_gl(struct libmpv_gpu_next_context *ctx, mpv_render_param *params)
{
    ctx->priv = talloc_zero(NULL, struct priv);
    struct priv *p = ctx->priv;

    mpv_opengl_init_params *gl_params =
    get_mpv_render_param(params, MPV_RENDER_PARAM_OPENGL_INIT_PARAMS, NULL);
    if (!gl_params || !gl_params->get_proc_address)
        return MPV_ERROR_INVALID_PARAMETER;

    p->gl_params = *gl_params;

    struct pl_log_params log_params = {
        .log_level = PL_LOG_DEBUG
    };

    if (mp_msg_test(ctx->log, MSGL_TRACE)) {
        log_params.log_cb = pl_log_cb;
        log_params.log_priv = ctx->log;
    }

    p->pl_log = pl_log_create(PL_API_VER, &log_params);
    p->gl = pl_opengl_create(p->pl_log, pl_opengl_params(
        .get_proc_addr_ex = (pl_voidfunc_t (*)(void*, const char*))gl_params->get_proc_address,
        .proc_ctx = gl_params->get_proc_address_ctx,
        .make_current = pl_callback_makecurrent_gl,
        .release_current = pl_callback_releasecurrent_gl,
        .priv = &p->gl_params
    ));

    if (!p->gl) {
        MP_ERR(ctx, "Failed to create libplacebo OpenGL context.\n");
        pl_log_destroy(&p->pl_log);
        return MPV_ERROR_UNSUPPORTED;
    }
    p->gpu = p->gl->gpu;

    p->ra = ra_pl_create(p->gpu, ctx->log, p->pl_log);
    if (!p->ra) {
        pl_opengl_destroy(&p->gl);
        pl_log_destroy(&p->pl_log);
        return MPV_ERROR_VO_INIT_FAILED;
    }

    ctx->ra = p->ra;
    ctx->gpu = p->gpu;
    return 0;
}

static int libmpv_gpu_next_wrap_fbo_gl(struct libmpv_gpu_next_context *ctx,
                    mpv_render_param *params, pl_tex *out_tex)
{
    struct priv *p = ctx->priv;
    *out_tex = NULL;

    mpv_opengl_fbo *fbo =
        get_mpv_render_param(params, MPV_RENDER_PARAM_OPENGL_FBO, NULL);
    if (!fbo)
        return MPV_ERROR_INVALID_PARAMETER;

    pl_tex tex = pl_opengl_wrap(p->gpu, pl_opengl_wrap_params(
        .framebuffer = fbo->fbo,
        .width = fbo->w,
        .height = fbo->h,
        .iformat = fbo->internal_format
    ));

    if (!tex) {
        MP_ERR(ctx, "Failed to wrap provided FBO as a libplacebo texture.\n");
        return MPV_ERROR_GENERIC;
    }

    *out_tex = tex;
    return 0;
}

static void libmpv_gpu_next_done_frame_gl(struct libmpv_gpu_next_context *ctx)
{
}

static void libmpv_gpu_next_destroy_gl(struct libmpv_gpu_next_context *ctx)
{
    struct priv *p = ctx->priv;
    if (!p)
        return;

    if (p->ra) {
        ra_pl_destroy(&p->ra);
    }

    pl_opengl_destroy(&p->gl);
    pl_log_destroy(&p->pl_log);
}

const struct libmpv_gpu_next_context_fns libmpv_gpu_next_context_gl = {
    .api_name = MPV_RENDER_API_TYPE_OPENGL,
    .init = libmpv_gpu_next_init_gl,
    .wrap_fbo = libmpv_gpu_next_wrap_fbo_gl,
    .done_frame = libmpv_gpu_next_done_frame_gl,
    .destroy = libmpv_gpu_next_destroy_gl,
};
#endif

#if HAVE_D3D11 && defined(PL_HAVE_D3D11)

#include <mpv/render_d3d11.h>

struct priv_d3d11 {
    pl_log pl_log;
    pl_d3d11 d3d11;
    pl_gpu gpu;
    struct ra_next *ra;
};

static int libmpv_gpu_next_init_d3d11(struct libmpv_gpu_next_context *ctx, mpv_render_param *params)
{
    ctx->priv = talloc_zero(NULL, struct priv_d3d11);
    struct priv_d3d11 *p = ctx->priv;

    mpv_d3d11_init_params *d3d11_params =
        get_mpv_render_param(params, MPV_RENDER_PARAM_D3D11_INIT_PARAMS, NULL);
    if (!d3d11_params || !d3d11_params->device)
        return MPV_ERROR_INVALID_PARAMETER;

    struct pl_log_params log_params = {
        .log_level = PL_LOG_DEBUG
    };
    p->pl_log = pl_log_create(PL_API_VER, &log_params);

    p->d3d11 = pl_d3d11_create(p->pl_log, pl_d3d11_params(
        .device = d3d11_params->device
    ));
    if (!p->d3d11) {
        MP_ERR(ctx, "Failed to create libplacebo D3D11 context.\n");
        pl_log_destroy(&p->pl_log);
        return MPV_ERROR_UNSUPPORTED;
    }
    p->gpu = p->d3d11->gpu;

    p->ra = ra_pl_create(p->gpu, ctx->log, p->pl_log);
    if (!p->ra) {
        pl_d3d11_destroy(&p->d3d11);
        pl_log_destroy(&p->pl_log);
        return MPV_ERROR_VO_INIT_FAILED;
    }

    ctx->ra = p->ra;
    ctx->gpu = p->gpu;
    return 0;
}

static int libmpv_gpu_next_wrap_fbo_d3d11(struct libmpv_gpu_next_context *ctx,
                    mpv_render_param *params, pl_tex *out_tex)
{
    struct priv_d3d11 *p = ctx->priv;
    *out_tex = NULL;

    mpv_d3d11_fbo *fbo =
        get_mpv_render_param(params, MPV_RENDER_PARAM_D3D11_FBO, NULL);
    if (!fbo || !fbo->tex)
        return MPV_ERROR_INVALID_PARAMETER;

    pl_tex tex = pl_d3d11_wrap(p->gpu, pl_d3d11_wrap_params(
        .tex = fbo->tex,
        .w = fbo->w,
        .h = fbo->h
    ));

    if (!tex) {
        MP_ERR(ctx, "Failed to wrap provided D3D11 texture as a libplacebo texture.\n");
        return MPV_ERROR_GENERIC;
    }

    *out_tex = tex;
    return 0;
}

static void libmpv_gpu_next_done_frame_d3d11(struct libmpv_gpu_next_context *ctx)
{
    struct priv_d3d11 *p = ctx->priv;
    pl_gpu_flush(p->gpu);
}

static void libmpv_gpu_next_destroy_d3d11(struct libmpv_gpu_next_context *ctx)
{
    struct priv_d3d11 *p = ctx->priv;
    if (!p)
        return;

    if (p->ra) {
        ra_pl_destroy(&p->ra);
    }

    pl_d3d11_destroy(&p->d3d11);
    pl_log_destroy(&p->pl_log);
}

const struct libmpv_gpu_next_context_fns libmpv_gpu_next_context_d3d11 = {
    .api_name = MPV_RENDER_API_TYPE_D3D11,
    .init = libmpv_gpu_next_init_d3d11,
    .wrap_fbo = libmpv_gpu_next_wrap_fbo_d3d11,
    .done_frame = libmpv_gpu_next_done_frame_d3d11,
    .destroy = libmpv_gpu_next_destroy_d3d11,
};
#endif

#if HAVE_VULKAN && defined(PL_HAVE_VULKAN)

struct priv_vulkan {
    pl_log pl_log;
    pl_vulkan vulkan;
    pl_gpu gpu;
    struct ra_next *ra;
    // Wrapper around the caller's VkImage, cached across frames instead of
    // rewrapped/destroyed every frame (see wrap_fbo/done_frame_vulkan below
    // for why: destroying it every frame requires a synchronous GPU drain
    // to avoid freeing a view the GPU is still using, which is what used to
    // make this backend's frame pacing track the video's GPU render time
    // instead of the display's). Recreated only when the caller hands us a
    // different VkImage (i.e. on resize).
    pl_tex wrapped_tex;
    VkImage wrapped_image;
    // Whether wrapped_tex is currently "held" (user side owns it). Tracked
    // here because release_ex on an unheld image is an error, and wrap_fbo
    // can run without a matching done_frame (get_target_size).
    bool wrapped_held;
    // The caller's fbo param for the in-progress render call; done_frame
    // reports the post-render image layout into its out_layout field. Only
    // valid for the duration of one mpv_render_context_render() call.
    mpv_vulkan_fbo *current_fbo;
    // Satisfies pl_vulkan_hold_ex's mandatory semaphore param (see
    // done_frame_vulkan). Timeline semaphore so re-signalling it every frame
    // needs no reset/consume step like a binary semaphore would.
    VkSemaphore hold_sem;
    uint64_t hold_sem_value;
    // Write-after-read guard (see wrap_fbo): signalled by an empty submit
    // whose implicit first sync scope covers everything earlier in this
    // queue's submission order — i.e. the host's last blit that read the
    // image — and waited on by libplacebo (via release_ex) before it writes.
    VkSemaphore guard_sem;
    uint64_t guard_sem_value;
    // Raw queue access for the guard submit. The host shares one VkQueue
    // with us (init params), and all submissions happen on its render
    // thread, so no cross-thread locking is needed here.
    VkQueue queue;
    PFN_vkQueueSubmit fp_queue_submit;
};

static int libmpv_gpu_next_init_vulkan(struct libmpv_gpu_next_context *ctx, mpv_render_param *params)
{
    ctx->priv = talloc_zero(NULL, struct priv_vulkan);
    struct priv_vulkan *p = ctx->priv;

    mpv_vulkan_init_params *vk_params =
        get_mpv_render_param(params, MPV_RENDER_PARAM_VULKAN_INIT_PARAMS, NULL);
    if (!vk_params || !vk_params->instance || !vk_params->phys_device || !vk_params->device)
        return MPV_ERROR_INVALID_PARAMETER;

    MP_INFO(ctx, "Vulkan init params:\n");
    MP_INFO(ctx, "  instance:    %p\n", vk_params->instance);
    MP_INFO(ctx, "  phys_device: %p\n", vk_params->phys_device);
    MP_INFO(ctx, "  device:      %p\n", vk_params->device);
    MP_INFO(ctx, "  proc_addr:   %p\n", vk_params->get_proc_addr);
    MP_INFO(ctx, "  qf_index:    %u\n", vk_params->queue_family_index);
    MP_INFO(ctx, "  q_index:     %u\n", vk_params->queue_index);

    struct pl_log_params log_params = {
        .log_level = PL_LOG_DEBUG,
        .log_cb    = pl_log_cb,
        .log_priv  = ctx->log,
    };
    p->pl_log = pl_log_create(PL_API_VER, &log_params);

    // Try with the caller-supplied get_proc_addr first. If libplacebo
    // rejects it, we fall back to NULL (let libplacebo use the Vulkan
    // loader directly) — this is safe because PL_HAVE_VK_PROC_ADDR
    // means we link vulkan-1.dll.
    //
    // Feature declaration: pl_vulkan_import loads functions of extensions
    // promoted to core purely by the device's apiVersion — but promoted-to-
    // core does not mean feature-enabled (synchronization2, pushDescriptor
    // are separate device features). Using them on a device created without
    // those features is undefined behavior (observed as random
    // VK_ERROR_DEVICE_LOST on NVIDIA).
    //
    // If the caller tells us what the device was actually created with
    // (enabled_features), pass that through verbatim. Otherwise assume the
    // bare libplacebo-required minimum (hostQueryReset + timelineSemaphore)
    // and cap the API version at 1.2 so libplacebo never picks up promoted
    // 1.3/1.4 entry points implicitly.
    VkPhysicalDeviceVulkan12Features vk12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .hostQueryReset = VK_TRUE,
        .timelineSemaphore = VK_TRUE,
    };
    VkPhysicalDeviceFeatures2 vkfeat = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &vk12,
    };
    const VkPhysicalDeviceFeatures2 *features = vk_params->enabled_features
        ? (const VkPhysicalDeviceFeatures2 *) vk_params->enabled_features
        : &vkfeat;
    uint32_t max_api_ver = vk_params->enabled_features ? 0 : VK_API_VERSION_1_2;

    PFN_vkGetInstanceProcAddr gpa = (PFN_vkGetInstanceProcAddr) vk_params->get_proc_addr;
    p->vulkan = pl_vulkan_import(p->pl_log, pl_vulkan_import_params(
        .instance   = (VkInstance) vk_params->instance,
        .get_proc_addr = gpa ? gpa : NULL,
        .phys_device = (VkPhysicalDevice) vk_params->phys_device,
        .device      = (VkDevice) vk_params->device,
        .features    = features,
        .extensions  = vk_params->enabled_extensions,
        .num_extensions = (int) vk_params->num_enabled_extensions,
        .max_api_version = max_api_ver,
        .queue_graphics = {
            .index = vk_params->queue_family_index,
            .count = 1,
        },
    ));
    if (!p->vulkan && gpa) {
        // If it failed with the caller's proc_addr, try again with
        // NULL to let libplacebo use the native vkGetInstanceProcAddr.
        MP_WARN(ctx, "Vulkan import with caller proc_addr failed; "
                "retrying with native vkGetInstanceProcAddr...\n");
        p->vulkan = pl_vulkan_import(p->pl_log, pl_vulkan_import_params(
            .instance   = (VkInstance) vk_params->instance,
            .get_proc_addr = NULL,
            .phys_device = (VkPhysicalDevice) vk_params->phys_device,
            .device      = (VkDevice) vk_params->device,
            .queue_graphics = {
                .index = vk_params->queue_family_index,
                .count = 1,
            },
            .features = features,
            .extensions  = vk_params->enabled_extensions,
            .num_extensions = (int) vk_params->num_enabled_extensions,
            .max_api_version = max_api_ver,
        ));
    }
    if (!p->vulkan) {
        MP_ERR(ctx, "Failed to import Vulkan device via libplacebo.\n");
        pl_log_destroy(&p->pl_log);
        return MPV_ERROR_UNSUPPORTED;
    }
    p->gpu = p->vulkan->gpu;

    p->ra = ra_pl_create(p->gpu, ctx->log, p->pl_log);
    if (!p->ra) {
        pl_vulkan_destroy(&p->vulkan);
        pl_log_destroy(&p->pl_log);
        return MPV_ERROR_VO_INIT_FAILED;
    }

    p->hold_sem = pl_vulkan_sem_create(p->gpu, pl_vulkan_sem_params(
        .type = VK_SEMAPHORE_TYPE_TIMELINE,
    ));
    p->guard_sem = pl_vulkan_sem_create(p->gpu, pl_vulkan_sem_params(
        .type = VK_SEMAPHORE_TYPE_TIMELINE,
    ));

    // Queue handle + submit fn for the guard submit in wrap_fbo. gpa may be
    // NULL (the retry path above); PL_HAVE_VK_PROC_ADDR guarantees we link
    // the Vulkan loader, so fall back to the native entry point.
    PFN_vkGetInstanceProcAddr gipa = gpa ? gpa : vkGetInstanceProcAddr;
    VkDevice dev = (VkDevice) vk_params->device;
    PFN_vkGetDeviceProcAddr gdpa = (PFN_vkGetDeviceProcAddr)
        gipa((VkInstance) vk_params->instance, "vkGetDeviceProcAddr");
    PFN_vkGetDeviceQueue get_queue = gdpa
        ? (PFN_vkGetDeviceQueue) gdpa(dev, "vkGetDeviceQueue") : NULL;
    p->fp_queue_submit = gdpa
        ? (PFN_vkQueueSubmit) gdpa(dev, "vkQueueSubmit") : NULL;
    if (get_queue) {
        get_queue(dev, vk_params->queue_family_index, vk_params->queue_index,
                  &p->queue);
    }

    if (!p->hold_sem || !p->guard_sem || !p->queue || !p->fp_queue_submit) {
        MP_ERR(ctx, "Failed to set up Vulkan interop sync objects.\n");
        if (p->hold_sem)
            pl_vulkan_sem_destroy(p->gpu, &p->hold_sem);
        if (p->guard_sem)
            pl_vulkan_sem_destroy(p->gpu, &p->guard_sem);
        ra_pl_destroy(&p->ra);
        pl_vulkan_destroy(&p->vulkan);
        pl_log_destroy(&p->pl_log);
        return MPV_ERROR_VO_INIT_FAILED;
    }

    ctx->ra = p->ra;
    ctx->gpu = p->gpu;
    return 0;
}

static int libmpv_gpu_next_wrap_fbo_vulkan(struct libmpv_gpu_next_context *ctx,
                    mpv_render_param *params, pl_tex *out_tex)
{
    struct priv_vulkan *p = ctx->priv;
    *out_tex = NULL;

    mpv_vulkan_fbo *fbo =
        get_mpv_render_param(params, MPV_RENDER_PARAM_VULKAN_FBO, NULL);
    if (!fbo || !fbo->image)
        return MPV_ERROR_INVALID_PARAMETER;

    VkImage image = (VkImage) fbo->image;

    if (p->wrapped_tex && (p->wrapped_image != image ||
                           p->wrapped_tex->params.w != fbo->w ||
                           p->wrapped_tex->params.h != fbo->h)) {
        // Caller handed us a different VkImage than last frame (a resize).
        // The size check guards against handle reuse: if the caller destroyed
        // its image and the driver gave the replacement the same handle value,
        // the cached wrapper's view references the destroyed image — trust the
        // handle alone and we render through a dead view (device lost).
        // The outgoing wrapper's view must not be freed while GPU work from
        // the last frame might still reference it — drain first. This is a
        // resize-only cost, not a per-frame one.
        pl_gpu_finish(p->gpu);
        pl_tex_destroy(p->gpu, &p->wrapped_tex);
        p->wrapped_image = VK_NULL_HANDLE;
        p->wrapped_held = false;
    }

    if (!p->wrapped_tex) {
        pl_tex tex = pl_vulkan_wrap(p->gpu, pl_vulkan_wrap_params(
            .image  = image,
            .width  = fbo->w,
            .height = fbo->h,
            .format = (VkFormat) fbo->format,
            .usage  = (VkImageUsageFlags) fbo->usage,
        ));
        if (!tex) {
            MP_ERR(ctx, "Failed to wrap VkImage as a libplacebo texture.\n");
            return MPV_ERROR_GENERIC;
        }
        p->wrapped_tex = tex;
        p->wrapped_image = image;
        p->wrapped_held = true; // freshly wrapped images start held
    }

    if (p->wrapped_held) {
        // Write-after-read guard: the host's blit that reads this image
        // lives in a command buffer submitted *after* our render (Qt
        // records its frame in one buffer, submitted at frame end). Without
        // a wait, libplacebo's first write barrier next frame has an empty
        // src scope and can overlap that still-executing read — UB. An
        // empty submit's signal op happens-after everything earlier in this
        // queue's submission order (which by now includes the host's frame
        // containing the blit), so releasing against it orders our next
        // write after that read. No CPU wait anywhere.
        p->guard_sem_value++;
        VkTimelineSemaphoreSubmitInfo tsinfo = {
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &p->guard_sem_value,
        };
        VkSubmitInfo sinfo = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = &tsinfo,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &p->guard_sem,
        };
        p->fp_queue_submit(p->queue, 1, &sinfo, VK_NULL_HANDLE);

        pl_vulkan_release_ex(p->gpu, pl_vulkan_release_params(
            .tex       = p->wrapped_tex,
            .layout    = VK_IMAGE_LAYOUT_UNDEFINED,
            .qf        = VK_QUEUE_FAMILY_IGNORED,
            .semaphore = { p->guard_sem, p->guard_sem_value },
        ));
        p->wrapped_held = false;
    }

    p->current_fbo = fbo;
    *out_tex = p->wrapped_tex;
    return 0;
}

static void libmpv_gpu_next_done_frame_vulkan(struct libmpv_gpu_next_context *ctx)
{
    struct priv_vulkan *p = ctx->priv;
    if (!p->wrapped_tex || p->wrapped_held)
        return;

    // Reclaim the image WITHOUT a layout transition (out_layout mode: query
    // the current layout instead of transitioning to one) and report that
    // layout to the caller via fbo->out_layout. Deliberately no transition
    // here: a transition recorded by libplacebo lives in *our* command
    // buffer, and its barrier's dst scope is empty (upstream hold_ex
    // semantics) — a host that can't wait on `semaphore` (Qt owns its own
    // vkQueueSubmit) has no way to order its read after that transition,
    // which is UB and crashed NVIDIA drivers in practice. By only reporting
    // the layout, the host performs the transition in its own command
    // buffer, in the same submission as its read — trivially ordered, no
    // cross-submission barrier chaining needed at all. This also submits
    // any still-buffered render commands (hold_ex ends and submits the
    // active command buffer). hold_sem exists purely to satisfy hold_ex's
    // mandatory semaphore param; nothing waits on it.
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    p->hold_sem_value++;
    p->wrapped_held = pl_vulkan_hold_ex(p->gpu, pl_vulkan_hold_params(
        .tex        = p->wrapped_tex,
        .out_layout = &layout,
        .qf         = VK_QUEUE_FAMILY_IGNORED,
        .semaphore  = { p->hold_sem, p->hold_sem_value },
    ));

    if (p->current_fbo) {
        p->current_fbo->out_layout = (int) layout;
        p->current_fbo = NULL;
    }
}

static void libmpv_gpu_next_destroy_vulkan(struct libmpv_gpu_next_context *ctx)
{
    struct priv_vulkan *p = ctx->priv;
    if (!p)
        return;

    if (p->wrapped_tex) {
        // Teardown isn't a hot path — a hard drain here to make sure no GPU
        // work still references the wrapper's view is fine (this replaces
        // the per-frame pl_gpu_finish() this backend used to do).
        pl_gpu_finish(p->gpu);
        pl_tex_destroy(p->gpu, &p->wrapped_tex);
    }

    if (p->hold_sem)
        pl_vulkan_sem_destroy(p->gpu, &p->hold_sem);
    if (p->guard_sem)
        pl_vulkan_sem_destroy(p->gpu, &p->guard_sem);

    if (p->ra)
        ra_pl_destroy(&p->ra);

    if (p->vulkan)
        pl_vulkan_destroy(&p->vulkan);

    pl_log_destroy(&p->pl_log);
}

const struct libmpv_gpu_next_context_fns libmpv_gpu_next_context_vulkan = {
    .api_name   = MPV_RENDER_API_TYPE_VULKAN,
    .init       = libmpv_gpu_next_init_vulkan,
    .wrap_fbo   = libmpv_gpu_next_wrap_fbo_vulkan,
    .done_frame = libmpv_gpu_next_done_frame_vulkan,
    .destroy    = libmpv_gpu_next_destroy_vulkan,
    .persistent_target_tex = true,
};
#endif
