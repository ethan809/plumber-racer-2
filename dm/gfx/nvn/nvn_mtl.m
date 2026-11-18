// Metal execution layer of the NVN null device, see nvn_mtl.h.
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <QuartzCore/CAMetalDisplayLink.h>
#import <MetalFX/MetalFX.h>
#undef I   // MetalFX brings in <complex.h>, whose I (the imaginary unit) would shadow the local tables named I below
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#include <mach/mach_time.h>
#include <mach/mach.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <fcntl.h>
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#include <os/proc.h>   // os_proc_available_memory: for the memory readout line
#endif
#include "nvn_mtl.h"
#include "nvn_obj.h"
#include "../tex/nvn_fmt_mtl.h"
#include "../tex/nvn_deswizzle.h"
#include "../tex/nvn_deswizzle_gpu.h"
#include "../tex/nvn_bc_decode.h"
#include "nvn_builtin.h"
#include "host_msl.h"
#include "nvn_dirty.h"
#include "nvn_depth_clear.h"
#include "nvn_refl.h"
#include "nvn_cover.h"
#include "nvn_border.h"
#include "nvn_primitive.h"
#include "nvn_timestamp.h"
#include "dmsh_chain.h"   // DMSH_CHAIN: the translation chain version of the B shader library expected by this source tree, one for every game (beside this file, tools/dmsh_chain.py --header)
#include "game_profile.h"   // game profile constants (profile id, the profile's switches)
// Texture slot table per stage: 0-127 the combined handles (nvnCommandBufferBindTexture), 128-255 the separate texture binds (NVN_SEPTEX_SLOT0, nvn_obj.h)
#define NTEX 256
// The NVN driver's constant segments c0 and c2 (byte offsets, slot counts) and the binding kinds of the shader reflection have one definition, which the shader build reads too (dmsh_build.py through dmsh_abi.py)
#include "dmsh_abi.h"
// c2 holds the handle tables (DMSH_C2_*), c0 the storage buffer table of each stage at DMSH_C0_SSBO_<stage> + 0x10 r (address 64 bits, size 32 bits). The replay fills one DMSH_DRV_SIZE buffer per segment a shader reads
_Static_assert(DMSH_C2_TEX + DMSH_C2_ROW * DMSH_C2_TEX_SLOTS == DMSH_C2_SEPTEX && DMSH_C2_SEPTEX + DMSH_C2_ROW * DMSH_C2_SEP_SLOTS == DMSH_C2_SAMP && DMSH_C2_SAMP + DMSH_C2_ROW * DMSH_C2_SEP_SLOTS <= DMSH_DRV_SIZE && NVN_SEPTEX_SLOT0 + DMSH_C2_SEP_SLOTS == NTEX, "NVN driver handle tables");
_Static_assert(DMSH_C0_CS_UBO + DMSH_C0_ROW * DMSH_C0_CS_UBO_SLOTS == DMSH_C0_SSBO_CS && DMSH_C0_SSBO_PS + DMSH_C0_ROW * DMSH_C0_SSBO_SLOTS <= DMSH_C0_POINT_SIZE && DMSH_C0_POINT_SIZE + 4 <= DMSH_C0_SAMPLE_TABLE &&
               DMSH_C0_SAMPLE_TABLE + DMSH_C0_SAMPLE_BYTES <= DMSH_C0_SAMPLE_INFO && DMSH_C0_SAMPLE_INFO + 16 <= DMSH_C0_BASE_INSTANCE && DMSH_C0_BASE_INSTANCE + 4 <= DMSH_DRV_SIZE, "NVN driver c0 tables");
// Geometry's counter/output addresses are c0+0x410/0x430 (NieR:Automata's MOVC/IADDC pairs); they use the logical NVN stage even when Metal executes a compute kernel.
static int c0_ssbo_base(int stage) { return stage == 0 ? DMSH_C0_SSBO_VS : stage == 1 ? DMSH_C0_SSBO_PS : stage == 2 ? DMSH_C0_SSBO_GS : stage == 3 ? DMSH_C0_SSBO_TS : stage == 4 ? DMSH_C0_SSBO_MS : stage == 5 ? DMSH_C0_SSBO_CS : -1; }   // NVN stages 0 vertex, 1 pixel, 2 geometry, 5 compute; slots 3 / 4 hold NVN2's task / mesh stages (hle_nvn.c st_slot), hull / domain never seen
// The handle a texture binding of slot reg reads: as c2 holds it (encode_args fills c2 the same way), a combined slot past 31 shares its word with an image slot, and a bound image
// is what lies there. A texture instruction on an image handle names such a slot: a size query of the image a compute shader writes (Unreal's GetDimensions on a RWTexture3D,
// dmsh_build.py decode_tex). DRAGON QUEST VII Reimagined's volumetric lightmap indirection patch read the empty texture slot 32 there, got a 1x1 size and wrote only cell (0, 0, 0)
static inline uint64_t tex_slot_h(const uint64_t *tex, const uint64_t *img, unsigned reg) {
    return reg >= 32 && reg - 32 < DMSH_C2_IMG_SLOTS && img[reg - 32] ? img[reg - 32] : reg < NTEX ? tex[reg] : 0;
}
static uint64_t c2_handle(int stage, unsigned b);   // the handle c2 holds at byte b (defined with the draw state, TEXA / SAMPA)
#include "nvn_hmap.h"
#include "../../host/dm.h"
#import <CommonCrypto/CommonDigest.h>
#include <stdarg.h>
#include <mach-o/dyld.h>
#include <mach-o/fat.h>
#include <libkern/OSByteOrder.h>
#include <libgen.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdatomic.h>
#include <errno.h>
#include <execinfo.h>
#include <os/lock.h>
#include <objc/runtime.h>
// Function parameters in this file are never retained again in the callee (the caller / global tables already hold them, and lifetime is guaranteed by defer_obj_release): saves a retain / release pair per call on the hot path
#pragma clang attribute push(__attribute__((objc_externally_retained)), apply_to = function)

#define MLOG(...) do { flockfile(stderr); fprintf(stderr, "[dm] [nvn-mtl] " __VA_ARGS__); fputc('\n', stderr); funlockfile(stderr); } while (0)   // one line under one lock, as LOG (dm.h)
#define MDIE(...) do { MLOG("FATAL " __VA_ARGS__); abort(); } while (0)

// Explicit experimental override for an already-built shader library from a
// user's own IPA. Do not enable by default: a matching input stamp and stable
// host ABI do not prove that an older translator produced compatible shaders.
static const char *nvn_shader_chain(void) {
    const char *s = getenv("DM_NVN_SHADER_CHAIN");
    if (!s || strlen(s) != 12) return DMSH_CHAIN;
    for (int i = 0; i < 12; i++) if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f'))) return DMSH_CHAIN;
    return s;
}

static void warm_start(void);
static void shmap_load(void);
static int prepare_on(void);
static int g_warm_started;
static void warm_claim(void);

static id<MTLDevice> g_dev;
// Experimental, opt-in bridge for pre-existing BC assets on Apple GPUs without
// BC support. RGBA8 is a temporary target; ASTC encoding is not yet integrated.
static int bc_cpu_fallback(void) {
    static int enabled = -1;
    if (enabled < 0) { const char *s = getenv("DM_NVN_BC_RGBA"); enabled = s && !strcmp(s, "1"); }
    return enabled && g_dev && !g_dev.supportsBCTextureCompression;
}
// Return the BC block family and normalized signedness for a Metal format.
// BC6H is deliberately excluded: its HDR pixels cannot be represented in RGBA8.
static int bc_rgba_kind(MTLPixelFormat pf, int *sign) {
    *sign = 0;
    switch (pf) {
        case MTLPixelFormatBC1_RGBA: case MTLPixelFormatBC1_RGBA_sRGB: return 1;
        case MTLPixelFormatBC2_RGBA: case MTLPixelFormatBC2_RGBA_sRGB: return 2;
        case MTLPixelFormatBC3_RGBA: case MTLPixelFormatBC3_RGBA_sRGB: return 3;
        case MTLPixelFormatBC4_RSnorm: *sign = 1; return 4;
        case MTLPixelFormatBC4_RUnorm: return 4;
        case MTLPixelFormatBC5_RGSnorm: *sign = 1; return 5;
        case MTLPixelFormatBC5_RGUnorm: return 5;
        case MTLPixelFormatBC7_RGBAUnorm: case MTLPixelFormatBC7_RGBAUnorm_sRGB: return 7;
        default: return 0;
    }
}
static int bc_rgba_srgb(MTLPixelFormat pf) {
    return pf == MTLPixelFormatBC1_RGBA_sRGB || pf == MTLPixelFormatBC2_RGBA_sRGB ||
           pf == MTLPixelFormatBC3_RGBA_sRGB || pf == MTLPixelFormatBC7_RGBAUnorm_sRGB;
}

// The shaders the host writes itself (built-in kernels, clears, copies, index conversions, geometry packing, the present: dm/gfx/msl), compiled when the host is built and loaded by
// name (host_msl_lib), so none is compiled at run time. The diagnostic overrides (DM_NVN_HILITE, DM_NVN_MAGSCAN, DM_NVN_TEXDUMP) still compile theirs from source
static id<MTLLibrary> host_lib(const char *name, NSError **e) { return host_msl_lib(g_dev, nil, name, e); }
static NSMutableArray *g_l3_gpu_leaks;
static int g_nvn_deswz_gpu;
static id<MTLCommandQueue> g_q;
static id<MTL4CommandQueue> g_q4;
static int g_classic_ok, g_classic;   // mtl_device_classic was asked / the device has no Metal 4 and only the classic part started (mtl_init)
static dispatch_queue_t g_feedback_q;
static _Atomic uint32_t g_present_qos;
#define ALLOC_MAX 8
static id<MTL4CommandAllocator> g_alloc4[ALLOC_MAX]; static int g_alloc_n, g_alloc_cur;   // Command allocator pool of the main submission: at start pick one the GPU has finished with, create a new one if there is none (wait only at the cap)
static id<MTL4CommandAllocator> g_up_alloc4[3];
static id<MTL4CommandAllocator> g_present_alloc4[3];
static id<MTL4CommandBuffer> g_present_cb[3];
static id<MTL4ArgumentTable> g_vt, g_ft, g_ct, g_up_ct, g_ot, g_mt;   // g_ot / g_mt: the object and mesh stages of an NVN2 mesh draw (mesh_draw_setup)
static int g_vbuf_cache = 1;   // DM_NVN_VBUF_CACHE=0 restores the uncached path for A/B measurements
static uint64_t g_vt_addr[31]; static uint32_t g_vt_valid;
static uint64_t g_vbuf_resolve, g_vbuf_reuse, g_vbuf_set, g_vbuf_skip, g_vbuf_copies, g_vbuf_copy_bytes;   // copies: vertex streams at addresses not 4-byte aligned (vbuf_unaligned)
static id<MTL4ArgumentTable> g_present_args;
static id<MTLBuffer> g_present_uv[3];
static uint64_t g_present_alloc_fence[3];
// Under g_pq_mu: per present allocator, the newest ticket its commit feedback reported done on the GPU. present_now resets an allocator only once this reaches the ticket that
// last used it. g_pq_done is not that: every display-link callback sets it to ticket - 1 (the previous present was committed, not run) and a present that was not submitted sets
// it to its own ticket. Waiting on g_pq_done, a display-link present never waited here: with three presents in flight, or two after a dropped frame, the allocator was reset
// under a present the GPU had not run, and that present's drawable read back black (t_present_alloc; docs/reference/frame-rate.md)
static uint64_t g_present_alloc_done[3];
static pthread_mutex_t g_pq_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_pq_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_link_cv = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t g_link_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_link_tick_ns, g_link_tick_n;
static uint64_t g_link_dropped;
static uint64_t g_link_empty;   // Under g_pq_mu: ticks that found no new frame queued (counted here, not as ticks minus presents: presents before the link starts and while it is rescued are not on ticks)
static double g_link_hold_until; static uint64_t g_link_held;   // Changed only on the display-link callback thread: the time the previous frame must stay on screen until by its present interval (0 = no hold), and the ticks that took no new frame because of it
static int g_pq_inflight;   // Under g_pq_mu: number of presents committed to the GPU whose completion callback has not arrived yet
int g_dm_present_link;   // Present pacing mode (runtime value of DM_PRESENT_LINK, modified directly during same-process alternating measurement): 0 = acquire the drawable itself; 1 = display link at 60 Hz; 2 = display link at 120 Hz, the game presents a frame every other display-link tick
static int g_present_latency;   // DM_PRESENT_LATENCY: the display link's frame latency in a window and in fullscreen alike (0 = not set: link_latency picks it)
static uint64_t g_pq_done;
static uint64_t g_pq_read;
static id<MTLResidencySet> g_res;
static int g_res_dirty;
static id<MTLSharedEvent> g_ev;
static _Atomic uint64_t g_fence;                // Highest committed fence value (incremented exclusively by the submit thread)
static int g_cut_copy;                          // a buffer â†’ texture copy was encoded since the last commit: the next pass boundary commits the segment (sub_cut)
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static _Atomic uint64_t g_done;                 // GPU complete and all CPU-visible effects through this fence published
static uint64_t g_arena_made;
static int g_allocprof = -1; static uint64_t g_ap_bytes[64], g_ap_fn[8];
// Diagnostic only: inclusive wall/CPU time on the replay thread. No GPU waits
// are added, and no clocks are read when DM_NVN_CPUPROF is absent or zero.
typedef struct { uint64_t wall, cpu; } cp_stamp;
static _Thread_local uint64_t g_cp_wall[8], g_cp_cpu[8], g_cp_calls[8], g_cp_last;
static _Thread_local uint64_t g_cp_geo_waited, g_cp_geo_safe;
static int cp_on(void) { static _Thread_local int on = -1; if (on < 0) { const char *e = getenv("DM_NVN_CPUPROF"); on = e && atoi(e); } return on; }
static cp_stamp cp_begin(void) {
    if (!cp_on()) return (cp_stamp){0};
    return (cp_stamp){clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW), clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID)};
}
static void cp_end(int phase, cp_stamp start) {
    if (!start.wall) return;
    g_cp_cpu[phase] += clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID) - start.cpu;
    g_cp_wall[phase] += clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW) - start.wall;
    g_cp_calls[phase]++;
}
static void cp_report(void) {
    if (!cp_on()) return;
    uint64_t now = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
    if (!g_cp_last) { g_cp_last = now; return; }
    if (now - g_cp_last < 5000000000ull) return;
    static const char *names[] = {"pass_end", "pass_begin", "pipeline", "arguments", "texture", "draw", "geometry", "index"};
    char line[1000] = {0}; size_t at = 0;
    for (int i = 0; i < 8; i++) if (g_cp_calls[i])
        at += snprintf(line + at, sizeof line - at, " %s %llu calls %.2f/%.2f ms CPU/wall;", names[i], (unsigned long long)g_cp_calls[i], g_cp_cpu[i] / 1e6, g_cp_wall[i] / 1e6);
    MLOG("[cpuprof] inclusive phases over %.2f s (nested phases overlap):%s", (now - g_cp_last) / 1e9, line);
    if (g_cp_geo_waited || g_cp_geo_safe) MLOG("[cpuprof] geometry CPU reads: waited %llu / proven safe %llu", (unsigned long long)g_cp_geo_waited, (unsigned long long)g_cp_geo_safe);
    memset(g_cp_wall, 0, sizeof g_cp_wall); memset(g_cp_cpu, 0, sizeof g_cp_cpu); memset(g_cp_calls, 0, sizeof g_cp_calls); g_cp_last = now;
    g_cp_geo_waited = g_cp_geo_safe = 0;
}
#define AP_B uint64_t _ap0 = g_allocprof > 0 ? g_dev.currentAllocatedSize : 0; cp_stamp _cp0 = cp_begin()
#define AP_E(i) do { if (g_allocprof > 0) { uint64_t _n = g_dev.currentAllocatedSize; if (_n > _ap0) g_ap_fn[i] += _n - _ap0; } cp_end(i, _cp0); } while (0)   // Per-frame arenas created so far (4 MiB each; for the memory report)
static _Atomic int g_last_commit_error;
static _Atomic int g_up_pending;

static pthread_mutex_t g_res_mu = PTHREAD_MUTEX_INITIALIZER;   // Dedicated lock for the residency set: res_add is called on paths that hold g_mu (arena_new), so it cannot share g_mu
// Wait ledger (DM_NVN_WAITS=1): every place a thread of the host or the game waits on the CPU for the GPU or for the encode thread, counted per frame (calls, calls that blocked,
// the time they blocked) and logged every 300 frames with the frame's GPU idle. The game's own SyncWait is split by how far its fence is behind the newest one (0: the last submission)
enum { WS_SYNC, WS_SYNC_NEWEST, WS_SYNC_POLL, WS_ACQ, WS_FINISH, WS_ALLOC, WS_VIS, WS_PG, WS_MDI, WS_READTEX, WS_T2T, WS_DIAG, WS_SNAP, WS_UPZERO, WS_UPALLOC, WS_DRAIN, WS_UBC, WS_OLDQ, WS_N };
static const char *const ws_name[WS_N] = {"SyncWait (game, older fence)", "SyncWait (game, newest fence)", "SyncWait polls", "SyncWait on an acquire sync", "QueueFinish", "allocator", "occlusion ring", "PASSGPU ring",
    "indirect-count multi-draw", "texture readback", "texture copy t2t", "diagnostics", "snapshot ring", "upload batches", "upload slot", "encode thread drain", "uniform copy worker", "old-queue command buffer"};
static _Atomic uint64_t g_ws_n[WS_N], g_ws_blk[WS_N], g_ws_ns[WS_N];
static int ws_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_NVN_WAITS"); v = e && atoi(e) > 0; } return v; }
static inline uint64_t ws_t0(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static inline void ws_put(int site, uint64_t t0, int blocked) {
    atomic_fetch_add_explicit(&g_ws_n[site], 1, memory_order_relaxed);
    if (blocked) { atomic_fetch_add_explicit(&g_ws_blk[site], 1, memory_order_relaxed); atomic_fetch_add_explicit(&g_ws_ns[site], ws_t0() - t0, memory_order_relaxed); }
}
// For the game's waits: when each fence's command buffer ended on the GPU and when its feedback handler ran (CACurrentMediaTime), so a wait that blocked tells its wake-up delay
static struct { _Atomic uint64_t fence, wfence; double gpu_end, fb, wake; _Atomic int done; } g_fend[64];
static _Atomic uint64_t g_ws_wake_n, g_ws_wake_ns, g_ws_fb_ns;
static void ws_wake_pair(unsigned k) {   // both halves known: the wake-up delay after the GPU end, counted once
    if (atomic_load(&g_fend[k].fence) != atomic_load(&g_fend[k].wfence) || atomic_exchange(&g_fend[k].done, 1)) return;
    atomic_fetch_add(&g_ws_wake_n, 1); atomic_fetch_add(&g_ws_wake_ns, (uint64_t)(fmax(0, g_fend[k].wake - g_fend[k].gpu_end) * 1e9)); atomic_fetch_add(&g_ws_fb_ns, (uint64_t)(fmax(0, g_fend[k].fb - g_fend[k].gpu_end) * 1e9));
}
static void ws_wake(uint64_t fence) {   // a blocking SyncWait on this fence has woken
    unsigned k = fence % 64; g_fend[k].wake = CACurrentMediaTime(); atomic_store(&g_fend[k].wfence, fence); ws_wake_pair(k);
    static int once; if (!once++) { char nm[64] = ""; pthread_getname_np(pthread_self(), nm, sizeof nm); MLOG("[waits] first blocking SyncWait on the newest fence: thread '%s'", nm); }
}
static void ws_fb(uint64_t fence, double gpu_end) {   // the fence's feedback handler
    unsigned k = fence % 64; if (atomic_load(&g_fend[k].fence) != fence) { atomic_store(&g_fend[k].done, 0); g_fend[k].gpu_end = gpu_end; g_fend[k].fb = CACurrentMediaTime(); atomic_store(&g_fend[k].fence, fence); }
    ws_wake_pair(k);
}
static _Atomic uint64_t g_ws_post[5];
static void ws_report(uint64_t frame, uint64_t frames) {
    char line[2048]; size_t o = 0;
    { uint64_t e = atomic_exchange(&g_ws_post[1], 0), sp = atomic_exchange(&g_ws_post[2], 0), ts = atomic_exchange(&g_ws_post[3], 0), oc = atomic_exchange(&g_ws_post[4], 0);
      if (e + sp + ts + oc) o += snprintf(line + o, sizeof line - o, " CPU-side writes at GPU end: events %.2f, samples-passed reports %.2f, timestamp reports %.2f, other counter reports %.2f;", (double)e / frames, (double)sp / frames, (double)ts / frames, (double)oc / frames); }
    uint64_t wn = atomic_exchange(&g_ws_wake_n, 0), wns = atomic_exchange(&g_ws_wake_ns, 0), fns = atomic_exchange(&g_ws_fb_ns, 0);
    if (wn) o += snprintf(line + o, sizeof line - o, " newest-fence SyncWait wakes %.3f ms after the GPU end (feedback handler at %.3f ms);", wns / 1e6 / wn, fns / 1e6 / wn);
    for (int i = 0; i < WS_N; i++) { uint64_t n = atomic_exchange(&g_ws_n[i], 0), b = atomic_exchange(&g_ws_blk[i], 0), ns = atomic_exchange(&g_ws_ns[i], 0);
        if (n && o < sizeof line - 160) o += snprintf(line + o, sizeof line - o, " %s %.2f/frame (blocked %.2f, %.3f ms);", ws_name[i], (double)n / frames, (double)b / frames, ns / 1e6 / frames); }
    MLOG("[waits] up to frame %llu, per frame:%s", (unsigned long long)frame, o ? line : " none");
}
#include "nvn_rec.h"   // deferred encoding (DM_NVN_ENC_THREAD)
static NSHashTable *g_res_objects;
static _Atomic unsigned g_image_heap_n;
static inline MTL4VisibilityOptions resource_vis(void) { return MTL4VisibilityOptionDevice | (g_image_heap_n ? MTL4VisibilityOptionResourceAlias : 0); }
// Residency churn by call site (the power line every 300 frames, mtl_res_stats): each commit with changes is a kernel call whose cost grows with what was added or removed
enum { RS_SITES = 16 }; static struct { int line; uint64_t n, bytes; } g_rs_site[RS_SITES]; static uint64_t g_rs_rm, g_rs_commits, g_rs_commit_ns;
#define res_add(r) res_add_at((r), __LINE__)
// Both run on guest threads too (pool_res from mtl_addr, uploads), whose implicit autorelease pools never drain; containsObject: / removeObject: on a weak NSHashTable leave the object
// autoreleased in the calling thread's pool (measured in a standalone test: the object outlived its last strong reference until the thread ended), so each has a pool of its own
static void res_add_at(id<MTLAllocation> allocation, int line) { @autoreleasepool {
    if (!allocation) return;
    // An indirectly addressed placed texture needs its heap resident. The heap owner below removes it only after the last texture/view has gone.
    id<MTLAllocation> r = [(id)allocation conformsToProtocol:@protocol(MTLResource)] && ((id<MTLResource>)allocation).heap ? (id)((id<MTLResource>)allocation).heap : allocation;
    pthread_mutex_lock(&g_res_mu);
    if (!g_res_objects) g_res_objects = [NSHashTable weakObjectsHashTable];
    if ([g_res_objects containsObject:r]) { pthread_mutex_unlock(&g_res_mu); return; }   // already resident (a shared texture resolved again): marking the set dirty would cost a commit that changes nothing
    { int k = 0; while (k < RS_SITES - 1 && g_rs_site[k].line && g_rs_site[k].line != line) k++; g_rs_site[k].line = line; g_rs_site[k].n++; g_rs_site[k].bytes += r.allocatedSize; }
    [g_res addAllocation:(id<MTLAllocation>)r];
    [g_res_objects addObject:r];
    g_res_dirty = 1;
    pthread_mutex_unlock(&g_res_mu);
} }
static void res_remove(id<MTLAllocation> r) { @autoreleasepool {
    if (!r) return;
    if ([(id)r conformsToProtocol:@protocol(MTLResource)] && ((id<MTLResource>)r).heap) return;
    pthread_mutex_lock(&g_res_mu);
    if (![g_res_objects containsObject:r]) { pthread_mutex_unlock(&g_res_mu); return; }
    [g_res removeAllocation:(id<MTLAllocation>)r];
    [g_res_objects removeObject:r];
    g_res_dirty = 1; g_rs_rm++;
    pthread_mutex_unlock(&g_res_mu);
} }
@interface NvImageHeap : NSObject { @public id<MTLHeap> heap; } @end
@implementation NvImageHeap
- (void)dealloc { res_remove(heap); atomic_fetch_sub(&g_image_heap_n, 1); }
@end
static char g_image_heap_owner_key;
// DM_RES_CHECK=1 (diagnostic): if textures / buffers the shader accesses through the argument block are not in the residency set, print the first 40 (textures are attributed to the actual allocation via parent texture / underlying buffer)
static int g_res_check_on;
static void res_check(id r, const char *what, int reg) {
    static int nbad;
    if (!g_res_check_on || !r || nbad >= 40) return;
    int is_tex = [r conformsToProtocol:@protocol(MTLTexture)], is_buf = [r conformsToProtocol:@protocol(MTLBuffer)];
    if (!is_tex && !is_buf) return;   // Samplers etc. are not allocations
    id<MTLResource> a = r; int hops = 0;
    if (is_tex) { id<MTLTexture> t = (id<MTLTexture>)a; while (t.parentTexture) { t = t.parentTexture; hops++; } a = t.buffer ? (id<MTLResource>)t.buffer : (id<MTLResource>)t; }
    pthread_mutex_lock(&g_res_mu); int ok = [g_res_objects containsObject:a.heap ? (id)a.heap : a]; pthread_mutex_unlock(&g_res_mu);
    if (!ok) { nbad++; id<MTLTexture> t = is_tex ? (id<MTLTexture>)r : nil;
        MLOG("[RED] missing residency #%d: %s slot %d %s %p (parent chain %d deep) allocated %lu B storage mode %lu%s", nbad, what, reg, is_tex ? "texture" : "buffer", (__bridge void *)a, hops, (unsigned long)a.allocatedSize, (unsigned long)a.storageMode,
             t ? [NSString stringWithFormat:@" format %lu %lux%lu type %lu usage %#lx", (unsigned long)t.pixelFormat, (unsigned long)t.width, (unsigned long)t.height, (unsigned long)t.textureType, (unsigned long)t.usage].UTF8String : ""); }
}
static void res_commit(void) {
    pthread_mutex_lock(&g_res_mu);
    if (g_res_dirty) { uint64_t t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW); [g_res commit]; g_res_dirty = 0; g_rs_commits++; g_rs_commit_ns += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0; }
    pthread_mutex_unlock(&g_res_mu);
}
// Since the last call: residency commits (count, time), additions by source line (count, MB; the four largest by bytes) and removals
void mtl_res_stats(char *out, size_t n) {
    pthread_mutex_lock(&g_res_mu);
    uint64_t tn = 0, tb = 0; for (int k = 0; k < RS_SITES; k++) { tn += g_rs_site[k].n; tb += g_rs_site[k].bytes; }
    size_t o = snprintf(out, n, "commits %llu / %.1f ms, added %llu, %.0f MB (", (unsigned long long)g_rs_commits, g_rs_commit_ns / 1e6, (unsigned long long)tn, tb / 1048576.0);
    for (int i = 0; i < 4 && o < n; i++) { int best = -1; for (int k = 0; k < RS_SITES; k++) if (g_rs_site[k].n && (best < 0 || g_rs_site[k].bytes > g_rs_site[best].bytes)) best = k;
        if (best < 0) break; o += snprintf(out + o, n - o, "%sline %d: %llu, %.0f MB", i ? ", " : "", g_rs_site[best].line, (unsigned long long)g_rs_site[best].n, g_rs_site[best].bytes / 1048576.0); g_rs_site[best].n = 0; }
    if (o < n) snprintf(out + o, n - o, "), removed %llu", (unsigned long long)g_rs_rm);
    memset(g_rs_site, 0, sizeof g_rs_site); g_rs_rm = g_rs_commits = g_rs_commit_ns = 0;
    pthread_mutex_unlock(&g_res_mu);
}

@interface NvDeferredRelease : NSObject { @public id object; uint64_t fence; int keep_resident; } @end
@implementation NvDeferredRelease @end
static NSMutableArray<NvDeferredRelease *> *g_deferred;
static pthread_mutex_t g_deferred_mu = PTHREAD_MUTEX_INITIALIZER;
// Deferred release has two meanings that must be kept apart (the root of the GPU memory fault on the first frame on entering Travincal):
//   defer_obj_release = this object is retired: drop the reference after the submissions that used it complete, and remove it from the residency set;
//   defer_obj_unref   = only the reference I hold is no longer wanted, the object itself may still be in use (the view cache of a texture ID may hold the base texture itself): only drop the reference, leave the residency set alone.
// Previously only the former existed: when the view cache was invalidated, a texture that 'is still the current storage of some serial' was also removed from the residency set. Invalidation used to happen only when the base texture was really replaced, so it was never hit;
// after the transparency chain's in-place composite swapped the storage of two textures, each of the two live textures was removed once this way, and afterwards they are used as attachments and sampled every frame as usual - the GPU accesses non-resident memory.
static void defer_obj_release_ex(id obj, int keep_resident) {
    if (!obj) return;
    NvDeferredRelease *d = [NvDeferredRelease new]; d->object = obj; d->fence = g_fence + 1; d->keep_resident = keep_resident;
    pthread_mutex_lock(&g_deferred_mu);
    if (!g_deferred) g_deferred = [NSMutableArray array];
    [g_deferred addObject:d];
    pthread_mutex_unlock(&g_deferred_mu);
}
static void defer_obj_release(id obj) { defer_obj_release_ex(obj, 0); }
static void defer_obj_unref(id obj) { defer_obj_release_ex(obj, 1); }
// Its own autorelease pool: mtl_sub_begin drains before it pushes the submission's pool, on the guest submit thread, whose implicit pool is never drained. The weak-table lookups and the residency-set removal
// autorelease the retired object, so whatever that thread drained (instead of the completion handler, which wins the race only when the CPU is idle) stayed alive for good: in Metroid Dread every
// re-upload of its map mask kept the old texture and its 32 MB upload staging block, and GPU memory grew by several GB within minutes under CPU load.
static void defer_drain(void) { @autoreleasepool {
    NSMutableArray<NvDeferredRelease *> *ready = [NSMutableArray array];
    pthread_mutex_lock(&g_deferred_mu);
    for (NSInteger i = g_deferred.count - 1; i >= 0; i--) {
        NvDeferredRelease *d = g_deferred[(NSUInteger)i];
        if (d->fence <= g_done) { [ready addObject:d]; [g_deferred removeObjectAtIndex:(NSUInteger)i]; }
    }
    pthread_mutex_unlock(&g_deferred_mu);
    for (NvDeferredRelease *d in ready) {
        id obj = d->object;
        if (!d->keep_resident && [obj conformsToProtocol:@protocol(MTLResource)]) {
            pthread_mutex_lock(&g_res_mu);
            int registered = [g_res_objects containsObject:obj];
            pthread_mutex_unlock(&g_res_mu);
            if (registered) res_remove((id<MTLResource>)obj);
        }
        d->object = nil;
    }
} }

// A profile list of shader keys the host asks the next shader build for (bf_want.tsv, dual_want.tsv): each key appended once to the file the environment names. A shader's key is
// its 16 hex digits, with or without the route suffix (NvShader keys are "<key>_b"), as dmsh_build.py reads the list
typedef struct { const char *env; pthread_mutex_t mu; char **keys; size_t n, cap; int loaded; } want_list;
static void want_note(want_list *w, const char *key) {
    const char *path = getenv(w->env);
    if (!path || !key || strlen(key) < 16 || (key[16] && key[16] != '_')) return;
    for (int i = 0; i < 16; i++) if (!((key[i] >= '0' && key[i] <= '9') || (key[i] >= 'a' && key[i] <= 'f') || (key[i] >= 'A' && key[i] <= 'F'))) return;
    char k16[17]; memcpy(k16, key, 16); k16[16] = 0;
    pthread_mutex_lock(&w->mu);
    if (!w->loaded) {
        FILE *f = fopen(path, "r"); char line[64];
        if (f) { while (fgets(line, sizeof line, f)) { char *e = strpbrk(line, "\r\n"); if (e) *e = 0; if (strlen(line) != 16) continue;
                if (w->n == w->cap) { w->cap = w->cap ? w->cap * 2 : 64; w->keys = realloc(w->keys, w->cap * sizeof *w->keys); }
                w->keys[w->n++] = strdup(line); } fclose(f); }
        w->loaded = 1;
    }
    for (size_t i = 0; i < w->n; i++) if (!strcmp(w->keys[i], k16)) { pthread_mutex_unlock(&w->mu); return; }
    if (w->n == w->cap) { w->cap = w->cap ? w->cap * 2 : 64; w->keys = realloc(w->keys, w->cap * sizeof *w->keys); }
    w->keys[w->n++] = strdup(k16);
    FILE *f = fopen(path, "a"); if (f) { fprintf(f, "%s\n", k16); fclose(f); }
    pthread_mutex_unlock(&w->mu);
}
static want_list g_bf_want = {"DM_NVN_BF_WANT", PTHREAD_MUTEX_INITIALIZER}, g_dual_want = {"DM_NVN_DUAL_WANT", PTHREAD_MUTEX_INITIALIZER};
void bf_want_note(const char *key) { want_note(&g_bf_want, key); }

// ---------- Memory pools ----------
typedef struct { uintptr_t lo, hi; uintptr_t base; __unsafe_unretained id<MTLBuffer> buf; uint64_t gpu; const void *key; int res; uint64_t born; } pool_ent;   // res: already in the residency set (on demand, see pool_res); born: when it was added (ns)
static pool_ent *g_pools; static int g_npool, g_cap;
static int g_pidx_ok, g_plast = -1;   // whether the addressâ†’pool index is valid, the last hit (pool_find)
static NSMutableArray *g_keep;                  // Holds the buffer references (pool_ent has non-owning pointers)
static uint64_t g_pool_adds;
static int g_pool_res_n;   // registered pools already in the residency set (for the memory readout; under g_mu)

#define C1_BANK_SIZE (4u << 20)
#define C1_DECL_SIZE (64u << 10)
static NSMutableArray *g_c1_banks;
static size_t g_c1_bank_off, g_c1_alloc_bytes, g_c1_data_bytes;
static uint64_t g_c1_count, g_c1_exclusive;
static NSMutableSet<NSString *> *g_c1_full;
static int g_c1_full_valid;
static NSDictionary<NSString *, NSString *> *g_cb_ext;   // cb_extent.tsv: shader key â†’ "reg:bytes ..." (how far into each uniform slot the shader can read); nil when stale or missing
static os_unfair_lock g_c1_lock = OS_UNFAIR_LOCK_INIT;

// DM_UDC_SELFTEST=1 (diagnostic, checks DontCare behavior on a real A19 device): private-storage Depth32Float_Stencil8 / RGBA8 (1952x900) are first cleared to write known values,
// then Load + DontCare without drawing, read back and compared pixel by pixel, printing one result line
static void udc_selftest(void) {
    if (!getenv("DM_UDC_SELFTEST")) return;
    id<MTLCommandQueue> q = [g_dev newCommandQueue];
    for (int depth = 1; depth >= 0; depth--) {
        NSUInteger w = 1952, h = 900;
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:depth ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatRGBA8Unorm width:w height:h mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; td.storageMode = MTLStorageModePrivate;
        id<MTLTexture> t = [g_dev newTextureWithDescriptor:td];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        for (int pass = 0; pass < 2; pass++) {
            MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
            MTLLoadAction la = pass ? MTLLoadActionLoad : MTLLoadActionClear; MTLStoreAction sa = pass ? MTLStoreActionDontCare : MTLStoreActionStore;
            if (depth) { rp.depthAttachment.texture = t; rp.depthAttachment.loadAction = la; rp.depthAttachment.storeAction = sa; rp.depthAttachment.clearDepth = 0.3125;
                         rp.stencilAttachment.texture = t; rp.stencilAttachment.loadAction = la; rp.stencilAttachment.storeAction = sa; rp.stencilAttachment.clearStencil = 0x5a; }
            else { rp.colorAttachments[0].texture = t; rp.colorAttachments[0].loadAction = la; rp.colorAttachments[0].storeAction = sa; rp.colorAttachments[0].clearColor = MTLClearColorMake(0.2, 0.4, 0.6, 0.8); }
            [[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
        }
        id<MTLBuffer> out = [g_dev newBufferWithLength:w * h * 4 options:MTLResourceStorageModeShared];
        id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
        [b copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1) toBuffer:out destinationOffset:0
            destinationBytesPerRow:w * 4 destinationBytesPerImage:w * h * 4 options:depth ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionNone];
        [b endEncoding]; [cb commit]; [cb waitUntilCompleted];
        const uint32_t *px = out.contents; NSUInteger bad = 0; for (NSUInteger i = 0; i < w * h; i++) if (px[i] != px[0]) bad++;
        MLOG("UDC self-check %s %s %lux%lu first value %08x, values differing from it %lu / %lu", g_dev.name.UTF8String, depth ? "Depth32F_S8" : "RGBA8", (unsigned long)w, (unsigned long)h, px[0], (unsigned long)bad, (unsigned long)(w * h));
    }
}
void mtl_init(void) {
    if (g_dev) return;
    const char *vbc = getenv("DM_NVN_VBUF_CACHE"); g_vbuf_cache = !(vbc && !strcmp(vbc, "0"));
    const char *link = getenv("DM_PRESENT_LINK");
    g_dm_present_link = link ? atoi(link) : 1; if (g_dm_present_link < 0 || g_dm_present_link > 2) g_dm_present_link = 1;   // Default 1 (real-device same-process alternation: when load exceeds one display-link period, long frames 15% â†’ 0 and frame intervals even out; no difference from 0 at a steady 60); DM_PRESENT_LINK=0 turns it off   // 1 = present on every tick of a 60 Hz display link; 2 = 120 Hz display link, the game presents a frame every other display-link tick (when behind, it can present on the next 8.33 ms display-link tick)
    const char *latency = getenv("DM_PRESENT_LATENCY");
    if (latency && atoi(latency) > 0) g_present_latency = atoi(latency);
    g_res_check_on = getenv("DM_RES_CHECK") != NULL;
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) MDIE("no Metal device");
    if (!g_dev.hasUnifiedMemory) MDIE("needs a unified-memory device (zero-copy memory pools)");
    g_q = [g_dev newCommandQueue];
    // A device without Metal 4 (the paravirtualized GPU of a virtual Mac, such as GitHub's macos-26 runner: no residency set, no MTL4CommandQueue) cannot run the NVN layer. The GL and
    // Vulkan layers draw on classic queues of their own and need only the device (mtl_device_classic): there the classic part starts and presents are dropped, which is all a
    // run-time shader capture needs (the NSP script's capture_shaders boots headless; GitHub issue #112: Balatro's capture aborted here)
    NSError *re = nil; const char *no4 = NULL;
    if (getenv("DM_TEST_NO_METAL4")) no4 = "DM_TEST_NO_METAL4";   // test switch: as on a virtual Mac; not set in the product
    else if (![g_dev supportsFamily:MTLGPUFamilyMetal4]) no4 = "no Metal 4 GPU family";
    else {
        MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
        rd.initialCapacity = 4096;
        g_res = [g_dev newResidencySetWithDescriptor:rd error:&re];
        if (!g_res) no4 = "failed to create the MTLResidencySet";
    }
    if (!no4) {
        dispatch_queue_attr_t feedback_attr = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0);
        g_feedback_q = dispatch_queue_create("dm.nvn.mtl4-feedback", feedback_attr);
        MTL4CommandQueueDescriptor *q4d = [MTL4CommandQueueDescriptor new]; q4d.feedbackQueue = g_feedback_q;
        g_q4 = [g_dev newMTL4CommandQueueWithDescriptor:q4d error:&re];
        if (!g_q4) no4 = "failed to create the MTL4CommandQueue";
    }
    if (no4) {
        const char *why = re ? re.localizedDescription.UTF8String : "";
        if (!g_classic_ok) MDIE("the Metal device %s has no Metal 4 (%s%s%s), which the NVN layer needs; a virtual Mac's paravirtualized GPU has none", g_dev.name.UTF8String, no4, *why ? ": " : "", why);
        g_res = nil; g_q4 = nil; g_classic = 1;
        g_ev = [g_dev newSharedEvent];
        g_keep = [NSMutableArray array];
        MLOG("[RED] Metal device %s has no Metal 4 (%s%s%s): the classic part only, nothing is presented (enough for a run-time shader capture)", g_dev.name.UTF8String, no4, *why ? ": " : "", why);
        return;
    }
    // BC textures are created and uploaded as BC, with no decompression path (dm/gfx/tex/README.md); without BC the first one aborts in Metal's descriptor validation mid-game
    // (GitHub issue #704: an iPhone 15, A16). The iOS front end refuses such a device before the launcher (main.m); this stops every other route at the device, with the reason
    if (!g_dev.supportsBCTextureCompression && !bc_cpu_fallback()) MDIE("the Metal device %s has no BC texture formats, which the games' textures use and the host does not decompress: an iPhone needs an A17 Pro or later, an iPad an M1 or later", g_dev.name.UTF8String);
    if (bc_cpu_fallback()) MLOG("[RED] experimental BC-to-RGBA bridge enabled: no sparse/BC6H support, high GPU memory use, gameplay NOT validated");
    [g_q addResidencySet:g_res];
    [g_q4 addResidencySet:g_res];
    { void crash_set_wait_dump(void (*fn)(void)); void mtl_wait_state_dump(void); crash_set_wait_dump(mtl_wait_state_dump); }
    g_alloc_n = 3;
    for (int i = 0; i < 3; i++) {
        MTL4CommandAllocatorDescriptor *ad = [MTL4CommandAllocatorDescriptor new];
        ad.label = [NSString stringWithFormat:@"nvn-mtl allocator %d", i];
        NSError *ae = nil;
        g_alloc4[i] = [g_dev newCommandAllocatorWithDescriptor:ad error:&ae];
        if (!g_alloc4[i]) MDIE("failed to create MTL4CommandAllocator[%d]: %s", i, ae.localizedDescription.UTF8String);
        g_up_alloc4[i] = [g_dev newCommandAllocatorWithDescriptor:ad error:&ae];
        if (!g_up_alloc4[i]) MDIE("failed to create upload MTL4CommandAllocator[%d]: %s", i, ae.localizedDescription.UTF8String);
        MTL4CommandAllocatorDescriptor *pad = [MTL4CommandAllocatorDescriptor new];
        pad.label = [NSString stringWithFormat:@"nvn-mtl present allocator %d", i];
        g_present_alloc4[i] = [g_dev newCommandAllocatorWithDescriptor:pad error:&ae];
        if (!g_present_alloc4[i]) MDIE("failed to create present MTL4CommandAllocator[%d]: %s", i, ae.localizedDescription.UTF8String);
        g_present_cb[i] = [g_dev newCommandBuffer];
        if (!g_present_cb[i]) MDIE("failed to create present MTL4CommandBuffer[%d]", i);
    }
    MTL4ArgumentTableDescriptor *atd = [MTL4ArgumentTableDescriptor new];
    atd.maxBufferBindCount = 31; atd.maxTextureBindCount = 1;
    NSError *ate = nil;
    g_vt = rec_table([g_dev newArgumentTableWithDescriptor:atd error:&ate]); if (!g_vt) MDIE("failed to create the vertex argument table: %s", ate.localizedDescription.UTF8String);
    g_vt_valid = 0;
    g_ft = rec_table([g_dev newArgumentTableWithDescriptor:atd error:&ate]); if (!g_ft) MDIE("failed to create the fragment argument table: %s", ate.localizedDescription.UTF8String);
    g_ct = rec_table([g_dev newArgumentTableWithDescriptor:atd error:&ate]); if (!g_ct) MDIE("failed to create the compute argument table: %s", ate.localizedDescription.UTF8String);
    g_ot = rec_table([g_dev newArgumentTableWithDescriptor:atd error:&ate]); if (!g_ot) MDIE("failed to create the object argument table: %s", ate.localizedDescription.UTF8String);
    g_mt = rec_table([g_dev newArgumentTableWithDescriptor:atd error:&ate]); if (!g_mt) MDIE("failed to create the mesh argument table: %s", ate.localizedDescription.UTF8String);
    g_up_ct = [g_dev newArgumentTableWithDescriptor:atd error:&ate]; if (!g_up_ct) MDIE("failed to create the upload compute argument table: %s", ate.localizedDescription.UTF8String);
    MTL4ArgumentTableDescriptor *pad = [MTL4ArgumentTableDescriptor new];
    pad.maxTextureBindCount = 1; pad.maxSamplerStateBindCount = 1; pad.maxBufferBindCount = 1;
    g_present_args = [g_dev newArgumentTableWithDescriptor:pad error:&ate];
    if (!g_present_args) MDIE("failed to create the present argument table: %s", ate.localizedDescription.UTF8String);
    for (int i = 0; i < 3; i++) {
        g_present_uv[i] = [g_dev newBufferWithLength:sizeof(float) * 8 options:MTLResourceStorageModeShared];
        if (!g_present_uv[i]) MDIE("failed to create present UV buffer[%d]", i);
        res_add(g_present_uv[i]);
    }
    g_ev = [g_dev newSharedEvent];
    g_keep = [NSMutableArray array];
    g_nvn_deswz_gpu = nvn_deswz_gpu_init(g_dev) == 0;
    dm_res_report_register(mtl_res_counts);
    if (!g_nvn_deswz_gpu) MLOG("[RED] GPU deswizzle init failed; swizzled textures fall back to the CPU");
    MLOG("Metal device %s", g_dev.name.UTF8String);
    udc_selftest();
}

void dm_mtl_l3_tick(uint32_t tick) {
    static int leak = -1;
    if (leak < 0) leak = getenv("DM_TEST_LEAK_GPU") != NULL;
    if (leak && tick % 100 == 0) {
        if (!g_l3_gpu_leaks) g_l3_gpu_leaks = [NSMutableArray array];
        id<MTLBuffer> leak = [g_dev newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];
        if (leak) [g_l3_gpu_leaks addObject:leak];
    }
}

uint64_t dm_mtl_l3_sample(uint32_t tick) {
    (void)tick;
    return g_dev ? g_dev.currentAllocatedSize : 0;
}

// No-copy buffers of finalized pools (weak references, under g_mu): the memory report counts how many are still alive and their length -- a buffer that outlives its pool is still strongly referenced by something else (a texture built directly on the pool, etc.)
static NSHashTable *g_dead_pools;
// Address â†’ pool cache per thread (mtl_addr / mtl_gpu_addr), valid while g_pool_gen is unchanged (every pool add / removal bumps it under g_mu), holding only pools already in the residency set.
// Every buffer binding of every draw used to take g_mu, which game threads creating pools and the GPU completion handler also take, and search the pool index. Two levels: a table direct-mapped
// by 4 KB page, then the last 8 pools found, which also fill the table. Breath of the Wild binds from more pools a frame than 8 entries alone could hold (about 2300 of its 21000 lookups a frame
// took the lock), and its uniform rings move to a new page every few draws (the table alone missed as often)
enum { PC_BITS = 10, PC_RECENT = 8 };
static _Atomic uint64_t g_pool_gen = 1;
typedef struct { uintptr_t lo, hi, base; __unsafe_unretained id<MTLBuffer> buf; uint64_t gpu; } pc_ent;
static __thread struct { uint64_t gen; unsigned next; pc_ent r[PC_RECENT], e[1u << PC_BITS]; } t_pc;   // an empty entry has hi 0
static inline unsigned pc_slot(uint64_t addr) { return (unsigned)(((addr >> 12) * 0x9E3779B97F4A7C15ull) >> (64 - PC_BITS)); }
static inline const pc_ent *pc_get(uint64_t addr, uint64_t len) {
    if (t_pc.gen != atomic_load_explicit(&g_pool_gen, memory_order_acquire)) return NULL;
    pc_ent *c = &t_pc.e[pc_slot(addr)];
    if (addr >= c->lo && addr + len <= c->hi) return c;
    for (unsigned k = 0; k < PC_RECENT; k++) { const pc_ent *x = &t_pc.r[k]; if (addr >= x->lo && addr + len <= x->hi) { *c = *x; return c; } }
    return NULL;
}
static void pc_put(int i, uint64_t addr) {   // caller holds g_mu, pool i is resident and holds addr
    uint64_t gen = atomic_load_explicit(&g_pool_gen, memory_order_relaxed);
    if (t_pc.gen != gen) { memset(t_pc.r, 0, sizeof t_pc.r); memset(t_pc.e, 0, sizeof t_pc.e); t_pc.gen = gen; }
    pc_ent c = {g_pools[i].lo, g_pools[i].hi, g_pools[i].base, g_pools[i].buf, g_pools[i].gpu};
    t_pc.e[pc_slot(addr)] = c; t_pc.r[t_pc.next++ % PC_RECENT] = c;
}
static _Atomic uint64_t g_reg_gen = 1;   // Change generation of memory pool / texture / sampler registration: one of the invalidation conditions for argument buffer reuse
// On-demand residency of memory pools: putting a zero-copy buffer into the residency set makes iOS read the whole pool into memory and pin it as wired, even a pool the game never uses
// (they measured on a 17 Pro Max a 512 MB render-target pool addressed 0 times in a run, with about 530 MB more device wired memory; a 4 GB A14 was ended by the system within seconds of launch because of it).
// So a pool is not put into the residency set when created; it is put in, and the residency set committed on the spot, the first time the GPU gets its address through mtl_addr / mtl_gpu_addr -- these are the only two ways the GPU gets pool addresses,
// and committing on the spot is earlier than any command buffer commit that uses it. CPU-side reads of pool contents through mtl_addr trigger it too (conservative, never misses). Caller holds g_mu
// A pool's zero-copy buffer is made the first time the GPU (or a host path reading through the replay) needs its address, not when the pool is created: a zero-copy buffer holds
// the pages it covers, so the memory of a pool the GPU never addresses as a buffer (texture storage, which the host uploads into textures of its own) could not be given back while
// the buffer lived (a buffer over released pages kept them wired, and the GPU still read the old bytes). Mario Kart World creates about 5,200 pools, of which 2,800 are never
// addressed. Caller holds g_mu
static uint64_t g_pool_wrapped;
static void pool_buf(int i) {
    pool_ent *p = &g_pools[i];
    if (p->buf) return;
    void *g = (void *)p->base; size_t n = ((p->hi + getpagesize() - 1) & ~((uintptr_t)getpagesize() - 1)) - p->base;
    // Over a mapping of its own of the same pages, writable. The dirty tracker has most of a pool's pages read-only by the time it is first addressed, and a zero-copy buffer made over
    // read-only pages takes GPU writes without an error and drops them (t_cpu_tex: a texture â†’ buffer copy left zeros). The game's mapping stays as the tracker wants it
    void *a = g; vm_address_t al = 0; vm_prot_t cur, max;
    void (^free_alias)(void *, NSUInteger) = nil;
    if (vm_remap(mach_task_self(), &al, n, 0, VM_FLAGS_ANYWHERE, mach_task_self(), (vm_address_t)g, FALSE, &cur, &max, VM_INHERIT_NONE) == KERN_SUCCESS) {
        if (!mprotect((void *)al, n, PROT_READ | PROT_WRITE)) { a = (void *)al; free_alias = ^(void *m, NSUInteger l) { vm_deallocate(mach_task_self(), (vm_address_t)m, l); }; }
        else vm_deallocate(mach_task_self(), al, n);
    }
    { static int warned; if (!free_alias && !warned++) MLOG("[RED] memory pool %p: no writable mapping of its own for its buffer: GPU writes to it may be lost (first case)", p->key); }
    id<MTLBuffer> buf = [g_dev newBufferWithBytesNoCopy:a length:n options:MTLResourceStorageModeShared deallocator:free_alias];
    // PokÃ©mon Legends: Z-A: twice on the Mac a small pool's buffer was refused although its pages were one read-write VM region, both times while another heavy job (a shader compile,
    // a second game instance) took the machine's memory; every run on an idle machine created all of them. Wiring the pages may fail under memory pressure: retry for up to about 2 s
    for (int t = 0; !buf && t < 12; t++) {
        usleep(10000u << (t < 4 ? t : 4));
        buf = [g_dev newBufferWithBytesNoCopy:a length:n options:MTLResourceStorageModeShared deallocator:free_alias];
        if (buf) MLOG("[RED] memory pool %p+%#lx: the zero-copy buffer was refused, created at retry %d", p->key, (unsigned long)(p->hi - p->lo), t + 1);
    }
    if (!buf) {   // say what the pages looked like: every VM region over the range, with its protection (an unmapped page in the range is the usual cause)
        for (vm_address_t r = (vm_address_t)a; r < (vm_address_t)a + n; ) {
            vm_size_t rs = 0; vm_region_basic_info_data_64_t vi; mach_msg_type_number_t cnt = VM_REGION_BASIC_INFO_COUNT_64; mach_port_t obj = MACH_PORT_NULL;
            if (vm_region_64(mach_task_self(), &r, &rs, VM_REGION_BASIC_INFO_64, (vm_region_info_t)&vi, &cnt, &obj) != KERN_SUCCESS) { MLOG("[RED]   no VM region at or after %#llx", (unsigned long long)r); break; }
            MLOG("[RED]   VM region %#llx+%#llx protection %d (max %d)", (unsigned long long)r, (unsigned long long)rs, vi.protection, vi.max_protection);
            r += rs;
        }
        MDIE("memory pool %p+%#lx: failed to create the zero-copy buffer over %p+%#zx", p->key, (unsigned long)(p->hi - p->lo), a, n);
    }
    p->buf = buf; p->gpu = buf.gpuAddress; [g_keep addObject:buf];
    // Diagnostic DM_NVN_POOLLOG: every pool wrapped, with the bytes wrapped so far and the device's allocated size (which pools make up the GPU allocation)
    static int plog = -1; if (plog < 0) plog = getenv("DM_NVN_POOLLOG") != NULL;
    g_pool_wrapped += n;
    if (plog) MLOG("pool wrapped %p+%#lx (%lu MB): %llu MB wrapped, GPU allocated %llu MB", p->key, (unsigned long)(p->hi - p->lo), (unsigned long)((p->hi - p->lo) >> 20),
                   (unsigned long long)(g_pool_wrapped >> 20), (unsigned long long)(g_dev.currentAllocatedSize >> 20));
}
static void pool_res(int i) {
    if (g_pools[i].res) return;
    uint64_t t0 = clock_gettime_nsec_np(CLOCK_MONOTONIC);
    pool_buf(i);
    res_add(g_pools[i].buf); res_commit(); g_pools[i].res = 1; g_pool_res_n++;
    double ms = (clock_gettime_nsec_np(CLOCK_MONOTONIC) - t0) / 1e6;   // on iOS this reads the whole pool into memory and wires it (see above): the time a frame stops for it
    static double total; total += ms;
    if (ms >= 20) MLOG("memory pool %p (%llu MB): %.0f ms to make its buffer resident (%.0f ms for all pools so far)", g_pools[i].key, (unsigned long long)((g_pools[i].hi - g_pools[i].lo) >> 20), ms, total);
}
// hle_svc.c zero_sweep_thread: fn(lo, hi) runs while no zero-copy buffer covers [lo, hi) and none can be made (pool_buf holds g_mu), or not at all (â†’ 0): a buffer holds the pages
// under it, so a new mapping there would leave the GPU on the old pages
size_t mtl_pool_unwrapped_do(uintptr_t lo, uintptr_t hi, size_t (*fn)(uintptr_t, uintptr_t)) {
    size_t r = 0; int wrapped = 0;
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_npool && !wrapped; i++) wrapped = g_pools[i].buf && g_pools[i].lo < hi && lo < g_pools[i].hi;
    if (!wrapped) r = fn(lo, hi);
    pthread_mutex_unlock(&g_mu);
    return r;
}
int mtl_pool_wrapped(const void *addr, uint64_t len) {   // whether a zero-copy buffer covers any of [addr, addr+len): pages under one cannot be given back (mtl_release_pages)
    int r = 0; uintptr_t lo = (uintptr_t)addr, hi = lo + len;
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_npool && !r; i++) r = g_pools[i].buf && g_pools[i].lo < hi && lo < g_pools[i].hi;
    pthread_mutex_unlock(&g_mu);
    return r;
}
void mtl_pool_add(const void *mem, uint64_t size) {
    mtl_init(); g_reg_gen++;
    uintptr_t pg = (uintptr_t)getpagesize(), lo = (uintptr_t)mem, hi = lo + size;
    uintptr_t a = lo & ~(pg - 1);
    pthread_mutex_lock(&g_mu);
    if (g_npool == g_cap) { g_cap = g_cap ? g_cap * 2 : 64; g_pools = realloc(g_pools, g_cap * sizeof *g_pools); }
    g_pidx_ok = 0; g_plast = -1; atomic_fetch_add(&g_pool_gen, 1);   // the address index and the per-thread caches are invalid (pool_find, pc_get)
    g_pools[g_npool++] = (pool_ent){lo, hi, a, nil, 0, mem, 0, clock_gettime_nsec_np(CLOCK_MONOTONIC)};   // no buffer and not in the residency set yet: both the first time it is addressed (pool_res)
    g_pool_adds++;
    pthread_mutex_unlock(&g_mu);
}

void mtl_pool_del(const void *mem) {
    g_reg_gen++;
    pthread_mutex_lock(&g_mu);
    // The no-copy buffer goes through deferred release (out of the residency set and unreferenced after submitted commands complete). It used to be removed only from g_keep while the residency set still held a strong reference: after the pool was freed the buffer stayed alive wrapping that guest memory,
    // and a new pool at the same address wrapped it again. Engage's title demo cycles scenes; GPU allocated grew to 5.4 GB and resident memory to 3.6 GB in 850 s.
    // Its own autorelease pool: this runs on the guest thread that finalizes the pool, whose implicit pool never drains, and adding to the weak table g_dead_pools left the buffer
    // autoreleased there, so every finalized pool buffer stayed alive for good (Ghost Trick: Phantom Detective, 125 buffers and 6.6 GB of GPU allocations 300 frames into the boot, 0 with the pool)
    @autoreleasepool {
    for (int i = 0; i < g_npool; i++)
        if (g_pools[i].key == mem) { id<MTLBuffer> b = g_pools[i].buf; if (g_pools[i].res) g_pool_res_n--;
            if (b) { [g_keep removeObject:b]; defer_obj_release(b); if (!g_dead_pools) g_dead_pools = [NSHashTable weakObjectsHashTable]; [g_dead_pools addObject:b]; }
            g_pools[i] = g_pools[--g_npool]; g_pidx_ok = 0; g_plast = -1; atomic_fetch_add(&g_pool_gen, 1); break; }
    }
    pthread_mutex_unlock(&g_mu);
}

// Address â†’ pool: the last hit first, then a binary search over indices sorted by start address, and a full linear scan only when that fails (pools overlapping).
// The Legend of Zelda: Breath of the Wild keeps about 1760 memory pools alive and every buffer binding of every draw looks one up; the old full scan took about 4% of the submit thread's samples (mtl_gpu_addr 222 / 6 s). Adding / removing pools invalidates the index, which is re-sorted when next used. Caller holds g_mu
static int *g_pidx; static int g_pidx_cap;
static int pidx_cmp(const void *a, const void *b) { uintptr_t x = g_pools[*(const int *)a].lo, y = g_pools[*(const int *)b].lo; return x < y ? -1 : x > y; }
static void pidx_sort(void) {
    if (g_pidx_ok) return;
    if (g_pidx_cap < g_npool) { g_pidx_cap = g_npool * 2; g_pidx = realloc(g_pidx, (size_t)g_pidx_cap * sizeof *g_pidx); }
    for (int i = 0; i < g_npool; i++) g_pidx[i] = i;
    qsort(g_pidx, (size_t)g_npool, sizeof *g_pidx, pidx_cmp); g_pidx_ok = 1;
}
static int pool_find(uint64_t addr, uint64_t len) {
    if (g_plast >= 0 && g_plast < g_npool && addr >= g_pools[g_plast].lo && addr + len <= g_pools[g_plast].hi) return g_plast;
    pidx_sort();
    int lo = 0, hi = g_npool - 1, hit = -1;   // the last one whose start â‰¤ addr
    while (lo <= hi) { int m = (lo + hi) / 2; if (g_pools[g_pidx[m]].lo <= addr) { hit = m; lo = m + 1; } else hi = m - 1; }
    if (hit >= 0 && addr + len <= g_pools[g_pidx[hit]].hi) return g_plast = g_pidx[hit];
    for (int i = 0; i < g_npool; i++) if (addr >= g_pools[i].lo && addr + len <= g_pools[i].hi) return g_plast = i;
    return -1;
}
int mtl_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off) {
    const pc_ent *c = pc_get(addr, len);
    if (c) { *buf = (__bridge void *)c->buf; *off = addr - c->base; return 1; }
    int ok = 0;
    pthread_mutex_lock(&g_mu);
    int i = pool_find(addr, len);
    if (i >= 0) { pool_res(i); pc_put(i, addr); *buf = (__bridge void *)g_pools[i].buf; *off = addr - g_pools[i].base; ok = 1; }
    pthread_mutex_unlock(&g_mu);
    return ok;
}
int mtl_gpu_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off, uint64_t *gpu) {
    const pc_ent *c = pc_get(addr, len);
    if (c) { *buf = (__bridge void *)c->buf; *off = addr - c->base; *gpu = c->gpu + *off; return 1; }
    int ok = 0;
    pthread_mutex_lock(&g_mu);
    int i = pool_find(addr, len);
    if (i >= 0) { pool_res(i); pc_put(i, addr); *buf = (__bridge void *)g_pools[i].buf; *off = addr - g_pools[i].base; *gpu = g_pools[i].gpu + *off; ok = 1; }
    pthread_mutex_unlock(&g_mu);
    return ok;
}

// ---------- Submission ----------
static void defer_reset(void); static void defer_fallback(void);
typedef struct { int kind; volatile uint32_t *val; uint32_t mode, v; uint64_t addr; uint32_t type; void *vis; uint32_t seg; } post_op;   // vis: the visibility slot table for kind 3, the readback for kind 4; seg: the segment it was encoded in (sub_cut)
static void mirror_rb_land(void *m);   // kind 4: a GPU copy's bytes read back into the destination texture's storage (mirror_readback)
static uint32_t g_seg;   // Index of the segment being encoded in this submission (0 until the first sub_cut)
typedef struct seg_clk seg_clk; static seg_clk *g_sclk; static void seg_clk_put(seg_clk *c);   // Counter timestamps by segment (see gpu_ticks)
// ---------- Conditional rendering (profile render_cond) ----------
// SetRenderEnableConditional(mode, rec) enables rendering if the 64-bit counter of the report at rec equals (RENDER_IF_EQUAL) or differs from (RENDER_IF_NOT_EQUAL) the one at
// rec + 16, until the next SetRenderEnable(true). Tears of the Kingdom wraps nearly every ClearColor / ClearDepthStencil this way: rec is set to 1 from the CPU when the target
// needs a clear, each clear is followed by ReportCounter(TIMESTAMP_TOP, rec), and rec + 16 is never written. A timestamp report puts 0 in the counter field (the time goes in the
// second field), so after one clear the two compare equal and later conditional clears of that target are skipped. The host used to ignore the condition and write the time
// into both fields, so every clear ran: the first tile-lighting round, the half-resolution scene depth its light volumes test against and the light accumulation were cleared
// before the composite read them, and the 3D scene came out black.
// Metal has no predication, so the condition is evaluated here at replay, on the CPU, from the records in memory. A report replayed earlier in this submission is not written yet
// (that happens when the GPU completes): a timestamp one counts with its known counter 0; a samples-passed result (an occlusion query, written from the visibility buffers) is
// unknown, and a condition on it keeps drawing, as before (occlusion culling only saves work). Inside a block evaluated false, draws and clears are dropped
static int g_rc_skip;   // a render-enable block evaluated false is active
// Neither table may drop an entry: a timestamp report forgotten leaves the condition reading the 1 the CPU wrote, so the clear runs again. They held 256 reports a submission of both kinds, and Splatoon Raiders' 285 ink
// queries a frame came first: every conditional clear after them ran, and the lit scene was wiped before the composite read it (a black screen with only the dialogue on it). Now only timestamps are kept per submission, and both tables grow
static uint64_t *g_rc_pend; static uint32_t g_rc_npend, g_rc_pcap;   // timestamp reports replayed in this submission: their counter is 0, though memory does not show it yet
static uint64_t *g_rc_occ; static uint32_t g_rc_ocap, g_rc_on;   // addresses ever written by a non-timestamp report (open addressing, a power of two): conditions on them keep drawing
static int rc_occ_has(uint64_t a) { if (!g_rc_ocap) return 0; for (uint32_t i = (uint32_t)(a >> 4) & (g_rc_ocap - 1);; i = (i + 1) & (g_rc_ocap - 1)) { if (g_rc_occ[i] == a) return 1; if (!g_rc_occ[i]) return 0; } }
static void rc_occ_add(uint64_t a) {
    if (rc_occ_has(a)) return;
    if ((g_rc_on + 1) * 2 > g_rc_ocap) { uint32_t oc = g_rc_ocap, nc = oc ? oc * 2 : 4096; uint64_t *old = g_rc_occ, *t = calloc(nc, 8); if (!t) MDIE("no memory for the query report table");
        for (uint32_t k = 0; k < oc; k++) if (old[k]) { uint32_t i = (uint32_t)(old[k] >> 4) & (nc - 1); while (t[i]) i = (i + 1) & (nc - 1); t[i] = old[k]; }
        g_rc_occ = t; g_rc_ocap = nc; free(old); }
    uint32_t i = (uint32_t)(a >> 4) & (g_rc_ocap - 1); while (g_rc_occ[i]) i = (i + 1) & (g_rc_ocap - 1);
    g_rc_occ[i] = a; g_rc_on++;
}
static void rc_note(uint64_t a, int kind) {
    if (kind == 2) { rc_occ_add(a); return; }   // rc_value asks this table first
    for (uint32_t i = 0; i < g_rc_npend; i++) if (g_rc_pend[i] == a) return;
    if (g_rc_npend == g_rc_pcap) { uint32_t nc = g_rc_pcap ? g_rc_pcap * 2 : 256; uint64_t *t = realloc(g_rc_pend, (size_t)nc * 8); if (!t) MDIE("no memory for the timestamp report table"); g_rc_pend = t; g_rc_pcap = nc; }
    g_rc_pend[g_rc_npend++] = a;
}
static int rc_value(uint64_t a, uint64_t *v) {   // 0 = unknown at encode time
    if (rc_occ_has(a)) return 0;
    for (uint32_t i = g_rc_npend; i-- > 0;) if (g_rc_pend[i] == a) { *v = 0; return 1; }
    *v = *(const volatile uint64_t *)a; return 1;
}
// ---------- Occlusion queries (NVN_COUNTER_TYPE_SAMPLES_PASSED) ----------
// NVN semantics: samples passed is a cumulative counter on the queue - ResetCounter zeroes it, every draw afterwards adds to it, and ReportCounter writes out the cumulative value at that moment (+ timestamp) when the GPU completes.
// Metal: every render pass attaches a visibility result buffer; draws during counting use counting mode, with a new slot per pass (no reliance on accumulation across passes); a reset empties the 'slot table since the reset',
// and a report hands the slot table together with its buffer (retained) to the completion callback to sum and write back; after a report a new slot is used, so later draws are not counted into the reported value.
// (Previously always wrote 0: occlusion query = 'fully occluded', campfires / light sources got culled - the character select screen calls ResetCounter 21 times per frame)
typedef struct { void *buf; uint32_t slot; } vis_slot;
typedef struct { int n; vis_slot s[]; } vis_list;
#define VIS_SLOTS 8192
typedef struct { __strong id<MTLBuffer> buf; uint64_t fence; } vis_buf;
// One visibility buffer per submission in flight, as many as command allocators: with 4, Age of Calamity (about twelve submissions a frame) waited in mtl_sub_begin for the previous frame's GPU work; its first battle went from 18.0 to 16.9 ms a frame on the Mac
enum { VIS_RING = ALLOC_MAX };
static vis_buf g_vis_ring[VIS_RING]; static int g_vis_ring_n, g_vis_ring_i = -1;
static id<MTLBuffer> g_vis; static uint32_t g_vis_n;   // This submission's visibility buffer and number of slots used
static int vis_in_ring(id<MTLBuffer> b) { for (int i = 0; i < g_vis_ring_n; i++) if (g_vis_ring[i].buf == b) return 1; return 0; }
static int g_vis_on, g_vis_counting, g_vis_cur = -1; static uint64_t g_vis_cur_pass;
static vis_slot *g_vs; static int g_nvs, g_vscap;      // Slots since the last reset
static uint64_t g_cnt_report[32], g_cnt_reset[32];       // Report / reset counts per counter type (diagnostics)
static void vis_clear(void) { for (int i = 0; i < g_nvs; i++) CFRelease(g_vs[i].buf); g_nvs = 0; g_vis_cur = -1; }
static vis_list *vis_snapshot(void) {
    vis_list *l = malloc(sizeof *l + g_nvs * sizeof(vis_slot)); l->n = g_nvs;
    for (int i = 0; i < g_nvs; i++) { l->s[i] = g_vs[i]; CFRetain(g_vs[i].buf); }
    return l;
}
// Target ranges of buffer writes (copies / inline uniform updates) already encoded in this submission but not yet executed on the GPU: reading these addresses on the CPU side at encode time yields old values (uv_proof avoids them when reading uniforms)
// Inline updates keep where their data sits in the update ring (g_pend_upd: index + 1 into g_upd), so a bindless handle they wrote can be read as the GPU will see it (pend_read).
// Monster Hunter Rise queues more than 256 writes in a submission (the old limit, after which every address counted as written). The writes are indexed by 4 KB page (a chain per hash
// bucket, newest first); a write over more than PEND_SPAN pages goes to a short list that every lookup scans
enum { PEND_MAX = 8192, PEND_HB = 2048, PEND_NODES = 16384, PEND_SPAN = 8, PEND_WIDE = 256 };
static uint64_t g_pend_lo[PEND_MAX], g_pend_hi[PEND_MAX]; static int g_npend, g_pend_over; static uint8_t g_pend_kind[PEND_MAX]; static uint16_t g_pend_upd[PEND_MAX];   // kind: 1 inline update, 2 copy, 3 fill
static int32_t g_pend_head[PEND_HB]; static uint32_t g_pend_hgen[PEND_HB], g_pend_gen = 1;   // a bucket's chain is valid when its generation is this submission's
static struct { int32_t e, next; } g_pend_node[PEND_NODES]; static int g_npnode;
static int32_t g_pend_wide[PEND_WIDE]; static int g_nwide;
typedef struct { uint64_t lo; size_t off; __unsafe_unretained id<MTLBuffer> buf; } upd_ent;   // the ring block lives until this submission completes (mtl_sub_update)
static upd_ent g_upd[PEND_MAX]; static int g_nupd;
// Other writes to guest memory in this submission, which the pend list does not hold: writable storage buffers bound to a shader, textureâ†’buffer copies, counter reports (ubp_get).
// Kept by 4 KB page (Breath of the Wild's particle dispatches bind about ten storage buffers each, hundreds a frame): a set of pages tagged with the submission (g_pend_gen) and with what
// wrote them. A counter report does not count against this submission's shadows: the host writes reports on the CPU when the whole submission has completed, so no draw of it reads one
// (Fire Emblem Engage writes its query reports into the page of a uniform block); it counts for later submissions while this one runs (gw_commit).
// A GPU write over more than SBW_WPAGES pages is kept as an interval instead (overlapping ones merge): a storage buffer bound with size 0 counts as the rest of its Metal buffer (sbw_len)
enum { SBW_N = 16384, SBW_WPAGES = 64, SBW_WN = 64 };
static struct { uint64_t page; uint32_t gen; uint8_t src; } g_sbwp[SBW_N]; static int g_nsbw, g_sbw_over;   // src: SBW_* bits; pages in the set this submission; over: a write could not be kept (set full, size not known)
static uint64_t g_sbw_wlo[SBW_WN], g_sbw_whi[SBW_WN]; static int g_nsbw_w;   // the wide writes of this submission
static int sbw_slot(uint64_t pg, int add) {   // open addressing; âˆ’1 = not there (or no room to add)
    unsigned h = (unsigned)((pg * 0x9E3779B97F4A7C15ull) >> 50) & (SBW_N - 1);
    for (int i = 0; i < 64; i++, h = (h + 1) & (SBW_N - 1)) {
        if (g_sbwp[h].gen != g_pend_gen) { if (!add) return -1; g_sbwp[h].page = pg; g_sbwp[h].gen = g_pend_gen; g_sbwp[h].src = 0; g_nsbw++; return (int)h; }
        if (g_sbwp[h].page == pg) return (int)h; }
    return -1;
}
enum { SBW_GPU = 1, SBW_REPORT = 2 };
static uint32_t g_sbw_gen;   // bumped by every write noted that this submission's shadows must respect (a cached shadow made before it is checked again)
static void drw_note(uint64_t lo, uint64_t n);   // the writes since the last drain (below g_wr_epoch)
static void sbw_note(uint64_t lo, uint64_t n, int src) {
    if (src == SBW_GPU) { g_sbw_gen++; drw_note(lo, n); }
    if (!n) { if (src == SBW_GPU) g_sbw_over = 1; return; }
    uint64_t hi = lo + n;
    if (src == SBW_GPU && ((hi - 1) >> 12) - (lo >> 12) >= SBW_WPAGES) {   // a report is 16 bytes, never wide
        for (int i = 0; i < g_nsbw_w; i++) if (lo <= g_sbw_whi[i] && g_sbw_wlo[i] <= hi) { g_sbw_wlo[i] = MIN(lo, g_sbw_wlo[i]); g_sbw_whi[i] = MAX(hi, g_sbw_whi[i]); return; }
        if (g_nsbw_w == SBW_WN) { g_sbw_over = 1; return; }
        g_sbw_wlo[g_nsbw_w] = lo; g_sbw_whi[g_nsbw_w++] = hi; return;
    }
    for (uint64_t pg = lo >> 12; pg <= (hi - 1) >> 12; pg++) { int k = sbw_slot(pg, 1); if (k < 0) { g_sbw_over = 1; return; } g_sbwp[k].src |= (uint8_t)src; }
}
static int sbw_hits(uint64_t lo, uint64_t hi) {
    for (int i = 0; i < g_nsbw_w; i++) if (lo < g_sbw_whi[i] && g_sbw_wlo[i] < hi) return 1;
    if (!g_nsbw) return 0;
    for (uint64_t pg = lo >> 12; pg <= (hi - 1) >> 12; pg++) { int k = sbw_slot(pg, 0); if (k >= 0 && (g_sbwp[k].src & SBW_GPU)) return 1; }
    return 0;
}
// The bytes a writable storage binding at Metal buffer offset mo can write: its size, else (bound with size 0, as Breath of the Wild's GPU particle dispatches bind their outputs, or a
// size not known) the rest of that buffer. A device pointer that runs past its buffer is out of bounds on Metal, which the host could not give the Switch's meaning anyway. Before, such
// a binding counted as a write anywhere, and every partly covered uniform shadow after it in the submission was refused (BotW shrines: about 240 more passes a frame, GitHub issue #534)
static uint64_t sbw_len(uint64_t size, void *mb, uint64_t mo) { if (size) return size; uint64_t len = ((__bridge id<MTLBuffer>)mb).length; return len > mo ? len - mo : 0; }
static inline unsigned pend_bucket(uint64_t page) { return (unsigned)((page * 0x9E3779B97F4A7C15ull) >> 53) & (PEND_HB - 1); }
static void pend_reset(void) { g_npend = g_pend_over = g_nupd = g_npnode = g_nwide = g_nsbw = g_nsbw_w = g_sbw_over = 0; if (!++g_pend_gen) { memset(g_pend_hgen, 0, sizeof g_pend_hgen); memset(g_sbwp, 0, sizeof g_sbwp); g_pend_gen = 1; } }
static void vc_gpu_write(uint64_t dst, uint64_t n);   // kept index ranges under the written range are not trusted again (ix_range)
static int pend_add(uint64_t dst, uint64_t n, int kind) {
    vc_gpu_write(dst, n); drw_note(dst, n ? n : 1);
    if (g_pend_over || g_npend == PEND_MAX) { g_pend_over = 1; return -1; }
    int e = g_npend; uint64_t p0 = dst >> 12, p1 = (dst + (n ? n : 1) - 1) >> 12;
    if (p1 - p0 < PEND_SPAN && g_npnode + (int)(p1 - p0 + 1) <= PEND_NODES)
        for (uint64_t pg = p0; pg <= p1; pg++) { unsigned b = pend_bucket(pg); int k = g_npnode++;
            g_pend_node[k].e = e; g_pend_node[k].next = g_pend_hgen[b] == g_pend_gen ? g_pend_head[b] : -1; g_pend_head[b] = k; g_pend_hgen[b] = g_pend_gen; }
    else if (g_nwide < PEND_WIDE) g_pend_wide[g_nwide++] = e;
    else { g_pend_over = 1; return -1; }
    g_pend_lo[e] = dst; g_pend_hi[e] = dst + n; g_pend_kind[e] = (uint8_t)kind; g_pend_upd[e] = 0; return g_npend++;
}
// The newest queued write overlapping [a, a+n): its index, -1 for none, -2 when it cannot be told (overflow)
static int pend_last(uint64_t a, uint64_t n) {
    if (g_pend_over) return -2;
    int best = -1; uint64_t p0 = a >> 12, p1 = (a + (n ? n : 1) - 1) >> 12;
    for (int i = 0; i < g_nwide; i++) { int e = g_pend_wide[i]; if (e > best && a < g_pend_hi[e] && a + n > g_pend_lo[e]) best = e; }
    if (p1 - p0 >= PEND_SPAN) { for (int e = g_npend - 1; e > best; e--) if (a < g_pend_hi[e] && a + n > g_pend_lo[e]) return e; return best; }
    for (uint64_t pg = p0; pg <= p1; pg++) { unsigned b = pend_bucket(pg); if (g_pend_hgen[b] != g_pend_gen) continue;
        for (int k = g_pend_head[b]; k >= 0 && g_pend_node[k].e > best; k = g_pend_node[k].next) { int e = g_pend_node[k].e; if (a < g_pend_hi[e] && a + n > g_pend_lo[e]) { best = e; break; } } }
    return best;
}
static int pend_hits(uint64_t a, uint64_t n) { return pend_last(a, n) != -1; }
// The n bytes at a as the last write queued in this submission leaves them, when that write is an inline update covering them all â†’ 1; otherwise â†’ 0 and *kind = what wrote them (0 = too many writes to tell)
static int pend_read(uint64_t a, uint64_t n, void *out, int *kind) {
    int e = pend_last(a, n); *kind = e >= 0 ? g_pend_kind[e] : 0;
    if (e < 0 || !g_pend_upd[e] || a < g_pend_lo[e] || a + n > g_pend_hi[e]) return 0;
    const upd_ent *u = &g_upd[g_pend_upd[e] - 1]; memcpy(out, (const uint8_t *)u->buf.contents + u->off + (a - u->lo), n); return 1;
}
typedef struct { uint64_t ser; uint16_t level, slice; } zskip_zero;
static zskip_zero g_zskip_zero[4096]; static int g_zskip_n, g_zskip_count;
static zskip_zero g_zskip_cand[8]; static uint8_t g_zskip_cand_ok[8];
static void zskip_clear(void);
static int tbdr_on(void);
static id g_cb;
static id<MTL4ComputeCommandEncoder> g_blit;
// Buffer operations may use distinct Metal GPU addresses for one allocation
// (overlapping pools or sparse mappings). Their encoder barriers need alias cache visibility.
static int g_blit_alias;
static inline MTL4VisibilityOptions blit_vis(void) { return resource_vis() | (g_blit_alias ? MTL4VisibilityOptionResourceAlias : 0); }
static NSMutableArray<dispatch_block_t> *g_commit_blocks;
// The uniform shadows taken once a shader with stores through loaded pointers has run (g_gml_on), each with the fence of the submission that took it: every shadow made or checked
// again after a GPU write was noted (ubp_get). A write log that shows a pool the prediction left out checks the shadows of its own submission and of the later ones encoded before
// it completed, the ones that relied on its writes (gml_read): the miss matters only if one of them covers a page the stores wrote
enum { GSR_N = 16384 };
static struct { uint64_t fence, lo, hi; } g_gsr[GSR_N]; static uint64_t g_gsr_n; static int g_gml_on;
static pthread_mutex_t g_gsr_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_alloc_fence[ALLOC_MAX]; static uint64_t g_alloc_waits, g_alloc_grown;
int mtl_wait(uint64_t fence, uint64_t timeout_ns);
static int mtl_wait_at(uint64_t fence, uint64_t timeout_ns, int site);
static void up_flush(void);
// The upload batch belongs only to the submit thread (the thread that called mtl_sub_begin): the batch's encoder cannot be used across threads. Every entry point asserts current thread == submit thread,
// and on mismatch prints the stack and aborts on the spot - so that 'another thread touched the batch' is exposed deterministically on Mac too, not by timing (real iOS device stuck at the main menu: the present thread flushed the submit thread's batch via q_cb).
static pthread_t g_sub_thread; static _Atomic int g_sub_thread_set;
static int on_sub_thread(void) { return atomic_load(&g_sub_thread_set) && pthread_equal(pthread_self(), g_sub_thread); }
static void up_owner_check(const char *where) {
    if (!atomic_load(&g_sub_thread_set) || on_sub_thread()) return;
    char name[64] = ""; pthread_getname_np(pthread_self(), name, sizeof name);
    MLOG("[RED] upload batches belong to the submit thread only: %s called on thread '%s'", where, name);
    void *bt[32]; int n = backtrace(bt, 32); backtrace_symbols_fd(bt, n, 2);
    abort();
}
// Every command buffer created outside the replay / upload batch goes through this. On the submit thread: flush this batch's uploads first, so that commands that then read or write these textures are ordered after the uploads;
// other threads (the present thread etc.) creating a command buffer do not touch the batch: they do not read or write textures uploaded in this batch; commands that need these textures all come after this replay submission (the batch is flushed before it)
// Old-queue command buffers (readback / diagnostics / present; to be deleted once M4-4 and M4-5 are migrated away): uploads are already on the Metal 4 queue, and there is no ordering guarantee between the two queues -
// wait for the upload to complete after flushing it, and only then can the old queue read the upload result (mtl_read_texels waited first and flushed afterwards, the readback beat the upload, 4 failures on t_nvn upload content)
static void up_wait_zero(_Atomic int *v, int site);
static id<MTLCommandBuffer> q_cb(void) { if (on_sub_thread()) { up_flush(); rec_drain(); up_wait_zero(&g_up_pending, WS_OLDQ); } return [g_q commandBuffer]; }
static post_op *g_post; static int g_npost, g_postcap;
// One autorelease pool per submission: submissions run on a guest thread (long-lived, no run loop, never drains the implicit pool), so objects autoreleased in the Metal replay
// (command buffers, encoders, pass descriptors, etc.) are never released unless put in a pool. Measured on the character select screen of a real iOS device: the host heap grows by about 47 objects per frame, 0.22 MiB/s.
void *objc_autoreleasePoolPush(void); void objc_autoreleasePoolPop(void *);
static void *g_sub_pool;
static int tbdr_on(void);
static void treg_finish_submission(void);

static uint64_t g_sub_serial;   // Submission index (used by the parallel precompile's 'predict only once per submission')
// The first submission: the prepare step (warm_start), before anything of it is encoded. No GPU work: the submit entry (hle_nvn.c) calls it outside the graphics gate
void mtl_prepare(void) {
    if (!g_warm_started && prepare_on()) { mtl_init(); shmap_load(); }
}
static void pkg_store_sweep(void); static void soft_sweep(void);
static void pst_drop_pool(const void *pool);   // caller holds g_smu
void mtl_sub_begin(void) {
    pkg_store_sweep(); soft_sweep();
    mtl_init(); g_sub_serial++; g_seg = 0; seg_clk_put(g_sclk); g_sclk = NULL; g_rc_npend = 0; g_rc_skip = 0;
    defer_drain();
    defer_reset();
    mtl_prepare();   // done by then in the submit entry; the replay tool (nvn_replay) reaches it here
    warm_claim();
    if (g_cb) MDIE("submit re-entered");
    g_sub_thread = pthread_self(); atomic_store(&g_sub_thread_set, 1);
    atomic_store(&g_last_commit_error, 0);
    g_sub_pool = objc_autoreleasePoolPush();
    g_cb = rec_cb_new();
    // Do not wait for the GPU mid-frame (hot real device: with a fixed rotation of 3, when the GPU cannot keep up every frame blocks here, and the submit thread runs at low frequency after waking, instruction count âˆ’18% but CPU time +49%):
    // pick an allocator the GPU has finished with; if all are in use create a new one; only at the cap wait for the oldest one. How far ahead it runs is bounded by the wait where the game acquires the swapchain texture, same as the old interface.
    int alloc_i = -1; uint64_t done = g_done;
    for (int i = 0; i < g_alloc_n; i++) if (g_alloc_fence[i] <= done) { alloc_i = i; break; }
    if (alloc_i < 0 && g_alloc_n < ALLOC_MAX) { NSError *ae = nil; MTL4CommandAllocatorDescriptor *ad = [MTL4CommandAllocatorDescriptor new];
        g_alloc4[g_alloc_n] = [g_dev newCommandAllocatorWithDescriptor:ad error:&ae]; if (!g_alloc4[g_alloc_n]) MDIE("failed to create a new MTL4CommandAllocator: %s", ae.localizedDescription.UTF8String);
        alloc_i = g_alloc_n++; g_alloc_grown++; }
    if (alloc_i < 0) { alloc_i = 0; for (int i = 1; i < g_alloc_n; i++) if (g_alloc_fence[i] < g_alloc_fence[alloc_i]) alloc_i = i; g_alloc_waits++; mtl_wait_at(g_alloc_fence[alloc_i], UINT64_MAX, WS_ALLOC); }
    g_alloc_cur = alloc_i;
    [g_alloc4[alloc_i] reset];
    [g_cb beginCommandBufferWithAllocator:g_alloc4[alloc_i]];
    g_commit_blocks = [NSMutableArray array];
    // The overflow buffer created when the previous submission ran out of slots (see g_vis_n >= VIS_SLOTS): retire it. It used to be just cleared here and stayed in the residency set for good (from d2r-mobile 2eb575877)
    if (g_vis && !vis_in_ring(g_vis)) defer_obj_release(g_vis);
    g_npost = 0; g_vis = nil; g_vis_n = 0; g_vis_cur = -1; pend_reset();
    zskip_clear(); g_zskip_count = 0;
    int pick = -1;
    for (int i = 0; i < g_vis_ring_n; i++) if (g_vis_ring[i].fence <= g_done) { pick = i; break; }
    if (pick < 0 && g_vis_ring_n < VIS_RING) pick = g_vis_ring_n++;
    if (pick < 0) { uint64_t oldest = UINT64_MAX; for (int i = 0; i < VIS_RING; i++) if (g_vis_ring[i].fence < oldest) { oldest = g_vis_ring[i].fence; pick = i; } if (oldest > g_done) mtl_wait_at(oldest, UINT64_MAX, WS_VIS); }
    g_vis_ring_i = pick; g_vis_ring[pick].fence = UINT64_MAX;
    if (!g_vis_ring[pick].buf) { g_vis_ring[pick].buf = [g_dev newBufferWithLength:VIS_SLOTS * 8 options:MTLResourceStorageModeShared]; res_add(g_vis_ring[pick].buf); }
    g_vis = g_vis_ring[pick].buf;
    void pg_sub_begin_ext(uint64_t fence); pg_sub_begin_ext(g_fence + 1);
    void arena_begin(void); arena_begin();
}

static void rp_end(void);
// DM_NVN_PASSGPU_SERIAL=1 (for measurement): before creating each encoder on the main command buffer, encode 'signal n+1 â†’ wait for signal n+1' - the signal is only raised once all prior work has completed,
// so the following encoder starts only after the previous one has fully finished; on TBDR passes no longer overlap in the pipeline, and the per-pass start-to-end span of DM_NVN_PASSGPU is that pass's exclusive time (the whole-frame total is inflated; look only at relative cost)
static void ser_gate(void) {
    static int on = -1; static id<MTLEvent> ev; static uint64_t n;
    if (on < 0) { on = getenv("DM_NVN_PASSGPU_SERIAL") != NULL; if (on) { ev = [g_dev newEvent]; MLOG("DM_NVN_PASSGPU_SERIAL: passes serialized (for measurement; times read high)"); } }
    if (!on) return;
    (void)ev; (void)n;
}
// Metal 4 conservative barriers (first version): a consumer barrier at the start of an encoder (all prior work â†’ this encoder) and a producer barrier at the end (this encoder and earlier â†’ later encoders).
// Only the one at the start is not enough: t_rt_range 'pass 1 â†’ pass 2 â†’ copy' read back black, and after adding the end barrier it is 19/19. Old-queue encoders (upload / diagnostics / present) do not go through these barriers.
static int g_pass_swrite;   // the open draw pass has shader writes (swrite_note)
static int g_pass_order;    // ordering barriers came in the open draw pass after its draws, none of which wrote (1 primitives, 2 fragments): the next draw that writes is ordered after them first (order_draw)
static int g_pass_late_res; // a draw of the open pass reads memory in a stage after the vertex stage (order_note)
static uint64_t g_order_vbar;   // ordering barriers kept inside the pass as a vertex-to-vertex barrier (prog_swrites)
static void img_fit_upgrade(void); static int g_imfit_npend;   // textures an image slot needs a format view of (img_fit), upgraded before the next draw pass opens
static int g_last_end_relaxed, g_game_barrier;   // the last encoder ended was a draw pass with the relaxed end barrier (nothing else since); the game issued nvnCommandBufferBarrier since the last draw pass began
static uint64_t g_bar_skip, g_bar_full;            // draw passes begun without / with the consumer barrier (DM_NVN_BARRIER=game)
static void gp_end(id e);
static inline void m4_end(id e) {
    gp_end(e);
    MTL4VisibilityOptions vis = e == g_blit ? blit_vis() : resource_vis(); if (e == g_blit) g_blit_alias = 0;
    if ([e conformsToProtocol:@protocol(MTL4CommandEncoder)]) [(id<MTL4CommandEncoder>)e barrierAfterStages:MTLStageAll beforeQueueStages:MTLStageAll visibilityOptions:vis];
    [e endEncoding]; g_last_end_relaxed = 0;
}
// DM_NVN_BARRIER=game (default full): draw passes are ordered by the game's own barriers. A draw pass ends with a barrier from its vertex and fragment stages to the fragment, compute and copy stages of
// later work, so the next pass's vertex work can overlap this pass's fragments (with the full barriers every pass waited for the previous one to finish completely: about 350 passes a frame in Breath of
// the Wild). The next draw pass adds the full consumer barrier when the game called nvnCommandBufferBarrier since the last one (NVN asks for one before results of rendering are read as textures, shader data or
// indirect arguments), or when anything other than a draw pass ran in between (compute, copies and uploads keep their full barriers). Only vertex-stage reads of the previous pass's results without a barrier change order
// DM_NVN_BARRIER=ab... alternates game and full with the DM_AB phases (measurement)
static int sw_mode(const char *name, int dflt); static inline int sw_on(int m);
static int barrier_game(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_NVN_BARRIER"); v = e && !strncmp(e, "ab", 2) ? sw_mode("DM_NVN_BARRIER", 0) : e && !strcmp(e, "game"); if (v == 1) MLOG("DM_NVN_BARRIER=game: draw passes are ordered by the game's own barriers (fragment work still has barriers between passes; vertex work may overlap the previous pass's fragments)"); } return sw_on(v); }
void mtl_barrier_stats(uint64_t *skip, uint64_t *full) { *skip = g_bar_skip; *full = g_bar_full; }
uint64_t mtl_cut_count(void);
// DM_NVN_GPUPROF=first frame[-last frame] (diagnostic): GPU time of every encoder of the main command buffers in those frames. A relaxed timestamp is written into a counter heap when an
// encoder begins (before its first draw or dispatch) and when it ends (after its fragment stage); nothing is serialized beyond the barriers the replay already sets, so the frame runs
// as usual. Printed as "[gpuprof]" lines once the frame's last command buffer completes: one per encoder, then a frame line with the render and compute / copy time, the attachment
// traffic and the span from the first start to the last end. dm/tools/gpuprof.py sums them by pass shape.
// DM_NVN_GPUPROF=every:<n> profiles one frame in every n (frames that are multiples of n), each reported on its own: for a device run whose scene cannot be timed to a frame number.
// A render pass line also gives what it loaded and stored of its attachments (KB, the bandwidth ledger's view: DontCare stores are not counted) and the Metal pixel formats of
// colour attachment 0 and of the depth / stencil attachment (0: none)
enum { GP_MAX = 4096 };   // the largest timestamp heap Metal makes
typedef struct { char kind; uint16_t w, h; uint8_t nc, depth, loads; uint32_t draws, seg; uint64_t pass; uint32_t ld_kb, st_kb; uint16_t cfmt, dfmt; } gp_lab;   // seg: the command buffer (segment) of the submission it is in
static id<MTL4CounterHeap> g_gp_heap; static gp_lab g_gp_lab[GP_MAX / 2]; static int g_gp_n, g_gp_state, g_gp_cur = -1; static long g_gp_f0 = -2, g_gp_f1, g_gp_every; static uint64_t g_gp_frame;
static const void *g_gp_e;   // the encoder g_gp_cur belongs to: m4_end also ends the upload / diagnostics / present encoders, which may end while a profiled pass is open (#531)
static _Atomic int g_gp_busy;   // a periodic profile's report has not been printed yet: the heap is still being read
static uint64_t g_bw_load, g_bw_store, g_gp_bw_l0, g_gp_bw_s0;   // the bandwidth ledger (defined with the pass statistics below) and its values at the last render pass end
static uint64_t g_stat_pass, g_stat_draw; static _Atomic uint64_t g_cur_frame; static uint32_t g_seg; static uint32_t g_gp_copies, g_gp_disp, g_gp_ubd;   // copies / dispatches encoded (for the compute encoders' labels)
static int gp_active(void) {
    if (__builtin_expect(g_gp_f0 == -1, 1)) return 0;
    if (g_gp_f0 == -2) { const char *e = getenv("DM_NVN_GPUPROF");
        if (e && !strncmp(e, "every:", 6)) { g_gp_every = atol(e + 6); g_gp_f0 = g_gp_every > 0 ? 0 : -1; if (g_gp_every > 0) MLOG("DM_NVN_GPUPROF: one frame in every %ld profiled", g_gp_every); }
        else { g_gp_f0 = e ? atol(e) : -1; g_gp_f1 = e && strchr(e, '-') ? atol(strchr(e, '-') + 1) : g_gp_f0; }
        if (g_gp_f0 < 0) return 0; }
    long f = (long)atomic_load(&g_cur_frame);
    if (g_gp_every > 0) return f > 0 && f % g_gp_every == 0;
    return f >= g_gp_f0 && f <= g_gp_f1;
}
static void gp_report(int n, gp_lab *lab, uint64_t frame);
static void gp_periodic_finish(void) {   // on the submit thread, before the next frame's first encoder can touch the heap
    if (!g_gp_every || g_gp_state != 1 || g_gp_frame == atomic_load(&g_cur_frame)) return;
    int n = g_gp_n; gp_lab *lab = malloc(sizeof *lab * (n ? n : 1)); memcpy(lab, g_gp_lab, sizeof *lab * n); uint64_t frame = g_gp_frame;
    g_gp_state = 2; atomic_store(&g_gp_busy, 1);
    static MTLSharedEventListener *listener; if (!listener) listener = [MTLSharedEventListener new];
    // g_fence covers all submissions of the sampled frame. No GPU wait or extra command buffer is needed, including for every:1.
    [g_ev notifyListener:listener atValue:g_fence block:^(id<MTLSharedEvent> event, uint64_t value) { (void)event; (void)value; gp_report(n, lab, frame); }];
}
static void gp_begin(id e, char kind, MTL4RenderPassDescriptor *d) {   // right after an encoder of the main command buffer is made
    gp_periodic_finish();
    if (g_gp_every > 0 && g_gp_state == 2 && !atomic_load(&g_gp_busy)) g_gp_state = 0;   // the previous periodic profile has been reported
    if (!gp_active() || g_gp_state == 2) return;
    if (!g_gp_heap) { MTL4CounterHeapDescriptor *hd = [MTL4CounterHeapDescriptor new]; hd.type = MTL4CounterHeapTypeTimestamp; hd.count = GP_MAX; NSError *er = nil;
        g_gp_heap = [g_dev newCounterHeapWithDescriptor:hd error:&er]; if (!g_gp_heap) { MLOG("[RED] GPUPROF: no counter heap: %s", er.description.UTF8String); g_gp_f0 = -1; return; } }
    if (!g_gp_state) { g_gp_state = 1; g_gp_n = 0; g_gp_frame = atomic_load(&g_cur_frame); [g_gp_heap invalidateCounterRange:NSMakeRange(0, GP_MAX)]; }
    if (2 * g_gp_n + 2 > GP_MAX) return;
    gp_lab l = {kind, 0, 0, 0, 0, 0, kind == 'R' ? (uint32_t)g_stat_draw : g_gp_copies, g_seg, g_stat_pass, 0, 0, 0, 0}; if (kind != 'R') { l.w = (uint16_t)g_gp_disp; l.h = (uint16_t)g_gp_ubd; }
    if (d) { for (int i = 0; i < 8; i++) { MTLRenderPassColorAttachmentDescriptor *c = d.colorAttachments[i]; if (!c.texture) continue;
            if (!l.nc) l.cfmt = (uint16_t)c.texture.pixelFormat; l.nc++; l.w = (uint16_t)MAX(c.texture.width >> c.level, 1); l.h = (uint16_t)MAX(c.texture.height >> c.level, 1); l.loads += c.loadAction == MTLLoadActionLoad; }
        MTLRenderPassDepthAttachmentDescriptor *z = d.depthAttachment;
        if (z.texture || d.stencilAttachment.texture) l.dfmt = (uint16_t)(z.texture ? z.texture : d.stencilAttachment.texture).pixelFormat;
        if (z.texture) { l.depth = 1; l.w = (uint16_t)MAX(z.texture.width >> z.level, 1); l.h = (uint16_t)MAX(z.texture.height >> z.level, 1); l.loads += z.loadAction == MTLLoadActionLoad; } }
    g_gp_lab[g_gp_n] = l; g_gp_cur = g_gp_n; g_gp_e = (__bridge const void *)e;
    if (kind == 'R') [(id<MTL4RenderCommandEncoder>)e writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed afterStage:MTLRenderStageVertex intoHeap:g_gp_heap atIndex:2 * g_gp_n];
    else [(id<MTL4ComputeCommandEncoder>)e writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed intoHeap:g_gp_heap atIndex:2 * g_gp_n];
    g_gp_n++;
}
static void gp_end(id e) {   // right before that encoder ends
    if (g_gp_cur < 0 || (__bridge const void *)e != g_gp_e) return;   // another queue's encoder: the open pass keeps its label and gets its end stamp from its own end
    int k = g_gp_cur; g_gp_cur = -1; g_gp_e = NULL;
    if (g_gp_lab[k].kind == 'R') { g_gp_lab[k].ld_kb = (uint32_t)((g_bw_load - g_gp_bw_l0) >> 10); g_gp_lab[k].st_kb = (uint32_t)((g_bw_store - g_gp_bw_s0) >> 10); }   // since the previous pass end (rp_end); sd_finish has set the stores
    if (g_gp_lab[k].kind == 'R') g_gp_lab[k].draws = (uint32_t)g_stat_draw - g_gp_lab[k].draws; else { g_gp_lab[k].draws = g_gp_copies - g_gp_lab[k].draws; g_gp_lab[k].w = (uint16_t)(g_gp_disp - g_gp_lab[k].w); g_gp_lab[k].h = (uint16_t)(g_gp_ubd - g_gp_lab[k].h); }
    if (g_gp_lab[k].kind == 'R') [(id<MTL4RenderCommandEncoder>)e writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed afterStage:MTLRenderStageFragment intoHeap:g_gp_heap atIndex:2 * k + 1];
    else [(id<MTL4ComputeCommandEncoder>)e writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed intoHeap:g_gp_heap atIndex:2 * k + 1];
}
static void gp_report(int n, gp_lab *lab, uint64_t frame) {   // the frame's last command buffer has completed
    NSData *r = [g_gp_heap resolveCounterRange:NSMakeRange(0, 2 * n)]; const uint64_t *t = r.bytes; if (!t) { free(lab); atomic_store(&g_gp_busy, 0); return; }
    double tick_ns = 1e9 / (double)[g_dev queryTimestampFrequency], tot[2] = {0}, span = 0; uint64_t t0 = 0, t1 = 0, ld = 0, st = 0; int np = 0;
    for (int i = 0; i < n; i++) { uint64_t a = t[2 * i], b = t[2 * i + 1]; if (!a || !b || b < a) continue;
        double us = (b - a) * tick_ns / 1e3; const gp_lab *l = &lab[i]; tot[l->kind != 'R'] += us;
        if (!t0 || a < t0) t0 = a; if (b > t1) t1 = b;
        if (l->kind == 'R') { np++; ld += l->ld_kb; st += l->st_kb;
            MLOG("[gpuprof] F%llu %4d %8.1f us render pass %llu %ux%u colors %u depth %u loads %u draws %u start %+.1f seg %u load %u KB store %u KB fmt %u/%u", (unsigned long long)frame, i, us, (unsigned long long)l->pass, l->w, l->h, l->nc, l->depth, l->loads, l->draws, (a - t[0]) * tick_ns / 1e3, l->seg, l->ld_kb, l->st_kb, l->cfmt, l->dfmt); }
        else MLOG("[gpuprof] F%llu %4d %8.1f us compute / copy after pass %llu start %+.1f seg %u copies %u (uniform updates %u) dispatches %u", (unsigned long long)frame, i, us, (unsigned long long)l->pass, (a - t[0]) * tick_ns / 1e3, l->seg, l->draws, l->h, l->w); }
    span = (t1 - t0) * tick_ns / 1e3;
    MLOG("[gpuprof] frame %llu: render %.2f ms, compute / copy %.2f ms over %d encoders (%d render passes, attachments load %.1f MB store %.1f MB), first start to last end %.2f ms", (unsigned long long)frame, tot[0] / 1e3, tot[1] / 1e3, n, np, ld / 1024.0, st / 1024.0, span / 1e3);
    free(lab); atomic_store(&g_gp_busy, 0);
}
static inline id<MTL4RenderCommandEncoder> enc_r_begin_pass(MTL4RenderPassDescriptor *d) {   // the draw passes (rp_begin_ex); see barrier_game
    id<MTL4RenderCommandEncoder> e = [g_cb renderCommandEncoderWithDescriptor:d];
    if (g_last_end_relaxed && !g_game_barrier && barrier_game() && !g_image_heap_n) g_bar_skip++;
    else { [e barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:resource_vis()]; g_bar_full++; }
    g_game_barrier = 0;
    [e setArgumentTable:g_vt atStages:MTLRenderStageVertex]; [e setArgumentTable:g_ft atStages:MTLRenderStageFragment]; gp_begin(e, 'R', d); return e;
}
static inline void enc_r_end_pass(id e) {
    if (!barrier_game() || g_image_heap_n) { m4_end(e); return; }
    gp_end(e); [(id<MTL4CommandEncoder>)e barrierAfterStages:MTLStageVertex | MTLStageMesh | MTLStageFragment beforeQueueStages:MTLStageFragment | MTLStageDispatch | MTLStageBlit visibilityOptions:resource_vis()];
    [e endEncoding]; g_last_end_relaxed = 1;
}
static inline id<MTL4RenderCommandEncoder> enc_r_begin(MTL4RenderPassDescriptor *d) { id<MTL4RenderCommandEncoder> e = [g_cb renderCommandEncoderWithDescriptor:d]; [e barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:resource_vis()]; [e setArgumentTable:g_vt atStages:MTLRenderStageVertex]; [e setArgumentTable:g_ft atStages:MTLRenderStageFragment]; gp_begin(e, 'R', d); return e; }
static inline void enc_r_end(id e) { m4_end(e); }
static inline void enc_r_pso(id<MTL4RenderCommandEncoder> e, id<MTLRenderPipelineState> p) { [e setRenderPipelineState:p]; }
static inline void enc_r_dss(id<MTL4RenderCommandEncoder> e, id<MTLDepthStencilState> s) { [e setDepthStencilState:s]; }
static inline void enc_r_viewport(id<MTL4RenderCommandEncoder> e, MTLViewport v) { [e setViewport:v]; }
static inline void enc_r_scissor(id<MTL4RenderCommandEncoder> e, MTLScissorRect r) { [e setScissorRect:r]; }
static inline void enc_r_cull(id<MTL4RenderCommandEncoder> e, MTLCullMode m) { [e setCullMode:m]; }
static inline void enc_r_winding(id<MTL4RenderCommandEncoder> e, MTLWinding w) { [e setFrontFacingWinding:w]; }
static inline void enc_r_fill(id<MTL4RenderCommandEncoder> e, MTLTriangleFillMode m) { [e setTriangleFillMode:m]; }
static inline void enc_r_depth_bias(id<MTL4RenderCommandEncoder> e, float u, float s, float c) { [e setDepthBias:u slopeScale:s clamp:c]; }
static inline void enc_r_depth_clip(id<MTL4RenderCommandEncoder> e, BOOL v) { [e setDepthClipMode:v ? MTLDepthClipModeClip : MTLDepthClipModeClamp]; }
static inline void enc_r_stencil_ref(id<MTL4RenderCommandEncoder> e, uint32_t f, uint32_t b) { (void)b; [e setStencilReferenceValue:f]; }
static inline void enc_r_blend_color(id<MTL4RenderCommandEncoder> e, float r, float g, float b, float a) { [e setBlendColorRed:r green:g blue:b alpha:a]; }
static inline void enc_r_visibility(id<MTL4RenderCommandEncoder> e, MTLVisibilityResultMode m, NSUInteger o) { [e setVisibilityResultMode:m offset:o]; }
// The table persists across encoders and submissions. Track every write to it, including converted streams,
// transform feedback and shader arguments; helper passes with their own tables do not change this state.
static inline void enc_r_vbuf(id<MTL4RenderCommandEncoder> e, NSUInteger i, uint64_t gpu) {
    (void)e; uint32_t bit = 1u << i;
    if (g_vbuf_cache && (g_vt_valid & bit) && g_vt_addr[i] == gpu) { g_vbuf_skip++; return; }
    if (rec_on()) rec_table_op(RO_T_ADDR, g_vt, gpu, i); else [g_vt setAddress:gpu atIndex:i];
    g_vt_addr[i] = gpu; g_vt_valid |= bit; g_vbuf_set++;
}
static inline void enc_r_args(id<MTL4RenderCommandEncoder> e, MTLRenderStages s, uint64_t gpu, NSUInteger i) {
    if (s == MTLRenderStageVertex) { enc_r_vbuf(e, i, gpu); return; }
    id<MTL4ArgumentTable> t = s == MTLRenderStageObject ? g_ot : s == MTLRenderStageMesh ? g_mt : g_ft;
    if (rec_on()) rec_table_op(RO_T_ADDR, t, gpu, i); else [t setAddress:gpu atIndex:i];
}
static inline void enc_r_draw(id<MTL4RenderCommandEncoder> e, MTLPrimitiveType p, NSUInteger v, NSUInteger n) { [e drawPrimitives:p vertexStart:v vertexCount:n]; }
static inline void enc_r_draw_instanced(id<MTL4RenderCommandEncoder> e, MTLPrimitiveType p, NSUInteger v, NSUInteger n, NSUInteger i, NSUInteger b) { [e drawPrimitives:p vertexStart:v vertexCount:n instanceCount:i baseInstance:b]; }
static inline void enc_r_draw_indirect(id<MTL4RenderCommandEncoder> e, MTLPrimitiveType p, uint64_t gpu) { [e drawPrimitives:p indirectBuffer:gpu]; }
static inline void enc_r_draw_indexed(id<MTL4RenderCommandEncoder> e, MTLPrimitiveType p, NSUInteger n, MTLIndexType t, uint64_t gpu, NSUInteger len, NSUInteger i, NSInteger bv, NSUInteger bi) { [e drawIndexedPrimitives:p indexCount:n indexType:t indexBuffer:gpu indexBufferLength:len instanceCount:i baseVertex:bv baseInstance:bi]; }
static inline void enc_r_draw_indexed_indirect(id<MTL4RenderCommandEncoder> e, MTLPrimitiveType p, MTLIndexType t, uint64_t gpu, NSUInteger len, uint64_t ind) { [e drawIndexedPrimitives:p indexType:t indexBuffer:gpu indexBufferLength:len indirectBuffer:ind]; }
static void ubd_land_compute(void);
static inline id<MTL4ComputeCommandEncoder> enc_c_begin(MTLComputePassDescriptor *d) { (void)d; ubd_land_compute(); id<MTL4ComputeCommandEncoder> e = [g_cb computeCommandEncoder]; [e barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:resource_vis()]; [e setArgumentTable:g_ct]; gp_begin(e, 'C', nil); return e; }
static inline id<MTLComputeCommandEncoder> enc_c_begin_concurrent(id<MTLCommandBuffer> cb) { return [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]; }
static inline void enc_c_end(id e) { m4_end(e); }
static inline void enc_c_pso(id e, id<MTLComputePipelineState> p) { [e setComputePipelineState:p]; }
static inline void enc_c_args(id e, uint64_t gpu, NSUInteger i) { (void)e; if (rec_on()) rec_table_op(RO_T_ADDR, g_ct, gpu, i); else [g_ct setAddress:gpu atIndex:i]; }
static inline void enc_c_dispatch(id e, MTLSize n, MTLSize t) { [e dispatchThreadgroups:n threadsPerThreadgroup:t]; }
static inline void enc_c_dispatch_indirect(id e, uint64_t ind, MTLSize t) { [(id<MTL4ComputeCommandEncoder>)e dispatchThreadgroupsWithIndirectBuffer:ind threadsPerThreadgroup:t]; }
static inline id enc_b_begin(id cb) { return [cb respondsToSelector:@selector(beginCommandBufferWithAllocator:)] ? [cb computeCommandEncoder] : [cb blitCommandEncoder]; }
static inline void enc_b_end(id e) { m4_end(e); }
static inline void enc_b_copy_buffer(id e, id<MTLBuffer> s, NSUInteger so, id<MTLBuffer> d, NSUInteger doff, NSUInteger n) { [e copyFromBuffer:s sourceOffset:so toBuffer:d destinationOffset:doff size:n]; }
static inline void enc_b_copy_buffer_texture(id e, id<MTLBuffer> s, NSUInteger so, NSUInteger row, NSUInteger img, MTLSize z, id<MTLTexture> d, NSUInteger sl, NSUInteger lv, MTLOrigin o) { [e copyFromBuffer:s sourceOffset:so sourceBytesPerRow:row sourceBytesPerImage:img sourceSize:z toTexture:d destinationSlice:sl destinationLevel:lv destinationOrigin:o]; }
static inline void enc_b_copy_texture_buffer(id e, id<MTLTexture> s, NSUInteger sl, NSUInteger lv, MTLOrigin o, MTLSize z, id<MTLBuffer> d, NSUInteger doff, NSUInteger row, NSUInteger img) { [e copyFromTexture:s sourceSlice:sl sourceLevel:lv sourceOrigin:o sourceSize:z toBuffer:d destinationOffset:doff destinationBytesPerRow:row destinationBytesPerImage:img]; }
// The same with a blit option: one plane of a combined depth / stencil texture (MTLBlitOptionDepthFromDepthStencil)
static inline void enc_b_copy_buffer_texture_opt(id e, id<MTLBuffer> s, NSUInteger so, NSUInteger row, NSUInteger img, MTLSize z, id<MTLTexture> d, NSUInteger sl, NSUInteger lv, MTLOrigin o, MTLBlitOption opt) { [e copyFromBuffer:s sourceOffset:so sourceBytesPerRow:row sourceBytesPerImage:img sourceSize:z toTexture:d destinationSlice:sl destinationLevel:lv destinationOrigin:o options:opt]; }
static inline void enc_b_copy_texture_buffer_opt(id e, id<MTLTexture> s, NSUInteger sl, NSUInteger lv, MTLOrigin o, MTLSize z, id<MTLBuffer> d, NSUInteger doff, NSUInteger row, NSUInteger img, MTLBlitOption opt) { [e copyFromTexture:s sourceSlice:sl sourceLevel:lv sourceOrigin:o sourceSize:z toBuffer:d destinationOffset:doff destinationBytesPerRow:row destinationBytesPerImage:img options:opt]; }
static inline void enc_b_copy_texture(id e, id<MTLTexture> s, NSUInteger sl, NSUInteger lv, MTLOrigin so, MTLSize z, id<MTLTexture> d, NSUInteger dl, NSUInteger dv, MTLOrigin doo) { [e copyFromTexture:s sourceSlice:sl sourceLevel:lv sourceOrigin:so sourceSize:z toTexture:d destinationSlice:dl destinationLevel:dv destinationOrigin:doo]; }
static void blit_note(void);   // Per-pass dump: a copy / buffer update ended the render pass (defined after PD)
// Ordering inside the copy encoder: Metal 4 does not order copies / fills within one encoder (no automatic hazard tracking); an operation reading or writing a range an earlier one wrote needs an intra-encoder barrier.
// Without it, SMBW's two updates per frame to the same uniform slot ("write 4 bytes âˆ’1, then 16 bytes of coefficients") occasionally landed in reverse order; the fountain water wave simulation took âˆ’1 as its self coefficient, diverged, and the water vertices were pushed to the top of the screen.
// Buffer operations check conflicts by guest address range: a barrier when an operation reads a range written earlier, or writes a range read or written earlier; texture operations get
// one whenever the encoder already holds an operation. Reads were not tracked at first, so a copy writing bytes an earlier copy reads could run first (seen in Cuphead while its pool
// allocator gave out overlapping blocks, see hle_nvn.c zcull_size; NVN runs copies in order for any game)
enum { BW_MAX = 64 };
static uint64_t g_bw_lo[BW_MAX], g_bw_hi[BW_MAX], g_br_lo[BW_MAX], g_br_hi[BW_MAX]; static int g_bw_n, g_br_n, g_blit_ops; static uint64_t g_blit_barriers;
static uint64_t g_blit_splits;   // copies / buffer updates / fills that ended an open draw pass (the residency set line; a pass ended this way stores and the next one reloads its attachments)
uint64_t mtl_blit_splits(void) { return g_blit_splits; }
static int enc_open(void);
static int g_blit_ct;   // the open copy encoder has the compute argument table set (dispatches share it, dispatch_enc)
static id<MTL4ComputeCommandEncoder> blit_open(void) { if (!g_blit) { ser_gate(); g_blit = enc_b_begin(g_cb); [g_blit barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:blit_vis()]; gp_begin(g_blit, 'C', nil); g_bw_n = g_br_n = 0; g_blit_ops = 0; g_blit_ct = 0; } return g_blit; }
static void ubd_flush(void); static int g_nubd;   // the inline updates deferred (see mtl_sub_update)
static int g_ubd_landing;
static id<MTL4ComputeCommandEncoder> blit_raw(void) {
    blit_note(); if (enc_open()) g_blit_splits++; rp_end(); id<MTL4ComputeCommandEncoder> e = blit_open();
    if (g_nubd && !g_ubd_landing) { g_ubd_landing = 1; ubd_flush(); g_ubd_landing = 0; }   // deferred inline updates come before this copy or dispatch in command order
    return e;
}
// Land the deferred inline updates before other GPU work outside the copy encoder (helper compute passes, the end of the submission, a GPU wait)
static void ubd_land(void) { if (g_nubd && !g_ubd_landing) { rp_end(); g_ubd_landing = 1; ubd_flush(); g_ubd_landing = 0; } }
static void ubd_land_compute(void) { if (g_nubd && !g_ubd_landing) { ubd_land(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; } } }   // before a compute encoder of its own (no other encoder may be open then)
static void blit_barrier(void) {
    static int off = -1; if (off < 0) { const char *e = getenv("DM_NVN_BLIT_ORDER"); off = e && !strcmp(e, "0"); if (off) MLOG("DM_NVN_BLIT_ORDER=0: no barriers inside the copy encoder (diagnostic; brings back the fountain's vertical stripes)"); }
    // Only the stages a compute encoder has: MTLStageAll here fails Metal validation (macOS 27: "must be a valid combination of MTLStages for the current encoder type")
    if (!off) [g_blit barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit beforeEncoderStages:MTLStageDispatch | MTLStageBlit visibilityOptions:blit_vis()];
    g_bw_n = g_br_n = 0; g_blit_ops = 0; g_blit_barriers++;
}
uint64_t mtl_blit_barriers(void) { return g_blit_barriers; }
static int bw_hit(uint64_t lo, uint64_t hi) { for (int i = 0; i < g_bw_n; i++) if (lo < g_bw_hi[i] && g_bw_lo[i] < hi) return 1; return 0; }
static int br_hit(uint64_t lo, uint64_t hi) { for (int i = 0; i < g_br_n; i++) if (lo < g_br_hi[i] && g_br_lo[i] < hi) return 1; return 0; }
// buffer â†’ buffer / fill: reads [rlo, rhi) (0, 0 when nothing is read), writes [wlo, whi)
static uint64_t g_wr_epoch;   // Bumped by every buffer write outside a render pass (copy / update / fill, texture â†’ buffer, dispatch): indirect-draw conversions made before it are stale (ixc_run)
static uint64_t g_mdx_epoch;  // Bumped by GPU writes that can change an indirect-count multi-draw's records or count (copy, fill, texture â†’ buffer, dispatch; see mdx_run)
// Writes since the last drain. A drain (mdi_wait: commit, then wait for the GPU) completes every GPU write encoded so far, so until the submission ends a CPU read of guest memory is
// stale only when a GPU write encoded after the drain overlaps it. The submission-wide sets (pend, sbw) cannot tell before from after, so each later write is also kept here: storage
// writes and copies, updates and fills (sbw_note, pend_add). The list is trusted only while every bump of g_wr_epoch since the drain came from a geometry kernel (g_wr_geo), whose
// storage writes sbw_note sees; another bump (transform feedback, for one) or an overflow sends readers back to the submission-wide sets. Before, every geometry draw of an
// indirect-count multi-draw drained the GPU for its count and its record, since the draw before it had dispatched its geometry kernel: about 1,900 drains a frame for three frames as
// Donkey Kong Bananza enters its first level (submissions of 330-520 ms of which 180 ms CPU)
enum { DRW_N = 512 };
static uint64_t g_drw_lo[DRW_N], g_drw_hi[DRW_N], g_drain_sub = ~0ull, g_drain_other, g_wr_geo; static int g_ndrw, g_drw_over;
static void drw_note(uint64_t lo, uint64_t n) {
    if (g_drain_sub != g_sub_serial || g_drw_over) return;
    if (!n || lo > UINT64_MAX - n) { g_drw_over = 1; return; }
    for (int i = g_ndrw - 1; i >= 0 && i >= g_ndrw - 8; i--) if (lo <= g_drw_hi[i] && g_drw_lo[i] <= lo + n) { g_drw_lo[i] = MIN(lo, g_drw_lo[i]); g_drw_hi[i] = MAX(lo + n, g_drw_hi[i]); return; }
    if (g_ndrw == DRW_N) { g_drw_over = 1; return; }
    g_drw_lo[g_ndrw] = lo; g_drw_hi[g_ndrw++] = lo + n;
}
static inline void args_dirty_all(void);
// Bindings re-encode after the drain (encode_args notes their storage writes only when it encodes, not when it reuses an argument buffer)
static void drw_drained(void) { g_drain_sub = g_sub_serial; g_drain_other = g_wr_epoch - g_wr_geo; g_ndrw = g_drw_over = 0; args_dirty_all(); }
static int drw_valid(void) { return g_drain_sub == g_sub_serial && !g_drw_over && g_wr_epoch - g_wr_geo == g_drain_other; }
static int drw_hits(uint64_t lo, uint64_t n) { for (int i = 0; i < g_ndrw; i++) if (lo < g_drw_hi[i] && g_drw_lo[i] < lo + n) return 1; return 0; }
int nvn_sparse_alias(uint64_t lo, uint64_t hi);   // hle_nvn.c: a sparse guest range may alias another pool at a different address
static id<MTL4ComputeCommandEncoder> blit_buf(uint64_t rlo, uint64_t rhi, uint64_t wlo, uint64_t whi) {
    g_wr_epoch++;
    g_blit_alias = 1;   // before blit_raw, so an encoder it opens starts with alias visibility
    id<MTL4ComputeCommandEncoder> e = blit_raw();
    g_blit_alias = 1;   // again: blit_raw may have ended an encoder in between
    // Sparse mappings can join unrelated physical runs, so conservatively order
    // their operations with everything before and after. Ordinary pools keep the
    // range-based batching, including separate Metal buffers over one CPU range.
    if (nvn_sparse_alias(rlo, rhi) || nvn_sparse_alias(wlo, whi)) { wlo = 0; whi = ~0ull; }
    if (g_blit_ops && (g_bw_n == BW_MAX || g_br_n == BW_MAX || (rhi > rlo && bw_hit(rlo, rhi)) || bw_hit(wlo, whi) || br_hit(wlo, whi))) blit_barrier();
    g_bw_lo[g_bw_n] = wlo; g_bw_hi[g_bw_n++] = whi; g_blit_ops++; g_gp_copies++;
    if (rhi > rlo) { g_br_lo[g_br_n] = rlo; g_br_hi[g_br_n++] = rhi; }
    return e;
}
// texture copies / readbacks: barrier if anything came before, and later operations must wait for it too
static id<MTL4ComputeCommandEncoder> blit(void) {
    id<MTL4ComputeCommandEncoder> e = blit_raw();
    if (g_blit_ops) blit_barrier();
    g_bw_lo[0] = 0; g_bw_hi[0] = ~0ull; g_bw_n = 1; g_br_n = 0; g_blit_ops = 1;   // record as "everything written": the next operation always gets a barrier
    return e;
}

void mtl_sub_copy(uint64_t src, uint64_t dst, uint64_t n) {
    pend_add(dst, n, 2); g_mdx_epoch++;
    void *sb, *db; uint64_t so, dof;
    if (!mtl_addr(src, n, &sb, &so) || !mtl_addr(dst, n, &db, &dof))
        MDIE("buffer copy address is in no memory pool src=%#llx dst=%#llx n=%#llx", (unsigned long long)src, (unsigned long long)dst, (unsigned long long)n);
    if (src != dst && src < dst + n && dst < src + n) { static int once;   // Metal leaves an overlapping copy undefined; seen only in Cuphead while its pool allocator gave out overlapping blocks (hle_nvn.c zcull_size)
        if (!once++) MLOG("[RED] buffer copy with overlapping source and destination (src %#llx dst %#llx, %llu bytes): the result is undefined", (unsigned long long)src, (unsigned long long)dst, (unsigned long long)n); }
    enc_b_copy_buffer(blit_buf(src, src + n, dst, dst + n), (__bridge id<MTLBuffer>)sb, so, (__bridge id<MTLBuffer>)db, dof, n);
}

// Inline uniform update (UpdateUniformBuffer): the data is copied into the host's own shared ring at encode time, then copied in order to the destination.
// Previously it was read from command memory only when the GPU executed - if the game recycles command memory before the GPU runs, it reads the bytes of the next batch of commands (SMBW's fountain water wave simulation got the wrong coefficients and diverged to Â±65504).
static id<MTLBuffer> g_ubs_buf; static size_t g_ubs_off, g_ubs_size;
// Inline updates inside an open render pass are deferred instead of ending the pass. Metal cannot copy into a buffer inside a render pass, so each update used to end it: all attachments stored,
// reloaded by the next pass, new encoders (Metroid Dread about 246 times a frame, Tears of the Kingdom about 50). NVN applies the update in command order, so the draws after it in the pass must
// read the new data: a draw whose uniform binding overlaps a deferred update reads a shadow. When the updates deferred in this pass cover all of the binding the shader can read, the shadow is
// made of them alone, latest first per byte (exactly what the GPU would read, whatever earlier frames still have in flight); when they cover only part of it, the rest comes from memory (ubp_get).
// The real copies are encoded in order when the pass ends (rp_end), so later passes and the CPU see them as before. The pass still ends before a draw that would need more: a partly covered binding
// whose rest memory cannot give, or a read of the deferred data through a vertex, index, indirect or storage buffer (ubd_conflict). Off in TBDR mode (it reorders passes). DM_NVN_UB_DEFER=0 turns it off
static size_t arena_alloc(size_t n, __strong id<MTLBuffer> *buf); static inline void args_dirty_all(void); static void ubd_dirty(uint64_t lo, uint64_t hi);
typedef struct { uint64_t lo, hi, gpu; size_t off; __unsafe_unretained id<MTLBuffer> buf; } ubd_ent;   // data at buf + off (the ubs ring; a ring block replaced meanwhile lives until this submission completes)
enum { UBD_MAX = 512 };
static ubd_ent g_ubd[UBD_MAX]; static int g_nubd;
static uint64_t g_ubd_deferred, g_ubd_conflicts;   // updates deferred, passes ended before a draw that needed more (the residency set line)
static int ubd_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_NVN_UB_DEFER"); v = !(e && !strcmp(e, "0")); if (!v) MLOG("DM_NVN_UB_DEFER=0: uniform updates inside a pass end the pass as before"); } return v && !tbdr_on(); }
void mtl_ubd_stats(uint64_t *deferred, uint64_t *conflicts) { *deferred = g_ubd_deferred; *conflicts = g_ubd_conflicts; }
// Across passes (DM_NVN_UB_LAZY=n, at most n updates kept past a pass end, default 160; 0 = landed at every pass end as before): the deferred updates are not landed when their pass ends but
// when something needs the memory itself: a copy or dispatch (they go into that compute encoder first), a draw that reads them other than as a uniform buffer the shadows cover
// (ubd_conflict), the end of the submission, more than n of them. An update with no pass
// open is deferred too, unless a compute encoder is open (it then goes in there). Breath of the Wild had about 45 small copy encoders between its render passes a frame, each with
// 10-50 Âµs of idle GPU before and after it: GPU time 9.3 â†’ 8.0 ms a frame, frame 11.9 â†’ 10.7 ms at the 120 fps cap (M2 Max), for about 0.8 ms more translation (more shadows).
// The list can then hold hundreds of updates: a page filter answers most lookups without the scan
static int ubd_lazy(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_NVN_UB_LAZY"); v = e ? atoi(e) : 160; if (v < 0) v = 0; } return v; }   // the most updates kept across a pass end (0 = off)
enum { UBD_PF = 1024 };
static uint64_t g_ubd_pf[UBD_PF];   // pages (address >> 12, +1) the deferred updates touch, open addressing; cleared when the list is landed
static int g_ubd_wide;              // an update spans more than 16 pages: the filter is skipped
static inline unsigned ubd_pf_slot(uint64_t pg) { return (unsigned)((pg * 0x9E3779B97F4A7C15ull) >> 54) & (UBD_PF - 1); }
static void ubd_pf_add(uint64_t lo, uint64_t hi) {
    if (((hi - 1) >> 12) - (lo >> 12) > 16) { g_ubd_wide = 1; return; }
    for (uint64_t pg = lo >> 12; pg <= (hi - 1) >> 12; pg++) { unsigned k = ubd_pf_slot(pg);
        while (g_ubd_pf[k] && g_ubd_pf[k] != pg + 1) k = (k + 1) & (UBD_PF - 1);
        g_ubd_pf[k] = pg + 1; }
}
static inline int ubd_pf_hit(uint64_t lo, uint64_t hi) {
    if (g_ubd_wide || ((hi - 1) >> 12) - (lo >> 12) > 16) return 1;
    for (uint64_t pg = lo >> 12; pg <= (hi - 1) >> 12; pg++) { unsigned k = ubd_pf_slot(pg);
        while (g_ubd_pf[k]) { if (g_ubd_pf[k] == pg + 1) return 1; k = (k + 1) & (UBD_PF - 1); } }
    return 0;
}
static int ubd_hits(uint64_t lo, uint64_t hi) {
    if (hi <= lo || !ubd_pf_hit(lo, hi)) return 0;
    for (int i = 0; i < g_nubd; i++) if (lo < g_ubd[i].hi && g_ubd[i].lo < hi) return 1;
    return 0;
}
// Whether the deferred updates cover [lo, hi) entirely (any order of overlap): the union of their ranges clipped to it, swept from lo
static int ubd_covers(uint64_t lo, uint64_t hi) {
    for (uint64_t at = lo; at < hi;) { uint64_t next = at;
        for (int i = 0; i < g_nubd; i++) if (g_ubd[i].lo <= at && g_ubd[i].hi > next) next = g_ubd[i].hi;
        if (next == at) return 0; at = next; }
    return 1;
}
// GPU address of the shadow of uniform range [lo, hi) (ubd_covers holds): one deferred update holding all of it and suitably aligned is used in place; otherwise the updates are laid into the arena in order
static uint64_t ubd_shadow(uint64_t lo, uint64_t hi) {
    for (int i = g_nubd - 1; i >= 0; i--) { const ubd_ent *e = &g_ubd[i];
        if (e->lo <= lo && hi <= e->hi) { int later = 0; for (int j = i + 1; j < g_nubd; j++) later |= lo < g_ubd[j].hi && g_ubd[j].lo < hi;
            uint64_t g = e->gpu + (lo - e->lo); if (!later && !(g & 15)) return g; break; } }
    id<MTLBuffer> ab; size_t o = arena_alloc((size_t)(hi - lo), &ab); uint8_t *d = (uint8_t *)ab.contents + o;
    for (int i = 0; i < g_nubd; i++) { const ubd_ent *e = &g_ubd[i]; uint64_t a = e->lo > lo ? e->lo : lo, b = e->hi < hi ? e->hi : hi;
        if (a < b) memcpy(d + (a - lo), (const uint8_t *)e->buf.contents + e->off + (a - e->lo), (size_t)(b - a)); }
    return ab.gpuAddress + o;
}
// Encode the deferred copies in order (rp_end, once the pass has ended)
static void ubd_flush(void) {
    int n = g_nubd; g_nubd = 0; memset(g_ubd_pf, 0, sizeof g_ubd_pf); g_ubd_wide = 0;
    for (int i = 0; i < n; i++) { const ubd_ent *e = &g_ubd[i]; void *db; uint64_t dof;
        if (!mtl_addr(e->lo, e->hi - e->lo, &db, &dof)) continue;
        enc_b_copy_buffer(blit_buf(0, 0, e->lo, e->hi), e->buf, e->off, (__bridge id<MTLBuffer>)db, dof, e->hi - e->lo); g_gp_ubd++; }
}
// The bytes [lo, lo+n) as the deferred updates leave them (ubd_covers holds): for bindless handles read from uniform memory (ubo_handle)
static void ubd_read(uint64_t lo, size_t n, void *out) {
    for (int i = 0; i < g_nubd; i++) { const ubd_ent *e = &g_ubd[i]; uint64_t a = e->lo > lo ? e->lo : lo, b = e->hi < lo + n ? e->hi : lo + n;
        if (a < b) memcpy((uint8_t *)out + (a - lo), (const uint8_t *)e->buf.contents + e->off + (a - e->lo), (size_t)(b - a)); }
}
// Bindings the deferred updates cover only in part. The rest of the binding is what memory holds when the pass runs on the GPU. The shadow then starts from memory as the CPU sees it at encode
// time and lays over it every inline update queued in this submission that overlaps it, in queue order (earlier passes' updates, encoded but not yet run, and this pass's deferred ones): exactly
// what the GPU would read. It is made only when nothing else can have written the range by then: no copy, fill or feedback write queued in this submission (pend kinds 2â€“4), no writable storage
// buffer, textureâ†’buffer copy or counter report over it in this submission (sbw), and no GPU write of an earlier submission still running (gw, by 4 KB page). Otherwise the pass ends as before.
// Breath of the Wild updates part of a bound block about 226 times a frame outdoors and ended a pass for nearly every one (about 900 MB of attachment loads and stores a frame on an iPhone).
// DM_NVN_UB_PARTIAL=0 turns it off
static int ub_partial_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_NVN_UB_PARTIAL"); v = !(e && !strcmp(e, "0")); if (!v) MLOG("DM_NVN_UB_PARTIAL=0: a binding the deferred uniform updates cover only in part ends the pass as before"); } return v; }
enum { GW_N = 4096, GW_PAGES = 4096 };
typedef struct { uint64_t page, fence; } gw_ent;
static gw_ent g_gw[GW_N];   // page ~0: two live pages met in the slot, so any page there counts as written until fence completes
static gw_ent g_gwx[GW_N];  // the same without the inline updates the host remembers (um_ent): what a uniform shadow cannot read from memory
static uint64_t g_gw_all;                            // every page counts as written until this fence completes (a submission whose writes could not all be listed)
static uint64_t g_ubp_made, g_ubp_reused, g_ubp_refused;   // partial shadows made / taken from the cache / refused (the pass ended), for the residency set line
static uint64_t g_ubp_why[5];   // why ubp_get refused: 0 too large or off, 1 other GPU writes in this submission (sbw), 2 GPU writes of an earlier submission still running, 3 a copy, fill or feedback write queued, 4 unreadable or too many writes
static inline unsigned gw_slot(uint64_t pg) { return (unsigned)((pg * 0x9E3779B97F4A7C15ull) >> 52) & (GW_N - 1); }
// Writes of GW_PAGES pages or more, kept whole. They used to count as a write anywhere (g_gw_all): one storage binding over 16 MB in each submission (Splatoon Raiders) refused every uniform shadow of the next one, and the
// pass ended for each. The same range comes back every submission, so few are live at a time; only when none of the slots is free does it still count as a write anywhere
enum { GWW_N = 32 };
static struct { uint64_t lo, hi, fence; } g_gww[GWW_N];
static void gww_put(uint64_t lo, uint64_t hi, uint64_t fence, uint64_t done) {
    int free_slot = -1;
    for (int i = 0; i < GWW_N; i++) { if (g_gww[i].lo == lo && g_gww[i].hi == hi) { if (fence > g_gww[i].fence) g_gww[i].fence = fence; return; } if (free_slot < 0 && g_gww[i].fence <= done) free_slot = i; }
    if (free_slot >= 0) { g_gww[free_slot].lo = lo; g_gww[free_slot].hi = hi; g_gww[free_slot].fence = fence; }
    else if (fence > g_gw_all) g_gw_all = fence;
}
static void gw_put_in(gw_ent *t, uint64_t lo, uint64_t hi, uint64_t fence, uint64_t done) {
    if (hi <= lo) return;
    uint64_t p0 = lo >> 12, p1 = (hi - 1) >> 12;
    if (p1 - p0 >= GW_PAGES) { gww_put(lo, hi, fence, done); return; }
    for (uint64_t pg = p0; pg <= p1; pg++) { unsigned s = gw_slot(pg);
        t[s].page = t[s].page == pg || t[s].fence <= done ? pg : ~0ull;
        if (fence > t[s].fence) t[s].fence = fence; }
}
static void gw_put(uint64_t lo, uint64_t hi, uint64_t fence, uint64_t done) { gw_put_in(g_gw, lo, hi, fence, done); gw_put_in(g_gwx, lo, hi, fence, done); }
static int gw_live_in(const gw_ent *t, uint64_t lo, uint64_t hi) {
    uint64_t done = g_done;   // read without the lock as elsewhere: a stale value only refuses more
    if (g_gw_all > done) return 1;
    for (int i = 0; i < GWW_N; i++) if (g_gww[i].fence > done && lo < g_gww[i].hi && g_gww[i].lo < hi) return 1;
    for (uint64_t pg = lo >> 12; pg <= (hi - 1) >> 12; pg++) { unsigned s = gw_slot(pg); if ((t[s].page == pg || t[s].page == ~0ull) && t[s].fence > done) return 1; }
    return 0;
}
static int gw_live(uint64_t lo, uint64_t hi) { return gw_live_in(g_gw, lo, hi); }
// Inline updates of submissions still running, remembered by byte. A partly covered uniform shadow takes the rest from memory, and memory does not hold them until their submission has run, so they used to refuse
// the shadow and end the pass. A game that updates a block every frame refused it for as long as the last frame's work was running, and the slower the GPU the longer: each refusal is one more pass, which made the GPU
// slower still. Splatoon Raiders updates 0x00..0x14 and 0x18..0x70 of its ink query block before each of about 150 queries a frame while the shader reads 0x00..0x70: 150 passes on a 6400x6400 target, about 6 fps.
// The CPU knows these bytes, so the shadow lays them over memory instead. A byte counts while its fence runs, so what the game writes later (or a copy of a later submission lands) is not overwritten once it has finished.
// A slot that a page still running holds leaves the other page to g_gwx: refused as before. DM_NVN_UBP_RUNNING=0 refuses them all as before
enum { UM_N = 256 };
typedef struct { uint64_t page, fence; uint64_t f[4096]; uint8_t val[4096]; } um_ent;   // f: each byte's fence; fence: the highest of them
static um_ent *g_um;
static inline unsigned um_slot(uint64_t pg) { return (unsigned)((pg * 0x9E3779B97F4A7C15ull) >> 40) & (UM_N - 1); }
static int um_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_NVN_UBP_RUNNING"); v = !(e && !strcmp(e, "0")); if (!v) MLOG("DM_NVN_UBP_RUNNING=0: inline updates of a running submission refuse partly covered uniform shadows"); } return v; }
static void um_put(uint64_t lo, uint64_t hi, const uint8_t *src, uint64_t fence, uint64_t done) {
    if (!g_um) g_um = calloc(UM_N, sizeof *g_um);   // about 13 MB of address space; only the pages a game updates are touched
    for (uint64_t at = lo, end; at < hi; at = end) { uint64_t pg = at >> 12; end = MIN((pg + 1) << 12, hi); um_ent *m = &g_um[um_slot(pg)];
        if (m->page != pg && m->fence > done) { gw_put_in(g_gwx, at, end, fence, done); continue; }
        m->page = pg; m->fence = fence;   // a page taken over: all its bytes' fences have finished
        unsigned o = (unsigned)(at & 4095), n = (unsigned)(end - at);
        memcpy(&m->val[o], src + (at - lo), n); for (unsigned i = 0; i < n; i++) m->f[o + i] = fence; }
}
// The submission's GPU writes go into the page tables with its fence (mtl_sub_commit)
static void gw_commit(uint64_t fence) {
    uint64_t done = g_done;
    if ((g_pend_over || g_sbw_over) && fence > g_gw_all) g_gw_all = fence;
    for (int e = 0; e < g_npend; e++) {
        gw_put_in(g_gw, g_pend_lo[e], g_pend_hi[e], fence, done);
        if (g_pend_kind[e] == 1 && g_pend_upd[e] && um_on()) { const upd_ent *u = &g_upd[g_pend_upd[e] - 1]; um_put(g_pend_lo[e], g_pend_hi[e], (const uint8_t *)u->buf.contents + u->off + (g_pend_lo[e] - u->lo), fence, done); }
        else gw_put_in(g_gwx, g_pend_lo[e], g_pend_hi[e], fence, done); }
    if (g_nsbw) for (int i = 0; i < SBW_N; i++) if (g_sbwp[i].gen == g_pend_gen) gw_put(g_sbwp[i].page << 12, (g_sbwp[i].page + 1) << 12, fence, done);
    for (int i = 0; i < g_nsbw_w; i++) gw_put(g_sbw_wlo[i], g_sbw_whi[i], fence, done);
}
// The remembered bytes of running submissions over a copy of memory (d holds [lo, hi); ready: the copy operations filling it, which must come first). Not left to the copy worker: the next commit changes these bytes
static void um_overlay(uint64_t lo, uint64_t hi, uint8_t *d, uint64_t ready) {
    if (!g_um) return;
    uint64_t done = g_done; int waited = 0;
    for (uint64_t at = lo, end; at < hi; at = end) { uint64_t pg = at >> 12; end = MIN((pg + 1) << 12, hi); const um_ent *m = &g_um[um_slot(pg)];
        if (m->page != pg || m->fence <= done) continue;
        for (uint64_t x = at; x < end; x++) if (m->f[x & 4095] > done) { if (!waited) { ubc_wait(ready); waited = 1; } d[x - lo] = m->val[x & 4095]; } }
}
// The writes queued in this submission from index since on that overlap [a, b), newest first, into out (room for every entry a lookup can meet); -1 when they cannot all be told
static int pend_since(uint64_t a, uint64_t b, int since, int *out) {
    if (g_pend_over) return -1;
    int n = 0; uint64_t p0 = a >> 12, p1 = (b - 1) >> 12;
    if (p1 - p0 >= PEND_SPAN) { for (int e = g_npend - 1; e >= since; e--) if (a < g_pend_hi[e] && g_pend_lo[e] < b) out[n++] = e; return n; }   // every entry, newest first
    for (int i = 0; i < g_nwide; i++) { int e = g_pend_wide[i]; if (e >= since && a < g_pend_hi[e] && g_pend_lo[e] < b) out[n++] = e; }
    for (uint64_t pg = p0; pg <= p1; pg++) { unsigned bk = pend_bucket(pg); if (g_pend_hgen[bk] != g_pend_gen) continue;
        for (int k = g_pend_head[bk]; k >= 0 && g_pend_node[k].e >= since; k = g_pend_node[k].next) { int e = g_pend_node[k].e; if (a < g_pend_hi[e] && g_pend_lo[e] < b) out[n++] = e; } }   // chains run newest first
    if (n > 1 && (g_nwide || p1 > p0)) {   // several lists: newest first, each once
        for (int i = 1; i < n; i++) { int v = out[i], j = i; while (j > 0 && out[j - 1] < v) { out[j] = out[j - 1]; j--; } out[j] = v; }
        int m = 1; for (int i = 1; i < n; i++) if (out[i] != out[m - 1]) out[m++] = out[i];
        n = m; }
    return n;
}
int nvn_host_readable(uint64_t a, uint64_t n);   // hle_nvn.c: guest memory and sparse pools
// Shadows made from memory, per range. One is kept while no write is queued after it (the draws after an update bind the same block again and again); the next one for the same range starts
// from it and lays over only the writes queued since (the game rewrites one block hundreds of times a submission). A range read whole from memory once is not read from memory again in the submission
enum { UBP_CACHE = 16, UBP_MAX_BYTES = 65536 };
static int g_ubp_idx[PEND_MAX + PEND_WIDE];   // every entry a lookup can meet: the chains of the range's pages hold each entry at most once after dedup, the wide list at most PEND_WIDE
typedef struct { uint64_t lo, hi, gpu; uint8_t *cpu; int npend; uint32_t gen, sbw; uint64_t ready; } ubp_ent;   // sbw: g_sbw_gen when the other writes were last checked; ready: copy ops that fill it (ubc_push)
static uint64_t g_ubp_ready;   // ready of the shadow ubp_get returned last (a CPU read of it waits for that: ubc_wait)
static ubp_ent g_ubpc[UBP_CACHE]; static unsigned g_ubpc_i;
// The bytes [lo, hi) as the GPU will read them at this point of the submission (see above): 1 with the shadow's GPU and CPU addresses, 0 when they cannot be known (the pass ends instead)
static int ubp_get(uint64_t lo, uint64_t hi, uint64_t *gpu, const uint8_t **cpu) {
    if (!ub_partial_on() || hi <= lo || hi - lo > UBP_MAX_BYTES) { g_ubp_why[0]++; return 0; }
    ubp_ent *c = NULL;
    for (int i = 0; i < UBP_CACHE; i++) if (g_ubpc[i].lo == lo && g_ubpc[i].hi == hi && g_ubpc[i].gen == g_pend_gen) { c = &g_ubpc[i]; break; }
    if (!c || c->sbw != g_sbw_gen) { if (g_sbw_over || sbw_hits(lo, hi)) { g_ubp_why[1]++; return 0; } if (c) c->sbw = g_sbw_gen;
        if (g_gml_on) { pthread_mutex_lock(&g_gsr_mu); uint64_t i = g_gsr_n++ % GSR_N; g_gsr[i].fence = g_fence + 1; g_gsr[i].lo = lo; g_gsr[i].hi = hi; pthread_mutex_unlock(&g_gsr_mu); } }
    if (c && c->npend == g_npend) { g_ubp_reused++; *gpu = c->gpu; *cpu = c->cpu; g_ubp_ready = c->ready; return 1; }
    int since = c ? c->npend : 0;
    if (!c && gw_live_in(g_gwx, lo, hi)) { g_ubp_why[2]++; return 0; }
    if (!c && !nvn_host_readable(lo, hi - lo)) { g_ubp_why[4]++; return 0; }
    int n = pend_since(lo, hi, since, g_ubp_idx); if (n < 0) { g_ubp_why[4]++; return 0; }
    // Newest first, keep the writes that still show through somewhere (a write only newer ones cover is never read, whatever wrote it); stop once they cover the range
    enum { COVN = 64 }; uint64_t cl[COVN], ch[COVN]; int nc = 0, nk = 0, exact = 1;   // covered intervals, sorted and disjoint; exact = 0 once they no longer fit
    for (int i = 0; i < n; i++) { int e = g_ubp_idx[i];
        uint64_t x = g_pend_lo[e] > lo ? g_pend_lo[e] : lo, y = g_pend_hi[e] < hi ? g_pend_hi[e] : hi;
        int j = 0; while (j < nc && ch[j] < y) j++;
        if (exact && j < nc && cl[j] <= x && y <= ch[j]) continue;   // newer writes cover all of it
        if (g_pend_kind[e] != 1 || !g_pend_upd[e]) { g_ubp_why[3]++; return 0; }   // a copy, fill or feedback write the CPU cannot see
        g_ubp_idx[nk++] = e;
        if (exact) {   // add [x, y) and merge neighbours
            int k = 0; while (k < nc && ch[k] < x) k++;
            int m = k; uint64_t nl = x, nh = y; while (m < nc && cl[m] <= y) { nl = MIN(nl, cl[m]); nh = MAX(nh, ch[m]); m++; }
            if (nc - (m - k) + 1 > COVN) exact = 0;
            else { memmove(&cl[k + 1], &cl[m], (nc - m) * sizeof *cl); memmove(&ch[k + 1], &ch[m], (nc - m) * sizeof *ch); nc = nc - (m - k) + 1; cl[k] = nl; ch[k] = nh;
                   if (nc == 1 && cl[0] == lo && ch[0] == hi) break; } } }
    uint64_t len = hi - lo;
    id<MTLBuffer> ab; size_t o = arena_alloc((size_t)len, &ab); uint8_t *d = (uint8_t *)ab.contents + o;
    int defer = len >= 512 && ubc_on(); uint64_t ready = 0;   // small ones (bindless handles read on the CPU) are copied here
    if (defer) ready = ubc_push(d, c ? c->cpu : (const uint8_t *)(uintptr_t)lo, (size_t)len);   // the previous shadow's own copies are queued before these
    else { if (c) ubc_wait(c->ready); memcpy(d, c ? c->cpu : (const uint8_t *)(uintptr_t)lo, (size_t)len); }
    if (!c) um_overlay(lo, hi, d, ready);   // a shadow made from the previous one has them already
    for (int i = nk - 1; i >= 0; i--) { int e = g_ubp_idx[i]; const upd_ent *u = &g_upd[g_pend_upd[e] - 1];   // oldest first: each byte ends with the newest write
        uint64_t x = g_pend_lo[e] > lo ? g_pend_lo[e] : lo, y = g_pend_hi[e] < hi ? g_pend_hi[e] : hi;
        const uint8_t *src = (const uint8_t *)u->buf.contents + u->off + (x - u->lo);
        if (defer) ready = ubc_push(d + (x - lo), src, (size_t)(y - x)); else memcpy(d + (x - lo), src, (size_t)(y - x)); }
    g_ubp_made++;
    if (!c) c = &g_ubpc[g_ubpc_i++ % UBP_CACHE];
    *c = (ubp_ent){lo, hi, ab.gpuAddress + o, d, g_npend, g_pend_gen, g_sbw_gen, ready};
    *gpu = c->gpu; *cpu = d; g_ubp_ready = ready; return 1;
}
void mtl_ubp_stats(uint64_t *made, uint64_t *reused, uint64_t *refused) { *made = g_ubp_made; *reused = g_ubp_reused; *refused = g_ubp_refused; }
void mtl_ubp_why(uint64_t w[5]) { memcpy(w, g_ubp_why, sizeof g_ubp_why); }
void mtl_sub_update(const void *data, uint64_t dst, uint64_t n) {
    if (!n) return;
    int pi = pend_add(dst, n, 1);
    void *db; uint64_t dof;
    if (!mtl_addr(dst, n, &db, &dof)) MDIE("uniform update target is in no memory pool dst=%#llx n=%#llx", (unsigned long long)dst, (unsigned long long)n);
    int defer = (enc_open() || (ubd_lazy() && !g_blit)) && g_nubd < UBD_MAX && ubd_on();
    size_t need = (n + 255u) & ~(size_t)255u;   // 256-aligned slots: a deferred update is read in place by draws as a uniform buffer
    if (!g_ubs_buf || g_ubs_off + need > g_ubs_size) {
        if (g_ubs_buf) defer_obj_release(g_ubs_buf);   // copies already encoded still reference the old block: release after this submission completes
        g_ubs_size = MAX((size_t)1 << 20, need);
        g_ubs_buf = [g_dev newBufferWithLength:g_ubs_size options:MTLResourceStorageModeShared];
        if (!g_ubs_buf) MDIE("uniform update ring allocation failed %#zx", g_ubs_size);
        res_add(g_ubs_buf); g_ubs_off = 0;
    }
    memcpy((uint8_t *)g_ubs_buf.contents + g_ubs_off, data, n);
    if (pi >= 0) { g_upd[g_nupd] = (upd_ent){dst, g_ubs_off, g_ubs_buf}; g_pend_upd[pi] = (uint16_t)++g_nupd; }
    if (defer) {
        // A deferred update this one covers whole never needs to land: the same landing writes these bytes after it, and the draws that read it hold its ring copy (ubd_shadow).
        // Games rewrite one block before every draw group (Breath of the Wild: the same 64 bytes), so the list stays short: every lookup below scans it, for most bindings of most draws
        int k = 0; for (int i = 0; i < g_nubd; i++) if (!(dst <= g_ubd[i].lo && g_ubd[i].hi <= dst + n)) g_ubd[k++] = g_ubd[i];
        g_nubd = k;
        g_ubd[g_nubd++] = (ubd_ent){dst, dst + n, g_ubs_buf.gpuAddress + g_ubs_off, g_ubs_off, g_ubs_buf}; ubd_pf_add(dst, dst + n); g_ubd_deferred++; ubd_dirty(dst, dst + n); }   // see ubd_shadow
    else enc_b_copy_buffer(blit_buf(0, 0, dst, dst + n), g_ubs_buf, g_ubs_off, (__bridge id<MTLBuffer>)db, dof, n);
    g_ubs_off += need;
}

// In-order buffer fill within the submission: all four bytes equal â†’ fillBuffer; otherwise copied in chunks from a 64 KiB pattern buffer
void mtl_sub_fill(uint64_t dst, uint64_t n, uint32_t value) {
    if (!n) return;
    pend_add(dst, n, 3); g_mdx_epoch++;
    void *db; uint64_t dof;
    if (!mtl_addr(dst, n, &db, &dof)) MDIE("buffer fill address is in no memory pool dst=%#llx n=%#llx", (unsigned long long)dst, (unsigned long long)n);
    uint8_t b0 = (uint8_t)value;
    if (value == (uint32_t)b0 * 0x01010101u) { [blit_buf(0, 0, dst, dst + n) fillBuffer:(__bridge id<MTLBuffer>)db range:NSMakeRange(dof, n) value:b0]; return; }
    enum { PAT = 0x10000 }; static __strong id<MTLBuffer> pat; static uint32_t pat_v; static int pat_ok;
    if (!pat_ok || pat_v != value) { uint32_t *tmp = malloc(PAT); for (unsigned i = 0; i < PAT / 4; i++) tmp[i] = value;
        pat = [g_dev newBufferWithBytes:tmp length:PAT options:MTLResourceStorageModeShared]; free(tmp); pat_v = value; pat_ok = 1; res_add(pat); }
    for (uint64_t off = 0; off < n; off += PAT) { uint64_t k = n - off < PAT ? n - off : PAT; enc_b_copy_buffer(blit_buf(0, 0, dst + off, dst + off + k), pat, 0, (__bridge id<MTLBuffer>)db, dof + off, k); }
}

static void post(post_op o) {
    if (g_npost == g_postcap) { g_postcap = g_postcap ? g_postcap * 2 : 64; g_post = realloc(g_post, g_postcap * sizeof *g_post); }
    g_post[g_npost++] = o;
}
void mtl_sub_event(volatile uint32_t *val, uint32_t mode, uint32_t v) { post((post_op){1, val, mode, v, 0, 0, NULL}); }
void mtl_sub_counter(uint64_t addr, uint32_t type) { sbw_note(addr, 16, SBW_REPORT); post((post_op){2, NULL, 0, 0, addr, type, NULL, g_seg}); }
void mtl_counter_stats(uint64_t rep[32], uint64_t rst[32]) { memcpy(rep, g_cnt_report, sizeof g_cnt_report); memcpy(rst, g_cnt_reset, sizeof g_cnt_reset); }

// Counter timestamps (614.4 MHz) are taken from the guest clock (hle_os.c guest_ns, the same epoch as svc GetSystemTick / cntpct, whose ticks are 19.2 MHz, 31.25 MHz on the Switch 2: guest_ticks).
// They used to come from CLOCK_MONOTONIC, which keeps running while the Mac sleeps while the guest clock (mach_absolute_time) does not, so the two differed by the sleep time; once timestamps were converted to ns, Mario halved its frame rate. In graphics unit tests guest_ns is a weak definition returning 0; fall back to the monotonic clock
uint64_t guest_ns(void);
static uint64_t gpu_ticks(void) { uint64_t g = guest_ns(); if (!g) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); g = (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec; } return nvn_timestamp_from_ns(g); }
// Counter timestamps by segment (profile gpu_timestamps segment; DM_NVN_GPU_TS=segment / submission overrides it). On hardware a timestamp report is written when the GPU reaches it, so two reports of one submission bracket the GPU work between them.
// The host writes all reports of a submission in its completion handler; with one time for all of them a game that measures its GPU time that way reads 0 (Breath of the Wild: frame-interval controller and dynamic resolution).
// In segment mode a report takes the GPU time of the segment it was encoded in (sub_cut): TIMESTAMP_TOP (0x10) that segment's GPU start, other types its GPU end. The writes stay in the last segment's handler, before the fence,
// so a game that waits for the fence reads one submission's reports together; a segment whose feedback has not arrived by then takes the last segment's times
#define SEG_CLK_N 1024
struct seg_clk { _Atomic int refs; _Atomic uint64_t t[SEG_CLK_N][2]; };   // per submission (g_sclk, allocated at its first cut): guest ns of each cut segment's GPU start / end (0 = no feedback yet)
static int seg_ts(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_NVN_GPU_TS"); v = e && *e ? !strcmp(e, "segment") : DM_GAME_GPU_TS_SEGMENT; MLOG("counter timestamps: %s (profile key gpu_timestamps; DM_NVN_GPU_TS=segment / submission)", v ? "GPU start / end time of their segment" : "one time for the whole submission"); } return v; }
static void seg_clk_put(seg_clk *c) { if (c && atomic_fetch_sub(&c->refs, 1) == 1) free(c); }
static uint64_t gpu_s_guest_ns(double s) {   // a feedback GPU time (seconds on the mach absolute time base) on the guest clock; 0 when unknown
    static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb);
    uint64_t g = guest_ns(), now = tb.denom ? mach_absolute_time() * tb.numer / tb.denom : 0, at = (uint64_t)(s * 1e9);
    if (s <= 0 || !g || at > now || now - at > g) return 0;
    return g - (now - at);
}
static uint64_t rep_ticks(const post_op *o, seg_clk *sc, uint32_t last, uint64_t c0, uint64_t c1) {   // the time a counter report records (614.4 MHz); c0 / c1: the last segment's GPU start / end on the guest clock
    int top = o->type == 0x10; uint64_t ns = 0;
    if (sc && o->seg < last && o->seg < SEG_CLK_N) ns = atomic_load(&sc->t[o->seg][!top]);
    if (!ns) ns = top ? c0 : c1;
    return ns ? nvn_timestamp_from_ns(ns) : gpu_ticks();
}

// ---------- Boot instrumentation (DM_BOOTPROF=1): metallib loading, pipeline creation (both happen synchronously during replay on the submit thread), texture upload, first-frame present ----------
static uint64_t bp_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec; }
static uint64_t g_bp_t0;   // Time of process start (when this module is loaded)
__attribute__((constructor)) static void bp_t0(void) { g_bp_t0 = bp_now(); }
static int bootprof(void) { static int v = -1; if (v < 0) v = getenv("DM_BOOTPROF") != NULL; return v; }
static uint64_t g_bp_lib_n, g_bp_lib_ns, g_bp_pso_n, g_bp_pso_ns, g_bp_up_bytes, g_bp_first_present;
static uint64_t g_up_total_ns;   // Cumulative upload time (same source as the per-frame @upload_ms)
// ---------- Per-frame GPU time aggregation ----------
#define GRING 1024
static struct { _Atomic uint64_t frame, ns, t0, t1; _Atomic int pending; } g_gpu[GRING];   // t0 / t1: the frame's earliest GPUStart / latest GPUEnd (mach absolute time, ns)
static _Atomic uint64_t g_cur_frame;
// ---------- Same-process alternating measurement (DM_NVN_UB, off by default: without the environment variable the code below only adds one 'is it enabled' check) ----------
// On a hot real device the system keeps adjusting power / frequency, and segment comparison (segments A and B each run for a minute) cannot resolve differences of <10%; within the same run, toggle by phase in turn and take differences of adjacent phase pairs, so the drift cancels.
// DM_NVN_UB="name=item,item;name=item", one phase every DM_NVN_UB_N frames (default 180, minimum 60): even phases = baseline (nothing changed), odd phases take one category in turn. Each item of a category:
//   @set:<global variable name>=<value>   on entering the phase set this int global variable to the value, on leaving restore it (address found by symbol name; switches to be used on a real device must have a line added to the registration table below)
//   <fragment shader key prefix> or *   the phase drops all draws of these fragment shaders (the picture will be wrong; it only measures 'how much this kind of draw is worth'; a nearly idle category needs a round of its own)
// Readings: each phase discards its first 45 frames (transition + present lag); for the remaining frames, the mean of the actual present interval (same source as perf.txt) and the frame counts in the three buckets 16.7 / 25 / â‰¥33 ms, one log line "UB present".
static int g_ub_n = -1, g_ub_N = 180; static _Atomic int g_ub_cur = -1; static char g_ub_name[16][16], g_ub_keys[16][16][48]; static int g_ub_nk[16];
static _Atomic int g_ubp_on, g_ubp_c[4]; static _Atomic uint64_t g_ubp_sum_us;
void mtl_ub_present(double iv) {   // Called once per frame by the present thread: actual present interval in ms (âˆ’1 = dropped, 0 = no reading)
    if (!atomic_load(&g_ubp_on) || iv == 0) return;
    if (iv < 0) { atomic_fetch_add(&g_ubp_c[3], 1); return; }
    atomic_fetch_add(&g_ubp_c[iv <= 20.8 ? 0 : iv <= 29.2 ? 1 : 2], 1); atomic_fetch_add(&g_ubp_sum_us, (uint64_t)(iv * 1000));
}
// Registered runtime switches: name â†’ address. Main-program symbols of an iOS release build are not in the dynamic symbol table and dlsym cannot find them (in the first real-device round all four categories were "global variable not found" and the whole round was wasted),
// so a switch to be measured on a real device must have a line added here; on Mac / in unit tests unregistered ones are still looked up by symbol name.
static int *ub_var(const char *nm) {
    extern int g_dm_present_link;
    static const struct { const char *n; int *p; } t[] = { {"g_dm_present_link", &g_dm_present_link} };
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) if (!strcmp(nm, t[i].n)) return t[i].p;
    return dlsym(RTLD_DEFAULT, nm);
}
static void ub_frame(uint64_t f) {
    if (g_ub_n < 0) { g_ub_n = 0; const char *e = getenv("DM_NVN_UB"), *n = getenv("DM_NVN_UB_N"); if (n && atoi(n) >= 60) g_ub_N = atoi(n);
        for (const char *q = e; q && *q && g_ub_n < 16; ) { const char *eq = strchr(q, '='), *end = strchr(q, ';'); if (!end) end = q + strlen(q); if (!eq || eq > end) break;
            int c = g_ub_n++; size_t l = (size_t)(eq - q); if (l > 15) l = 15; memcpy(g_ub_name[c], q, l);
            for (const char *k = eq + 1; k < end && g_ub_nk[c] < 16; ) { size_t kl = 0; while (k + kl < end && k[kl] != ',') kl++; size_t cl = kl > 47 ? 47 : kl; memcpy(g_ub_keys[c][g_ub_nk[c]++], k, cl); k += kl; if (k < end) k++; }
            q = *end ? end + 1 : end; }
        if (g_ub_n) MLOG("DM_NVN_UB: %d categories, %d frames per phase (measured alternately in one process; phases that drop draws render wrong)", g_ub_n, g_ub_N); }
    if (!g_ub_n) return;
    uint64_t ph = f / (uint64_t)g_ub_N, r = f % (uint64_t)g_ub_N;
    if (r == 45) { for (int i = 0; i < 4; i++) atomic_store(&g_ubp_c[i], 0); atomic_store(&g_ubp_sum_us, 0); atomic_store(&g_ubp_on, 1); }
    if (r != 0) return;
    if (f && atomic_load(&g_ubp_on)) { atomic_store(&g_ubp_on, 0); uint64_t pp = ph - 1; int pc = (pp & 1) ? (int)((pp / 2) % (uint64_t)g_ub_n) : -1;
        int c0 = atomic_load(&g_ubp_c[0]), c1 = atomic_load(&g_ubp_c[1]), c2 = atomic_load(&g_ubp_c[2]), c3 = atomic_load(&g_ubp_c[3]), n_ = c0 + c1 + c2;
        MLOG("UB present %llu %s frames %d mean %.3f ms 2 ticks %d 3 ticks %d 4+ ticks %d discarded %d", (unsigned long long)pp, pc < 0 ? "base" : g_ub_name[pc], n_, n_ ? atomic_load(&g_ubp_sum_us) / 1000.0 / n_ : 0.0, c0, c1, c2, c3); }
    int nc = (ph & 1) ? (int)((ph / 2) % (uint64_t)g_ub_n) : -1;
    static int *sp[16]; static int sv[16], ns;   // Variables set by the previous phase: restore first
    for (int i = 0; i < ns; i++) *sp[i] = sv[i]; ns = 0;
    for (int i = 0; nc >= 0 && i < g_ub_nk[nc] && ns < 16; i++) { const char *k = g_ub_keys[nc][i]; if (strncmp(k, "@set:", 5)) continue;
        char nm[48]; const char *eq = strchr(k, '='); if (!eq) continue; size_t l = (size_t)(eq - (k + 5)); if (l >= sizeof nm) continue; memcpy(nm, k + 5, l); nm[l] = 0;
        int *a_ = ub_var(nm); static int bad, ok;
        if (!a_) { if (bad++ < 8) MLOG("[RED] DM_NVN_UB: global variable not found: %s (on a device, register it in the ub_var table first)", nm); continue; }
        sp[ns] = a_; sv[ns] = *a_; ns++; *a_ = atoi(eq + 1); if (ok++ < 8) MLOG("DM_NVN_UB: %s = %d (was %d)", nm, atoi(eq + 1), sv[ns - 1]); }
    atomic_store(&g_ub_cur, nc);
}
static int ub_skip(const char *key) {   // Whether this phase should drop draws of this fragment shader
    int uc = atomic_load(&g_ub_cur); if (uc < 0) return 0;
    for (int i = 0; i < g_ub_nk[uc]; i++) { const char *k = g_ub_keys[uc][i]; if (k[0] == '@') continue; if (k[0] == '*' || !strncmp(key, k, strlen(k))) return 1; }
    return 0;
}
static long ab_len(void); static void ab_frame(uint64_t f);
// DM_GPU_CAPTURE=<frame>[:<frames>] (diagnostic): a Metal GPU trace of those frames (default 1) for Xcode's Metal debugger (per-draw cost, shader profiler), or DM_GPU_CAPTURE=gpu:<ms>[:<after frame>]
// of one frame in the first scene, after that frame, whose GPU time passes <ms> in 60 frames in a row (a single long frame in a light scene fired the first iPhone capture). Read again
// when it changes, so the Mac's DM_CTL "env DM_GPU_CAPTURE=..." can set it while the game runs. iOS needs a build with MetalCaptureEnabled (DM_METAL_CAPTURE=1 when the project is
// made), the Mac MTL_CAPTURE_ENABLED=1 in the environment. It captures the Metal 4 queue (Metal refuses to capture a Metal 4 device), so the old-queue helpers (present, some uploads)
// are not in it. The trace is dm_<date>-<time>_f<frame>.gputrace in DM_GPU_CAPTURE_DIR, by default gpu_captures/ in the working directory (the app's Documents on iOS, the run folder
// on the Mac). A trace takes 1.5-2 GB, so before each capture that folder alone is pruned: of its dm_*.gputrace entries the newest DM_GPU_CAPTURE_KEEP - 1 stay (default 2 with the new
// one); nothing outside it is removed
static long g_cap_f0 = -2, g_cap_n; static int g_cap_on;
static NSString *gpu_capture_dir(void) {
    const char *e = getenv("DM_GPU_CAPTURE_DIR"); NSString *dir;
    if (e && *e) dir = @(e); else { char cwd[1024]; if (!getcwd(cwd, sizeof cwd)) strcpy(cwd, "."); dir = [@(cwd) stringByAppendingPathComponent:@"gpu_captures"]; }
    NSError *err = nil;
    if (![NSFileManager.defaultManager createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:&err]) { MLOG("[RED] DM_GPU_CAPTURE: cannot make the folder %s: %s", dir.UTF8String, err.localizedDescription.UTF8String); return nil; }
    return dir;
}
static void gpu_capture_prune(NSString *dir) {   // only dm_*.gputrace entries directly in the capture folder; the newest DM_GPU_CAPTURE_KEEP - 1 stay
    const char *e = getenv("DM_GPU_CAPTURE_KEEP"); long keep = e ? atol(e) : 2; if (keep < 1) keep = 1;
    NSFileManager *fm = NSFileManager.defaultManager; NSMutableArray<NSURL *> *traces = [NSMutableArray array];
    for (NSURL *u in [fm contentsOfDirectoryAtURL:[NSURL fileURLWithPath:dir isDirectory:YES] includingPropertiesForKeys:@[NSURLContentModificationDateKey] options:0 error:nil]) {
        NSString *n = u.lastPathComponent; if ([n hasPrefix:@"dm_"] && [n hasSuffix:@".gputrace"]) [traces addObject:u]; }
    [traces sortUsingComparator:^NSComparisonResult(NSURL *a, NSURL *b) { NSDate *da = nil, *db = nil;
        [a getResourceValue:&da forKey:NSURLContentModificationDateKey error:nil]; [b getResourceValue:&db forKey:NSURLContentModificationDateKey error:nil];
        return [db ?: NSDate.distantPast compare:da ?: NSDate.distantPast]; }];   // newest first
    for (NSUInteger i = (NSUInteger)keep - 1; i < traces.count; i++) if ([fm removeItemAtURL:traces[i] error:nil]) MLOG("DM_GPU_CAPTURE: removed %s (older captures beyond DM_GPU_CAPTURE_KEEP=%ld)", traces[i].lastPathComponent.UTF8String, keep);
}
static void gpu_capture_frame(uint64_t f) {
    static char last[32];   // read again when it changes
    { const char *e = getenv("DM_GPU_CAPTURE"); if (!g_cap_on && e && strncmp(e, last, sizeof last - 1)) { snprintf(last, sizeof last, "%s", e); g_cap_f0 = -2; } }
    static double gpu_thr; static long after; static int over;
    if (g_cap_f0 == -2) { const char *e = getenv("DM_GPU_CAPTURE"); gpu_thr = 0; over = 0;
        if (e && !strncmp(e, "gpu:", 4)) { gpu_thr = atof(e + 4); const char *c = strchr(e + 4, ':'); after = c ? atol(c + 1) : 0; g_cap_n = 1; g_cap_f0 = gpu_thr > 0 ? LONG_MAX : -1;   // armed: the frame is not known yet
            if (gpu_thr > 0) MLOG("DM_GPU_CAPTURE: the first frame after frame %ld that follows 60 frames in a row with more than %.1f ms of GPU time is captured", after, gpu_thr); }
        else { g_cap_f0 = e ? atol(e) : -1; g_cap_n = e && strchr(e, ':') ? atol(strchr(e, ':') + 1) : 1; if (g_cap_n < 1) g_cap_n = 1; if (g_cap_f0 <= 0) g_cap_f0 = -1; } }
    if (g_cap_f0 < 0) return;
    if (gpu_thr > 0 && !g_cap_on && f > 8 && (long)f >= after) { double g = mtl_gpu_ms(f - 8); over = g >= gpu_thr ? over + 1 : g >= 0 ? 0 : over; if (over >= 60) g_cap_f0 = (long)f; }
    MTLCaptureManager *m = [MTLCaptureManager sharedCaptureManager];
    if (!g_cap_on && f == (uint64_t)g_cap_f0) {
        g_cap_f0 = -1;   // one capture per setting: a failure below does not retry every frame
        if (![m supportsDestination:MTLCaptureDestinationGPUTraceDocument]) { MLOG("[RED] DM_GPU_CAPTURE: this build cannot write GPU traces (iOS: MetalCaptureEnabled, a build made with DM_METAL_CAPTURE=1; Mac: MTL_CAPTURE_ENABLED=1)"); return; }
        NSString *dir = gpu_capture_dir(); if (!dir) return;
        gpu_capture_prune(dir);
        time_t now = time(NULL); struct tm tm; localtime_r(&now, &tm); char ts[32]; strftime(ts, sizeof ts, "%Y%m%d-%H%M%S", &tm);
        NSString *path = [dir stringByAppendingPathComponent:[NSString stringWithFormat:@"dm_%s_f%llu.gputrace", ts, (unsigned long long)f]];
        MTLCaptureDescriptor *d = [MTLCaptureDescriptor new]; d.captureObject = g_q4 ? (id)g_q4 : (id)g_dev; d.destination = MTLCaptureDestinationGPUTraceDocument; d.outputURL = [NSURL fileURLWithPath:path];
        NSError *err = nil;
        if ([m startCaptureWithDescriptor:d error:&err]) { g_cap_on = 1; g_cap_f0 = (long)f; MLOG("DM_GPU_CAPTURE: capturing frames %llu to %llu into %s", (unsigned long long)f, (unsigned long long)(f + g_cap_n - 1), path.UTF8String); }
        else MLOG("[RED] DM_GPU_CAPTURE: the capture did not start: %s", err.localizedDescription.UTF8String);
    } else if (g_cap_on && f >= (uint64_t)g_cap_f0 + (uint64_t)g_cap_n) { [m stopCapture]; g_cap_on = 0; g_cap_f0 = -1; gpu_thr = 0; MLOG("DM_GPU_CAPTURE: capture stopped at frame %llu (the trace is written in the background)", (unsigned long long)f); }
}
void mtl_set_frame(uint64_t f) { atomic_store(&g_cur_frame, f); if (g_ub_n) ub_frame(f); if (ab_len()) ab_frame(f); gpu_capture_frame(f); }
// GPU timing registration: gpu_tag_begin(frame) before commit, gpu_tag_done(frame, GPU start, GPU end) on completion (seconds on the CACurrentMediaTime base).
// Old-queue command buffers go through gpu_tag_f; the Metal 4 main submission backfills in the feedback of mtl_sub_commit (when switching to Metal 4 all the call sites were lost and the GPU column of the timing line became empty)
static void gpu_tag_begin(uint64_t f) {
    unsigned k = f % GRING;
    if (atomic_load(&g_gpu[k].frame) != f) { atomic_store(&g_gpu[k].frame, f); atomic_store(&g_gpu[k].ns, 0); atomic_store(&g_gpu[k].pending, 0); atomic_store(&g_gpu[k].t0, UINT64_MAX); atomic_store(&g_gpu[k].t1, 0); }
    atomic_fetch_add(&g_gpu[k].pending, 1);
}
static void gpu_tag_done(uint64_t f, double g0, double g1) {
    unsigned k = f % GRING;
    if (atomic_load(&g_gpu[k].frame) != f) return;
    double d = g1 - g0;
    if (d > 0 && g0 > 0) {   // a command buffer can complete with no start time (0, while a movie's frames upload each frame in Big Brain Academy: Brain vs. Brain), and its span would be the whole clock
        atomic_fetch_add(&g_gpu[k].ns, (uint64_t)(d * 1e9));
        uint64_t s0 = (uint64_t)(g0 * 1e9), s1 = (uint64_t)(g1 * 1e9), v;
        for (v = atomic_load(&g_gpu[k].t0); s0 < v && !atomic_compare_exchange_weak(&g_gpu[k].t0, &v, s0);) {}
        for (v = atomic_load(&g_gpu[k].t1); s1 > v && !atomic_compare_exchange_weak(&g_gpu[k].t1, &v, s1);) {}
    }
    atomic_fetch_sub(&g_gpu[k].pending, 1);
}
static void gpu_tag_f(id<MTLCommandBuffer> cb, uint64_t f) { gpu_tag_begin(f); [cb addCompletedHandler:^(id<MTLCommandBuffer> c) { gpu_tag_done(f, c.GPUStartTime, c.GPUEndTime); }]; }
void gpu_tag(id<MTLCommandBuffer> cb) { gpu_tag_f(cb, atomic_load(&g_cur_frame)); }
double mtl_gpu_ms(uint64_t f) {
    unsigned k = f % GRING;
    if (atomic_load(&g_gpu[k].frame) != f || atomic_load(&g_gpu[k].pending)) return -1;
    return atomic_load(&g_gpu[k].ns) / 1e6;
}
int mtl_gpu_span(uint64_t f, uint64_t *t0, uint64_t *t1) {
    unsigned k = f % GRING;
    if (atomic_load(&g_gpu[k].frame) != f || atomic_load(&g_gpu[k].pending)) return 0;
    *t0 = atomic_load(&g_gpu[k].t0); *t1 = atomic_load(&g_gpu[k].t1);
    return *t1 >= *t0 && *t1;
}
static void rp_flush(void);
static const char *cctx_snap(void);   // What is in this submission (printed into the failure line on error; defined before mtl_exec)
static void lc_land(void);
// A submission's CPU-side writes at its end (events, samples-passed and counter reports): run once, by its feedback handler or, when a game's SyncWait wakes on the
// GPU's own signal of the fence first, by that waiter (DM_NVN_FAST_WAKE, on by default): the feedback handler comes 0.26 ms after the GPU end on an M2 Max
typedef struct { _Atomic uint64_t fence; _Atomic int claimed; int n; post_op *ops; seg_clk *sc; uint32_t seg_last; uint64_t t_commit; } post_rec;   // t_commit: guest ns when it was committed
#define POST_RING 128
static post_rec g_postr[POST_RING];
static pthread_mutex_t g_post_mu = PTHREAD_MUTEX_INITIALIZER;
// A successful claim holds the lock until post_run finishes. Ownership alone
// cannot publish completion: another waiter must see the actual writes first.
static int post_claim(post_rec *r, uint64_t fence) {
    pthread_mutex_lock(&g_post_mu);
    if (r->fence != fence || r->claimed) { pthread_mutex_unlock(&g_post_mu); return 0; }
    return 1;
}
static void post_run(post_rec *r, uint64_t c0, uint64_t c1) {
    // A guest SyncWait may run completion on a thread whose outer pool never drains.
    // MTLBuffer.contents may autorelease its receiver; keep that lifetime inside the completed copy.
    @autoreleasepool {
    int n = r->n; post_op *ops = r->ops; seg_clk *sc = r->sc; uint32_t seg_last = r->seg_last;
    for (int i = 0; i < n; i++) {
        if (ops[i].kind == 1) { if (ops[i].mode == 1) *ops[i].val += ops[i].v; else *ops[i].val = ops[i].v; }
        else if (ops[i].kind == 3) {   // Samples passed: the sum of all slots since the reset
            vis_list *l = ops[i].vis; uint64_t sum = 0;
            for (int k = 0; k < l->n; k++) { sum += ((const uint64_t *)((__bridge id<MTLBuffer>)l->s[k].buf).contents)[l->s[k].slot]; CFRelease(l->s[k].buf); }
            free(l); uint64_t rec[2] = {sum, gpu_ticks()}; memcpy((void *)ops[i].addr, rec, sizeof rec);
        }
        else if (ops[i].kind == 4) mirror_rb_land(ops[i].vis);
        else { uint64_t t = rep_ticks(&ops[i], sc, seg_last, c0, c1), rec[2] = {(ops[i].type == 0 || ops[i].type == 0x10) && !DM_GAME_RENDER_COND ? t : 0, t}; memcpy((void *)ops[i].addr, rec, sizeof rec); }   // render_cond: a timestamp report's counter field is 0, as on the GPU (see rc_value)
    }
    free(ops); seg_clk_put(sc); r->ops = NULL; r->sc = NULL;
    atomic_store(&r->claimed, 1);
    pthread_mutex_unlock(&g_post_mu);
    }
}
// Both GPU feedback and an early waiter complete records in submission order.
// Delayed callbacks carry their fence, never the identity of a recyclable slot.
static void post_complete(uint64_t fence, uint64_t c0, uint64_t c1) {
    uint64_t now = guest_ns();
    for (uint64_t f = g_done + 1; f <= fence; f++) {
        post_rec *r = &g_postr[f % POST_RING];
        if (!post_claim(r, f)) continue;
        uint64_t start = r->t_commit;
        if (r->sc && r->seg_last > 0 && r->seg_last <= SEG_CLK_N) {
            uint64_t pe = atomic_load(&r->sc->t[r->seg_last - 1][1]); if (pe > start) start = pe;
        }
        if (start > now) start = now;
        post_run(r, f == fence && c1 ? c0 : seg_ts() ? start : 0, f == fence && c1 ? c1 : seg_ts() ? now : 0);
    }
    pthread_mutex_lock(&g_mu); if (fence > g_done) g_done = fence; pthread_cond_broadcast(&g_cv); pthread_mutex_unlock(&g_mu);
}
uint64_t mtl_sub_commit(void) {
    lc_land();   // clears still waiting (the command list ends with them)
    defer_fallback();   // Deferred merged pass: at the end of the submission there are still P draws not replayed (the normal path never gets here)
    rp_flush(); ubd_land();
    if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    up_flush();
    g_cut_copy = 0;
    treg_finish_submission();   // Late descriptor views retire behind this submission's fence.
    uint64_t fence = ++g_fence;
    gw_commit(fence);
    if (g_vis_ring_i >= 0) g_vis_ring[g_vis_ring_i].fence = fence;
    // CPU-visible effects after completion: copy the op table into the callback
    int n = g_npost; post_op *ops = n ? malloc(n * sizeof *ops) : NULL; post_rec *pr = &g_postr[fence % POST_RING];
    if (ws_on()) for (int i = 0; i < n; i++) { int k = g_post[i].kind == 2 ? (g_post[i].type == 0 || g_post[i].type == 0x10 ? 3 : 4) : g_post[i].kind == 3 ? 2 : 1; atomic_fetch_add(&g_ws_post[k], 1); }   // 1 event, 2 samples passed, 3 timestamp, 4 other counter
    if (n) memcpy(ops, g_post, n * sizeof *ops);
    NSArray<dispatch_block_t> *commit_blocks = [g_commit_blocks copy];
    MTL4CommitOptions *o = [MTL4CommitOptions new];
    [o addFeedbackHandler:^(id<MTL4CommitFeedback> f) { (void)f; for (dispatch_block_t block in commit_blocks) block(); }];
    uint64_t gpu_f = atomic_load(&g_cur_frame); gpu_tag_begin(gpu_f);
    NSString *cctx = @(cctx_snap());
    seg_clk *sc = g_sclk; uint32_t seg_last = g_seg; g_sclk = NULL;
    pthread_mutex_lock(&g_post_mu);
    if (!atomic_load(&pr->claimed) && pr->fence) MDIE("post ring: the writes of fence %llu never ran", (unsigned long long)pr->fence);
    pr->n = n; pr->ops = ops; pr->sc = sc; pr->seg_last = seg_last; pr->t_commit = guest_ns(); atomic_store(&pr->claimed, 0); atomic_store(&pr->fence, fence);
    pthread_mutex_unlock(&g_post_mu);
    [o addFeedbackHandler:^(id<MTL4CommitFeedback> cb) {
        gpu_tag_done(gpu_f, cb.GPUStartTime, cb.GPUEndTime);
        if (ws_on()) ws_fb(fence, cb.GPUEndTime);
        uint64_t c0 = 0, c1 = 0; if (seg_ts()) { c0 = gpu_s_guest_ns(cb.GPUStartTime); c1 = gpu_s_guest_ns(cb.GPUEndTime); }
        if (cb.error) { atomic_store(&g_last_commit_error, 1); static int n, nv; int victim = [cb.error.description containsString:@"InnocentVictim"]; if (victim ? nv++ < 1 : n++ < 8) MLOG("[RED] command buffer error details fence=%llu: %s | GPU %.3fâ†’%.3f", (unsigned long long)fence, cb.error.description.UTF8String, cb.GPUStartTime, cb.GPUEndTime); if (!victim) MLOG("[RED] state of the failed submission fence=%llu: %s", (unsigned long long)fence, cctx.UTF8String); MLOG("[RED] command buffer execution error: %s", cb.error.localizedDescription.UTF8String); }
        post_complete(fence, c0, c1);
        defer_drain();
    }];
    void arena_commit(uint64_t fence); arena_commit(fence);
    res_commit();
    if (g_gp_state == 1 && !gp_active()) { int gn = g_gp_n; gp_lab *gl = malloc(sizeof(gp_lab) * (gn ? gn : 1)); memcpy(gl, g_gp_lab, sizeof(gp_lab) * gn); uint64_t gf = g_gp_frame; g_gp_state = 2; atomic_store(&g_gp_busy, 1);
        [o addFeedbackHandler:^(id<MTL4CommitFeedback> f) { (void)f; gp_report(gn, gl, gf); }]; }
    [g_cb endCommandBuffer];
    rec_commit(g_cb, o);
    rec_signal(g_ev, fence);
    rec_drain();
    g_alloc_fence[g_alloc_cur] = fence;
    g_cb = nil;
    g_commit_blocks = nil;
    objc_autoreleasePoolPop(g_sub_pool); g_sub_pool = NULL;
    return fence;
}

uint64_t mtl_last_fence(void) { return g_fence; }
int mtl_last_commit_error(void) { return atomic_load(&g_last_commit_error); }

static int fast_wake(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_NVN_FAST_WAKE"); v = !(e && !strcmp(e, "0")); MLOG("game SyncWaits wake on the GPU's signal of the fence%s (DM_NVN_FAST_WAKE)", v ? "" : ": off, on the feedback handler"); } return v; }
// A SyncWait woken by the GPU's signal: the CPU-side writes of every submission up to the fence that no feedback handler has run yet, run here. The fence's last segment ended
// now (c1); it started at the later of its commit and the previous segment's end (c0), as near as the CPU can tell without its feedback
static void post_early(uint64_t fence) {
    post_complete(fence, 0, 0);
}
static int mtl_wait_at(uint64_t fence, uint64_t timeout_ns, int site) {
    if (g_done >= fence) { if (ws_on()) ws_put(site, 0, 0); return 1; }
    uint64_t w0 = ws_t0();
    if ((site == WS_SYNC || site == WS_SYNC_NEWEST || site == WS_ACQ) && fast_wake() && g_ev && fence <= g_fence) {   // the game's own wait: wake on the GPU's signal (rec_signal(g_ev, fence) after each submission)
        uint64_t ms = timeout_ns / 1000000ull; if (ms > 3600000ull) ms = 3600000ull;
        if ([g_ev waitUntilSignaledValue:fence timeoutMS:ms]) post_early(fence);
        int ok = g_done >= fence;
        if (ws_on()) { ws_put(site, w0, 1); ws_wake(fence); }
        if (ok || timeout_ns < 1000000ull) return ok;
        // The event and feedback waits share a finite budget; a timeout must not buy another full wait.
        if (timeout_ns != UINT64_MAX) {
            uint64_t used = ws_t0() - w0;
            if (used >= timeout_ns) return 0;
            timeout_ns -= used;
        }
    }
    struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl);
    uint64_t ns = (uint64_t)dl.tv_nsec + (timeout_ns > 3600000000000ull ? 3600000000000ull : timeout_ns);
    dl.tv_sec += ns / 1000000000ull; dl.tv_nsec = ns % 1000000000ull;
    pthread_mutex_lock(&g_mu);
    int r = 0;
    while (g_done < fence) if ((r = pthread_cond_timedwait(&g_cv, &g_mu, &dl))) break;
    int ok = g_done >= fence;
    pthread_mutex_unlock(&g_mu);
    if (ws_on()) { ws_put(site, w0, 1); if (site == WS_SYNC_NEWEST || site == WS_SYNC || site == WS_ACQ) ws_wake(fence); }
    return ok;
}
// The game's SyncWait (hle_nvn.c), with what it waits for in the wait ledger
int mtl_wait(uint64_t fence, uint64_t timeout_ns) { return mtl_wait_at(fence, timeout_ns, !timeout_ns ? WS_SYNC_POLL : fence >= g_fence ? WS_SYNC_NEWEST : WS_SYNC); }
int mtl_wait_acq(uint64_t fence, uint64_t timeout_ns) { return mtl_wait_at(fence, timeout_ns, !timeout_ns ? WS_SYNC_POLL : WS_ACQ); }

static void mtl_finish_gpu_at(int site) {
    if (g_fence) mtl_wait_at(g_fence, UINT64_MAX, site);
    if (on_sub_thread()) rec_publish();   // a batch's commit may still be a record in the encode thread's current block (rec_commit_upload), as in up_begin and q_cb
    up_wait_zero(&g_up_pending, site);
}
#define mtl_finish_gpu() mtl_finish_gpu_at(WS_DIAG)
void mtl_present_flush(void);
// nvnQueueFinish waits for the GPU work submitted so far, not for queued presents to reach the screen: with a window, the display link takes one present per tick, so waiting for them
// made every QueueFinish cost one or two display frames. Fire Emblem: Three Houses' loader submits a copy and calls QueueFinish for each resource it uploads, and its first loading screen
// took 78 s on the iPhone (5 s windowless on the Mac). A swapchain texture is not handed back before its previous present is processed (AcquireTexture waits for that ticket)
void mtl_finish(void) { mtl_finish_gpu_at(WS_FINISH); }
static void present_drop_for_suspend(void);
static void snap_drain(void);
void mtl_drain(void) { present_drop_for_suspend(); mtl_present_flush(); snap_drain(); mtl_init(); id<MTLCommandBuffer> cb = q_cb(); [cb commit]; [cb waitUntilCompleted]; mtl_finish_gpu(); }

// ================= Render replay =================
#include "../../host/front.h"
static int g_red_once[162];
#define RED_ONCE(k, ...) do { if (!g_red_once[k]++) MLOG("[RED] " __VA_ARGS__); } while (0)
// NVN vertex streams 0â€“15 â†’ Metal vertex buffer slots: 0â€“13 go to 16â€“29 (30 is the constant fallback buffer), 14 and 15 to 14 and 15. Metal has only the 31 slots 0â€“30; translated shaders use only buffer(0) (the argument buffer),
// and the host's own vertex functions read no vertex buffers. 13 Sentinels uses stream 15, and the old 16 + stream went out of range (Metal assertion: buffer index (31) must be < 31)
#define VB_SLOT(st) ((st) < 14 ? 16 + (st) : (st))

// ---------- Texture â†’ MTLTexture ----------
static int fmt_compressed(int f) { return (f >= 0x41 && f <= 0x50) || (f >= 0x79 && f <= 0x94); }
static int fmt_depth(int f) { return f >= 0x31 && f <= 0x36; }
typedef struct { MTLPixelFormat a, b; } pfv_pair;
static const pfv_pair g_pfv_srgb[] = {
    {MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Unorm_sRGB},
    {MTLPixelFormatBGRA8Unorm, MTLPixelFormatBGRA8Unorm_sRGB},
    {MTLPixelFormatBC1_RGBA, MTLPixelFormatBC1_RGBA_sRGB},
    {MTLPixelFormatBC2_RGBA, MTLPixelFormatBC2_RGBA_sRGB},
    {MTLPixelFormatBC3_RGBA, MTLPixelFormatBC3_RGBA_sRGB},
    {MTLPixelFormatBC7_RGBAUnorm, MTLPixelFormatBC7_RGBAUnorm_sRGB},
};
static int tex_need_pfv(MTLPixelFormat base, MTLPixelFormat view) {
    if (base == view) return 0;
    for (size_t i = 0; i < sizeof g_pfv_srgb / sizeof *g_pfv_srgb; i++)
        if ((base == g_pfv_srgb[i].a && view == g_pfv_srgb[i].b) || (base == g_pfv_srgb[i].b && view == g_pfv_srgb[i].a)) return 0;
    return 1;
}
// Cache tables on the replay hot path always use nvn_hmap (C hash, keys not boxed); the table stores CFBridgingRetain'ed objects, CFRelease'd on replace / delete
static void hm_rel(void *v) { CFRelease(v); }
static __unsafe_unretained id hm_get_id(const nvn_hmap *m, const void *k) { return (__bridge id)nvn_hm_get(m, k); }
static void hm_put_id(nvn_hmap *m, const void *k, id o) { void *old = nvn_hm_put(m, k, (void *)CFBridgingRetain(o)); if (old) CFRelease(old); }
static void hm_del_id(nvn_hmap *m, const void *k) { void *old = nvn_hm_del(m, k); if (old) CFRelease(old); }
static nvn_hmap g_by_serial = {.klen = 8};     // Serial â†’ MTLTexture / NvPkg / NvProg
static pthread_mutex_t g_smu = PTHREAD_MUTEX_INITIALIZER;
// Storage of textures the game copies from (soft_harden): those stay whole, as the copy needs what the game has at that level. Mario Kart World copies source textures into its
// 8192x8192 terrain arrays; a skipped copy left them zeros, which ASTC decodes as magenta: a purple road
static nvn_hmap g_soft_hard = {.klen = 8};   // under g_smu
// Serial table generation: +1 when entries are added / removed or a packaged texture is replaced. Downstream caches (texture views, programs) record the generation at fetch time; if it is unchanged they can be used directly without another table lookup
static _Atomic uint64_t g_ser_gen = 1;
// Textures shared by storage (non-packaged): key = memory pool + offset + texture description. An NVN texture object is only a descriptor; an object rebuilt over the same storage with the same description is the same texture;
// SMBW rebuilds its transient render targets on new objects every frame and never Finalizes them, so building one per serial leaked about 13 MB of GPU memory per frame. Textures here are kept resident by the sharing table,
// and when the serial table replaces / deletes them it only drops its reference (defer_obj_unref), never removing them from the residency set. Entries go when their pool is finalized (mtl_alias_drop_pool)
// or another texture takes over their storage (alias_take_over).
typedef struct { uint64_t pool; int64_t off; nvn_texdesc d; } alias_key;
static nvn_hmap g_alias = {.klen = sizeof(alias_key)};
static nvn_hmap g_alias_seen = {.klen = sizeof(alias_key)};   // key â†’ g_fence + 1 at its last lookup or creation (alias_take_over spares entries looked up lately)
// Live GPU-only image descriptions, including those not used yet. A dynamic-resolution family declares its largest image before any particular resolution is used.
// Keeping only descriptions here lets its first allocation cover that family exactly, without allocating every resolution or rounding to a power of two.
static nvn_hmap g_image_desc = {.klen = 8};   // serial â†’ malloc'ed tex_t, under g_smu
static int image_shape(const tex_t *t) {
    return t && t->pool && !t->has_layout && t->d.flags == (NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_IMAGE)
        && t->d.target == NVN_TEXTURE_TARGET_2D && t->d.depth == 1 && t->d.levels == 1 && t->d.samples <= 1 && !t->d.stride;
}
static void image_desc_drop_pool(const void *pool) {   // caller holds g_smu
    for (uint32_t i = 0; i < g_image_desc.cap;) {
        if (g_image_desc.vals[i] && ((tex_t *)g_image_desc.vals[i])->pool == pool) {
            uint64_t serial; memcpy(&serial, g_image_desc.keys + (size_t)i * 8, 8);
            free(nvn_hm_del(&g_image_desc, &serial));   // deletion shifts entries: examine this slot again
        } else i++;
    }
}
static CFMutableSetRef g_alias_set;   // Textures in the sharing table (non-owning; used to ask "is this shared")
static uint64_t g_alias_evicted;   // shared textures evicted on pool Finalize so far (for the memory report)
static int alias_has(id o) { return g_alias_set && o && CFSetContainsValue(g_alias_set, (__bridge const void *)o); }
// On memory pool Finalize, evict the textures built on that pool from the sharing table (collect the keys first, then delete: deletion shifts entries back). The sharing table used not to evict: Mario's pools stay for good, so entries are bounded;
// Engage frees pools on scene changes and rebuilds on new ones, so the old pools' render targets and other private textures stayed in the table forever (about 0.8 GB more GPU memory over 850 s of the title demo), and linear textures built directly on a pool also kept the whole pool buffer alive.
// Also, when a pool object's address is reused, old entries would hand a texture on the old storage to the new pool's texture at the same offset and description. Evicted textures go through deferred release; if the serial table still holds one, that is just one more reference
static void treg_drop_pool_views(const void *pool);
static void cpu_tex_drop_pool(const void *pool);
void mtl_alias_drop_pool(const void *pool) {
    if (!pool) return;
    pthread_mutex_lock(&g_smu);
    image_desc_drop_pool(pool);
    uint32_t n = 0; alias_key *ks = NULL;
    for (uint32_t i = 0; i < g_alias.cap; i++) {
        if (!g_alias.vals[i]) continue;
        alias_key k; memcpy(&k, g_alias.keys + (size_t)i * g_alias.klen, sizeof k);
        if (k.pool != (uint64_t)(uintptr_t)pool) continue;
        alias_key *nk = realloc(ks, (n + 1) * sizeof *ks); if (!nk) break; ks = nk; ks[n++] = k;
    }
    for (uint32_t j = 0; j < n; j++) {
        void *old = nvn_hm_del(&g_alias, &ks[j]);
        if (!old) continue;
        nvn_hm_del(&g_alias_seen, &ks[j]);
        id o = CFBridgingRelease(old);
        if (g_alias_set) CFSetRemoveValue(g_alias_set, (__bridge const void *)o);
        defer_obj_release(o);
    }
    if (n) { g_ser_gen++; g_alias_evicted += n; }
    pst_drop_pool(pool);   // the storage table's entries of the pool (see g_pst)
    pthread_mutex_unlock(&g_smu);
    free(ks);
    cpu_tex_drop_pool(pool);
    treg_drop_pool_views(pool);   // outside g_smu: the texture ID registry lock g_rmu is taken before g_smu
}
// A new shared texture takes over storage: entries of the same pool whose storage overlaps the new texture's under another description go, unless a live texture object still resolves
// to them (an object rebuilt over the same storage with the same description must keep finding it: SMBW's transient render targets) or a lookup found them in the last ALIAS_IDLE submissions
// (Breath of the Wild finalizes its transient render targets before it submits and makes them again the next frame: targets sharing storage would evict each other every frame). On NVN the memory now belongs to the new texture;
// the old private Metal texture only held a copy nothing can name again. STEINS;GATE RE:BOOT decodes every image it shows into one scratch region of a pool that lives for good and
// finalizes the texture before the next one: each image added an entry, 468 of them and 1.58 GB of GPU memory 24,000 frames into a new game. A finalized object resolved through g_ghost
// keeps its own reference, so the entry's reference is only unref'd while one does; otherwise the texture is retired (deferred until the submissions that used it complete).
// Cached texture-ID views follow g_ser_gen. Caller holds g_smu
enum { ALIAS_IDLE = 16 };
static uint64_t g_alias_taken;   // shared textures evicted when another texture took over their storage (for the memory report)
static nvn_hmap g_ghost;   // defined below (finalized texture serials)
static int serial_holds(nvn_hmap *m, id o) { for (uint32_t i = 0; i < m->cap; i++) if (m->vals[i] == (__bridge void *)o) return 1; return 0; }
static void alias_take_over(const alias_key *nk) {
    nvn_texlayout L; nvn_tex_layout(&nk->d, &L);
    if (!L.size) return;
    int64_t lo = nk->off, hi = nk->off + (int64_t)L.size;
    uint32_t n = 0; alias_key ks[16];
    for (uint32_t i = 0; i < g_alias.cap && n < 16; i++) {
        if (!g_alias.vals[i]) continue;
        alias_key k; memcpy(&k, g_alias.keys + (size_t)i * g_alias.klen, sizeof k);
        if (k.pool != nk->pool || !memcmp(&k, nk, sizeof k) || k.off >= hi) continue;
        nvn_texlayout kl; nvn_tex_layout(&k.d, &kl);
        uint64_t seen = (uint64_t)(uintptr_t)nvn_hm_get(&g_alias_seen, &k);
        if (k.off + (int64_t)kl.size <= lo || (seen && g_fence + 1 - seen < ALIAS_IDLE) || serial_holds(&g_by_serial, (__bridge id)g_alias.vals[i])) continue;
        ks[n++] = k;
    }
    for (uint32_t j = 0; j < n; j++) {
        void *old = nvn_hm_del(&g_alias, &ks[j]);
        if (!old) continue;
        nvn_hm_del(&g_alias_seen, &ks[j]);
        id o = CFBridgingRelease(old);
        if (g_alias_set) CFSetRemoveValue(g_alias_set, (__bridge const void *)o);
        if (serial_holds(&g_ghost, o)) defer_obj_unref(o); else defer_obj_release(o);
    }
    if (n) { g_ser_gen++; g_alias_taken += n; }
}
static __unsafe_unretained id serial_get(uint64_t k) { pthread_mutex_lock(&g_smu); __unsafe_unretained id o = hm_get_id(&g_by_serial, &k); pthread_mutex_unlock(&g_smu); return o; }
static void serial_put(uint64_t k, id o) { pthread_mutex_lock(&g_smu); id old = hm_get_id(&g_by_serial, &k); if (old) { if (alias_has(old)) defer_obj_unref(old); else defer_obj_release(old); } hm_put_id(&g_by_serial, &k, o); g_ser_gen++; pthread_mutex_unlock(&g_smu); }
static void serial_del(uint64_t k) { pthread_mutex_lock(&g_smu); id old = hm_get_id(&g_by_serial, &k); if (old) { if (alias_has(old)) defer_obj_unref(old); else defer_obj_release(old); hm_del_id(&g_by_serial, &k); g_ser_gen++; } pthread_mutex_unlock(&g_smu); }   // nothing to drop: the texture ID caches stay valid
// Texture serials owned by a live NVN texture object (TextureInitialize until Finalize or re-initialization, hle_nvn.c). Command records and texture pool registrations snapshot the
// texture object (tex_snap in hle_nvn.c, mtl_reg_tex), so replay can resolve a texture the game has already finalized: Breath of the Wild records each frame's transient render targets
// (the HUD captures, the 640x360 half-resolution buffers) and finalizes their objects before it submits. Those serials went back into g_by_serial and stayed there for good, as no second
// Finalize comes: about 10 a frame, 200,000 after an hour in the first shrine. A dead serial resolves through g_ghost instead, a small table whose oldest entry goes when a new one comes,
// so a record replayed again soon still finds its texture
enum { NGHOST = 512 };
static nvn_hmap g_tex_live = {.klen = 8};   // serial â†’ (void *)1, under g_smu
static nvn_hmap g_ghost = {.klen = 8}; static uint64_t g_ghost_ring[NGHOST]; static unsigned g_ghost_next; static uint64_t g_ghost_puts;
void mtl_tex_init(const tex_t *t) {
    if (!t->serial) return;
    tex_t *desc = image_shape(t) ? malloc(sizeof *desc) : NULL; if (desc) *desc = *t;
    pthread_mutex_lock(&g_smu);
    nvn_hm_put(&g_tex_live, &t->serial, (void *)1);
    if (desc) free(nvn_hm_put(&g_image_desc, &t->serial, desc));
    pthread_mutex_unlock(&g_smu);
}
static void obj_drop(id o) { if (alias_has(o)) defer_obj_unref(o); else defer_obj_release(o); }   // caller holds g_smu
static __unsafe_unretained id tex_get(uint64_t k) {
    pthread_mutex_lock(&g_smu); __unsafe_unretained id o = hm_get_id(&g_by_serial, &k); if (!o && g_ghost.n) o = hm_get_id(&g_ghost, &k); pthread_mutex_unlock(&g_smu); return o;
}
static void tex_put(uint64_t k, id o) {   // serials from 2^62 up are external textures (mtl_ext_tex), never initialized by the game
    pthread_mutex_lock(&g_smu);
    nvn_hmap *m = k >= (1ull << 62) || nvn_hm_get(&g_tex_live, &k) ? &g_by_serial : &g_ghost;
    id old = hm_get_id(m, &k);
    if (old) obj_drop(old);
    else if (m == &g_ghost) {
        uint64_t *slot = &g_ghost_ring[g_ghost_next++ % NGHOST];
        if (*slot) { id ev = hm_get_id(&g_ghost, slot); if (ev) obj_drop(ev); hm_del_id(&g_ghost, slot); }
        *slot = k; g_ghost_puts++;
    }
    hm_put_id(m, &k, o); g_ser_gen++;
    pthread_mutex_unlock(&g_smu);
}
// Host record of a packaged texture: the current MTLTexture, the upload stamp, and the global write generation at the last check
// spare: the texture the last re-upload replaced, kept resident and written by the next re-upload once the GPU has passed spare_fence (a texture the CPU rewrites every frame, a movie frame,
// alternates between two textures instead of allocating one per frame: Link's Awakening's title movie made and freed an 8 MB texture a frame)
@interface NvPkg : NSObject { @public id<MTLTexture> tex, spare, view; uint64_t stamp, seen, hash, spare_fence, vgen, vseen; const void *span; int sparse, vbase; uint8_t *edge[2]; uint32_t edge_n[2];
    uint64_t clean_at, vone_gen; const void *vone; uint8_t *rmem; uint64_t rsize, born, used; const void *rpool; int reup, released; } @end   // r*: its storage while it is in the storage table (profile tex_release: pkg_store), born / used: frames   // hash: full hash of the uploaded content under DM_NVN_DIRTYCHECK; span: its tracked range (nvn_dirty_span, NULL = untracked);
// sparse: in a sparse virtual pool, read through its page table (hle_nvn.c nvn_sparse_*), vgen the page table's generation at the upload and vseen at the last look; view: what it is bound as (sparse_view)
// edge: the uploaded bytes of the storage's first and last host pages where other data shares them (pkg_edges_take)
@implementation NvPkg
- (void)dealloc { res_remove(tex); if (spare) res_remove(spare); free(edge[0]); free(edge[1]); }
@end
static uint64_t g_pkg_uploads, g_pkg_reuploads, g_pkg_miss, g_pkg_edge_skips;
// Profile tex_release, packaged textures: a texture the game loaded from its files (a packaged layout) that it has not written since is kept by its storage and description
// (g_pst), so a texture object made again over the same storage takes the upload rather than reading the storage (the game re-creates about one texture in 17 so: 585 of 10,240 in
// a race of Mario Kart World), and once it has lived 120 frames unchanged the memory of its storage's whole pages goes back (pkg_store_sweep, nvn_dirty_release). A write later
// faults and dirties it: the next use uploads what the game wrote. A pool with a zero-copy buffer keeps its pages (the buffer holds them), so such textures upload on the CPU
// while their pool has none (pkg_upload_). An entry goes when its storage is written while no texture uses it, or with its pool
typedef struct { uint64_t mem, sig; } pst_key;
static nvn_hmap g_pst = {.klen = sizeof(pst_key)};   // under g_smu
static uint64_t g_pst_reused, g_pst_released;
static pst_key pst_key_of(const tex_t *t) { uint64_t h = 1469598103934665603ull; const uint8_t *d = (const uint8_t *)&t->d; for (size_t i = 0; i < sizeof t->d; i++) h = (h ^ d[i]) * 1099511628211ull; return (pst_key){(uint64_t)(uintptr_t)t->mem, h}; }
int nvn_sparse_pool(const void *pool);
// Off: Mario Kart World streams finer levels into a texture's storage after its first use, and the re-upload that follows read the levels whose pages had gone back (zeros:
// magenta ASTC over the whole course). Re-uploading only the levels the game wrote, the others copied from the texture, would make it safe
enum { PKG_RELEASE = 0 };
static int pst_eligible(const tex_t *t) { return PKG_RELEASE && DM_GAME_TEX_RELEASE && t->has_layout && t->mem && !(t->pool && nvn_sparse_pool(t->pool)); }
int mtl_pool_wrapped(const void *addr, uint64_t len);
static void pst_drop_pool(const void *pool) {   // caller holds g_smu
    for (uint32_t i = 0; i < g_pst.cap; i++) {
        NvPkg *pk = (__bridge NvPkg *)g_pst.vals[i]; if (!pk || pk->rpool != pool) continue;
        pst_key k; memcpy(&k, g_pst.keys + (size_t)i * g_pst.klen, sizeof k); hm_del_id(&g_pst, &k); i = (uint32_t)-1;   // the table may have moved its slots: from the start
    }
}
// Every 120 frames, on the submit thread (where packaged textures are uploaded): give back the storage pages of the packaged textures in g_pst that have lived 120 frames unchanged,
// and drop the entries whose storage was written while no texture used them for 600 frames (the game took that memory for something else)
static void pkg_store_sweep(void) {
    static uint64_t last; uint64_t f = atomic_load(&g_cur_frame);
    if (!PKG_RELEASE || !DM_GAME_TEX_RELEASE || f - last < 120) return;
    last = f;
    NSMutableArray<NvPkg *> *all = [NSMutableArray array]; NSMutableData *keys = [NSMutableData data];
    pthread_mutex_lock(&g_smu);
    for (uint32_t i = 0; i < g_pst.cap; i++) if (g_pst.vals[i]) { [all addObject:(__bridge NvPkg *)g_pst.vals[i]]; [keys appendBytes:g_pst.keys + (size_t)i * g_pst.klen length:sizeof(pst_key)]; }
    pthread_mutex_unlock(&g_smu);
    const pst_key *ks = keys.bytes; uint64_t freed = 0;
    for (NSUInteger i = 0; i < all.count; i++) {
        NvPkg *pk = all[i];
        int clean = pk->span && !pk->reup && !nvn_dirty_since_span(pk->span, pk->rmem, pk->rsize, pk->stamp);
        if (!clean) { if (f - pk->used > 600) { pthread_mutex_lock(&g_smu); if (hm_get_id(&g_pst, &ks[i]) == pk) hm_del_id(&g_pst, &ks[i]); pthread_mutex_unlock(&g_smu); } continue; }
        if (pk->released || f - pk->born < 120 || mtl_pool_wrapped(pk->rmem, pk->rsize)) continue;
        freed += nvn_dirty_release(pk->rmem, pk->rsize, NULL); pk->released = 1;
    }
    g_pst_released += freed;
    static unsigned n; if (freed && (n++ < 8 || !(n & (n - 1)))) MLOG("packaged textures: %llu MB of storage given back (%llu MB so far, %lu textures kept by storage, %llu made again from a kept upload)",
        (unsigned long long)(freed >> 20), (unsigned long long)(g_pst_released >> 20), (unsigned long)all.count, (unsigned long long)g_pst_reused);
}
// The whole pages only level 0 of a soft texture holds, less the last (a texture made again a little lower over the same bytes, as the game does, has level 1 start inside it): nothing reads
// them again, as uploads start at level 1. Not under a zero-copy buffer, which holds its pages (mtl_pool_wrapped)
typedef struct { uint64_t serial, born, n; uint8_t *mem; } soft_pend;
static soft_pend *g_soft_pend; static uint32_t g_nsoft_pend, g_soft_pend_cap;
static uint64_t g_soft_released;
static void tex_soft_release(const tex_t *t) {   // on the submit thread, as soft_sweep: when it has lived SOFT_WAIT frames
    size_t pg = (size_t)getpagesize(); uint64_t l1 = nvn_tex_level_offset(&t->d, 1);
    if (l1 <= pg) return;
    if (g_nsoft_pend == g_soft_pend_cap && !(g_soft_pend = realloc(g_soft_pend, (g_soft_pend_cap = g_soft_pend_cap ? g_soft_pend_cap * 2 : 256) * sizeof *g_soft_pend))) MDIE("soft texture list allocation failed");
    g_soft_pend[g_nsoft_pend++] = (soft_pend){t->serial, atomic_load(&g_cur_frame), l1 - pg, t->mem};
}
enum { SOFT_WAIT = 300 };
static void soft_sweep(void) {
    static uint64_t last; uint64_t f = atomic_load(&g_cur_frame);
    if (f - last < 30 || !g_nsoft_pend) return;
    last = f; uint32_t keep = 0; uint64_t freed = 0;
    for (uint32_t i = 0; i < g_nsoft_pend; i++) {
        soft_pend *e = &g_soft_pend[i];
        if (f - e->born < SOFT_WAIT) { g_soft_pend[keep++] = *e; continue; }
        id have = tex_get(e->serial); uint64_t k = (uint64_t)(uintptr_t)e->mem;   // gone: the game may have that memory in use for something else
        pthread_mutex_lock(&g_smu); int hard = nvn_hm_get(&g_soft_hard, &k) != NULL; pthread_mutex_unlock(&g_smu);
        if (hard || ![have isKindOfClass:[NvPkg class]] || mtl_pool_wrapped(e->mem, e->n)) continue;
        freed += nvn_dirty_release(e->mem, e->n, NULL); ((NvPkg *)have)->released = 1;
    }
    g_nsoft_pend = keep; g_soft_released += freed;
    static unsigned n; if (freed && (n++ < 4 || !(n & (n - 1)))) MLOG("soft textures: %llu MB of finest levels' storage given back so far", (unsigned long long)(g_soft_released >> 20));
}
static uint64_t g_pfv_upgrades;
// DM_NVN_DIRTYCHECK=1 (diagnostic, compared against ground truth): fully hash the pool content on every use; changed but not detected by dirty-page tracking = a missed detection
static int dirtycheck(void) { static int v = -1; if (v < 0) v = getenv("DM_NVN_DIRTYCHECK") != NULL; return v; }
static uint64_t full_hash(const uint8_t *m, uint64_t n) { uint64_t h = 1469598103934665603ull; for (uint64_t i = 0; i < n; i++) h = (h ^ m[i]) * 1099511628211ull; return h; }
// Writes beside a texture. The tracker marks whole host pages (16 KB) dirty, and guest memory pools are only 4 KB aligned: the first and last pages of a texture's storage are shared with
// whatever lies beside it. PokÃ©mon Sword keeps each texture of its GFPAK models in a pool of its own, with the pool's object 4 KB before the storage, in the same host page, and writes there
// every frame: 73,232 re-uploads in 70 s of play from a save, every one checked of unchanged content, 12 GB of upload staging in 8 minutes. So the bytes of the storage in those edge pages
// are kept at the upload (pkg_edges_take), and a texture whose only written pages are its edges, with those bytes unchanged, is protected again instead of re-uploaded (pkg_edges_clean).
// The edges are the whole storage when it lies within two pages (no page wholly inside)
static void pkg_edge_ranges(const tex_t *t, uintptr_t r[2][2], uintptr_t *ia, uintptr_t *ie) {
    uintptr_t pg = (uintptr_t)getpagesize(), a = (uintptr_t)t->mem, e = a + t->L.size;
    *ia = (a + pg - 1) & ~(pg - 1); *ie = e & ~(pg - 1);
    if (*ie <= *ia) { r[0][0] = a; r[0][1] = e; r[1][0] = r[1][1] = 0; *ia = *ie = 0; return; }
    r[0][0] = a; r[0][1] = *ia; r[1][0] = *ie; r[1][1] = e;
}
static void pkg_edges_take(NvPkg *pk, const tex_t *t) {   // right after an upload, with its stamp in pk->stamp
    for (int i = 0; i < 2; i++) { free(pk->edge[i]); pk->edge[i] = NULL; pk->edge_n[i] = 0; }
    if (pk->sparse || !pk->span || !t->mem) return;
    uintptr_t r[2][2], ia, ie; pkg_edge_ranges(t, r, &ia, &ie);
    for (int i = 0; i < 2; i++) if (r[i][1] > r[i][0]) { pk->edge_n[i] = (uint32_t)(r[i][1] - r[i][0]); pk->edge[i] = malloc(pk->edge_n[i]); memcpy(pk->edge[i], (const void *)r[i][0], pk->edge_n[i]); }
    // A write after the upload's stamp may have reached the copy but not the upload: then there is no copy to trust, and the next write re-uploads as before
    if (nvn_dirty_since_span(pk->span, t->mem, t->L.size, pk->stamp)) for (int i = 0; i < 2; i++) { free(pk->edge[i]); pk->edge[i] = NULL; pk->edge_n[i] = 0; }
}
static int pkg_edges_clean(NvPkg *pk, const tex_t *t) {   // the covered pages were written since the upload: 1 = only edge pages, and the texture's bytes there are as uploaded
    if (!pk->edge_n[0] && !pk->edge_n[1]) return 0;
    uintptr_t r[2][2], ia, ie; pkg_edge_ranges(t, r, &ia, &ie);
    uint64_t st = nvn_dirty_now(); nvn_dirty_protect(t->mem, t->L.size);   // take the stamp, protect, then read (as pkg_upload): a later write is seen at the next check
    if (ie > ia && nvn_dirty_since_span(pk->span, (const void *)ia, ie - ia, pk->stamp)) return 0;
    for (int i = 0; i < 2; i++) if (pk->edge_n[i] && memcmp(pk->edge[i], (const void *)r[i][0], pk->edge_n[i])) return 0;
    pk->stamp = st; g_pkg_edge_skips++;
    return 1;
}
// What a render pass drew, read back into the storage (rt_mirror_flush), is already the content of the host texture it was read from: that write is not a change to upload. The handler
// takes a stamp, protects the storage again once it has written it and keeps the texture's bytes in its edge pages (pkg_absorb_note); at the next resolution the texture keeps its host
// texture when that is still the one read back, no page wholly inside it was written since and its edge bytes are unchanged (pkg_absorb_take), as pkg_edges_clean judges neighbours'
// writes. Tears of the Kingdom draws into four CPU-written targets (256x256, 192x192, 96x96 and 500x400) every frame, one of them loading its old content: re-uploaded after every
// readback, 31,249 uploads in 200 s where there had been 19. Its 192x192 and 96x96 ones share a host page, so the readback of one dirtied the other
// Until the readback lands, the storage is being written with what the host texture already holds: the texture keeps it rather than uploading the half-written storage (pkg_absorb_pending)
static pthread_mutex_t g_abs_mu = PTHREAD_MUTEX_INITIALIZER; static struct { uint64_t serial, stamp; void *tex; uint8_t *edge[2]; uint32_t edge_n[2]; } g_abs[16]; static _Atomic int g_nabs;
static struct { uint64_t serial; int n; } g_absp[16]; static _Atomic int g_nabsp;   // the counts are read without the lock first (tex_mtl_)
static void pkg_absorb_pending(uint64_t serial, int d) {
    pthread_mutex_lock(&g_abs_mu);
    int i = 0; while (i < g_nabsp && g_absp[i].serial != serial) i++;
    if (i == g_nabsp && d > 0 && g_nabsp < (int)(sizeof g_absp / sizeof g_absp[0])) { g_absp[i].serial = serial; g_absp[i].n = 0; g_nabsp++; }
    if (i < g_nabsp && !(g_absp[i].n += d)) g_absp[i] = g_absp[--g_nabsp];
    pthread_mutex_unlock(&g_abs_mu);
}
static int pkg_absorb_waits(uint64_t serial) {
    pthread_mutex_lock(&g_abs_mu); int r = 0;
    for (int i = 0; i < g_nabsp && !r; i++) r = g_absp[i].serial == serial;
    pthread_mutex_unlock(&g_abs_mu); return r;
}
static void abs_free(int i) { CFRelease(g_abs[i].tex); free(g_abs[i].edge[0]); free(g_abs[i].edge[1]); }   // only this reference: the texture itself lives on
static void pkg_absorb_note(const tex_t *t, void *tex) {   // right after the readback was written into the storage; tex: retained, released here or by pkg_absorb_take
    uint64_t st = nvn_dirty_now(); nvn_dirty_protect(t->mem, t->L.size);   // stamp, protect, then read (as pkg_upload)
    uintptr_t r[2][2], ia, ie; pkg_edge_ranges(t, r, &ia, &ie);
    uint8_t *edge[2] = {0}; uint32_t edge_n[2] = {0};
    for (int k = 0; k < 2; k++) if (r[k][1] > r[k][0]) { edge_n[k] = (uint32_t)(r[k][1] - r[k][0]); edge[k] = malloc(edge_n[k]); memcpy(edge[k], (const void *)r[k][0], edge_n[k]); }
    pthread_mutex_lock(&g_abs_mu);
    int i = 0; while (i < g_nabs && g_abs[i].serial != t->serial) i++;
    if (i < g_nabs) abs_free(i);
    else if (g_nabs < (int)(sizeof g_abs / sizeof g_abs[0])) g_nabs++;
    else { pthread_mutex_unlock(&g_abs_mu); CFRelease(tex); free(edge[0]); free(edge[1]); return; }   // full: that one is uploaded again, as before
    g_abs[i].serial = t->serial; g_abs[i].stamp = st; g_abs[i].tex = tex;
    for (int k = 0; k < 2; k++) { g_abs[i].edge[k] = edge[k]; g_abs[i].edge_n[k] = edge_n[k]; }
    pthread_mutex_unlock(&g_abs_mu);
}
static void pkg_absorb_take(NvPkg *pk, const tex_t *t) {
    pthread_mutex_lock(&g_abs_mu);
    int i = 0; while (i < g_nabs && g_abs[i].serial != t->serial) i++;
    if (i == g_nabs) { pthread_mutex_unlock(&g_abs_mu); return; }
    typeof(g_abs[0]) e = g_abs[i]; g_abs[i] = g_abs[--g_nabs];
    pthread_mutex_unlock(&g_abs_mu);
    id tex = CFBridgingRelease(e.tex);   // ARC drops this reference on return
    uintptr_t r[2][2], ia, ie; pkg_edge_ranges(t, r, &ia, &ie);
    int ok = tex == pk->tex && pk->span && !pk->sparse && !(ie > ia && nvn_dirty_since_span(pk->span, (const void *)ia, ie - ia, e.stamp));
    for (int k = 0; k < 2 && ok; k++) ok = e.edge_n[k] == (r[k][1] > r[k][0] ? r[k][1] - r[k][0] : 0) && (!e.edge_n[k] || !memcmp(e.edge[k], (const void *)r[k][0], e.edge_n[k]));
    if (ok) {
        pk->stamp = pk->seen = e.stamp; if (dirtycheck()) pk->hash = full_hash(t->mem, t->L.size);
        for (int k = 0; k < 2; k++) { free(pk->edge[k]); pk->edge[k] = e.edge[k]; pk->edge_n[k] = e.edge_n[k]; e.edge[k] = NULL; }   // the bytes as written: a neighbour's write in an edge page is judged by them (pkg_edges_clean)
    }
    free(e.edge[0]); free(e.edge[1]);
}
// Usage an NVN image takes: shader writes, and for R32Uint / R32Sint the shader-atomic usage too, since Metal runs texture atomics only on such textures (dummy_img_atomic) and NVN does
// not say which images a shader will use atomically
static inline int pf_r32i(MTLPixelFormat pf) { return pf == MTLPixelFormatR32Uint || pf == MTLPixelFormatR32Sint; }
static MTLTextureUsage img_usage(const tex_t *t, MTLPixelFormat pf) {
    if (!(t->d.flags & NVN_TEXTURE_FLAGS_IMAGE)) return 0;
    return MTLTextureUsageShaderWrite | (pf_r32i(pf) ? MTLTextureUsageShaderAtomic : 0);
}
// Metal leaves MTLTextureUsageShaderAtomic out of a texture's usage property (0x3 for a texture made with 0x23), so it cannot be read back: every shader-writable R32Uint / R32Sint
// texture the host makes has it (img_usage, tex_upgrade, dummy_img_atomic), and that is what this tests
static int tex_atomic_capable(id<MTLTexture> base) { return pf_r32i(base.pixelFormat) && (base.usage & MTLTextureUsageShaderWrite); }
// Profile tex_soft (DM_NVN_TEX_SOFT=0 / 1): a large block-compressed texture the game loaded from its files is kept without its finest level, on the GPU and in guest memory: its host
// texture is that of level 1 (half the size, one level less, sampled through the same normalized coordinates), and the pages only level 0 holds go back to the system (tex_soft_release)
// (later: soft_sweep, which leaves time for a first copy) and are never read again. About three quarters of a texture's bytes, twice over. Mario Kart World's race does not fit an iPhone 18 Pro otherwise. Level numbers the game gives
// (views, copies) are one lower here: tex_soft_level. Reads of the size in a shader see the smaller size
static int tex_soft(const tex_t *t) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_NVN_TEX_SOFT"); on = e && *e ? atoi(e) != 0 : DM_GAME_TEX_SOFT; if (on) MLOG("large block-compressed textures kept without their finest level (profile tex_soft; DM_NVN_TEX_SOFT=0 turns it off)"); }
    if (!on || !t->has_layout || !t->mem || t->d.levels < 3 || (t->d.target != 1 && t->d.target != 4)) return 0;
    if (t->pool && t->pool->magic == MAGIC('P', 'O', 'O', 'L') && (t->pool->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL)) return 0;   // a sparse pool's: sparse_upload, tiles that follow the game's mappings
    if (MAX(t->d.width, t->d.height) < 512 || MIN(t->d.width, t->d.height) < 256) return 0;
    int bw, bh; if (!nvn_fmt_bpb(t->d.format, &bw, &bh) || bw < 2) return 0;
    uint64_t k = (uint64_t)(uintptr_t)t->mem; pthread_mutex_lock(&g_smu); int hard = nvn_hm_get(&g_soft_hard, &k) != NULL; pthread_mutex_unlock(&g_smu);
    return !hard;
}
// A level the game names, as the host texture's level: 0 when there is none (level 0 of a soft texture)
static int tex_soft_level(const tex_t *t, unsigned *level) { if (!t || !tex_soft(t)) return 1; if (!*level) return 0; --*level; return 1; }
static MTLTextureDescriptor *tex_desc(const tex_t *t, int packaged) {
    uint32_t pf = 0; uint8_t swz[4];
    if (!nvn_fmt_to_mtl(t->d.format, &pf, swz) || !pf) { RED_ONCE(0, "texture format %#x has no Metal equivalent (first case)", t->d.format); return nil; }
    int signed_bc = 0;
    if (packaged && bc_cpu_fallback() && bc_rgba_kind((MTLPixelFormat)pf, &signed_bc))
        pf = bc_rgba_srgb((MTLPixelFormat)pf) ? MTLPixelFormatRGBA8Unorm_sRGB : MTLPixelFormatRGBA8Unorm;
    // Metal's limits: 16384 in 2D, 2048 in 3D, 2048 array layers; beyond them Metal aborts the process on an assertion, so they are caught here with a red line
    if (t->d.width > 16384 || t->d.height > 16384 || t->d.depth > 2048) { RED_ONCE(71, "texture size exceeds Metal limits: %uÃ—%uÃ—%u levels %u format %#x target %d (first case ser=%llu)", t->d.width, t->d.height, t->d.depth, t->d.levels, t->d.format, t->d.target, (unsigned long long)t->serial); return nil; }
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.pixelFormat = (MTLPixelFormat)pf;
    d.width = t->d.width ? t->d.width : 1; d.height = t->d.height ? t->d.height : 1; d.depth = 1; d.arrayLength = 1;
    d.mipmapLevelCount = t->d.levels ? t->d.levels : 1;
    if (packaged && tex_soft(t)) { d.width = MAX(d.width >> 1, 1u); d.height = MAX(d.height >> 1, 1u); d.mipmapLevelCount--; }
    switch (t->d.target) {
        // 1D / 1D array are created as one-row 2D / 2D array textures: translated shaders never declare texture1d (0 in all five games' libraries) and always sample them as 2D; a Metal 1D texture bound to a texture2d argument is a type mismatch with undefined reads.
        // Super Mario 3D World + Bowser's Fury's material shaders sample 64x1 R32F 1D lookup tables heavily every frame (on Maxwell a 2D sample of a 1D texture reads that one row)
        case 0: d.textureType = MTLTextureType2D; d.height = 1; break;
        case 2: d.textureType = MTLTextureType3D; d.depth = t->d.depth; break;
        case 3: d.textureType = MTLTextureType2DArray; d.arrayLength = t->d.height; d.height = 1; break;
        case 4: d.textureType = MTLTextureType2DArray; d.arrayLength = t->d.depth; break;
        case 5: d.textureType = MTLTextureType2DMultisample; d.sampleCount = t->d.samples; d.mipmapLevelCount = 1; break;
        case 6: d.textureType = MTLTextureType2DMultisampleArray; d.sampleCount = t->d.samples; d.arrayLength = t->d.depth; d.mipmapLevelCount = 1; break;
        // Cube textures are created as 6-layer 2D arrays and sampled through cube / cube array views made by tex_fit: Metal only allows cube array views of a 2D array base texture (not of a cube base).
        // The Legend of Zelda: Breath of the Wild has 1512 shaders that sample the environment light probes as a cube array while a single cube texture is bound (texture ID 1064 onwards): with mismatched types the reads are undefined, and every lit surface came out black
        case 8: d.textureType = MTLTextureType2DArray; d.arrayLength = 6; break;
        // Cube arrays likewise, as a 2D array of all their faces: NVN gives a cube array's depth in layer-faces (a multiple of 6), where a Metal cube array's arrayLength counts cubes.
        // Created as a cube array with arrayLength = depth, DRAGON QUEST XI S's 510-face one (85 cubes, made when its prologue movie ends) asked for 510 cubes, past Metal's 341, and Metal aborted the process
        case 9: d.textureType = MTLTextureType2DArray; d.arrayLength = t->d.depth; break;
        case 10: RED_ONCE(1, "buffer textures (TARGET_BUFFER) not implemented"); return nil;
        default: d.textureType = MTLTextureType2D; break;
    }
    // Fields NVN takes that Metal refuses: Metal aborts the process on them (-[MTLTextureDescriptorInternal validateWithDevice:]). An array or 3D texture of no layers gets one; one of
    // more than 2048 is not made. More levels than the size holds (on NVN the extra ones are 1x1; a mod's 1x1 texture declaring 9, MacNativeGames f864eb0): made with the levels the
    // size holds, and copies to the others are skipped (tex_level_ok). A multisample texture of a sample count the GPU lacks (Apple GPUs have 2 and 4; NVN also 8 and 16), or of
    // one sample, has no Metal form and is not made, as a texture past Metal's size limits (fewer samples would leave shaders that read sample 4 or higher reading past the texture)
    if (d.depth < 1) d.depth = 1;
    if (d.arrayLength < 1) d.arrayLength = 1;
    if (d.arrayLength > 2048) { RED_ONCE(113, "texture of %lu layers exceeds Metal's 2048: not made (format %#x target %d, first case ser=%llu)", (unsigned long)d.arrayLength, t->d.format, t->d.target, (unsigned long long)t->serial); return nil; }
    NSUInteger big = MAX(d.width, d.height), fit = 1;
    if (d.textureType == MTLTextureType3D) big = MAX(big, d.depth);
    while (big >> fit) fit++;
    if (d.mipmapLevelCount > fit) {
        RED_ONCE(114, "texture %uÃ—%uÃ—%u declares %lu levels, its size holds %lu: made with %lu, the others are not kept (format %#x, first case ser=%llu)", t->d.width, t->d.height, t->d.depth,
                 (unsigned long)d.mipmapLevelCount, (unsigned long)fit, (unsigned long)fit, t->d.format, (unsigned long long)t->serial);
        d.mipmapLevelCount = fit;
    }
    if (d.textureType == MTLTextureType2DMultisample || d.textureType == MTLTextureType2DMultisampleArray) {
        if (d.sampleCount < 2 || ![g_dev supportsTextureSampleCount:d.sampleCount]) {
            RED_ONCE(115, "multisample texture of %lu samples: this GPU has no such texture, not made (format %#x, first case ser=%llu)", (unsigned long)d.sampleCount, t->d.format, (unsigned long long)t->serial);
            return nil;
        }
    }
    d.storageMode = MTLStorageModePrivate;   // Render targets and assets both go in GPU private storage (lossless compression available)
    d.usage = MTLTextureUsageShaderRead | (fmt_compressed(t->d.format) || packaged ? 0 : MTLTextureUsageRenderTarget)
            | img_usage(t, d.pixelFormat);   // Image usage: shader-writable (compute kernels generating the 3D color lookup table, etc.
    // A depth-stencil texture has the format-view usage from the start: its stencil is read through an X32_Stencil8 view (stencil-mode views and textures, the depth-stencil copy), which Metal's
    // contract allows only with that usage. Apple GPUs read the view correctly without it, but the API validation layer aborts the process at the first such view (PR #106, found on Luigi's
    // Mansion 3), and upgrading later would recreate a depth buffer in use. Unlike colour formats it costs nothing measured: on an M4 Pro a 2560x1440 one has the same allocated size and the same
    // GPU time over eight load / store passes with and without it
    if (d.pixelFormat == MTLPixelFormatDepth32Float_Stencil8) d.usage |= MTLTextureUsagePixelFormatView;
    return d;
}
// In-place upgrade: when the original texture was not created with PixelFormatView (off by default, to keep Apple GPU lossless compression) but a different-format view is needed, create a new one with the original spec + PixelFormatView, copy over every slice Ã— level, and replace the registration.
// Count: across the 4 screens character select / camp / wilderness / combat, 17473 textures needed a different-format view 0 times, so this is a fallback and does not rely on predicting which textures need it
// The same upgrade adds render-target usage for a whole-level clear (clear_tex) of a texture made without it: Monster Hunter Rise clears a 32x80 packaged one
static id<MTLTexture> tex_upgrade(const tex_t *t, id<MTLTexture> base, MTLTextureUsage add, MTLPixelFormat view_pf) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.textureType = base.textureType; d.pixelFormat = base.pixelFormat; d.width = base.width; d.height = base.height; d.depth = base.depth;   // Keep the original format, only add the format-view usage
    d.arrayLength = base.arrayLength; d.mipmapLevelCount = base.mipmapLevelCount; d.sampleCount = base.sampleCount;
    d.storageMode = base.storageMode; d.cpuCacheMode = base.cpuCacheMode; d.hazardTrackingMode = base.hazardTrackingMode;
    d.resourceOptions = base.resourceOptions; d.usage = base.usage | add | (tex_atomic_capable(base) ? MTLTextureUsageShaderAtomic : 0);   // the usage property leaves the atomic usage out
    // A linear texture aliases the guest pool. Adding usage must keep that backing: a new standalone texture copies the old pixels once, but later GPU writes never reach the CPU map.
    id<MTLBuffer> backing = base.buffer;
    id<MTLTexture> upgraded = backing ? [backing newTextureWithDescriptor:d offset:base.bufferOffset bytesPerRow:base.bufferBytesPerRow] : [g_dev newTextureWithDescriptor:d];
    if (!upgraded) { RED_ONCE(45, "texture upgrade failed ser=%llu format %lu â†’ %lu usage %#lx", t->serial, (unsigned long)base.pixelFormat, (unsigned long)view_pf, (unsigned long)d.usage); return base; }
    res_add(upgraded);
    id<MTL4ComputeCommandEncoder> b = backing ? nil : blit();   // shared backing already contains the pixels; copying it onto itself would overlap
    NSUInteger slices = base.textureType == MTLTextureType3D ? 1 : base.arrayLength * (base.textureType == MTLTextureTypeCube || base.textureType == MTLTextureTypeCubeArray ? 6 : 1);
    for (NSUInteger l = 0; !backing && l < base.mipmapLevelCount; l++) {
        NSUInteger w = MAX(base.width >> l, 1u), h = MAX(base.height >> l, 1u), z = base.textureType == MTLTextureType3D ? MAX(base.depth >> l, 1u) : 1;
        if (base.textureType == MTLTextureType3D)
            enc_b_copy_texture(b, base, 0, l, MTLOriginMake(0, 0, 0), MTLSizeMake(w, h, z), upgraded, 0, l, MTLOriginMake(0, 0, 0));
        else for (NSUInteger s = 0; s < slices; s++)
            enc_b_copy_texture(b, base, s, l, MTLOriginMake(0, 0, 0), MTLSizeMake(w, h, 1), upgraded, s, l, MTLOriginMake(0, 0, 0));
    }
    id holder = tex_get(t->serial);
    if ([holder isKindOfClass:[NvPkg class]]) { id<MTLTexture> old = ((NvPkg *)holder)->tex; ((NvPkg *)holder)->tex = upgraded; defer_obj_release(old); g_ser_gen++;
        if (((NvPkg *)holder)->view) { defer_obj_unref(((NvPkg *)holder)->view); ((NvPkg *)holder)->view = nil; ((NvPkg *)holder)->vbase = 0; } }   // a sparse texture's view of the old one: bound whole until its next upload
    else {
        // Every name of the old texture moves to the upgraded one: other serials sharing its storage (live in g_by_serial, finalized in g_ghost) and the sharing table. Only the asking serial moved before,
        // so the others kept reading the old copy while writes went to the new one, and the old texture stayed in g_alias_set. Slots are replaced in place (no insert, so no table resize during the walk);
        // the old texture is retired once the submissions that used it complete, the view caches follow g_ser_gen (PR #107)
        pthread_mutex_lock(&g_smu);
        nvn_hmap *tables[] = {&g_by_serial, &g_ghost}; int named = 0, shared = 0;
        for (unsigned j = 0; j < 2; j++) for (uint32_t i = 0; i < tables[j]->cap; i++) {
            if (tables[j]->vals[i] != (__bridge void *)base) continue;
            uint64_t serial; memcpy(&serial, tables[j]->keys + (size_t)i * tables[j]->klen, sizeof serial);
            void *old = tables[j]->vals[i]; tables[j]->vals[i] = (void *)CFBridgingRetain(upgraded); CFRelease(old); named |= serial == t->serial;
        }
        for (uint32_t i = 0; i < g_alias.cap; i++) {
            if (g_alias.vals[i] != (__bridge void *)base) continue;
            void *old = g_alias.vals[i]; g_alias.vals[i] = (void *)CFBridgingRetain(upgraded); CFRelease(old); shared = 1;
        }
        if (shared) { CFSetRemoveValue(g_alias_set, (__bridge const void *)base); CFSetAddValue(g_alias_set, (__bridge const void *)upgraded); }
        g_ser_gen++;
        pthread_mutex_unlock(&g_smu);
        if (!named) tex_put(t->serial, upgraded);
        defer_obj_release(base);
    }
    MLOG("texture upgraded to allow %s ser=%llu base format %lu â†’ view format %lu (upgrades=%llu)", add & MTLTextureUsageShaderWrite ? "image writes" : add & MTLTextureUsageRenderTarget ? "render-target use" : "format views", t->serial, (unsigned long)base.pixelFormat, (unsigned long)view_pf, ++g_pfv_upgrades);
    return upgraded;
}
static id<MTLTexture> tex_upgrade_pfv(const tex_t *t, id<MTLTexture> base, MTLPixelFormat view_pf) { return tex_upgrade(t, base, MTLTextureUsagePixelFormatView, view_pf); }
// Packaged texture upload: first take the stamp, then set read-only, then read the pool memory and deswizzle into a staging buffer, and blit into a new private texture.
// The stamp comes first: a write that faults while the pages are being protected gets a generation above it and leaves its page writable, and its bytes may land after
// the read; a stamp taken after the protect covered that generation, so the write and every later one to that page went unseen and the texture stayed stale
// (MacNativeGames a166602, missing letters in Tears of the Kingdom's glyph cache). The same order holds wherever a stamp is taken with a protect.
// Batching: uploads within one replay submission share one upload command buffer (compute encoder deswizzle + a unified blit), flushed before the replay command buffer is committed;
// the same queue executes in commit order â‡’ uploads are always finished before the first sample. Uploads outside a submission are flushed on the spot. The staging area is cut from 32 MB private blocks, held by the command buffer until completion.
static _Atomic uint64_t g_upload_ns;   // Cumulative time of deswizzle + upload encoding on the submit thread (timing instrumentation)
static _Atomic uint64_t g_upload_gpu_ns;
static id<MTL4CommandBuffer> g_up_cb;
static id<MTL4ComputeCommandEncoder> g_up_ce;
static _Atomic int g_up_alloc_busy[3];
// Waiting for upload batches to finish (g_up_pending, g_up_alloc_busy): the batches' completion handler broadcasts after it updates them, so waiters sleep until then (they used to poll every 0.2â€“1 ms;
// Three Houses' loader calls QueueFinish after each resource, each one waited up to a millisecond late). The 10 ms timeout is only a safety net
static pthread_mutex_t g_up_wmu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_up_wcv = PTHREAD_COND_INITIALIZER;
static void up_wake(void) { pthread_mutex_lock(&g_up_wmu); pthread_cond_broadcast(&g_up_wcv); pthread_mutex_unlock(&g_up_wmu); }
static void up_wait_zero(_Atomic int *v, int site) {
    if (!atomic_load(v)) { if (ws_on()) ws_put(site, 0, 0); return; }
    uint64_t w0 = ws_t0();
    pthread_mutex_lock(&g_up_wmu);
    while (atomic_load(v)) { struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl); dl.tv_nsec += 10000000; if (dl.tv_nsec >= 1000000000) { dl.tv_sec++; dl.tv_nsec -= 1000000000; } pthread_cond_timedwait(&g_up_wcv, &g_up_wmu, &dl); }
    pthread_mutex_unlock(&g_up_wmu);
    if (ws_on()) ws_put(site, w0, 1);
}
static unsigned g_up_seq;
#define UP_BLOCK_SIZE (32u * 1024u * 1024u)
// Each upload slot keeps its staging resident until its next batch (up_begin waits for the previous batch). Grow the first block to the slot's previous batch demand, not just its first texture:
// a small first texture followed by a large one otherwise allocates and retires a fresh 32 MB overflow every frame. Keep one overflow as well, at most 64 MB per slot / 192 MB across all three.
// Bursts beyond that budget and individual uploads larger than UP_BLOCK_SIZE still use temporary blocks, retired only after their GPU work. DM_NVN_UPLOAD_REUSE=0 keeps the original one-block policy.
#define UP_SLOT_BLOCK (4u * 1024u * 1024u)
static __strong id<MTLBuffer> g_up_slot_blk[3], g_up_slot_over[3]; static unsigned g_up_cur_slot;
static size_t g_up_slot_want[3], g_up_batch_bytes; static int g_up_over_used;
static int up_reuse(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_NVN_UPLOAD_REUSE"); on = !e || atoi(e); } return on; }
static int up_retained(id<MTLBuffer> b, unsigned slot) { return b && (b == g_up_slot_blk[slot] || b == g_up_slot_over[slot]); }
typedef struct {
    __unsafe_unretained id<MTLBuffer> src;
    size_t off, row, image;
    MTLSize size;
    __unsafe_unretained id<MTLTexture> dst;   // Strongly referenced by g_up_texs until the flush is encoded
    NSUInteger slice, level;
    MTLOrigin org;   // where in the level the copy lands (0 but for a sparse texture's tiles, sparse_upload)
} up_blit;
static up_blit *g_up_blits; static int g_nup_blits, g_up_blitcap;
static NSMutableArray *g_up_texs, *g_up_views;   // Targets and temporary integer-write views kept through their upload
static id<MTLBuffer> g_up_block;
static size_t g_up_block_off, g_up_block_size;
static NSMutableArray *g_up_keep;
// Limit overflow staging per batch. The three upload slots bound in-flight
// batches; up_begin publishes queued commits before waiting for a slot. A single
// texture larger than the budget gets its own block (uploads are not tiled here).
static size_t g_up_staging_bytes;
#define UP_BATCH_BYTES (128u * 1024u * 1024u)


// The upload command buffer was discarded by the system (dropped along during GPU error recovery; the first frame on entering Travincal drops one every run): the objects of this batch of textures are built and the timestamps updated, but the content was never written in,
// and if the pool memory on the game side no longer changes it will never be re-uploaded. Record this batch of target textures, and the next time any of them is fetched treat it as dirty and re-upload once.
// DM_TEST_UP_LOST=<batch index> (for testing): treat that batch of uploads as dropped, to verify the re-upload path.
static NSHashTable *g_up_lost; static os_unfair_lock g_up_lost_lk = OS_UNFAIR_LOCK_INIT; static _Atomic int g_up_lost_n; static _Atomic uint64_t g_up_lost_redo;   // Weak-reference table: if a texture in the record is never fetched again and is released normally, this table does not keep it alive
static int up_lost_take(id tex) {   // This texture's upload was dropped: take it out of the record and return 1
    if (!tex) return 0; os_unfair_lock_lock(&g_up_lost_lk); int hit = [g_up_lost containsObject:tex]; if (hit) { [g_up_lost removeObject:tex]; atomic_store(&g_up_lost_n, (int)g_up_lost.allObjects.count); atomic_fetch_add(&g_up_lost_redo, 1); } os_unfair_lock_unlock(&g_up_lost_lk); return hit;
}
static void up_diag(void);
static void up_flush(void) {
    up_owner_check("up_flush");
    if (!g_up_cb) return;
    if (g_up_ce) {
        m4_end(g_up_ce); g_up_ce = nil;
    }
    id<MTL4ComputeCommandEncoder> b = g_nup_blits ? g_up_ce = [g_up_cb computeCommandEncoder] : nil;
    if (b) [b barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:resource_vis()];
    for (int i = 0; b && i < g_nup_blits; i++) {
        up_blit *x = &g_up_blits[i];
        enc_b_copy_buffer_texture(b, x->src, x->off, x->row, x->image, x->size, x->dst, x->slice, x->level, x->org);
    }
    if (b) { m4_end(b); g_up_ce = nil; }
    id<MTL4CommandBuffer> cb = g_up_cb;
    unsigned slot = (g_up_seq - 1) % 3;
    // Only this upload batch uses its overflow staging. Retire it on upload
    // completion, even when the main rendering submission is still being built.
    NSMutableArray *staging = g_up_keep;
    if (g_up_block && !up_retained(g_up_block, slot)) [staging addObject:g_up_block];
    NSMutableArray *views = g_up_views; g_up_views = nil;
    g_up_keep = nil; g_up_block = nil;
    g_up_block_off = g_up_block_size = g_up_staging_bytes = 0;
    NSMutableSet *targets = [NSMutableSet setWithArray:g_up_texs]; for (int i = 0; i < g_nup_blits; i++) if (g_up_blits[i].dst) [targets addObject:g_up_blits[i].dst];
    NSArray *dsts = targets.allObjects;
    static int lost_at = -2; static uint64_t batch_n; if (lost_at == -2) { const char *e = getenv("DM_TEST_UP_LOST"); lost_at = e ? atoi(e) : -1; } int fake_lost = dsts.count && lost_at > 0 && (int)++batch_n == lost_at;
    MTL4CommitOptions *o = [MTL4CommitOptions new];
    atomic_fetch_add(&g_up_pending, 1);
    [o addFeedbackHandler:^(id<MTL4CommitFeedback> f) {
        if (f.error) MLOG("[RED] upload command buffer execution error: %s", f.error.description.UTF8String);
        if (f.error || fake_lost) {
            os_unfair_lock_lock(&g_up_lost_lk); if (!g_up_lost) g_up_lost = [NSHashTable weakObjectsHashTable]; for (id d in dsts) [g_up_lost addObject:d]; int n = (int)g_up_lost.allObjects.count; atomic_store(&g_up_lost_n, n); os_unfair_lock_unlock(&g_up_lost_lk);
            pthread_mutex_lock(&g_smu); g_ser_gen++; pthread_mutex_unlock(&g_smu);   // Make the view cache of every texture ID take the slow path once (on a cache hit tex_mtl is not passed through and this record is not seen)
            MLOG("%supload command buffer dropped: the %lu textures of this batch are marked unwritten and re-uploaded on next use (%d awaiting re-upload)", fake_lost ? "DM_TEST_UP_LOST: " : "", (unsigned long)dsts.count, n);
        }
        double d = f.GPUEndTime - f.GPUStartTime;
        if (d > 0) atomic_fetch_add(&g_upload_gpu_ns, (uint64_t)(d * 1e9));
        @autoreleasepool {
            for (id<MTLResource> obj in staging) res_remove(obj);
            [staging removeAllObjects];
            [views removeAllObjects];   // upload-only integer views share their targets' residency
            res_commit();
        }
        atomic_store(&g_up_alloc_busy[slot], 0);
        atomic_fetch_sub(&g_up_pending, 1);
        up_wake();
    }];
    res_commit();
    [cb endCommandBuffer];
    rec_commit_upload(cb, o);
    // Outside a submission nothing else hands the encode thread its current block until the next submission, so a batch flushed on the spot stayed a record and every wait for it
    // (nvnQueueFinish on any thread, up_begin on another) waited for that submission: hand it over now (from PR #141, which found the nvnQueueFinish half)
    if (!g_cb) rec_publish();
    if (up_reuse()) g_up_slot_want[slot] = MAX(g_up_slot_want[slot], g_up_batch_bytes);
    g_up_cb = nil;
    up_diag();
    g_nup_blits = 0;
    [g_up_texs removeAllObjects];
}

static void up_begin(void) {
    up_owner_check("up_begin");
    if (g_up_cb) return;
    unsigned slot = g_up_seq++ % 3;
    // The slot's last batch may still be only a commit record in the encode thread's current block (rec_commit_upload), which is handed over when it fills or the submission drains:
    // waiting without handing it over waited for a batch never committed. Little Nightmares II's first render passes zero-filled enough new targets to come round to a slot within one
    // block, and its RHI thread waited for good inside nvnQueueSubmitCommands (q_cb drains before it waits for the same reason)
    if (atomic_load(&g_up_alloc_busy[slot])) rec_publish();
    up_wait_zero(&g_up_alloc_busy[slot], WS_UPALLOC);
    atomic_store(&g_up_alloc_busy[slot], 1); g_up_cur_slot = slot;
    g_up_batch_bytes = 0; g_up_over_used = 0;
    [g_up_alloc4[slot] reset];
    g_up_cb = [g_dev newCommandBuffer];
    [g_up_cb beginCommandBufferWithAllocator:g_up_alloc4[slot]];
    g_up_ce = [g_up_cb computeCommandEncoder];
    [g_up_ce barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:resource_vis()];
    [g_up_ce setArgumentTable:g_up_ct];
    if (!g_up_texs) g_up_texs = [NSMutableArray array];
    if (!g_up_keep) g_up_keep = [NSMutableArray array];
}

// Diagnostic readout (low frequency; printed only when the block switch count changed and â‰¥ 1 second since the last line, at most 200 lines per game): how many times an upload batch ran out of staging block and switched blocks midway; GPU allocated, residency set object count, deferred-release queue length.
// If the old block is not retired on a switch it stays in the residency set and can never be freed, and "GPU allocated" climbs one block at a time with the switch count
static uint64_t g_up_blocks_new, g_up_swaps, g_up_swap_bytes; static NSHashTable *g_up_old;   // g_up_old: old blocks switched out (weak references); the diagnostic line counts how many are still alive
static void up_diag(void) { @autoreleasepool {
    // Weak-table enumeration autoreleases the buffers it finds. Do not keep
    // completed batches alive until the whole rendering submission ends.
    // Print a line when the block switch count changes; then one more line 5 seconds later (that batch's commit is long done and deferred retirement drained, so "GPU allocated" is the settled number)
    static uint64_t said_swaps = ~0ull, last_ns; static int lines, follow; uint64_t now = clock_gettime_nsec_np(CLOCK_MONOTONIC);
    int chg = g_up_swaps != said_swaps, fol = follow && now - last_ns >= 5000000000ull; if ((!chg && !fol) || lines >= 200) return;
    if (chg && last_ns && now - last_ns < 1000000000ull) return; last_ns = now; said_swaps = g_up_swaps; lines++; follow = chg;
    unsigned long alive = 0, inres = 0; uint64_t abytes = 0; NSArray *olds = g_up_old.allObjects;
    pthread_mutex_lock(&g_res_mu); unsigned long nres = (unsigned long)g_res_objects.allObjects.count; for (id<MTLBuffer> o in olds) { alive++; abytes += o.length; inres += [g_res_objects containsObject:o]; } pthread_mutex_unlock(&g_res_mu);
    pthread_mutex_lock(&g_deferred_mu); unsigned long ndef = (unsigned long)g_deferred.count; pthread_mutex_unlock(&g_deferred_mu);
    MLOG("upload staging blocks%s: %llu created in total, swapped within a batch %llu times (%llu MB of old blocks swapped out); swapped-out blocks still alive now %lu, %llu MB, still in the residency set %lu; GPU allocated %llu MB, residency set %lu objects, deferred-release queue %lu", fol && !chg ? " (settled)" : "",
         (unsigned long long)g_up_blocks_new, (unsigned long long)g_up_swaps, (unsigned long long)(g_up_swap_bytes >> 20), alive, (unsigned long long)(abytes >> 20), inres, (unsigned long long)(g_dev.currentAllocatedSize >> 20), nres, ndef);
} }
static id<MTLBuffer> up_alloc(size_t n, size_t *off) {
    size_t need = (n + 255u) & ~((size_t)255u);
    size_t extra = MAX(need, UP_BLOCK_SIZE);
    // This call starts one texture's upload: all prior deswizzles and copies are
    // complete records, so the batch can be split without ending the render pass.
    if (g_up_staging_bytes && (!g_up_block || need > g_up_block_size - g_up_block_off)
        && extra > UP_BATCH_BYTES - MIN(g_up_staging_bytes, UP_BATCH_BYTES)) {
        up_flush();
        up_begin();
    }
    g_up_batch_bytes += MIN(need, UP_BLOCK_SIZE - g_up_batch_bytes);   // demand saturates at one retained block
    if (!g_up_block && need <= UP_BLOCK_SIZE) {   // first allocation of the batch: the slot's own block, grown to the next power of two that holds it (a movie frame re-uploaded every frame, 8 MB in Link's Awakening)
        __strong id<MTLBuffer> *sb = &g_up_slot_blk[g_up_cur_slot];
        size_t demand = up_reuse() ? MAX(need, g_up_slot_want[g_up_cur_slot]) : need;
        size_t want = UP_SLOT_BLOCK; while (want < demand) want <<= 1;
        if (*sb && (*sb).length < want) { defer_obj_release(*sb); *sb = nil; }   // the slot's previous batch has finished (up_begin waited for it)
        if (!*sb) { *sb = [g_dev newBufferWithLength:want options:MTLResourceStorageModePrivate]; if (!*sb) MDIE("failed to create the swizzled-upload staging block %#zx", want); res_add(*sb); g_up_blocks_new++; }
        g_up_block = *sb; g_up_block_size = (*sb).length; g_up_block_off = 0;
        g_up_staging_bytes += (*sb).length;
    }
    if (!g_up_block || g_up_block_off + need > g_up_block_size) {
        // Block switch: the old block is still referenced by copy / decompress commands already recorded in this batch and cannot be freed on the spot; hand it to g_up_keep, to be retired later with the other staging after the flush commits (removed from the residency set on retire).
        // Previously g_up_block was simply replaced with the new block here: nobody retired the old block, yet it stayed in the residency set, so each extra block switch in an upload batch leaked one block (usually 32 MB) of GPU memory.
        // The old block and the "current block" retired at flush are two objects, so nothing is retired twice
        if (g_up_block) { g_up_swaps++; g_up_swap_bytes += g_up_block_size; if (!up_retained(g_up_block, g_up_cur_slot)) { @autoreleasepool { if (!g_up_old) g_up_old = [NSHashTable weakObjectsHashTable]; [g_up_old addObject:g_up_block]; } } static int leak = -1; if (leak < 0) leak = getenv("DM_TEST_UP_LEAK") != NULL;
            if (!leak && !up_retained(g_up_block, g_up_cur_slot)) { if (!g_up_keep) g_up_keep = [NSMutableArray array]; [g_up_keep addObject:g_up_block]; } }
        if (up_reuse() && need <= UP_BLOCK_SIZE && !g_up_over_used) {
            __strong id<MTLBuffer> *sb = &g_up_slot_over[g_up_cur_slot];
            if (!*sb) { *sb = [g_dev newBufferWithLength:UP_BLOCK_SIZE options:MTLResourceStorageModePrivate]; if (!*sb) MDIE("failed to create the overflow staging block"); res_add(*sb); g_up_blocks_new++; }
            g_up_block = *sb; g_up_block_size = (*sb).length; g_up_over_used = 1;
        } else {
            g_up_block_size = n > UP_BLOCK_SIZE ? need : UP_BLOCK_SIZE;
            g_up_block = [g_dev newBufferWithLength:g_up_block_size options:MTLResourceStorageModePrivate];
            res_add(g_up_block); g_up_blocks_new++;
        }
        if (!g_up_block) MDIE("failed to create the swizzled-upload staging block %#zx", g_up_block_size);
        g_up_staging_bytes += g_up_block_size;
        g_up_block_off = 0;
    }
    *off = g_up_block_off;
    g_up_block_off += need;
    return g_up_block;
}

// Deswizzle parameter blocks: 64 bytes per cell, cut sequentially from a 64 KB shared-storage block; a full block is replaced by a new one, and the old block is released lazily by fence
static id<MTLBuffer> g_up_pblock; static size_t g_up_pblock_off;
static id<MTLBuffer> up_param_alloc(size_t *off) {
    if (!g_up_pblock || g_up_pblock_off + 64 > g_up_pblock.length) {
        if (g_up_pblock) defer_obj_release(g_up_pblock);
        g_up_pblock = [g_dev newBufferWithLength:64 << 10 options:MTLResourceStorageModeShared];
        if (!g_up_pblock) MDIE("failed to create the deswizzle parameter block");
        res_add(g_up_pblock); g_up_pblock_off = 0;
    }
    *off = g_up_pblock_off; g_up_pblock_off += 64;
    return g_up_pblock;
}
static void up_blit_add(id<MTLBuffer> src, size_t off, size_t row, size_t image, MTLSize size,
                        id<MTLTexture> dst, NSUInteger slice, NSUInteger level) {
    if (g_nup_blits == g_up_blitcap) {
        g_up_blitcap = g_up_blitcap ? g_up_blitcap * 2 : 256;
        g_up_blits = realloc(g_up_blits, g_up_blitcap * sizeof *g_up_blits);
    }
    g_up_blits[g_nup_blits++] = (up_blit){src, off, row, image, size, dst, slice, level, MTLOriginMake(0, 0, 0)};
}
static void up_blit_at(id<MTLBuffer> src, size_t off, size_t row, size_t image, MTLSize size, id<MTLTexture> dst, NSUInteger slice, NSUInteger level, MTLOrigin org) {
    up_blit_add(src, off, row, image, size, dst, slice, level); g_up_blits[g_nup_blits - 1].org = org;
}

static id<MTLTexture> pkg_upload_(const tex_t *t, uint64_t *stamp, id<MTLTexture> reuse, MTLTextureUsage keep);
static int g_direct_upload_on = -1; static uint64_t g_direct_uploads, g_direct_dispatches;
static int direct_upload_on(void) { if (g_direct_upload_on < 0) { const char *e=getenv("DM_NVN_DIRECT_UPLOAD"); g_direct_upload_on=!e||atoi(e); } return g_direct_upload_on; }
int nvn_host_readable(uint64_t a, uint64_t n);   // hle_nvn.c: guest memory and sparse pools
int nvn_sparse_pool(const void *pool); uint64_t nvn_sparse_gen(void); uint64_t nvn_sparse_copy(const void *pool, void *dst, uint64_t a, uint64_t n);
int nvn_sparse_since(const void *pool, uint64_t a, uint64_t n, uint64_t gen, uint64_t stamp, const void **one); void nvn_sparse_protect(const void *pool, uint64_t a, uint64_t n); int nvn_sparse_mapped(const void *pool, uint64_t a, uint64_t n);
const void *nvn_sparse_phys(const void *pool, uint64_t a, uint64_t n);
// reuse: a texture of the same description, idle on the GPU, to upload into instead of a new one; keep: usage the texture it replaces was upgraded to (tex_upgrade), which the new one
// is made with: a texture drawn into and re-uploaded (a CPU write into one the game renders into) was made without render-target use again and upgraded at its next attachment, a
// second texture and a copy of every level for each re-upload, and its spare (of other usage) was never reused
static id<MTLTexture> pkg_upload(const tex_t *t, uint64_t *stamp, id<MTLTexture> reuse, MTLTextureUsage keep) {
    struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
    id<MTLTexture> r = pkg_upload_(t, stamp, reuse, keep);
    clock_gettime(CLOCK_MONOTONIC, &b); uint64_t d_ = (uint64_t)((b.tv_sec - a.tv_sec) * 1000000000ll + b.tv_nsec - a.tv_nsec);
    atomic_fetch_add(&g_upload_ns, d_); g_up_total_ns += d_;
    return r;
}
uint64_t mtl_upload_ns_take(void) { return atomic_exchange(&g_upload_ns, 0); }
uint64_t mtl_upload_gpu_ns_take(void) { return atomic_exchange(&g_upload_gpu_ns, 0); }
void mtl_create_stats(uint64_t *lib, uint64_t *pso) { *lib = g_bp_lib_n; *pso = g_bp_pso_n; }
void mtl_create_ns(uint64_t *lib_ns, uint64_t *pso_ns) { *lib_ns = g_bp_lib_ns; *pso_ns = g_bp_pso_ns; }   // Cumulative time spent synchronously creating shader libraries / pipelines on the spot (the long-frame log takes the per-frame difference)
uint64_t mtl_upload_bytes(void) { return g_bp_up_bytes; }   // Cumulative texture upload bytes
// Diagnostic DM_NVN_ZBLOCK=1: all-zero 16-byte blocks of each ASTC texture uploaded on the CPU, per level (an all-zero ASTC block decodes as the error colour, magenta). tex_mtl_ logs
// a texture whose bound levels hold any (zb_report), with each level's zero blocks, its total and whether all its pages are mapped (sparse pools)
static int zblock_on(void) { static int v = -1; if (v < 0) { v = getenv("DM_NVN_ZBLOCK") != NULL; if (v) MLOG("DM_NVN_ZBLOCK: counting all-zero blocks of ASTC textures at upload (diagnostic)"); } return v; }
static uint64_t g_zb_ser; static uint32_t g_zb[16], g_zb_tot[16]; static int g_zb_n;
static void zb_report(const tex_t *t, unsigned base, const char *why) {
    if (!zblock_on() || g_zb_ser != t->serial) return;
    int bad = 0; for (int l = (int)base; l < g_zb_n; l++) bad |= g_zb[l] != 0;
    g_zb_ser = 0; if (!bad) return;
    static int nl; if (nl++ >= 400) return;
    char b[512]; size_t o = 0; int sp = t->pool && nvn_sparse_pool(t->pool);
    for (int l = 0; l < g_zb_n && o < sizeof b - 48; l++) { uint64_t lo = nvn_tex_level_offset(&t->d, (unsigned)l), hi = l + 1 < (int)t->d.levels ? nvn_tex_level_offset(&t->d, (unsigned)l + 1) : t->L.size;
        o += (size_t)snprintf(b + o, sizeof b - o, " L%d %u/%u%s", l, g_zb[l], g_zb_tot[l], sp ? (nvn_sparse_mapped(t->pool, (uint64_t)(uintptr_t)t->mem + lo, hi - lo) ? "" : " unmapped") : ""); }
    MLOG("[zb] %s ser=%llu format %#x %ux%u levels %u flags %#x mem=%p size %#llx sparse %d bound from level %u frame %llu:%s", why, (unsigned long long)t->serial, t->d.format, t->d.width, t->d.height, t->d.levels,
         t->d.flags, t->mem, (unsigned long long)t->L.size, sp, base, (unsigned long long)atomic_load(&g_cur_frame), b);
    const char *dir = getenv("DM_NVN_ZBLOCK"); static int nd; static uint64_t seen[16];   // DM_NVN_ZBLOCK=<folder>: also the raw storage of the first such texture of each format and size (up to 16), zb_<serial>.bin (unmapped pages as 0)
    uint64_t kind = (uint64_t)t->d.format << 48 | (uint64_t)t->d.width << 24 | t->d.height; int dup = 0; for (int i = 0; i < nd; i++) dup |= seen[i] == kind;
    if (dir && dir[0] == '/' && nd < 16 && !dup) { seen[nd] = kind; uint64_t n = sp ? MAX(t->L.size, 2u << 20) : t->L.size; uint8_t *m = malloc(n); char fp[1024]; FILE *f;   // a sparse texture: 2 MB, and zbm_<serial>.bin a byte per 4 KB page (1 mapped)
        if (m) { if (sp) nvn_sparse_copy(t->pool, m, (uint64_t)(uintptr_t)t->mem, n); else memcpy(m, t->mem, n);
            snprintf(fp, sizeof fp, "%s/zb_%llu.bin", dir, (unsigned long long)t->serial); if ((f = fopen(fp, "wb"))) { fwrite(m, 1, n, f); fclose(f); nd++; }
            if (sp) { for (uint64_t o = 0; o < n; o += 0x1000) m[o >> 12] = (uint8_t)nvn_sparse_mapped(t->pool, (uint64_t)(uintptr_t)t->mem + o, 0x1000);
                snprintf(fp, sizeof fp, "%s/zbm_%llu.bin", dir, (unsigned long long)t->serial); if ((f = fopen(fp, "wb"))) { fwrite(m, 1, n >> 12, f); fclose(f); } }
            free(m); }
        MLOG("[zb] ser=%llu: block height log2 %d (packaged %d), stride %u, depth %u, target %d, samples %d", (unsigned long long)t->serial, (int)t->L.tileH, t->d.pkg_block_h, t->d.stride, t->d.depth, t->d.target, t->d.samples); }
}
// A packed 16-bit format that Metal has only with red and blue exchanged (RGB5, RGB5A1, RGB565: nvn_fmt_to_mtl's r/b swap): the bit position of red in Metal's format (blue is at bit 0), else 0.
// Every host texture of these formats holds Metal's order, as Metal renders, clears and copies into it; only guest memory holds the guest's (red at bit 0). Bytes are converted where they cross:
// uploads (pkg_upload_), buffer â†” texture copies (rb_swap_gpu), ReadTexels and readbacks into storage (rb_swap_rows). Textures that read guest memory in place (linear and buffer textures, on a
// pool's buffer) take the swap when sampled instead (tex_by_id). Tears of the Kingdom's Depths: smeared rock with red and blue exchanged (GitHub issue #183)
static int rb_packed_shift(int f) {
    uint32_t pf = 0; uint8_t sw[4];
    if (!nvn_fmt_to_mtl(f, &pf, sw) || sw[0] != NVN_SWZ_B) return 0;
    return pf == MTLPixelFormatB5G6R5Unorm ? 11 : pf == MTLPixelFormatBGR5A1Unorm ? 10 : 0;
}
// Exchanges the 5-bit red and blue fields of a region's 16-bit texels in place (rows of wb bytes, row / img apart); green and alpha stay. The same exchange both ways
static void rb_swap_rows(uint8_t *p, size_t wb, size_t rows, size_t nz, size_t row, size_t img, int shift) {
    uint16_t m = (uint16_t)(31u | 31u << shift);
    for (size_t z = 0; z < nz; z++) for (size_t y = 0; y < rows; y++) {
        uint16_t *q = (uint16_t *)(p + z * img + y * row);
        for (size_t i = 0; i < wb / 2; i++) q[i] = (uint16_t)((q[i] & ~m) | (q[i] & 31u) << shift | (q[i] >> shift & 31u));
    }
}
// Diagnostic DM_NVN_POOLLOG: a texture whose Metal allocation is 256 MB or more (which textures make up the GPU allocation)
static void big_tex_log(const tex_t *t, id<MTLTexture> mt, const char *where) {
    static int on = -1; if (on < 0) on = getenv("DM_NVN_POOLLOG") != NULL;
    if (!on || !mt || mt.allocatedSize < (256u << 20)) return;
    if (t->pool && nvn_sparse_pool(t->pool)) {   // how much of a sparse one the game maps now
        uint64_t m = 0, n = 0; for (uint64_t o = 0; o < t->L.size; o += 0x10000, n++) m += nvn_sparse_mapped(t->pool, (uint64_t)(uintptr_t)t->mem + o, 0x10000) != 0;
        MLOG("big texture ser=%llu: %llu of %llu 64 KB pages mapped (%llu MB)", (unsigned long long)t->serial, (unsigned long long)m, (unsigned long long)n, (unsigned long long)(m >> 4));
    }
    MLOG("big texture (%s) ser=%llu %ux%ux%u format %#x target %d levels %u flags %#x: %llu MB, GPU allocated %llu MB; pool %s%s, storage %p + %#llx", where, (unsigned long long)t->serial, t->d.width, t->d.height, t->d.depth,
         t->d.format, t->d.target, (unsigned)t->d.levels, (unsigned)t->d.flags, (unsigned long long)(mt.allocatedSize >> 20), (unsigned long long)(g_dev.currentAllocatedSize >> 20),
         t->pool ? "yes" : "none", t->pool && nvn_sparse_pool(t->pool) ? " (sparse)" : "", t->mem, (unsigned long long)t->off);
}
// ---------- Sparse textures as Metal placement sparse textures ----------
// A texture in a sparse virtual pool holds only the pages the game maps into it, and its virtual size can be far beyond what it uses: Mario Kart World's terrain caches are three
// 8192Ã—8192 arrays of 68 layers with 12 levels (1.45 GB, 5.8 GB, 1.45 GB), filled from physical pools of 4.7 GB in all. Made as plain Metal textures, they allocated 8.7 GB at the first
// loading screen and the GPU ran out of memory (kIOGPUCommandBufferCallbackErrorOutOfMemory, the game froze), after a CPU gather of the whole texture as large again.
// When NVN's 64 KB sparse tile and Metal's tile at a 64 KB page are the same rectangle of texels, and NVN's tiled levels are Metal's levels before its mip tail (both checked here;
// they hold for both of Mario Kart World's formats on an M5), each 64 KB page of the texture's storage is one tile of the Metal texture: a placement sparse texture (no memory of its
// own) maps a tile of a placement heap for each page the game maps, unmaps it when the game unmaps the page, and uploads only the tiles whose pages were mapped or written since the last
// upload, each deswizzled on its own as a one-block-row 2D texture. The mip tail (a few small levels) is always mapped and uploaded as a whole. Any other texture keeps the dense path.
// Tiles come from placement heaps (sp_alloc), given back to their heap when the texture is released (after the GPU is done with it). DM_NVN_SPARSE_TEX=0 turns the path off
// Heaps of 64 MB (1,024 tiles; a tile reference keeps 12 bits for the tile); a heap whose tiles are all free goes back to the system while the others have room for half a heap,
// and its slot makes a new heap when tiles are needed again. With heaps of 256 MB that were never given back, Mario Kart World held 9 (2.3 GB) for 1.7 GB of mapped tiles
enum { SP_HEAP_TILES = 1024, SP_TILE = 65536, SP_BATCH = 256u << 20 };
typedef struct { __unsafe_unretained id<MTLHeap> heap; uint32_t nfree, *free; } sp_heap;
static sp_heap *g_sph; static int g_nsph; static NSMutableArray *g_sph_keep; static os_unfair_lock g_sp_lk = OS_UNFAIR_LOCK_INIT;
static uint64_t g_sp_mapped;   // tiles mapped now (under g_sp_lk)
static id<MTLHeap> sp_heap_new(void) {
    MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
    hd.type = MTLHeapTypePlacement; hd.storageMode = MTLStorageModePrivate; hd.size = (NSUInteger)SP_HEAP_TILES * SP_TILE; hd.maxCompatiblePlacementSparsePageSize = MTLSparsePageSize64;
    id<MTLHeap> heap = [g_dev newHeapWithDescriptor:hd];
    if (!heap) { MLOG("[RED] sparse textures: no placement heap of %u MB", SP_HEAP_TILES * SP_TILE >> 20); return nil; }
    if (!g_sph_keep) g_sph_keep = [NSMutableArray array];
    [g_sph_keep addObject:heap]; res_add(heap); res_commit();
    return heap;
}
static int sp_alloc(uint32_t *ref) {   // a tile of a placement heap: ref = (heap << 12 | tile) + 1; caller holds g_sp_lk
    for (int h = 0; h < g_nsph; h++) if (g_sph[h].heap && g_sph[h].nfree) { *ref = ((uint32_t)h << 12 | g_sph[h].free[--g_sph[h].nfree]) + 1; g_sp_mapped++; return 1; }
    for (int h = 0; h < g_nsph; h++) if (!g_sph[h].heap) {   // a slot whose heap went back: a new heap there (its tiles are all free)
        if (!(g_sph[h].heap = sp_heap_new())) return 0;
        MLOG("sparse textures: placement heap slot %d made again (%llu MB of tiles mapped)", h + 1, (unsigned long long)(g_sp_mapped * SP_TILE >> 20));
        return sp_alloc(ref);
    }
    id<MTLHeap> heap = sp_heap_new(); if (!heap) return 0;
    g_sph = realloc(g_sph, (size_t)(g_nsph + 1) * sizeof *g_sph);
    sp_heap *s = &g_sph[g_nsph++]; s->heap = heap; s->free = malloc(SP_HEAP_TILES * sizeof *s->free); s->nfree = 0;
    for (uint32_t i = SP_HEAP_TILES; i-- > 0; ) s->free[s->nfree++] = i;
    MLOG("sparse textures: placement heap #%d of %u MB (%llu MB of tiles mapped)", g_nsph, SP_HEAP_TILES * SP_TILE >> 20, (unsigned long long)(g_sp_mapped * SP_TILE >> 20));
    return sp_alloc(ref);
}
static void sp_free(uint32_t ref) {   // caller holds g_sp_lk
    ref--; sp_heap *s = &g_sph[ref >> 12]; s->free[s->nfree++] = ref & 0xfff; g_sp_mapped--;
    if (s->nfree < SP_HEAP_TILES || !s->heap) return;
    uint64_t room = 0; for (int h = 0; h < g_nsph; h++) if (&g_sph[h] != s && g_sph[h].heap) room += g_sph[h].nfree;
    if (room < SP_HEAP_TILES / 2) return;   // keep it: the others are nearly full
    id<MTLHeap> heap = s->heap; s->heap = nil;
    [g_sph_keep removeObjectIdenticalTo:heap]; defer_obj_release(heap);   // released once the submissions that may still read its tiles are done
    static unsigned n; if (n++ < 8 || !(n & (n - 1))) MLOG("sparse textures: an empty placement heap given back (%u so far; %llu MB of tiles mapped)", n, (unsigned long long)(g_sp_mapped * SP_TILE >> 20));
}

// A GPU-written sparse texture's tiles follow its physical pages, as on hardware, where the content is the physical page's: one tile per 64 KB physical page, shared by every place
// and texture the game maps that page into (placement heap tiles may alias), counted in g_sp_phys (physical address â†’ tile reference | count << 32). Mario Kart World's terrain
// index maps come in pairs over the same physical pages: a tile per texture and place left what the GPU drew through one mapping invisible through the other, and the ground flat
// With the profile's tex_release an uploaded texture's tiles follow their physical pages too (the key adds the format and the block height, which decide what the bytes deswizzle to),
// with the dirty-page time of their last upload: another place the page is mapped takes the tile as it is, and a tile is uploaded again only when the game writes the page. The page
// itself is then given back (nvn_dirty_release): its bytes are the tile's, and no upload reads them again until the game writes it
typedef struct { uint32_t ref, n, gone; uint64_t stamp; } sp_ent;   // stamp: dirty-page time of the tile's last upload, 0 = not uploaded (a GPU-written one never is); gone: the page was given back, so the tile is all that holds its bytes
static nvn_hmap g_sp_phys = {.klen = 8};   // key (sp_key) â†’ sp_ent
static uint64_t g_sp_shared;   // mappings that took the tile of a physical page already mapped elsewhere
static uint64_t g_sp_released;   // bytes of uploaded tiles' pages given back (nvn_dirty_release)
int mtl_pool_wrapped(const void *addr, uint64_t len);
static uint64_t sp_key(const void *phys, int rt, int format, int bh) { return rt ? (uint64_t)(uintptr_t)phys | 1ull << 63 : (uint64_t)(uintptr_t)phys | (uint64_t)(format & 0xff) << 48 | (uint64_t)(bh & 0x7f) << 56; }
static sp_ent *sp_phys_ref(uint64_t key, uint32_t *ref) {   // caller holds g_sp_lk
    sp_ent *e = nvn_hm_get(&g_sp_phys, &key);
    if (e) { *ref = e->ref; e->n++; g_sp_shared++; return e; }
    if (!sp_alloc(ref)) return NULL;
    e = calloc(1, sizeof *e); if (!e) MDIE("sparse tile table allocation failed");
    e->ref = *ref; e->n = 1; nvn_hm_put(&g_sp_phys, &key, e); return e;
}
static void sp_phys_unref(uint64_t key, uint32_t ref) {   // caller holds g_sp_lk
    sp_ent *e = nvn_hm_get(&g_sp_phys, &key);
    if (!e || e->ref != ref) { sp_free(ref); return; }
    if (--e->n) return;
    if (e->gone) return;   // mapped again later, the page has nothing to upload from: the system may have taken it (on an iPhone it does: textures of zeros, black or missing). The tile stays, in place of the page, until the game writes the page
    nvn_hm_del(&g_sp_phys, &key); free(e); sp_free(ref);
}
@interface NvSparse : NSObject { @public uint32_t rt, layers, ntiled, tw, th, ntail, tx[16], ty[16], lw[16], lh[16], bh_log[16]; uint64_t ntile, lofs[16], tail_off, base[16], gen, stamp; uint32_t *map, *tail; uint64_t *phys; } @end   // phys: per tile, the sp_key of the physical page whose tile it maps (0: a tile of its own)
@implementation NvSparse
- (void)dealloc {   // the texture is gone, so is the GPU's use of its tiles (it was released through deferred release): its tiles go back to their heaps
    os_unfair_lock_lock(&g_sp_lk);
    for (uint64_t i = 0; map && i < ntile; i++) if (map[i]) { if (phys && phys[i]) sp_phys_unref(phys[i], map[i]); else sp_free(map[i]); }   // no tables: given up before it was made (the dense path made the texture)
    for (uint64_t i = 0; tail && i < (uint64_t)layers * ntail; i++) if (tail[i]) sp_free(tail[i]);
    os_unfair_lock_unlock(&g_sp_lk);
    free(map); free(tail); free(phys);
}
@end
static char g_sp_key; static NSMapTable *g_sp_tex; static int g_sp_rt_n;   // g_sp_rt_n: sparse targets made (a texture lookup checks for one only once there is any)   // texture serial â†’ its placement sparse texture (weak: it lives as long as the replay holds it)

// Mapping operations gathered per heap and queued before the upload batch that writes the tiles (rec_tex_map)
typedef struct { NSMutableData *ops[64]; NSMutableData *unmap; } sp_ops;
static void sp_op(sp_ops *o, int heap, MTLSparseTextureMappingMode mode, MTLRegion r, NSUInteger level, NSUInteger slice, NSUInteger tile) {
    MTL4UpdateSparseTextureMappingOperation op = {mode, r, level, slice, tile};
    NSMutableData * __strong *d = mode == MTLSparseTextureMappingModeUnmap ? &o->unmap : &o->ops[heap & 63];
    if (!*d) *d = [NSMutableData data];
    [*d appendBytes:&op length:sizeof op];
}
static void sp_emit(sp_ops *o, id<MTLTexture> mt) {
    if (o->unmap) { rec_tex_map(mt, nil, o->unmap); o->unmap = nil; }
    for (int h = 0; h < 64; h++) if (o->ops[h]) {   // heap index h, h + 64, ...: the operations of each heap kept apart by the heap they name
        NSMutableData *all = o->ops[h]; o->ops[h] = nil;
        const MTL4UpdateSparseTextureMappingOperation *p = all.bytes; NSUInteger n = all.length / sizeof *p;
        for (int k = h; k < g_nsph; k += 64) {
            NSMutableData *mine = [NSMutableData data];
            for (NSUInteger i = 0; i < n; i++) if ((int)(p[i].heapOffset >> 12) == k) { MTL4UpdateSparseTextureMappingOperation q = p[i]; q.heapOffset &= 0xfff; [mine appendBytes:&q length:sizeof q]; }
            if (mine.length) rec_tex_map(mt, g_sph[k].heap, mine);
        }
    }
}

// CPU-written staging for the tiles (up_alloc's blocks are the GPU deswizzler's, private): shared chunks kept by the upload batch until its GPU work is done (g_up_keep, as the dense
// CPU path's); a chunk belongs to one batch, so it is dropped at every flush
static id<MTLBuffer> g_sp_stg; static size_t g_sp_stg_off;
static id<MTLBuffer> sp_stage(size_t n, size_t *off) {
    if (!g_sp_stg || g_sp_stg_off + n > g_sp_stg.length) {
        g_sp_stg = [g_dev newBufferWithLength:MAX(n, (size_t)32 << 20) options:MTLResourceStorageModeShared];
        if (!g_sp_stg) MDIE("sparse texture staging allocation failed");
        res_add(g_sp_stg); [g_up_keep addObject:g_sp_stg]; g_sp_stg_off = 0;
    }
    *off = g_sp_stg_off; g_sp_stg_off += (n + 255) & ~(size_t)255;
    return g_sp_stg;
}

// â†’ the texture, or nil when it does not qualify (the dense path then makes it). rt: a texture the GPU writes (a render target or image), made by tex_mtl_ rather than uploaded:
// its tiles are mapped and unmapped as the game maps its pages, and nothing is uploaded into them (what the GPU wrote is the content). Mario Kart World's terrain targets are
// four 16384x16384 R8 and six 8192x8192 targets in sparse pools (3.2 GB if dense), of which the game maps 0 to 2 MB each at creation
static id<MTLTexture> sparse_upload(const tex_t *t, uint64_t *stamp, MTLTextureUsage keep, int rt) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_NVN_SPARSE_TEX"); on = !e || atoi(e); }
    if (!on || !t->pool || !nvn_sparse_pool(t->pool) || t->L.size < (64u << 20) || (t->d.target != 1 && t->d.target != 4) || t->d.samples > 1 || !g_dev.supportsPlacementSparse
        || rb_packed_shift(t->d.format)) return nil;
    uint32_t tw, th, td; int ntiled = nvn_tex_sparse_tiles(&t->d, &tw, &th, &td);
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    if (ntiled <= 0 || ntiled > 16 || td != 1 || !bpb || tw % bw || th % bh) return nil;
    if (!g_sp_tex) g_sp_tex = [NSMapTable strongToWeakObjectsMapTable];
    NSNumber *key = @(t->serial);
    id<MTLTexture> mt = [g_sp_tex objectForKey:key]; NvSparse *sp = mt ? objc_getAssociatedObject(mt, &g_sp_key) : nil;
    unsigned levels = t->d.levels ? t->d.levels : 1;
    if (!sp) {
        MTLTextureDescriptor *d = tex_desc(t, !rt);
        if (!d) return nil;
        d.usage |= keep & (MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView);
        d.storageMode = MTLStorageModePrivate; d.placementSparsePageSize = MTLSparsePageSize64;
        MTLSize ms = [g_dev sparseTileSizeWithTextureType:d.textureType pixelFormat:d.pixelFormat sampleCount:1 sparsePageSize:MTLSparsePageSize64];
        static int said;
        if (ms.width != tw || ms.height != th || ms.depth != 1 || [g_dev sparseTileSizeInBytesForSparsePageSize:MTLSparsePageSize64] != SP_TILE) {
            if (!said++) MLOG("[RED] sparse texture ser=%llu format %#x: NVN's tile %ux%u is not Metal's %lux%lu; made dense", (unsigned long long)t->serial, t->d.format, tw, th, (unsigned long)ms.width, (unsigned long)ms.height);
            return nil;
        }
        sp = [NvSparse new]; sp->rt = (uint32_t)rt; g_sp_rt_n += rt;
        sp->layers = t->d.target == 4 ? t->d.depth : 1; sp->ntiled = (uint32_t)ntiled; sp->tw = tw; sp->th = th;
        for (int l = 0; l < ntiled; l++) {
            sp->lw[l] = MAX(t->d.width >> l, 1u); sp->lh[l] = MAX(t->d.height >> l, 1u);
            sp->tx[l] = (sp->lw[l] + tw - 1) / tw; sp->ty[l] = (sp->lh[l] + th - 1) / th;
            sp->lofs[l] = nvn_tex_level_offset(&t->d, (unsigned)l);
            unsigned tH, tD; nvn_tex_level_tiles(&t->d, (unsigned)l, &tH, &tD); sp->bh_log[l] = tH;
            nvn_texdesc s = {.target = 1, .format = t->d.format, .width = tw, .height = th, .depth = 1, .levels = 1, .pkg_block_h = (int)tH + 1};
            nvn_texlayout sl; nvn_tex_layout(&s, &sl);
            if (sp->lofs[l] % SP_TILE || (uint64_t)nvn_tex_level_gobs(&t->d, (unsigned)l) * 64 != (uint64_t)sp->tx[l] * (tw / bw) * bpb || sl.size != SP_TILE || tD) {
                if (!said++) MLOG("[RED] sparse texture ser=%llu format %#x level %d: its tiles are not one 64 KB page each (offset %#llx, %u GOBs, tile %llu bytes); made dense",
                                  (unsigned long long)t->serial, t->d.format, l, (unsigned long long)sp->lofs[l], nvn_tex_level_gobs(&t->d, (unsigned)l), (unsigned long long)sl.size);
                return nil;
            }
            sp->base[l] = sp->ntile; sp->ntile += (uint64_t)sp->layers * sp->tx[l] * sp->ty[l];
        }
        mt = [g_dev newTextureWithDescriptor:d];
        // NVN's tiled levels are Metal's levels before its tail; with every level tiled Metal names a tail level past the last one (7 for an 8192x8192 RGBA8 target of 5 levels)
        if (!mt || ((unsigned)ntiled < levels ? mt.firstMipmapInTail != (NSUInteger)ntiled : mt.firstMipmapInTail < (NSUInteger)levels)) {
            if (!said++) MLOG("[RED] sparse texture ser=%llu: Metal's mip tail starts at level %lu, NVN's at %d; made dense", (unsigned long long)t->serial, (unsigned long)mt.firstMipmapInTail, ntiled);
            return nil;
        }
        sp->ntail = (unsigned)ntiled < levels ? (uint32_t)((mt.tailSizeInBytes + SP_TILE - 1) / SP_TILE) : 0;
        sp->tail_off = (unsigned)ntiled < levels ? nvn_tex_level_offset(&t->d, (unsigned)ntiled) : t->L.layer_size;
        sp->map = calloc(sp->ntile ? sp->ntile : 1, sizeof *sp->map); sp->tail = calloc((size_t)sp->layers * (sp->ntail ? sp->ntail : 1), sizeof *sp->tail);
        if (!sp->map || !sp->tail) MDIE("sparse texture state allocation failed");
        objc_setAssociatedObject(mt, &g_sp_key, sp, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        [g_sp_tex setObject:mt forKey:key];
        res_add(mt);
        MLOG("sparse texture ser=%llu %ux%ux%u format %#x, %u levels: a placement sparse texture of %llu tiles of %ux%u, mapped as the game maps its pages (%llu MB if dense)",
             (unsigned long long)t->serial, t->d.width, t->d.height, sp->layers, t->d.format, levels, (unsigned long long)sp->ntile, tw, th, (unsigned long long)(t->L.size >> 20));
    }
    uint64_t now = nvn_dirty_now(), gen = nvn_sparse_gen();
    if (stamp) { *stamp = now; nvn_sparse_protect(t->pool, (uint64_t)(uintptr_t)t->mem, t->L.size); }
    sp_ops ops = {0};
    uint8_t *page = malloc(SP_TILE), *tailbuf = NULL; uint64_t staged = 0, uploads = 0;
    if (!page) MDIE("sparse texture tile buffer allocation failed");
    up_begin(); g_sp_stg = nil;
    for (uint32_t y = 0; y < sp->layers; y++) {
        uint64_t lay = (uint64_t)(uintptr_t)t->mem + (uint64_t)y * t->L.layer_size;
        for (uint32_t l = 0; l < sp->ntiled; l++) {
            nvn_texdesc s = {.target = 1, .format = t->d.format, .width = sp->tw, .height = sp->th, .depth = 1, .levels = 1, .pkg_block_h = (int)sp->bh_log[l] + 1};
            size_t pitch = (size_t)(sp->tw / bw) * bpb, rows = sp->th / bh;
            for (uint32_t ty = 0; ty < sp->ty[l]; ty++) for (uint32_t tx = 0; tx < sp->tx[l]; tx++) {
                uint64_t k = (uint64_t)ty * sp->tx[l] + tx, a = lay + sp->lofs[l] + k * SP_TILE;
                uint64_t mi = sp->base[l] + ((uint64_t)y * sp->ty[l] + ty) * sp->tx[l] + tx; uint32_t *m = &sp->map[mi];
                int want = nvn_sparse_mapped(t->pool, a, SP_TILE), fresh = 0;
                MTLRegion r = MTLRegionMake3D(tx, ty, 0, 1, 1, 1);
                int share = sp->rt || DM_GAME_TEX_RELEASE;   // tiles that follow their physical pages (sp_ent)
                if (share && !sp->phys && !(sp->phys = calloc(sp->ntile ? sp->ntile : 1, sizeof *sp->phys))) MDIE("sparse texture state allocation failed");
                const void *pp = share && want ? nvn_sparse_phys(t->pool, a, SP_TILE) : NULL;
                uint64_t ph = pp ? sp_key(pp, (int)sp->rt, t->d.format, (int)sp->bh_log[l]) : 0;
                if (*m && (!want || (sp->phys && sp->phys[mi] != ph))) {   // unmapped, or another physical page at this place: its tile goes
                    sp_op(&ops, (int)((*m - 1) >> 12), MTLSparseTextureMappingModeUnmap, r, l, y, 0);
                    os_unfair_lock_lock(&g_sp_lk); if (sp->phys && sp->phys[mi]) sp_phys_unref(sp->phys[mi], *m); else sp_free(*m); os_unfair_lock_unlock(&g_sp_lk);
                    *m = 0; if (sp->phys) sp->phys[mi] = 0;
                }
                if (!want) continue;
                sp_ent *se = NULL;
                if (!*m) {
                    os_unfair_lock_lock(&g_sp_lk); int ok = ph ? (se = sp_phys_ref(ph, m)) != NULL : sp_alloc(m); os_unfair_lock_unlock(&g_sp_lk);
                    if (!ok) continue;
                    if (sp->phys) sp->phys[mi] = ph;
                    sp_op(&ops, (int)((*m - 1) >> 12), MTLSparseTextureMappingModeMap, r, l, y, *m - 1); fresh = 1;
                } else if (ph) { os_unfair_lock_lock(&g_sp_lk); se = nvn_hm_get(&g_sp_phys, &ph); os_unfair_lock_unlock(&g_sp_lk); }
                if (sp->rt) continue;
                if (se) { if (se->stamp && !nvn_dirty_since(pp, SP_TILE, se->stamp)) continue; }   // the page's tile is as the game last wrote it
                else if (!fresh && sp->gen && !nvn_sparse_since(t->pool, a, SP_TILE, sp->gen, sp->stamp, NULL)) continue;
                nvn_sparse_copy(t->pool, page, a, SP_TILE);
                size_t so; id<MTLBuffer> stg = sp_stage(pitch * rows, &so);
                if (nvn_deswizzle(&s, 0, 0, page, (uint8_t *)stg.contents + so, pitch)) { RED_ONCE(9, "deswizzle failed format %#x target %d", t->d.format, t->d.target); continue; }
                up_blit_at(stg, so, pitch, pitch * rows, MTLSizeMake(MIN(sp->tw, sp->lw[l] - tx * sp->tw), MIN(sp->th, sp->lh[l] - ty * sp->th), 1), mt, y, l, MTLOriginMake(tx * sp->tw, ty * sp->th, 0));
                staged += pitch * rows; uploads++;
                if (se && stamp) {   // the tile now holds the page: its time, and the page's memory goes back (the bytes are in the tile; no pool buffer may hold the page)
                    se->stamp = now;
                    if (!mtl_pool_wrapped(pp, SP_TILE) && !mtl_pool_wrapped((const void *)(uintptr_t)a, SP_TILE)) { size_t got = nvn_dirty_release(pp, SP_TILE, page); g_sp_released += got; if (got) se->gone = 1; }
                }
                if (staged >= SP_BATCH) { sp_emit(&ops, mt); [g_up_texs addObject:mt]; up_flush(); up_begin(); g_sp_stg = nil; staged = 0; }   // a first upload of many tiles goes out in batches
            }
        }
        if (!sp->ntail) continue;
        int fresh = 0;
        for (uint32_t i = 0; i < sp->ntail; i++) {
            uint32_t *m = &sp->tail[(size_t)y * sp->ntail + i];
            if (*m) continue;
            os_unfair_lock_lock(&g_sp_lk); int ok = sp_alloc(m); os_unfair_lock_unlock(&g_sp_lk);
            if (!ok) continue;
            sp_op(&ops, (int)((*m - 1) >> 12), MTLSparseTextureMappingModeMap, MTLRegionMake3D(i, 0, 0, 1, 1, 1), sp->ntiled, y, *m - 1); fresh = 1;
        }
        uint64_t ta = lay + sp->tail_off, tn = t->L.layer_size - sp->tail_off;
        if (sp->rt) continue;
        if (!fresh && sp->gen && !nvn_sparse_since(t->pool, ta, tn, sp->gen, sp->stamp, NULL)) continue;
        if (!tailbuf && !(tailbuf = malloc(tn ? tn : 1))) MDIE("sparse texture mip tail buffer allocation failed");
        nvn_sparse_copy(t->pool, tailbuf, ta, tn);
        const uint8_t *src = tailbuf - ((uint64_t)y * t->L.layer_size + sp->tail_off);   // nvn_deswizzle addresses the texture's storage: only the tail's bytes are read
        for (unsigned l = sp->ntiled; l < levels; l++) {
            uint32_t w = MAX(t->d.width >> l, 1u), h = MAX(t->d.height >> l, 1u);
            size_t pitch = (size_t)((w + bw - 1) / bw) * bpb, rows = (h + bh - 1) / bh, so;
            id<MTLBuffer> stg = sp_stage(pitch * rows, &so);
            if (nvn_deswizzle(&t->d, l, y, src, (uint8_t *)stg.contents + so, pitch)) { RED_ONCE(9, "deswizzle failed format %#x target %d", t->d.format, t->d.target); break; }
            up_blit_add(stg, so, pitch, pitch * rows, MTLSizeMake(w, h, 1), mt, y, l);
        }
    }
    sp_emit(&ops, mt);
    free(page); free(tailbuf); g_sp_stg = nil;
    sp->gen = gen; sp->stamp = now;
    [g_up_texs addObject:mt];
    if (!g_cb) up_flush();
    static uint64_t logged; if (uploads && logged++ < 20) { os_unfair_lock_lock(&g_sp_lk); uint64_t mapped = g_sp_mapped; os_unfair_lock_unlock(&g_sp_lk);
        MLOG("sparse texture ser=%llu: %llu tiles uploaded; %llu tiles (%llu MB) mapped in all sparse textures, %llu mappings so far shared a physical page's tile, %llu MB of their pages given back", (unsigned long long)t->serial, (unsigned long long)uploads, (unsigned long long)mapped, (unsigned long long)(mapped * SP_TILE >> 20), (unsigned long long)g_sp_shared, (unsigned long long)(g_sp_released >> 20)); }
    return mt;
}

// Diagnose the physical tile-geometry problem before attempting any BC ->
// ASTC sparse mapping. Matching compressed bytes per 4x4 block is insufficient:
// sparse mapping coordinates are counted in *Metal's* tiles, not NVN's pages.
// We must re-tile converted blocks when these footprints differ. This probe
// intentionally does not allocate a texture or enable a broken fallback.
static void bc_astc_sparse_probe(const tex_t *t, MTLPixelFormat guest_pf) {
    static unsigned logged;
    if (logged++ >= 12 || !g_dev.supportsPlacementSparse) return;
    uint32_t nw = 0, nh = 0, nd = 0;
    int ntiled = nvn_tex_sparse_tiles(&t->d, &nw, &nh, &nd);
    MTLPixelFormat astc = bc_rgba_srgb(guest_pf) ? MTLPixelFormatASTC_4x4_sRGB : MTLPixelFormatASTC_4x4_LDR;
    MTLSize metal = [g_dev sparseTileSizeWithTextureType:(t->d.target == 4 ? MTLTextureType2DArray : MTLTextureType2D)
                                             pixelFormat:astc sampleCount:1 sparsePageSize:MTLSparsePageSize64];
    MLOG("[RED] A16 sparse BC -> ASTC pending: ser=%llu format=%#x NVN tile=%ux%ux%u Metal ASTC 4x4 tile=%lux%lux%lu (tiled levels=%d); mapping needs tile-layout translation",
         (unsigned long long)t->serial, t->d.format, nw, nh, nd,
         (unsigned long)metal.width, (unsigned long)metal.height,
         (unsigned long)metal.depth, ntiled);
}

static id<MTLTexture> pkg_upload_(const tex_t *t, uint64_t *stamp, id<MTLTexture> reuse, MTLTextureUsage keep) {
    uint32_t guest_pf = 0; uint8_t guest_swz[4]; int bc_signed = 0;
    nvn_fmt_to_mtl(t->d.format, &guest_pf, guest_swz);
    int bc_kind = bc_cpu_fallback() ? bc_rgba_kind((MTLPixelFormat)guest_pf, &bc_signed) : 0;
    if (bc_kind && t->d.target == 2) {
        RED_ONCE(160, "BC fallback does not support 3D textures (format %#x)", t->d.format);
        return nil;
    }
    // A sparse BC texture needs a per-tile decode/cache architecture. Do not
    // accidentally allocate it as RGBA8: a virtual 8192x8192x68 array would
    // take tens of gigabytes and kill the app before a useful log was written.
    if (bc_kind && t->pool && nvn_sparse_pool(t->pool)) {
        bc_astc_sparse_probe(t, (MTLPixelFormat)guest_pf);
        RED_ONCE(159, "BC fallback has no sparse texture support (format %#x, %ux%ux%u)", t->d.format, t->d.width, t->d.height, t->d.depth);
        return nil;
    }
    { id<MTLTexture> s = sparse_upload(t, stamp, keep, 0); if (s) return s; }   // a large texture of a sparse pool: only its mapped tiles take memory
    MTLTextureDescriptor *d = tex_desc(t, 1);
    if (!d) return nil;
    d.usage |= keep & (MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView);
    int rb_shift = rb_packed_shift(t->d.format);   // the RGB565 family is repacked into Metal's order on the CPU (rb_packed_shift)
    // On the CPU while its pool has no zero-copy buffer: the GPU's upload would give it one (mtl_addr), and iOS reads a buffer's whole pool into memory and wires it, for a pool the GPU may otherwise never address
    int keep_cpu = (pst_eligible(t) || DM_GAME_TEX_SOFT) && t->has_layout && !mtl_pool_wrapped(t->mem, t->L.size);
    unsigned sk = tex_soft(t) ? 1 : 0;   // source level = host level + sk
    int direct = !bc_kind && !sk && !keep_cpu && !rb_shift && stamp && direct_upload_on() && g_nvn_deswz_gpu && !(t->pool && nvn_sparse_pool(t->pool)) && nvn_deswz_gpu_direct_supported(&t->d);
    if (direct) d.usage |= MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView;
    id<MTLTexture> mt = reuse && reuse.pixelFormat == d.pixelFormat && reuse.width == d.width && reuse.height == d.height && reuse.mipmapLevelCount == d.mipmapLevelCount && reuse.textureType == d.textureType
                        && (reuse.usage & d.usage) == d.usage
                        ? reuse : [g_dev newTextureWithDescriptor:d];
    big_tex_log(t, mt, "package upload");
    if (!mt) { RED_ONCE(2, "failed to create MTLTexture %ux%u format %#x target %d", t->d.width, t->d.height, t->d.format, t->d.target); return nil; }
    if (g_allocprof > 0) { static int n; if (n++ < 30 || n % 500 == 0) MLOG("new texture (%s) #%d ser=%llu %ux%ux%u format %#x target %d levels %u storage %p", __func__, n, (unsigned long long)t->serial, t->d.width, t->d.height, t->d.depth, t->d.format, t->d.target, t->d.levels, (void *)t->mem); }
    res_add(mt);   // Both first upload and re-upload register residency here (Metal 4 does not make resources resident automatically; a missed registration = GPU address page fault)
    int sparse = t->pool && nvn_sparse_pool(t->pool);
    if (stamp) {   // no stamp: a one-time upload of content the CPU will not change (tex_init_fill), with no write tracking
        *stamp = nvn_dirty_now();
        if (sparse) nvn_sparse_protect(t->pool, (uint64_t)(uintptr_t)t->mem, t->L.size); else nvn_dirty_protect(t->mem, t->L.size); }
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    unsigned layers = (t->d.target == 4 || t->d.target == 9) ? t->d.depth : t->d.target == 8 ? 6 : t->d.target == 3 ? t->d.height : 1;   // a cube array's depth counts layer-faces (tex_desc)
    size_t total = 0;
    for (unsigned l = 0; l < d.mipmapLevelCount; l++) {
        uint32_t w = MAX(t->d.width >> (l + sk), 1u), h = MAX(t->d.height >> (l + sk), 1u), z = t->d.target == 2 ? MAX(t->d.depth >> l, 1u) : 1;
        total += (bc_kind ? (size_t)w * h * 4 : (size_t)((w + bw - 1) / bw) * bpb * ((h + bh - 1) / bh)) * z * layers;
    }
    g_bp_up_bytes += total;
    void *mb; uint64_t mo;
    int use_gpu = !bc_kind && !keep_cpu && !rb_shift && !sparse && g_nvn_deswz_gpu && mtl_addr((uint64_t)t->mem, t->L.size, &mb, &mo);   // a sparse pool's texture is gathered through its page table on the CPU
    if (g_nvn_deswz_gpu && !use_gpu && !sparse && !rb_shift && !keep_cpu) RED_ONCE(40, "swizzled texture storage is not in a memory pool; falling back to the CPU");
    if (use_gpu) {
        // The GPU reads the memory pool at execution time, which is closer to a real Switch than a snapshot at CPU upload time; the dirty-page stamp is still taken before encoding, and later writes still trigger a re-upload.
        up_begin();
        if (direct && !(mo & 7)) {
            id<MTLTexture> view = [mt newTextureViewWithPixelFormat:MTLPixelFormatRGBA16Uint textureType:MTLTextureType2DArray levels:NSMakeRange(0,mt.mipmapLevelCount) slices:NSMakeRange(0,mt.arrayLength)];
            if (view) {
                if (!g_up_views) g_up_views=[NSMutableArray array];[g_up_views addObject:view];
                int ok=1;
                for(unsigned l=0;l<d.mipmapLevelCount;l++) {
                    size_t poff;id<MTLBuffer> params=up_param_alloc(&poff);
                    if(nvn_deswz_gpu_direct_m4(g_up_ce,g_up_ct,params,poff,(__bridge id<MTLBuffer>)mb,mo,&t->d,l,view)){ok=0;break;}
                    g_direct_dispatches++;
                }
                if(ok){[g_up_texs addObject:mt];g_direct_uploads++;if(!g_cb)up_flush();return mt;}
                // Any encoded writes must finish before fallback copies replace
                // them; both versions read the same source and write all texels.
                [g_up_ce barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
            }
        }
        size_t stg_off;
        id<MTLBuffer> stg = up_alloc(total ? total : 16, &stg_off);
        size_t off = 0; int enc_ok = g_up_ce != nil, nb0 = g_nup_blits;
        for (unsigned l = 0; enc_ok && l < d.mipmapLevelCount; l++) {
            uint32_t w = MAX(t->d.width >> (l + sk), 1u), h = MAX(t->d.height >> (l + sk), 1u), z = t->d.target == 2 ? MAX(t->d.depth >> l, 1u) : 1;
            size_t pitch = (size_t)((w + bw - 1) / bw) * bpb, rows = (h + bh - 1) / bh;
            for (unsigned y = 0; y < layers; y++) {
                size_t poff; id<MTLBuffer> params = up_param_alloc(&poff);   // Metal 4 has no setBytes: the parameter block of each deswizzle is cut from a shared block (no new GPU buffer each time)
                if (nvn_deswz_gpu_encode_m4(g_up_ce, g_up_ct, params, poff, (__bridge id<MTLBuffer>)mb, mo, &t->d, l + sk, y, stg, stg_off + off, pitch)) { enc_ok = 0; break; }
                up_blit_add(stg, stg_off + off, pitch, pitch * rows, MTLSizeMake(w, h, z), mt, y, l);
                off += pitch * rows * z;
            }
        }
        if (enc_ok) {
            [g_up_texs addObject:mt];
            if (!g_cb) up_flush();
            return mt;
        }
        g_nup_blits = nb0;   // Roll back the copies already recorded for this texture: the already-encoded deswizzle only writes the staging area and is harmless; this texture takes the CPU path below instead
        if (!g_cb) up_flush();
        RED_ONCE(41, "GPU deswizzle does not support format %#x target %d; falling back to the CPU", t->d.format, t->d.target);
    }

    // A sparse pool's texture is read through a copy gathered from its page table, unmapped pages as 0 (a mip tail at a 4 KB offset has no host alias; reading the alias stopped the host with SIGBUS)
    const uint8_t *src = t->mem; uint8_t *part = NULL;
    if (sparse) {
        if (!(part = malloc(t->L.size ? t->L.size : 1))) MDIE("sparse texture copy of %llu bytes failed", (unsigned long long)t->L.size);
        nvn_sparse_copy(t->pool, part, (uint64_t)(uintptr_t)t->mem, t->L.size); src = part;
    }
    up_begin();
    id<MTLBuffer> stg = [g_dev newBufferWithLength:total ? total : 16 options:MTLResourceStorageModeShared];
    res_add(stg); [g_up_keep addObject:stg];
    size_t off = 0;
    int zb = zblock_on() && t->d.format >= 0x79 && t->d.format <= 0x94; if (zb) { g_zb_ser = t->serial; g_zb_n = (int)MIN(d.mipmapLevelCount, 16u); memset(g_zb, 0, sizeof g_zb); memset(g_zb_tot, 0, sizeof g_zb_tot); }
    uint8_t *bc_linear = NULL; size_t bc_linear_cap = 0;
    for (unsigned l = 0; l < d.mipmapLevelCount; l++) {
        uint32_t w = MAX(t->d.width >> (l + sk), 1u), h = MAX(t->d.height >> (l + sk), 1u), z = t->d.target == 2 ? MAX(t->d.depth >> l, 1u) : 1;
        size_t pitch = (size_t)((w + bw - 1) / bw) * bpb, rows = (h + bh - 1) / bh;
        for (unsigned y = 0; y < layers; y++) {
            if (bc_kind) {
                // Decode after deswizzling, keeping the game's compressed
                // guest storage untouched. Only the Metal allocation differs.
                size_t compressed_size = pitch * rows, rgba_size = (size_t)w * h * 4;
                if (compressed_size > bc_linear_cap) {
                    uint8_t *next = realloc(bc_linear, compressed_size);
                    if (!next) MDIE("BC fallback staging allocation failed (%zu bytes)", compressed_size);
                    bc_linear = next; bc_linear_cap = compressed_size;
                }
                if (nvn_deswizzle(&t->d, l + sk, y, src, bc_linear, pitch) ||
                    nvn_bc_decode_rgba8((unsigned)bc_kind, bc_signed, bc_linear, compressed_size,
                                        w, h, (uint8_t *)stg.contents + off, rgba_size)) {
                    RED_ONCE(161, "BC fallback failed to deswizzle/decode format %#x level %u", t->d.format, l + sk);
                    memset((uint8_t *)stg.contents + off, 0, rgba_size);
                }
                up_blit_add(stg, off, (size_t)w * 4, rgba_size, MTLSizeMake(w, h, 1), mt, y, l);
                off += rgba_size;
                continue;
            }
            if (nvn_deswizzle(&t->d, l + sk, y, src, (uint8_t *)stg.contents + off, pitch)) { RED_ONCE(9, "deswizzle failed format %#x target %d", t->d.format, t->d.target); break; }
            if (rb_shift) rb_swap_rows((uint8_t *)stg.contents + off, pitch, rows, z, pitch, pitch * rows, rb_shift);
            if (zb && l < 16) { const uint64_t *q = (const uint64_t *)((const uint8_t *)stg.contents + off); size_t nb = pitch * rows * z / 16;
                for (size_t i = 0; i < nb; i++) g_zb[l] += !(q[2 * i] | q[2 * i + 1]);
                g_zb_tot[l] += (uint32_t)nb; }
            up_blit_add(stg, off, pitch, pitch * rows, MTLSizeMake(w, h, z), mt, y, l);
            off += pitch * rows * z;
        }
    }
    free(bc_linear);
    free(part);
    up_flush();
    return mt;
}
// Zero newly created non-packaged textures: a new Metal texture's content is undefined, while a freshly mapped memory pool on the Switch reads as 0. Games depend on it: SMBW's fountain water surface uses a 512Ã—1 RGBA16F wave texture
// that every frame is "read last frame â†’ render into a render target â†’ copy back" and is never cleared; the undefined initial value keeps swinging through the feedback and pushes the water vertices to the top of the screen (the tall panels above the fountain).
// Copied level by level / slice by slice from one shared all-zero buffer, executed with the upload batch (ordered before this replay). Depth / stencil use clear load actions instead:
// Pokemon Brilliant Diamond's movie quad tests GEQUAL against a newly allocated window depth buffer without clearing it. Metal's undefined initial depth hid every movie even after VP8 decoding.
// Multisample colour and block-compressed formats are not filled by this path.
static size_t mtl_pf_bytes(MTLPixelFormat pf) {
    switch ((int)pf) {
        case 10: case 11: case 12: case 13: case 14: return 1;                                   // R8
        case 20: case 22: case 23: case 24: case 25: case 30: case 31: case 32: case 33: case 34: return 2;   // R16 / RG8
        case 40: case 41: case 42: case 43: return 2;                                            // packed 16-bit colour
        case 53: case 54: case 55: case 60: case 62: case 63: case 64: case 65:
        case 70: case 71: case 72: case 73: case 74: case 80: case 81:
        case 90: case 91: case 92: case 93: case 94: return 4;                                   // R32 / RG16 / RGBA8 / BGRA8 / 10-10-10-2 / 11-11-10 / 9-9-9-5
        case 103: case 104: case 105: case 110: case 112: case 113: case 114: case 115: return 8;   // RG32 / RGBA16
        case 123: case 124: case 125: return 16;                                                 // RGBA32
    }
    return 0;
}
// Shared all-zero buffer (for zeroing new textures and scissored zero clears): replaced by a bigger one when too small; the old one is released after in-flight submissions finish
static id<MTLBuffer> zero_buf(size_t need) {
    static id<MTLBuffer> zb; static size_t zn;
    if (need > zn) {
        size_t n = MAX(need, (size_t)4 << 20);
        id<MTLBuffer> b = [g_dev newBufferWithLength:n options:MTLResourceStorageModeShared];
        if (!b) return nil;
        memset(b.contents, 0, n); res_add(b);
        if (zb) defer_obj_release(zb);
        zb = b; zn = n;
    }
    return zb;
}
static uint64_t g_tex_zeroed;
static void tex_zero_fill(id<MTLTexture> mt) {
    static int warned;
    if (!atomic_load(&g_sub_thread_set) || !on_sub_thread()) return;   // the upload batch belongs to the submit thread only
    MTLTextureType tt = mt.textureType;
    MTLPixelFormat pf = mt.pixelFormat;
    int depth = pf == MTLPixelFormatDepth16Unorm || pf == MTLPixelFormatDepth32Float || pf == MTLPixelFormatDepth32Float_Stencil8;
    int stencil = pf == MTLPixelFormatStencil8 || pf == MTLPixelFormatDepth32Float_Stencil8;
    if (depth || stencil) {
        up_begin();
        if (g_up_ce) { m4_end(g_up_ce); g_up_ce = nil; }
        NSUInteger slices = mt.arrayLength * (tt == MTLTextureTypeCube || tt == MTLTextureTypeCubeArray ? 6 : 1);
        for (NSUInteger l = 0; l < mt.mipmapLevelCount; l++) for (NSUInteger y = 0; y < slices; y++) {
            MTL4RenderPassDescriptor *pd = [MTL4RenderPassDescriptor new];
            if (depth) { pd.depthAttachment.texture = mt; pd.depthAttachment.level = l; pd.depthAttachment.slice = y;
                pd.depthAttachment.loadAction = MTLLoadActionClear; pd.depthAttachment.storeAction = MTLStoreActionStore; pd.depthAttachment.clearDepth = 0; }
            if (stencil) { pd.stencilAttachment.texture = mt; pd.stencilAttachment.level = l; pd.stencilAttachment.slice = y;
                pd.stencilAttachment.loadAction = MTLLoadActionClear; pd.stencilAttachment.storeAction = MTLStoreActionStore; pd.stencilAttachment.clearStencil = 0; }
            id<MTL4RenderCommandEncoder> e = [g_up_cb renderCommandEncoderWithDescriptor:pd];
            [e barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:resource_vis()]; m4_end(e);
        }
        g_up_ce = [g_up_cb computeCommandEncoder];
        [g_up_ce barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:resource_vis()];
        [g_up_ce setArgumentTable:g_up_ct]; [g_up_texs addObject:mt];
        if (!g_cb) up_flush();
        return;
    }
    if (tt == MTLTextureType2DMultisample || tt == MTLTextureType2DMultisampleArray || tt == MTLTextureTypeTextureBuffer) return;
    size_t bpp = mtl_pf_bytes(mt.pixelFormat);
    if (!bpp) { if (!warned++) MLOG("new-texture zeroing: pixel format %lu not cleared (compressed / unsupported)", (unsigned long)mt.pixelFormat); return; }
    NSUInteger slices = mt.arrayLength * (tt == MTLTextureTypeCube || tt == MTLTextureTypeCubeArray ? 6 : 1);
    size_t need = (size_t)mt.width * bpp * mt.height * mt.depth;   // level 0, one slice is the largest
    if (need > (64u << 20)) { static int big; if (!big++) MLOG("new-texture zeroing: %lux%lux%lu exceeds 64 MB, not cleared", (unsigned long)mt.width, (unsigned long)mt.height, (unsigned long)mt.depth); return; }
    id<MTLBuffer> zb = zero_buf(need); if (!zb) return;
    up_begin();
    for (NSUInteger l = 0; l < mt.mipmapLevelCount; l++) {
        NSUInteger w = MAX(mt.width >> l, 1u), h = MAX(mt.height >> l, 1u), z = tt == MTLTextureType3D ? MAX(mt.depth >> l, 1u) : 1;
        for (NSUInteger y = 0; y < slices; y++) up_blit_add(zb, 0, w * bpp, w * bpp * h, MTLSizeMake(w, h, z), mt, y, l);
    }
    [g_up_texs addObject:mt];
    if (!g_cb) up_flush();
    if (!g_tex_zeroed++) MLOG("new-texture zeroing: on (new Metal textures are undefined; new Switch memory reads 0)");
}

// Initial content: what a memory pool's storage holds at nvnMemoryPoolInitialize is the pool's content, also in a pool the CPU cannot access afterwards (CPU_NO_ACCESS), where the
// game never writes it again. Rhythm Heaven Groove reads each texture file into memory and makes a CPU_NO_ACCESS | GPU_CACHED pool on it (245 by its first screens), with BC7 textures
// that have no packaged layout; as GPU textures of zeros its sprite shader discarded every fragment (alpha 0) and the screen stayed black. A new texture in such a pool whose storage
// holds data is filled from it once, as a GPU texture that can still be rendered into (no re-upload on page writes, unlike packaged / CPU-written ones). Left out: storage of zeros
// (the zero fill does the same), COMPRESSIBLE textures (their memory is meaningless without the compression tags), multisample, depth / stencil and sparse ones.
// In a pool the CPU can write, the same holds for storage the game flushed (nvnMemoryPoolFlushMappedRange) before the texture's first use, when the GPU has not written it yet
// (nvn_flushed in hle_nvn.c: Breath of the Wild's sky scattering tables, copied into their pools and flushed; without them the sky was brownish grey). DM_NVN_FLUSH_FILL=0 leaves
// those out, as before. In a CPU_UNCACHED pool the CPU's writes reach memory with no flush, and the GPU reads that memory: what the storage holds at the texture's first use is its
// content there too, flushed or not. Donkey Kong Country: Tropical Freeze's title logo and menu panels (ASTC 5x5, no packaged layout) sit in CPU_UNCACHED | GPU_CACHED pools
// (flags 0x22) whose storage holds their texels at first use (the 912x512 logo: 242,374 of 376,832 bytes non-zero), never flushed; as GPU textures of zeros they were magenta
int nvn_flushed(const void *p, uint64_t n);   // hle_nvn.c
static int mem_any(const uint8_t *p, uint64_t n) {   // whether any byte is non-zero (stops at the first)
    uint64_t i = 0, v;
    for (; i + 8 <= n; i += 8) { memcpy(&v, p + i, 8); if (v) return 1; }
    for (; i < n; i++) if (p[i]) return 1;
    return 0;
}
static int tex_init_fill(const tex_t *t, id<MTLTexture> mt) {
    if (!t->mem || !t->pool || (t->pool->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) || t->has_layout) return 0;
    if (t->d.samples > 1 || (t->d.flags & NVN_TEXTURE_FLAGS_COMPRESSIBLE) || ((int)mt.pixelFormat >= 250 && (int)mt.pixelFormat <= 262)) return 0;   // 250..262: Depth16Unorm â€¦ X24_Stencil8 (some names are macOS-only)
    int no_cpu = (t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS) != 0, uncached = (t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED) != 0;
    static int flush_fill = -1; if (flush_fill < 0) { const char *e = getenv("DM_NVN_FLUSH_FILL"); flush_fill = !e || atoi(e) != 0; }
    if (!no_cpu && !uncached && (!flush_fill || !nvn_flushed(t->mem, t->L.size))) return 0;
    if (!atomic_load(&g_sub_thread_set) || !on_sub_thread() || !mem_any(t->mem, t->L.size)) return 0;   // the upload batch belongs to the submit thread
    if (pkg_upload(t, NULL, mt, 0) != mt) return 0;
    int why = no_cpu ? 0 : uncached ? 1 : 2;
    static uint64_t n[3]; if (n[why]++ < 4) MLOG("initial content: ser=%llu %ux%ux%u format %#x filled from its storage, %s (%llu so far)", (unsigned long long)t->serial, t->d.width, t->d.height, t->d.depth, t->d.format,
                                                 (const char *[]){"a CPU_NO_ACCESS pool's", "a CPU_UNCACHED pool's", "flushed by the CPU before its first use"}[why], (unsigned long long)n[why]);
    return 1;
}

static int clear_rect_zero(int i);
// Linear textures (LINEAR) in a CPU-visible pool: the game writes the pool memory directly from the CPU (SMBW's glyph cache: 2048Ã—8192 R8, IMAGE|LINEAR, pool CPU_UNCACHED|GPU_CACHED,
// the font engine rasterizes glyphs into it on the CPU). A GPU-private texture never sees those writes â†’ all menu / dialog text is blank. On unified memory, create a linear texture directly on the pool buffer (zero copy): the GPU reads the current content at execution time, same as NVN.
static id<MTLTexture> tex_linear(const tex_t *t) {
    // LINEAR_RENDER_TARGET is a linear layout too (SMBW's sky colour ramps: four 64Ã—1 RGBA16F, written by the CPU every frame). Also in a pool the CPU cannot access, whose buffer holds what
    //   its storage held at creation (see tex_init_fill): Rhythm Heaven Groove's 228 linear R8 textures (sprite masks), read from files into CPU_NO_ACCESS pools, were private textures of zeros
    if (!(t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) || !t->mem || !t->pool) return nil;
    if (t->d.target != NVN_TEXTURE_TARGET_2D || t->d.levels > 1 || t->d.samples > 1) return nil;
    uint32_t pf = 0; uint8_t swz[4]; int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    if (!nvn_fmt_to_mtl(t->d.format, &pf, swz) || !pf || !bpb || bw != 1 || bh != 1) return nil;
    size_t row = t->L.layer_size / (t->d.height ? t->d.height : 1);   // same explicit or aligned stride as GetStorageSize
    void *mb; uint64_t mo;
    if (!mtl_addr((uint64_t)t->mem, (uint64_t)row * (t->d.height ? t->d.height : 1), &mb, &mo)) { RED_ONCE(64, "linear texture storage is not in a memory pool (%ux%u format %#x)", t->d.width, t->d.height, t->d.format); return nil; }
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)mb;
    NSUInteger al = [g_dev minimumLinearTextureAlignmentForPixelFormat:(MTLPixelFormat)pf];
    if (mo % al || row % al) { RED_ONCE(65, "linear texture offset %#llx / row pitch %zu not aligned to %lu bytes; created as a private texture (CPU writes invisible)", (unsigned long long)mo, row, (unsigned long)al); return nil; }
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:(MTLPixelFormat)pf width:t->d.width ? t->d.width : 1 height:t->d.height ? t->d.height : 1 mipmapped:NO];
    d.resourceOptions = b.resourceOptions;
    d.usage = MTLTextureUsageShaderRead | img_usage(t, (MTLPixelFormat)pf) | (t->d.flags & NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET ? MTLTextureUsageRenderTarget : 0);
    return [b newTextureWithDescriptor:d offset:mo bytesPerRow:row];
}
// CPU-written block-linear textures (WriteTexels): the game gave no packaged layout and changes the content from the CPU each frame (Engage's movie frames: a Y plane R8 + a UV plane RG8, WriteTexels every frame).
// Only packaged textures used to be uploaded from guest memory, so such a texture became a GPU-only texture whose content stayed 0. Their serials are recorded and they resolve through the packaged path (upload + re-upload on dirty pages)
// Every texture resolve asks about CPU writes. Mix serial keys: consecutive serials in a callback-free CFSet made misses probe long bucket runs (see reference/frame-rate.md).
// Finalize removes live membership, but command records and registered descriptors can still resolve the serial. Keep their classification until the storage pool is finalized;
// a copied flag would miss writes made after recording. Retired metadata can grow while a pool lives: expiring it earlier requires tracking ownership of all snapshots.
static pthread_mutex_t g_cpu_tex_mu = PTHREAD_MUTEX_INITIALIZER;
static nvn_hmap g_cpu_tex = {.klen = 8}, g_cpu_tex_retired = {.klen = 8};   // serial â†’ pool (or (void *)1 for a live test texture without a pool)
static void cpu_tex_drop_pool(const void *pool) {
    pthread_mutex_lock(&g_cpu_tex_mu);
    nvn_hmap *maps[] = {&g_cpu_tex, &g_cpu_tex_retired};
    for (unsigned m = 0; m < sizeof maps / sizeof *maps; m++)
        for (uint32_t i = 0; i < maps[m]->cap;)
            if (maps[m]->vals[i] == pool) { uint64_t serial; memcpy(&serial, maps[m]->keys + (size_t)i * 8, 8); nvn_hm_del(maps[m], &serial); }
            else i++;
    pthread_mutex_unlock(&g_cpu_tex_mu);
}
// The GPU owns the texture's content: its host texture was made to be rendered into or written by shaders (nvn_ft_poll in hle_nvn.c never re-uploads such a texture from guest memory)
int mtl_tex_gpu_owned(uint64_t serial) {
    __unsafe_unretained id o = tex_get(serial);
    if (!o || ![o conformsToProtocol:@protocol(MTLTexture)]) return 0;
    return (((id<MTLTexture>)o).usage & (MTLTextureUsageRenderTarget | MTLTextureUsageShaderWrite)) != 0;
}
void mtl_tex_cpu_written(const tex_t *t) {
    uint64_t serial = t->serial;
    if (!serial) return;
    // nvn_ft_poll can mark a snapshot after Finalize. Serialize the live check with release (g_smu â†’ g_cpu_tex_mu), so that mark goes straight to retired metadata.
    pthread_mutex_lock(&g_smu);
    pthread_mutex_lock(&g_cpu_tex_mu);
    int live = nvn_hm_get(&g_tex_live, &serial) != NULL;
    if ((live || (t->pool && t->mem)) && !nvn_hm_get(&g_cpu_tex, &serial) && !nvn_hm_get(&g_cpu_tex_retired, &serial)) {
        nvn_hm_put(live ? &g_cpu_tex : &g_cpu_tex_retired, &serial, t->pool ? (void *)t->pool : (void *)1);
        g_ser_gen++;   // cached texture IDs resolve again through the packaged path (tex_by_id view_pkg)
        static int n; if (n++ < 16) MLOG("CPU-written texture ser=%llu %ux%ux%u format %#x target %d levels %u: uploaded / re-uploaded like a swizzled texture", (unsigned long long)serial, t->d.width, t->d.height, t->d.depth, t->d.format, t->d.target, t->d.levels);
    }
    pthread_mutex_unlock(&g_cpu_tex_mu);
    pthread_mutex_unlock(&g_smu);
}
unsigned mtl_cpu_tex_count(void) { pthread_mutex_lock(&g_cpu_tex_mu); unsigned n = g_cpu_tex.n; pthread_mutex_unlock(&g_cpu_tex_mu); return n; }
static int tex_cpu_written(uint64_t serial) {
    pthread_mutex_lock(&g_cpu_tex_mu);
    int r = nvn_hm_get(&g_cpu_tex, &serial) != NULL || nvn_hm_get(&g_cpu_tex_retired, &serial) != NULL;
    pthread_mutex_unlock(&g_cpu_tex_mu);
    return r;
}
// Whether the host takes a texture's content from its storage (tex_mtl_'s packaged path): a packaged layout, CPU-written, or a sampled texture in a sparse virtual pool, which holds what
// the CPU wrote into the physical pages mapped there (Monster Hunter Rise: the hunter's innerwear, ASTC 6x6 with no packaged layout, came out magenta, and the face black) and is read
// through the page table. Render targets, images, linear and buffer textures stay GPU textures (as ft_add). mtl_read_texels reads these from the storage too, on whatever thread asks
static int tex_from_storage(const tex_t *t) {
    return (t->has_layout || tex_cpu_written(t->serial) || (t->pool && (t->pool->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) && t->d.target != NVN_TEXTURE_TARGET_BUFFER
            && !(t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_IMAGE)) && nvn_sparse_pool(t->pool))) && t->mem;
}
// Storage the CPU wrote before the texture's first use (every game): until the GPU first uses a texture it has not written it, and a block-compressed one it can never render into
// (BC and ASTC formats are not colour-renderable; GPU copies into it are written into its storage too, tex_storage_dst). So a sampled block-compressed texture in a pool the CPU can
// write, with no packaged layout, whose storage holds data at its first use (tex_mtl_, submit thread) holds what the CPU wrote, flushed or not, before or after nvnTextureInitialize:
// it is made CPU-written, uploaded now and re-uploaded when its pages change. A one-time fill (tex_init_fill) is not enough: that GPU texture goes into the sharing table (g_alias),
// and a new texture object over the same storage with the same description gets it back after the CPU has rewritten the storage. The Great Ace Attorney Chronicles (Japanese release
// 0100D7F00FB1A000, GitHub issue #568) decodes its court-record pictures (256x256 to 2048x2048 BC1, a CPU_CACHED pool, no flush) into storage it reuses, building a new texture over
// it for each item: they were black, and once filled once, showed the previous item's picture. On Switch 1, uncompressed textures cannot be told from render targets (NVN has no flag
// for that, so tex_desc gives them render-target use) and keep the flushed-range fill.
// Switch 2 (NVN2) memory is coherent: the driver's flushes are bare returns, so the GPU reads what the CPU stored, flushed or not, and a texture first used as a sampled texture
// (sampled: its first resolution comes from a texture ID binding, tex_by_id) that the GPU has not written holds what its storage holds, whatever its format. So there an uncompressed
// texture is taken from its storage too when its first use is a sample, unless the sharing table already holds a GPU texture over the same storage (a render target drawn through
// another object, which a sample must see). Mario Kart World stores its terrain tiles (260x260 R16 height, RG8 material, RGBA8 and R11G11B10F colour, 68x68 RGBA16; a CPU_CACHED pool)
// into reused slots before nvnTextureInitialize and never flushes them: as GPU textures of zeros the ground came out flat and untextured and Peach Beach had no sea (from the
// kartnative fork, 313e90f9). A texture first used as an attachment, a copy destination or an image stays a GPU texture. DM_NVN_STORAGE_FIRST_USE=0 turns this off (to measure it)
// What the sharing table's key leaves out of a texture's description; returns the texture's Metal format, 0 if it has none.
// IMAGE permits shader writes; it does not change storage. Xenoblade Chronicles 3's depth and stencil-sampling objects differ only by this flag. Separate Metal textures
// made the stencil read zero and rejected every deferred-lighting pass. Actual image writes add Metal usage before opening their encoder (image_prepare).
// An sRGB format and its linear pair (g_pfv_srgb) over the same storage are one texture, each object served as a view in its own format (no pixel-format-view usage
// needed: tex_need_pfv). Mario Kart World draws each decoded movie frame into an RGBA8 render target and composites it through an RGBA8_SRGB texture over the same pool
// and offset; as two Metal textures the composite kept the picture from before the movie and the screen stood still for the whole movie
static uint32_t alias_key_norm(alias_key *ak) {
    ak->d.flags &= ~NVN_TEXTURE_FLAGS_IMAGE;
    uint32_t apf = 0; uint8_t asw[4];
    if (nvn_fmt_to_mtl(ak->d.format, &apf, asw) && apf && asw[0] == NVN_SWZ_R && asw[1] == NVN_SWZ_G && asw[2] == NVN_SWZ_B && asw[3] == NVN_SWZ_A)
        for (size_t i = 0; i < sizeof g_pfv_srgb / sizeof *g_pfv_srgb; i++)
            if (apf == g_pfv_srgb[i].a || apf == g_pfv_srgb[i].b) { ak->d.format = 0x40000000 | (int)g_pfv_srgb[i].a; break; }
    return apf;
}
int dm_sdk_ounce(void);
static int tex_cpu_content(const tex_t *t, int sampled) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_NVN_STORAGE_FIRST_USE"); on = !e || atoi(e) != 0; }
    int bc = fmt_compressed(t->d.format), nvn2 = sampled && dm_sdk_ounce() && !(t->d.format >= NVN_FORMAT_STENCIL8 && t->d.format <= NVN_FORMAT_DEPTH32F_STENCIL8);
    if (!on || !t->mem || !t->pool || t->has_layout || !(bc || nvn2) || t->d.samples > 1 || t->d.target == NVN_TEXTURE_TARGET_BUFFER) return 0;
    if (t->pool->flags & (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL)) return 0;   // the CPU cannot write it (tex_init_fill fills it once) / sparse (tex_from_storage)
    if (t->d.flags & (NVN_TEXTURE_FLAGS_IMAGE | NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_SPARSE)) return 0;
    if (!mem_any(t->mem, t->L.size)) return 0;
    if (!bc) {
        alias_key ak; memset(&ak, 0, sizeof ak); ak.pool = (uint64_t)(uintptr_t)t->pool; ak.off = t->off; ak.d = t->d; alias_key_norm(&ak);
        pthread_mutex_lock(&g_smu); int shared = hm_get_id(&g_alias, &ak) != nil; pthread_mutex_unlock(&g_smu);
        if (shared) return 0; }
    static uint64_t n[2]; if (n[!bc]++ < 4) MLOG("texture written by the CPU before its first use ser=%llu %ux%ux%u format %#x pool flags %#x: uploaded from its storage, re-uploaded when it changes (%s, %llu so far)",
                                                 (unsigned long long)t->serial, t->d.width, t->d.height, t->d.depth, t->d.format, t->pool->flags, bc ? "block-compressed" : "NVN2, sampled first", (unsigned long long)n[!bc]);
    return 1;
}
static id<MTLTexture> tex_mtl_(const tex_t *ct);
static id<MTLTexture> tex_mtl(const tex_t *ct) { AP_B; id<MTLTexture> r = tex_mtl_(ct); AP_E(4); return r; }
// The first resolution of a texture for a texture ID binding (tex_by_id) is a sample (tex_cpu_content); tex_mtl_ takes the flag on entry, so nothing it resolves in turn inherits it
static _Thread_local int g_res_sampled;
static id<MTLTexture> tex_mtl_sampled(const tex_t *ct) { g_res_sampled = 1; id<MTLTexture> r = tex_mtl(ct); g_res_sampled = 0; return r; }
// A sparse texture is bound from its first level whose pages, and those of every smaller level, are all mapped: a view of levels [r, n) samples as with a minimum LOD of r.
// The translator drops TEX's LOD clamp and residency predicate, so Monster Hunter Rise's streamed textures were sampled at levels it had not mapped yet; those read 0, which ASTC
// decodes as its error colour (magenta clothes on the title screen). Single-layer 2D textures only; the view follows each upload, as the mappings change only with one
static void sparse_view(NvPkg *pk, const tex_t *t) {
    unsigned n = (unsigned)pk->tex.mipmapLevelCount, r = n;
    if (t->d.target == NVN_TEXTURE_TARGET_2D && n > 1)
        while (r > 0) { uint64_t lo = nvn_tex_level_offset(&t->d, r - 1), hi = r < n ? nvn_tex_level_offset(&t->d, r) : t->L.size;
            if (!nvn_sparse_mapped(t->pool, (uint64_t)(uintptr_t)t->mem + lo, hi - lo)) break;
            r--; }
    else r = 0;
    if (r == n) r = n - 1;   // nothing mapped: the smallest level (reads 0)
    if (pk->view) defer_obj_unref(pk->view);   // a view shares its texture's allocation: drop only the reference, the residency set keeps the texture
    pk->view = r ? [pk->tex newTextureViewWithPixelFormat:pk->tex.pixelFormat textureType:pk->tex.textureType levels:NSMakeRange(r, n - r) slices:NSMakeRange(0, 1)] : nil;
    if (r != (unsigned)pk->vbase) { static int nl; if (nl++ < 6) MLOG("sparse texture ser=%llu %ux%u: sampled from level %u of %u (finer levels not mapped)", (unsigned long long)t->serial, t->d.width, t->d.height, r, n); }
    pk->vbase = (int)r;
}
static uint64_t g_texinfo_ser[8]; static int g_texinfo_n;
// Distinct compressible image descriptions at one NVN address alias physical memory, even while all their descriptor objects are live (LM3's resolution families).
// Single-description storage keeps the ordinary allocation. Families share an exact-sized placement heap, found through the existing alias table rather than a pool-lifetime owner.
// Different formats, offsets, CPU-visible, sparse, linear, display, layered and multisample images keep their existing path. Inherited texels are not translated between incompatible
// layouts: a changed layout is initialized by its GPU writer. Creating its descriptor must not zero another alias's live content in the earlier upload batch.
static id<MTLTexture> image_texture(const tex_t *t, MTLTextureDescriptor *d) {
    if (!image_shape(t) || !t->pool->mem || !(t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS) || nvn_sparse_pool(t->pool)) return nil;
    MTLSizeAndAlign need = [g_dev heapTextureSizeAndAlignWithDescriptor:d];
    if (!need.size) return nil;
    NvImageHeap *owner = nil; MTLTextureDescriptor *largest = d; NSUInteger capacity = need.size; int family = 0;
    pthread_mutex_lock(&g_smu);
    for (uint32_t i = 0; i < g_alias.cap; i++) if (g_alias.vals[i]) {
        alias_key k; memcpy(&k, g_alias.keys + (size_t)i * g_alias.klen, sizeof k);
        if (k.pool != (uint64_t)(uintptr_t)t->pool || k.off != t->off || k.d.format != t->d.format) continue;
        id<MTLTexture> other = (__bridge id)g_alias.vals[i];
        NvImageHeap *o = objc_getAssociatedObject(other, &g_image_heap_owner_key);
        // Metal omits ShaderAtomic from usage; tex_desc gives every writable R32 integer image that capability.
        MTLTextureUsage usage = other.usage | (tex_atomic_capable(other) ? MTLTextureUsageShaderAtomic : 0);
        if (o && o->heap.size >= need.size && usage == d.usage && (!owner || o->heap.size < owner->heap.size)) owner = o;
    }
    if (!owner) for (uint32_t i = 0; i < g_image_desc.cap; i++) if (g_image_desc.vals[i]) {
        const tex_t *other = g_image_desc.vals[i];
        if (other->pool != t->pool || other->off != t->off || other->d.format != t->d.format) continue;
        family |= other->d.width != t->d.width || other->d.height != t->d.height;
        MTLTextureDescriptor *od = tex_desc(other, 0); if (!od) continue;
        MTLSizeAndAlign sz = [g_dev heapTextureSizeAndAlignWithDescriptor:od];
        if (sz.size > capacity) { capacity = sz.size; largest = od; }
    }
    pthread_mutex_unlock(&g_smu);
    int fresh = !owner;
    if (fresh) {
        if (!family) return nil;
        MTLHeapDescriptor *hd = [MTLHeapDescriptor new]; hd.type = MTLHeapTypePlacement; hd.size = capacity;
        hd.storageMode = d.storageMode; hd.cpuCacheMode = d.cpuCacheMode; hd.hazardTrackingMode = d.hazardTrackingMode;
        id<MTLHeap> heap = [g_dev newHeapWithDescriptor:hd]; if (!heap) return nil;
        owner = [NvImageHeap new]; owner->heap = heap; atomic_fetch_add(&g_image_heap_n, 1);
    }
    id<MTLTexture> mt = [owner->heap newTextureWithDescriptor:d offset:0]; if (!mt) return nil;
    objc_setAssociatedObject(mt, &g_image_heap_owner_key, owner, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    res_add(mt);
    if (fresh) {
        // Initialize the whole family, including storage outside a first-used small resolution. Only a new heap is zeroed; later descriptors alias its existing bytes.
        id<MTLTexture> zero = largest == d ? mt : [owner->heap newTextureWithDescriptor:largest offset:0];
        if (!zero) return nil;
        objc_setAssociatedObject(zero, &g_image_heap_owner_key, owner, OBJC_ASSOCIATION_RETAIN_NONATOMIC); tex_zero_fill(zero);
    }
    return mt;
}
// The host texture a texture already has, made and refreshed by nothing: for a texture whose storage is gone (a memory pool finalized, tex_mtl_ and tex_by_id). nil if it has none
static id<MTLTexture> tex_made(const tex_t *t) {
    id have = tex_get(t->serial);
    if ([have isKindOfClass:[NvPkg class]]) { NvPkg *pk = have; return pk->view ? pk->view : pk->tex; }
    return have;
}
// The game copies from this texture: if it is soft, it goes whole from here on, uploaded again now with its finest level. Its storage still holds that level unless the texture has
// lived SOFT_WAIT frames unused as a source
static void soft_harden(const tex_t *t) {
    if (!t || !tex_soft(t) || (t->pool && t->pool->magic != MAGIC('P', 'O', 'O', 'L'))) return;
    id have = tex_get(t->serial); NvPkg *pk = [have isKindOfClass:[NvPkg class]] ? have : nil;
    if (pk && pk->released) { static uint64_t n; if (!n++) MLOG("[RED] copy from a soft texture whose finest level has gone back (first case ser=%llu %ux%u format %#x): copy of that level skipped", (unsigned long long)t->serial, t->d.width, t->d.height, t->d.format); return; }
    uint64_t k = (uint64_t)(uintptr_t)t->mem; pthread_mutex_lock(&g_smu); nvn_hm_put(&g_soft_hard, &k, (void *)1); pthread_mutex_unlock(&g_smu);
    static uint64_t nh; if (nh++ < 4 || !(nh & (nh - 1))) MLOG("soft textures: kept whole, the game copies from it: ser=%llu %ux%ux%u format %#x (%llu so far)", (unsigned long long)t->serial, t->d.width, t->d.height, t->d.depth, t->d.format, (unsigned long long)nh);
    if (!pk) return;
    uint64_t st; id<MTLTexture> nt = pkg_upload(t, &st, nil, pk->tex.usage);
    if (!nt) return;
    if (pk->spare) defer_obj_release(pk->spare);
    pk->spare = pk->tex; pk->spare_fence = g_fence + 1; pk->tex = nt; g_ser_gen++; pk->stamp = st; pk->span = nvn_dirty_span(t->mem, t->L.size); pk->seen = st; pkg_edges_take(pk, t);
}
static id<MTLTexture> tex_mtl_(const tex_t *ct) {
    tex_t *t = (tex_t *)ct; int sampled = g_res_sampled; g_res_sampled = 0;   // see tex_mtl_sampled
    if (!t || !t->serial) return nil;
    if (t->magic != MAGIC('T', 'E', 'X', 0)) { RED_ONCE(70, "texture object no longer valid (magic %#x, first case ser=%llu): no MTLTexture created", t->magic, (unsigned long long)t->serial); return nil; }
    // t->pool is the game's object, named by a copy of the texture that an ID or a record keeps past its life: nvnMemoryPoolFinalize clears the magic, and the game then frees the memory, which
    // then reads as anything. Everything below reads it: garbage in its vp sent an upload through a page table that was never there (nvn_sparse_protect, FINAL FANTASY RESONANCE, GitHub issue #934)
    if (t->pool && t->pool->magic != MAGIC('P', 'O', 'O', 'L')) {
        id<MTLTexture> made = tex_made(t);
        static uint64_t n; if (n++ < 4) MLOG("texture of a memory pool that is gone ser=%llu %ux%u format %#x: %s (%llu so far)", (unsigned long long)t->serial, t->d.width, t->d.height, t->d.format,
                                             made ? "keeps its host texture, not refreshed" : "no host texture made", (unsigned long long)n);
        return made;
    }
    int packaged = tex_from_storage(t);
    {   // Diagnostic DM_NVN_TEXINFO=WxH: log storage and description of textures of that size on first resolution (to tell whether two textures share storage and why they were not merged)
        static int ti = -1; static uint32_t tw, th; static uint64_t last[64]; static int nl;
        if (ti > 0 && t->d.width == tw && t->d.height == th && g_texinfo_n < 8) { int dup = 0; for (int i = 0; i < g_texinfo_n; i++) dup |= g_texinfo_ser[i] == t->serial; if (!dup && !(t->d.flags & 8)) g_texinfo_ser[g_texinfo_n++] = t->serial; }   // remember the non-render-target ones (no COMPRESSIBLE) for DM_NVN_TEXRAW=info
        if (ti < 0) { const char *e = getenv("DM_NVN_TEXINFO"); ti = e && sscanf(e, "%ux%u", &tw, &th) == 2; }
        if (ti && t->d.width == tw && t->d.height == th) {
            int seen = 0; for (int i = 0; i < nl; i++) if (last[i] == t->serial) seen = 1;
            static uint64_t n; if (!seen && (n++ < 40 || n % 600 == 0)) { if (nl < 64) last[nl++] = t->serial; else last[n % 64] = t->serial;
                uint64_t nz = 0; if (t->mem && t->L.size <= (64u << 20)) { uint8_t *m = t->mem, *part = NULL;   // non-zero bytes in guest memory: content the CPU wrote (a sparse pool's texture through its page table)
                    if (t->pool && nvn_sparse_pool(t->pool) && (part = malloc(t->L.size))) { nvn_sparse_copy(t->pool, part, (uint64_t)(uintptr_t)t->mem, t->L.size); m = part; }
                    for (uint64_t i = 0; i < t->L.size; i++) nz += m[i] != 0;
                    free(part); }
                MLOG("TEXINFO ser=%llu swizzled=%d has_layout=%d mem=%p pool=%p(%#x) off=%#llx target=%d format=%#x flags=%#x samples=%d %ux%ux%u levels=%u stride=%u block_h=%d storage %llu bytes non-zero %llu",
                     (unsigned long long)t->serial, packaged, t->has_layout, t->mem, t->pool, t->pool ? t->pool->flags : 0, (unsigned long long)t->off, t->d.target, t->d.format, t->d.flags, t->d.samples,
                     t->d.width, t->d.height, t->d.depth, t->d.levels, t->d.stride, t->d.pkg_block_h, (unsigned long long)t->L.size, (unsigned long long)nz); }
        }
    }
    id have = tex_get(t->serial);
    if (!packaged && !have && DM_GAME_TEX_DIRTY_UPLOAD && t->mem && on_sub_thread()) { int nvn_ft_written(const tex_t *); packaged = nvn_ft_written(t); }   // first use of a texture the CPU wrote since init without a flush (hle_nvn.c)
    if (!packaged && !have && on_sub_thread() && tex_cpu_content(t, sampled)) { mtl_tex_cpu_written(t); packaged = 1; }   // storage the CPU wrote before its first use (tex_cpu_content)
    if (packaged) {
        NvPkg *pk = [have isKindOfClass:[NvPkg class]] ? have : nil;   // resolved as GPU-only before the CPU wrote it: dropped and rebuilt as packaged
        if (pk) {   // Already uploaded: check the covered pages only if there has been a guest write since the last check
            if (g_nabs) pkg_absorb_take(pk, t);
            if (g_nabsp && pkg_absorb_waits(t->serial)) return pk->view ? pk->view : pk->tex;
            uint64_t now = nvn_dirty_now();
            int tracked = pk->span != NULL;   // kept from the upload: a texture's pool stays registered while the texture lives
            int clean = tracked && (now == pk->seen || !nvn_dirty_since_span(pk->span, t->mem, t->L.size, pk->stamp));
            if (pk->sparse) {   // A walk of the page table on every binding was two thirds of Mario Kart World's submit thread in a Mac race (its 8192x8192x68 arrays). With no mapping changed since the
                // last clean walk the runs are the same, so the one range they lie in tells whether any of them was written (from the kartnative fork, b0046eb3). And the look starts from the last time the texture
                // was found clean, not from its upload: generations only rise, and its main pool is written all the time, while the texture's pages rarely are
                uint64_t g = nvn_sparse_gen(), quiet = nvn_dirty_quiet_now(), from = MAX(pk->stamp, pk->clean_at);
                if (now == pk->seen && g == pk->vseen) clean = 1;
                else if (pk->vone && g == pk->vone_gen && !nvn_dirty_range_since(pk->vone, from)) clean = 1;
                else { const void *one = NULL; clean = !nvn_sparse_since(t->pool, (uint64_t)(uintptr_t)t->mem, t->L.size, pk->vgen, from, &one); pk->vone = clean ? one : NULL; pk->vone_gen = g; }
                if (clean && quiet) pk->clean_at = quiet;
                pk->vseen = g; }
            pk->seen = now;
            if (clean && atomic_load_explicit(&g_up_lost_n, memory_order_relaxed) && up_lost_take(pk->tex)) { clean = 0; uint64_t k = atomic_load(&g_up_lost_redo); if (k <= 3 || !atomic_load(&g_up_lost_n)) MLOG("re-uploading a texture whose upload was dropped ser=%llu (%llu in total, %d left in the record)", (unsigned long long)t->serial, (unsigned long long)k, atomic_load(&g_up_lost_n)); }
            pk->used = atomic_load(&g_cur_frame);
            if (clean) {
                if (dirtycheck() && !pk->sparse && full_hash(t->mem, t->L.size) != pk->hash && !g_pkg_miss++)
                    MLOG("[RED] dirty-page miss: texture ser=%llu pool content changed but was not detected", t->serial);
                return pk->view ? pk->view : pk->tex;
            }
            if (!pk->sparse && pkg_edges_clean(pk, t)) return pk->view ? pk->view : pk->tex;   // only the pages it shares with other data were written
            // Pool content changed (streamed assets written into the same storage, a movie frame): switch to another texture, as in-flight draws still read the old one. That is the spare
            // when the GPU is done with it (see NvPkg), else a new one; the old one becomes the spare (pkg_upload_ made nt resident)
            id<MTLTexture> reuse = nil;
            if (pk->spare) { pthread_mutex_lock(&g_mu); uint64_t done = g_done; pthread_mutex_unlock(&g_mu); if (pk->spare_fence <= done && pk->spare.usage == pk->tex.usage) reuse = pk->spare; }
            uint64_t st, vg = nvn_sparse_gen(); id<MTLTexture> nt = pkg_upload(t, &st, reuse, pk->tex.usage);
            if (nt) { id<MTLTexture> old = pk->tex; pk->tex = nt;
                if (pk->spare && pk->spare != nt) defer_obj_release(pk->spare);   // still in use, or of another shape: retired as before
                if (pk->sparse) sparse_view(pk, t);
                zb_report(t, pk->sparse ? (unsigned)pk->vbase : 0, "re-upload");
                pk->spare = old; pk->spare_fence = g_fence + 1; g_ser_gen++; pk->stamp = st; pk->vgen = pk->vseen = vg; pk->span = nvn_dirty_span(t->mem, t->L.size); pk->seen = st; g_pkg_reuploads++; pk->reup++; if (dirtycheck() && !pk->sparse) pk->hash = full_hash(t->mem, t->L.size); pkg_edges_take(pk, t); }
            return pk->view ? pk->view : pk->tex;
        }
        pst_key sk; int store = pst_eligible(t);
        if (store) {   // the same storage and description as a texture uploaded before and not written since: that upload (see g_pst)
            sk = pst_key_of(t);
            pthread_mutex_lock(&g_smu); NvPkg *hit = hm_get_id(&g_pst, &sk); pthread_mutex_unlock(&g_smu);
            if (hit && hit->span && !hit->reup && !nvn_dirty_since_span(hit->span, t->mem, t->L.size, hit->stamp)) {
                hit->used = atomic_load(&g_cur_frame); g_pst_reused++; tex_put(t->serial, hit); return hit->view ? hit->view : hit->tex; }
            if (hit) { pthread_mutex_lock(&g_smu); hm_del_id(&g_pst, &sk); pthread_mutex_unlock(&g_smu); }
        }
        pk = [NvPkg new];
        pk->sparse = t->pool && nvn_sparse_pool(t->pool); pk->vgen = pk->vseen = nvn_sparse_gen();
        pk->tex = pkg_upload(t, &pk->stamp, nil, 0);
        if (!pk->tex) return nil;
        if (store) { pk->rmem = t->mem; pk->rsize = t->L.size; pk->rpool = t->pool; pk->born = pk->used = atomic_load(&g_cur_frame);
            pthread_mutex_lock(&g_smu); hm_put_id(&g_pst, &sk, pk); pthread_mutex_unlock(&g_smu); }
        pk->span = nvn_dirty_span(t->mem, t->L.size);
        if (tex_soft(t) && DM_GAME_TEX_RELEASE) tex_soft_release(t);
        pk->seen = pk->stamp; g_pkg_uploads++; if (dirtycheck() && !pk->sparse) pk->hash = full_hash(t->mem, t->L.size);   // "nothing written since" starts at the stamp: a write that faulted during the upload moved "now" past it
        pkg_edges_take(pk, t);
        if (pk->sparse) sparse_view(pk, t);
        zb_report(t, pk->sparse ? (unsigned)pk->vbase : 0, "upload");
        tex_put(t->serial, pk);
        return pk->view ? pk->view : pk->tex;
    }
    if (have && g_sp_rt_n && t->pool && nvn_sparse_pool(t->pool) && [have conformsToProtocol:@protocol(MTLTexture)]) {   // the game changed a sparse pool's mappings: a sparse target follows them
        NvSparse *sp = objc_getAssociatedObject(have, &g_sp_key);
        if (sp && sp->rt && sp->gen != nvn_sparse_gen()) sparse_upload(t, NULL, 0, 1);
    }
    if (have) return have;
    if (t->d.target == 10) {   // Buffer texture (TARGET_BUFFER): create the texture-buffer view directly on the memory pool buffer (zero copy; the GPU reads the pool's current content at execution time, same as NVN semantics)
        uint32_t pf = 0; uint8_t swz[4]; int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
        void *mb; uint64_t mo;
        if (!nvn_fmt_to_mtl(t->d.format, &pf, swz) || !pf || !bpb) { RED_ONCE(36, "buffer texture format %#x not mapped", t->d.format); return nil; }
        if (!t->mem || !mtl_addr((uint64_t)t->mem, (uint64_t)t->d.width * bpb, &mb, &mo)) { RED_ONCE(37, "buffer texture storage is not in a memory pool"); return nil; }
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)mb;
        NSUInteger al = [g_dev minimumLinearTextureAlignmentForPixelFormat:(MTLPixelFormat)pf];
        // Metal requires 16-byte alignment for texture buffer views (minimumTextureBufferAlignmentForPixelFormat), and an Apple GPU addresses such a view from the 16-byte boundary below its
        // offset (measured on an M4 Pro: R32Float views at offsets 4, 8 and 12 read elements 0, 0 and 0 instead of 1, 2 and 3, shared and private buffers, bound directly or in an argument
        // buffer). Unreal puts buffer textures at 4-, 8- and 12-byte offsets in its pools (Octopath Traveler 0: offset 0x168; FINAL FANTASY RESONANCE's particle data), so the view is created
        // there and shaders read it through an aligned copy (tbuf_shadow). Writes through such a view cannot be redirected. Below the element size the view stays a dummy
        if (mo % al) {
            if (mo % (uint64_t)bpb) { RED_ONCE(38, "buffer texture offset %#llx not aligned to its %d-byte elements", (unsigned long long)mo, bpb); return nil; }
            if (t->d.flags & NVN_TEXTURE_FLAGS_IMAGE) RED_ONCE(119, "image buffer texture at pool offset %#llx, below Metal's %lu-byte alignment: shader writes to it land at the 16-byte boundary below", (unsigned long long)mo, (unsigned long)al);
            static int once; if (!once++) MLOG("buffer texture at pool offset %#llx, below Metal's %lu-byte alignment: created there, read through an aligned copy (tbuf_shadow)", (unsigned long long)mo, (unsigned long)al);
        }
        MTLTextureDescriptor *d = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:(MTLPixelFormat)pf width:t->d.width resourceOptions:b.resourceOptions
                                                                                         usage:MTLTextureUsageShaderRead | img_usage(t, (MTLPixelFormat)pf)];
        size_t row = ((size_t)t->d.width * bpb + al - 1) / al * al;
        id<MTLTexture> mt = [b newTextureWithDescriptor:d offset:mo bytesPerRow:row];
        if (!mt) { RED_ONCE(39, "cannot create buffer texture: %u elements format %#x", t->d.width, t->d.format); return nil; }
        res_add(mt);
        tex_put(t->serial, mt);
        return mt;
    }
    alias_key ak; memset(&ak, 0, sizeof ak); ak.pool = (uint64_t)(uintptr_t)t->pool; ak.off = t->off; ak.d = t->d;
    uint32_t apf = alias_key_norm(&ak);
    pthread_mutex_lock(&g_smu); __unsafe_unretained id shared = hm_get_id(&g_alias, &ak); if (shared) nvn_hm_put(&g_alias_seen, &ak, (void *)(uintptr_t)(g_fence + 1)); pthread_mutex_unlock(&g_smu);
    if (shared && (ak.d.format & 0x40000000) && ((id<MTLTexture>)shared).pixelFormat != (MTLPixelFormat)apf) {
        id<MTLTexture> v = [(id<MTLTexture>)shared newTextureViewWithPixelFormat:(MTLPixelFormat)apf];
        if (v) { static int n; if (n++ < 8) MLOG("texture ser=%llu %ux%u format %#x shares the storage of an sRGB / linear twin: a view in Metal format %u", (unsigned long long)t->serial, t->d.width, t->d.height, t->d.format, apf);
            tex_put(t->serial, v); return v; }
        RED_ONCE(146, "no %u view of a shared texture of format %lu: the two formats stay apart", apf, (unsigned long)((id<MTLTexture>)shared).pixelFormat);
    }
    if (shared) { tex_put(t->serial, shared); return shared; }
    id<MTLTexture> mt = tex_linear(t);
    if (!mt) {
        MTLTextureDescriptor *d = tex_desc(t, 0);
        if (!d) return nil;
        mt = image_texture(t, d);
        if (!mt) mt = sparse_upload(t, NULL, d.usage, 1);   // a large one in a sparse pool: only the tiles the game maps take memory (sparse_upload)
        if (!mt) { mt = [g_dev newTextureWithDescriptor:d]; big_tex_log(t, mt, "texture");
            if (mt) { res_add(mt); if (!tex_init_fill(t, mt)) tex_zero_fill(mt); } }
        if (!mt) { RED_ONCE(2, "failed to create MTLTexture %ux%u format %#x target %d", t->d.width, t->d.height, t->d.format, t->d.target); return nil; }
    }
    if (g_allocprof > 0) { static int n; if (n++ < 30 || n % 500 == 0) MLOG("new texture (%s) #%d ser=%llu %ux%ux%u format %#x target %d levels %u storage %p", __func__, n, (unsigned long long)t->serial, t->d.width, t->d.height, t->d.depth, t->d.format, t->d.target, t->d.levels, (void *)t->mem); }
    if (getenv("DM_NVN_TEXCPU") && t->pool && !(t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS)) MLOG("non-swizzled texture on a CPU-visible pool ser=%llu %ux%ux%u format %#x target %d levels %u texture flags %#x pool flags %#x storage %p", (unsigned long long)t->serial, t->d.width, t->d.height, t->d.depth, t->d.format, t->d.target, t->d.levels, t->d.flags, t->pool->flags, (void *)t->mem);
    res_add(mt);
    pthread_mutex_lock(&g_smu);
    if (!g_alias_set) g_alias_set = CFSetCreateMutable(NULL, 0, NULL);
    alias_take_over(&ak);
    hm_put_id(&g_alias, &ak, mt); CFSetAddValue(g_alias_set, (__bridge const void *)mt); nvn_hm_put(&g_alias_seen, &ak, (void *)(uintptr_t)(g_fence + 1));
    pthread_mutex_unlock(&g_smu);
    tex_put(t->serial, mt);
    return mt;
}
// Memory readout line (always on, low frequency, capped): resident memory, GPU allocated, deferred-release queue, residency set object count, registered memory pools / peak / tracking table capacity, cumulative texture re-uploads.
// Purpose: pin down "crashes with no log". The system gives no notice before killing a process for memory, so only readouts left beforehand can show which number kept climbing.
// Called by the frame hook every 300 frames; printed only when resident or GPU allocated changed by â‰¥ 64 MB since the last line, otherwise one line every 3000 frames; capped at 400 lines per game. All integers (the frame hook runs on the guest thread's small stack, so no floating-point formatting)
void mtl_mem_stat(unsigned long long frame, unsigned long long rss_mb) {
    static unsigned long long last_rss, last_gpu, last_frame; static int lines; unsigned long long gpu = (unsigned long long)(g_dev.currentAllocatedSize >> 20);
    unsigned long long dr = rss_mb > last_rss ? rss_mb - last_rss : last_rss - rss_mb, dg = gpu > last_gpu ? gpu - last_gpu : last_gpu - gpu;
    if (lines >= 400 || (lines && dr < 64 && dg < 64 && frame - last_frame < 3000)) return; lines++; last_rss = rss_mb; last_gpu = gpu; last_frame = frame;
    pthread_mutex_lock(&g_deferred_mu); unsigned long nd = (unsigned long)g_deferred.count; pthread_mutex_unlock(&g_deferred_mu);
    pthread_mutex_lock(&g_res_mu); unsigned long nr = (unsigned long)g_res_objects.count; pthread_mutex_unlock(&g_res_mu);
    int live = 0, peak = 0, cap = 0; nvn_dirty_stat(&live, &peak, &cap);
    // Diagnostic DM_NVN_POOLLOG: the Metal textures the serial table holds, by kind, and the largest (where the GPU allocation goes)
    { static int on = -1; if (on < 0) on = getenv("DM_NVN_POOLLOG") != NULL;
      if (on) @autoreleasepool {
        NSMutableSet *seen = [NSMutableSet set]; uint64_t by[7] = {0}; unsigned nby[7] = {0}; const char *kn[7] = {"render target", "private", "shared / managed", "on a buffer or heap", "placement sparse", "other", "packaged (uploaded from guest memory)"};
        typedef struct { uint64_t sz; unsigned w, h, d, lv; int pf; uint64_t ser; } big_t; big_t top[12] = {0};
        pthread_mutex_lock(&g_smu);
        for (uint32_t i = 0; i < g_by_serial.cap; i++) {
            id o = (__bridge id)g_by_serial.vals[i]; if (!o) continue;
            if ([o isKindOfClass:[NvPkg class]]) { NvPkg *pk = o; id<MTLTexture> t = pk->tex; while (t.parentTexture) t = t.parentTexture;
                if (t && ![seen containsObject:t]) { [seen addObject:t]; by[6] += t.allocatedSize; nby[6]++; }
                if (pk->spare && ![seen containsObject:pk->spare]) { [seen addObject:pk->spare]; by[6] += pk->spare.allocatedSize; }
                continue; }
            if (![o conformsToProtocol:@protocol(MTLTexture)]) continue;
            id<MTLTexture> t = o; while (t.parentTexture) t = t.parentTexture;
            if ([seen containsObject:t]) continue; [seen addObject:t];
            int k = t.buffer || t.heap ? 3 : t.isSparse ? 4 : t.usage & MTLTextureUsageRenderTarget ? 0 : t.storageMode == MTLStorageModePrivate ? 1 : t.storageMode != MTLStorageModeMemoryless ? 2 : 5;
            uint64_t sz = k == 3 ? 0 : t.allocatedSize; by[k] += sz; nby[k]++;
            for (int j = 0; j < 12; j++) if (sz > top[j].sz) { memmove(&top[j + 1], &top[j], (11 - j) * sizeof *top);
                top[j] = (big_t){sz, (unsigned)t.width, (unsigned)t.height, (unsigned)(t.textureType == MTLTextureType3D ? t.depth : t.arrayLength), (unsigned)t.mipmapLevelCount, (int)t.pixelFormat, g_by_serial.keys ? *(uint64_t *)((uint8_t *)g_by_serial.keys + (size_t)i * g_by_serial.klen) : 0}; break; }
        }
        pthread_mutex_unlock(&g_smu);
        MLOG("textures up to frame %llu: %s %u %llu MB, %s %u %llu MB, %s %u %llu MB, %s %u, %s %u %llu MB, %s %u %llu MB, %s %u %llu MB", frame, kn[0], nby[0], (unsigned long long)(by[0] >> 20), kn[1], nby[1], (unsigned long long)(by[1] >> 20),
             kn[2], nby[2], (unsigned long long)(by[2] >> 20), kn[3], nby[3], kn[4], nby[4], (unsigned long long)(by[4] >> 20), kn[5], nby[5], (unsigned long long)(by[5] >> 20), kn[6], nby[6], (unsigned long long)(by[6] >> 20));
        for (int j = 0; j < 12 && top[j].sz; j++) MLOG("  largest texture %d: %ux%ux%u, %u levels, Metal format %d, %llu MB (serial %llu)", j + 1, top[j].w, top[j].h, top[j].d, top[j].lv, top[j].pf, (unsigned long long)(top[j].sz >> 20), (unsigned long long)top[j].ser);
      } }
    pthread_mutex_lock(&g_smu); unsigned ns = g_by_serial.n, ashared = g_alias.n, nghost = g_ghost.n; uint64_t ghost_puts = g_ghost_puts;
    unsigned aorph = 0;   // shared textures no live texture object or finalized serial resolves to: kept only until their storage is taken over or their pool goes (alias_take_over)
    { CFMutableSetRef held = CFSetCreateMutable(NULL, 0, NULL);
      for (uint32_t i = 0; i < g_by_serial.cap; i++) if (g_by_serial.vals[i]) CFSetAddValue(held, g_by_serial.vals[i]);
      for (uint32_t i = 0; i < g_ghost.cap; i++) if (g_ghost.vals[i]) CFSetAddValue(held, g_ghost.vals[i]);
      for (uint32_t i = 0; i < g_alias.cap; i++) if (g_alias.vals[i] && !CFSetContainsValue(held, g_alias.vals[i])) aorph++;
      CFRelease(held); }
    pthread_mutex_unlock(&g_smu);
    unsigned long dead_n = 0; unsigned long long dead_mb = 0;
    // Objects read from a weak table come back retained + autoreleased: the frame hook runs on a guest thread with no autorelease pool that drains, so without this wrapper every buffer counted would be pinned for good (the readout creating the leak itself)
    int pres = 0; @autoreleasepool { pthread_mutex_lock(&g_mu); pres = g_pool_res_n; for (id<MTLBuffer> b in g_dead_pools) { dead_n++; dead_mb += b.length; } pthread_mutex_unlock(&g_mu); } dead_mb >>= 20;   // host objects held by serial (textures / packaged textures / programs): if it climbs along with GPU allocated, textures are not being released
    // os_proc_available_memory: how far this process is from being terminated by the system for memory, iOS only; the Mac writes "none". On a phone, a crash with no log is most likely this number reaching 0
    char avail[32] = "none";
#if TARGET_OS_IPHONE
    snprintf(avail, sizeof avail, "%llu MB", (unsigned long long)(os_proc_available_memory() >> 20));
#endif
    MLOG("memory up to frame %llu: resident %llu MB, GPU allocated %llu MB, deferred-release queue %lu, residency set %lu, serial objects %u, memory pools %d (resident %d; peak %d / tracking table capacity %d; freed but alive %lu, %llu MB), shared textures %u (%u of them unreferenced; evicted with pools %llu, on storage takeover %llu in total), texture re-uploads %llu in total, per-frame arena blocks created %llu in total, finalized textures still referenced %u (small table, %llu put in total); headroom to the system memory limit %s", frame, rss_mb, gpu, nd, nr, ns, live, pres, peak, cap, dead_n, dead_mb, ashared, aorph, (unsigned long long)g_alias_evicted, (unsigned long long)g_alias_taken, (unsigned long long)g_pkg_reuploads, (unsigned long long)g_arena_made, nghost, (unsigned long long)ghost_puts, avail);
    { void mtl_vc_stats(unsigned *, uint64_t *, uint64_t *), mtl_tb2_stats(uint64_t *, uint64_t *, uint64_t *, uint64_t *, uint64_t *); uint64_t mtl_tex_mis_count(void); unsigned vn; uint64_t vh, vm, tb, tm, tf, tc, tcb;
      mtl_vc_stats(&vn, &vh, &vm); mtl_tb2_stats(&tb, &tm, &tf, &tc, &tcb);
      MLOG("texture views by texture (vc_get): %u textures hold cached views; %llu views reused, %llu made in total; texture buffers bound as 2D rows for wrapped 1D fetches %llu times, %llu made, %llu failed, %llu copied into aligned rows (%llu MB) "
           "(tbuf_as_2d); texture type mismatches no view bridges %llu in total (tex_mis_note)",
           vn, (unsigned long long)vh, (unsigned long long)vm, (unsigned long long)tb, (unsigned long long)tm, (unsigned long long)tf, (unsigned long long)tc, (unsigned long long)(tcb >> 20), (unsigned long long)mtl_tex_mis_count()); }
    if (g_allocprof > 0) { char ab[256]; int o = 0; for (int i = 0; i < g_alloc_n && o < 200; i++) o += snprintf(ab + o, sizeof ab - o, " #%d %llu MB", i, (unsigned long long)(g_alloc4[i].allocatedSize >> 20));
        for (int i = 0; i < 3 && o < 240; i++) o += snprintf(ab + o, sizeof ab - o, " up%d %llu pres%d %llu", i, (unsigned long long)(g_up_alloc4[i].allocatedSize >> 20), i, (unsigned long long)(g_present_alloc4[i].allocatedSize >> 20));
        MLOG("command allocator usage:%s", ab);
        MLOG("GPU allocation growth by function: pass end %llu MB, pass begin %llu MB, pipeline fetch %llu MB, argument buffers %llu MB, texture fetch %llu MB", g_ap_fn[0] >> 20, g_ap_fn[1] >> 20, g_ap_fn[2] >> 20, g_ap_fn[3] >> 20, g_ap_fn[4] >> 20); }
    if (g_allocprof > 0) { int top[4] = {-1, -1, -1, -1}; for (int o = 0; o < 64; o++) for (int k = 0; k < 4; k++) if (top[k] < 0 || g_ap_bytes[o] > g_ap_bytes[top[k]]) { for (int j = 3; j > k; j--) top[j] = top[j - 1]; top[k] = o; break; }
        MLOG("GPU allocation growth by op: op%d %llu MB, op%d %llu MB, op%d %llu MB, op%d %llu MB", top[0], g_ap_bytes[top[0]] >> 20, top[1], g_ap_bytes[top[1]] >> 20, top[2], g_ap_bytes[top[2]] >> 20, top[3], g_ap_bytes[top[3]] >> 20); }
}
void mtl_pkg_stats(uint64_t *uploads, uint64_t *reuploads) { *uploads = g_pkg_uploads; *reuploads = g_pkg_reuploads; }
uint64_t mtl_pkg_edge_skips(void) { return g_pkg_edge_skips; }
uint64_t mtl_pkg_miss(void) { return g_pkg_miss; }
void mtl_vc_drop(uint64_t ser);
static void treg_retire_serial(uint64_t ser);
void mtl_tex_release(tex_t *t) {
    if (!t->serial) return;
    pthread_mutex_lock(&g_smu); nvn_hm_del(&g_tex_live, &t->serial); free(nvn_hm_del(&g_image_desc, &t->serial));
    pthread_mutex_lock(&g_cpu_tex_mu);
    void *pool = nvn_hm_del(&g_cpu_tex, &t->serial);
    if (pool && t->pool && t->mem) nvn_hm_put(&g_cpu_tex_retired, &t->serial, pool);
    pthread_mutex_unlock(&g_cpu_tex_mu);
    pthread_mutex_unlock(&g_smu);
    serial_del(t->serial);
    mtl_vc_drop(t->serial);
    treg_retire_serial(t->serial);
}

// ---------- Render passes: SetRenderTargets opens a new pass; clears before the first command in the pass are folded into the load action ----------
static struct {
    p_set_rt rt; int has_rt;
    uint8_t clear_c[8]; MTLClearColor cc[8];
    uint8_t clear_d, clear_s; double cd; uint32_t cs;
    p_scissor sc; int has_sc;
} R;

// ---------- Zero-source skip ----------
// Shader fact table (statically proven keys): 'the textures it samples are all 0 and the coefficient is finite â‡’ the output is all 0'. Coefficient = component comp of vector vec of fragment-stage constant buffer register ubo_slot.
//   No shader is in the table yet: a proof comes from reading the shader's translation line by line (dmsh_b/<chain version>/<key>_b.metal); unit tests add their own (zskip_test_add).
typedef struct { char key[17]; int ubo_slot, vec, comp; } zskip_fact;
static zskip_fact g_zskip_facts[16];
static int g_zskip_nf;
void zskip_test_add(const char *key, int ubo_slot, int vec, int comp) {
    if (!key || g_zskip_nf >= 16) return;
    snprintf(g_zskip_facts[g_zskip_nf].key, sizeof g_zskip_facts[0].key, "%s", key);
    g_zskip_facts[g_zskip_nf].ubo_slot = ubo_slot; g_zskip_facts[g_zskip_nf].vec = vec; g_zskip_facts[g_zskip_nf].comp = comp; g_zskip_nf++;
}
int zskip_test_count(void) { return g_zskip_count; }
static int g_zskip_env = -1;   // Cache of DM_NVN_ZSKIP (âˆ’1 = not read). This function is called on every draw, and getenv once took about 8% of the submit path; unit tests set it back to âˆ’1 after toggling the switch mid-run
static int zskip_on(void) { if (g_zskip_env < 0) { const char *e = getenv("DM_NVN_ZSKIP"); g_zskip_env = !e || strcmp(e, "0"); } return g_zskip_env && tbdr_on(); }
static void zskip_clear(void) { g_zskip_n = 0; memset(g_zskip_cand_ok, 0, sizeof g_zskip_cand_ok); }
static void zskip_del(uint64_t ser) { for (int i = 0; i < g_zskip_n;) if (g_zskip_zero[i].ser == ser) g_zskip_zero[i] = g_zskip_zero[--g_zskip_n]; else i++; }
static int zskip_has(uint64_t ser, int lv, int sl) { for (int i = 0; i < g_zskip_n; i++) if (g_zskip_zero[i].ser == ser && g_zskip_zero[i].level == lv && g_zskip_zero[i].slice == sl) return 1; return 0; }
static void zskip_add(uint64_t ser, int lv, int sl) { if (!ser || zskip_has(ser, lv, sl) || g_zskip_n >= 4096) return; g_zskip_zero[g_zskip_n++] = (zskip_zero){ser, (uint16_t)lv, (uint16_t)sl}; }
static int zskip_zero_clear(int i) { return R.clear_c[i] && R.cc[i].red == 0 && R.cc[i].green == 0 && R.cc[i].blue == 0 && R.cc[i].alpha == 0; }
static void zskip_commit_clears(void) { for (int i = 0; i < 8; i++) if (g_zskip_cand_ok[i]) zskip_add(g_zskip_cand[i].ser, g_zskip_cand[i].level, g_zskip_cand[i].slice); }
static int zskip_target_zero(const tex_t *t, const view_t *v, int hv) { int lv = hv && v->set_levels ? v->base_level : 0, sl = hv && v->set_layers ? v->min_layer : 0; return t && zskip_has(t->serial, lv, sl); }
#ifdef NVN_RT_RANGE_TEST
static int g_rr_checking;
static void rr_begin(void), rr_finish(int ranged);
static void rr_snap(const p_draw *, const p_draw_idx *);
static void rr_reset(void);
#endif
static id<MTL4RenderCommandEncoder> g_enc;
static int enc_open(void) { return g_enc != nil; }
// Target info of the current render pass: determined once from the attachments at rp_begin and shared by all draws in the pass (pipeline key, depth-stencil, viewport / scissor all depend only on it)
static struct { uint32_t cfmt[8], dfmt, sfmt, samples; int hd, hs; NSUInteger tw, th; uint64_t serial, reg_gen, ser_gen, range_dirty; int range_external, range_depth_nonro, layered; __unsafe_unretained id<MTLTexture> range_color, range_depth; } g_pass;
static int g_rp_layered;   // the pass about to open is for a vertex shader that writes the layer (draw â†’ rp_begin_ex)
// Slice-merged passes (DM_NVN_SLICE_MERGE, on by default; 0 off): consecutive passes that each draw into another layer of one level of the same array or cube texture,
// with nothing between them but state, uniform updates and draws (no game barrier, clear, copy or dispatch: sm_scan), run as one layered pass. Each draw reaches its layer through a vertex
// amplification view mapping (count 1, renderTargetArrayIndexOffset), which needs pipelines built with maxVertexAmplificationCount 2 (sm_amp_get; a pipeline with the default 1 ignores
// the mapping and every draw lands in the first layer). On Apple GPUs a render pass costs about 35-40 us whatever it draws (M2 Max): Breath of the Wild filters its environment cube map
// every tenth frame in 132 one-draw passes, 22 levels of 6 faces, and that frame's GPU time grew by about 5 ms
static struct { int on, lo, hi; } g_sm_plan;   // set by draw() for the pass rp_begin_ex is about to open
static int g_sm_pass, g_sm_lo, g_sm_hi, g_sm_cur, g_sm_applied;   // the open pass is slice-merged: its attachment holds layers lo..hi; cur: the layer the game's target names now; applied: the offset set on the encoder (-1 none)
static uint64_t g_sm_npass, g_sm_ncont, g_sm_nwait;   // slice-merged passes, layer switches taken into them, draws that left one because their pipeline was not built yet
static int g_pso_amp;   // pso_get: the pipeline of a slice-merged pass (sm_amp_get)
void mtl_sm_stats(uint64_t *np, uint64_t *nc, uint64_t *nw) { *np = g_sm_npass; *nc = g_sm_ncont; *nw = g_sm_nwait; }
static id<MTLRenderPipelineState> g_enc_pso; static id<MTLDepthStencilState> g_enc_dss;   // Already set on this encoder (not set again if identical)
// Dynamic state already sent on the open render encoder: each draw used to send all nine (stencil reference, viewport, scissor, cull, winding, fill, depth bias, depth clip, blend colour);
// now a value is sent only when it differs. g_ed.ok holds one bit per item and is cleared for every new encoder and after a clear triangle (which sets its own)
static struct { unsigned ok; uint32_t sref; MTLViewport vp; MTLScissorRect sc; int cull, wind, fill, clip; float bias[3], blend[4]; } g_ed;
enum { ED_SREF = 1, ED_VP = 2, ED_SC = 4, ED_CULL = 8, ED_WIND = 16, ED_FILL = 32, ED_BIAS = 64, ED_CLIP = 128, ED_BLEND = 256 };
static int g_enc_vis, g_enc_has_vis;   // This encoder's current visibility count mode (0 off / slot number + 1), and whether a visibility buffer is attached
static int g_pso_dirty = 1, g_dss_dirty = 1;   // Pipeline / depth-stencil related state was changed during replay
static uint64_t g_stat_args_enc, g_stat_args_reuse;   // Argument buffer: new-encode / reuse counts
static uint64_t g_stat_draw, g_stat_pass, g_bw_load, g_bw_store, g_pass_full_rw, g_udc_saved_bytes;   // Bandwidth ledger: attachment load / store bytes, number of full load-store passes
// DM_NVN_PASSDUMP: records pass attachments / samples / presents one by one (textures by serial) for offline data-flow analysis; the pass numbers of P / D / R / S lines agree (= g_stat_pass after the pass is opened)
// DM_NVN_PASSDUMP=frame number, or start-end (inclusive at both ends)
static int g_kd_force;   // the draw DM_NVN_KEYDUMP matched: force the per-binding records on (see kd_draw)
static _Atomic long g_pd_a0 = -2, g_pd_a1;   // the frames dumped (inclusive); -2 = DM_NVN_PASSDUMP not read yet, -3 = DM_NVN_PASSDUMP=every:<n>
static long g_pd_every;
static inline int passdump(void) {
    if (g_kd_force) return 1;
#ifdef NVN_RT_RANGE_TEST
    if (g_rr_checking) return 0;
#endif
    if (__builtin_expect(g_pd_a0 == -1, 1)) return 0;   // off (the usual case): answered first, it is asked for every binding of every draw
    if (g_pd_a0 == -2) { const char *e = getenv("DM_NVN_PASSDUMP");
        if (e && !strncmp(e, "every:", 6)) { g_pd_every = atol(e + 6); g_pd_a1 = -1; g_pd_a0 = g_pd_every > 0 ? -3 : -1; }   // one frame in every n, halfway between two DM_NVN_GPUPROF=every:n frames
        else { g_pd_a1 = e && strchr(e, '-') ? atol(strchr(e, '-') + 1) : e ? atol(e) : -1; g_pd_a0 = e ? atol(e) : -1; } }
    long f = (long)atomic_load(&g_cur_frame), a0 = g_pd_a0;
    if (a0 == -3) return f > 0 && f % g_pd_every == g_pd_every / 2;
    return a0 >= 0 && f >= a0 && f <= g_pd_a1;
}
// The next n frames' passes, on request (the Mac's DM_CTL "passdump n", hle_hid.c): for a scene that a fixed DM_NVN_PASSDUMP frame range cannot be timed to
void dm_gfx_passdump_next(long n) { long f = (long)atomic_load(&g_cur_frame); g_pd_a1 = f + n; g_pd_a0 = f + 1; }
#define PD(...) do { if (passdump()) { flockfile(stderr); fprintf(stderr, "[pass] F%llu ", (unsigned long long)atomic_load(&g_cur_frame)); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); funlockfile(stderr); } } while (0)
static void blit_note(void) { if (g_enc) PD("E %llu copy / buffer update ends the render pass", g_stat_pass); }
__attribute__((weak)) void dm_guest_frame_tick(void) {}   // hle_os.c provides the strong definition (DM_DETCLOCK deterministic guest clock); a no-op when graphics unit tests do not link the host   // diag.c provides the strong definition (DM_TICK_HOLD=E:SHOT unfreeze); a no-op when graphics unit tests do not link the host
__attribute__((weak)) void dm_cheats_present(void) {}   // hle_cheats.c provides the strong definition (the AtmosphÃ¨re cheat pass per presented frame); a no-op when graphics unit tests do not link the host
__attribute__((weak)) int dm_detclock_enabled(void) { return 0; }   // Same as above: the host's hle_os.c provides the strong definition; when graphics unit tests do not link the host, the deterministic clock is treated as off
__attribute__((weak)) uint64_t guest_ns(void) { return 0; }          // Same as above: read only when the deterministic clock is on; never reached in graphics unit tests
static uint64_t att_bytes(id<MTLTexture> t, NSUInteger level) {
    if (!t) return 0;
    NSUInteger w = MAX(t.width >> level, 1), h = MAX(t.height >> level, 1);
    NSUInteger bpp = t.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 ? 5 : t.pixelFormat == MTLPixelFormatRGBA16Float || t.pixelFormat == MTLPixelFormatRG32Float ? 8 :
                     t.pixelFormat == MTLPixelFormatRGBA32Float ? 16 : t.pixelFormat == MTLPixelFormatR8Unorm || t.pixelFormat == MTLPixelFormatStencil8 ? 1 :
                     t.pixelFormat == MTLPixelFormatRG8Unorm || t.pixelFormat == MTLPixelFormatR16Float ? 2 : 4;
    return (uint64_t)w * h * bpp * t.sampleCount;
}
// The render attachment selects a particular mip, while MTLTexture.width/height still describe the base level. Empty attachments do not take part in the bounds.
static NSUInteger mip_extent(NSUInteger size, NSUInteger level) {
    return level < sizeof(size) * 8 ? MAX(size >> level, 1) : 1;
}
static void pass_dims(MTL4RenderPassDescriptor *pd, NSUInteger nc, NSUInteger *w, NSUInteger *h) {
    *w = *h = 0;
    for (NSUInteger i = 0; i < nc + 2; i++) {
        MTLRenderPassAttachmentDescriptor *a = i < nc ? pd.colorAttachments[i] : i == nc ? pd.depthAttachment : pd.stencilAttachment;
        if (!a.texture) continue;
        NSUInteger aw = mip_extent(a.texture.width, a.level), ah = mip_extent(a.texture.height, a.level);
        *w = *w ? MIN(*w, aw) : aw; *h = *h ? MIN(*h, ah) : ah;
    }
    if (!*w) { *w = MAX(pd.renderTargetWidth, 1); *h = MAX(pd.renderTargetHeight, 1); }   // attachment-less pass (rp_begin_ex_): its explicit size
}

// ---------- DM_NVN_PASSGPU=<file>: per-pass GPU time (stage-boundary timestamp counters) ----------
// Each render pass samples 4 points (vertex start/end, fragment start/end), a compute pass 2 points; resolved after the command buffer completes and accumulated by 'index of the pass within the frame'.
// Each line: frame idx kind gpu_ms inc_ms w h fmt0 dfmt load draws t0_ns t1_ns (kind: R render / C compute; load: L load / C clear for each color attachment + depth)
// On TBDR passes overlap in the pipeline (the later pass's vertex stage is earlier than the end of the earlier pass's fragment stage): gpu_ms = this pass's start-to-end span (including queueing) and cannot be summed;
// attribution uses the timeline increment inc_ms = t1 âˆ’ max(t0, latest t1 of the passes completed so far); the sum of the increments of all passes = total time the GPU is busy (command buffers complete in submission order)
typedef struct { uint64_t frame, pass; int idx, kind, w, h, draws, slot; uint32_t cfmt0, dfmt; char load[12]; } pg_pass;   // pass: global pass number (same as the pass number in the DM_NVN_PASSDUMP lines, for joining the two records)
#define PG_SLOTS 4096   // Counter sample buffer limit 32768 bytes = 4096 timestamps (measured from the error on this machine)
#define PG_RING 8
static int g_pg = -1; static FILE *g_pg_f;
static id<MTLCounterSampleBuffer> g_pg_buf[PG_RING]; static int g_pg_b; static uint32_t g_pg_used;
static pg_pass *g_pg_pass; static int g_pg_np, g_pg_cap, g_pg_open = -1;
static uint64_t g_pg_frame = ~0ull; static int g_pg_idx;
static pthread_mutex_t g_pg_mu = PTHREAD_MUTEX_INITIALIZER;
static inline void args_dirty_all(void);
typedef struct { double sum, last; uint64_t n; pg_pass meta; } pg_stat;
static pg_stat g_pg_st[256]; static uint64_t g_pg_rep_frame;
static int pg_on(void) {
    if (g_pg >= 0) return g_pg;
    const char *e = getenv("DM_NVN_PASSGPU"); g_pg = 0;
    if (!e) return 0;
    if (![g_dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) { MLOG("[RED] DM_NVN_PASSGPU: this device does not support counter sampling at stage boundaries"); return 0; }
    id<MTLCounterSet> ts = nil; for (id<MTLCounterSet> c in g_dev.counterSets) if ([c.name isEqualToString:MTLCommonCounterSetTimestamp]) ts = c;
    if (!ts) { MLOG("[RED] DM_NVN_PASSGPU: no timestamp counter set"); return 0; }
    MTLCounterSampleBufferDescriptor *d = [MTLCounterSampleBufferDescriptor new];
    d.counterSet = ts; d.storageMode = MTLStorageModeShared; d.sampleCount = PG_SLOTS;
    for (int i = 0; i < PG_RING; i++) { NSError *er = nil; g_pg_buf[i] = [g_dev newCounterSampleBufferWithDescriptor:d error:&er]; if (!g_pg_buf[i]) { MLOG("[RED] cannot create the counter sample buffer: %s", er.localizedDescription.UTF8String); return 0; } }
    if (!(g_pg_f = fopen(e, "w"))) { MLOG("[RED] DM_NVN_PASSGPU cannot open %s", e); return 0; }
    fprintf(g_pg_f, "frame\tidx\tkind\tgpu_ms\tinc_ms\tw\th\tfmt0\tdfmt\tload\tdraws\tt0_ns\tt1_ns\tpass\n");
    MLOG("[passgpu] per-pass GPU timing â†’ %s", e);
    return g_pg = 1;
}
static void pg_sub_begin(uint64_t fence) {   // This submission uses block fence % PG_RING of the ring; first wait for the submission that used it PG_RING submissions ago to complete
    if (!pg_on()) return;
    if (fence > PG_RING) mtl_wait_at(fence - PG_RING, UINT64_MAX, WS_PG);
    g_pg_b = (int)(fence % PG_RING); g_pg_used = 0; g_pg_np = 0; g_pg_open = -1;
}
static pg_pass *pg_new(int kind, int nslot) {
    if (!pg_on() || g_pg_used + nslot > PG_SLOTS) return NULL;
    uint64_t f = atomic_load(&g_cur_frame); if (f != g_pg_frame) { g_pg_frame = f; g_pg_idx = 0; }
    if (g_pg_np == g_pg_cap) { g_pg_cap = g_pg_cap ? g_pg_cap * 2 : 128; g_pg_pass = realloc(g_pg_pass, g_pg_cap * sizeof *g_pg_pass); }
    pg_pass *p = &g_pg_pass[g_pg_np++]; memset(p, 0, sizeof *p);
    p->frame = f; p->pass = g_stat_pass; p->idx = g_pg_idx++; p->kind = kind; p->slot = (int)g_pg_used; p->draws = -(int)g_stat_draw;
    g_pg_used += nslot;
    return p;
}
static void pg_render(MTL4RenderPassDescriptor *pd, NSUInteger nc) {
    (void)pd; (void)nc;
    return;
#if 0
    pg_pass *p = pg_new(0, 4); if (!p) return;
    MTLRenderPassSampleBufferAttachmentDescriptor *a = pd.sampleBufferAttachments[0];
    a.sampleBuffer = g_pg_buf[g_pg_b];
    a.startOfVertexSampleIndex = p->slot; a.endOfVertexSampleIndex = p->slot + 1; a.startOfFragmentSampleIndex = p->slot + 2; a.endOfFragmentSampleIndex = p->slot + 3;
    id<MTLTexture> t0 = nil; int k = 0;
    for (NSUInteger i = 0; i < nc && i < 8; i++) { id<MTLTexture> t = pd.colorAttachments[i].texture; if (!t) continue;
        if (!t0) t0 = t; if (k < 10) p->load[k++] = pd.colorAttachments[i].loadAction == MTLLoadActionClear ? 'C' : 'L'; }
    id<MTLTexture> dt = pd.depthAttachment.texture;
    if (dt) { if (k < 10) { p->load[k++] = '|'; p->load[k++] = pd.depthAttachment.loadAction == MTLLoadActionClear ? 'C' : 'L'; } p->dfmt = (uint32_t)dt.pixelFormat; }
    if (!t0) t0 = dt;
    NSUInteger pw, ph; pass_dims(pd, nc, &pw, &ph);
    p->w = (int)pw; p->h = (int)ph; p->cfmt0 = t0 && t0 != dt ? (uint32_t)t0.pixelFormat : 0;
    g_pg_open = g_pg_np - 1;
#endif
}
static id<MTLComputeCommandEncoder> pg_compute(void) {
    pg_pass *p = pg_new(1, 2);
    ser_gate();
    if (!p) { args_dirty_all(); return enc_c_begin(NULL); }
    MTLComputePassDescriptor *cd = [MTLComputePassDescriptor computePassDescriptor];
    cd.sampleBufferAttachments[0].sampleBuffer = g_pg_buf[g_pg_b];
    cd.sampleBufferAttachments[0].startOfEncoderSampleIndex = p->slot; cd.sampleBufferAttachments[0].endOfEncoderSampleIndex = p->slot + 1;
    p->draws = 0;
    args_dirty_all();
    return enc_c_begin(cd);
}
static void pg_end_render(void) { if (g_pg_open >= 0) { g_pg_pass[g_pg_open].draws += (int)g_stat_draw; g_pg_open = -1; } }
static void pg_commit(id<MTLCommandBuffer> cb) {   // Hand this submission's pass table to the completion callback for resolving
    if (!g_pg || !g_pg_np) return;
    int n = g_pg_np; pg_pass *ps = malloc(n * sizeof *ps); memcpy(ps, g_pg_pass, n * sizeof *ps);
    id<MTLCounterSampleBuffer> buf = g_pg_buf[g_pg_b]; uint32_t used = g_pg_used;
    [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
        (void)c;
        NSData *d = [buf resolveCounterRange:NSMakeRange(0, used)];
        const MTLCounterResultTimestamp *ts = d.bytes;
        pthread_mutex_lock(&g_pg_mu);
        for (int i = 0; d && i < n; i++) {
            pg_pass *p = &ps[i]; uint64_t a, b;
            #define OK(x) ((x) != MTLCounterErrorValue && (x) != 0)
            if (p->kind == 0) { a = OK(ts[p->slot].timestamp) ? ts[p->slot].timestamp : ts[p->slot + 2].timestamp; b = OK(ts[p->slot + 3].timestamp) ? ts[p->slot + 3].timestamp : ts[p->slot + 1].timestamp; }
            else { a = ts[p->slot].timestamp; b = ts[p->slot + 1].timestamp; }
            if (!OK(a) || !OK(b) || b < a) continue;
            #undef OK
            static uint64_t prev_end;   // Latest end time of the passes completed so far (completion callbacks are serialized in submission order)
            double ms = (b - a) / 1e6;   // Timestamps are in ns, same as the CPU (measured 1:1 via sampleTimestamps on this machine)
            double inc = b > prev_end ? (b - (a > prev_end ? a : prev_end)) / 1e6 : 0;
            if (b > prev_end) prev_end = b;
            fprintf(g_pg_f, "%llu\t%d\t%c\t%.4f\t%.4f\t%d\t%d\t%u\t%u\t%s\t%d\t%llu\t%llu\t%llu\n", (unsigned long long)p->frame, p->idx, p->kind ? 'C' : 'R', ms, inc,
                    p->w, p->h, p->cfmt0, p->dfmt, p->load, p->draws, (unsigned long long)a, (unsigned long long)b, (unsigned long long)p->pass);
            if (p->idx < 256 && p->frame > 300) { pg_stat *st = &g_pg_st[p->idx]; st->sum += inc; st->n++; st->last = inc; st->meta = *p; }
            if (p->frame >= g_pg_rep_frame + 300 && p->frame > 300) {   // Report the top five every 300 frames (accumulated from frame 300)
                g_pg_rep_frame = p->frame; double tot = 0; int top[5] = {-1, -1, -1, -1, -1};
                for (int k = 0; k < 256; k++) if (g_pg_st[k].n) { double m = g_pg_st[k].sum / g_pg_st[k].n; tot += m;
                    for (int j = 0; j < 5; j++) if (top[j] < 0 || m > g_pg_st[top[j]].sum / g_pg_st[top[j]].n) { for (int q = 4; q > j; q--) top[q] = top[q - 1]; top[j] = k; break; } }
                char line[1024]; size_t o = 0;
                for (int j = 0; j < 5 && top[j] >= 0; j++) { pg_stat *t = &g_pg_st[top[j]];
                    o += snprintf(line + o, sizeof line - o, " #%d %.2f ms %dx%d fmt%u d%u %s %d draws;", top[j], t->sum / t->n, t->meta.w, t->meta.h, t->meta.cfmt0, t->meta.dfmt, t->meta.load, t->meta.draws); }
                MLOG("[passgpu] up to frame %llu: sum of per-pass means per frame %.2f ms; top five:%s", (unsigned long long)p->frame, tot, line);
                fflush(g_pg_f);
            }
        }
        pthread_mutex_unlock(&g_pg_mu);
        free(ps);
    }];
}
void ts_after(void);
void ms_after(void);
// ---------- Store lookahead: at pass end, decide from the later commands of the same submission whether each attachment's write-back is redundant ----------
// On TBDR every attachment is written back to memory in full when the pass ends; Switch (Maxwell immediate-mode rendering) has no such cost. If an attachment is cleared in full before being read, the write-back is redundant (DontCare).
// For the decision see mtl_sub_index / sdx_decide; only what can be proven within the same submission is accepted, and anything still undecided at the end of the submission is always written back.
typedef struct { uint64_t ser, bytes; uint32_t lv, sl; int idx; uint8_t depth, state, ridx, drawn, why, load, wrote, wwhy, img, rast; const char *wkey; int wslot; } sd_att;   // depth: 0 color 1 depth 2 stencil (the two parts of a combined depth-stencil texture are decided separately)   // state 0 undecided 1 needed 2 redundant; idx: color attachment index, -1 for depth
// Depth and stencil have independent store decisions, even in one combined texture.
// Eight color targets therefore need ten records (XC2 uses the full set).
enum { SD_ATT_MAX = 8 + 2 };
static sd_att g_sd[SD_ATT_MAX]; static int g_nsd;
static int g_udc_ab = -1, g_udc_ab_on;
// Attachments this pass did not write (depth / stencil test only, color write mask 0, etc.) get their write-back changed to DontCare: the content is already in memory, saving one full write-back.
// On by default: on Mac, same-process on / off screenshots follow the same distribution as the null control; on A19 Pro, private Depth32F_S8 / RGBA8 read back after DontCare differ in 0 places.
// DM_NVN_UNWRITTEN_DC=0 turns optional savings off (for equivalence comparison); the image-only color store rule in sd_finish remains mandatory.
static int udc_on(void) {
    if (g_udc_ab < 0) { const char *e = getenv("DM_NVN_UNWRITTEN_DC"); g_udc_ab = !(e && !strcmp(e, "0"));
        const char *a = getenv("DM_NVN_UDC_AB"); if (a && !strcmp(a, "1")) { g_udc_ab = 0; g_udc_ab_on = 0; } }
    return g_udc_ab || g_udc_ab_on;
}
// DM_NVN_TBDR=0 disables lookahead and pass merging. DM_NVN_UNWRITTEN_DC=0 also disables optional unwritten-attachment savings; neither disables image-write preservation.
// Only for equivalence comparison (the same replay run on / off, comparing the picture byte by byte), not a product configuration
// Mario: off by default (only DM_NVN_TBDR=1 turns it on). These optimizations infer read / write relations from "which textures are bound in texture slots", while 38% of this game's shaders use bindless textures (handles in uniform buffers, NVN_RB_TEXU),
//   so the inference would miss readers - e.g. deciding an attachment is never read afterwards and not storing it. Turn them back on once they recognize bindless readers.
static int tbdr_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_NVN_TBDR"); v = e && !strcmp(e, "1"); if (!v) MLOG("TBDR optimizations off by default (this game uses bindless textures; DM_NVN_TBDR=1 turns them on)"); } return v; }
static int rt_range_on(void) { static int v = -1; if (v < 0) { const char *e=getenv("DM_NVN_RT_RANGE"); v=e && !strcmp(e,"1"); } return v; }   // Narrow candidate, off by default
// ---------- In-process A/B phases (DM_AB, measurement) ----------
// DM_AB=<frames> splits the run into phases of that many frames (frame f is in phase f / frames) and alternates the switches whose value is ab (sw_mode; DM_NVN_BARRIER=ab so far):
// ab = on in every second phase (phase 0 with all of them off), ab:<k>[,<k>...]/<n> = on in the phases whose number modulo n is one of the k (four configurations of two switches: one
// ab:1,3/4, the other ab:2,3/4). Both configurations then run in the same process, scene and thermal state, which separate runs on a phone cannot give. A switch reads its mode once
// (sw_mode) and asks sw_on at each use; one that must stay the same over a pass latches it when the pass begins
static _Atomic long g_ab_phase; static long g_ab_len = -1;
static long ab_len(void) { if (g_ab_len < 0) { const char *e = getenv("DM_AB"); g_ab_len = e ? atol(e) : 0; if (g_ab_len < 0) g_ab_len = 0; if (g_ab_len) MLOG("DM_AB: switches set to ab alternate every %ld frames (phase 0 with all of them off)", g_ab_len); } return g_ab_len; }
enum { SW_MAX = 8 }; static const char *g_sw_name[SW_MAX]; static int g_sw_mode[SW_MAX], g_sw_n;
static inline int sw_on_at(int m, long ph) { if (m < 2) return m; int n = (m - 2) & 15, mask = (m - 2) >> 4; return mask >> (ph % n) & 1; }
// The mode of switch <name>: 0 off, 1 on, 2 + n + 16 * mask = on in the phases whose number modulo n is in mask; ab without DM_AB is off
static int sw_mode(const char *name, int dflt) {
    const char *e = getenv(name); int m = dflt, mask = 2, n = 2;
    if (e && !strncmp(e, "ab", 2)) {
        if (e[2] == ':') { const char *sl = strchr(e, '/'); int nn = sl ? atoi(sl + 1) : 0, mk = 0;
            for (const char *q = e + 3; sl && q < sl; ) { int k = atoi(q); if (k >= 0 && k < 15) mk |= 1 << k; while (q < sl && *q != ',') q++; if (q < sl) q++; }
            if (nn >= 2 && nn <= 15 && mk && mk < (1 << nn)) { mask = mk; n = nn; } }
        m = ab_len() ? 2 + n + 16 * mask : 0;
        if (!ab_len()) MLOG("[RED] %s=%s needs DM_AB=<frames>: the switch is off", name, e); }
    else if (e) m = atoi(e) ? 1 : 0;
    if (m >= 2) MLOG("%s: on in the DM_AB phases whose number modulo %d is in the set %#x", name, n, mask);
    if (g_sw_n < SW_MAX) { g_sw_name[g_sw_n] = name; g_sw_mode[g_sw_n++] = m; }
    return m;
}
static inline int sw_on(int m) { return sw_on_at(m, atomic_load_explicit(&g_ab_phase, memory_order_relaxed)); }
// Each phase is reported once its frames' GPU times have arrived (8 frames late) as an [ab] line: the switches on in it, GPU time and whole frame time (p50 / p90 / mean over the phase
// without its first quarter, while the GPU clock and the frame step settle), and attachment traffic and render passes per frame (the bandwidth ledger)
enum { AB_MAXF = 4096 };
static float g_ab_gpu[AB_MAXF], g_ab_fr[AB_MAXF]; static int g_ab_nf; static long g_ab_ph = -1; static uint64_t g_ab_bw0[2], g_ab_pass0, g_ab_f0;
static uint64_t g_ab_bw1[2], g_ab_pass1, g_ab_f1;   // the counters at the switch to the current phase (taken in ab_frame), copied to *0 when the previous phase is reported
static int flt_cmp(const void *a, const void *b) { float x = *(const float *)a, y = *(const float *)b; return x < y ? -1 : x > y; }
static void ab_frame(uint64_t f) {   // frame f begins (mtl_set_frame): the phase follows the frame number
    long ph = (long)(f / (uint64_t)g_ab_len);
    if (atomic_load(&g_ab_phase) != ph || f % (uint64_t)g_ab_len == 0) { g_ab_bw1[0] = g_bw_load; g_ab_bw1[1] = g_bw_store; g_ab_pass1 = g_stat_pass; g_ab_f1 = f; }
    atomic_store(&g_ab_phase, ph);
}
static void ab_report(long ph) {
    int n = g_ab_nf, skip = n / 4, m = n - skip; if (m < 10) return;
    float *g = malloc(sizeof(float) * (size_t)m), *fr = malloc(sizeof(float) * (size_t)m); double gs = 0, fs = 0;
    for (int i = 0; i < m; i++) { g[i] = g_ab_gpu[skip + i]; fr[i] = g_ab_fr[skip + i]; gs += g[i]; fs += fr[i]; }
    qsort(g, (size_t)m, sizeof *g, flt_cmp); qsort(fr, (size_t)m, sizeof *fr, flt_cmp);
    double nf = g_ab_f1 > g_ab_f0 ? (double)(g_ab_f1 - g_ab_f0) : 1;
    char on[160] = ""; size_t o = 0;
    for (int i = 0; i < g_sw_n && o < sizeof on; i++) if (g_sw_mode[i] >= 2 && sw_on_at(g_sw_mode[i], ph)) {
        const char *nm = g_sw_name[i]; if (!strncmp(nm, "DM_NVN_", 7)) nm += 7; else if (!strncmp(nm, "DM_", 3)) nm += 3;
        o += (size_t)snprintf(on + o, sizeof on - o, "%s%s", o ? "," : "", nm); }
    MLOG("[ab] phase %ld %s: %d frames, GPU p50 %.2f / p90 %.2f / mean %.2f ms, frame p50 %.2f / p90 %.2f / mean %.2f ms; per frame: attachments load %.0f MB store %.0f MB, passes %.1f",
         ph, o ? on : "base", m, g[m / 2], g[m * 9 / 10], gs / m, fr[m / 2], fr[m * 9 / 10], fs / m,
         (g_ab_bw1[0] - g_ab_bw0[0]) / nf / 1048576.0, (g_ab_bw1[1] - g_ab_bw0[1]) / nf / 1048576.0, (g_ab_pass1 - g_ab_pass0) / nf);
    free(g); free(fr);
}
// Once per frame with that frame's GPU time and whole frame time (hle_nvn.c, 8 frames late); the phase is the one frame f ran in
void mtl_ab_gpu(uint64_t f, double gpu_ms, double frame_ms) {
    if (!ab_len()) return;
    long ph = (long)(f / (uint64_t)g_ab_len);
    if (ph != g_ab_ph) {
        if (g_ab_ph >= 0) ab_report(g_ab_ph);
        g_ab_ph = ph; g_ab_nf = 0;
        memcpy(g_ab_bw0, g_ab_bw1, sizeof g_ab_bw0); g_ab_pass0 = g_ab_pass1; g_ab_f0 = g_ab_f1;
    }
    if (g_ab_nf < AB_MAXF) { g_ab_gpu[g_ab_nf] = (float)gpu_ms; g_ab_fr[g_ab_nf] = (float)frame_ms; g_ab_nf++; }
}

static uint64_t g_sd_dead_bytes, g_sd_n_dead, g_sd_n_all;
// ---------- Cross-submission attachment validity ledger ----------
// Entry = subresource (texture serial, level, layer) + state, surviving across submissions:
//   XV_INVALID       content was not written back / not generated and must not be read. The next touch must be a full Clear, or use as an attachment with load action DontCare / Clear; otherwise it is a violation.
//   XV_PENDING_CLEAR reserved for 'clear-only pending across submissions' (not implemented): accepts only full, unscissored clears with the write mask fully open; clearing again while pending = replace the clear value without landing it;
//                    as an attachment with load DontCare / Clear â†’ the pending clear is voided, with load Load â†’ changed to a Clear load; depth-stencil is recorded per plane.
// Violation = report a failure + that stable key (format / width / height / level) permanently disabled (in memory + persisted to DM_XV_FILE, read at the next launch) + all entries of the same texture deleted. The fallback can only detect, not undo (in that frame the attachment holds old content).
// Its first use: color attachments 'scanned to the end undecided' at the end of a submission are not written back (sd_finish) - conditions: its first touch in this submission is a full clear / DontCare load (reset every frame before use), it is not a window texture, and the key is not disabled.
// Switch DM_NVN_XV=0 turns it off; has no effect when tbdr_on()==0.
// Trial period (XV_PROBE): for a key's first XV_TRIAL times it is only 'shadow-accounted' - written back as usual, but its next touch is tracked all the same; only after XV_TRIAL consecutive compliant touches does the write-back really start being skipped.
//   On a read / Load during the trial period: only disable that key (within this process, not persisted, no failure), the picture is intact - in practice there are readers as early as the launch / menu screens (upscale target 36), and not every fresh install should get one wrong frame.
enum { XV_INVALID = 1, XV_PENDING_CLEAR = 2, XV_PROBE = 3 };
#define XV_TRIAL 3
typedef struct { uint64_t ser, tag; uint32_t lv, sl; uint8_t state; } xv_ent;
static xv_ent g_xv[64]; static int g_xv_n, g_xv_block, g_xv_init;
static uint64_t g_xv_disabled[128]; static int g_xv_ndis;
static struct { uint64_t tag; int good; } g_xv_trial[128]; static int g_xv_ntrial;   // Number of consecutive compliant touches for each key
static int xv_trial_get(uint64_t tag, int add) { for (int i = 0; i < g_xv_ntrial; i++) if (g_xv_trial[i].tag == tag) return g_xv_trial[i].good += add; if (g_xv_ntrial < 128) { g_xv_trial[g_xv_ntrial].tag = tag; return g_xv_trial[g_xv_ntrial++].good = add; } return 0; }
static uint64_t g_xv_present[128]; static int g_xv_npresent;
static uint64_t g_xv_dc, g_xv_viol;
typedef struct { uint64_t ser; uint32_t lv, sl; uint8_t good; } xv_touch;
static xv_touch g_xv_touch[128]; static int g_xv_ntouch;
static uint64_t xv_tag(const tex_t *t, uint32_t lv) { return ((uint64_t)t->d.format << 40) ^ ((uint64_t)t->d.width << 24) ^ ((uint64_t)t->d.height << 8) ^ lv; }
// Stable key (unchanged across launches, does not use the texture serial): format / width / height / level, then mixed with the formats of the color attachments of the pass where the write-back is skipped and this attachment's index - textures of the same format and size but different purpose (e.g. transmittance 27 and upscale target 36) do not share a key
static uint64_t xv_tag_pass(const tex_t *t, uint32_t lv, int idx, const p_set_rt *rt) { uint64_t h = xv_tag(t, lv) ^ ((uint64_t)(idx + 1) << 60); for (int i = 0; i < rt->n && i < 8; i++) h = (h ^ (rt->color[i] ? (uint64_t)rt->color[i]->d.format + 1 : 0)) * 0x100000001b3ull; return h; }
static void xv_init(void) {
    if (g_xv_init) return; g_xv_init = 1;
    const char *p = getenv("DM_XV_FILE"); if (!p || !*p) return;
    FILE *f = fopen(p, "r"); if (!f) return;
    char line[256], why[160]; unsigned long long tag; int ch, n = 0;
    while (fgets(line, sizeof line, f) && g_xv_ndis < 128) if (sscanf(line, "%llx\t%159[^\t]\t%d", &tag, why, &ch) >= 1) g_xv_disabled[g_xv_ndis++] = tag, n++;
    fclose(f); MLOG("read %d disable records", n);
}
static int xv_disabled(uint64_t tag) { xv_init(); for (int i = 0; i < g_xv_ndis; i++) if (g_xv_disabled[i] == tag) return 1; return 0; }
static xv_ent *xv_find(uint64_t ser, uint32_t lv, uint32_t sl) { for (int i = 0; i < g_xv_n; i++) if (g_xv[i].ser == ser && g_xv[i].lv == lv && g_xv[i].sl == sl) return &g_xv[i]; return NULL; }
static void xv_drop(uint64_t ser, uint32_t lv, uint32_t sl) { for (int i = 0; i < g_xv_n; i++) if (g_xv[i].ser == ser && g_xv[i].lv == lv && g_xv[i].sl == sl) { g_xv[i] = g_xv[--g_xv_n]; return; } }
static void xv_drop_all_of(uint64_t ser) { for (int i = 0; i < g_xv_n;) if (g_xv[i].ser == ser) g_xv[i] = g_xv[--g_xv_n]; else i++; }
static void xv_set(uint64_t ser, uint32_t lv, uint32_t sl, uint8_t state, uint64_t tag) {
    xv_init(); if (g_xv_block || xv_disabled(tag)) return;
    xv_ent *e = xv_find(ser, lv, sl); if (e) { e->state = state; e->tag = tag; return; }
    if (g_xv_n < 64) g_xv[g_xv_n++] = (xv_ent){ser, tag, lv, sl, state};
}
static void xv_violation(const xv_ent *e, const char *why) {
    xv_init();
    if (e->state == XV_PROBE) {   // Trial period: the content is still there and the picture is intact - only disable the key (this process)
        if (!xv_disabled(e->tag) && g_xv_ndis < 128) g_xv_disabled[g_xv_ndis++] = e->tag;
        PD("V xv reader seen during probation: key=%016llx texture=%llu level=%u reason=%s â†’ key disabled", (unsigned long long)e->tag, (unsigned long long)e->ser, e->lv, why);
        xv_drop_all_of(e->ser); return; }
    g_xv_viol++;
    int seen = xv_disabled(e->tag); if (!seen && g_xv_ndis < 128) g_xv_disabled[g_xv_ndis++] = e->tag;
    MLOG("[RED] xv tag=%016llx texture=%llu level=%u layer=%u reason=%s pass=%llu", (unsigned long long)e->tag, (unsigned long long)e->ser, e->lv, e->sl, why, (unsigned long long)g_stat_pass);
    const char *p = getenv("DM_XV_FILE"); if (p && *p) { FILE *f = fopen(p, "a"); if (f) { fprintf(f, "%016llx\t%s\t%llu\n", (unsigned long long)e->tag, why, (unsigned long long)g_stat_pass); fclose(f); } }
    xv_drop_all_of(e->ser);
}
static void xv_check(uint64_t ser, uint32_t lv, uint32_t sl, const char *why) { xv_ent *e = xv_find(ser, lv, sl); if (e && (e->state == XV_INVALID || e->state == XV_PROBE)) { xv_ent copy = *e; xv_violation(&copy, why); } }
static void xv_good(xv_ent *e) { if (e->state == XV_PROBE) xv_trial_get(e->tag, 1); xv_drop(e->ser, e->lv, e->sl); }   // Compliant touch: counts once toward the trial period, and the entry returns to valid
static void xv_touch_add(uint64_t ser, uint32_t lv, uint32_t sl, int good) { for (int i = 0; i < g_xv_ntouch; i++) if (g_xv_touch[i].ser == ser && g_xv_touch[i].lv == lv && g_xv_touch[i].sl == sl) return; if (g_xv_ntouch < 128) g_xv_touch[g_xv_ntouch++] = (xv_touch){ser, lv, sl, (uint8_t)good}; }
static int xv_first_good(uint64_t ser, uint32_t lv, uint32_t sl) { for (int i = 0; i < g_xv_ntouch; i++) if (g_xv_touch[i].ser == ser && g_xv_touch[i].lv == lv && g_xv_touch[i].sl == sl) return g_xv_touch[i].good; return 0; }
static int xv_present(uint64_t ser) { for (int i = 0; i < g_xv_npresent; i++) if (g_xv_present[i] == ser) return 1; return 0; }
static void xv_forward(void);
static int tbdr_on(void);
static int xv_on(void) { static int off = -1; if (off < 0) { const char *e = getenv("DM_NVN_XV"); off = e && !strcmp(e, "0"); } return !off && tbdr_on(); }
void mtl_xv_window_tex(uint64_t serial) { if (!xv_present(serial) && g_xv_npresent < 128) g_xv_present[g_xv_npresent++] = serial; }   // Window (swapchain) texture: never enters the ledger
void xv_test_counts(int *dc, int *viol) { if (dc) *dc = (int)g_xv_dc; if (viol) *viol = (int)g_xv_viol; }
static void sd_finish(void);
// Load elision (fully covering first draw): if the draw that opens the pass can be proven to cover a color attachment in full (first_draw_covers), that attachment's load becomes DontCare -
// the first draw does not blend and writes all channels, every covered pixel is rewritten by it, and the original content is not visible (Switch immediate-mode rendering has no such cost; on TBDR load = reading the whole thing from memory into the tile)
static const p_draw *g_fd_dw; static const p_draw_idx *g_fd_di;   // The draw that opens the pass (one of non-indexed / indexed); non-null only while draw() is opening a pass
static uint64_t g_fd_n;         // Cumulative number of attachments judged elidable
static int g_fd_why;            // Reason for the most recent failure (Q line of PASSDUMP: 1 no program 2 shader 3 attachment 4 blend/mask/raster 5 depth-stencil test 6 vertex count 7 position attribute 8 format/stream 9 out of bounds 10 address 11 geometry)
static int first_draw_covers(int i, id<MTLTexture> t, NSUInteger level);
static int pf_is_float(MTLPixelFormat f) {   // Float color formats that can store NaN (sentinel choice for the load-elision probe; RGB9E5 cannot store NaN and does not count)
    switch (f) { case MTLPixelFormatR16Float: case MTLPixelFormatRG16Float: case MTLPixelFormatRGBA16Float: case MTLPixelFormatR32Float:
        case MTLPixelFormatRG32Float: case MTLPixelFormatRGBA32Float: case MTLPixelFormatRG11B10Float: return 1; default: return 0; }
}
static int fz_load_covers(int i, id<MTLTexture> t, NSUInteger level);
static NSMutableArray *g_lp;    // DM_NVN_LOADPROBE=1 / 2 / 3 (two constant sentinel values / NaN - under float formats normal shader output cannot reach NaN, ruling out numeric coincidence): attachments judged elidable are cleared to the sentinel color instead, and at pass end the remaining sentinel pixels are read back and counted (diagnostic: should be 0)
static void lp_finish(void);
// Same-target continuation: when the game calls SetRenderTargets again on the same set of targets (with no copy / dispatch / clear in between) the pass is not ended - on TBDR this saves one full write-back + load.
// Basis for equivalence: NVN's Barrier between two passes requires the latter to see all writes of the former. Within one Metal render encoder, later draws' reads and writes of the attachment itself (blend, depth / stencil test)
// are ordered with earlier draws in submission order; but 'sampling' the attachment texture is not covered (reading and writing the same attachment within one pass is undefined). So continuation happens only when later draws do not sample these attachments:
// if during continuation a draw needs to sample them, or a clear arrives, the pass is ended before it, returning to the separate semantics. Copies / dispatches end the render encoder anyway
static int g_enc_mode;          // How the current encoder was opened (rp_begin_ex argument: 0 all attachments 1 only for clears 2 depth omitted)
// Only the synthetic raster draw of geo_draw may open a pass without these
// colors. geo_draw ends that pass before returning to ordinary command replay.
static int g_geo_omit_colors, g_enc_geo_nocolor;
static int g_rt_range_active, g_rt_range_pending;
// The render area of the open pass (0: its whole attachments). A draw pass on a very large target that loads what it keeps can stop at the far corner of its scissor: the GPU then does not load and store what lies beyond,
// which keeps its contents. Splatoon Raiders' 6400x6400 ink map, each stamp's pass scissored to 1952x2940 at (0, 3460): about 0.95 â†’ 0.32 ms a pass. Only the draw path opens such a pass, only from targets of 16 M pixels, and only when
// the area is at most half of it; a draw whose scissor leaves the area, and anything else that needs the pass (rp_begin), ends it first. DM_NVN_RP_AREA=0 turns it off
static NSUInteger g_rp_aw, g_rp_ah; static void rt_dims(NSUInteger *tw, NSUInteger *th);
static int rpa_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_NVN_RP_AREA"); v = !(e && !strcmp(e, "0")); if (!v) MLOG("DM_NVN_RP_AREA=0: passes on very large targets cover the whole target"); } return v; }
static int g_rp_whole;   // the pass about to open must cover the whole target (rp_begin)
static int g_rt_merged;         // The current pass has already been continued (at least one same-target SetRenderTargets was merged in)
static uint64_t g_rt_merge_n;   // Cumulative continuation count
static int g_defer_replaying;
static void fz_end(void);   // Wrap-up of the readable-tile pass merge (defined before draw)
static void defer_fallback(void);
// DM_NVN_ENC_LABEL=1: label every render encoder with 'p<index of the render pass within the frame> <width>x<height> <first 8 digits of the first draw's fragment key> d<draw count>'; in Instruments (xctrace Game Performance / Metal System Trace)
// the encoder track and the GPU interval table carry this name directly, and shader_map.tsv maps it back to the shader; off by default (one string allocation per pass)
static int g_el_w, g_el_h, g_el_draws; static char g_el_key[12];
static int enc_label_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_NVN_ENC_LABEL"); v = e && atoi(e); } return v; }
static int g_nbcrt; static void bcrt_flush(void);   // shadows of block-compressed render targets used by the open pass (see bcrt_attach)
// A pass that draws into a texture the host uploads from its storage (tex_storage_dst) writes that storage on the hardware, as a GPU copy does (mirror_readback): what the pass drew is
// read back into the storage when it ends, so that a later upload from the storage, or the CPU reading or copying it, carries the drawing. Tears of the Kingdom draws the normal maps of
// its terrain tiles (260x260 RG8, 9 levels, in a CPU-cached pool) after a flush over their storage made them CPU-written. Its resource defragmentation (main+0x5bc988) later moves each
// tile with memcpy and re-creates the texture at the new address, which uploaded the zeros the storage held: a tile's grass got no length and its ground no light (dark, bare meadows a
// few seconds after loading)
static struct { tex_t t; void *tex; unsigned level, slice, n; int att; } g_rtm[8]; static int g_nrtm;   // n: the layers the pass draws from slice on; att: the colour slot
static int tex_storage_dst(const tex_t *t);
static void mirror_readback(const tex_t *t, id<MTLTexture> d, unsigned level, unsigned layer0, const NVNcopyRegion *r, const char *from, int absorb);
static void rt_mirror_add(const tex_t *t, id<MTLTexture> mt, NSUInteger level, NSUInteger slice, int att) {
    if (!mt || t->d.target == NVN_TEXTURE_TARGET_3D || !tex_storage_dst(t)) return;
    for (int i = 0; i < g_nrtm; i++) if (g_rtm[i].t.serial == t->serial && g_rtm[i].level == level && g_rtm[i].slice == slice) return;
    if (g_nrtm == (int)(sizeof g_rtm / sizeof g_rtm[0])) { RED_ONCE(131, "more than %d storage-backed render targets in one pass: the rest are not read back into their storage", g_nrtm); return; }
    g_rtm[g_nrtm++] = (typeof(g_rtm[0])){*t, (void *)CFBridgingRetain(mt), (unsigned)level, (unsigned)slice, 1, att};
}
// Once the pass's layers are known: a layered or slice-merged pass (renderTargetArrayLength) draws into that many layers of each attachment from its attached slice on, all read back
static void rt_mirror_layers(MTL4RenderPassDescriptor *pd) {
    for (int i = 0; i < g_nrtm; i++) { MTLRenderPassColorAttachmentDescriptor *c = pd.colorAttachments[g_rtm[i].att];
        g_rtm[i].slice = (unsigned)c.slice; g_rtm[i].n = pd.renderTargetArrayLength > 1 ? (unsigned)pd.renderTargetArrayLength : 1; }
}
// After the pass has ended. The readbacks open the copy encoder, which is closed again: callers of rp_end may open an encoder of their own next and rely on none being open
// (with it left open, Tears of the Kingdom stopped presenting on loading its opening at frame 1983)
static void rt_mirror_flush(void) {
    int n = g_nrtm; g_nrtm = 0;
    for (int i = 0; i < n; i++) {
        id<MTLTexture> mt = CFBridgingRelease(g_rtm[i].tex);
        NVNcopyRegion r = {0, 0, 0, (int)MAX(mt.width >> g_rtm[i].level, 1u), (int)MAX(mt.height >> g_rtm[i].level, 1u), (int)g_rtm[i].n};
        mirror_readback(&g_rtm[i].t, mt, g_rtm[i].level, g_rtm[i].slice, &r, "render", 1);
    }
    if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
}
static void rp_end(void) { if (g_enc) { AP_B;
    if (enc_label_on()) { static uint64_t lf = ~0ull, base; uint64_t f = atomic_load(&g_cur_frame); if (f != lf) { lf = f; base = g_stat_pass - 1; }   // Intra-frame serial: the same pass has the same name every frame, so the table can aggregate by name
        [(id)g_enc setLabel:[NSString stringWithFormat:@"p%02llu %dx%d %s d%d", (unsigned long long)(g_stat_pass - base), g_el_w, g_el_h, g_el_key[0] ? g_el_key : "-", g_el_draws]]; }
#ifdef NVN_RT_RANGE_TEST
    int ranged=g_rt_range_active && !g_rt_range_pending;
#endif
    sd_finish(); enc_r_end_pass(g_enc); g_gp_bw_l0 = g_bw_load; g_gp_bw_s0 = g_bw_store;   // the profile's per-pass ledger counts from the previous pass end, profiled or not
    zskip_commit_clears(); g_enc = nil; g_enc_geo_nocolor = 0; g_rt_merged = 0; g_sm_pass = 0; g_rt_range_active = g_rt_range_pending = 0; pg_end_render(); ts_after(); ms_after(); if (g_lp.count) lp_finish(); fz_end();
    if (g_nubd && g_nubd >= ubd_lazy()) ubd_flush();   // the inline updates deferred in this pass, in order (see mtl_sub_update); across passes they wait for the next copy or dispatch (blit_raw)
    if (g_nbcrt) bcrt_flush();   // blocks drawn into shadows go back into their compressed textures
    if (g_nrtm) rt_mirror_flush();
#ifdef NVN_RT_RANGE_TEST
    rr_finish(ranged);
#endif
    AP_E(0);
} }
void pg_sub_begin_ext(uint64_t fence) { pg_sub_begin(fence); }
void pg_commit_ext(id<MTLCommandBuffer> cb) { pg_commit(cb); }
// The format an attachment is actually attached with: the view's format if the view changes the format and can be attached, otherwise the base texture's. Shared by attaching (attach) and pipeline prediction (pso_predict_rest) so both agree --
// prediction used the base format, so for a pass written through an sRGB view (Mario's last pass) the predicted pipeline had the wrong format and the real one was compiled at the first draw (from d2r-mobile e2a0c9a9b)
// A compressed texture as a render target (The Legend of Zelda: Breath of the Wild a 32x32 RGTC1 (BC4) texture, Tears of the Kingdom a 32x32x32 3D one drawn one depth plane per pass, both written
// through an RGBA16UI view: one pixel = one 8-byte block, i.e. BC4 encoded on the GPU). Metal can neither attach a compressed format as a colour attachment nor create compressed <-> uncompressed views
// (the pipeline descriptor validation aborts the process). A BC texture with an uncompressed view whose pixel size equals the block size is drawn through a shadow (bcrt_attach); any other compressed target
// is left out (a pass with no attachment at all skips its draws), with a red line
static int mtl_fmt_block(MTLPixelFormat f) { return (f >= MTLPixelFormatBC1_RGBA && f <= MTLPixelFormatBC7_RGBAUnorm_sRGB) || (f >= MTLPixelFormatPVRTC_RGB_2BPP && f < MTLPixelFormatGBGR422); }
static int nvn_fmt_bc(uint32_t f) { return f >= 0x41 && f <= 0x50; }   // NVN BC1â€“7 / BC6H formats (0x41 RGB_DXT1 â€¦ 0x50 BC6H unsigned)
static int bc_block_bytes(MTLPixelFormat f) {   // bytes per 4x4 block of a BC format, 0 for other formats
    if (f < MTLPixelFormatBC1_RGBA || f > MTLPixelFormatBC7_RGBAUnorm_sRGB) return 0;
    return f <= MTLPixelFormatBC1_RGBA_sRGB || f == MTLPixelFormatBC4_RUnorm || f == MTLPixelFormatBC4_RSnorm ? 8 : 16;
}
// A depth texture drawn through a colour view of the same pixel size (Luigi's Mansion 3: a D16 texture through an R16 view at boot) is drawn the same way: Metal can neither attach a depth format
// as a colour attachment nor make a colour view of a depth texture, and the pipeline descriptor validation aborts the process. Only formats whose Metal pixel is the NVN pixel qualify (D16 and D32F;
// D24S8 is Depth32Float_Stencil8 here, whose bytes are not the game's)
static int shrt_block(MTLPixelFormat f, int *side) {   // bytes per block of a texture drawn through a shadow, and the block's side in pixels; 0 for any other format
    int bb = bc_block_bytes(f); if (bb) { *side = 4; return bb; }
    *side = 1; return f == MTLPixelFormatDepth16Unorm ? 2 : f == MTLPixelFormatDepth32Float ? 4 : 0;
}
static int mtl_fmt_depth(MTLPixelFormat f) { return f == MTLPixelFormatDepth16Unorm || f == MTLPixelFormatDepth32Float || f == MTLPixelFormatDepth32Float_Stencil8 || f == MTLPixelFormatX32_Stencil8 || f == MTLPixelFormatStencil8; }
// The shadow's format: the view's, when the base is BC (or a depth format, above) and the view is an uncompressed colour format with one block's bytes per pixel
static int bcrt_view(id<MTLTexture> base, const view_t *v, int has_v, MTLPixelFormat *pf_out) {
    uint32_t pf = 0; uint8_t swz[4]; int bw = 0, bh = 0, side = 0, bb = base ? shrt_block(base.pixelFormat, &side) : 0;
    if (!bb || !has_v || !v->set_format || !nvn_fmt_to_mtl(v->format, &pf, swz) || !pf || mtl_fmt_block((MTLPixelFormat)pf) || mtl_fmt_depth((MTLPixelFormat)pf)) return 0;
    if (nvn_fmt_bpb(v->format, &bw, &bh) != bb || bw != 1 || bh != 1) return 0;
    *pf_out = (MTLPixelFormat)pf; return 1;
}
// Drawing into a BC texture: the pass draws into an uncompressed shadow of blocks x blocks in the view's format. The level / layer / depth plane's blocks are copied into it through a buffer before the pass
// and back, byte for byte, after it (bcrt_flush, from rp_end). The shadow is an attachment of its own: always loaded and stored, and kept out of the load-skip / store-elision / zero-source bookkeeping,
// which keys by the BC texture's serial. Shadows and staging buffers stay resident and are reused per attachment number, format and size, so a pass per depth plane adds no residency churn.
// From PR #11 (digitalhivehub), where Tears of the Kingdom first drew this way
typedef struct { id<MTLTexture> base, shadow; id<MTLBuffer> buf; NSUInteger level, slice, z, w, h, bw, bh, bpb; } bcrt_t;
static bcrt_t g_bcrt[8];
static __strong id<MTLTexture> g_bcrt_sh[8]; static __strong id<MTLBuffer> g_bcrt_buf[8];
static int bcrt_is(id<MTLTexture> t) { for (int i = 0; i < g_nbcrt; i++) if (t && g_bcrt[i].shadow == t) return 1; return 0; }
static void bcrt_flush(void) {
    int n = g_nbcrt; g_nbcrt = 0;
    for (int i = 0; i < n; i++) { bcrt_t e = g_bcrt[i]; g_bcrt[i] = (bcrt_t){0};
        NSUInteger row = e.bw * e.bpb, img = row * e.bh;
        enc_b_copy_texture_buffer(blit(), e.shadow, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(e.bw, e.bh, 1), e.buf, 0, row, img);
        enc_b_copy_buffer_texture(blit(), e.buf, 0, row, img, MTLSizeMake(e.w, e.h, 1), e.base, e.slice, e.level, MTLOriginMake(0, 0, e.z));
    }
}
// a holds the BC texture with its level and layer / plane already set; returns 1 when a now holds the shadow
static int bcrt_attach(MTLRenderPassAttachmentDescriptor *a, id<MTLTexture> base, const view_t *v, int has_v) {
    MTLPixelFormat spf; int k = g_nbcrt; if (k >= 8 || !bcrt_view(base, v, has_v, &spf)) return 0;
    int is3d = base.textureType == MTLTextureType3D, side = 1;
    NSUInteger bpb = (NSUInteger)shrt_block(base.pixelFormat, &side);
    NSUInteger lv = a.level, sl = is3d ? 0 : a.slice, z = is3d ? a.depthPlane : 0, w = mip_extent(base.width, lv), h = mip_extent(base.height, lv), bw = (w + side - 1) / side, bh = (h + side - 1) / side;
    NSUInteger row = bw * bpb, img = row * bh;
    if (!g_bcrt_sh[k] || g_bcrt_sh[k].pixelFormat != spf || g_bcrt_sh[k].width != bw || g_bcrt_sh[k].height != bh) {
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:spf width:bw height:bh mipmapped:NO];
        d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; d.storageMode = MTLStorageModePrivate;
        id<MTLTexture> sh = [g_dev newTextureWithDescriptor:d]; id<MTLBuffer> buf = [g_dev newBufferWithLength:img options:MTLResourceStorageModePrivate];
        if (!sh || !buf) { RED_ONCE(74, "render target in a compressed or depth texture: cannot create its %lux%lu-block shadow (Metal format %lu)", (unsigned long)bw, (unsigned long)bh, (unsigned long)spf); return 0; }
        if (g_bcrt_sh[k]) { defer_obj_release(g_bcrt_sh[k]); defer_obj_release(g_bcrt_buf[k]); }
        res_add(sh); res_add(buf); g_bcrt_sh[k] = sh; g_bcrt_buf[k] = buf;
    }
    id<MTLTexture> sh = g_bcrt_sh[k]; id<MTLBuffer> buf = g_bcrt_buf[k];
    enc_b_copy_texture_buffer(blit(), base, sl, lv, MTLOriginMake(0, 0, z), MTLSizeMake(w, h, 1), buf, 0, row, img);
    enc_b_copy_buffer_texture(blit(), buf, 0, row, img, MTLSizeMake(bw, bh, 1), sh, 0, 0, MTLOriginMake(0, 0, 0));
    g_bcrt[g_nbcrt++] = (bcrt_t){base, sh, buf, lv, sl, z, w, h, bw, bh, bpb};
    a.texture = sh; a.level = 0; a.slice = 0; a.depthPlane = 0;
    static int once[2]; if (!once[side == 1]++) MLOG("render target in a %s texture: view format %#x drawn into a %lux%lu-block shadow (Metal format %lu) and copied back after the pass (first case pass %llu)",
                                       side == 1 ? "depth" : "compressed", v->format, (unsigned long)bw, (unsigned long)bh, (unsigned long)spf, g_stat_pass + 1);
    return 1;
}
static MTLPixelFormat rt_view_fmt(id<MTLTexture> base, const view_t *v, int has_v) {
    MTLPixelFormat spf;
    if (bcrt_view(base, v, has_v, &spf)) return spf;
    if (base && mtl_fmt_block(base.pixelFormat)) {
        RED_ONCE(78, "render target is a compressed-format texture (Metal format %lu %lux%lu layers %lu, view format change %d â†’ NVN %#x): not attached; draws to this attachment are lost", (unsigned long)base.pixelFormat, (unsigned long)base.width, (unsigned long)base.height,
                 (unsigned long)base.arrayLength, has_v ? v->set_format : 0, has_v ? v->format : 0);
        return MTLPixelFormatInvalid;
    }
    if (!base || !has_v || !v->set_format) return base.pixelFormat;
    uint32_t pf = 0; uint8_t swz[4];
    if (!nvn_fmt_to_mtl(v->format, &pf, swz) || !pf || pf == (uint32_t)base.pixelFormat) return base.pixelFormat;
    if (tex_need_pfv(base.pixelFormat, (MTLPixelFormat)pf) && !(base.usage & MTLTextureUsagePixelFormatView)) return base.pixelFormat;
    return (MTLPixelFormat)pf;
}
// The same texture level / layer attached at more than one colour slot: only the first is attached, later ones are not (their writes are dropped). The G-buffer passes of The Legend of Zelda: Breath of the Wild attach the albedo and another buffer twice each
// (c1 = c2, c3 = c4), and the channel write masks of c2 / c4 are always 0 (cmask 0x00f0f0ff). The Switch's ROP writes only the masked channels and the two slots do not interfere;
// on Apple's TBDR each attachment has its own tile memory and the whole tile is stored back at the end of the pass, so the copy stored later (loaded, never written) overwrites the writes of the earlier one -- the order differs per 32x32 tile,
// the albedo kept only scattered squares, and the whole 3D picture went black. A red line is printed when a dropped slot is used with a non-zero write mask (those writes are really lost)
static int rt_dup_slot(const p_set_rt *r, int i) {
    if (!r->color[i]) return 0;
    uint8_t lv = r->has_cview[i] && r->cview[i].set_levels ? r->cview[i].base_level : 0; uint16_t ly = r->has_cview[i] && r->cview[i].set_layers ? r->cview[i].min_layer : 0;
    for (int j = 0; j < i; j++) {
        if (!r->color[j] || r->color[j]->serial != r->color[i]->serial) continue;   // compare serials: records hold copies of the texture objects (tex_snap), so the pointers differ
        uint8_t lj = r->has_cview[j] && r->cview[j].set_levels ? r->cview[j].base_level : 0; uint16_t yj = r->has_cview[j] && r->cview[j].set_layers ? r->cview[j].min_layer : 0;
        if (lj == lv && yj == ly) return 1;
    }
    return 0;
}
// The layers an attached render target view covers from its attached slice (3D: depth plane) on: the view's layer count (0: the rest), else every layer the texture has from there
static NSUInteger att_layers(MTLRenderPassAttachmentDescriptor *a, const view_t *v, int has_v) {
    id<MTLTexture> t = a.texture; if (!t) return 1;
    NSUInteger all = t.textureType == MTLTextureType3D ? MAX(t.depth >> a.level, 1u) : t.arrayLength * (t.textureType == MTLTextureTypeCube || t.textureType == MTLTextureTypeCubeArray ? 6 : 1);
    NSUInteger from = t.textureType == MTLTextureType3D ? a.depthPlane : a.slice, n = all > from ? all - from : 1;
    if (has_v && v->set_layers && v->nlayers) n = MIN(n, (NSUInteger)v->nlayers);
    return MAX(n, 1u);
}
static void attach(MTLRenderPassAttachmentDescriptor *a, const tex_t *t, const view_t *v, int has_v) {
    id<MTLTexture> base = tex_mtl(t);
    // A texture made without render-target use (packaged or CPU-written, uploaded from its storage) drawn into: give it that use, as a clear of one does (clear_tex). Metal's validation
    // stops at such an attachment; without it Tears of the Kingdom drew its terrain normal maps into textures made for reading only
    if (base && !(base.usage & MTLTextureUsageRenderTarget) && !mtl_fmt_block(base.pixelFormat) && base.sampleCount == 1) base = tex_upgrade(t, base, MTLTextureUsageRenderTarget, base.pixelFormat);
    a.texture = base;
    if (has_v && v->set_levels) a.level = v->base_level;
    if (has_v && v->set_layers) { if (t->d.target == 2) a.depthPlane = v->min_layer; else a.slice = v->min_layer; }
    if (base && mtl_fmt_block(base.pixelFormat)) { if (!bcrt_attach(a, base, v, has_v)) { rt_view_fmt(base, v, has_v); a.texture = nil; } return; }   // compressed formats: a shadow or nothing (see rt_view_fmt)
    if (base && mtl_fmt_depth(base.pixelFormat) && bcrt_attach(a, base, v, has_v)) return;   // a depth texture through a colour view: a shadow (see shrt_block)
    // A view that changes the format is attached as a Metal view of that format (pipeline formats follow the attachment). SMBW's last pass writes the RGBA8 swapchain texture through an RGBA8_SRGB view, so the hardware encodes sRGB on write;
    // attaching the base texture instead stored the linear values as they were and the whole picture came out dark (every value decoded from sRGB once more, on the Mac and the iPhone). The view lives until this submission completes
    if (base && has_v && v->set_format) {
        MTLPixelFormat pf = rt_view_fmt(base, v, has_v);
        if (pf != base.pixelFormat) {
            id<MTLTexture> vw = [base newTextureViewWithPixelFormat:pf];
            if (vw) { a.texture = vw; defer_obj_unref(vw); }
            else RED_ONCE(67, "cannot create the render target view: format %#x (base texture Metal format %lu)", v->format, (unsigned long)base.pixelFormat);
        } else { uint32_t want = 0; uint8_t swz[4];
            if (nvn_fmt_to_mtl(v->format, &want, swz) && want && want != (uint32_t)base.pixelFormat)
                RED_ONCE(68, "render target view format %#x cannot be attached (base texture Metal format %lu was created without the format-view usage): drawing in the base format", v->format, (unsigned long)base.pixelFormat); }
    }
}
// mode 1 (clear_only): a pass opened only to land pending clears, attaching only the attachments that have clears; mode 2: the lookahead proves this pass does not use depth-stencil, so depth is not attached.
// clear_only: a pass opened only to land pending clears, attaching only the attachments that have clears - the other attachments are not touched by this pass, and attaching them costs a pointless load and store on TBDR
// (Mac character select screen: the color-clear pass drags along a 400x225 depth load and store)
static id<MTL4RenderCommandEncoder> rp_begin_ex(int clear_only);
static NSUInteger g_rp_aw; static int g_rp_whole;
static id<MTL4RenderCommandEncoder> rp_begin(void) { if (g_enc && g_rp_aw) rp_end(); g_rp_whole = 1; id<MTL4RenderCommandEncoder> e = rp_begin_ex(0); g_rp_whole = 0; return e; }   // the callers draw anywhere in the target (see g_rp_aw)
static id<MTL4RenderCommandEncoder> rp_begin_ex_(int clear_only);
static id<MTL4RenderCommandEncoder> rp_begin_ex(int clear_only) { AP_B; id<MTL4RenderCommandEncoder> e = rp_begin_ex_(clear_only); AP_E(1); return e; }
static void bare_dims(NSUInteger *w, NSUInteger *h);
static id<MTL4RenderCommandEncoder> rp_begin_ex_(int clear_only) {
    if (g_enc) return g_enc;
    memset(g_zskip_cand_ok, 0, sizeof g_zskip_cand_ok);
    g_enc_mode = clear_only; g_enc_geo_nocolor = g_geo_omit_colors; g_pass_swrite = g_pass_order = g_pass_late_res = 0; g_sm_pass = 0;
    if (!R.has_rt) { RED_ONCE(3, "no render target set before the draw"); return nil; }
    if (g_imfit_npend) img_fit_upgrade();   // no pass is open here: the upgrade's copy may run (img_fit)
    // The version of the attachment-range candidate is taken before capturing any attachment; if it changes during setup, there is no continuation.
    uint64_t reg0 = 0, ser0 = 0, dirty0 = 0;
    if (rt_range_on()) { reg0 = g_reg_gen; ser0 = g_ser_gen; }
    if (rt_range_on()) dirty0=nvn_dirty_now();
    if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    MTL4RenderPassDescriptor *pd = [MTL4RenderPassDescriptor new];
    int any = 0;
    for (int i = 0; i < R.rt.n; i++) {
        if (!R.rt.color[i] || (clear_only == 1 && !R.clear_c[i]) || (g_geo_omit_colors && !R.clear_c[i])) continue;
        if (rt_dup_slot(&R.rt, i)) continue;   // a slot attached twice is not attached (see rt_dup_slot)
        attach(pd.colorAttachments[i], R.rt.color[i], &R.rt.cview[i], R.rt.has_cview[i]);
        if (!pd.colorAttachments[i].texture) continue;
        any = 1;
        pd.colorAttachments[i].loadAction = R.clear_c[i] ? MTLLoadActionClear : MTLLoadActionLoad;
        pd.colorAttachments[i].clearColor = R.cc[i];
        if (R.clear_c[i]) PD("C %llu c%d cleared to %g,%g,%g,%g", g_stat_pass + 1, i, R.cc[i].red, R.cc[i].green, R.cc[i].blue, R.cc[i].alpha);
        if (bcrt_is(pd.colorAttachments[i].texture)) { pd.colorAttachments[i].storeAction = MTLStoreActionStore; continue; }   // a shadow (bcrt_attach): loaded and stored as is, outside the bookkeeping below
        int cov = tbdr_on() && !R.clear_c[i] && (g_fd_dw || g_fd_di) && fz_load_covers(i, pd.colorAttachments[i].texture, pd.colorAttachments[i].level);
        if (!R.clear_c[i] && (g_fd_dw || g_fd_di) && !cov) PD("Q %llu c%d load cannot be skipped why=%d", g_stat_pass + 1, i, g_fd_why);
        if (cov) {
            static int probe = -1; if (probe < 0) { probe = getenv("DM_NVN_LOADPROBE") ? atoi(getenv("DM_NVN_LOADPROBE")) : 0; if (probe) MLOG("DM_NVN_LOADPROBE: attachments whose load can be skipped are cleared to a sentinel color and read back to check"); }
            if (probe) { pd.colorAttachments[i].loadAction = MTLLoadActionClear; int pm = probe == 1 && pf_is_float(pd.colorAttachments[i].texture.pixelFormat) ? 3 : probe;   // Float formats use a NaN sentinel by default: a fixed sentinel value can be hit by real output (it has been hit in practice)
                pd.colorAttachments[i].clearColor = pm == 3 ? MTLClearColorMake(NAN, NAN, NAN, NAN) : probe == 2 ? MTLClearColorMake(0.777, 0.222, 0.555, 0.888) : MTLClearColorMake(0.123, 0.456, 0.789, 0.321);
                if (!g_lë¾½ß¦òµë(š+myÓ%÷B#"Òu÷'G…÷6W¶ÆõÒÂÒÂ"ÒuöG'…öã°¢v†–ÆR†Â"’²V–çC3%÷BÒÒ†²"’ò#²–b†uöG'…÷6W¶ÕÒÃÒ#"’ÒÒ²²VÇ6R"ÒÓ²Ð¢–b†ãÒuöG'…öâ’&WGW&â°¢V–çC3%÷BfBÒuöG'…÷6W¶Ó°¢f÷"‡V–çC3%÷B²ÒÆò²²²Âu÷'G…öâbbu÷'G…÷6W¶µÒÂfC²²²²’–b‚†uö†ö—7Eöâbb†ö—7E÷6¶—VB†u÷6G…ö÷¶u÷'G…÷6W¶µÕÒ’’’&WGW&â²òòF†RF&vWB6†ævW2v–â&Vf÷&RF†Rf—'7BG&s¢7FWGvòöbF†R÷&–v–æÂÆææ–ærv—fW2W2vVÆÀ¢f÷"†–çB¢Ò²¢Â"ç'Bæâbb¢Âƒ²¢²²’²–b‚"ç'Bæ6öÆ÷%¶¥Ò’6öçF–çVS²–çB&²Ò6G…ö'V6¶WB…"ç'Bæ6öÆ÷%¶¥ÒÓç6W&–ÂÂ“²–b†&²Â’6öçF–çVS°¢V–çC3%÷BSÒu÷6G…ö'7E¶&µÒÂSÒu÷6G…ö'7E¶&²²Ó°¢v†–ÆR†SÂS’²V–çC3%÷BÒÒ†S²S’ò#²–b†u÷6G…¶ÕÒç6WÂfB’SÒÒ²²VÇ6RSÒÓ²Ð¢f÷"ƒ²SÂu÷6G…ö'7E¶&²²Òbbu÷6G…¶SÒç6WÓÒfC²S²²’–b†u÷6G…¶SÒæ¶–æBÓÒ4E…õ$TB’&WGW&â²Ð¢&WGW&â°§Ð§7FF–2fö–BgW6U÷G'’‡fö–B’°¢–b†uög¢æ7F—fRÇÂuöW†V5öæW‡BÇÂuö†ö—7Eöâ²"â„ô•5EôÔ‚ÇÂ"ç'Bæâ’&WGW&ã°¢–b‚g¥÷&Vf–ÇFW"‚’’&WGW&ã°¢òòöæS¢g&öÒF†RV&Æ–W"72FòF†RæW‡B6WE&VæFW%F&vWG2ÂöæÇ’G&w2æB7FFR&RÆÆ÷vV@¢6öç7BV–çC…÷B§ÒåTÄÂÂ§#&÷ÒåTÄÂÂ§&WW6U³…Ó²–çBç&WW6RÒ²6öç7B÷6WE÷'B§#"ÒåTÄÃ²÷66—76÷"62Ò"ç63²–çB†5÷62Ò"æ†5÷63°¢g¥ö7W'6÷"7W"Ò²çÒuöW†V5öæW‡GÓ²÷ö†G"ƒ²6öç7Bfö–B¦°¢f÷"†–çBwV&BÒ²wV&BÂ#bb##²wV&B²²’°¢–b‚g¥öæW‡B‚f7W"ÂgÂf‚Âf’’&WGW&ã°¢7v—F6‚†‚æ÷’°¢66Rõô4õ•õC%C¢66RõôE$uõDUƒ¢66Rõô4õ•ô#%C¢66Rõô4õ•õC$#¢66Rõô4ÄT%õDUƒ¢66RõôDÅ53¢66RõôD•5D4ƒ¢66Rõõ$TäDU%ô4ôäC ¢66Rõô4ÄT%ô4ôÄõ#¢66Rõô4ÄT%ôE3¢66RõõUDDUõT#¢66Rõô4õ•ô#$#¢66Rõô4ÄT%ô%Tc¢&WGW&ã°¢66Rõõ44•54õ#¢62Ò¢†6öç7B÷66—76÷"¢–²†5÷62Ò²'&V³°¢66Rõõ4UEõ%C¢°¢–çBVæBÒ"æ6ÆV%öBÇÂ"æ6ÆV%÷3²f÷"†–çB’Ò²’Âƒ²’²²’VæBÃÒ"æ6ÆV%ö5¶•Ó°¢–b‡'E÷6ÖR‚e"ç'BÂ’’°¢–b†uöVæ5öÖöFRÓÒÇÂVæBÇÂ"æ†5÷'BÇÂF&G%ööâ‚’ÇÂç&WW6RãÒ‚’&WGW&ã°¢&WW6U¶ç&WW6R²µÒÒ²6öçF–çVS°¢Ð¢#"Ò²#&÷Ò²6öçF–çVS°¢Ð¢FVfVÇC¢'&V³°¢Ð¢Ð¢–b‚#"ÇÂ'E÷6ÖR‚e"ç'BÂ#"’’&WGW&ã°¢òòGvó¢g&öÒF†R7F'BöbF†RÆFW"72Fò—G2f—'7BG&rÂöæÇ’6ÆV'2†æ÷B6ÆV&–ær6†&VBGF6†ÖVçG2’æB7FFR&RÆÆ÷vV@¢6öç7BV–çC…÷B¦6Å³…Ó²–çBæ6ÂÒ²V–çC…÷B6Ç%³…ÒÒ³Ó²ÕDÄ6ÆV$6öÆ÷"65³…Ó²–çB6EöbÒÂ75öbÒ²F÷V&ÆR6GbÒ²V–çC3%÷B77bÒ°¢–b‚g¥öæW‡B‚f7W"ÂgÂf‚Âf’’&WGW&ã²òò6öçF–çVRg&öÒF†R7W'6÷"öb7FWöæR†Ç&VG’7BF†RÆFW"72w24UEõ%BÂ¶VW–ærF†R7&÷72Ö†æFÆRò&WGW&â7F6²÷6—F–öâ¢f÷"†–çBwV&BÒ²wV&BÂ#²wV&B²²’°¢–b†‚æ÷ÓÒõôE$rÇÂ‚æ÷ÓÒõôE$uô”E‚’'&V³°¢7v—F6‚†‚æ÷’°¢66Rõô4õ•õC%C¢66RõôE$uõDUƒ¢66Rõô4õ•ô#%C¢66Rõô4õ•õC$#¢66Rõô4ÄT%õDUƒ¢66RõôDÅ53¢66RõôD•5D4ƒ¢66Rõõ4UEõ%C¢66RõõUDDUõT#¢66Rõô4õ•ô#$#¢66Rõô4ÄT%ô%Tc¢66Rõõ$TäDU%ô4ôäC¢&WGW&ã°¢66Rõõ44•54õ#¢62Ò¢†6öç7B÷66—76÷"¢–²†5÷62Ò²'&V³°¢66Rõô4ÄT%ô4ôÄõ#¢²6öç7Bö6ÆV%ö6öÆ÷"¦2Ò²–b†æ6ÂãÒ‚ÇÂ2Óæ–æFW‚ÂÇÂ2Óæ–æFW‚ãÒ#"ÓæâÇÂ#"Óæ6öÆ÷%¶2Óæ–æFW…ÒÇÂ2ÓæÖ6²Ò†b’&WGW&ã°¢6öç7BFW…÷B§BÒ#"Óæ6öÆ÷%¶2Óæ–æFW…Ó°¢–b††5÷62bb‡62ç‚ÃÒbb62ç’ÃÒbb62ç‚²62çrãÒ†–çB—BÓæBçv–GF‚bb62ç’²62æ‚ãÒ†–çB—BÓæBæ†V–v‡B’’&WGW&ã°¢6Ç%¶2Óæ–æFW…ÒÒ²65¶2Óæ–æFW…ÒÒÕDÄ6ÆV$6öÆ÷$Ö¶R†2Óç&v&³ÒÂ2Óç&v&³ÒÂ2Óç&v&³%ÒÂ2Óç&v&³5Ò“²6Å¶æ6Â²µÒÒ²'&V³²Ð¢66Rõô4ÄT%ôE3¢²6öç7Bö6ÆV%öG2¦2Ò²–b†æ6ÂãÒ‚ÇÂ#"ÓæFWF‚’&WGW&ã°¢–b†2ÓæFWF…öÖ6²’²6EöbÒ²6GbÒ2ÓæFWFƒ²Ò–b†2Óç7FVæ6–ÅöÖ6²’²75öbÒ²77bÒ‡V–çC3%÷B–2Óç7FVæ6–Ã²Ò6Å¶æ6Â²µÒÒ²'&V³²Ð¢FVfVÇC¢'&V³°¢Ð¢–b‚g¥öæW‡B‚f7W"ÂgÂf‚Âf’’&WGW&ã°¢Ð¢òòF‡&VS¢Væ–öâöbGF6†ÖVçG2‡F†RV&Æ–W"72w2GF6†ÖVçG2¶VWF†V—"÷&–v–æÂ–æF–6W2ÂF†÷6RVæ—VRFòF†RÆFW"72&RVæFVB’Â6öç6—7FVçB6—¦W2Â6ö×F–&ÆRFWF€¢÷6WE÷'BRÒ"ç'C²–çBâÒ"ç'Bæã²–çC…÷B&Ö³…Ó²ÖV×6WB‡&ÖÂÓÂ6—¦Vöb&Ö“°¢f÷"†–çB²Ò²²Â#"Óæâbb²Âƒ²²²²’²–b‚#"Óæ6öÆ÷%¶µÒ’6öçF–çVS°¢–çB¢ÒÓ²f÷"†–çBÒÒ²ÒÂ"ç'Bæã²Ò²²’–b…"ç'Bæ6öÆ÷%¶ÕÒbb7V%öW‚e"ç'BÂÒÂ#"Â²’’²¢ÒÓ²'&V³²Ð¢–b†¢Â’²–b†âãÒ‚’&WGW&ã²¢Òâ²³²Ræ6öÆ÷%¶¥ÒÒ#"Óæ6öÆ÷%¶µÓ²Ræ7f–Wu¶¥ÒÒ#"Óæ7f–Wu¶µÓ²Ræ†5ö7f–Wu¶¥ÒÒ#"Óæ†5ö7f–Wu¶µÓ²Ð¢VÇ6R–b†6Ç%¶µÒ’&WGW&ã²òòF†R7F'BöbF†RÆFW"72vçG2Fò6ÆV"âGF6†ÖVçBF†RV&Æ–W"72—2G&v–ærFó¢gFW"ÖW&v–ærF†—26ÆV"†2æ÷v†W&RFòvð¢&Ö¶µÒÒ†–çC…÷B–£²Ð¢RæâÒã°¢²V–çCcE÷B3²–çBÃÂƒ²'E÷7V"‚eRÂÂg3ÂfÃÂgƒ“²–b‚Ræ6öÆ÷%³Ò’&WGW&ã°¢V–çC3%÷BsÒÔ‚…Ræ6öÆ÷%³ÒÓæBçv–GF‚ãâÃÂR’ÂƒÒÔ‚…Ræ6öÆ÷%³ÒÓæBæ†V–v‡BãâÃÂR“°¢f÷"†–çB¢Ò²¢Âã²¢²²’²V–çCcE÷B5ó²–çBÅòÂ…ó²'E÷7V"‚eRÂ¢Âg5òÂfÅòÂg…ò“²–b‚Ræ6öÆ÷%¶¥ÒÇÂÔ‚…Ræ6öÆ÷%¶¥ÒÓæBçv–GF‚ãâÅòÂR’ÒsÇÂÔ‚…Ræ6öÆ÷%¶¥ÒÓæBæ†V–v‡BãâÅòÂR’Òƒ’&WGW&ã²ÒÐ¢–çBFWF…ög&öÕ÷#"Ò°¢–b…"ç'BæFWF‚bb#"ÓæFWF‚bb7V%öW‚e"ç'BÂ‚Â#"Â‚’’&WGW&ã°¢–b‚"ç'BæFWF‚bb#"ÓæFWF‚’²RæFWF‚Ò#"ÓæFWFƒ²RæGf–WrÒ#"ÓæGf–Ws²Ræ†5öGf–WrÒ#"Óæ†5öGf–Ws²FWF…ög&öÕ÷#"Ò²Ð¢–b‚†6EöbÇÂ75öb’bbFWF…ög&öÕ÷#"’&WGW&ã²òòF†RFWF‚—2Ç&VG’–âW6R–âF†RV&Æ–W"73¢F†RFWF‚6ÆV"BF†R7F'BöbF†RÆFW"72†2æ÷v†W&RFòvð¢òòf÷W#¢G&w2öbF†RV&Æ–W"72×W7Bæ÷B6×ÆRGF6†ÖVçG2'&÷Vv‡B–â'’F†RÆFW"73²v†VâF†RFWF‚6öÖW2g&öÒF†RÆFW"72ÂFWF‚×7FVæ6–ÂFW7F–ær×W7Bæ÷B&RVæ&ÆV@¢7FF–2V–çCcE÷BFW…³eÕ´åDU…ÒÂ–Öu³eÕ³3%Ó²ÖVÖ7’‡FW‚Â2çFW‚Â6—¦VöbFW‚“²ÖVÖ7’†–ÖrÂ2æ–ÖrÂ6—¦Vöb–Ör“°¢6öç7B&öu÷B§rÒ2ç&ös²G5÷7BG2Ò2æG3°¢7W"Ò†g¥ö7W'6÷"—²çÒuöW†V5öæW‡GÓ°¢f÷"ƒ³²’²òòF†Rf—'7BÆö÷—FW&F–öâ6†V6·2F†RG&rF†B÷Vç2F†R&VæFW"72†7W'&VçB7FFR’ÂF†Vâ6†V6·2F†RG&w2–âF†R&ævRöæR'’öæP¢–b†FWF…ög&öÕ÷#"bb†G2æGBÇÂG2æGrÇÂG2ç7B’’&WGW&ã°¢f÷"†–çB¢Ò"ç'Bæã²¢Âã²¢²²’–b‡&öu÷6×ÆW5÷6W"‡rÂFW‚Â–ÖrÂRæ6öÆ÷%¶¥ÒÓç6W&–Â’’&WGW&ã°¢–b†FWF…ög&öÕ÷#"bb&öu÷6×ÆW5÷6W"‡rÂFW‚Â–ÖrÂRæFWF‚Óç6W&–Â’’&WGW&ã°¢f÷"ƒ³²’°¢6öç7BV–çC…÷B§²÷ö†G"‡²6öç7Bfö–B¦°¢–b‚g¥öæW‡B‚f7W"ÂgÂf‡Âf’’&WGW&ã°¢–b‡ÓÒ#&÷’v÷FòÆææVC°¢–b†‡æ÷ÓÒõô$”äEõ$ôu$Ò’rÒ‚†6öç7Bö&–æE÷&öw&Ò¢–’Óç&ös°¢VÇ6R–b†‡æ÷ÓÒõô$”äEõ5DDR’²6öç7Bö&–æE÷7FFR¦"Ò²–b†"Óæ¶–æBÓÒR’ÖVÖ7’‚fG2Â"Óæ'—FW2Â6—¦VöbG2“²Ð¢VÇ6R–b†‡æ÷ÓÒõô$”äEõDU‚ÇÂ‡æ÷ÓÒõô$”äEõDUƒÇÂ‡æ÷ÓÒõô$”äEô”Ôr’²ö&–æEö†æFÆW2C²6öç7Bö&–æEö†æFÆW2¦"Ò&–æE÷&V2†‡æ÷ÂÂgC“²&…ög&W6‚†"“²–çBÆ–ÒÒ‡æ÷ÓÒõô$”äEô”Ôrò3"¢åDUƒ°¢–b†"Óç7FvRãÒbb"Óç7FvRÂbbb"Óæf—'7BãÒbb"Óæf—'7B²"ÓæâÃÒÆ–Ò’f÷"†–çB²Ò²²Â"Óæã²²²²’²–b†‡æ÷ÓÒõô$”äEô”Ôr’–Öu¶"Óç7FvUÕ¶"Óæf—'7B²µÒÒ"Óæ…¶µÓ²VÇ6RFW…¶"Óç7FvUÕ¶"Óæf—'7B²µÒÒ"Óæ…¶µÓ²ÒÐ¢VÇ6R–b†‡æ÷ÓÒõôE$rÇÂ‡æ÷ÓÒõôE$uô”E‚’'&V³°¢Ð¢Ð§ÆææVC ¢²òòf—fS¢&R×67&VVâF†Rf—'7BG&röbF†RÆFW"72‡7FF–26öæF—F–öç2’ÂFòfö–BGF6†–ærF†RVæ–öâöbGF6†ÖVçG2f÷"æ÷F†–æs¢G&6²F†R&öw&ÒæBFW‡GW&R&–æF–æw2WFòF†BG&p¢6öç7B&öu÷B§s"Ò2ç&ös²7FF–2V–çCcE÷BC%³eÕ´åDU…ÒÂ“%³eÕ³3%Ó²ÖVÖ7’‡C"Â2çFW‚Â6—¦VöbC"“²ÖVÖ7’†“"Â2æ–ÖrÂ6—¦Vöb“"“°¢7W"Ò†g¥ö7W'6÷"—²çÒuöW†V5öæW‡GÓ°¢f÷"ƒ³²’²6öç7BV–çC…÷B§²÷ö†G"‡²6öç7Bfö–B¦²–b‚g¥öæW‡B‚f7W"ÂgÂf‡Âf’’&WGW&ã²–b‡ÓÒ’'&V³°¢–b†‡æ÷ÓÒõô$”äEõ$ôu$Ò’s"Ò‚†6öç7Bö&–æE÷&öw&Ò¢–’Óç&ös°¢VÇ6R–b†‡æ÷ÓÒõô$”äEõDU‚ÇÂ‡æ÷ÓÒõô$”äEõDUƒÇÂ‡æ÷ÓÒõô$”äEô”Ôr’²ö&–æEö†æFÆW2C²6öç7Bö&–æEö†æFÆW2¦"Ò&–æE÷&V2†‡æ÷ÂÂgC“²&…ög&W6‚†"“²–çBÆ–ÒÒ‡æ÷ÓÒõô$”äEô”Ôrò3"¢åDUƒ°¢–b†"Óç7FvRãÒbb"Óç7FvRÂbbb"Óæf—'7BãÒbb"Óæf—'7B²"ÓæâÃÒÆ–Ò’f÷"†–çB²Ò²²Â"Óæã²²²²’²–b†‡æ÷ÓÒõô$”äEô”Ôr’“%¶"Óç7FvUÕ¶"Óæf—'7B²µÒÒ"Óæ…¶µÓ²VÇ6RC%¶"Óç7FvUÕ¶"Óæf—'7B²µÒÒ"Óæ…¶µÓ²ÒÐ¢Ð¢çe&ör§#"Òs"ò&öuövWB‡s"’¢æ–Ã²çe6†FW"§g3"Ò#"ò#"Óç6…³Ò¢æ–ÂÂ¦g3"Ò#"ò#"Óç6…³Ò¢æ–Ã°¢–b‚g3"ÇÂg3"’&WGW&ã°¢–çBF÷V6†W2Ò°¢f÷"†–çB7BÒ²7BÂS²7B²²’²õ÷Vç6fU÷Vç&WF–æVBçe6†FW"§6‚Ò#"Óç6…·7EÓ²–b‚6‚ÇÂ6‚Óç7FvRãÒb’6öçF–çVS°¢f÷"†–çB’Ò²’Â6‚Óç&bææ#²’²²’²6öç7Bçfå÷&&–æB¦&"Òg6‚Óç&bæ%¶•Ó²V–çCcE÷Bƒ°¢–b†&"Óæ¶–æBÓÒådåõ$%ô”Ôr’‚Ò&"Óç&VrÂ3"ò“%·6‚Óç7FvUÕ¶&"Óç&VuÒ¢²VÇ6R–b†&"Óæ¶–æBÓÒådåõ$%õDU‚ÇÂ&"Óæ¶–æBÓÒådåõ$%õD%Tb’‚ÒFW…÷6Æ÷Eö‚‡C%·6‚Óç7FvUÒÂ“%·6‚Óç7FvUÒÂ&"Óç&Vr“²VÇ6R6öçF–çVS°¢V–çCcE÷B6W"Ò‚ò…÷6W"†‚’¢²–b‚6W"’6öçF–çVS°¢–çB¢ÒÓ²f÷"†–çBÒÒ²ÒÂ"ç'Bæã²Ò²²’–b…"ç'Bæ6öÆ÷%¶ÕÒbb"ç'Bæ6öÆ÷%¶ÕÒÓç6W&–ÂÓÒ6W"’²¢ÒÓ²'&V³²Ð¢–b†¢Â’6öçF–çVS°¢–b‡6‚Òg3"ÇÂ&"Óæ¶–æBÒådåõ$%õDU‚ÇÂ†g3"Óç&bæf5öÖ—†VE¶&"Óç&VrãâeÒãâ†&"Óç&Vrbc2’b’’&WGW&ã°¢–çBç’Ò°¢f÷"†–çB2Ò²2Âg3"Óç&bææf3²2²²’–b†g3"Óç&bæf5÷&Vu¶5ÒÓÒ&"Óç&Vr’²–çBÂÒg3"Óç&bæf5öÆö6å¶5Ó°¢–b†ÂãÒ3"ÇÂ†g3"Óç&bæf5÷&ö¢ãâ2bò‡g3"Óç&bç67%öÆö6âãâÂb’¢g3"Óç&bçWeöGG%¶ÅÒÂbbg3"Óç&bçWeöfeöGG%¶ÅÒÂ’’&WGW&ã²ç’Ò²Ð¢–b‚ç’’&WGW&ã°¢F÷V6†W2Ò²ÒÐ¢–b‚F÷V6†W2’&WGW&ã²òòF†RÆFW"72FöW2æ÷B&VBF†RV&Æ–W"72w2GF6†ÖVçG3¢ÖW&v–ær†2æò&VæVf—@¢–çB–FVçBÒ²f÷"†–çB2Ò²2Âƒ²2²²’–b†g3"Óç&bæ6öÆ÷%öÖ6²ãâ2b’²–b‡&Ö¶5ÒÂ’&WGW&ã²–b‡&Ö¶5ÒÒ2’–FVçBÒ²Ð¢–b‚–FVçBbbg3"Óç&bææfò’&WGW&ã°¢Ð¢ÖV×6WB‚fuög¢ÂÂ6—¦Vöbuög¢“°¢uög¢æ7F—fRÒ²uög¢ç†6RÒ²uög¢æãÒ"ç'Bæã²uög¢æç&WW6RÒç&WW6S²uög¢ç'CÒ"ç'C²uög¢ç'C"Ò§##²uög¢ç'C%ö÷Ò#&÷²ÖVÖ7’†uög¢ç&WW6RÂ&WW6RÂ6—¦Vöb&WW6R“²ÖVÖ7’†uög¢ç&ÖÂ&ÖÂ‚“°¢f÷"†–çB¢Ò²¢Âƒ²¢²²’uög¢æÆÖ¶¥ÒÒ¢Â"ç'Bæâò‡V–çC…÷B–¢¢†fc°¢g¥÷&ö&Uö6÷c"‡#"ÂÂ&ÖÂ6Ç"Âuög¢æãÂâ“°¢uög¢ç6¶—¶uög¢æç6¶—²µÒÒ#&÷²f÷"†–çB’Ò²’Âæ6Ã²’²²’uög¢ç6¶—¶uög¢æç6¶—²µÒÒ6Å¶•Ó°¢f÷"†–çB’Ò²’Âuög¢æç6¶—²’²²’uö†ö—7E÷6¶—¶uö†ö—7Eöâ²µÒÒuög¢ç6¶—¶•Ó°¢"ç'BÒS°¢ÖVÖ7’†uög¢æ6Ç#"Â6Ç"Â‚“²ÖVÖ7’†uög¢æ63"Â62Â6—¦Vöb62“²uög¢æ6C&bÒ‡V–çC…÷B–6Eöc²uög¢æ73&bÒ‡V–çC…÷B–75öc²uög¢æ6C"Ò6Gc²uög¢æ73"Ò77c°¢f÷"†–çB²Ò²²Âƒ²²²²’–b†6Ç%¶µÒ’²"æ6ÆV%ö5·&Ö¶µÕÒÒ²"æ65·&Ö¶µÕÒÒ65¶µÓ²Ð¢–b†6Eöb’²"æ6ÆV%öBÒ²"æ6BÒ6Gc²Ò–b†75öb’²"æ6ÆV%÷2Ò²"æ72Ò77c²Ð¢–b‚uög¥öâ²²’ÔÄôr‚'&VF&ÆR×F–ÆR72ÖW&vS¢f—'7B66R72VÆÇR‚VB‡—6–6ÂGF6†ÖVçG2’"Âu÷7FE÷72²Ââ“°¢B‚$bVÆÇRÖW&vRv—F‚F†RæW‡B72‡&VF&ÆRF–ÆR“¢‡—6–6ÂGF6†ÖVçG2VBÂ&Wf–÷W272VB"Âu÷7FE÷72²ÂâÂ"ç'BæâÓÒâòuög¢æã¢uög¢æã“°§Ð§7FF–2–CÄÕDÄgVæ7F–öãâ&eöfâ„çe6†FW"¦g2Â6öç7B–çB¦g‚ÂV–çC…÷Bf&Ö6²Â6öç7BV–çC…÷B¦ÆÖ’²òòF†R&VF&ÆRF–ÆRfW'6–öâ—27V6–Æ—¦VB'’6öç7FçG2†66†VB¢÷5÷Væf—%öÆö6µöÆö6²‚fuög¥öfåöÆö6²“°¢–b‚uö&eöfåö66†R’²uö&eöfåö66†RÒ´å4×WF&ÆTF–7F–öæ'’F–7F–öæ'•Ó²uö&eöÆ–%ö66†RÒ´å4×WF&ÆTF–7F–öæ'’F–7F–öæ'•Ó²Ð¢6—¦U÷B¶ÂÒ7G&ÆVâ†g2Óæ¶W’“²å57G&–ær¦&6RÒ´å57G&–ær7G&–æuv—F„f÷&ÖC¤"Râ§2"Â†–çB’†¶Ââ"ò¶ÂÒ"¢¶Â’Âg2Óæ¶W•Ó°¢–CÄÕDÄÆ–'&'“âÆ–"Òuö&eöÆ–%ö66†U¶&6UÓ°¢–b‚Æ–"’°¢å4W'&÷"¦RÒæ–Ã²å57G&–ær§F‚Ò´å57G&–ær7G&–æuv—F„f÷&ÖC¤"TòTö&bæÖWFÆÆ–""Âuö×6ÅöF—"Â&6UÓ°¢Æ–"ÒÆ–%ög&öÕöf–ÆR‡F‚ÂfR“°¢–b‚Æ–"’²ÔÄôr‚%µ$TEÒf–ÆVBFòÆöBF†R&VF&ÆR×F–ÆRfW'6–öâW3¢W2"ÂF‚åUDc…7G&–ærÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“²uö&eöÆ–%ö66†U¶&6UÒÒ†–CÄÕDÄÆ–'&'“â”å4çVÆÂæçVÆÃ²Òòòæ÷BÆöFVBv–âf÷"WfW'’G&r†f%÷F–ÆU÷Æâ6·2BV6‚öæR¢VÇ6Ruö&eöÆ–%ö66†U¶&6UÒÒÆ–#°¢Ð¢–b‚Æ–"ÇÂ†–B–Æ–"ÓÒå4çVÆÂæçVÆÂ’²÷5÷Væf—%öÆö6µ÷VæÆö6²‚fuög¥öfåöÆö6²“²&WGW&âæ–Ã²Ð¢å4×WF&ÆU7G&–ær¦²Ò´å4×WF&ÆU7G&–ær7G&–æuv—F„f÷&ÖC¤"S'‚"Âf&Ö6µÓ°¢f÷"†–çB’Ò²’Âg2Óç&bææf3²’²²’¶²VæDf÷&ÖC¤"ÂVB"Âg…¶•ÕÓ°¢f÷"†–çB¢Ò²¢Âƒ²¢²²’¶²VæDf÷&ÖC¤"òS'‚"ÂÆÖ¶¥ÕÓ°¢å57G&–ær¦66†T¶W’Ò´å57G&–ær7G&–æuv—F„f÷&ÖC¤"TÂT"Â&6RÂµÓ°¢–CÄÕDÄgVæ7F–öãâbÒuö&eöfåö66†U¶66†T¶W•Ó°¢–b†b’²÷5÷Væf—%öÆö6µ÷VæÆö6²‚fuög¥öfåöÆö6²“²&WGW&â†–B–bÓÒå4çVÆÂæçVÆÂòæ–Â¢c²Ð¢ÕDÄgVæ7F–öä6öç7FçEfÇVW2¦7bÒ´ÕDÄgVæ7F–öä6öç7FçEfÇVW2æWuÓ°¢f÷"†–çB’Ò²’Âg2Óç&bææf3²’²²’²–çBbÒg…¶•Ó²¶7b6WD6öç7FçEfÇVS¢gbG—S¤ÕDÄFFG—T–çBD–æFWƒ¢„å5T–çFVvW"–•Ó²Ð¢f÷"†–çBÒ²Âƒ²²²’²&ööÂ"Òf&Ö6²ãâb²¶7b6WD6öç7FçEfÇVS¢f"G—S¤ÕDÄFFG—T&ööÂD–æFWƒ¢„å5T–çFVvW"’ƒ3"²•Ó²Ð¢f÷"†–çB¢Ò²¢Âg2Óç&bææfó²¢²²’f÷"†–çBÒ²Âƒ²²²’²&ööÂ"ÒÆÖ·ÒÓÒg2Óç&bæfõö6öÆ÷%¶¥Ó²¶7b6WD6öç7FçEfÇVS¢f"G—S¤ÕDÄFFG—T&ööÂD–æFWƒ¢„å5T–çFVvW"’ƒcB²‚¢¢²•Ó²Ð¢å4W'&÷"¦RÒæ–Ã²bÒfãEöæ÷FR…¶Æ–"æWtgVæ7F–öåv—F„æÖS¤&Ö–ã"6öç7FçEfÇVW3¦7bW'&÷#¢fUÒÂÆ–"Â&Ö–ã"Â7b“°¢–b‚b’²ÔÄôr‚%µ$TEÒ&VF&ÆR×F–ÆRfW'6–öâ7V6–Æ—¦F–öâf–ÆVBW2‚W2“¢W2"Âg2Óæ¶W’Â²åUDc…7G&–ærÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“²uö&eöfåö66†U¶66†T¶W•ÒÒ†–CÄÕDÄgVæ7F–öãâ”å4çVÆÂæçVÆÃ²÷5÷Væf—%öÆö6µ÷VæÆö6²‚fuög¥öfåöÆö6²“²&WGW&âæ–Ã²Ð¢uö&eöfåö66†U¶66†T¶W•ÒÒc²÷5÷Væf—%öÆö6µ÷VæÆö6²‚fuög¥öfåöÆö6²“²&WGW&âc°§Ð§7FF–2–çBV&õöfÆöB†–çB6Æ÷BÂV–çCe÷B&VbÂfÆöB¦÷WB’²òò6ö×öæVçB6×öbfV7F÷"fV2öbF†RVæ–f÷&Ò6Æ÷B‡&VBöâF†R5R6–FS²æ÷B&VB–b—BfÆÇ2v—F†–â'VffW"w&—FRVæF–ærW†V7WF–öâ–âF†—27V&Ö—B¢–çB6ÂÒ&Vbãâ‚ÂfV2Ò&Vbãâ"bc2Â6×Ò&Vbb3°¢–b‡6Æ÷BÂÇÂ6Æ÷BãÒbÇÂ6ÂÂÇÂ6ÂãÒb’&WGW&â°¢6öç7Bö&–æEö'Vb§RÒe2çV&õ·6Æ÷EÕ·6ÅÓ²V–çCcE÷BöfbÒ‡V–çCcE÷B—fV2¢b²‡V–çCcE÷B–6×¢C°¢–b‚RÓæFG"ÇÂöfb²BâRÓç6—¦RÇÂVæEö†—G2‡RÓæFG"²öfbÂB’’&WGW&â°¢fö–B¦Ö#²V–çCcE÷BÖó²6öç7BV–çC…÷B¦&6S°¢–b‚×FÅöFG"‡RÓæFG"²öfbÂBÂfÖ"ÂfÖò’ÇÂ†&6RÒ²…õö'&–FvR–CÄÕDÄ'VffW#â–Ö"6öçFVçG5Ò’’&WGW&â°¢ÖVÖ7’†÷WBÂ&6R²ÖòÂB“²&WGW&â°§Ð§7FF–2–çB§6¶—öf7Eöf÷"†6öç7B6†"¦¶W’Â§6¶—öf7B¦÷WB’°¢–b‚¶W’’&WGW&â°¢f÷"†–çB’Ò²’Âu÷§6¶—öæc²’²²’–b‚7G&æ6×†¶W’Âu÷§6¶—öf7G5¶•Òæ¶W’Â7G&ÆVâ†u÷§6¶—öf7G5¶•Òæ¶W’’’’²¦÷WBÒu÷§6¶—öf7G5¶•Ó²&WGW&â²Òòò'VçF–ÖR¶W—26''’F†Rö"7Vff—€¢&WGW&â°§Ð§7FF–2–çB§6¶—÷f–Wu÷¦W&ò†–çBF–B’°¢–b‚F–BÇÂF–BãÒuöçG&VrÇÂu÷G&Vu·F–EÒæö²’&WGW&â°¢G&Vu÷B§"Òfu÷G&Vu·F–EÓ²6öç7BFW…÷B§BÒg"ÓçC²–çBÇcÒ"Óæ†5÷bbb"Óçbç6WEöÆWfVÇ2ò"Óçbæ&6UöÆWfVÂ¢°¢–çBæÂÒ"Óæ†5÷bbb"Óçbç6WEöÆWfVÇ2bb"ÓçbææÆWfVÇ2âò"ÓçbææÆWfVÇ2¢°¢–çB6ÃÒ"Óæ†5÷bbb"Óçbç6WEöÆ–W'2ò"ÓçbæÖ–åöÆ–W"¢°¢–çBç2Ò"Óæ†5÷bbb"Óçbç6WEöÆ–W'2bb"ÓçbææÆ–W'2âò"ÓçbææÆ–W'2¢°¢f÷"†–çBÂÒ²ÂÂæÃ²Â²²’f÷"†–çB2Ò²2Âç3²2²²’–b‚§6¶—ö†2‡BÓç6W&–ÂÂÇc²ÂÂ6Ã²2’’&WGW&â°¢&WGW&â°§Ð¢òòv†–ÆRâö66ÇW6–öâVW'’—26÷VçF–ærÂ6¶—VBG&r6†÷VÆB†fR6öçG&–'WFVB76VB6×ÆW3¢—BöæÇ’FöW2æ÷BÖGFW"–b'F†W&R—2æò&W÷'B&Vf÷&RF†RæW‡B&W6WB"à¢òòÆöö¶–ærf÷'v&B–â&WÆ’÷&FW"g&öÒgFW"F†—2G&s¢6×ÆR6÷VçB&W6WBVæ6÷VçFW&VBf—'7BÓâFöW2æ÷BÖGFW"‡&WGW&â“²6×ÆR6÷VçB&W÷'BVæ6÷VçFW&VBf—'7BÂ÷"F†RVæB6ææ÷B&R6VVâ‡7V&Ö—BVæG2’ÓâÖGFW'2‡&WGW&â§7FF–2–çB§6¶—÷f—5öFVB‡fö–B’°¢–b‚u÷f—5ö6÷VçF–ær’&WGW&â°¢6öç7BV–çC…÷B§ÒuöW†V5öæW‡C²–çB6’Ò²–b‚’&WGW&â°¢f÷"†–çBwV&BÒ²wV&BÂ#²wV&B²²’²÷ö†G"ƒ²ÖVÖ7’‚f‚ÂÂ6—¦Vöb‚“²6öç7Bfö–B¦Ò²ƒ°¢–b†‚æ÷ÓÒõô¥TÕ’²ÖVÖ7’‚gÂÂ‚“²6öçF–çVS²Ð¢–b†‚æ÷ÓÒõôTäB’²ÒFVfW%ö6öçB‚f6’“²–b‚’&WGW&â²6öçF–çVS²Ð¢–b†‚æ÷ÓÒõô4ÄÂ’&WGW&â°¢–b†‚æ÷ÓÒõõ$U4UEô4õTåDU"ÇÂ‚æ÷ÓÒõõ$Uõ%Eô4õTåDU"’²V–çCcE÷BC²ÖVÖ7’‚gBÂÂ‚“²–b‡BÓÒådåô4õTåDU%õE•Uõ4ÕÄU5õ54TB’&WGW&â‚æ÷ÓÒõõ$U4UEô4õTåDU#²Ð¢³Ò‚ç6—¦S²Ð¢&WGW&â°§Ð§7FF–2–çB§6¶—ö6åöG&r„çe&ör§"’°¢òò6¶—6öæF—F–öç2†ÆÂ×W7B†öÆB“¢F†Rg&vÖVçB¶W’—2–âF†Rf7BF&ÆRÂF†Rf–Wr&÷VæBFòWfW'’FW‡GW&R6Æ÷BFV6Æ&VB'’—G2&VfÆV7F–öâ6÷fW'2öæÇ’7V'&W6÷W&6W2¶æ÷vâFò&RÆÂ¦W&òÂF†R6öVff–6–VçB—2f–æ—FRÀ¢òòæòFWF‚w&—FRæBæò7FVæ6–ÂÂæ÷B6÷VçF–ærâö66ÇW6–öâVW'’ÂæBWfW'’6öÆ÷"GF6†ÖVçBFò&Rw&—GFVâ—2Ç&VG’†6æF–FFR6ÆV&VBVçF—&VÇ’Fò–âF†—2&VæFW"72Â÷"¶æ÷vâÆÂ¦W&ò’ÒÒw&—F–ær÷fW"à¢òò&V6öç2f÷"æ÷B†öÆF–ær‡F†RâÆ–æRöb54ETÕ“¢FWF‚ò7FVæ6–Â"ö66ÇW6–öâVW'’6÷VçF–ærv—F‚&W÷'B&Vf÷&RF†R&W6WB26öVff–6–VçBVç&VF&ÆR÷"æöâÖf–æ—FRB–ÖvR6Æ÷BR6×ÆVBf–Wræ÷B¶æ÷vâÆÂ¦W&òbæòF&vWBrF&vWB—2æ÷B ¢–b…2ç&7EöF—66&BÇÂ§6¶—ööâ‚’ÇÂ"’&WGW&â°¢çe6†FW"¦g2Òæ–Ã²f÷"†–çB7BÒ²7BÂS²7B²²’–b‡"Óç6…·7EÒbb"Óç6…·7EÒÓç7FvRÓÒ’g2Ò"Óç6…·7EÓ°¢§6¶—öf7Bc²fÆöB6öVfc²–çBv‡’Ò°¢–b‚g2ÇÂ§6¶—öf7Eöf÷"†g2Óæ¶W’Âfb’’&WGW&â°¢–b…2æG2æGrÇÂ2æG2ç7B’v‡’Ò°¢VÇ6R–b‚§6¶—÷f—5öFVB‚’’v‡’Ò#°¢VÇ6R–b†bçV&õ÷6Æ÷BãÒbb‚V&õöfÆöBƒÂ‡V–çCe÷B’‚†bçV&õ÷6Æ÷BÃÂ‚’Â†bçfV2ÃÂ"’Âbæ6ö×’Âf6öVfb’ÇÂ—6f–æ—FR†6öVfb’’’v‡’Ò3²òòV&õ÷6Æ÷BÂ¢F†—26†FW"†2æò6öVff–6–Vç@¢f÷"†–çB’Ò²v‡’bb’Âg2Óç&bææ#²’²²’²çfå÷&&–æB¦"Òfg2Óç&bæ%¶•Ó°¢–b†"Óæ¶–æBÓÒådåõ$%ô”Ôr’²v‡’ÒC²'&V³²Ð¢–b†"Óæ¶–æBÒådåõ$%õDU‚bb"Óæ¶–æBÒådåõ$%õD%Tb’6öçF–çVS°¢–çBF–BÒ†–çB’‡FW…÷6Æ÷Eö‚…2çFW…³ÒÂ2æ–Öu³ÒÂ"Óç&Vr’b†ffffb“²–b‚§6¶—÷f–Wu÷¦W&ò‡F–B’’v‡’ÒS²Ð¢–b‚v‡’bb‚"æ†5÷'BÇÂ"ç'Bæâ’’v‡’Òc°¢f÷"†–çB’Ò²v‡’bb’Â"ç'Bæâbb’Âƒ²’²²’²–b‚"ç'Bæ6öÆ÷%¶•Ò’6öçF–çVS°¢–çBVæF–ærÒuöVæ2òu÷§6¶—ö6æEööµ¶•Ò¢§6¶—÷¦W&õö6ÆV"†’“²òò&VæFW"72Ç&VG’÷Vã¢Æöö²BF†R6æF–FFW2æ÷FVBv†VâF†R&VæFW"72v2÷VæVB†æ÷B–WBG&vâFò“²æ÷B÷Vã¢Æöö²BF†RVæF–ær6ÆV'0¢–b‚VæF–ærbb†uöVæ2ÇÂ"æ6ÆV%ö5¶•ÒÇÂ§6¶—÷F&vWE÷¦W&ò…"ç'Bæ6öÆ÷%¶•ÒÂe"ç'Bæ7f–Wu¶•ÒÂ"ç'Bæ†5ö7f–Wu¶•Ò’’’v‡’Òs²Ð¢–b‡v‡’’B‚$âVÆÇR¦W&ò×6÷W&6RG&ræ÷B6¶—VB&V6öãÒVB6WBVB"Âu÷7FE÷72²Âv‡’Âu÷§6¶—öâ“°¢&WGW&âv‡“°§Ð§7FF–2–çBWe÷&ööb„çe6†FW"§g2Â–çBÆö6âÂ6öç7BöG&r¦GrÂ6öç7BöG&uö–G‚¦F’’²òòF†R–çFW'öÆFVB÷WGWBÆö6âBWfW'’—†VÂ6VçFW"—2W†7FÇ’F†—2—†VÂw2FW†VÂ6VçFW#¢RÒ‡‚²’ò"ÂbÒƒÒ’’ò"ÂrÒÂf–Ww÷'BæB66—76÷"6÷fW"F†Rv†öÆR&Æö6°¢–b†Æö6âÂÇÂÆö6âãÒ3"ÇÂg2Óç&bç÷5öGG"Â’&WGW&â°¢–çBF—&V7BÒg2Óç&bçWeöGG%¶Æö6åÒãÒÂfbÒg2Óç&bçWeöfeöGG%¶Æö6åÒãÒ°¢–b‚F—&V7Bbbfb’&WGW&â°¢fÆöB65³EÒÒ³ÂÂÂÓ²òò72×F‡&÷Vv‚2Ö—2Ò66ÆRÂöfg6WB ¢–b‚F—&V7B’f÷"†–çBÒ²ÂC²²²’–b‚V&õöfÆöBƒÂg2Óç&bçWeöfe¶Æö6åÕ·ÒÂg65·Ò’’&WGW&â°¢å5T–çFVvW"rÒu÷72çGrÂ‚Òu÷72çFƒ°¢–b…2æ†5÷gbb…2çg³Õ³ÒÓÒbb2çg³Õ³ÒÓÒbb2çg³Õ³%ÒÓÒ†fÆöB•rbb2çg³Õ³5ÒÓÒ†fÆöB”‚’’&WGW&â°¢–b…"æ†5÷62bb…"ç62ç‚ÃÒbb"ç62ç’ÃÒbb"ç62ç‚²"ç62çrãÒ†–çB•rbb"ç62ç’²"ç62æ‚ãÒ†–çB”‚’’&WGW&â°¢fÆöB³cEÕ³EÒÂ³cEÕ³EÓ²–çBâÒGròGrÓæ6÷VçB¢F’Óæ6÷VçC°¢–b†GG%÷&VB‡g2Óç&bç÷5öGG"ÂGrÂF’Â’ÇÂGG%÷&VB†F—&V7Bòg2Óç&bçWeöGG%¶Æö6åÒ¢g2Óç&bçWeöfeöGG%¶Æö6åÒÂGrÂF’Â’’&WGW&â°¢f÷"†–çB²Ò²²Âã²²²²’²fÆöBrÒg2Óç&bç÷5÷sòãb¢¶µÕ³5Ó°¢fÆöBRÒF—&V7Bò¶µÕ³Ò¢fÖb…¶µÕ³ÒÂ65³ÒÂ65³Ò’ÂbÒF—&V7Bò¶µÕ³Ò¢fÖb…¶µÕ³ÒÂ65³%ÒÂ65³5Ò“²òòF†R6ÖRfÖ2F†R6†FW ¢–b‡rÒãbÇÂRÒ…¶µÕ³Ò²ãb’¢ãVbÇÂbÒƒãbÒ¶µÕ³Ò’¢ãVb’²B‚$bVÆÇRWb&ööc¢fW'FW‚VB÷6—F–öâVrÂVrWbVrÂVr‡66ÆRòöfg6WBVrÂVrÂVrÂVr’"Âu÷7FE÷72Â²Â¶µÕ³ÒÂ¶µÕ³ÒÂRÂbÂ65³ÒÂ65³ÒÂ65³%ÒÂ65³5Ò“²&WGW&â²ÒÐ¢&WGW&â°§Ð¢òò&ö¦V7F—fR6×ÆR†F×6…öfWF6‚ç’w&ö¢rÂ–âæÕõ‚ç‡’ò–âæÕõ‚çr’&VG2F†—2—†VÂw2FW†VÂ6VçFW#¢‚—2F†RfW'FW‚6†FW"w26Æ—(i"67&VVâf'––ær‡67%öÆö6âÂF×6…öfWF6‚ç’67&VVå÷f'––æw2’À¢òòæ÷F†–ær&WGvVVâ—BæBF†R&7FW&—¦W"†æòFW76VÆÆF–öâ÷"vVöÖWG'’7FvR’ÂF†Rf–Ww÷'B—2F†Rv†öÆRr‚‚F&vWBv—F‚æò7v—§¦ÆRÂæBF†Rv–æF÷r÷&–v–â—2F†RWW"ÆVgB†Æ÷vW"ÆVg@¢òòfÆ—2F†Rf–Ww÷'B’âVæÆ–¶RWe÷&ööbæ÷F†–ær—2&VBg&öÒF†RG&rw2fW'F–6W3¢F†Rf'––ær—2F†—2g&vÖVçBw2÷6—F–öâ'’6öç7G'V7F–öâÂWFòF†R–çFW'öÆF–öâw2&÷VæF–ær‡F†R6×ÆP¢òòÆæG2v—F†–âfÆöB&V6—6–öâöbF†RFW†VÂ6VçFW"Âv†–6‚F†RF–ÆR&VB&WGW&ç2W†7FÇ’§7FF–2–çB67%÷&ööb„çe&ör§"Â–çBÆö6âÂå5T–çFVvW"rÂå5T–çFVvW"‚’°¢çe6†FW"§g2Ò"Óç6…³Ó°¢–b‚g2ÇÂÆö6âÂÇÂÆö6âãÒ3"ÇÂ‡g2Óç&bç67%öÆö6âãâÆö6âb’ÇÂ"Óç6…³%ÒÇÂ"Óç6…³5ÒÇÂ"Óç6…³EÒ’&WGW&â°¢–b†u÷VWVUö÷&–v–åöÆÂÇÂFöÖ–5öÆöEöW‡Æ–6—B‚fuö÷&–v–åöÆÂÂÖVÖ÷'•ö÷&FW%÷&VÆ†VB’’&WGW&â°¢–b…2çg÷7u³Õ³ÒÇÂ2çg÷7u³Õ³ÒÇÂ2çg÷7u³Õ³%ÒÇÂ2çg÷7u³Õ³5Ò’&WGW&â°¢&WGW&â2æ†5÷gÇÂ…2çg³Õ³ÒÓÒbb2çg³Õ³ÒÓÒbb2çg³Õ³%ÒÓÒ†fÆöB•rbb2çg³Õ³5ÒÓÒ†fÆöB”‚“°§Ð¢òòöæRG&röbF†RÆFW"73¢fW&–g’æB&öGV6RF†R7V6–Æ—¦F–öââ&WGW&ç2Ò6ææ÷B&R&÷fVâ‡F†R6ÆÆW"VæG2F†RÖW&vVB&VæFW"72æB&V÷Vç2v—F‚F†R÷&–v–æÂF&vWG2§7FF–2–çBg¥÷%öG&r„çe&ör§"Â6öç7BöG&r¦GrÂ6öç7BöG&uö–G‚¦F’ÂV–çC…÷BÆÖ³…ÒÂõ÷7G&öær–CÄÕDÄgVæ7F–öãâ¦fâ’°¢çe6†FW"§g2Ò"Óç6…³ÒÂ¦g2Ò"Óç6…³Ó²¦fâÒæ–Ã°¢–b‚g2ÇÂg2’²B‚$bVÆÇRæ÷B&÷f&ÆS¢æòfW'FW‚òg&vÖVçB6†FW""Âu÷7FE÷72“²&WGW&â²Ð¢–b‚uög¢ç'C"æFWF‚bb"ç'BæFWF‚bb…2æG2æGBÇÂ2æG2æGrÇÂ2æG2ç7B’’²B‚$bVÆÇRæ÷B&÷f&ÆS¢FWF‚6öÖW2g&öÒF†R&Wf–÷W272æBF†—2G&r†2FWF‚ò7FVæ6–ÂFW7F–æröâ"Âu÷7FE÷72“²&WGW&â²Ð¢–çBg…³3%Ó²f÷"†–çB’Ò²’Â3#²’²²’g…¶•ÒÒÓ²V–çC…÷Bf&Ö6²Ò°¢f÷"†–çB7BÒ²7BÂS²7B²²’²õ÷Vç6fU÷Vç&WF–æVBçe6†FW"§6‚Ò"Óç6…·7EÓ²–b‚6‚ÇÂ6‚Óç7FvRãÒb’6öçF–çVS°¢f÷"†–çB’Ò²’Â6‚Óç&bææ#²’²²’²6öç7Bçfå÷&&–æB¦"Òg6‚Óç&bæ%¶•Ó²V–çCcE÷Bƒ°¢–b†"Óæ¶–æBÓÒådåõ$%ô”Ôr’‚Ò"Óç&VrÂ3"ò2æ–Öu·6‚Óç7FvUÕ¶"Óç&VuÒ¢²VÇ6R–b†"Óæ¶–æBÓÒådåõ$%õDU‚ÇÂ"Óæ¶–æBÓÒådåõ$%õD%Tb’‚ÒFW…÷6Æ÷Eö‚…2çFW…·6‚Óç7FvUÒÂ2æ–Öu·6‚Óç7FvUÒÂ"Óç&Vr“²VÇ6R6öçF–çVS°¢V–çCcE÷B6W"Ò‚ò…÷6W"†‚’¢²–b‚6W"’6öçF–çVS°¢–çB¢ÒÓ²f÷"†–çBÒÒ²ÒÂuög¢æã²Ò²²’–b…"ç'Bæ6öÆ÷%¶ÕÒbb"ç'Bæ6öÆ÷%¶ÕÒÓç6W&–ÂÓÒ6W"’²¢ÒÓ²'&V³²Ð¢–b†¢Â’6öçF–çVS²òòv†B—26×ÆVB—2æ÷BâGF6†ÖVçBöbF†RV&Æ–W"70¢–b‡6‚Òg2ÇÂ"Óæ¶–æBÒådåõ$%õDU‚’²B‚$bVÆÇRæ÷B&÷f&ÆS¢æöâÖg&vÖVçB7FvR÷"æöâ×FW‡GW&R6Æ÷B&VG2F†R&Wf–÷W272w2GF6†ÖVçB"Âu÷7FE÷72“²&WGW&â²Ð¢–b†g2Óç&bæf5öÖ—†VE¶"Óç&VrãâeÒãâ†"Óç&Vrbc2’b’²B‚$bVÆÇRæ÷B&÷f&ÆS¢6Æ÷BVB†2Ö—†VBW6R"Âu÷7FE÷72Â"Óç&Vr“²&WGW&â²Ð¢V–çCcE÷B5ó²–çBÇbÂ6Ã²'E÷7V"‚e"ç'BÂ¢Âg5òÂfÇbÂg6Â“°¢–b†ÇbÇÂ6ÂÇÂÔ‚…"ç'Bæ6öÆ÷%¶¥ÒÓæBçv–GF‚ÂR’Òu÷72çGrÇÂÔ‚…"ç'Bæ6öÆ÷%¶¥ÒÓæBæ†V–v‡BÂR’Òu÷72çF‚’²B‚$bVÆÇRæ÷B&÷f&ÆS¢F†RGF6†ÖVçB—2æ÷BÆWfVÂÆ–W"Â÷"—G26—¦RF–ffW'2"Âu÷7FE÷72“²&WGW&â²Ð¢²òòF†R6×ÆVBf–Wr×W7BÇ6ò&RW†7FÇ’F†RGF6†ÖVçB—G6VÆb‡&Wf–÷W6Ç’öæÇ’F†RGF6†ÖVçB&V–ærÆWfVÂÆ–W"v26†V6¶VBÂæ÷BF†Rf–Wr&÷VæB'’F†R6†FW"ÒÒv†VâF†R6ÖRFW‡GW&R—26×ÆVBF‡&÷Vv‚f–WröbÆWfVÂ²òÆ–W"²ò6†ævVBf÷&ÖBÀ¢òòF†R&ööbv÷VÆB7F–ÆÂ†öÆBÂF†R6×Æ–ærv÷VÆB&R6†ævVBFò&VBöâ×F–ÆRÆWfVÂòF†R÷&–v–æÂf÷&ÖBÂæBF†R–7GW&Rv÷VÆB&Rw&öær’âF†Rf–Wrw27F'F–ærÆWfVÂòÆ–W"×W7B&RæB—G2f÷&ÖB×W7BWVÂF†RGF6†ÖVçBw27GVÂf÷&ÖBÂ÷F†W'v—6RFòæ÷BÖW&vP¢–çBF–EòÒ†–çB’†‚b†ffffb“²fö–B§GeòÒåTÄÃ²–b‡F–Eò’FW…öf7E÷&–B‡F–EòÂgGeò“²õ÷Vç6fU÷Vç&WF–æVB–CÄÕDÅFW‡GW&SâGbÒ…õö'&–FvR–CÄÕDÅFW‡GW&Sâ—Geó°¢V–çC3%÷BeòÒGbç&VçEFW‡GW&Rò‡V–çC3%÷B—Gbç&VçEFW‡GW&Rç—†VÄf÷&ÖB¢‡V–çC3%÷B—Gbç—†VÄf÷&ÖC²V–çC…÷B7w¥õ³EÓ°¢–b…"ç'Bæ†5ö7f–Wu¶¥Òbb"ç'Bæ7f–Wu¶¥Òç6WEöf÷&ÖBbb†çfåöf×E÷Fõö×FÂ…"ç'Bæ7f–Wu¶¥Òæf÷&ÖBÂfeòÂ7w¥ò’bbeò’’eòÒ²òòF†RGF6†ÖVçB—2GF6†VBF‡&÷Vv‚f÷&ÖBÖ6†æv–ærf–Ws¢F¶RF†Bf÷&ÖC²–bVç&V6övæ—¦VBÂFòæ÷BÖW&vP¢ÕDÅ—†VÄf÷&ÖBbÒ„ÕDÅ—†VÄf÷&ÖB–eó²ÕDÅFW‡GW&U7v—§¦ÆT6†ææVÇ2G7rÒGbç7v—§¦ÆS²òò&VBöbF†RF–ÆRF¶W2æò6†ææVÂÖ–ærV—F†W"†Ç†öbâ$t%‚GF6†ÖVçBÂFW…ö'•ö–B¢–çBG7uö–BÒG7rç&VBÓÒÕDÅFW‡GW&U7v—§¦ÆU&VBbbG7ræw&VVâÓÒÕDÅFW‡GW&U7v—§¦ÆTw&VVâbbG7ræ&ÇVRÓÒÕDÅFW‡GW&U7v—§¦ÆT&ÇVRbbG7ræÇ†ÓÒÕDÅFW‡GW&U7v—§¦ÆTÇ†°¢–b‚GbÇÂ‡Gbç&VçEFW‡GW&Rbb‡Gbç&VçE&VÆF—fTÆWfVÂÇÂGbç&VçE&VÆF—fU6Æ–6R’’ÇÂGbç—†VÄf÷&ÖBÒbÇÂG7uö–B’°¢7FF–2–çBÆs²–b†Ær²²Â‚’Äôr‚%¶çfâÖ×FÅÒöâÖ6†—&VBÖW&vS¢6Æ÷BVB6×ÆVBf–Wr†&6RÆWfVÂVÇRÆ–W"VÇRf÷&ÖBVÇRW2’FöW2æ÷BÖF6‚F†RGF6†ÖVçB†ÆWfVÂÆ–W"f÷&ÖBVÇR“²æ÷BÖW&vVB†g&ÖRVÆÇR72VÆÇR’"Â"Óç&VrÀ¢GbbbGbç&VçEFW‡GW&Rò‡Vç6–væVBÆöær—Gbç&VçE&VÆF—fTÆWfVÂ¢VÂÂGbbbGbç&VçEFW‡GW&Rò‡Vç6–væVBÆöær—Gbç&VçE&VÆF—fU6Æ–6R¢VÂÂ‡Vç6–væVBÆöær—Gbç—†VÄf÷&ÖBÂG7uö–Bò""¢"Â7v—§¦ÆVB"Â‡Vç6–væVBÆöær–bÀ¢‡Vç6–væVBÆöærÆöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR’Âu÷7FE÷72“°¢B‚$bVÆÇRæ÷B&÷f&ÆS¢6Æ÷BVB6×ÆVBf–Wr—2æ÷BF†RGF6†ÖVçB—G6VÆb†ÆWfVÂòÆ–W"òf÷&ÖBò6†ææVÂÖ–ærF–ffW"’"Âu÷7FE÷72Â"Óç&Vr“²&WGW&â²ÒÐ¢–çBç’Ò°¢f÷"†–çB2Ò²2Âg2Óç&bææf3²2²²’–b†g2Óç&bæf5÷&Vu¶5ÒÓÒ"Óç&Vr’°¢–b‚†g2Óç&bæf5÷&ö¢ãâ2bò67%÷&ööb‡"Âg2Óç&bæf5öÆö6å¶5ÒÂu÷72çGrÂu÷72çF‚’¢We÷&ööb‡g2Âg2Óç&bæf5öÆö6å¶5ÒÂGrÂF’’’’²B‚$bVÆÇRæ÷B&÷f&ÆS¢6Æ÷BVB6×ÆR6ö÷&F–æFR—2æ÷BF†—2—†VÂ†–çFW'öÆFVB÷WGWBVB’"Âu÷7FE÷72Â"Óç&VrÂg2Óç&bæf5öÆö6å¶5Ò“²&WGW&â²Ð¢g…¶5ÒÒ£²f&Ö6²ÃÒ‡V–çC…÷B’ƒRÃÂ¢“²ç’Ò²Ð¢–b‚ç’’²B‚$bVÆÇRæ÷B&÷f&ÆS¢6Æ÷BVB6×ÆR—2æ÷B–âF†R&Ww&—F&ÆRF&ÆR"Âu÷7FE÷72Â"Óç&Vr“²&WGW&â²ÒÒÐ¢ÖV×6WB†ÆÖÂ†fbÂ‚“°¢f÷"†–çB2Ò²2Âƒ²2²²’–b†g2Óç&bæ6öÆ÷%öÖ6²ãâ2b’²–b†uög¢ç&Ö¶5ÒÂ’²B‚$bVÆÇRæ÷B&÷f&ÆS¢÷WGWB6öÆ÷"VB†2æòÖF6†–ærGF6†ÖVçB"Âu÷7FE÷72Â2“²&WGW&â²ÒÆÖ¶uög¢ç&Ö¶5ÕÒÒ‡V–çC…÷B–3²Ð¢–çB–FVçBÒf&Ö6³²f÷"†–çB2Ò²2Âƒ²2²²’–b‚†g2Óç&bæ6öÆ÷%öÖ6²ãâ2b’bbuög¢ç&Ö¶5ÒÒ2’–FVçBÒ°¢–b†–FVçB’²f÷"†–çBÒ²Âƒ²²²’–b†ÆÖ·ÒÒ†fb’uög¢çw&÷FU·ÒÒ²–b†g¥ö6†V6µööâ‚’’g¥÷6æ÷F¶R†GrÂF’“²&WGW&â²ÒòòFöW2æ÷B&VBF†RF–ÆRæBF†R÷WGWB—2æ÷B6†–gFVC¢W6RF†R÷&–v–æÂfW'6–öâ‡F†Rw&—FRÖ6²7F–ÆÂföÆÆ÷w2F†RÖ–ær¢–b‚g2Óç&bææfò’²&e÷vçEöæ÷FR†g2Óæ¶W’“²B‚$bVÆÇRæ÷B&÷f&ÆS¢æò&VF&ÆR×F–ÆRfW'6–öâ†÷WGWG2v÷VÆBæVVBÖ÷f–ær’"Âu÷7FE÷72“²&WGW&â²Ð¢¦fâÒ&eöfâ†g2Âg‚Âf&Ö6²ÂÆÖ“°¢–b‚¦fâ’&WGW&â°¢ÖV×6WB‚fuög¥öFW62ÂÂ6—¦Vöbuög¥öFW62“²uög¥öFW62æ¶–æBÒ²uög¥öFW62æf&Ö6²Òf&Ö6³²f÷"†–çB’Ò²’Â3#²’²²’uög¥öFW62æg…¶•ÒÒ†–çC…÷B–g…¶•Ó°¢f÷"†–çBÒ²Âƒ²²²’–b†ÆÖ·ÒÒ†fb’uög¢çw&÷FU·ÒÒ°¢–b†g¥ö6†V6µööâ‚’’g¥÷6æ÷F¶R†GrÂF’“°¢uög¥öG&w2²³²&WGW&â°§Ð¢òòF–væ÷7F–2DÕôådåô´U”ETÕÓÆg&vÖVçB6†FW"¶W’&Vf—ƒå³£Æ6÷VçBÂFVfVÇBcå³£Æg&ÖR–çFW'fÃåÕÓ¢vF6‚öæÇ’F†—26Æ72öbG&w2†f÷"&VÂFWf–6W3²æòg&ÖRçVÖ&W"FòwVW72’âF†Rf—'7BâÖF6†–ærG&w2f÷&6R54ETÕw2W"Ö&–æF–ærÆ–æW2öâ„²ò"òC ¢òòFW‡GW&W2v—F‚f÷&ÖBòf–Wrò6†ææVÂ7v—§¦ÆRÂ6×ÆW'2v—F‚ÄôB6Æ×2’ÂÇW2¶¶EÒÆ–æW3¢fW'FW‚GG&–'WFRf÷&ÖG2ÂV6‚fW'FW‚7G&VÒ†f—'7BBfW'F–6W2öbW"×fW'FW‚7G&V×2ÂÆÂ–ç7Fæ6W2öbW"Ö–ç7Fæ6R7G&V×2’æBWfW'’fÇVRöbV6‚Væ–f÷&Ò'VffW"öb&÷F‚7FvW2à¢òò'VâF†R6ÖR7v—F6‚öæ6RöâF†RÖ2æBöæ6RöâF†RFWf–6RæB6ö×&R—FVÒ'’—FVÒFò6VRv†–6‚–çWBF–ffW'2öâF†RFWf–6R…7WW"Ö&–ò4Bv÷&ÆB²&÷w6W"w2gW'’w2ÖvVçF7V&W2–âÓ¢'F–6ÆRVSFFSsSc3csfVRÂ•†öæRöæÇ’Âæ÷B&W&öGV6–&ÆRöâF†RÖ2§7FF–2fö–B¶EöfÆöG2†6öç7B6†"§v†BÂ–çB&VrÂV–çCcE÷BFG"ÂV–çCcE÷B6—¦RÂV–çCcE÷B6’°¢fö–B¦Ö#²V–çCcE÷BÖó²6öç7BV–çC…÷B¦&6S°¢–b‚FG"ÇÂ6—¦R’²g&–çFb‡7FFW'"Â%¶¶EÒW2"VBVæ&÷VæEÆâ"Âv†BÂ&Vr“²&WGW&ã²Ð¢–b‚×FÅöFG"†FG"ÂÂfÖ"ÂfÖò’ÇÂ†&6RÒ²…õö'&–FvR–CÄÕDÄ'VffW#â–Ö"6öçFVçG5Ò’’²g&–çFb‡7FFW'"Â%¶¶EÒW2"VBFG#ÒR6ÆÇ‚æ÷B–âÖVÖ÷'’ööÅÆâ"Âv†BÂ&VrÂ‡Vç6–væVBÆöærÆöær–FG"“²&WGW&ã²Ð¢V–çCcE÷BâÒÔ”â‡6—¦RÂ6’òC²6öç7BfÆöB§bÒ†6öç7BfÆöB¢’†&6R²Öò“°¢g&–çFb‡7FFW'"Â%¶¶EÒW2"VBFG#ÒR6ÆÇ‚6—¦SÒVÆÇRW5Æâ"Âv†BÂ&VrÂ‡Vç6–væVBÆöærÆöær–FG"Â‡Vç6–væVBÆöærÆöær—6—¦RÂVæEö†—G2†FG"Â6—¦R’ò"‡VæF–ærw&—FW2–âF†—27V&Ö—76–öâ’"¢""“°¢f÷"‡V–çCcE÷B’Ò²’Âã²’³Òb’°¢6†"Æ–æU³#EÓ²6—¦U÷BòÒ°¢f÷"‡V–çCcE÷B²Ò“²²Ââbb²Â’²bbbòÂ6—¦VöbÆ–æRÒ#C²²²²’ò³Ò6ç&–çFb†Æ–æR²òÂ6—¦VöbÆ–æRÒòÂ"Rãfr"Âe¶µÒ“°¢g&–çFb‡7FFW'"Â%¶¶EÒ²R3fÆÇ‚W5Æâ"Â‡Vç6–væVBÆöærÆöær’†’¢B’ÂÆ–æR“°¢Ð§Ð§7FF–2–çB¶EöG&r„çe&ör§"Â6öç7BöG&r¦GrÂ6öç7BöG&uö–G‚¦F’’°¢7FF–26†"¶W•³#EÓ²7FF–2–çBÆVgBÒÓÂWfW'“²7FF–2V–çCcE÷BÆ7BÒãVÆÃ°¢–b†ÆVgBÂ’²ÆVgBÒ²6öç7B6†"¦RÒvWFVçb‚$DÕôådåô´U”ETÕ"“°¢–b†Rbb¦R’²6—¦U÷BâÒ7G&77â†RÂ#¢"“²–b†ââ#2’âÒ#3²ÖVÖ7’†¶W’ÂRÂâ“²¶W•¶åÒÒ²6öç7B6†"¦2Ò7G&6‡"†RÂs¢r“²ÆVgBÒ2òFö’†2²’¢c²WfW'’Ò2bb7G&6‡"†2²Âs¢r’òFö’‡7G&6‡"†2²Âs¢r’²’¢°¢ÔÄôr‚$DÕôådåô´U”ETÕ¢g&vÖVçB6†FW"W2ÂVBG&w2ÆövvVB–âFWF–ÂW2…¶¶EÒò·75ÒÆ–æW2’"Â¶W’ÂÆVgBÂWfW'’ò"ÂBÖ÷7Böæ6RWfW'’F†BÖç’g&ÖW2"¢""“²ÒÐ¢–b†ÆVgBÃÒÇÂ"Óç6…³ÒÇÂ7G&æ6×‡"Óç6…³ÒÓæ¶W’Â¶W’Â7G&ÆVâ†¶W’’’’&WGW&â°¢V–çCcE÷Bg"ÒFöÖ–5öÆöB‚fuö7W%ög&ÖR“°¢–b†WfW'’âbbÆ7BÒãVÆÂbbg"ÂÆ7B²‡V–çCcE÷B–WfW'’’&WGW&â²òòBÖ÷7Böæ6RWfW'’WfW'–g&ÖW3¢6×ÆW27&VB÷fW"F†R6W76–öâÂæòæVVBFòwVW72F†Rg&ÖRöbF†R66VæP¢Æ7BÒg#²ÆVgBÒÓ°¢–çB6÷VçBÒGròGrÓæ6÷VçB¢F’Óæ6÷VçBÂ–ç7BÒGròGrÓæ–ç7Fæ6W2¢F’Óæ–ç7Fæ6W2Âf—'7BÒGròGrÓæf—'7B¢Â&–ç7BÒGròGrÓæ&6Uö–ç7Fæ6R¢F’Óæ&6Uö–ç7Fæ6S°¢g&–çFb‡7FFW'"Â%¶¶EÒbVÆÇR72VÆÇRg3ÒW2g3ÒW2&–ÒVBfW'F–6W2VB–ç7Fæ6W2VBf—'7BVB&6U÷fW'FW‚VB&6Uö–ç7Fæ6RVBF&vWBW5Æâ"Â‡Vç6–væVBÆöærÆöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR’Âu÷7FE÷72À¢"Óç6…³Òò"Óç6…³ÒÓæ¶W’¢"Ò"Â"Óç6…³ÒÓæ¶W’ÂGròGrÓç&–Ò¢F’Óç&–ÒÂ6÷VçBÂ–ç7BÂf—'7BÂF’òF’Óæ&6U÷fW'FW‚¢Â&–ç7BÂ"ç'Bæâbb"ç'Bæ6öÆ÷%³Òò'–W2"¢&æò"“°¢f÷"†–çB’Ò²’Â2ææGG"bb’Âc²’²²’g&–çFb‡7FFW'"Â%¶¶EÒGG"VBf÷&ÖBR7‚öfg6WBWR7G&VÒWUÆâ"Â’Â‡Vç6–væVB•2æGG%¶•Òæf×BÂ‡Vç6–væVB•2æGG%¶•Òæöfg6WBÂ‡Vç6–væVB•2æGG%¶•Òç7G&VÒ“°¢f÷"†–çB5òÒ²5òÂ2æç7G&VÒbb5òÂc²5ò²²’°¢V–çC3%÷B7BÒ2ç7G&VÕ·5õÒç7G&–FRÂGbÒ2ç7G&VÕ·5õÒæF—f—6÷#°¢g&–çFb‡7FFW'"Â%¶¶EÒ7G&VÒVB7G&–FRWRF—f—6÷"WR'VffW"R6ÆÇ‚6—¦RVÆÇUÆâ"Â5òÂ7BÂGbÂ‡Vç6–væVBÆöærÆöær•2çf'Ve·5õÒæFG&W72Â‡Vç6–væVBÆöærÆöær•2çf'Ve·5õÒç6—¦R“°¢–b‚2çf'Ve·5õÒæFG&W72ÇÂ7B’6öçF–çVS°¢V–çCcE÷Bg&öÒÒ‡V–çCcE÷B’†Gbò&–ç7B¢†F’òF’Óæ&6U÷fW'FW‚¢f—'7B’’¢7BÂâÒ‡V–çCcE÷B—7B¢†Gbò‡V–çCcE÷B”Ô”â†–ç7BÂ‚’¢B“°¢–b†g&öÒÂ2çf'Ve·5õÒç6—¦R’¶EöfÆöG2‚'7G&VÒ"Â5òÂ2çf'Ve·5õÒæFG&W72²g&öÒÂÔ”â†âÂ2çf'Ve·5õÒç6—¦RÒg&öÒ’ÂS"“°¢Ð¢f÷"†–çB7FrÒ²7FrÂ#²7Fr²²’²çe6†FW"§6‚Ò"Óç6…·7FuÓ²–b‚6‚’6öçF–çVS°¢f÷"†–çB’Ò²’Â6‚Óç&bææ#²’²²’–b‡6‚Óç&bæ%¶•Òæ¶–æBÓÒådåõ$%ô4%Tbbb6‚Óç&bæ%¶•Òç&VrÂb’°¢6öç7Bö&–æEö'Vb§RÒe2çV&õ·7FuÕ·6‚Óç&bæ%¶•Òç&VuÓ²¶EöfÆöG2‡7Frò&g&vÖVçBVæ–f÷&Ò"¢'fW'FW‚Væ–f÷&Ò"Â6‚Óç&bæ%¶•Òç&VrÂRÓæFG"ÂRÓç6—¦RÂC“b“²ÒÐ¢&WGW&â°§Ð¢òòF–væ÷7F–2DÕôådåôÔu44ãÓÆg&vÖVçB¶W“ã£Ç66âWfW'’âg&ÖW2ÂFVfVÇB3ã£Æg&ÖW2W"†6RÂFVfVÇBcã¢öâÖFWf–6R6VÆbÖ6†V6²f÷"F†RÖvVçF7V&W2…7WW"Ö&–ò4Bv÷&ÆB²&÷w6W"w2gW'’ÓÂ•†öæRöæÇ’Âæ÷B&W&öGV6–&ÆRöâF†RÖ2’à¢òò–â66âg&ÖRÂ26ööâ2V6‚&VæFW"72VæG2Â6ö×WFR¶W&æVÂ&VG2—G26öÆ÷"GF6†ÖVçG2‡&VBvöW2F‡&÷Vv‚F†R6ÖRf÷&ÖBFV6öFR2F†RvÖRw26†FW"6×ÆW2’æBvF†W'2W"gƒb&Æö6³¢F†RÖvVçF—†VÂ6÷VçB…"æB"&÷F‚ãÒã‚öbF†RÆ&vW7B6†ææVÂÂrÃÒãöb—B’À¢òòF†RæöâÖf–æ—FR—†VÂ6÷VçBæBV6‚6†ææVÂw2Ö–âòÖ‚ââGF6†ÖVçBv—F‚&Æö6²†öÆF–æröæR6–ævÆRÖvVçFfÇVRF‡&÷Vv†÷WB†W†7FÇ’v†BF†R7V&W2–âF†R•†öæR67&VVç6†÷B&RÂÆ–væVBFòF†Rb×—†VÂw&–B’÷"v—F‚æöâÖf–æ—FRfÇVW2vWG2¶ÖuÒÆ–æR†–æFW‚öbF†R72v—F†–âF†Rg&ÖRÂF—7Fæ6Rg&öÒF†R72†öÆF–ærF†—2'F–6ÆRÂGF6†ÖVçB6W&–Âòf÷&ÖBò6—¦RÂ&Æö6²÷6—F–öâæBfÇVR’à¢òòf÷W"†6W2&÷FFS¢†6RG&w22W7VÂÂ†6RG&÷2WfW'’G&röbF†—2g&vÖVçB6†FW"Â†6W2"ò2¶VWF†RvVöÖWG'’æB&WÆ6RF†Rg&vÖVçB÷WGWBv—F‚6öç7FçB†w&W’ò'&–v‡BÖvVçF’â6ö×&–ærF†R†6W2FVÆÇ2v†WF†W"F†—2G&r6W6W2—@¢òòæBv†WF†W"—G2g&vÖVçB÷WGWB‡FW‡GW&R&VG2ÂfÇVR&ævR’ÖGFW'3²F†Rf—'7B72v†W&RF†RÖvVçF&Æö6·26†÷rFVÆÇ2v†–6‚7FWvöW2w&öæp§7FF–26†"uö×5ö¶W•³#EÓ²7FF–2–çBuö×5öWfW'’Âuö×5÷ÆVâÂuö×5ö–æ—C°§7FF–2–çB×5ööâ‡fö–B“°§7FF–26öç7B6†"¦6öç7Bµö×5÷†6U³EÒÒ²&æ÷&ÖÂ"Â'6†FW"G&÷VB"Â'6†FW"÷WGWG26öç7FçBw&’"Â'6†FW"÷WGWG26öç7FçB'&–v‡BÖvVçF'Ó°§7FF–2–çB×5÷†6Uöæ÷r‡fö–B’²&WGW&â†–çB’†FöÖ–5öÆöB‚fuö7W%ög&ÖR’ò‡V–çCcE÷B–uö×5÷ÆVâRB“²Ð¢òò†6W2"ò3¢F†—26†FW"w2G&w2¶VWF†V—"vVöÖWG'’æBF†Rg&vÖVçB7FvR—2&WÆ6VB'’6öç7FçB÷WGWB†w&W’ãRò'&–v‡BÖvVçFSÃ"ÃS’âF†R—VÆ–æR66†R¶W’6'&–W2F†R&WÆ6VÖVçBgVæ7F–öâÂ¶WB'Bg&öÒF†R÷&–v–æÂ—VÆ–æS²F†W6R—VÆ–æW2&Ræ÷B&V6÷&FVB–âF†R&Wv&ÒÆ—7@§7FF–2–CÄÕDÄgVæ7F–öãâ×5ög5ö÷fW'&–FR„çe6†FW"¦g2’°¢–b‚g2ÇÂ×5ööâ‚’ÇÂ7G&æ6×†g2Óæ¶W’Âuö×5ö¶W’Â7G&ÆVâ†uö×5ö¶W’’’’&WGW&âæ–Ã°¢–çB‚Ò×5÷†6Uöæ÷r‚“²–b‡‚Â"’&WGW&âæ–Ã°¢7FF–2–CÄÕDÄgVæ7F–öãâfå³%Ó²7FF–2–çBf–ÆVC°¢–b‚få³Òbbf–ÆVB’²å4W'&÷"¦RÒæ–Ã²–CÄÕDÄÆ–'&'“âÆ–"Ò¶uöFWbæWtÆ–'&'•v—F…6÷W&6S¤"6–æ6ÇVFRÆÖWFÅ÷7FFÆ–#åÆçW6–æræÖW76RÖWFÃµÆâ ¢&g&vÖVçBfÆöCB×6r‚’²&WGW&âfÆöCBƒãRÂãRÂãRÂã“²ÕÆæg&vÖVçBfÆöCB×6Ò‚’²&WGW&âfÆöCBƒSãÂ"ãÂSãÂãb“²ÕÆâ"÷F–öç3¦æ–ÂW'&÷#¢fUÓ°¢få³ÒÒ¶Æ–"æWtgVæ7F–öåv—F„æÖS¤&×6r%Ó²få³ÒÒ¶Æ–"æWtgVæ7F–öåv—F„æÖS¤&×6Ò%Ó°¢–b‚få³ÒÇÂfå³Ò’²f–ÆVBÒ²ÔÄôr‚%µ$TEÒDÕôådåôÔu44ã¢6ææ÷B6ö×–ÆRF†R6öç7FçBg&vÖVçB6†FW'2W2"ÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“²ÒÐ¢&WGW&âf–ÆVBòæ–Â¢få·‚Ò%Ó°§Ð§7FF–2–çB×5ööâ‡fö–B’°¢–b‚uö×5ö–æ—B’²uö×5ö–æ—BÒ²6öç7B6†"¦RÒvWFVçb‚$DÕôådåôÔu44â"“°¢–b†Rbb¦R’²6—¦U÷BâÒ7G&77â†RÂ#¢"“²–b†ââ#2’âÒ#3²ÖVÖ7’†uö×5ö¶W’ÂRÂâ“²uö×5ö¶W•¶åÒÒ°¢6öç7B6†"¦2Ò7G&6‡"†RÂs¢r“²uö×5öWfW'’Ò2òFö’†2²’¢3²2Ò2ò7G&6‡"†2²Âs¢r’¢åTÄÃ²uö×5÷ÆVâÒ2òFö’†2²’¢c°¢–b†uö×5öWfW'’Â’uö×5öWfW'’Ò3²–b†uö×5÷ÆVâÂuö×5öWfW'’’uö×5÷ÆVâÒc°¢ÔÄôr‚$DÕôådåôÔu44ã¢g&vÖVçB6†FW"W3²WfW'’VBg&ÖW2Â66ç2V6‚72w26öÆ÷"GF6†ÖVçG2f÷"ÖvVçF&Æö6·3²WfW'’VBg&ÖW2Â7v—F6†W2†6Rƒæ÷&ÖÂòG&÷F†—26†FW"w2G&w2ò"—B÷WGWG26öç7FçBw&’ò2—B÷WGWG26öç7FçB'&–v‡BÖvVçF’…¶ÖuÒÆ–æW2’"Âuö×5ö¶W’Âuö×5öWfW'’Âuö×5÷ÆVâ“²ÒÐ¢&WGW&âuö×5ö¶W•³ÒÒ°§Ð§7FF–2V–çCcE÷Buö×5÷72ÒãVÆÂÂuö×5÷g&ÖRÒãVÆÂÂuö×5öG&vâÂuö×5öG&÷VC°¢òò&WGW&ç2ÒF†—2†6RG&÷2F†RG&p§7FF–2–çB×5öG&r„çe&ör§"’°¢–b‚×5ööâ‚’ÇÂ"Óç6…³ÒÇÂ7G&æ6×‡"Óç6…³ÒÓæ¶W’Âuö×5ö¶W’Â7G&ÆVâ†uö×5ö¶W’’’’&WGW&â²òòFVfW'&VBG&r&WGW&ç2&Vf÷&RF†—2v†Vâ&V6÷&FVBæBöæÇ’'&—fW2†W&Rv†Vâ&WÆ–VC¢WfW'’G&r76W2W†7FÇ’öæ6P¢V–çCcE÷BbÒFöÖ–5öÆöB‚fuö7W%ög&ÖR“°¢–b†bò‡V–çCcE÷B–uö×5÷ÆVâRBÓÒ’²uö×5öG&÷VB²³²&WGW&â²Ð¢uö×5öG&vâ²³²–b†uö×5÷g&ÖRÒb’²uö×5÷g&ÖRÒc²uö×5÷72Òu÷7FE÷73²ÒòòF†R72v†W&RF†—2'F–6ÆRf—'7BG&w2F†—2g&ÖP¢&WGW&â°§Ð§7FF–26öç7B6†"µö×5ö×6ÅµÒÐ¢"6–æ6ÇVFRÆÖWFÅ÷7FFÆ–#åÆçW6–æræÖW76RÖWFÃµÆâ ¢&¶W&æVÂfö–B×2‡FW‡GW&S&CÆfÆöBÂ66W73£§&VCâBµ·FW‡GW&Rƒ•ÕÒÂFWf–6RV–çB¦òµ¶'VffW"ƒ•ÕÒÂV–çC"rµ·F‡&VE÷÷6—F–öåö–åöw&–EÕÒÂV–çC"Frµ·F‡&VFw&÷W÷÷6—F–öåö–åöw&–EÕÒÅÆâ ¢"V–çBÆ’µ·F‡&VEö–æFW…ö–å÷F‡&VFw&÷WÕÒÂV–çB6’µ·6–ÖFw&÷Wö–æFW…ö–å÷F‡&VFw&÷WÕÒÂV–çB6Âµ·F‡&VEö–æFW…ö–å÷6–ÖFw&÷WÕÒ’µÆâ ¢"F‡&VFw&÷WV–çB6…³…Õ³…Ó²&ööÂ–âÒrç‚Âõ³Òbbrç’Âõ³Ó²fÆöCB2Ò–âòBç&VB†r’¢fÆöCBƒ“²fÆöBÒÒÖ‚†2ç"ÂÖ‚†2ærÂ2æ"’“µÆâ ¢"&ööÂ&BÒ–âbb†—6æâ†2ç"’ÇÂ—6æâ†2ær’ÇÂ—6æâ†2æ"’ÇÂ—6–æb†2ç"’ÇÂ—6–æb†2ær’ÇÂ—6–æb†2æ"’“µÆâ ¢"&ööÂÖrÒ–âbb&BbbÒâãbb2ç"ãÒã‚¢Òbb2æ"ãÒã‚¢Òbb2ærÃÒã¢ÓµÆâ ¢"V–çB"Ò5÷G—SÇV–çCâ†Ö‚†2ç"Âã’’ÂvrÒ5÷G—SÇV–çCâ†Ö‚†2ærÂã’’Â"Ò5÷G—SÇV–çCâ†Ö‚†2æ"Âã’“µÆâ ¢"V–çBe³…ÒÒ·6–ÖE÷7VÒ†ÖròR¢R’Â6–ÖE÷7VÒ†&BòR¢R’Â6–ÖEöÖ–â†–âò"¢ãR’Â6–ÖEöÖ‚†–âò"¢R’Â6–ÖEöÖ–â†–âòvr¢ãR’Â6–ÖEöÖ‚†–âòvr¢R’Â6–ÖEöÖ–â†–âò"¢ãR’Â6–ÖEöÖ‚†–âò"¢R—ÓµÆâ ¢"–b‡6ÂÓÒ’f÷"†–çB²Ò²²Âƒ²²²²’6…·6•Õ¶µÒÒe¶µÓµÆâ ¢"F‡&VFw&÷Wö&'&–W"†ÖVÕöfÆw3£¦ÖVÕ÷F‡&VFw&÷W“²–b†Æ’’&WGW&ãµÆâ ¢"f÷"†–çB2Ò²2Âƒ²2²²’²e³Ò³Ò6…·5Õ³Ó²e³Ò³Ò6…·5Õ³Ó²f÷"†–çB²Ò#²²Âƒ²²³Ò"’²e¶µÒÒÖ–â‡e¶µÒÂ6…·5Õ¶µÒ“²e¶²²ÒÒÖ‚‡e¶²²ÒÂ6…·5Õ¶²²Ò“²ÒÕÆâ ¢"FWf–6RV–çB§Òò²B²B¢‡Frç’¢õ³%Ò²Frç‚“²³ÒÒe³ÒÂe³ÒÃÂbÂ‡e³%ÒÓÒe³5Òbbe³EÒÓÒe³UÒbbe³eÒÓÒe³uÒòRÃÂ3¢R“²³ÒÒe³5Ó²³%ÒÒe³UÓ²³5ÒÒe³uÓ²ÕÆâ#°¢6FVf–æRÕ5ô%TbƒgRÃÂ#¢6FVf–æRÕ5ôÔ„R#S`§G—VFVb7G'V7B²V–çCcE÷B75ö–G‚Â6W#²–çBGBÂF—7BÂf×C²V–çC3%÷BrÂ‚Âöfc²Ò×5öVçC°§G—VFVb7G'V7B²õ÷7G&öær–CÄÕDÄ'VffW#â#²V–çCcE÷Bg&ÖRÂfVæ6S²V–çC3%÷BW6VC²–çBâÂ'W7’ÂgVÆÃ²×5öVçBU´Õ5ôÔ„UÓ²Ò×5÷66ã°§7FF–2×5÷66âuö×5÷5³%Ó²7FF–2–çBuö×5ö7W"ÒÓ²òòv†–6‚'VffW"F†—266âg&ÖRw&—FW2‚ÓÒæ÷B66ææ–ærF†—2g&ÖR§7FF–2V–çCcE÷Buö×5ö7W&bÒãVÆÂÂuö×5öf—'7C°§7FF–2õ÷7G&öær–CÄÕDÅFW‡GW&Sâuö×5öGE³…Ó²7FF–2å5T–çFVvW"uö×5öÇe³…ÒÂuö×5÷6Å³…Ó²7FF–2V–çCcE÷Buö×5ö6W%³…Ó²7FF–2–çBuö×5öæ²7FF–2V–çCcE÷Buö×5ö73°§7FF–2–çBuö×5÷6WÒÓÂuö×5÷62Âuö×5ö†—FbÂuö×5öÖ†&Æ²Âuö×5÷66ã²7FF–2V–çCcE÷Buö×5öcÂuö×5öcÂuö×5÷Ös²òòF†R†6R&V–ærFÆÆ–VB†g&ÖRòg&ÖW2W"†6R’æB—G2F÷FÇ0§7FF–2fö–B×5÷öÆÂ‡fö–B’°¢f÷"†–çB²Ò²²Â#²²²²’²×5÷66â§2Òfuö×5÷5¶µÓ²–b‚2Óæ'W7’ÇÂuö×5ö7W"ÓÒ²ÇÂ2ÓæfVæ6RâuöFöæR’6öçF–çVS°¢–çB6WÒ†–çB’‡2Óæg&ÖRò‡V–çCcE÷B–uö×5÷ÆVâ“°¢–b‡6WÒuö×5÷6W’°¢–b†uö×5÷6WãÒ’ÔÄôr‚%¶ÖuÒ†6RVB‚W2’g&ÖW2VÆÇ^(	2VÆÇS¢66ææVBVBg&ÖW2Âg&ÖW2v—F‚Væ–f÷&ÒÖvVçF&Æö6·2†v†öÆRl9sb&Æö6²öböæRÖvVçFfÇVR’VBÂBÖ÷7BVB&Æö6·2–âöæRg&ÖS²ÖvVçF—†VÇ2–âF†RGF6†ÖVçG2öbF†R6†FW"w272RãbW"g&ÖR‚VBg&ÖW2“²F†—26†FW"G&vâVÆÇRF–ÖW2ÂG&÷VBVÆÇRF–ÖW26òf""À¢uö×5÷6WRBÂµö×5÷†6U¶uö×5÷6WREÒÂ‡Vç6–væVBÆöærÆöær–uö×5öcÂ‡Vç6–væVBÆöærÆöær–uö×5öcÂuö×5÷62Âuö×5ö†—FbÂuö×5öÖ†&Æ²Âuö×5÷66âò†F÷V&ÆR–uö×5÷Öròuö×5÷66â¢ãÂuö×5÷66âÀ¢‡Vç6–væVBÆöærÆöær–uö×5öG&vâÂ‡Vç6–væVBÆöærÆöær–uö×5öG&÷VB“°¢uö×5÷6WÒ6W²uö×5÷62Òuö×5ö†—FbÒuö×5öÖ†&Æ²Òuö×5÷66âÒ²uö×5÷ÖrÒ²uö×5öcÒ2Óæg&ÖS²Ð¢uö×5÷62²³²uö×5öcÒ2Óæg&ÖS²–çBf&Æ²ÒÂ6VVâÒ°¢f÷"†–çB’Ò²’Â2Óæã²’²²’²6öç7B×5öVçB¦RÒg2ÓæU¶•Ó²6öç7BV–çC3%÷B¦òÒ†6öç7BV–çC3%÷B¢’‚†6öç7BV–çC…÷B¢—2Óæ"æ6öçFVçG2²RÓæöfb“°¢V–çC3%÷BGrÒ†RÓçr²R’òbÂF‚Ò†RÓæ‚²R’òc²V–çCcE÷B&BÒÂÖrÒ²–çBV&Æ²ÒÂÖ&Æ²ÒÂæÂÒ²6†"Æ–æU³CÓ²6—¦U÷BÆòÒ°¢f÷"‡V–çC3%÷BBÒ²BÂGr¢Fƒ²B²²’²6öç7BV–çC3%÷B§Òò²B²B¢C²V–çC3%÷BæÖrÒ³Òb†fffbÂæ&BÒ³Òãâbbƒvffc²&B³Òæ&C²Ör³ÒæÖs°¢V–çC3%÷Bç‚ÒÔ”âƒgRÂRÓçrÒBRGr¢b’¢Ô”âƒgRÂRÓæ‚ÒBòGr¢b“°¢–çBVæ’Ò³Òãâ3²òòF†Rv†öÆR&Æö6²†öÆG2öæRfÇVP¢–çBÆÆÒÒæÖrÓÒçƒ°¢–b†ÆÆÒ’Ö&Æ²²³²–b†ÆÆÒbbVæ’bbç‚ãÒcB’V&Æ²²³°¢–b‚‚†ÆÆÒbbVæ’bbç‚ãÒcB’ÇÂæ&B’bbæÂÂ‚bbÆòÂ6—¦VöbÆ–æRÒ#’²fÆöB"ÂrÂ#²ÖVÖ7’‚g"Âg³ÒÂB“²ÖVÖ7’‚frÂg³%ÒÂB“²ÖVÖ7’‚f"Âg³5ÒÂB“²æÂ²³°¢Æò³Ò6ç&–çFb†Æ–æR²ÆòÂ6—¦VöbÆ–æRÒÆòÂ"²WRÂWRW2Ö‚VrVrVræöâÖf–æ—FRWUÒ"ÂBRGr¢bÂBòGr¢bÂVæ’ò"Væ–f÷&Ò"¢""Â"ÂrÂ"Âæ&B“²ÒÐ¢f&Æ²³ÒV&Æ³²–b†RÓçF—7BÓÒbbRÓæGBÓÒ’²uö×5÷Ör³ÒÖs²6VVâÒ²Ð¢–b‡V&Æ²ÇÂ&B’ÔÄôr‚%¶ÖuÒbVÆÇR†6RVÆÇR72VÆÇR‡6†FW"w272R¶B’GF6†ÖVçBVB6W#ÒVÆÇRf÷&ÖBVBWW‚WS¢Væ–f÷&ÒÖvVçF&Æö6·2VB†ÆÂÖÖvVçF&Æö6·2VB’ÂæöâÖf–æ—FR—†VÇ2VÆÇRW2"Â‡Vç6–væVBÆöærÆöær—2Óæg&ÖRÂ‡Vç6–væVBÆöærÆöær’‡2Óæg&ÖRò‡V–çCcE÷B–uö×5÷ÆVâRB’À¢‡Vç6–væVBÆöærÆöær–RÓç75ö–G‚ÂRÓçF—7BÂRÓæGBÂ‡Vç6–væVBÆöærÆöær–RÓç6W"ÂRÓæf×BÂRÓçrÂRÓæ‚ÂV&Æ²ÂÖ&Æ²Â‡Vç6–væVBÆöærÆöær–&BÂÆ–æR“°¢Ð¢–b†f&Æ²’uö×5ö†—Fb²³²–b†f&Æ²âuö×5öÖ†&Æ²’uö×5öÖ†&Æ²Òf&Æ³²uö×5÷66â³Ò6VVã°¢–b‡2ÓægVÆÂ’ÔÄôr‚%¶ÖuÒbVÆÇR66â'VffW"gVÆÃ²F†R&VÖ–æ–ær76W2öbF†—2g&ÖRvW&Ræ÷B66ææVB"Â‡Vç6–væVBÆöærÆöær—2Óæg&ÖR“°¢2Óæ'W7’Ò²2ÓæâÒ²2ÓçW6VBÒ²2ÓægVÆÂÒ²Ð§Ð§fö–B×5öæ÷FR„ÕDÃE&VæFW%74FW67&—F÷"§BÂå5T–çFVvW"æ2’°¢uö×5öæÒ²–b‚×5ööâ‚’’&WGW&ã°¢V–çCcE÷BbÒFöÖ–5öÆöB‚fuö7W%ög&ÖR“°¢–b†bÒuö×5ö7W&b’²òòæWrg&ÖS¢F†R&Wf–÷W266âg&ÖRw26÷VçG2&R&ö6W76VBöæ6R—G27V&Ö—76–öâ6ö×ÆWFW0¢uö×5ö7W&bÒc²uö×5öf—'7BÒu÷7FE÷72²²–b†uö×5ö7W"ãÒ’uö×5ö7W"ÒÓ°¢–b†bR‡V–çCcE÷B–uö×5öWfW'’ÓÒ’²f÷"†–çB²Ò²²Â#²²²²’–b‚uö×5÷5¶µÒæ'W7’’²uö×5ö7W"Ò³²'&V³²Ð¢–b†uö×5ö7W"ãÒ’²×5÷66â§2Òfuö×5÷5¶uö×5ö7W%Ó°¢–b‚2Óæ"’²2Óæ"Ò¶uöFWbæWt'VffW%v—F„ÆVæwFƒ¤Õ5ô%Tb÷F–öç3¤ÕDÅ&W6÷W&6U7F÷&vTÖöFU6†&VEÓ²–b‚2Óæ"’²uö×5ö7W"ÒÓ²&WGW&ã²Ò&W5öFB‡2Óæ"“²Ð¢2Óæ'W7’Ò²2Óæg&ÖRÒc²2ÓæâÒ²2ÓçW6VBÒ²2ÓægVÆÂÒ²2ÓæfVæ6RÒ²ÒÒÐ¢×5÷öÆÂ‚“°¢–b†uö×5ö7W"Â’&WGW&ã°¢uö×5ö72Òu÷7FE÷72²°¢f÷"„å5T–çFVvW"’Ò²’Âæ2bb’Âƒ²’²²’²–CÄÕDÅFW‡GW&SâBÒBæ6öÆ÷$GF6†ÖVçG5¶•ÒçFW‡GW&S²–b‚B’6öçF–çVS°¢uö×5öGE¶uö×5öæÒÒC²uö×5öÇe¶uö×5öæÒÒBæ6öÆ÷$GF6†ÖVçG5¶•ÒæÆWfVÃ²uö×5÷6Å¶uö×5öæÒÒBæ6öÆ÷$GF6†ÖVçG5¶•Òç6Æ–6S²uö×5ö6W%¶uö×5öæÒÒ"ç'Bæ6öÆ÷%¶•Òò"ç'Bæ6öÆ÷%¶•ÒÓç6W&–Â¢²uö×5öæ²³²Ð§Ð§fö–B×5ögFW"‡fö–B’°¢–b‚uö×5öæ’&WGW&ã°¢–çBæÒuö×5öæ²uö×5öæÒ°¢–b†uö×5ö7W"ÂÇÂu÷7FE÷72Òuö×5ö72’²f÷"†–çB’Ò²’Âæ²’²²’uö×5öGE¶•ÒÒæ–Ã²&WGW&ã²Ð¢7FF–2–çBf–ÆVC²7FF–2–CÄÕDÄ6ö×WFU—VÆ–æU7FFSâ3²7FF–2–CÄÕDÃD&wVÖVçEF&ÆSâC²7FF–2–çBÆövvVEöGC°¢–b†f–ÆVB’&WGW&ã°¢–b‚2’²å4W'&÷"¦RÒæ–Ã²–CÄÕDÄÆ–'&'“âÆ–"Ò¶uöFWbæWtÆ–'&'•v—F…6÷W&6S¤†µö×5ö×6Â’÷F–öç3¦æ–ÂW'&÷#¢fUÓ²–CÄÕDÄgVæ7F–öãâfâÒÆ–"ò¶Æ–"æWtgVæ7F–öåv—F„æÖS¤&×2%Ò¢æ–Ã°¢2Òfâò¶uöFWbæWt6ö×WFU—VÆ–æU7FFUv—F„gVæ7F–öã¦fâW'&÷#¢fUÒ¢æ–Ã°¢ÕDÃD&wVÖVçEF&ÆTFW67&—F÷"¦BÒ´ÕDÃD&wVÖVçEF&ÆTFW67&—F÷"æWuÓ²BæÖ…FW‡GW&T&–æD6÷VçBÒ²BæÖ„'VffW$&–æD6÷VçBÒ²BÒ&V5÷F&ÆR…¶uöFWbæWt&wVÖVçEF&ÆUv—F„FW67&—F÷#¦BW'&÷#¢fUÒ“°¢–b‚2ÇÂB’²f–ÆVBÒ²ÔÄôr‚%µ$TEÒDÕôådåôÔu44ã¢6ææ÷B7&VFRF†R6ö×WFR—VÆ–æRW2"ÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“²&WGW&ã²ÒÐ¢×5÷66â§2Òfuö×5÷5¶uö×5ö7W%Ó²–çBF—7BÒuö×5÷g&ÖRÓÒ2Óæg&ÖRbbuö×5÷72ÒãVÆÂò†–çB’‚†–çCcE÷B–u÷7FE÷72Ò†–çCcE÷B–uö×5÷72’¢““““°¢f÷"†–çB’Ò²’Âæ²’²²’²–CÄÕDÅFW‡GW&SâBÒuö×5öGE¶•Ó²uö×5öGE¶•ÒÒæ–Ã°¢ÕDÅ—†VÄf÷&ÖBbÒBç—†VÄf÷&ÖC°¢–çBö²ÒbÓÒÕDÅ—†VÄf÷&ÖE$t$…Væ÷&ÒÇÂbÓÒÕDÅ—†VÄf÷&ÖE$t$…Væ÷&Õ÷5$t"ÇÂbÓÒÕDÅ—†VÄf÷&ÖD$u$…Væ÷&ÒÇÂbÓÒÕDÅ—†VÄf÷&ÖD$u$…Væ÷&Õ÷5$t"ÇÂbÓÒÕDÅ—†VÄf÷&ÖE$t#%Væ÷&Ð¢ÇÂbÓÒÕDÅ—†VÄf÷&ÖE$s#fÆöBÇÂbÓÒÕDÅ—†VÄf÷&ÖE$t$dfÆöBÇÂbÓÒÕDÅ—†VÄf÷&ÖE$t$eVæ÷&ÒÇÂbÓÒÕDÅ—†VÄf÷&ÖE$t$3$fÆöC°¢–b‚ö²ÇÂ‡BçW6vRbÕDÅFW‡GW&UW6vU6†FW%&VB’ÇÂBç6×ÆT6÷VçBâÇÂ‡Bçv–GF‚ãâuö×5öÇe¶•Ò’ÂcBÇÂBç7F÷&vTÖöFRÓÒÕDÅ7F÷&vTÖöFTÖVÖ÷'–ÆW72ÇÂ‡BçFW‡GW&UG—RÒÕDÅFW‡GW&UG—S$BbbBçFW‡GW&UG—RÒÕDÅFW‡GW&UG—S$D'&’’’6öçF–çVS°¢–b‡F—7BÓÒbbÆövvVEöGB²²’ÔÄôr‚%¶ÖuÒGF6†ÖVçBVBöbF†R6†FW"w273¢6W#ÒVÆÇRf÷&ÖBVÇRVÇW‚VÇRÆWfVÂVÇRÆ–W"VÇR7F÷&vRVÇRW6vRR6Ç‚6ö×&W76–öâG—RVÆBÆÆ÷tuT÷F–Ö—¦VD6öçFVçG2VB"Â’Â‡Vç6–væVBÆöærÆöær–uö×5ö6W%¶•ÒÂ‡Vç6–væVBÆöær—bÀ¢‡Vç6–væVBÆöær—Bçv–GF‚Â‡Vç6–væVBÆöær—Bæ†V–v‡BÂ‡Vç6–væVBÆöær–uö×5öÇe¶•ÒÂ‡Vç6–væVBÆöær–uö×5÷6Å¶•ÒÂ‡Vç6–væVBÆöær—Bç7F÷&vTÖöFRÂ‡Vç6–væVBÆöær—BçW6vRÂ†Æöær—Bæ6ö×&W76–öåG—RÂ†–çB—BæÆÆ÷tuT÷F–Ö—¦VD6öçFVçG2“°¢V–çC3%÷BrÒ‡V–çC3%÷B”Ô‚‡Bçv–GF‚ãâuö×5öÇe¶•ÒÂVÂ’Â‚Ò‡V–çC3%÷B”Ô‚‡Bæ†V–v‡Bãâuö×5öÇe¶•ÒÂVÂ’ÂGrÒ‡r²R’òbÂF‚Ò†‚²R’òbÂæVVBÒb²Gr¢F‚¢c°¢–b‡2ÓæâãÒÕ5ôÔ„RÇÂ2ÓçW6VB²æVVBâÕ5ô%Tb’²2ÓægVÆÂÒ²6öçF–çVS²Ð¢–CÄÕDÅFW‡GW&SâbÒC°¢–b‡BçFW‡GW&UG—RÒÕDÅFW‡GW&UG—S$BÇÂuö×5öÇe¶•ÒÇÂuö×5÷6Å¶•Ò’bÒ·BæWuFW‡GW&Uf–Wuv—F…—†VÄf÷&ÖC§bFW‡GW&UG—S¤ÕDÅFW‡GW&UG—S$BÆWfVÇ3¤å4Ö¶U&ævR†uö×5öÇe¶•ÒÂ’6Æ–6W3¤å4Ö¶U&ævR†uö×5÷6Å¶•ÒÂ•Ó°¢–b‚b’6öçF–çVS°¢V–çC3%÷B¦òÒ‡V–çC3%÷B¢’‚‡V–çC…÷B¢—2Óæ"æ6öçFVçG2²2ÓçW6VB“²õ³ÒÒs²õ³ÒÒƒ²õ³%ÒÒGs²õ³5ÒÒ°¢×5öVçB¦RÒg2ÓæU·2Óæâ²µÓ²RÓç75ö–G‚Òu÷7FE÷72Òuö×5öf—'7C²RÓç6W"Òuö×5ö6W%¶•Ó²RÓæGBÒ“²RÓçF—7BÒF—7C²RÓæf×BÒ†–çB—c²RÓçrÒs²RÓæ‚Òƒ²RÓæöfbÒ2ÓçW6VC²2ÓçW6VB³ÒæVVC°¢–b†uö&Æ—B’²Væ5ö%öVæB†uö&Æ—B“²uö&Æ—BÒæ–Ã²Ð¢–CÄÕDÃD6ö×WFT6öÖÖæDVæ6öFW#â6RÒ†–CÄÕDÃD6ö×WFT6öÖÖæDVæ6öFW#â–Væ5ö5ö&Vv–â„åTÄÂ“²¶6R6WD&wVÖVçEF&ÆS¦EÓ°¢¶B6WEFW‡GW&S§bæwU&W6÷W&6T”BD–æFWƒ£Ó²¶B6WDFG&W73§2Óæ"æwTFG&W72²RÓæöfbD–æFWƒ£Ó°¢Væ5ö5÷6ò†6RÂ2“²Væ5ö5öF—7F6‚†6RÂÕDÅ6—¦TÖ¶R‡GrÂF‚Â’ÂÕDÅ6—¦TÖ¶RƒbÂbÂ’“²Væ5ö5öVæB†6R“°¢&w5öF—'G•öÆÂ‚“²2ÓæfVæ6RÒuöfVæ6R²°¢–b‡bÒB’FVfW%öö&¥÷Vç&Vb‡b“°¢Ð§Ð¢òò6ÖR×72fVVF&6²6×Æ–æs¢FW‡GW&Rf–WrF†—2G&rw2g&vÖVçB7FvR6×ÆW2—2W†7FÇ’F†RÆWfVÂæBÆ–W"öb6öÆ÷"GF6†ÖVçBöbF†R7W'&VçB&VæFW"72ÂæBF†R72w&—FW2F†BGF6†ÖVç@¢òò†vÂw2÷7BÖVffV7B&öw&Ò–â7WW"Ö&–ò4Bv÷&ÆB²&÷w6W"w2gW'’Â6F–âFöBæB7WW"Ö&–òöG—76W’w&—FW2F†RFöæRÖÖVB$t#"–ÖvR2âGF6†ÖVçBv†–ÆR6×Æ–ær—B’âöâF†R7v—F6‚F†R&V@¢òò&WGW&ç2F†R6öçFVçBg&öÒ&Vf÷&RF†—2G&s²öâÖWFÂÂw&—F–æræB6×Æ–ærF†R6ÖRGF6†ÖVçBv—F†–âöæR&VæFW"72—2VæFVf–æVC¢F†RÓB&ò†Vç2Fò&VB6÷'&V7FÇ’Âv†–ÆRF†R#&òæBF†RÆRÓ`¢òòæ÷ræBF†Vâ&VBv†öÆRgƒb&Æö6·22ƒÃÃ’ÂƒÃÃ’÷"ƒÃÃ’†&Æö6·2öbF†R6ö×&W76VBGF6†ÖVçB6Vv‡Bv†–ÆRæ÷F†W"F–ÆR7F÷&W2F†VÒ’à¢òòF†Rf—ƒ¢&Vf÷&R7V6‚G&r÷Vç2—G2&VæFW"72ÂVæBF†R7W'&VçB&VæFW"72‡F†RGF6†ÖVçBÆæG2–âÖVÖ÷'’’Â6÷’F†BÆWfVÂæBÆ–W"–çFò6æ6†÷BFW‡GW&R–âF†R&WÆ’7G&VÒÂÖ¶RF†—2G&p¢òò6×ÆRF†R6æ6†÷C²ÆFW"G&w2&V÷VâF†R&VæFW"722W7VÂ†ÆöF–ærF†RGF6†ÖVçB’âöæÇ’6–ævÆRÖÆWfVÂÂ6–ævÆRÖÆ–W"$Bf–Ww2F†B&RW†7FÇ’F†RGF6†ÖVçBw2ÆWfVÂæBÆ–W"&R†æFÆVC²f–Ww0¢òò÷fW"6WfW&ÂÆWfVÇ2÷"Æ–W'2†Ö—Ö6†–â76W2’&RÆVgB2F†W’&RÂ÷F†W"6†W2&R&W÷'FVBöæ6Rà¢òòv†–6‚G&w3¢F†÷6RF†Bw&—FRF†R6×ÆVBGF6†ÖVçB†6†ææVÂÖ6²æBg&vÖVçB÷WGWB’÷"föÆÆ÷rG&röbF†R6ÖRÖWFÂ72F†Bw&÷FR—C²G&rF†B6×ÆW2âGF6†ÖVçBæ÷F†–ær–â—G272†0¢òòw&—GFVâ&VG2—BF—&V7FÇ’ÂæBÆFW"G&röbF†B72F†Bw&—FW2—B7F'G2æWr72†öâF–ÆVBuRF†RV&Æ–W"&VG2–â÷F†W"F–ÆW26÷VÆB6VRF†W6Rw&—FW2’âv†–6‚6æ6†÷C¢ådâÖ¶W2&VæFW&–æp¢òòf—6–&ÆRFòFW‡GW&R&VG2öæÇ’F‡&÷Vv‚çfä6öÖÖæD'VffW$&'&–W"Â6ò6æ6†÷BF¶Vâ6–æ6RF†RvÖRw2Æ7B&'&–W"†æB–âF†—27V&Ö—76–öâ’†öÆG2WfW'—F†–ærF†RvÖRÖ’W‡V7BFò&VC²ÆFW"7V6€¢òòG&w2&WW6R—Bv—F†÷WBVæF–ærF†R72ÂVæÆW72F†R72Ç6ò&VBF†BGF6†ÖVçBF—&V7FÇ’â'&VF‚öbF†Rv–ÆBw2rÖ'VffW"FV6Ç2w&—FRæB6×ÆRF†V—"÷vâF&vWG2Â&÷WBCcG&w2g&ÖR÷WFFö÷'3 ¢òòg&W6‚6æ6†÷Bf÷"V6‚Föö²F†RÖ2w2uRF–ÖRg&öÒ&÷WBBFò#‚×2g&ÖRÂ&WW6RVçF–Â&'&–W"æVVG2&÷WBFVâg&ÖR„uRF–ÖRv—F†–âF†Ræö—6RöbæöæR’à¢òòvÖR&'&–W"&WGvVVâ7V6‚G&w2Ö¶W2F†RæW‡BöæRF¶Rg&W6‚6æ6†÷BÂæBFV'2öbF†R¶–ævFöÒw26‡&–æW26WBöæR&Vf÷&RV6‚FV6Âƒƒ"Â&÷WB#c2g&ÖR“¢&÷WB#ƒ@¢òò6æ6†÷G2g&ÖRÂV6‚72VæBæBgVÆÂ×6—¦R6÷–W2„v—D‡V"3s3r’âG&rF†B6×ÆW2F†RGF6†ÖVçBB—G2÷vâ—†VÂ&VG2F†RF–ÆR–ç7FVB†f%÷F–ÆU÷Æâ“¢æò72VæBÂæò6÷’à¢òòDÕôådåôd%õ4äÓæöæRÂÆÂg&W6‚6æ6†÷Bf÷"WfW'’G&r6×Æ–ærâGF6†ÖVçBöb—G272†F–væ÷7F–2“²DÕôådåôd%õD”ÄSÓæòF–ÆR&VG0§7FF–2–çBuöf%öÖöFRÒÓ²òòæöæRÂw&—FW'2v—F‚&WW6RVçF–Â&'&–W"Â"WfW'’6×Æ–ærG&rÂg&W6€§7FF–2V–çCcE÷Buöf%ö&"Âuöf%÷&WW6RÂuöf%÷7Æ—C²òòvÖR&'&–W'26òf"„õô$%$”U"v—F‚&—G2“²G&w2F†B&WW6VB6æ6†÷C²76W2VæFVB&Vf÷&Rw&—FRFòâGF6†ÖVçBâV&Æ–W"G&r&V@§7FF–2V–çCcE÷Buöf%÷72ÒT”åCcEôÔƒ²7FF–2V–çC3%÷Buöf%÷vÖ6²Âuöf%÷&Ö6²Âuöf%öG&VC²òòF†R7W'&VçBÖWFÂ&VæFW"73¢6öÆ÷"GF6†ÖVçG2w&—GFVâ6òf"Â6×ÆVBF—&V7FÇ“²G&r†2&VB—G2FWF‚GF6†ÖVçB†÷"6÷VÆB†fS¢&–æFÆW72§G—VFVb7G'V7B²V–çCcE÷B6W"Â&"Â7V"ÂW6S²–çBÇbÂ6Ã²õ÷7G&öær–CÄÕDÅFW‡GW&Sâ6æ²Òf%ö66†U÷C²òòW6S¢v†Vâ—Bv2Æ7BF¶Vâ÷"&WW6VB†uöf&5÷W6R§7FF–2f%ö66†U÷Buöf&5³…Ó²7FF–2V–çCcE÷Buöf&5÷W6S°§7FF–2–çBf%öÖöFR‡fö–B’°¢–b†uöf%öÖöFRÂ’²6öç7B6†"¦RÒvWFVçb‚$DÕôådåôd%õ4ä"“²uöf%öÖöFRÒRbb7G&6×†RÂ#"’ò¢Rbb7G&6×†RÂ&ÆÂ"’ò"¢°¢–b†uöf%öÖöFRÒ’ÔÄôr‚'6ÖR×72fVVF&6²6×Æ–ær6æ6†÷C¢W2„DÕôådåôd%õ4ä’"Âuöf%öÖöFRò&g&W6‚öæRf÷"WfW'’G&r6×Æ–ærâGF6†ÖVçBöb—G272"¢&öfb"“²Ð¢&WGW&âuöf%öÖöFS°§Ð§7FF–2–æÆ–æRfö–Bf%÷75÷7–æ2‡fö–B’²V–çCcE÷BÒu÷7FE÷72²uöVæ3²–b‡Òuöf%÷72’²uöf%÷72Ò²uöf%÷vÖ6²Òuöf%÷&Ö6²Òuöf%öG&VBÒ²ÒÐ§7FF–2–æÆ–æR–çBf%÷6†R†6öç7Bf%ö66†U÷B¦RÂÕDÅ—†VÄf÷&ÖBbÂå5T–çFVvW"rÂå5T–çFVvW"‚’²&WGW&âRÓç6æbbRÓç6æç—†VÄf÷&ÖBÓÒbbbRÓç6æçv–GF‚ÓÒrbbRÓç6ææ†V–v‡BÓÒƒ²Ð§7FF–2–æÆ–æR–çBf%öÆ—fR†6öç7Bf%ö66†U÷B¦R’²&WGW&âRÓç6W"bbRÓæ&"ÓÒuöf%ö&"bbRÓç7V"ÓÒu÷7V%÷6W&–Ã²ÒòòÆFW"G&rÖ’7F–ÆÂ&WW6R—@¢òòF†R6æ6†÷BöbâGF6†ÖVçBw2ÆWfVÂæBÆ–W#¢F†R66†VBöæR‡fÆ–C¢¦g&W6‚Ò’÷"FW‡GW&RFò6÷’–çFò‚¦g&W6‚Ò²F†RVçG'’—2F†VâfÆ–Bg&öÒF†—2&'&–W"æB7V&Ö—76–öâöâ’à¢òòæWr¶W’F¶W2âVçG'’v†÷6RFW‡GW&RÇ&VG’†2—G2f÷&ÖBæB6—¦RÂöæRæòÆFW"G&r6â&WW6Rf—'7BÂVÇ6RF†RÆV7B&V6VçFÇ’W6VBâF¶–ærF†RVçG&–W2–âGW&âÖFRFW‡GW&Rf÷ ¢òòæV&Ç’WfW'’6æ6†÷Bv†Vâg&ÖR6æ6†÷G26WfW&Â6—¦W3¢FV'2öbF†R¶–ævFöÒw2BÓRg&ÖRƒcƒƒ“bF÷vâFòCƒ##B’V6‚ÖFRöæRÂBÓbt"öbFW‡GW&W2WfW'’3g&ÖW2öâF†RÖ0¢òòæB&W6–FVæ7’6öÖÖ—Bf÷"V6‚FW‡GW&RFFVBæBV6‚&VÖ÷fVB„v—D‡V"3“3§7FF–2f%ö66†U÷B¦f%÷6æ‡V–çCcE÷B6W"Â–çBÇbÂ–çB6ÂÂÕDÅ—†VÄf÷&ÖBbÂå5T–çFVvW"rÂå5T–çFVvW"‚Â–çBf÷&6RÂ–çB¦g&W6‚’°¢f%ö66†U÷B¦RÒåTÄÃ°¢f÷"†–çB’Ò²’Â‚bbS²’²²’–b†uöf&5¶•Òç6æbbuöf&5¶•Òç6W"ÓÒ6W"bbuöf&5¶•ÒæÇbÓÒÇbbbuöf&5¶•Òç6ÂÓÒ6Â’RÒfuöf&5¶•Ó°¢–b†Rbbf%÷6†R†RÂbÂrÂ‚’’²RÓç6W"Ò²RÒåTÄÃ²ÒòòF†RGF6†ÖVçBv2ÖFRv–â–âæ÷F†W"f÷&ÖB÷"6—¦S¢—G26æ6†÷B—2æòÆöævW"F†—2¶W’w0¢–b†Rbbf÷&6Rbbf%öÆ—fR†R’’²RÓçW6RÒ²¶uöf&5÷W6S²¦g&W6‚Ò²&WGW&âS²Ð¢–b‚R’²òò'’&æ³¢6ÖR6†RæBæ÷BÆ—fRÂæ÷BÆ—fRÂ6ÖR6†R†WfW'’VçG'’Æ—fR’ÂÆV7B&V6VçFÇ’W6VBv—F†–â&æ°¢–çB&W7BÒÃÂ3°¢f÷"†–çB’Ò²’Âƒ²’²²’²f%ö66†U÷B¦2Òfuöf&5¶•Ó²–çB"Òf%öÆ—fR†2’ò†f%÷6†R†2ÂbÂrÂ‚’ò"¢2’¢f%÷6†R†2ÂbÂrÂ‚’ò¢°¢–b‡"Â&W7BÇÂ‡"ÓÒ&W7Bbb2ÓçW6RÂRÓçW6R’’²&W7BÒ#²RÒ3²ÒÐ¢–b†RÓç6æbbf%÷6†R†RÂbÂrÂ‚’’²FVfW%öö&¥÷&VÆV6R†RÓç6æ“²RÓç6æÒæ–Ã²Ð¢Ð¢–b‚RÓç6æ’²ÕDÅFW‡GW&TFW67&—F÷"¦BÒ´ÕDÅFW‡GW&TFW67&—F÷"FW‡GW&S$DFW67&—F÷%v—F…—†VÄf÷&ÖC§bv–GFƒ§r†V–v‡C¦‚Ö—ÖVC¤äõÓ°¢BçW6vRÒÕDÅFW‡GW&UW6vU6†FW%&VBÂÕDÅFW‡GW&UW6vU—†VÄf÷&ÖEf–Ws²Bç7F÷&vTÖöFRÒÕDÅ7F÷&vTÖöFU&—fFS°¢RÓç6æÒ¶uöFWbæWuFW‡GW&Uv—F„FW67&—F÷#¦EÓ²–b‚RÓç6æ’²RÓç6W"Ò²&WGW&âåTÄÃ²Ò&W5öFB†RÓç6æ“²ÒòògVÆÂVWVR&'&–W'26W&FR6÷’g&öÒF†RG&w2F†B&VBF†RV&Æ–W"6öçFVç@¢RÓç6W"Ò6W#²RÓæÇbÒÇc²RÓç6ÂÒ6Ã²RÓæ&"Òuöf%ö&#²RÓç7V"Òu÷7V%÷6W&–Ã²RÓçW6RÒ²¶uöf&5÷W6S²¦g&W6‚Ò²&WGW&âS°§Ð¢òòÒÒÒÒ6ÖR×72fVVF&6²&VBg&öÒF†RF–ÆRÒÒÒÐ¢òòG&rv†÷6RWfW'’6×ÆRöbâGF6†ÖVçBöb—G2÷Vâ72—2&ö¦V7F—fR6×ÆRB—G2fW'FW‚6†FW"w26Æ—(i"67&VVâf'––ær‡67%÷&ööb’&VG2F†—2—†VÂöbF†RGF6†ÖVçC ¢òòöâÆRw2uW2F†R&VF&ÆR×F–ÆRfW'6–öâƒÆ¶W“åö&bÂF×6…öfWF6‚ç’’&VG2—Bg&öÒF†RF–ÆR…µ¶6öÆ÷"†²•ÕÒ’Âv†–6‚†öÆG2WfW'’V&Æ–W"g&vÖVçBöbF†R72–â÷&FW"âF†B—2v†Bådà¢òòwV&çFVW27V6‚G&rgFW"&'&–W"†æBöæRöbF†RfÇVW2—BÆVfW2VæFVf–æVBv—F†÷WBöæR’Â6òF†RG&ræVVG2æò6æ6†÷BæBF†R72vöW2öââ'&VF‚öbF†Rv–ÆBw2æBFV'0¢òòöbF†R¶–ævFöÒw2rÖ'VffW"FV6Ç2ƒFC6CVCC†3&S†C’ÂvC#63#cfc†c"’6×ÆRF†V—"F&vWG26òâöæÇ’–ââ÷Vâ72†—G2f÷&ÖG2æÖRF†R—VÆ–æR’ÂöæÇ’6–ævÆRÖÆWfVÂ$Bf–Wp¢òòF†B—2F†RGF6†ÖVçBw2ÆWfVÂæBÆ–W"–â—G2÷vâf÷&ÖBv—F‚æò6†ææVÂÖ–ærÂæBæ÷Bv—F‚F†RÇ†FW7BÂÆöv–2÷W&F–öâÂ×VÇF—6×Æ–ærÂvVöÖWG'’÷"6Æ–6RÖÖW&vVBG&r÷ ¢òòF†RD$E"÷F–Ö—¦F–öç2‡F†V—"ÖW&vVB76W2ÖGF6†ÖVçG2F†V×6VÇfW2’âF†R—VÆ–æR—2'V–ÇB–âF†R&6¶w&÷VæBöâf—'7BW6RÂ26Õö×övWBFöW3¢VçF–Â—B—2F†W&RF†RG&p¢òòF¶W26æ6†÷BÂ6òæòg&ÖRv—G2f÷"F†R6ö×–ÆW"âÆÖ—2F†R–FVçF—G’÷fW"F†R6†FW"w2÷WGWG2†âGF6†ÖVçB—BFöW2æ÷Bw&—FR¶VW2—G26öçFVçBÂ2–â6õöFW62§7FF–2–çBf%÷F–ÆUööâ‡fö–B’²7FF–2–çBbÒÓ²–b‡bÂ’²6öç7B6†"¦RÒvWFVçb‚$DÕôådåôd%õD”ÄR"“²bÒ†Rbb7G&6×†RÂ#"’“²–b‚b’ÔÄôr‚'6ÖR×72fVVF&6²6×Æ–æs¢æòF–ÆR&VG2„DÕôådåôd%õD”ÄSÓ’"“²Ò&WGW&âc²Ð§7FF–2–çBvVõöG&v–ær‡fö–B“²òòG&röbF†RvVöÖWG'’F‚†çfåövVöÖWG'’æ–æ2’Âv†÷6R—VÆ–æR—2—G2÷và§7FF–2V–çCcE÷Buöf%÷F–ÆUöG&w2Âuöf%÷F–ÆU÷v—BÂuöf%÷G6W²7FF–2–CÄÕDÄgVæ7F–öãâuöf%÷Ffã²7FF–2V–çC…÷Buöf%÷FÆÖ³…Ó²7FF–26õög¥öFW62uöf%÷FFW63°¦VçVÒ²d%õEôÔ‚Ò#SbÓ°§7FF–27G'V7B²6õö¶W’³²–CÄÕDÅ&VæFW%—VÆ–æU7FFSâ3²–çBFöæS²Òuöf%÷G´d%õEôÔ…Ó²7FF–2–çBuöf%÷Göã²7FF–2÷5÷Væf—%öÆö6²uöf%÷GöÆö6²Òõ5õTäd•%ôÄô4µô”ä•C°§7FF–2–çBf%÷F–ÆU÷&VG’„çe6†FW"§g2Âçe6†FW"¦g2Â–CÄÕDÄgVæ7F–öãâfâÂ6öç7BV–çC…÷BÆÖ³…ÒÂ6öç7B6õög¥öFW62¦g¢’°¢6öç7BV–çC…÷B¦ÃÒu÷6õöÆÖ²u÷6õöÆÖÒÆÖ²òòF†R¶W’6õövWBÖ¶W2f÷"F†—2G&r†FW7Eö¶W’æBÆ÷ö¶W’&VBF†RÖ–ær¢6õö¶W’³²6õö¶W•öf–ÆÂ‚f²Âg2Âg2Âu÷72æ6f×BÂu÷72æFf×BÂu÷72ç6f×BÂu÷72ç6×ÆW2Â“²u÷6õöÆÖÒÃ°¢ÖVÖ7’†²æÆÖÂÆÖÂ‚“²²æg6fâÒ…õö'&–FvRfö–B¢–fã°¢–b††ÕövWEö–B‚fu÷6òÂf²’’&WGW&â°¢÷5÷Væf—%öÆö6µöÆö6²‚fuöf%÷GöÆö6²“²–çB’Ò²f÷"ƒ²’Âuöf%÷Göã²’²²’–b‚ÖVÖ6×‚fuöf%÷G¶•Òæ²Âf²Â6—¦Vöb²’’'&V³²òòäôÄ”åB†'Vw&öæR×7W7–6–÷W2ÖÖVÖ÷'’Ö6ö×&—6öâ“¢6õö¶W•öf–ÆÂ¦W&öW2F†Rv†öÆR¶W’f—'7BÂFF–ær–æ6ÇVFVB†2f÷"6Õö×övWBw2F&ÆR¢–b†’Âuöf%÷Göâ’²–B2Òuöf%÷G¶•ÒæFöæRòuöf%÷G¶•Òç2¢æ–Ã²÷5÷Væf—%öÆö6µ÷VæÆö6²‚fuöf%÷GöÆö6²“°¢–b‚2’&WGW&â²òò'V–ÆF–ærÂ÷"f–ÆVB†—G26æ6†÷B7F—2¢†Õ÷WEö–B‚fu÷6òÂf²Â2“°¢6õ÷6¶W’6³²6†"6†W…·6—¦Vöb6²¢"²Ó²6õ÷6¶W•ög&öÒ‚f²Âg2Âg2ÂÂg¢Âg6²“²6õ÷6¶W•ö†W‚‚g6²Â6†W‚“²6õ÷&V6÷&B‚%%ÇBW5Æâ"Â6†W‚“²6õ÷W6W%÷&V6÷&B‚%%ÇBW5Æâ"Â6†W‚“²òò&Wv&ÖVBæW‡BF–ÖR†¶–æB¢&eöfâ¢&WGW&â²Ð¢–b†uöf%÷GöâãÒd%õEôÔ‚’²÷5÷Væf—%öÆö6µ÷VæÆö6²‚fuöf%÷GöÆö6²“²7FF–2–çBöæ6S²–b‚öæ6R²²’ÔÄôr‚'6ÖR×72fVVF&6²6×Æ–æs¢VBF–ÆR×&VF–ær—VÆ–æW2'V–ÇC²ÆFW"öæW2¶VWF†R6æ6†÷B"Âd%õEôÔ‚“²&WGW&â²Ð¢uöf%÷G¶•Òæ²Ò³²uöf%÷G¶•Òç2Òæ–Ã²uöf%÷G¶•ÒæFöæRÒ²uöf%÷Göâ²³²÷5÷Væf—%öÆö6µ÷VæÆö6²‚fuöf%÷GöÆö6²“°¢ÕDÅ&VæFW%—VÆ–æTFW67&—F÷"¦BÒ6õöFW62‚f²Âg2Âg2ÂfâÂ“°¢F—7F6…ö7–æ2†F—7F6…övWEövÆö&Å÷VWVR…õ5ô4Ä55õUD”Ä•E’Â’Âç²å4W'&÷"¦RÒæ–Ã²–CÄÕDÅ&VæFW%—VÆ–æU7FFSâ2Ò6õö'V–ÆB†BÂfR“°¢–b‚2’ÔÄôr‚%µ$TEÒ6ÖR×72fVVF&6²6×Æ–æs¢æòF–ÆR×&VF–ær—VÆ–æRf÷"fW'FW‚W2g&vÖVçBW3¢W2†—G2G&w2¶VWF†R6æ6†÷B’"Âg2Óæ¶W’Âg2Óæ¶W’ÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“°¢÷5÷Væf—%öÆö6µöÆö6²‚fuöf%÷GöÆö6²“²uöf%÷G¶•Òç2Ò3²uöf%÷G¶•ÒæFöæRÒ²÷5÷Væf—%öÆö6µ÷VæÆö6²‚fuöf%÷GöÆö6²“²Ò“°¢&WGW&â°§Ð¢òòF†RFW‡GW&R6Æ÷G2†&—BW"6Æ÷BÓ#SR’v†÷6RGF6†ÖVçBF†—2G&r&VG2g&öÒF†RF–ÆS²ÒæöæR‡F†Vâuöf%÷FfâæBF†R&W7B&Ræ÷BF†—2G&rw2§7FF–2–çBf%÷F–ÆU÷Æâ„çe&ör§"ÂV–çC3%÷BvÒÂV–çCcE÷BF–ÆU³EÒ’°¢çe6†FW"§g2Ò"Óç6…³ÒÂ¦g2Ò"Óç6…³Ó²ÖV×6WB‡F–ÆRÂÂB¢6—¦Vöb§F–ÆR“°¢–b†uöf%öÖöFRÒÇÂuöVæ2ÇÂg2ÇÂg2Óç&bæf5÷&ö¢ÇÂg2Óç&bæ6öÆ÷%ö¶æ÷vâÇÂg2Óç&bææfòÇÂf%÷F–ÆUööâ‚’ÇÂF&G%ööâ‚’’&WGW&â°¢–b†FW7Eööâ‚’ÇÂÆ÷ööâ‚’ÇÂu÷72ç6×ÆW2âÇÂ2ç&7EöF—66&BÇÂu÷6Õ÷72ÇÂvVõöG&v–ær‚’ÇÂ×5ög5ö÷fW'&–FR†g2’’&WGW&â°¢å5T–çFVvW"rÒu÷72çGrÂ‚Òu÷72çFƒ°¢–çBg…³3%Ó²f÷"†–çB’Ò²’Â3#²’²²’g…¶•ÒÒÓ²V–çC…÷Bf&Ö6²Ò°¢f÷"†–çB’Ò²’Âg2Óç&bææ#²’²²’²6öç7Bçfå÷&&–æB¦"Òfg2Óç&bæ%¶•Ó²–b†"Óæ¶–æBÒådåõ$%õDU‚ÇÂ"Óç&VrãÒåDU‚’6öçF–çVS°¢–çBF–BÒ†–çB’‡FW…÷6Æ÷Eö‚…2çFW…³ÒÂ2æ–Öu³ÒÂ"Óç&Vr’b†ffffb“²–b‚F–B’6öçF–çVS°¢÷5÷Væf—%öÆö6µöÆö6²‚fu÷&×R“²V–çCcE÷B6W"ÒF–BÂuöçG&Vrbbu÷G&Vu·F–EÒæö²òu÷G&Vu·F–EÒçBç6W&–Â¢²fö–B§GbÒåTÄÃ²–b‡6W"’FW…öf7E÷&–B‡F–BÂgGb“²–CÄÕDÅFW‡GW&SâBÒ…õö'&–FvR–CÄÕDÅFW‡GW&Sâ—Gc²÷5÷Væf—%öÆö6µ÷VæÆö6²‚fu÷&×R“°¢–b‚6W"ÇÂB’6öçF–çVS°¢f÷"†–çB²Ò²²Â"ç'Bæâbb²Âƒ²²²²’²–b‚"ç'Bæ6öÆ÷%¶µÒÇÂ"ç'Bæ6öÆ÷%¶µÒÓç6W&–ÂÒ6W"ÇÂ‚‡vÒÂuöf%÷vÖ6²’ãâ²b’’6öçF–çVS²òòæ÷F†–ær–âF†—272w&÷FR—C¢&VBF—&V7FÇ’Â2&Vf÷&P¢V–çCcE÷B6W#²–çBÇbÂ6Ã²'E÷7V"‚e"ç'BÂ²Âf6W"ÂfÇbÂg6Â“°¢–CÄÕDÅFW‡GW&Sâ&6RÒBç&VçEFW‡GW&RòBç&VçEFW‡GW&R¢C²ÕDÅFW‡GW&U7v—§¦ÆT6†ææVÇ27rÒBç7v—§¦ÆS°¢–b‡BçFW‡GW&UG—RÒÕDÅFW‡GW&UG—S$BÇÂBæÖ—ÖÆWfVÄ6÷VçBÒÇÂBç6×ÆT6÷VçBÒÇÂ&6Rç&VçEFW‡GW&RÇÂ‡Bç&VçEFW‡GW&RòBç&VçE&VÆF—fTÆWfVÂ¢’Ò„å5T–çFVvW"–Ç`¢ÇÂ‡Bç&VçEFW‡GW&RòBç&VçE&VÆF—fU6Æ–6R¢’Ò„å5T–çFVvW"—6ÂÇÂBç—†VÄf÷&ÖBÒ„ÕDÅ—†VÄf÷&ÖB–u÷72æ6f×E¶µÒÇÂ"ÓæGGBÒÕDÅFW‡GW&UG—S$@¢ÇÂ7rç&VBÒÕDÅFW‡GW&U7v—§¦ÆU&VBÇÂ7ræw&VVâÒÕDÅFW‡GW&U7v—§¦ÆTw&VVâÇÂ7ræ&ÇVRÒÕDÅFW‡GW&U7v—§¦ÆT&ÇVRÇÂ7ræÇ†ÒÕDÅFW‡GW&U7v—§¦ÆTÇ†¢ÇÂÔ‚†&6Rçv–GF‚ãâÇbÂVÂ’ÒrÇÂÔ‚†&6Ræ†V–v‡BãâÇbÂVÂ’Ò‚ÇÂ†g2Óç&bæf5öÖ—†VE¶"Óç&VrãâeÒãâ†"Óç&Vrbc2’b’’'&V³°¢–çBç’ÒÂö²Ò°¢f÷"†–çB2Ò²2Âg2Óç&bææf3²2²²’–b†g2Óç&bæf5÷&Vu¶5ÒÓÒ"Óç&Vr’²–b†g2Óç&bæf5÷&ö¢ãâ2bbb67%÷&ööb‡"Âg2Óç&bæf5öÆö6å¶5ÒÂrÂ‚’’ç’Ò²VÇ6Rö²Ò²Ð¢–b‚ö²ÇÂç’’'&V³°¢f÷"†–çB2Ò²2Âg2Óç&bææf3²2²²’–b†g2Óç&bæf5÷&Vu¶5ÒÓÒ"Óç&Vr’g…¶5ÒÒ³°¢f&Ö6²ÃÒ‡V–çC…÷B’ƒRÃÂ²“²F–ÆU¶"Óç&VrãâeÒÃÒVÆÂÃÂ†"Óç&Vrbc2“°¢'&V³²ÒÐ¢–b‚f&Ö6²’&WGW&â°¢V–çC…÷BÆÖ³…Ó²ÖV×6WB†ÆÖÂ†fbÂ‚“²f÷"†–çBÒ²Âƒ²²²’–b†g2Óç&bæ6öÆ÷%öÖ6²ãâb’ÆÖ·ÒÒ‡V–çC…÷B—°¢6õög¥öFW62C²ÖV×6WB‚fBÂÂ6—¦VöbB“²Bæ¶–æBÒ²Bæf&Ö6²Òf&Ö6³²f÷"†–çB’Ò²’Â3#²’²²’Bæg…¶•ÒÒ†–çC…÷B–g…¶•Ó°¢–CÄÕDÄgVæ7F–öãâfâÒ&eöfâ†g2Âg‚Âf&Ö6²ÂÆÖ“°¢–b‚fâÇÂf%÷F–ÆU÷&VG’‡g2Âg2ÂfâÂÆÖÂfB’’²uöf%÷F–ÆU÷v—B³ÒfâÒæ–Ã²ÖV×6WB‡F–ÆRÂÂB¢6—¦Vöb§F–ÆR“²&WGW&â²Ð¢uöf%÷FfâÒfã²ÖVÖ7’†uöf%÷FÆÖÂÆÖÂ‚“²uöf%÷FFW62ÒC°¢&WGW&â°§Ð§fö–B×FÅöf%÷7FG2‡V–çCcE÷B§6æ2ÂV–çCcE÷B§&WW6VBÂV–çCcE÷B§F–ÆRÂV–çCcE÷B§v—B’²§6æ2Òuöf%ö6÷VçC²§&WW6VBÒuöf%÷&WW6S²§F–ÆRÒuöf%÷F–ÆUöG&w3²§v—BÒuöf%÷F–ÆU÷v—C²ÒòòF†R÷vW"Æ–æRw2fVVF&6²6÷VçG2††ÆUöçfâæ2¢òò6ÆÆVB&Vf÷&RF†RG&r÷Vç2—G2&VæFW"73¢–b6æ6†÷B—2æVVFVBÂVæBF†R7W'&VçB&VæFW"72Â6÷’F†R6æ6†÷B†÷"&WW6RfÆ–BöæR’æB&Vv—7FW"F†R7V'7F—GWF–öâ‡W6VB'’F†—2G&rw2&wVÖVçBVæ6öF–ær§7FF–2fö–Bf%÷&W&R„çe&ör§"’°¢uöG&u÷6W²³²uöf%öâÒ°¢–b†uöf%÷&V&–æB’²&w5öF—'G•öÆÂ‚“²uöf%÷&V&–æBÒ²ÒòòF†R&Wf–÷W2G&r†B7V'7F—GWFVB&–æF–æs¢&RÖVæ6öFRv—F‚F†R÷&–v–æÂ&–æF–æw2F†—2F–ÖP¢–b‚"ÇÂ"Óç6…³ÒÇÂ"ç'BæâÇÂuög¢æ7F—fRÇÂuöFVfW%÷&WÆ––ærÇÂf%öÖöFR‚’’&WGW&ã°¢çe6†FW"¦g2Ò"Óç6…³Ó°¢f%÷75÷7–æ2‚“°¢V–çC3%÷BvÒÒ²òòF†RGF6†ÖVçG2F†—2G&rw&—FW0¢f÷"†–çB²Ò²²Â"ç'Bæâbb²Âƒ²²²²’–b…"ç'Bæ6öÆ÷%¶µÒbb‚…2æ6†Ö6²æÖ6²ãâƒB¢²’’b†b’bb†g2Óç&bæ6öÆ÷%ö¶æ÷vâòg2Óç&bæ6öÆ÷%öÖ6²ãâ²b¢’’vÒÃÒRÃÂ³°¢–b†uöVæ2bb‡vÒbuöf%÷&Ö6²’’²uöf%÷7Æ—B²³²'öVæB‚“²f%÷75÷7–æ2‚“²ÒòòâV&Æ–W"G&röbF†—272&VBâGF6†ÖVçBF†—2öæRw&—FW3¢—B×W7Bæ÷B6VRF†W6Rw&—FW0¢–çBG7rÒ"ç'BæFWF‚bb‚…2æG2æGBbb2æG2æGr’ÇÂ…2æG2ç7Bbb…2ç7E÷vÖ6µ³ÒÇÂ2ç7E÷vÖ6µ³Ò’bb…2æG2æf–ÅöbÒådåõ5DTä4”Åôõô´TUÇÂ2æG2ç¦f–ÅöbÒådåõ5DTä4”Åôõô´TUÇÂ2æG2ç§75öbÒådåõ5DTä4”Åôõô´TU ¢ÇÂ2æG2æf–Åö"Òådåõ5DTä4”Åôõô´TUÇÂ2æG2ç¦f–Åö"Òådåõ5DTä4”Åôõô´TUÇÂ2æG2ç§75ö"Òådåõ5DTä4”Åôõô´TU’’“°¢–b†uöVæ2bbG7rbbuöf%öG&VB’²uöf%÷7Æ—B²³²'öVæB‚“²f%÷75÷7–æ2‚“²ÒòòF†R6ÖRf÷"—G2FWF‚æB7FVæ6–Â‡6VR'F5ööâ¢–b…"ç'BæFWF‚bb†g2Óç&bæ&–æFÆW72ÇÂg2Óç&bæG–â’’uöf%öG&VBÒ°¢V–çCcE÷BF–ÆU³EÓ²–çBçF–ÆRÒf%÷F–ÆU÷Æâ‡"ÂvÒÂF–ÆR“°¢f÷"†–çB’Ò²’Âg2Óç&bææ"bbuöf%öâÂC²’²²’²6öç7Bçfå÷&&–æB¦"Òfg2Óç&bæ%¶•Ó²–b†"Óæ¶–æBÒådåõ$%õDU‚ÇÂ"Óç&VrãÒåDU‚’6öçF–çVS°¢–çBF–BÒ†–çB’‡FW…÷6Æ÷Eö‚…2çFW…³ÒÂ2æ–Öu³ÒÂ"Óç&Vr’b†ffffb“²–b‚F–B’6öçF–çVS°¢÷5÷Væf—%öÆö6µöÆö6²‚fu÷&×R“²V–çCcE÷B6W"ÒF–BÂuöçG&Vrbbu÷G&Vu·F–EÒæö²òu÷G&Vu·F–EÒçBç6W&–Â¢²÷5÷Væf—%öÆö6µ÷VæÆö6²‚fu÷&×R“°¢–b‚6W"’6öçF–çVS°¢–b…"ç'BæFWF‚bb"ç'BæFWF‚Óç6W&–ÂÓÒ6W"’uöf%öG&VBÒ°¢f÷"†–çB²Ò²²Â"ç'Bæâbb²Â‚bbuöf%öâÂC²²²²’²–b‚"ç'Bæ6öÆ÷%¶µÒÇÂ"ç'Bæ6öÆ÷%¶µÒÓç6W&–ÂÒ6W"’6öçF–çVS°¢–b‡F–ÆU¶"Óç&VrãâeÒãâ†"Óç&Vrbc2’b’²B‚$âVÆÇR6ÖR×72fVVF&6²6×Æ–æs¢6Æ÷BVB&VG2GF6†ÖVçBVB6W#ÒVÆÇRg&öÒF†RF–ÆR"Âu÷7FE÷72²uöVæ2Â"Óç&VrÂ²Â‡Vç6–væVBÆöærÆöær—6W"“²6öçF–çVS²Òòòf%÷F–ÆU÷Æà¢V–çCcE÷B6W#²–çBÇbÂ6Ã²'E÷7V"‚e"ç'BÂ²Âf6W"ÂfÇbÂg6Â“°¢÷5÷Væf—%öÆö6µöÆö6²‚fu÷&×R“²fö–B§GbÒåTÄÃ²FW…öf7E÷&–B‡F–BÂgGb“²–CÄÕDÅFW‡GW&SâBÒ…õö'&–FvR–CÄÕDÅFW‡GW&Sâ—Gc²÷5÷Væf—%öÆö6µ÷VæÆö6²‚fu÷&×R“°¢–b‚B’6öçF–çVS°¢–çBGWÒ²f÷"†–çBÒ²Âuöf%öã²²²’GWÃÒuöf%÷7&5·ÒÓÒC²–b†GW’6öçF–çVS°¢–CÄÕDÅFW‡GW&Sâ&6RÒBç&VçEFW‡GW&RòBç&VçEFW‡GW&R¢C²å5T–çFVvW"FÂÒBç&VçEFW‡GW&RòBç&VçE&VÆF—fTÆWfVÂ¢ÂG2ÒBç&VçEFW‡GW&RòBç&VçE&VÆF—fU6Æ–6R¢°¢–b‡FÂ²BæÖ—ÖÆWfVÄ6÷VçBÃÒ„å5T–çFVvW"–ÇbÇÂFÂâ„å5T–çFVvW"–Çb’6öçF–çVS²òòF†R6×ÆVBÆWfVÇ2Fòæ÷B–æ6ÇVFRF†RGF6†ÖVçBw2ÆWfVÂ†Ö—vVæW&F–öâWF2â“¢æ÷BfVVF&6°¢å5T–çFVvW"FâÒBçFW‡GW&UG—RÓÒÕDÅFW‡GW&UG—S$D'&’ÇÂBçFW‡GW&UG—RÓÒÕDÅFW‡GW&UG—T7V&T'&’ÇÂBçFW‡GW&UG—RÓÒÕDÅFW‡GW&UG—S$D×VÇF—6×ÆT'&’òBæ'&”ÆVæwF‚¢‡BçFW‡GW&UG—RÓÒÕDÅFW‡GW&UG—T7V&T'&’òb¢’¢BçFW‡GW&UG—RÓÒÕDÅFW‡GW&UG—T7V&Ròb¢°¢–b‡G2²FâÃÒ„å5T–çFVvW"—6ÂÇÂG2â„å5T–çFVvW"—6Â’6öçF–çVS²òòæ÷"—G2Æ–W#¢æ÷BfVVF&6°¢–b†uöf%öÖöFRÓÒbb‚‡vÒÂuöf%÷vÖ6²’ãâ²b’’²uöf%÷&Ö6²ÃÒRÃÂ³²6öçF–çVS²Òòòæ÷F†–ær–âF†—272†2w&—GFVâ—B6òf#¢&VBF—&V7FÇ¢òòÖ—Ö6†–â76W2‡&VæFW"ÆWfVÂâv†–ÆRF†Rf–Wr6÷fW'2F†R6†–âÂ6×Æ–ærÆ÷vW"ÆWfVÂ“¢F†RG&r6×ÆW2f–WröbF†RÆWfVÇ2&VÆ÷râ–ç7FVBÂv—F‚F†R6ÖR&6Ræ@¢òò6òF†R6ÖRÆWfVÂçVÖ&W'2â6×Æ–ærF†RgVÆÂf–Wr&VBF†RGF6†ÖVçBw2ÆWfVÂFöó¢Æ–æV"Ö—f–ÇFW"fWF6†W2F†RæW‡BÆWfVÂWfVâBvV–v‡BÂæBöâÆRw2uW2à¢òòGF6†VBÆWfVÂ&VBF‡&÷Vv‚FW‡GW&R†öÆG2æòFVf–æVBFFâFöæ¶W’¶öær&æç¦w2Vçf—&öæÖVçB7V&Rƒ#‡ƒ#‚Â‚ÆWfVÇ2Â#s#bÂÆWfVÇ2âãrV6‚G&vâg&öÒF†RöæP¢òò&÷fRv—F‚âW‡Æ–6—BÆWfVÂ’v÷BæâFW†VÇ2F†Bv“²—G2&VfÆV7F–öç27&VBF†VÒ÷fW"F†R66VæRF‡&÷Vv‚F†R7V&Rw2&Vf–ÇFW"ÂæB&ÆööÒGW&æVBF†VÒ–çFòv†—FR&Æ÷F6†W0¢–b‡BæÖ—ÖÆWfVÄ6÷VçBÒbb„å5T–çFVvW"–ÇbâFÂ’°¢–CÄÕDÅFW‡GW&SâW6RÒ¶&6RæWuFW‡GW&Uf–Wuv—F…—†VÄf÷&ÖC§Bç—†VÄf÷&ÖBFW‡GW&UG—S§BçFW‡GW&UG—RÆWfVÇ3¤å4Ö¶U&ævR‡FÂÂ„å5T–çFVvW"–ÇbÒFÂ’6Æ–6W3¤å4Ö¶U&ævR‡G2ÂFâ’7v—§¦ÆS§Bç7v—§¦ÆUÓ°¢–b‚W6R’²$TEôôä4Rƒƒ"Â'6ÖR×72fVVF&6²6×Æ–æs¢6ææ÷BÖ¶RF†Rf–WröbÆWfVÇ2VÇR²VB&VÆ÷rF†RGF6†ÖVçB†f÷&ÖBVÇR“²ÆVgB2—2"Â‡Vç6–væVBÆöær—FÂÂÇbÒ†–çB—FÂÂ‡Vç6–væVBÆöær—Bç—†VÄf÷&ÖB“²6öçF–çVS²Ð¢–b†uöf%÷W6U¶uöf%öåÒbbuöf%÷W6U¶uöf%öåÒÒW6R’FVfW%öö&¥÷Vç&Vb†uöf%÷W6U¶uöf%öåÒ“°¢uöf%÷7&5¶uöf%öåÒÒC²uöf%÷W6U¶uöf%öåÒÒW6S²uöf%÷&–E¶uöf%öåÒÒW6RæwU&W6÷W&6T”Båö–×Ã²uöf%öâ²³°¢7FF–2–çBöæ6S²–b‚öæ6R²²’ÔÄôr‚'6ÖR×72fVVF&6²6×Æ–æs¢6†FW"W2&VæFW'2ÆWfVÂVBöbf–WröbÆWfVÇ2VÇR²VÇS²—B6×ÆW2f–WröbF†RÆWfVÇ2&VÆ÷r†Ö—Ö6†–â72’"Âg2Óæ¶W’ÂÇbÂ‡Vç6–væVBÆöær—FÂÂ‡Vç6–væVBÆöær—BæÖ—ÖÆWfVÄ6÷VçB“°¢6öçF–çVS°¢Ð¢–b‡BæÖ—ÖÆWfVÄ6÷VçBÒÇÂFâÒ’²òòÆ–W"76W2ÂæBÖ—Ö6†–â76W2&VæFW&–ærF†Rf–Wrw2&6RÆWfVÃ¢æ÷B6æ6†÷GFV@¢7FF–2–çBöæ6S²–b‚öæ6R²²’ÔÄôr‚'6ÖR×72fVVF&6²6×Æ–æs¢6†FW"W2&VæFW'2ÆWfVÂVBÆ–W"VBv†–ÆR—G2f–Wr6÷fW'2ÆWfVÇ2VÇR²VÇRÂÆ–W'2VÇR²VÇS²ÆVgB2—2†Ö—Ö6†–â÷"Æ–W"72’"Âg2Óæ¶W’ÂÇbÂ6ÂÂ‡Vç6–væVBÆöær—FÂÂ‡Vç6–væVBÆöær—BæÖ—ÖÆWfVÄ6÷VçBÂ‡Vç6–væVBÆöær—G2Â‡Vç6–væVBÆöær—Fâ“°¢6öçF–çVS²Ð¢–b‡BçFW‡GW&UG—RÒÕDÅFW‡GW&UG—S$BÇÂBç6×ÆT6÷VçBÒÇÂG2Ò„å5T–çFVvW"—6ÂÇÂ"ÓæGGBÒÕDÅFW‡GW&UG—S$BÇÂ&6Rç&VçEFW‡GW&R’°¢$TEôôä4RƒƒÂ'6ÖR×72fVVF&6²6×Æ–æs¢F†R6×ÆVBf–Wr—2æ÷B6–ævÆRÖÆWfVÂ$Bf–WröbF†RGF6†ÖVçBw2ÆWfVÂæBÆ–W"‡G—RVÇRÆWfVÇ2VÇR²VÇRÆ–W"VÇRÂGF6†ÖVçBÆWfVÂVBÆ–W"VBÂ6†FW"W2“²ÆVgB2—2"Â‡Vç6–væVBÆöær—BçFW‡GW&UG—RÂ‡Vç6–væVBÆöær—FÂÂ‡Vç6–væVBÆöær—BæÖ—ÖÆWfVÄ6÷VçBÂ‡Vç6–væVBÆöær—G2ÂÇbÂ6ÂÂg2Óæ¶W’“°¢6öçF–çVS²Ð¢å5T–çFVvW"rÒÔ‚†&6Rçv–GF‚ãâÇbÂVÂ’Â‚ÒÔ‚†&6Ræ†V–v‡BãâÇbÂVÂ“²–çBg&W6ƒ°¢f%ö66†U÷B¦RÒf%÷6æ‡6W"ÂÇbÂ6ÂÂ&6Rç—†VÄf÷&ÖBÂrÂ‚Âuöf%öÖöFRÓÒ"Âfg&W6‚“²–b‚R’6öçF–çVS°¢–CÄÕDÅFW‡GW&Sâ6æÒRÓç6æ°¢–b†g&W6‚’°¢–b†uöVæ2’'öVæB‚“°¢–CÄÕDÃD6ö×WFT6öÖÖæDVæ6öFW#â&RÒ&Æ—B‚“²Væ5ö%ö6÷•÷FW‡GW&R†&RÂ&6RÂ„å5T–çFVvW"—6ÂÂ„å5T–çFVvW"–ÇbÂÕDÄ÷&–v–äÖ¶RƒÂÂ’ÂÕDÅ6—¦TÖ¶R‡rÂ‚Â’Â6æÂÂÂÕDÄ÷&–v–äÖ¶RƒÂÂ’“°¢ÒVÇ6R–b†uöVæ2bb†uöf%÷&Ö6²ãâ²b’’'öVæB‚“²òòâV&Æ–W"G&röbF†—272&VBF†RGF6†ÖVçBF—&V7FÇ“¢—B×W7Bæ÷B6VRF†—272w2w&—FW0¢–b‚uöVæ2’f%÷75÷7–æ2‚“°¢ÕDÅFW‡GW&U7v—§¦ÆT6†ææVÇ27rÒBç7v—§¦ÆS°¢–çB6ÖRÒBç—†VÄf÷&ÖBÓÒ6æç—†VÄf÷&ÖBbb7rç&VBÓÒÕDÅFW‡GW&U7v—§¦ÆU&VBbb7ræw&VVâÓÒÕDÅFW‡GW&U7v—§¦ÆTw&VVâbb7ræ&ÇVRÓÒÕDÅFW‡GW&U7v—§¦ÆT&ÇVRbb7ræÇ†ÓÒÕDÅFW‡GW&U7v—§¦ÆTÇ†°¢–CÄÕDÅFW‡GW&SâW6RÒ6ÖRò6æ¢·6ææWuFW‡GW&Uf–Wuv—F…—†VÄf÷&ÖC§Bç—†VÄf÷&ÖBFW‡GW&UG—S¤ÕDÅFW‡GW&UG—S$BÆWfVÇ3¤å4Ö¶U&ævRƒÂ’6Æ–6W3¤å4Ö¶U&ævRƒÂ’7v—§¦ÆS§7uÓ°¢–b‚W6R’²$TEôôä4RƒƒÂ'6ÖR×72fVVF&6²6×Æ–æs¢6ææ÷B7&VFRF†R6æ6†÷Bf–Wr†f÷&ÖBVÇR“²ÆVgB2—2"Â‡Vç6–væVBÆöær—Bç—†VÄf÷&ÖB“²6öçF–çVS²Ð¢–b†uöf%÷W6U¶uöf%öåÒbbuöf%÷W6U¶uöf%öåÒÒW6R’FVfW%öö&¥÷Vç&Vb†uöf%÷W6U¶uöf%öåÒ“°¢uöf%÷7&5¶uöf%öåÒÒC²uöf%÷W6U¶uöf%öåÒÒW6S²uöf%÷&–E¶uöf%öåÒÒW6RæwU&W6÷W&6T”Båö–×Ã²uöf%öâ²³°¢–b‚g&W6‚’²uöf%÷&WW6R²³²B‚$âVÆÇR6ÖR×72fVVF&6²6×Æ–æs¢GF6†ÖVçBVB6W#ÒVÆÇRÆWfVÂVBÆ–W"VC¢F†—2G&r6×ÆW2F†R6æ6†÷BF¶Vâ6–æ6RF†RÆ7B&'&–W""Âu÷7FE÷72²uöVæ2Â²Â‡Vç6–væVBÆöærÆöær—6W"ÂÇbÂ6Â“²6öçF–çVS²Ð¢uöf%ö6÷VçB²³°¢–b†uöf%ö6÷VçBÓÒÇÂuöf%ö6÷VçBÓÒÇÂuöf%ö6÷VçBÓÒÇÂuöf%ö6÷VçBRÓÒ¢ÔÄôr‚'6ÖR×72fVVF&6²6×Æ–æs¢6æ6†÷B2VÆÇRÂVÆÇRG&w2&WW6VBöæRÂVÆÇR&VBF†RF–ÆR‚VÆÇRv—FVBf÷"F†V—"—VÆ–æR’ÂVÆÇR76W2VæFVB&Vf÷&Rw&—FRFòâGF6†ÖVçB&VBF—&V7FÇ’†g&ÖRVÆÇR72VÆÇRÂ6†FW"W26Æ÷BVB6×ÆW2GF6†ÖVçBVB6W#ÒVÆÇRVÇW‚VÇRf÷&ÖBVÇR’"Â‡Vç6–væVBÆöærÆöær–uöf%ö6÷VçBÂ‡Vç6–væVBÆöærÆöær–uöf%÷&WW6RÂ‡Vç6–væVBÆöærÆöær–uöf%÷F–ÆUöG&w2Â‡Vç6–væVBÆöærÆöær–uöf%÷F–ÆU÷v—BÂ‡Vç6–væVBÆöærÆöær–uöf%÷7Æ—BÀ¢‡Vç6–væVBÆöærÆöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR’Âu÷7FE÷72²uöVæ2Âg2Óæ¶W’Â"Óç&VrÂ²Â‡Vç6–væVBÆöærÆöær—6W"Â‡Vç6–væVBÆöær—rÂ‡Vç6–væVBÆöær–‚Â‡Vç6–væVBÆöær–&6Rç—†VÄf÷&ÖB“°¢B‚$âVÆÇR6ÖR×72fVVF&6²6×Æ–æs¢GF6†ÖVçBVB6W#ÒVÆÇRÆWfVÂVBÆ–W"VB6÷–VB–çFò6æ6†÷C²F†—2G&r6×ÆW2F†R6æ6†÷B"Âu÷7FE÷72²uöVæ2Â²Â‡Vç6–væVBÆöærÆöær—6W"ÂÇbÂ6Â“°¢ÒÐ¢uöf%÷vÖ6²ÃÒvÓ°¢–b†uöf%öâ’²uöf%÷6WÒuöG&u÷6W²&w5öF—'G•öÆÂ‚“²uöf%÷&V&–æBÒ²Ð¢–b†çF–ÆR’²uöf%÷G6WÒuöG&u÷6W²uöf%÷F–ÆUöG&w2²³°¢–b†uöf%÷F–ÆUöG&w2ÓÒÇÂuöf%÷F–ÆUöG&w2ÓÒÇÂuöf%÷F–ÆUöG&w2RÓÒ¢ÔÄôr‚'6ÖR×72fVVF&6²6×Æ–æs¢G&r2VÆÇR&VG2—G2GF6†ÖVçG2g&öÒF†RF–ÆRÂVÆÇR6æ6†÷G26òf"†g&ÖRVÆÇRÂ6†FW"W2ÂGF6†ÖVçG2R7‚’"Â‡Vç6–væVBÆöærÆöær–uöf%÷F–ÆUöG&w2Â‡Vç6–væVBÆöærÆöær–uöf%ö6÷VçBÀ¢‡Vç6–væVBÆöærÆöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR’Âg2Óæ¶W’Âuöf%÷FFW62æf&Ö6²“²Ð§Ð¢òòÒÒÒÒÒÒÒÒÒÒ–æF—&V7BG&w2ÖWFÂ6ææ÷BF¶R2F†W’&RÒÒÒÒÒÒÒÒÒÐ¢òòVG2æBG&–ævÆRfç2„G&t'&—4–æF—&V7B’ÂæB–æFW†VBVG2òfç2ò‚Ö&—B–æF–6W2„G&tVÆVÖVçG4–æF—&V7B’âF†R¶6÷VçBÂ–ç7Fæ6W2Âf—'7BÂââçÒ&V6÷&B—2W7VÆÇ’w&—GFVâ'’F†RuRV&Æ–W ¢òò–âF†Rg&ÖR„‡—'VÆRv'&–÷'3¢vRöb6ÆÖ—G’G&w2uRÖG&—fVâVffV7G22–æF—&V7BVG2–â—G2f—'7B&GFÆR’Â6òF†R5R6ææ÷BW‡æB—BB&WÆ’â6ö×WFR726öçfW'G2V6‚&V6÷&B–çFð¢òòÕDÄG&t–æFW†VE&–Ö—F—fW4–æF—&V7D&wVÖVçG2–âF†Rg&ÖR&VæÂæBF†RG&r&V6öÖW2â–æFW†VB–æF—&V7BG&s¢'&—2–æFW‚F†R6†&VBu÷VEö–"òuöfåö–"v—F‚f—'7B2F†R&6RfW'FW‚À¢òò–æFW†VBöæW2&RW‡æFVB–çFò3"Ö&—BÆ—7BöbBÖ÷7B•„5ô4–æF–6W2âF†R6ö×WFR726ææ÷B'Vâ–ç6–FR&VæFW"72âv†VâF†Rf—'7B7V6‚G&r—2ÖWBÂ—†5÷'Vâ6öçfW'G2—BæBWfW'¢òò7V6‚G&rgFW"—BWFòF†RæW‡B'VffW"w&—FR÷WG6–FR&VæFW"72†6÷’ÂWFFRòf–ÆÂÂFW‡GW&R(i"'VffW"ÂF—7F6ƒ¢F†W’'V×u÷w%öWö6‚’Â6òV6‚&V6÷&B—2&VBgFW"—G2Æ7Bw&—FW ¢òòæBF†RÆFW"G&w2f–æBF†V—"6öçfW'6–öââöæÇ’G&rÖWB–ç6–FRâ÷Vâ&VæFW"72v—F‚æò6öçfW'6–öâ–WB7Æ—G2F†B72à¢6FVf–æR•„5ôÔ‚#€¢6FVf–æR•„5ô4ƒ2¢c3ƒB§G—VFVb7G'V7B²V–çCcE÷B–æBÂ–'Vc²–çC3%÷B&–ÒÂ—G—S²V–çCcE÷B&w2Â—†#²V–çC3%÷B—†ÆVã²Ò—†5öVçC²òò&w2¢F†R&V6÷&B6÷VÆBæ÷B&R6öçfW'FV@§7FF–2—†5öVçBuö—†5´•„5ôÔ…Ó²7FF–2–çBuö—†5öã²7FF–2V–çCcE÷Buö—†5öWö6‚Âuö—†5÷7V"Âuö—†5÷'Vç2Âuö—†5÷7Æ—G3°¢òòF†R¶W&æVÇ3¢FÒövg‚ö×6Âö—†2æÖWFÀ§7FF–2–çB—†5÷vçFVB†6öç7BöG&r¦GrÂ6öç7BöG&uö–G‚¦F’’°¢–b†Gr’&WGW&âGrÓæ–æBbb†GrÓç&–ÒÓÒbÇÂGrÓç&–ÒÓÒr“°¢&WGW&âF’Óæ–æBbb†F’Óç&–ÒÓÒbÇÂF’Óç&–ÒÓÒrÇÂF’Óæ—G—RÓÒ“°§Ð§7FF–26öç7B—†5öVçB¦—†5öf–æB†6öç7BöG&r¦GrÂ6öç7BöG&uö–G‚¦F’’°¢–b†uö—†5öWö6‚Òu÷w%öWö6‚ÇÂuö—†5÷7V"Òu÷7V%÷6W&–Â’&WGW&âåTÄÃ°¢V–çCcE÷B–æBÒGròGrÓæ–æB¢F’Óæ–æBÂ–"ÒGrò¢F’Óæ–'Vc²–çB&–ÒÒGròGrÓç&–Ò¢F’Óç&–ÒÂ—BÒGròÓ¢F’Óæ—G—S°¢f÷"†–çB’Ò²’Âuö—†5öã²’²²’–b†uö—†5¶•Òæ–æBÓÒ–æBbbuö—†5¶•Òæ–'VbÓÒ–"bbuö—†5¶•Òç&–ÒÓÒ&–Òbbuö—†5¶•Òæ—G—RÓÒ—B’&WGW&âfuö—†5¶•Ó°¢&WGW&âåTÄÃ°§Ð§7FF–2fö–B—†5÷F¶R†6öç7BöG&r¦GrÂ6öç7BöG&uö–G‚¦F’’°¢–b†uö—†5öâÂ•„5ôÔ‚bb—†5öf–æB†GrÂF’’’uö—†5¶uö—†5öâ²µÒÒ†—†5öVçB—¶GròGrÓæ–æB¢F’Óæ–æBÂGrò¢F’Óæ–'VbÂGròGrÓç&–Ò¢F’Óç&–ÒÂGròÓ¢F’Óæ—G—RÂÂÂÓ°§Ð§7FF–2fö–B—†5÷'Vâ†6öç7BöG&r¦GrÂ6öç7BöG&uö–G‚¦F’’°¢7FF–2–CÄÕDÄ6ö×WFU—VÆ–æU7FFSâ5³%Ó²7FF–2–CÄÕDÃD&wVÖVçEF&ÆSâC²7FF–2–çBf–ÆVC°¢–b†f–ÆVB’&WGW&ã°¢–b‚B’²å4W'&÷"¦RÒæ–Ã²–CÄÕDÄÆ–'&'“âÆ–"Ò†÷7EöÆ–"‚&—†2"ÂfR“°¢f÷"†–çB²Ò²²Â"bbÆ–#²²²²’²–CÄÕDÄgVæ7F–öãâbÒ¶Æ–"æWtgVæ7F–öåv—F„æÖS¦²ò&—†5öW‡æB"¢&—†5ö'&—2%Ó²5¶µÒÒbò¶uöFWbæWt6ö×WFU—VÆ–æU7FFUv—F„gVæ7F–öã¦bW'&÷#¢fUÒ¢æ–Ã²Ð¢ÕDÃD&wVÖVçEF&ÆTFW67&—F÷"¦BÒ´ÕDÃD&wVÖVçEF&ÆTFW67&—F÷"æWuÓ²BæÖ„'VffW$&–æD6÷VçBÒ²BÒ5³Òbb5³Òò&V5÷F&ÆR…¶uöFWbæWt&wVÖVçEF&ÆUv—F„FW67&—F÷#¦BW'&÷#¢fUÒ’¢æ–Ã°¢–b‚B’²f–ÆVBÒ²ÔÄôr‚%µ$TEÒ–æF—&V7BG&r6öçfW'6–öã¢6ææ÷B7&VFRF†R6ö×WFR—VÆ–æRW2†–æF—&V7BG&w2öbVG2òfç2ò‚Ö&—B–æF–6W2&R6¶—VB’"ÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“²&WGW&ã²ÒÐ¢–b†uö—†5öWö6‚Òu÷w%öWö6‚ÇÂuö—†5÷7V"Òu÷7V%÷6W&–ÂÇÂuö—†5öââ•„5ôÔ‚Ò‚’²uö—†5öâÒ²uö—†5öWö6‚Òu÷w%öWö6ƒ²uö—†5÷7V"Òu÷7V%÷6W&–Ã²Ð¢–çBãÒuö—†5öã²—†5÷F¶R†GrÂF’“°¢–b†uöW†V5öæW‡Bbbu÷&WEöâÃÒ‚’²òòÆFW"G&w2–â&WÆ’÷&FW"†7W'&VçBÆ—7B(i"6ÆÆW'2(i"F†R&W7BöbF†—27V&Ö—76–öâ’ÂWFòF†RæW‡B'VffW"w&—FP¢g¥ö7W'6÷"7W"Ò²çÒuöW†V5öæW‡BÂç&WEöâÒu÷&WEöçÓ²ÖVÖ7’†7W"ç&WBÂu÷&WBÂ6—¦Vöb‚¦u÷&WB’¢‡6—¦U÷B–u÷&WEöâ“°¢6öç7BV–çC…÷B§²÷ö†G"ƒ²6öç7Bfö–B¦°¢f÷"†–çBwV&BÒ²wV&BÂ#bbuö—†5öâÂ•„5ôÔ‚bbg¥öæW‡B‚f7W"ÂgÂf‚Âf“²wV&B²²’°¢–b†‚æ÷ÓÒõôD•5D4‚ÇÂ‚æ÷ÓÒõô4õ•ô#$"ÇÂ‚æ÷ÓÒõõUDDUõT"ÇÂ‚æ÷ÓÒõô4ÄT%ô%TbÇÂ‚æ÷ÓÒõô4õ•õC$"’'&V³°¢öG&rGC²öG&uö–G‚—C²6öç7BöG&r¦BÒ‚æ÷ÓÒõôE$ròG&u÷&V2†Â‚ç6—¦RÂfGB’¢åTÄÃ²6öç7BöG&uö–G‚§‚Ò‚æ÷ÓÒõôE$uô”E‚òG&uö–G…÷&V2†Â‚ç6—¦RÂf—B’¢åTÄÃ°¢–b†Bbb—†5÷vçFVB†BÂåTÄÂ’’—†5÷F¶R†BÂåTÄÂ“°¢VÇ6R–b‡‚bb—†5÷vçFVB„åTÄÂÂ‚’’—†5÷F¶R„åTÄÂÂ‚“°¢Ð¢Ð¢–b†uöVæ2’²uö—†5÷7Æ—G2²³²B‚$ÒVÆÇR–æF—&V7BG&r6öçfW'6–öã¢&VæFW"727Æ—B†W&R"Âu÷7FE÷72“²'öVæB‚“²Ð¢–b†uö&Æ—B’²Væ5ö%öVæB†uö&Æ—B“²uö&Æ—BÒæ–Ã²Ð¢–çBâÒuö—†5öâÒãÂæÒ²–b†âÃÒ’&WGW&ã°¢G—VFVb7G'V7B²V–çCcE÷B&V2Â&w3²V–çC3%÷B&–ÒÂ6²Ò¦ö#°¢G—VFVb7G'V7B²V–çCcE÷B&V2Â–"Â÷WBÂ&w3²V–çC3%÷B&–ÒÂ—7¢Â–ÆVâÂ6²ÒV¦ö#°¢–CÄÕDÄ'VffW#â#²6—¦U÷BòÒ&VæöÆÆö2‚‡6—¦U÷B–â¢ƒ3"²6—¦Vöb†¦ö"’’Âf"“²V–çCcE÷BrÒ"æwTFG&W72²ó²òòâ&wVÖVçB&V6÷&G2ƒ3"'—FW2'B’ÂF†VâF†R'&—2¦ö'0¢¦ö"¦¢Ò†¦ö"¢’‚‡V–çC…÷B¢–"æ6öçFVçG2²ò²‡6—¦U÷B–â¢3"“°¢–CÄÕDÃD6ö×WFT6öÖÖæDVæ6öFW#â6RÒVæ5ö5ö&Vv–â„åTÄÂ“²¶6R6WD&wVÖVçEF&ÆS¦EÓ°¢f÷"†–çB’Ò²’Âã²’²²’°¢—†5öVçB¦RÒfuö—†5¶ã²•Ó²V–çCcE÷B&V2Â–"Ò²fö–B¦Ö#²V–çCcE÷BÖó²RÓæ&w2Ò°¢–b‚×FÅöwUöFG"†RÓæ–æBÂRÓæ—G—RÂòb¢#ÂfÖ"ÂfÖòÂg&V2’’6öçF–çVS°¢–b†RÓæ—G—RÂ’²òòG&t'&—4–æF—&V7BVG2òfç3¢F†R6†&VB–æFW‚'VffW'2ÂöæRF‡&VBW"&V6÷&@¢–b†RÓç&–ÒÓÒr’VEö–%öÖ¶R‚“²VÇ6Rfåö–%öÖ¶R‚“°¢RÓæ—†"ÒRÓç&–ÒÓÒròu÷VEöwR¢uöfåöwS²RÓæ—†ÆVâÒ‡V–çC3%÷B’†RÓç&–ÒÓÒròu÷VEö–"æÆVæwF‚¢uöfåö–"æÆVæwF‚“°¢¥¶æ²µÒÒ†¦ö"—·&V2Âr²3"¢‡V–çCcE÷B–’Â‡V–çC3%÷B–RÓç&–ÒÂRÓæ—†ÆVâòGÓ²RÓæ&w2Òr²3"¢‡V–çCcE÷B–“°¢ÒVÇ6R²òòG&tVÆVÖVçG4–æF—&V7C¢W‡æFVB–çFò3"Ö&—BÆ—7BÂöæRF‡&VBW"÷WGWB–æFW€¢–b‚×FÅöwUöFG"†RÓæ–'VbÂÂfÖ"ÂfÖòÂf–"’’6öçF–çVS°¢–CÄÕDÄ'VffW#âö#²6—¦U÷BöòÒ&VæöÆÆö2„•„5ô4¢B²6—¦Vöb†V¦ö"’Âfö"“°¢V¦ö"¦¢Ò†V¦ö"¢’‚‡V–çC…÷B¢–ö"æ6öçFVçG2²öò²•„5ô4¢B“°¢¦¢Ò†V¦ö"—·&V2Â–"Âö"æwTFG&W72²öòÂr²3"¢‡V–çCcE÷B–’Â‡V–çC3%÷B–RÓç&–ÒÂRÓæ—G—RÓÒòR¢RÓæ—G—RÓÒò'R¢GRÂ‡V–çC3%÷B”Ô”â‚‚…õö'&–FvR–CÄÕDÄ'VffW#â–Ö"’æÆVæwF‚ÒÖòÂ†fffffffgVÆÂ’Â•„5ô4Ó°¢¶B6WDFG&W73¦ö"æwTFG&W72²öò²•„5ô4¢BD–æFWƒ£Ó²¶6R6WD6ö×WFU—VÆ–æU7FFS§5³ÕÓ°¢¶6RF—7F6…F‡&VFw&÷W3¤ÕDÅ6—¦TÖ¶R„•„5ô4òcBÂÂ’F‡&VG5W%F‡&VFw&÷W¤ÕDÅ6—¦TÖ¶RƒcBÂÂ•Ó°¢RÓæ—†"Òö"æwTFG&W72²öó²RÓæ—†ÆVâÒ•„5ô4¢C²RÓæ&w2Òr²3"¢‡V–çCcE÷B–“°¢²7FF–2–çBöæ6S²–b‚öæ6R²²’ÔÄôr‚&–æF—&V7B–æFW†VBG&r‡&–Ö—F—fRVBÂVBÖ'—FR–æF–6W2’W‡æFVBFò3"Ö&—B–æF–6W2öâF†RuRÂBÖ÷7BVB"ÂRÓç&–ÒÂRÓæ—G—RÓÒò¢RÓæ—G—RÓÒò"¢BÂ•„5ô4“²Ð¢Ð¢Ð¢–b†æ’²¶B6WDFG&W73¦r²‡V–çCcE÷B–â¢3"D–æFWƒ£Ó²¶6R6WD6ö×WFU—VÆ–æU7FFS§5³ÕÓ²¶6RF—7F6…F‡&VFw&÷W3¤ÕDÅ6—¦TÖ¶R‚„å5T–çFVvW"–æÂÂ’F‡&VG5W%F‡&VFw&÷W¤ÕDÅ6—¦TÖ¶RƒÂÂ•Ó²Ð¢Væ5ö5öVæB†6R“°¢&w5öF—'G•öÆÂ‚“°¢–b‚uö—†5÷'Vç2²²’ÔÄôr‚&–æF—&V7BG&r6öçfW'6–öã¢f—'7BF–ÖRÂVBG&w2‡&V6÷&G2f÷"VG2òfç2ò‚Ö&—B–æF–6W2&V6öÖR–æFW†VB–æF—&V7B&wVÖVçG2–â6ö×WFR72&Vf÷&RF†R&VæFW"72’"Ââ“°¢VÇ6R–b†uö—†5÷'Vç2R3cÓÒ’ÔÄôr‚&–æF—&V7BG&r6öçfW'6–öã¢VÆÇRF–ÖW2Â&VæFW"727Æ—BVÆÇRF–ÖW2"Â‡Vç6–væVBÆöærÆöær–uö—†5÷'Vç2Â‡Vç6–væVBÆöærÆöær–uö—†5÷7Æ—G2“°¢òòDÕôådåô•„4ÄôsÓÆãã¢Æöw2F†Rf—'7Bâ6öçfW'6–öç2‡&V6÷&B(i"&wVÖVçG2’öæ6RF†RuR†2'VâF†VÐ¢7FF–2–çBÆrÒÓ²–b†ÆrÂ’²6öç7B6†"§bÒvWFVçb‚$DÕôådåô•„4Äôr"“²ÆrÒbòFö’‡b’¢²Ð¢f÷"†–çB’Ò²’ÂâbbÆrâ²’²²’²6öç7B—†5öVçB¦RÒfuö—†5¶ã²•Ó²–b‚RÓæ&w2’6öçF–çVS²ÆrÒÓ°¢6öç7BV–çC3%÷B§7&2Ò†6öç7BV–çC3%÷B¢’‡V–çGG%÷B–RÓæ–æBÂ¦&rÒ†6öç7BV–çC3%÷B¢’‚†6öç7BV–çC…÷B¢–"æ6öçFVçG2²ò²3"¢‡6—¦U÷B–’“°¢–çB&–ÒÒRÓç&–ÒÂ—BÒRÓæ—G—S²Vç6–væVBÆöærÆöærg"Ò‡Vç6–væVBÆöærÆöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR’Â72Òu÷7FE÷72Â7Æ—G2Òuö—†5÷7Æ—G3²–CÄÕDÄ'VffW#â¶VWÒ#°¢¶uö6öÖÖ—Eö&Æö6·2FDö&¦V7C¥ç²‡fö–B–¶VW²ÔÄôr‚&–æF—&V7BG&r6öçfW'6–öâg&ÖRVÆÇR72VÆÇS¢&–Ö—F—fRVB–æFW‚G—RVB&V6÷&BWRÂWRÂWRÂWR(i"&wVÖVçG2WRÂWRÂWRÂVBÂWR‡7Æ—G26òf"VÆÇR’"Âg"Â72Â&–ÒÂ—BÂ7&5³ÒÂ7&5³ÒÂ7&5³%ÒÂ7&5³5ÒÂ&u³ÒÂ&u³ÒÂ&u³%ÒÂ†–çB–&u³5ÒÂ&u³EÒÂ7Æ—G2“²ÕÓ²Ð§Ð¢òòÒÒÒÒÒÒÒÒÒÒ‚Ö&—B–æF–6W2w&—GFVâ'’F†RuRV&Æ–W"–âF†R7V&Ö—76–öâÒÒÒÒÒÒÒÒÒÐ¢òòÖWFÂ†2æò‚Ö&—B–æFW‚G—RÂ6òG&rv–FVç2‚Ö&—B–æF–6W2FòbÖ&—BöâF†R5RB&WÆ’âv†Vâ'VffW"w&—FRVæ6öFVBV&Æ–W"–âF†—27V&Ö—76–öâ†6÷’òWFFRòf–ÆÂÀ¢òò6VRVæEö†—G2’6÷fW'2F†VÒÂF†R5R&VG2F†R'—FW2g&öÒ&Vf÷&RF†Bw&—FR…FV'2öbF†R¶–ævFöÒ’âF†÷6RG&w2&Rv–FVæVBöâF†RuR–ç7FVC¢6ö×WFR72&VG2F†P¢òò‚Ö&—B–æF–6W2g&öÒbÖ'—FRÖÆ–væVBFG&W72Âv—F‚F†R'—FR6†–gB26öç7FçB†–æFW‚FFÖ’7F'Bç—v†W&R’ÂæBw&—FW2bÖ&—BöæW2–çFòF†Rg&ÖR&VæÂv†–6‚Æ—fW0¢òòVçF–ÂF†R7V&Ö—76–öâ6ö×ÆWFW2â2–â—†5÷'VâÂ6ö×WFR726ææ÷B'Vâ–ç6–FR&VæFW"73¢F†Rf—'7B7V6‚G&rv–FVç2—G6VÆbæBWfW'’7V6‚G&rgFW"—BWFòF†RæW‡@¢òò'VffW"w&—FR÷WG6–FR&VæFW"72Â6òöæÇ’G&rÖWB–ç6–FRâ÷Vâ&VæFW"72v—F‚æòv–FVæ–ær–WB7Æ—G2—BâG&w2v†÷6R–æF–6W2æòw&—FR6÷fW'2¶VWF†R5Rv–FVæ–ærà¢òòFFVBg&öÒÖ4æF—fTvÖW2rv–FVå÷S…öwR†v—F‡V"æ6öÒöÇ“SC’ôÖ4æF—fTvÖW2ÂÔ•B’à¢6FVf–æRt”DUôÔ‚#S`§G—VFVb7G'V7B²V–çCcE÷B–'Vc²–çC3%÷B6÷VçC²V–çCcE÷BwS²Òv–FUöVçC²òòwS¢F†RbÖ&—B–æF–6W2–âF†Rg&ÖR&Væƒ¢æ÷Bv–FVæVB§7FF–2v–FUöVçBu÷v–FUµt”DUôÔ…Ó²7FF–2–çBu÷v–FUöã²7FF–2V–çCcE÷Bu÷v–FUöWö6‚Âu÷v–FU÷7V"Âu÷v–FU÷'Vç2Âu÷v–FUöG&w2Âu÷v–FU÷7Æ—G3°¢òòF†R¶W&æVÂv–FU÷Sƒ¢FÒövg‚ö×6Â÷v–FRæÖWFÀ§7FF–2–çBv–FU÷vçFVB†6öç7BöG&uö–G‚¦F’’²&WGW&âF’Óæ–æBbbF’Óæ—G—RÓÒbbF’Óç&–ÒÒbbbF’Óç&–ÒÒrbbF’Óæ6÷VçBâbbVæEö†—G2†F’Óæ–'VbÂ‡V–çCcE÷B–F’Óæ6÷VçB“²Ð§7FF–26öç7Bv–FUöVçB§v–FUöf–æB†6öç7BöG&uö–G‚¦F’’°¢–b†u÷v–FUöWö6‚Òu÷w%öWö6‚ÇÂu÷v–FU÷7V"Òu÷7V%÷6W&–Â’&WGW&âåTÄÃ°¢f÷"†–çB’Ò²’Âu÷v–FUöã²’²²’–b†u÷v–FU¶•Òæ–'VbÓÒF’Óæ–'Vbbbu÷v–FU¶•Òæ6÷VçBÓÒF’Óæ6÷VçB’&WGW&âfu÷v–FU¶•Ó°¢&WGW&âåTÄÃ°§Ð§7FF–2fö–Bv–FU÷F¶R†6öç7BöG&uö–G‚¦F’’²–b†u÷v–FUöâÂt”DUôÔ‚bbv–FUöf–æB†F’’’u÷v–FU¶u÷v–FUöâ²µÒÒ‡v–FUöVçB—¶F’Óæ–'VbÂF’Óæ6÷VçBÂÓ²Ð§7FF–2fö–Bv–FU÷'Vâ†6öç7BöG&uö–G‚¦F’’°¢7FF–2–CÄÕDÄ6ö×WFU—VÆ–æU7FFSâ3²7FF–2–CÄÕDÃD&wVÖVçEF&ÆSâC²7FF–2–çBf–ÆVC°¢–b†f–ÆVB’&WGW&ã°¢–b‚B’²å4W'&÷"¦RÒæ–Ã²–CÄÕDÄÆ–'&'“âÆ–"Ò†÷7EöÆ–"‚'v–FR"ÂfR“²–CÄÕDÄgVæ7F–öãâbÒÆ–"ò¶Æ–"æWtgVæ7F–öåv—F„æÖS¤'v–FU÷S‚%Ò¢æ–Ã°¢2Òbò¶uöFWbæWt6ö×WFU—VÆ–æU7FFUv—F„gVæ7F–öã¦bW'&÷#¢fUÒ¢æ–Ã°¢ÕDÃD&wVÖVçEF&ÆTFW67&—F÷"¦BÒ´ÕDÃD&wVÖVçEF&ÆTFW67&—F÷"æWuÓ²BæÖ„'VffW$&–æD6÷VçBÒ²BÒ2ò&V5÷F&ÆR…¶uöFWbæWt&wVÖVçEF&ÆUv—F„FW67&—F÷#¦BW'&÷#¢fUÒ’¢æ–Ã°¢–b‚B’²f–ÆVBÒ²ÔÄôr‚%µ$TEÒ‚Ö&—B–æFW‚uRW‡ç6–öã¢6ææ÷B7&VFRF†R6ö×WFR—VÆ–æRW2‡F†R5RW‡æG2F†VÒ2&Vf÷&R’"ÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“²&WGW&ã²ÒÐ¢–b†u÷v–FUöWö6‚Òu÷w%öWö6‚ÇÂu÷v–FU÷7V"Òu÷7V%÷6W&–ÂÇÂu÷v–FUöâât”DUôÔ‚Ò‚’²u÷v–FUöâÒ²u÷v–FUöWö6‚Òu÷w%öWö6ƒ²u÷v–FU÷7V"Òu÷7V%÷6W&–Ã²Ð¢–çBãÒu÷v–FUöã²v–FU÷F¶R†F’“°¢–b†uöW†V5öæW‡Bbbu÷&WEöâÃÒ‚’²òòÆFW"G&w2–â&WÆ’÷&FW"†7W'&VçBÆ—7B(i"6ÆÆW'2(i"F†R&W7BöbF†—27V&Ö—76–öâ’ÂWFòF†RæW‡B'VffW"w&—FP¢g¥ö7W'6÷"7W"Ò²çÒuöW†V5öæW‡BÂç&WEöâÒu÷&WEöçÓ²ÖVÖ7’†7W"ç&WBÂu÷&WBÂ6—¦Vöb‚¦u÷&WB’¢‡6—¦U÷B–u÷&WEöâ“°¢6öç7BV–çC…÷B§²÷ö†G"ƒ²6öç7Bfö–B¦°¢f÷"†–çBwV&BÒ²wV&BÂ#bbu÷v–FUöâÂt”DUôÔ‚bbg¥öæW‡B‚f7W"ÂgÂf‚Âf“²wV&B²²’°¢–b†‚æ÷ÓÒõôD•5D4‚ÇÂ‚æ÷ÓÒõô4õ•ô#$"ÇÂ‚æ÷ÓÒõõUDDUõT"ÇÂ‚æ÷ÓÒõô4ÄT%ô%TbÇÂ‚æ÷ÓÒõô4õ•õC$"’'&V³°¢öG&uö–G‚—C²6öç7BöG&uö–G‚§‚Ò‚æ÷ÓÒõôE$uô”E‚òG&uö–G…÷&V2†Â‚ç6—¦RÂf—B’¢åTÄÃ°¢–b‡‚bbv–FU÷vçFVB‡‚’’v–FU÷F¶R‡‚“°¢Ð¢Ð¢–çBâÒu÷v–FUöâÒã²–b†âÃÒ’&WGW&ã°¢–b†uöVæ2’²u÷v–FU÷7Æ—G2²³²B‚$ÒVÆÇR‚Ö&—B–æFW‚uRW‡ç6–öã¢&VæFW"727Æ—B†W&R"Âu÷7FE÷72“²'öVæB‚“²Ð¢–b†uö&Æ—B’²Væ5ö%öVæB†uö&Æ—B“²uö&Æ—BÒæ–Ã²Ð¢G—VFVb7G'V7B²V–çCcE÷B7&2ÂG7C²V–çC3%÷BâÂ6†–gC²Òv¦ö#°¢–CÄÕDÄ'VffW#â¦#²6—¦U÷B¦òÒ&VæöÆÆö2‚‡6—¦U÷B–â¢6—¦Vöb‡v¦ö"’Âf¦"“²v¦ö"¦§Ò‡v¦ö"¢’‚‡V–çC…÷B¢–¦"æ6öçFVçG2²¦ò“°¢–CÄÕDÃD6ö×WFT6öÖÖæDVæ6öFW#â6RÒVæ5ö5ö&Vv–â„åTÄÂ“²¶6R6WD&wVÖVçEF&ÆS¦EÓ²¶6R6WD6ö×WFU—VÆ–æU7FFS§5Ó°¢f÷"†–çB’Ò²’Âã²’²²’°¢v–FUöVçB¦RÒfu÷v–FU¶ã²•Ó²fö–B¦Ö#²V–çCcE÷BÖòÂ7&3²RÓæwRÒ°¢–b‚×FÅöwUöFG"†RÓæ–'VbÂ‡V–çCcE÷B–RÓæ6÷VçBÂfÖ"ÂfÖòÂg7&2’’6öçF–çVS²òòF†RG&r&W÷'G2—B…$TEôôä4R#"¢–CÄÕDÄ'VffW#âö#²6—¦U÷BöòÒ&VæöÆÆö2‚‡6—¦U÷B–RÓæ6÷VçB¢"Âfö"“°¢§¶•ÒÒ‡v¦ö"—·7&2bãWVÆÂÂö"æwTFG&W72²öòÂ‡V–çC3%÷B–RÓæ6÷VçBÂ‡V–çC3%÷B’‡7&2bR—Ó°¢¶B6WDFG&W73¦¦"æwTFG&W72²¦ò²‡V–çCcE÷B–’¢6—¦Vöb‡v¦ö"’D–æFWƒ£Ó°¢¶6RF—7F6…F‡&VFw&÷W3¤ÕDÅ6—¦TÖ¶R‚‚„å5T–çFVvW"–RÓæ6÷VçB²c2’òcBÂÂ’F‡&VG5W%F‡&VFw&÷W¤ÕDÅ6—¦TÖ¶RƒcBÂÂ•Ó°¢RÓæwRÒö"æwTFG&W72²öó°¢Ð¢Væ5ö5öVæB†6R“°¢&w5öF—'G•öÆÂ‚“°¢u÷v–FUöG&w2³Ò‡V–çCcE÷B–ã°¢–b‚u÷v–FU÷'Vç2²²’ÔÄôr‚#‚Ö&—B–æFW‚uRW‡ç6–öã¢f—'7BF–ÖRÂVBG&w2‡F†R–æF–6W2†fRuR×6–FRw&—FW2–âF†—27V&Ö—76–öã²6ö×WFR72W‡æG2F†VÒFòb&—G2&Vf÷&RF†R&VæFW"72’"Ââ“°¢VÇ6R–b†u÷v–FU÷'Vç2R3cÓÒ’ÔÄôr‚#‚Ö&—B–æFW‚uRW‡ç6–öã¢VÆÇRF–ÖW2ÂVÆÇRG&w2Â&VæFW"727Æ—BVÆÇRF–ÖW2"Â‡Vç6–væVBÆöærÆöær–u÷v–FU÷'Vç2Â‡Vç6–væVBÆöærÆöær–u÷v–FUöG&w2Â‡Vç6–væVBÆöærÆöær–u÷v–FU÷7Æ—G2“°§Ð¢òòF–væ÷7F–2DÕôådåõT%ô4„T4³Ó¢V6‚6†F÷rÖFRg&öÒÖVÖ÷'’‡V'övWB’—26†V6¶VBv–ç7BF†RuRâF†R72VæG2ÂF†RFVfW'&VBWFFW2ÆæBÂF†R&WÆ’v—G2f÷"F†RuR†2â–æF—&V7BÖ6÷Vç@¢òò×VÇF’ÖG&rFöW2’ÂæBF†R'—FW2F†RG&rv÷VÆB†fR&VBg&öÒF†R6†F÷r&R6ö×&VBv—F‚ÖVÖ÷'’Âv†–6‚æ÷r†öÆG2W†7FÇ’v†BF†RuR&VG2âF†RG&rF†VâvöW2öâ2v—F†÷WBF†R6†F÷w0§7FF–2fö–BÖF•÷v—B‡fö–B“°§7FF–2V–çCcE÷Bu÷V'µ÷6ÖRÂu÷V'µöF–fc°§7FF–2–çBV'ö6†V6µööâ‡fö–B’²7FF–2–çBbÒÓ²–b‡bÂ’²bÒvWFVçb‚$DÕôådåõT%ô4„T4²"’ÒåTÄÃ²–b‡b’ÔÄôr‚$DÕôådåõT%ô4„T4³¢WfW'’Væ–f÷&Ò6†F÷rÖFRg&öÒÖVÖ÷'’—26ö×&VBv—F‚F†RuRw2ÖVÖ÷'’gFW"F†RWFFW2ÆæB‡6Æ÷r’"“²Ò&WGW&âc²Ð§7FF–2fö–BV'ö6†V6²‡V–çCcE÷BÆòÂV–çCcE÷B†’Â6öç7BV–çC…÷B§6‚’°¢V&5÷v—B†u÷V'÷&VG’“²òòF†R6†F÷rV'övWB§W7B&WGW&æVBÖ’7F–ÆÂ&Rf–ÆÆVB'’F†R6÷’v÷&¶W ¢6—¦U÷BâÒ‡6—¦U÷B’††’ÒÆò“²V–çC…÷B§F×ÒÖÆÆö2†â“²ÖVÖ7’‡F×Â6‚Ââ“°¢ÖF•÷v—B‚“°¢6öç7BV–çC…÷B¦ÒÒ†6öç7BV–çC…÷B¢’‡V–çGG%÷B–Æó²6—¦U÷B&BÒÂf—'7BÒ°¢f÷"‡6—¦U÷B’Ò²’Âã²’²²’–b‡F×¶•ÒÒÕ¶•Ò’²–b‚&B²²’f—'7BÒ“²Ð¢–b†&B’²u÷V'µöF–fb²³²–b†u÷V'µöF–fbÃÒ’ÔÄôr‚%µ$TEÒVæ–f÷&Ò6†F÷r6†V6³¢²R6ÆÇ‚²W§UÒF–ffW'2g&öÒF†RuRw2ÖVÖ÷'’–âW§R'—FW2Âf—'7BB²W§R‡6†F÷rS'‚ÂÖVÖ÷'’S'‚’g&ÖRVÆÇR72VÆÇR"À¢‡Vç6–væVBÆöærÆöær–ÆòÂâÂ&BÂf—'7BÂF×¶f—'7EÒÂÕ¶f—'7EÒÂ‡Vç6–væVBÆöærÆöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR’Âu÷7FE÷72“²Ð¢VÇ6Ru÷V'µ÷6ÖR²³°¢–b‚†u÷V'µ÷6ÖR²u÷V'µöF–fb’R#ÓÒ’ÔÄôr‚'Væ–f÷&Ò6†F÷r6†V6³¢VÆÇRWVÂÂVÆÇRF–ffW&VçB"Â‡Vç6–væVBÆöærÆöær–u÷V'µ÷6ÖRÂ‡Vç6–væVBÆöærÆöær–u÷V'µöF–fb“°¢g&VR‡F×“°§Ð¢òòv†WF†W"F†—2G&ræVVG2Ö÷&Rg&öÒF†R–æÆ–æRWFFW2FVfW'&VB–âF†R÷Vâ72F†âF†RVæ–f÷&Ò6†F÷w2v—fR‡6VR×FÅ÷7V%÷WFFR“¢Væ–f÷&Ò&–æF–ærF†W’6÷fW"öæÇ’–â'Bv†÷6R&W7@¢òò6ææ÷B&R&VBg&öÒÖVÖ÷'’‡V'övWC²&–æF–ær6÷VçG2öæÇ’2f"2F†R6†FW"6â&VB—BÂ6%öÆVâ’Â&–æFÆW72†æFÆR–â7V6‚'BÂ÷"F†V—"FF&VB2fW'F–6W2Â–æF–6W2Â–æF—&V7@¢òò&wVÖVçG2÷"7F÷&vR'VffW"âF†VâF†R72VæG2f—'7BæBF†RWFFW2ÆæB&Vf÷&RF†RG&rÂ2&Vf÷&P§7FF–2–çBV&Eö6öæfÆ–7E÷6‚„çe6†FW"§6‚Â–çB7B’²òòöæR7FvRw2&–æF–æw2†Væ6öFUö&w2&–æG2F†RfW'FW‚Âg&vÖVçBæB6ö×WFR7FvW2¢f÷"†–çB’Ò²’Â6‚Óç&bææ#²’²²’²6öç7Bçfå÷&&–æB¦"Òg6‚Óç&bæ%¶•Ó°¢–b†"Óæ¶–æBÓÒådåõ$%ôtÔTÒ’²–b‡V&Eö†—G2ƒÂT”åCcEôÔ‚’’&WGW&â²6öçF–çVS²Òòò&VG2ç’ööÀ¢–b†"Óç&VrãÒb’6öçF–çVS°¢6öç7Bö&–æEö'Vb§RÒ"Óæ¶–æBÓÒådåõ$%õ4%Teòe2ç76&õ·7EÕ¶"Óç&VuÒ¢e2çV&õ·7EÕ¶"Óç&VuÓ°¢–b‚RÓæFG"’6öçF–çVS°¢–b†"Óæ¶–æBÓÒådåõ$%ô4%Tb’²V–çCcE÷BrÂVÂÒ6%öÆVâ‡6‚Â"Óç&VrÂRÓç6—¦R“²6öç7BV–çC…÷B¦3°¢–b‡V&Eö†—G2‡RÓæFG"ÂRÓæFG"²VÂ’bbV&Eö6÷fW'2‡RÓæFG"ÂRÓæFG"²VÂ’’°¢–b‚V'övWB‡RÓæFG"ÂRÓæFG"²VÂÂfrÂf2’’²u÷V'÷&VgW6VB²³²&WGW&â²Ð¢–b‡V'ö6†V6µööâ‚’’²V'ö6†V6²‡RÓæFG"ÂRÓæFG"²VÂÂ2“²&WGW&â²ÒÒÐ¢VÇ6R–b†"Óæ¶–æBÓÒådåõ$%õDU…RÇÂ"Óæ¶–æBÓÒådåõ$%õ4ÕRÇÂ"Óæ¶–æBÓÒådåõ$%õ4%TeR’²V–çCcE÷B‚ÒRÓæFG"²†"Óæâò¢"Óæöfb’ÂâÒ"Óæâò‡VÆÂ¢"Óæâ¢"Óæ¶–æBÓÒådåõ$%õ4%TeRò‚¢C²V–çCcE÷Bs²6öç7BV–çC…÷B¦3²òòâ'&“¢—G2†æFÆW2B†°¢–b‡V&Eö†—G2†‚Â‚²â’bbV&Eö6÷fW'2†‚Â‚²â’bbV'övWB†‚Â‚²âÂfrÂf2’’²u÷V'÷&VgW6VB²³²&WGW&â²ÒÐ¢VÇ6R–b†"Óæ¶–æBÓÒådåõ$%õ4%TeÇÂ"Óæ¶–æBÓÒådåõ$%õT$õ’²–b‡V&Eö†—G2‡RÓæFG"ÂRÓæFG"²‡RÓç6—¦RòRÓç6—¦R¢’’’&WGW&â²ÒÐ¢&WGW&â°§Ð§7FF–2–çBV&Eö6öæfÆ–7B„çe&ör§"Â6öç7BöG&r¦GrÂ6öç7BöG&uö–G‚¦F’’°¢f÷"†–çB5òÒ²5òÂc²5ò²²’–b…2çf'Ve·5õÒæFG&W72bbV&Eö†—G2…2çf'Ve·5õÒæFG&W72Â2çf'Ve·5õÒæFG&W72²2çf'Ve·5õÒç6—¦R’’&WGW&â°¢–b†F’bbF’Óæ–'VbbbV&Eö†—G2†F’Óæ–'VbÂF’Óæ–'Vb²‡V–çCcE÷B–F’Óæ6÷VçB¢B’’&WGW&â°¢V–çCcE÷B–æBÒGròGrÓæ–æB¢F’òF’Óæ–æB¢²–b†–æBbbV&Eö†—G2†–æBÂ–æB²3"’’&WGW&â°¢f÷"†–çB7BÒ²"bb7BÂ#²7B²²’–b‡"Óç6…·7EÒbbV&Eö6öæfÆ–7E÷6‚‡"Óç6…·7EÒÂ7B’’&WGW&â°¢&WGW&â°§Ð¢òòF†R6ÖRf÷"F—7F6ƒ¢WFFW2¶WB7BF†V—"72„DÕôådåõT%ôÄ¥’’ÆæB&Vf÷&RF—7F6‚F†B&VG2÷"w&—FW2F†V—"ÖVÖ÷'’F‡&÷Vv‚7F÷&vR'VffW"÷"—G2–æF—&V7B&wVÖVçG2Â÷"&VG2¢òò'BöbVæ–f÷&Ò&–æF–ærF†R6†F÷w26ææ÷Bv—fR†VÇ6RF†W’v÷VÆBÆæBgFW"—BÂBF†RæW‡B6÷’–çFòF†RVæ6öFW"—B6†&W2ÂF—7F6…öVæ2’âVæ–f÷&Ò&VG2F†W’6÷fW"F¶R6†F÷w2†Væ6öFUö&w2§7FF–2–çBV&Eö6öæfÆ–7Eö72„çe6†FW"¦72ÂV–çCcE÷B–æB’²&WGW&â†–æBbbV&Eö†—G2†–æBÂ–æB²"’’ÇÂV&Eö6öæfÆ–7E÷6‚†72ÂR“²Ð¢òòG&w2öbâ–æF—&V7BÖ6÷VçB×VÇF’ÖG&r††ÆUöçfâæ2çfä6öÖÖæD'VffW$×VÇF”G&tVÆVÖVçG4–æF—&V7D6÷VçB“¢G&r’—2ÖFRöæÇ’v†–ÆR’ÂF†RS3"6÷VçBF†RvÖRw2uR7VÆÆ–ærw&÷FRV&Æ–W"–âF†P¢òò7V&Ö—76–öââÖWFÂ6ææ÷B6¶—G&w2'’6÷VçB–âuRÖVÖ÷'’âÖ÷7B7V6‚G&w2&RÖ6¶VBöâF†RuR†ÖG…÷'Vâ&VÆ÷r“²F†R&W7B‡VG2Âfç2æB‚Ö&—B–æF–6W2Âv†÷6R&V6÷&G2—†5÷'Và¢òò6öçfW'G2ÂæB&V6÷&G2÷"6÷VçG2–âæòÖWFÂ'VffW"’v—B†W&S¢&Vf÷&RF†Rf—'7B7V6‚G&rgFW"uRw&—FW2†F—7F6‚Â6÷’ÂWFFR÷"f–ÆÂ6–æ6RF†RÆ7Bv—BÂu÷w%öWö6ƒ²÷"æWp¢òò7V&Ö—76–öâ’F†R&WÆ’VæG2F†R&VæFW"72Â6öÖÖ—G2v†B—B†2Væ6öFVBæBv—G2f÷"—BÂF†Vâ&VG2F†R6÷VçBöâF†R5S²ÆFW"G&w2öbF†Rw&÷WW6RF†R6ÖRfÇVP§7FF–2–CÄÕDÅ6†&VDWfVçCâuöÖF•öWc²7FF–2V–çCcE÷BuöÖF•÷bÂuöÖF•÷v—G2ÂuöÖF•öWö6‚ÒãVÆÂÂuöÖF•÷7V"ÂuöÖF•öFG#²7FF–2V–çC3%÷BuöÖF•ö6÷VçC°§7FF–2fö–BÖF•÷v—B‡fö–B’°¢'öVæB‚“²V&EöÆæB‚“°¢–b†uö&Æ—B’²Væ5ö%öVæB†uö&Æ—B“²uö&Æ—BÒæ–Ã²Ð¢WöfÇW6‚‚“²&W5ö6öÖÖ—B‚“°¢¶uö6"VæD6öÖÖæD'VffW%Ó°¢&V5ö6öÖÖ—B†uö6"Âæ–Â“°¢–b‚uöÖF•öWb’uöÖF•öWbÒ¶uöFWbæWu6†&VDWfVçEÓ°¢&V5÷6–væÂ†uöÖF•öWbÂ²¶uöÖF•÷b“²&V5öG&–â‚“°¢V–çCcE÷BsÒw5÷C‚“²–çBö²Ò¶uöÖF•öWbv—EVçF–Å6–væÆVEfÇVS¦uöÖF•÷bF–ÖV÷WDÕ3£Ó²–b‡w5ööâ‚’’w5÷WB…u5ôÔD’ÂsÂ“°¢–b‚ö²’$TEôôä4Rƒ“rÂ&–æF—&V7BÖ6÷VçB×VÇF’ÖG&s¢F†RuRF–Bæ÷Bf–æ—6‚v—F†–â3²F†R6÷VçB—2&VBç—v’"“°¢VÇ6RG'uöG&–æVB‚“°¢uö6"Ò&V5ö6%öæWr‚“°¢¶uö6"&Vv–ä6öÖÖæD'VffW%v—F„ÆÆö6F÷#¦uöÆÆö3E¶uöÆÆö5ö7W%ÕÓ°¢–b‚uöÖF•÷v—G2²²’ÔÄôr‚&–æF—&V7BÖ6÷VçB×VÇF’ÖG&s¢f—'7Bv—Bf÷"F†RuR×w&—GFVâ6÷VçBBg&ÖRVÆÇR72VÆÇR"Â‡Vç6–væVBÆöærÆöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR’Âu÷7FE÷72“°§Ð§7FF–2–çBÖF•öÆ—fR‡V–çCcE÷B6çBÂ–çB’’°¢–b†6çBÒuöÖF•öFG"ÇÂuöÖF•öWö6‚Òu÷w%öWö6‚ÇÂuöÖF•÷7V"Òu÷7V%÷6W&–Â’°¢òòæòw&—FR6–æ6RF†—27V&Ö—76–öâw2Æ7BG&–â6â†fR6†ævVBF†R6÷VçB†G'u÷fÆ–B“¢&VB—Bv—F†÷WBæ÷F†W ¢–çBg&W6‚ÒG'u÷fÆ–B‚’bbG'uö†—G2†6çBÂB’bbçfå÷7'6UöÆ–2†6çBÂ6çB²B“°¢–b‚g&W6‚bb†uöÖF•öWö6‚Òu÷w%öWö6‚ÇÂuöÖF•÷7V"Òu÷7V%÷6W&–Â’’²ÖF•÷v—B‚“²uöÖF•öWö6‚Òu÷w%öWö6ƒ²uöÖF•÷7V"Òu÷7V%÷6W&–Ã²Ð¢V–çC3%÷BbÒ°¢–b†çfåö†÷7E÷&VF&ÆR†6çBÂB’’ÖVÖ7’‚gbÂ†6öç7Bfö–B¢’‡V–çGG%÷B–6çBÂB“°¢VÇ6R$TEôôä4Rƒ“‚Â&–æF—&V7BÖ6÷VçB×VÇF’ÖG&s¢F†R6÷VçBBR6ÆÇ‚—2–ââVæÖVBvRöb7'6RööÂÂæ÷F†–ærG&vâ"Â‡Vç6–væVBÆöærÆöær–6çB“°¢uöÖF•öFG"Ò6çC²uöÖF•ö6÷VçBÒc°¢Ð¢&WGW&â‡V–çC3%÷B–’ÂuöÖF•ö6÷VçC°§Ð¢òòÒÒÒÒÒÒÒÒÒÒ–æF—&V7BÖ6÷VçB×VÇF’ÖG&w2Ö6¶VBöâF†RuRÒÒÒÒÒÒÒÒÒÐ¢òòF†Rv—B&÷fR†VÆBF†R7V&Ö—GF–ærF‡&VBf÷"F†RuRBWfW'’uRÖG&—fVâ73¢–âÖöç7FW"‡VçFW"&—6Rw2¶×W&f–ÆÆvRÂBöbF†R#×27V&Ö—76–öâFöö²öââ•†öæRvW&RF†Bv—@¢òòƒb×2öbv÷&²’ÂæBF†RvÖR&âB3n(	3SBg2â–ç7FVB6ö×WFR726÷–W2V6‚7V6‚G&rw2&V6÷&B–çFòF†Rg&ÖR&VæÂ÷"¦W&÷2f÷"G&r7BF†R6÷VçBÂæBF†RG&r&VG2F†R6÷“ ¢òò7BF†R6÷VçB—BG&w2æ÷F†–ærÂæBF†R5RæWfW"v—G2â2–â—†5÷'VâÂ6ö×WFR726ææ÷B'Vâ–ç6–FR&VæFW"73¢BF†Rf—'7B7V6‚G&rv—F‚æò6÷’–WBÂÖG…÷'Vâ6÷–W2—BæBWfW'¢òò7V6‚G&rgFW"—B–â&WÆ’÷&FW"WFòF†RæW‡Bw&—FRF†B6÷VÆB6†ævR&V6÷&B÷"6÷VçB†F—7F6‚Â6÷’Âf–ÆÂ÷"FW‡GW&R(i"'VffW"6÷’Âv†–6‚'V×uöÖG…öWö6ƒ²Væ–f÷&ÒWFFRöæÇ¢òòv†Vâ—B6÷fW'2öæS¢uöÖG…÷7F÷’Â6òöæÇ’G&rÖWB–ç6–FRâ÷Vâ&VæFW"727Æ—G2—Bâ6÷–W2&Rf÷VæB'’&V6÷&BæB6÷VçBFG&W72–ââ÷VâÖFG&W76–ærF&ÆRÂfÆ–Bf÷"öæRuöÖG…÷Fp¢òò„¶×W&f–ÆÆvRG&w2F†R6ÖR&V6÷&G2v—F‚Gvò6÷VçG2&WGvVVâF†R6ÖRw&—FW3¢¶W–VB'’F†R&V6÷&BÆöæRÂF†R6V6öæB×VÇF’ÖG&rf÷VæBæò6÷’æBv—FVBf÷"F†RuRV6‚g&ÖR¢6FVf–æRÔE…õ4ÄõE2c3ƒBòò÷vW"öbGvó²66âf–ÆÇ2BÖ÷7BV'FW §G—VFVb7G'V7B²V–çCcE÷B–æBÂ6çBÂ÷WBÂFs²ÒÖG…öVçC²òò÷WC¢uRFG&W72öbF†R6÷’Âv†VâF†R&V6÷&B÷"6÷VçB—2–âæòÖWFÂ'VffW"‡F†÷6RG&w2v—BÂÖF•öÆ—fR§7FF–2ÖG…öVçBuöÖG…´ÔE…õ4ÄõE5Ó²7FF–2–çBuöÖG…öã²7FF–2V–çCcE÷BuöÖG…÷FrÒÂuöÖG…öWÒãVÆÂÂuöÖG…÷7V"ÂuöÖG…÷'Vç2ÂuöÖG…÷7Æ—G2ÂuöÖG…öG&w2ÂuöÖG…öf7C²7FF–26öç7BV–çC…÷B¦uöÖG…÷7F÷°¢òòF†R¶W&æVÂÖG…öÖ6³¢FÒövg‚ö×6ÂöÖG‚æÖWFÀ§7FF–2–çBÖG…÷vçFVB†6öç7BöG&r¦GrÂ6öç7BöG&uö–G‚¦F’’²&WGW&âGròGrÓæÖF•ö6çBbb—†5÷vçFVB†GrÂåTÄÂ’¢F’ÓæÖF•ö6çBbb—†5÷vçFVB„åTÄÂÂF’“²Ð§7FF–2ÖG…öVçB¦ÖG…÷6Æ÷B‡V–çCcE÷B–æBÂV–çCcE÷B6çB’²òò—G2VçG'’–âF†—2FrÂ÷"F†Rg&VR6Æ÷Bf÷"—@¢f÷"‡V–çCcE÷B‚Ò‚†–æBâ6çB¢„CdS„dT#ƒccS”dC“7VÆÂ’¢ƒ”S3ss”#“tcDt3WVÆÂ’ãâS³²‚Ò†‚²’b„ÔE…õ4ÄõE2Ò’’²ÖG…öVçB¦RÒfuöÖG…¶…Ó²–b†RÓçFrÒuöÖG…÷FrÇÂ†RÓæ–æBÓÒ–æBbbRÓæ6çBÓÒ6çB’’&WGW&âS²Ð§Ð§7FF–2–çBÖG…öf–æB‡V–çCcE÷B–æBÂV–çCcE÷B6çBÂV–çCcE÷B¦÷WB’²òò(i"v†VâÖG…÷'Vâ†2F¶VâF†—2G&r‚¦÷WC¢—G26÷’Â÷"¢–b†uöÖG…öWÒuöÖG…öWö6‚ÇÂuöÖG…÷7V"Òu÷7V%÷6W&–Â’&WGW&â°¢6öç7BÖG…öVçB¦RÒÖG…÷6Æ÷B†–æBÂ6çB“²–b†RÓçFrÒuöÖG…÷Fr’&WGW&â°¢¦÷WBÒRÓæ÷WC²&WGW&â°§Ð§7FF–2fö–BÖG…÷'Vâ†6öç7BöG&r¦GrÂ6öç7BöG&uö–G‚¦F’’°¢7FF–2–CÄÕDÄ6ö×WFU—VÆ–æU7FFSâ3²7FF–2–CÄÕDÃD&wVÖVçEF&ÆSâC²7FF–2–çBf–ÆVC°¢–b†f–ÆVB’&WGW&ã°¢–b‚B’²å4W'&÷"¦RÒæ–Ã²–CÄÕDÄÆ–'&'“âÆ–"Ò†÷7EöÆ–"‚&ÖG‚"ÂfR“²–CÄÕDÄgVæ7F–öãâbÒ¶Æ–"æWtgVæ7F–öåv—F„æÖS¤&ÖG…öÖ6²%Ó°¢2Òbò¶uöFWbæWt6ö×WFU—VÆ–æU7FFUv—F„gVæ7F–öã¦bW'&÷#¢fUÒ¢æ–Ã°¢ÕDÃD&wVÖVçEF&ÆTFW67&—F÷"¦BÒ´ÕDÃD&wVÖVçEF&ÆTFW67&—F÷"æWuÓ²BæÖ„'VffW$&–æD6÷VçBÒ²BÒ2ò&V5÷F&ÆR…¶uöFWbæWt&wVÖVçEF&ÆUv—F„FW67&—F÷#¦BW'&÷#¢fUÒ’¢æ–Ã°¢–b‚B’²f–ÆVBÒ²ÔÄôr‚%µ$TEÒ–æF—&V7BÖ6÷VçB×VÇF’ÖG&rÖ6¶–æs¢6ææ÷B7&VFRF†R6ö×WFR—VÆ–æRW2‡F†÷6RG&w2v—Bf÷"F†RuR’"ÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“²&WGW&ã²ÒÐ¢–b†uöÖG…öWÒuöÖG…öWö6‚ÇÂuöÖG…÷7V"Òu÷7V%÷6W&–ÂÇÂuöÖG…öââÔE…õ4ÄõE2òB’²uöÖG…÷Fr²³²uöÖG…öâÒ²uöÖG…öWÒuöÖG…öWö6ƒ²uöÖG…÷7V"Òu÷7V%÷6W&–Ã²Ð¢VçVÒ²Ô‚ÒÔE…õ4ÄõE2òBÓ°¢7FF–27G'V7B²ÖG…öVçB¦S²V–çC3%÷B’Âv÷&G3²ÒµÔ…Ó²–çBçÒ°¢V–çCcE÷BÆòÒãVÆÂÂ†’Ò²òòF†R‡VÆÂöbF†R&V6÷&G2æB6÷VçG2F¶VâÂf÷"F†RVæ–f÷&ÒWFFR6†V6°¢6FVf–æRÔE…õD´R„”äBÂ4åBÂ’Âr’Fò²ÖG…öVçB¦UòÒÖG…÷6Æ÷B‚„”äB’Â„4åB’“²–b†UòÓçFrÒuöÖG…÷FrbbçÂÔ‚’²¦UòÒ†ÖG…öVçB—²„”äB’Â„4åB’ÂÂuöÖG…÷FwÓ²uöÖG…öâ²³²À¢¶çÒæRÒUó²¶çÒæ’Ò‡V–çC3%÷B’„’“²¶ç²µÒçv÷&G2Ò…r“²ÆòÒÔ”â†ÆòÂÔ”â‚„”äB’Â„4åB’’“²†’ÒÔ‚††’ÂÔ‚‚„”äB’²B¢…r’Â„4åB’²B’“²ÒÒv†–ÆRƒ¢–b†Gr’ÔE…õD´R†GrÓæ–æBÂGrÓæÖF•ö6çBÂGrÓæÖF•ö’ÂB“²VÇ6RÔE…õD´R†F’Óæ–æBÂF’ÓæÖF•ö6çBÂF’ÓæÖF•ö’ÂR“°¢–b†uöW†V5öæW‡Bbbu÷&WEöâÃÒ‚’²òòÆFW"G&w2–â&WÆ’÷&FW"†7W'&VçBÆ—7B(i"6ÆÆW'2(i"F†R&W7BöbF†—27V&Ö—76–öâ¢g¥ö7W'6÷"7W"Ò²çÒuöW†V5öæW‡BÂç&WEöâÒu÷&WEöçÓ²ÖVÖ7’†7W"ç&WBÂu÷&WBÂ6—¦Vöb‚¦u÷&WB’¢‡6—¦U÷B–u÷&WEöâ“°¢6öç7BV–çC…÷B§²÷ö†G"ƒ²6öç7Bfö–B¦°¢f÷"†–çBwV&BÒ²wV&BÂ#bbçÂÔ‚bbg¥öæW‡B‚f7W"ÂgÂf‚Âf“²wV&B²²’°¢–b†‚æ÷ÓÒõôD•5D4‚ÇÂ‚æ÷ÓÒõô4õ•ô#$"ÇÂ‚æ÷ÓÒõô4ÄT%ô%TbÇÂ‚æ÷ÓÒõô4õ•õC$"’'&V³°¢–b†‚æ÷ÓÒõõUDDUõT"’²V–çCcE÷BBÂã²ÖVÖ7’‚fBÂÂ‚“²ÖVÖ7’‚fâÂ†6öç7BV–çC…÷B¢–²‚Â‚“°¢–çB†—BÒ²f÷"†–çB²Ò²BÂ†’bbÆòÂB²âbb²Âçbb†—C²²²²’²6öç7BÖG…öVçB¦RÒ¶µÒæS²†—BÒ†BÂRÓæ–æB²B¢¶µÒçv÷&G2bbRÓæ–æBÂB²â’ÇÂ†BÂRÓæ6çB²BbbRÓæ6çBÂB²â“²Ð¢–b††—B’²uöÖG…÷7F÷Ò²'&V³²Ò6öçF–çVS²Ð¢öG&rGC²öG&uö–G‚—C²6öç7BöG&r¦BÒ‚æ÷ÓÒõôE$ròG&u÷&V2†Â‚ç6—¦RÂfGB’¢åTÄÃ²6öç7BöG&uö–G‚§‚Ò‚æ÷ÓÒõôE$uô”E‚òG&uö–G…÷&V2†Â‚ç6—¦RÂf—B’¢åTÄÃ°¢–b†BbbÖG…÷vçFVB†BÂåTÄÂ’’ÔE…õD´R†BÓæ–æBÂBÓæÖF•ö6çBÂBÓæÖF•ö’ÂB“°¢VÇ6R–b‡‚bbÖG…÷vçFVB„åTÄÂÂ‚’’ÔE…õD´R‡‚Óæ–æBÂ‚ÓæÖF•ö6çBÂ‚ÓæÖF•ö’ÂR“°¢Ð¢Ð¢7VæFVbÔE…õD´P¢–b‚ç’&WGW&ã°¢–b†uöVæ2’²uöÖG…÷7Æ—G2²³²B‚$ÒVÆÇR–æF—&V7BÖ6÷VçB×VÇF’ÖG&rÖ6¶–æs¢&VæFW"727Æ—B†W&R"Âu÷7FE÷72“²'öVæB‚“²Ð¢–b†uö&Æ—B’²Væ5ö%öVæB†uö&Æ—B“²uö&Æ—BÒæ–Ã²Ð¢G—VFVb7G'V7B²V–çCcE÷B&V2Â6çBÂ÷WC²V–çC3%÷B’Âv÷&G3²ÒÖ¦ö#°¢–çBæ¢Ò†ç²c2’bãc3°¢–CÄÕDÄ'VffW#â#²6—¦U÷BòÒ&VæöÆÆö2‚‡6—¦U÷B–æ¢¢6—¦Vöb†Ö¦ö"’²‡6—¦U÷B–ç¢3"Âf"“²òòF†R¦ö'2‡FFVBFòv†öÆRF‡&VFw&÷W2’ÂF†VâF†R6÷–W23"'—FW2'@¢Ö¦ö"¦¢Ò†Ö¦ö"¢’‚‡V–çC…÷B¢–"æ6öçFVçG2²ò“²V–çCcE÷BörÒ"æwTFG&W72²ò²‡V–çCcE÷B–æ¢¢6—¦Vöb†Ö¦ö"“°¢ÖV×6WB†¢ÂÂ‡6—¦U÷B–æ¢¢6—¦Vöb†Ö¦ö"’“°¢f÷"†–çB²Ò²²Âç²²²²’²ÖG…öVçB¦RÒ¶µÒæS²fö–B¦Ö#²V–çCcE÷BÖòÂ&rÂ6s°¢–b‚×FÅöwUöFG"†RÓæ–æBÂB¢¶µÒçv÷&G2ÂfÖ"ÂfÖòÂg&r’ÇÂ×FÅöwUöFG"†RÓæ6çBÂBÂfÖ"ÂfÖòÂf6r’’6öçF–çVS°¢RÓæ÷WBÒör²3"¢‡V–çCcE÷B–³²¥¶µÒÒ†Ö¦ö"—·&rÂ6rÂRÓæ÷WBÂ¶µÒæ’Â¶µÒçv÷&G7Ó²Ð¢–CÄÕDÃD6ö×WFT6öÖÖæDVæ6öFW#â6RÒVæ5ö5ö&Vv–â„åTÄÂ“²¶6R6WD&wVÖVçEF&ÆS¦EÓ²¶B6WDFG&W73¦"æwTFG&W72²òD–æFWƒ£Ó²¶6R6WD6ö×WFU—VÆ–æU7FFS§5Ó°¢¶6RF—7F6…F‡&VFw&÷W3¤ÕDÅ6—¦TÖ¶R‚„å5T–çFVvW"–æ¢òcBÂÂ’F‡&VG5W%F‡&VFw&÷W¤ÕDÅ6—¦TÖ¶RƒcBÂÂ•Ó°¢Væ5ö5öVæB†6R“°¢&w5öF—'G•öÆÂ‚“°¢uöÖG…öG&w2³Ò‡V–çCcE÷B–ç°¢–b‚uöÖG…÷'Vç2²²’ÔÄôr‚&–æF—&V7BÖ6÷VçB×VÇF’ÖG&rÖ6¶–æs¢f—'7BF–ÖRÂVBG&w2‡F†V—"&V6÷&G2&R6÷–VBöâF†RuRÂ¦W&÷27BF†R6÷VçBÂ&Vf÷&RF†R&VæFW"72’"Âç“°¢VÇ6R–b†uöÖG…÷'Vç2R3cÓÒ’ÔÄôr‚&–æF—&V7BÖ6÷VçB×VÇF’ÖG&rÖ6¶–æs¢VÆÇRF–ÖW2ÂVÆÇRG&w2‚VÆÇRVæ6öFVBÆöæRÂÖG…öæW‡B’Â&VæFW"727Æ—BVÆÇRF–ÖW3²v—G2f÷"F†RuR×w&—GFVâ6÷VçBVÆÇR"À¢‡Vç6–væVBÆöærÆöær–uöÖG…÷'Vç2Â‡Vç6–væVBÆöærÆöær–uöÖG…öG&w2Â‡Vç6–væVBÆöærÆöær–uöÖG…öf7BÂ‡Vç6–væVBÆöærÆöær–uöÖG…÷7Æ—G2Â‡Vç6–væVBÆöærÆöær–uöÖF•÷v—G2“°§Ð¢òòF†RG&w2öböæR×VÇF’ÖG&r&R&V6÷&FVB&6²Fò&6²v—F‚F†R6ÖR7FFRÂæBöæÇ’F†Rf—'7BfWr&RÆ—fS¢V6‚öæR&WÆ–VB&–v‡BgFW"F†RöæR&Vf÷&R—B†æò÷&WGvVVâF†VÒÀ¢òòF†R6ÖR÷Vâ72’—2Væ6öFVBÆöæRÂv—F†÷WBF†R&W7BöbG&rw2v÷&²Âv†–6‚6÷7Bv†öÆRG&rf÷"V6‚Ö6¶VBöæR†&÷WBãR+W2öââÓB&òÂÃg&ÖR–â¶×W&f–ÆÆvR§7FF–27G'V7B²6öç7BV–çC…÷B¦æW‡C²6öç7Bfö–B¦Væ3²V–çCcE÷B6çBÂ–rÂ–ÂÂ73²–çB“²ÕDÅ&–Ö—F—fUG—R×²ÕDÄ–æFW…G—R—C²ÒuöÖG†3°§7FF–2fö–BÖG…ö6†–â‡V–çCcE÷B6çBÂ–çB’ÂÕDÅ&–Ö—F—fUG—R×ÂÕDÄ–æFW…G—R—BÂV–çCcE÷B–rÂV–çCcE÷B–Â’°¢uöÖG†2ÒuöFVfW%÷&WÆ––ærÇÂuöW†V5öæW‡Bò‡G—Vöb†uöÖG†2’—³Ò¢‡G—Vöb†uöÖG†2’—¶uöW†V5öæW‡BÂ…õö'&–FvR6öç7Bfö–B¢–uöVæ2Â6çBÂ–rÂ–ÂÂu÷7FE÷72Â’Â×Â—GÓ°§Ð§7FF–2–çBÖG…öæW‡B†6öç7BöG&r¦GrÂ6öç7BöG&uö–G‚¦F’’°¢V–çCcE÷B6çBÒGròGrÓæÖF•ö6çB¢F’ÓæÖF•ö6çBÂ÷WC²–çB’ÒGròGrÓæÖF•ö’¢F’ÓæÖF•ö“°¢–b‚uöÖG†2ææW‡BÇÂuöÖG†2ææW‡BÒuöW†V5ö7W"ÇÂuöVæ2ÇÂuöÖG†2æVæ2Ò…õö'&–FvR6öç7Bfö–B¢–uöVæ2ÇÂuöÖG†2ç72Òu÷7FE÷72ÇÂuöÖG†2æ6çBÒ6çBÇÂuöÖG†2æ’²Ò¢ÇÂuöFVfW%÷&WÆ––ærÇÂ76GV×‚’ÇÂÖG…öf–æB†GròGrÓæ–æB¢F’Óæ–æBÂ6çBÂf÷WB’ÇÂ÷WB’&WGW&â°¢–b†Gr’Væ5÷%öG&uö–æF—&V7B†uöVæ2ÂuöÖG†2æ×Â÷WB“²VÇ6RVæ5÷%öG&uö–æFW†VEö–æF—&V7B†uöVæ2ÂuöÖG†2æ×ÂuöÖG†2æ—BÂuöÖG†2æ–rÂuöÖG†2æ–ÂÂ÷WB“°¢u÷7FEöG&r²³²uöÖG…öf7B²³²uöÖG†2ææW‡BÒuöW†V5öæW‡C²uöÖG†2æ’Ò“°¢&WGW&â°§Ð¢6–æ6ÇVFR&çfåöfVVF&6²æ–æ2  ¢6–æ6ÇVFR&çfåövVöÖWG'’æ–æ2 §7FF–2–çBvVõöG&v–ær‡fö–B’²&WGW&âuövVõ÷&öw&ÒÒæ–ÂÇÂuövVõöÖW6‚æöã²Ð ¢òòÆ§’6ÆV'2â6WBöbF&vWG2F†BvWG2öæÇ’6ÆV'2&Vf÷&RF†RvÖR6WG2÷F†W"F&vWG2W6VBFò÷Vâ72§W7BFòÆæBF†VÒ‡'öfÇW6‚’Â7F÷&–ærF†R6ÆV&VBGF6†ÖVçG2ÂæBF†RæW‡@¢òò72F†BGF6†VBF†VÒÆöFVBF†VÒ&6²â'&VF‚öbF†Rv–ÆB6ÆV'2—G2rÖ'VffW"öæRF&vWBBF–ÖR†FWF‚v—F‚F†RÆ–v‡F–ærF&vWBÂF†VâÆ&VFòÂæ÷&ÖÇ2ÂF†Rf÷W'F‚’ÂF†VâGF6†W2ÆÀ¢òòf—fS¢f÷W"76W27F÷&VB#rÔ"öb6ÆV&VBÖVÖ÷'’Bcƒ“æBF†RrÖ'VffW"72ÆöFVB—Bv–ââ7V6‚6ÆV'2æ÷rv—B†uöÆ2“¢F†RæW‡B72F†BGF6†W2F†R6ÖRFW‡GW&RÆWfVÂæ@¢òòÆ–W"F‡&÷Vv‚F†R6ÖRf–WrF¶W2F†R6ÆV"2—G2ÆöB7F–öââF†R&W7BÆæB2&Vf÷&RÂ–â÷&FW"ÂV6‚–â72öb—G2÷vâ†Æ5öÆæB’Â&Vf÷&Rç—F†–ærVÇ6R6÷VÆB6VRF†V—"FW‡GW&W3¢¢òòG&rv†÷6R&öw&Ò6×ÆW2öæRÂF†R72öbG&rF†BFöW2æ÷BGF6‚—B†&Vf÷&RF†B72÷Vç2’ÂæBç’÷W&F–öâ÷F†W"F†âF&vWG2Â6ÆV'2Â7FFRÂVæ–f÷&ÒWFFW2æB'VffW ¢òò6÷–W2÷"f–ÆÇ2ÂæBF†RVæBöb6öÖÖæBÆ—7B†æ÷Böb7V"ÖÆ—7B—B6ÆÇ2’âæWvW"6ÆV"öbF†R6ÖR7V'&W6÷W&6R&WÆ6W2âöÆFW"öæRâöfb–âD$E"ÖöFR†—G2†ö—7F–ærÖ÷fW26ÆV"ÖöæÇ’76W2—G2÷vâv’’à¢òòDÕôådåôÄ¥•ô4ÄT#ÓGW&ç2—Böf`§G—VFVb7G'V7B²÷6WE÷'B'C²V–çC…÷B6ÆV%ö5³…Ó²ÕDÄ6ÆV$6öÆ÷"65³…Ó²V–çC…÷B6ÆV%öBÂ6ÆV%÷3²F÷V&ÆR6C²V–çC3%÷B73²ÒÆ5öw&÷W°¦VçVÒ²Ä5ôÔ‚Ò‚Ó°§7FF–2Æ5öw&÷WuöÆ5´Ä5ôÔ…Ó²7FF–2–çBuöæÆ3°§7FF–2V–çCcE÷BuöÆ5÷F¶VâÂuöÆ5öÆæFVC²òò6ÆV'2ÆFW"72Föö²2—G2ÆöB7F–öâò6ÆV"ÖöæÇ’76W27F–ÆÂ÷VæVBf÷"F†VÒ‡F†R&W6–FVæ7’6WBÆ–æR§fö–B×FÅöÆ5÷7FG2‡V–çCcE÷B§F¶VâÂV–çCcE÷B¦ÆæFVB’²§F¶VâÒuöÆ5÷F¶Vã²¦ÆæFVBÒuöÆ5öÆæFVC²Ð§7FF–2–çBÆ5ööâ‡fö–B’²7FF–2–çBbÒÓ²–b‡bÂ’²6öç7B6†"¦RÒvWFVçb‚$DÕôådåôÄ¥•ô4ÄT""“²bÒ†Rbb7G&6×†RÂ#"’“²–b‚b’ÔÄôr‚$DÕôådåôÄ¥•ô4ÄT#Ó¢6ÆV'2v—F‚æòG&r÷Vâ72öbF†V—"÷vâ2&Vf÷&R"“²Ò&WGW&âbbbF&G%ööâ‚“²Ð§7FF–2–çBÆ5÷6ÖR†6öç7B÷6WE÷'B¦Â–çB¶Â6öç7B÷6WE÷'B¦"Â–çB¶"’²òòF†R6ÖR7V'&W6÷W&6RF‡&÷Vv‚F†R6ÖRf–Wr†²Òƒ¢FWF‚ò7FVæ6–Â¢V–çCcE÷B3Â3#²–çBÃÂÃ"ÂƒÂƒ#²'E÷7V"†Â¶Âg3ÂfÃÂgƒ“²'E÷7V"†"Â¶"Âg3"ÂfÃ"Âgƒ"“°¢–b‚3ÇÂ3Ò3"ÇÂÃÒÃ"ÇÂƒÒƒ"’&WGW&â°¢–çBƒÒ¶ÓÒ‚òÓæ†5öGf–Wr¢Óæ†5ö7f–Wu¶¶ÒÂƒ"Ò¶"ÓÒ‚ò"Óæ†5öGf–Wr¢"Óæ†5ö7f–Wu¶¶%Ó°¢&WGW&âƒÓÒƒ"bb‚ƒÇÂÖVÖ6×†¶ÓÒ‚òfÓæGf–Wr¢fÓæ7f–Wu¶¶ÒÂ¶"ÓÒ‚òf"ÓæGf–Wr¢f"Óæ7f–Wu¶¶%ÒÂ6—¦Vöb‡f–Wu÷B’’“°§Ð¢òòÆæBWfW'’v—F–ær6ÆV"ÂöÆFW7Bf—'7BÂV6‚w&÷W–â72öb—G2÷vâ‡'öfÇW6‚w2v’§7FF–2fö–BÆ5öÆæB‡fö–B’°¢–b‚uöæÆ2’&WGW&ã°¢–b†uöVæ2’'öVæB‚“°¢–çBâÒuöæÆ3²uöæÆ2Ò°¢G—Vöb…"’6fRÒ#°¢f÷"†–çBv’Ò²v’Âã²v’²²’²6öç7BÆ5öw&÷W¦rÒfuöÆ5¶v•Ó²–çBç’ÒrÓæ6ÆV%öBÇÂrÓæ6ÆV%÷3²f÷"†–çB’Ò²’Âƒ²’²²’ç’ÃÒrÓæ6ÆV%ö5¶•Ó°¢–b‚ç’ÇÂuö6"’6öçF–çVS°¢"ç'BÒrÓç'C²"æ†5÷'BÒ²ÖVÖ7’…"æ6ÆV%ö2ÂrÓæ6ÆV%ö2Â6—¦Vöb"æ6ÆV%ö2“²ÖVÖ7’…"æ62ÂrÓæ62Â6—¦Vöb"æ62“°¢"æ6ÆV%öBÒrÓæ6ÆV%öC²"æ6ÆV%÷2ÒrÓæ6ÆV%÷3²"æ6BÒrÓæ6C²"æ72ÒrÓæ73°¢'ö&Vv–åöW‚ƒ“²'öVæB‚“²uöÆ5öÆæFVB²³²Ð¢"Ò6fS°§Ð¢òò6WE&VæFW%F&vWG2v—F‚6ÆV'2VæF–æræBæòG&r6–æ6S¢F†R6ÆV'2v—B–ç7FVBöb÷Væ–ær70§7FF–2fö–BÆ5÷W6‚‡fö–B’°¢–b†uöæÆ2ÓÒÄ5ôÔ‚’Æ5öÆæB‚“°¢Æ5öw&÷W¦rÒfuöÆ5¶uöæÆ5Ó°¢rÓç'BÒ"ç'C²ÖVÖ7’†rÓæ6ÆV%ö2Â"æ6ÆV%ö2Â6—¦VöbrÓæ6ÆV%ö2“²ÖVÖ7’†rÓæ62Â"æ62Â6—¦VöbrÓæ62“²rÓæ6ÆV%öBÒ"æ6ÆV%öC²rÓæ6ÆV%÷2Ò"æ6ÆV%÷3²rÓæ6BÒ"æ6C²rÓæ72Ò"æ73°¢f÷"†–çBv’Ò²v’ÂuöæÆ3²v’²²’²Æ5öw&÷W¦òÒfuöÆ5¶v•Ó²òòöÆFW"6ÆV'2öbF†R6ÖR7V'&W6÷W&6W2&R÷fW'w&—GFVà¢f÷"†–çB’Ò²’Âƒ²’²²’–b†òÓæ6ÆV%ö5¶•Ò’f÷"†–çB¢Ò²¢Âƒ²¢²²’–b†rÓæ6ÆV%ö5¶¥Ò’²V–çCcE÷B3Â3#²–çBÃÂÃ"ÂƒÂƒ#²'E÷7V"‚fòÓç'BÂ’Âg3ÂfÃÂgƒ“²'E÷7V"‚frÓç'BÂ¢Âg3"ÂfÃ"Âgƒ"“°¢–b‡3bb3ÓÒ3"bbÃÓÒÃ"bbƒÓÒƒ"’²òÓæ6ÆV%ö5¶•ÒÒ²'&V³²ÒÐ¢–b‚†òÓæ6ÆV%öBÇÂòÓæ6ÆV%÷2’bb†rÓæ6ÆV%öBÇÂrÓæ6ÆV%÷2’’²V–çCcE÷B3Â3#²–çBÃÂÃ"ÂƒÂƒ#²'E÷7V"‚fòÓç'BÂ‚Âg3ÂfÃÂgƒ“²'E÷7V"‚frÓç'BÂ‚Âg3"ÂfÃ"Âgƒ"“°¢–b‡3bb3ÓÒ3"bbÃÓÒÃ"bbƒÓÒƒ"’²–b†rÓæ6ÆV%öB’òÓæ6ÆV%öBÒ²–b†rÓæ6ÆV%÷2’òÓæ6ÆV%÷2Ò²ÒÒÐ¢uöæÆ2²³°¢ÖV×6WB…"æ6ÆV%ö2ÂÂ6—¦Vöb"æ6ÆV%ö2“²"æ6ÆV%öBÒ"æ6ÆV%÷2Ò°§Ð¢òòv†WF†W"F†R&öw&Ò6×ÆW2FW‡GW&Rv—F‚v—F–ær6ÆV"†&öw&Òv—F‚&–æFÆW72÷"G–æÖ–2†æFÆW26÷VçG226×Æ–ær§7FF–2–çBÆ5÷6×ÆVB„çe&ör§"’°¢–b‚"’&WGW&â°¢f÷"†–çB7BÒ²7BÂS²7B²²’²õ÷Vç6fU÷Vç&WF–æVBçe6†FW"§6‚Ò"Óç6…·7EÓ²–b‚6‚ÇÂ6‚Óç7FvRãÒb’6öçF–çVS°¢–b‡6‚Óç&bæ&–æFÆW72ÇÂ6‚Óç&bæG–â’&WGW&â°¢f÷"†–çB’Ò²’Â6‚Óç&bææ#²’²²’²6öç7Bçfå÷&&–æB¦"Òg6‚Óç&bæ%¶•Ó²V–çCcE÷Bƒ°¢–b†"Óæ¶–æBÓÒådåõ$%ô”Ôr’‚Ò"Óç&VrÂ3"ò2æ–Öu·6‚Óç7FvUÕ¶"Óç&VuÒ¢²VÇ6R–b†"Óæ¶–æBÓÒådåõ$%õDU‚ÇÂ"Óæ¶–æBÓÒådåõ$%õD%Tb’‚ÒFW…÷6Æ÷Eö‚…2çFW…·6‚Óç7FvUÒÂ2æ–Öu·6‚Óç7FvUÒÂ"Óç&Vr“²VÇ6R6öçF–çVS°¢V–çCcE÷B6W"Ò‚ò…÷6W"†‚’¢²–b‚6W"’6öçF–çVS°¢f÷"†–çBv’Ò²v’ÂuöæÆ3²v’²²’²6öç7BÆ5öw&÷W¦rÒfuöÆ5¶v•Ó°¢f÷"†–çB²Ò²²Âƒ²²²²’–b†rÓæ6ÆV%ö5¶µÒbbrÓç'Bæ6öÆ÷%¶µÒbbrÓç'Bæ6öÆ÷%¶µÒÓç6W&–ÂÓÒ6W"’&WGW&â°¢–b‚†rÓæ6ÆV%öBÇÂrÓæ6ÆV%÷2’bbrÓç'BæFWF‚bbrÓç'BæFWF‚Óç6W&–ÂÓÒ6W"’&WGW&â²ÒÒÐ¢&WGW&â°§Ð¢òòF†RG&r&÷WBFò÷Vâ73¢—G2GF6†ÖVçG2F¶RF†Rv—F–ær6ÆV'2öbF†R6ÖR7V'&W6÷W&6W2‡F†RæWvW7BöæS²F†R72w2÷vâVæF–ær6ÆV"—2æWvW"7F–ÆÂ’ÂF†R&W7BÆæBf—'7@§7FF–2fö–BÆ5÷F¶R„çe&ör§"’°¢–b†Æ5÷6×ÆVB‡"’’²Æ5öÆæB‚“²&WGW&ã²Ð¢f÷"†–çBv’ÒuöæÆ2Ò²v’ãÒ²v’ÒÒ’²Æ5öw&÷W¦rÒfuöÆ5¶v•Ó°¢f÷"†–çB’Ò²’Âƒ²’²²’–b†rÓæ6ÆV%ö5¶•Ò¢f÷"†–çB¢Ò²¢Â"ç'Bæâbb¢Âƒ²¢²²’–b…"ç'Bæ6öÆ÷%¶¥Òbb'EöGW÷6Æ÷B‚e"ç'BÂ¢’bbçfåöf×Eö&2…"ç'Bæ6öÆ÷%¶¥ÒÓæBæf÷&ÖB’bbÆ5÷6ÖR‚frÓç'BÂ’Âe"ç'BÂ¢’’°¢–b‚"æ6ÆV%ö5¶¥Ò’²"æ6ÆV%ö5¶¥ÒÒ²"æ65¶¥ÒÒrÓæ65¶•Ó²Ð¢rÓæ6ÆV%ö5¶•ÒÒ²uöÆ5÷F¶Vâ²³²'&V³²Ð¢–b‚†rÓæ6ÆV%öBÇÂrÓæ6ÆV%÷2’bb"ç'BæFWF‚bbÆ5÷6ÖR‚frÓç'BÂ‚Âe"ç'BÂ‚’’°¢–b†rÓæ6ÆV%öB’²–b‚"æ6ÆV%öB’²"æ6ÆV%öBÒ²"æ6BÒrÓæ6C²ÒrÓæ6ÆV%öBÒ²Ð¢–b†rÓæ6ÆV%÷2’²–b‚"æ6ÆV%÷2’²"æ6ÆV%÷2Ò²"æ72ÒrÓæ73²ÒrÓæ6ÆV%÷2Ò²Ð¢uöÆ5÷F¶Vâ²³²ÒÐ¢Æ5öÆæB‚“°§Ð¢òòF†R÷W&F–öç2v—F–ær6ÆV"6âv—B7&÷73¢F†W’æV—F†W"&VBæ÷"w&—FRFW‡GW&W0§7FF–2–çBÆ5ö¶VW2‡V–çC3%÷B÷’°¢7v—F6‚†÷’°¢66Rõõ4UEõ%C¢66Rõô4ÄT%ô4ôÄõ#¢66Rõô4ÄT%ôE3¢66RõôE$s¢66RõôE$uô”Eƒ¢66Rõô$Tt”ã¢66Rõô¥TÕ¢66Rõô4ÄÃ ¢66Rõõ44•54õ#¢66Rõô$”äEõ$ôu$Ó¢66Rõô$”äEõ5DDS¢66Rõô$”äEõdEE#¢66Rõô$”äEõe5E$TÓ¢66Rõô$”äEõd%Tc¢66Rõô$”äEõd%Tc ¢66Rõô$”äEõT$ó¢66Rõô$”äEõT$õõ4ÔÄÃ¢66Rõô$”äEõ54$ó¢66Rõô$”äEõDUƒ¢66Rõô$”äEõDUƒ¢66Rõô$”äEõ4Õ¢66Rõô$”äEô”Ôs¢66Rõõ5DTä4”Ã ¢66RõõôÅ•ôôde4UC¢66Rõõd”Uuõ%E3¢66RõôDUD…õ$ätS¢66RõôDUD…ô4ÄÕ¢66Rõõeõ5t•¥¤ÄS¢66Rõô$ÄTäEô4ôÄõ#¢66RõôÅ„õ$Tc¢66RõõU4…ô4ôå5C¢66Rõô4õ•õ5E$”DS ¢66RõõDeô$”äC¢66RõõDeõ5DDS¢66Rõõ$5EôD•44$C¢66Rõõ$”Õõ$U5D%C¢66Rõô$%$”U#¢66Rõõ4UEõDU…õôôÃ¢66Rõõ4UEõ4ÕõôôÃ¢66Rõõ$TäDU%ô4ôäC ¢66RõõUDDUõT#¢66Rõô4õ•ô#$#¢66Rõô4ÄT%ô%Tc¢&WGW&â°¢FVfVÇC¢&WGW&â°¢Ð§Ð §7FF–2fö–BF—7F6‚†6öç7BöF—7F6‚¦G“°¢òòÒÒÒÒÒÒÒÒÒÒ6ö×WFRF—7F6†W2Ö÷fVB†VBöbF†R&VæFW"72F†W’v÷VÆB7Æ—B„DÕôådåô55ô„ô•5B’ÒÒÒÒÒÒÒÒÒÐ¢òòÖWFÂ6ææ÷B'Vâ6ö×WFRF—7F6‚–ç6–FR&VæFW"72Â6òF—7F6‚&WGvVVâGvòG&w2öböæR72VæG2—BÂæBöâF–ÆRÖ&6VBuRWfW'’GF6†ÖVçB—27F÷&VBæBÆöFVBv–ã ¢òò'&VF‚öbF†Rv–ÆB'Vç2öæRuR'F–6ÆRF—7F6‚W"VÖ—GFW"&WGvVVâF†RrÖ'VffW"w2VffV7BG&w2Â6—‚FòV–v‡B7Æ—G2öbcƒ“rÖ'VffW"g&ÖR†&÷WB3SÔ"öbGF6†ÖVç@¢òòG&ff–2’âv†VâG&r÷Vç272ÂF†R÷2gFW"—B&R66ææVBWFòF†Rf—'7B÷F†Bv÷VÆBVæBF†R72ç—v“²V6‚F—7F6‚ÖWBöâF†Rv’'Vç2&–v‡Bv’Â&Vf÷&RF†R70¢òò÷Vç2Â–bÖ÷f–ær—B†VBöbF†RG&w2&Vf÷&R—B–âF†B7G&WF6‚6†ævW2æ÷F†–ærF†W’&VB÷"w&—FS¢—Bw&—FW2æòÖVÖ÷'’F†÷6RG&w2&VB‡fW'FW‚Â–æFW‚Â–æF—&V7BÂVæ–f÷&Òæ@¢òò7F÷&vR'VffW'3²7F÷&vR'VffW'26÷VçB2w&—GFVâ'’F†RF—7F6‚’Â6×ÆW2æò6öÆ÷W"GF6†ÖVçBöbF†R72Âw&—FW2æò–ÖvRÂæBF†RG&w2w&—FRæòÖVÖ÷'’†æò7F÷&vR'VffW"À¢òò–ÖvR÷"G&ç6f÷&ÒfVVF&6²’â&WÆ’F†Vâ6¶—2—BÂæBF†R727F—2÷Vâ7&÷72—BâF—7F6‚F†B6×ÆW2F†R72w2FWF‚GF6†ÖVçB6VW2—B2F†R72&VvâÂ6ò—BÖ÷fW0¢òòöæÇ’v†–ÆRæòG&r&Vf÷&R—B–âF†R72†2FWF‚w&—FW2öâ‡VçF–ÂF†VâF†BFWF‚—2F†R6ÖS²'&VF‚öbF†Rv–ÆBw2'F–6ÆR6öÆÆ—6–öâ6×ÆW2F†RrÖ'VffW"FWF‚’âöâ'’FVfVÇ@¢òòƒöfb“¢'&VF‚öbF†Rv–ÆBB“öââÓ"Ö‚Â#‚(i"‚6ö×WFRVæ6öFW'2æBC’(i"#’&VæFW"76W2g&ÖRÂg&ÖRSã"(i"‚ã‚×0§7FF–2–çB76†ö—7Eööâ‡fö–B’²7FF–2–çBÒÒÓ²–b†ÒÂ’²6öç7B6†"¦RÒvWFVçb‚$DÕôådåô55ô„ô•5B"“²ÒÒ†Rbb7G&6×†RÂ#"’“²–b‚Ò’ÔÄôr‚$DÕôådåô55ô„ô•5CÓ¢6ö×WFRF—7F6†W2&WGvVVâF†RG&w2öb72VæB—B"“²Ò&WGW&âÓ²Ð§G—VFVb7G'V7B²V–çCcE÷BÆòÂ†“²Ò76…÷&æs°¦VçVÒ²54…õ$âÒ#BÂ54…ôÔ‚ÒcBÓ°§7FF–26öç7BV–çC…÷B¦uö76…ö÷´54…ôÔ…Ó²7FF–2–çBuö76…öã²7FF–2V–çCcE÷Buö76…öÖ÷fVBÂuö76…öFWF‚Âuö76…÷7F÷³…Ó²òòF—7F6†W2'Vâ†VBæBæ÷B–WBÖWB'’&WÆ“²Ö÷fVBÂöbF†VÒ6×Æ–ærF†R72w2FWFƒ²v‡’66ç27F÷V@§7FF–2–çB76…÷F¶R†6öç7BV–çC…÷B§’²f÷"†–çB’Ò²’Âuö76…öã²’²²’–b†uö76…ö÷¶•ÒÓÒ’²uö76…ö÷¶•ÒÒuö76…ö÷²ÒÖuö76…öåÓ²&WGW&â²Ò&WGW&â²Ð§7FF–2–çB76…öFB†76…÷&ær§"Â–çB¦âÂV–çCcE÷BÂV–çCcE÷B7¢’°¢–b‚’&WGW&â°¢òò7'6Rf—'GVÂFG&W76W26âæÖRF†R6ÖR7F÷&vRF‡&÷Vv‚F–ffW&Vç@¢òòööÇ2âwVW7BÖFG&W726ö×&—6öâ6ææ÷BW7F&Æ—6‚–æFWVæFVæ6RF†W&Rà¢–b‚7¢ÇÂ7¢âT”åCcEôÔ‚ÒÇÂ¦âãÒ54…õ$âÇÂçfå÷7'6UöÆ–2†Â²7¢’’&WGW&â°¢%²‚¦â’²µÒÒ†76…÷&ær—¶Â²7§Ó²&WGW&â°§Ð§7FF–2–çB76…ö†—B†6öç7B76…÷&ær¦Â–çBæÂ6öç7B76…÷&ær¦"Â–çBæ"’²f÷"†–çB’Ò²’Âæ²’²²’f÷"†–çB¢Ò²¢Âæ#²¢²²’–b†¶•ÒæÆòÂ%¶¥Òæ†’bb%¶¥ÒæÆòÂ¶•Òæ†’’&WGW&â²&WGW&â²Ð§7FF–2–çB76…öGF6†ÖVçEö†—B‡V–çCcE÷BFG"ÂV–çCcE÷B6—¦R’°¢–b‚FG"’&WGW&â°¢76…÷&ær66W75³Ó²–çBæÒ²–b‚76…öFB†66W72ÂfæÂFG"Â6—¦R’’&WGW&â°¢f÷"†–çB²Ò²²Â“²²²²’°¢6öç7BFW…÷B§BÒ²ÓÒ‚ò"ç'BæFWF‚¢²Â"ç'Bæâò"ç'Bæ6öÆ÷%¶µÒ¢åTÄÃ°¢–b‚BÇÂBÓæÖVÒ’6öçF–çVS°¢76…÷&ær7F÷&vU³Ó²–çBç2Ò°¢–b‚76…öFB‡7F÷&vRÂfç2Â‡V–çCcE÷B—BÓæÖVÒÂBÓäÂç6—¦R’ÇÂ76…ö†—B†66W72ÂæÂ7F÷&vRÂç2’’&WGW&â°¢Ð¢&WGW&â°§Ð¢òòF†RÖVÖ÷'’G&r&VG2&W6–FW2FW‡GW&W3²v†Vâ—B6ææ÷B&RÆ—7FVB÷"F†RG&rÖ’w&—FRÖVÖ÷'§7FF–2–çB76…öG&u÷&VG2„çe&ör§"Â6öç7BöG&r¦GrÂ6öç7BöG&uö–G‚¦F’Â76…÷&ær§"Â–çB¦â’°¢–b‚"ÇÂ"Óç6…³%ÒÇÂ"Óç6…³5ÒÇÂ"Óç6…³EÒÇÂ2ç&7EöF—66&BÇÂ†GrbbGrÓæÖF•ö6çB’ÇÂ†F’bb†F’ÓæÖF•ö6çBÇÂF’Óæ–æB’’’&WGW&â°¢f÷"†–çB²Ò²²Âc²²²²’–b…2çf'Ve¶µÒç6—¦Rbb76…öFB‡"ÂâÂ2çf'Ve¶µÒæFG&W72Â2çf'Ve¶µÒç6—¦R’’&WGW&â²òòâV×G’7G&VÒ&VG2æòwVW7BÖVÖ÷'¢–b†F’bb76…öFB‡"ÂâÂF’Óæ–'VbÂ‡V–çCcE÷B’†F’Óæ6÷VçBâòF’Óæ6÷VçB¢’¢B²b’’&WGW&â°¢–b‚76…öFB‡"ÂâÂGròGrÓæ–æB¢F’òF’Óæ–æB¢Â3"’’&WGW&â°¢f÷"†–çB7BÒ²7BÂ#²7B²²’²õ÷Vç6fU÷Vç&WF–æVBçe6†FW"§6‚Ò"Óç6…·7EÓ²–b‚6‚’6öçF–çVS°¢–b‡6‚Óç&bæ&–æFÆW72ÇÂ6‚Óç&bæG–â’&WGW&â°¢f÷"†–çB’Ò²’Â6‚Óç&bææ#²’²²’²6öç7Bçfå÷&&–æB¦"Òg6‚Óç&bæ%¶•Ó°¢7v—F6‚†"Óæ¶–æB’°¢66Rådåõ$%ô4%Tc¢66Rådåõ$%õT$õ¢–b†"Óç&VrÂbbb76…öFB‡"ÂâÂ2çV&õ·7EÕ¶"Óç&VuÒæFG"Â2çV&õ·7EÕ¶"Óç&VuÒç6—¦R’’&WGW&â²'&V³°¢66Rådåõ$%õDUƒ¢²V–çCcE÷B‚ÒFW…÷6Æ÷Eö‚…2çFW…·7EÒÂ2æ–Öu·7EÒÂ"Óç&Vr“²–çBF–BÒ†–çB’†‚b†ffffb“²–b‚F–B’'&V³°¢÷5÷Væf—%öÆö6µöÆö6²‚fu÷&×R“²FW…÷BBÒF–BÂuöçG&Vrbbu÷G&Vu·F–EÒæö²òu÷G&Vu·F–EÒçB¢‡FW…÷B—³Ó²÷5÷Væf—%öÆö6µ÷VæÆö6²‚fu÷&×R“°¢òòÆ–æV"æB'VffW"FW‡GW&W26â&VBF—&V7FÇ’g&öÒööÀ¢òò'VffW'2â–æ6ÇVFR7F÷&vRWfVâv†Vâ&VfÆV7F–öâ6ÆÇ2—BDU‚à¢–b‚Bç6W&–ÂÇÂ76…öFB‡"ÂâÂ‡V–çCcE÷B—BæÖVÒÂBäÂç6—¦R’’&WGW&â°¢'&V³²Ð¢66Rådåõ$%õDU…S¢66Rådåõ$%õDU„C¢66Rådåõ$%õDU„¢&WGW&â°¢66Rådåõ$%õ4%Te¢66Rådåõ$%õ4%TeS¢66Rådåõ$%ô”Ôs¢66Rådåõ$%õD%Tc¢66Rådåõ$%ôtÔTÓ¢&WGW&â²òòÖ’w&—FRÖVÖ÷'’Â÷"&VG2ÖVÖ÷'’æ÷BÆ—7FVB†W&P¢FVfVÇC¢'&V³²ÒÒÐ¢&WGW&â°§Ð¢òòF—7F6‚B—G2Æ6R–âF†R7G&VÒ…22—B—2F†W&R“¢—G27F÷&vR'VffW'2†6÷VçFVB2w&—GFVâ’ÂæBv†WF†W"—B6×ÆW26öÆ÷W"GF6†ÖVçBöbF†R72ƒ’ÂöæÇ’F†RFWF‚ƒ"§7FF–2–çB76…ö72„çe6†FW"¦72Â6öç7BöF—7F6‚¦GÂ76…÷&ær§rÂ–çB¦çrÂ–çB¦FWF‚’°¢–b‚72ÇÂ72Óæ'V–ÇF–âÇÂGÓæ–æBÇÂ72Óç&bæ&–æFÆW72ÇÂ72Óç&bæG–â’&WGW&â°¢f÷"†–çB’Ò²’Â72Óç&bææ#²’²²’²6öç7Bçfå÷&&–æB¦"Òf72Óç&bæ%¶•Ó°¢7v—F6‚†"Óæ¶–æB’°¢66Rådåõ$%õ4%Te¢–b†"Óç&VrãÒbÇÂ76…öGF6†ÖVçEö†—B…2ç76&õ³UÕ¶"Óç&VuÒæFG"Â2ç76&õ³UÕ¶"Óç&VuÒç6—¦R’ÇÂ76…öFB‡rÂçrÂ2ç76&õ³UÕ¶"Óç&VuÒæFG"Â2ç76&õ³UÕ¶"Óç&VuÒç6—¦R’’&WGW&â²'&V³°¢66Rådåõ$%ô4%Tc¢66Rådåõ$%õT$õ¢–b†"Óç&VrãÒbÇÂ76…öGF6†ÖVçEö†—B…2çV&õ³UÕ¶"Óç&VuÒæFG"Â2çV&õ³UÕ¶"Óç&VuÒç6—¦R’’&WGW&â²'&V³°¢66Rådåõ$%õ4%TeS¢66Rådåõ$%ô”Ôs¢66Rådåõ$%õDU…S¢66Rådåõ$%õDU„C¢66Rådåõ$%õDU„¢66Rådåõ$%ôtÔTÓ¢&WGW&â°¢66Rådåõ$%õDUƒ¢66Rådåõ$%õD%Tc¢²V–çCcE÷B‚ÒFW…÷6Æ÷Eö‚…2çFW…³UÒÂ2æ–Öu³UÒÂ"Óç&Vr“²–çBF–BÒ†–çB’†‚b†ffffb“²–b‚F–B’'&V³°¢÷5÷Væf—%öÆö6µöÆö6²‚fu÷&×R“²FW…÷BBÒF–BÂuöçG&Vrbbu÷G&Vu·F–EÒæö²òu÷G&Vu·F–EÒçB¢‡FW…÷B—³Ó²÷5÷Væf—%öÆö6µ÷VæÆö6²‚fu÷&×R“°¢–b‚Bç6W&–Â’&WGW&â°¢f÷"†–çB²Ò²²Â"ç'Bæâbb²Âƒ²²²²’–b‡FW…÷7F÷&vUö÷fW&Æ‚gBÂ"ç'Bæ6öÆ÷%¶µÒ’’&WGW&â°¢–b‡FW…÷7F÷&vUö÷fW&Æ‚gBÂ"ç'BæFWF‚’’¦FWF‚Ò°¢'&V³²Ð¢FVfVÇC¢'&V³²ÒÐ¢&WGW&â°§Ð§7FF–2fö–B75ö†ö—7E÷G'’„çe&ör§#Â6öç7BöG&r¦GsÂ6öç7BöG&uö–G‚¦F“’°¢–b‚uöW†V5öæW‡BÇÂu÷&WEöââ‚ÇÂuög¥ö6†V6¶–ærÇÂuöFVfW%÷&WÆ––ærÇÂuöæÆ2ÇÂ"æ6ÆV%öBÇÂ"æ6ÆV%÷2’&WGW&ã°¢f÷"†–çB’Ò²’Âƒ²’²²’–b…"æ6ÆV%ö5¶•Ò’&WGW&ã²òò6ÆV"F†—272F¶W22—G2ÆöB7F–öâ6öÖW2&Vf÷&Rç—F†–ærÖ÷fVBF—7F6‚6÷VÆB&V@¢7FF–276…÷&ær&E´54…õ$åÓ²–çBç&BÒ°¢–b‚76…öG&u÷&VG2‡#ÂGsÂF“Â&BÂfç&B’’²uö76…÷7F÷³Ò²³²&WGW&ã²Ð¢g¥ö7W'6÷"7W"Ò²çÒuöW†V5öæW‡GÓ²f÷"†–çB²Ò²²Âu÷&WEöã²²²²’7W"ç&WE¶7W"ç&WEöâ²µÒÒu÷&WE¶µÓ°¢7FF–2G—Vöb…2’76fS²ÖVÖ7’‚e76fRÂe2Â6—¦Vöb2“²G—Vöb…"’'6fRÒ#²–çB6õöF—'G’Òu÷6õöF—'G’ÂG75öF—'G’ÒuöG75öF—'G’ÂÖ÷fVBÒ°¢–çBGw&÷FRÒ"ç'BæFWF‚bb…2æG2æGrÇÂ2æG2ç7B“²òòV—F†W"ÆæS¢6ö×WFRFW‡GW&Rf–WrÖ’&VB7FVæ6–À¢f÷"†–çBwV&BÒ²wV&BÂC“c²wV&B²²’°¢6öç7BV–çC…÷B§²÷ö†G"‡²6öç7Bfö–B¦°¢–b‚g¥öæW‡B‚f7W"ÂgÂf‡Âf’’'&V³°¢–b†‡æ÷ÓÒõô$%$”U"’²V–çC3%÷B&—G3²ÖVÖ7’‚f&—G2ÂÂ6—¦Vöb&—G2“²–b†&—G2’²uö76…÷7F÷³UÒ²³²'&V³²Ò6öçF–çVS²Ð¢–b†‡æ÷ÓÒõôE$rÇÂ‡æ÷ÓÒõôE$uô”E‚’°¢öG&rFGs²öG&uö–G‚FF“²6öç7BöG&r¦BÒ‡æ÷ÓÒõôE$ròG&u÷&V2†Â‡ç6—¦RÂgFGr’¢åTÄÃ²6öç7BöG&uö–G‚§‚Ò‡æ÷ÓÒõôE$uô”E‚òG&uö–G…÷&V2†Â‡ç6—¦RÂgFF’’¢åTÄÃ°¢&öu÷&W6öÇfR‚“²õ÷Vç6fU÷Vç&WF–æVBçe&ör§"Ò2ç&örò&öuövWB…2ç&ör’¢æ–Ã°¢–b‚76…öG&u÷&VG2‡"ÂBÂ‚Â&BÂfç&B’’²uö76…÷7F÷³Ò²³²'&V³²Ð¢–b…"ç'BæFWF‚bb…2æG2æGrÇÂ2æG2ç7B’’Gw&÷FRÒ°¢6öçF–çVS²Ð¢–b†‡æ÷ÓÒõôD•5D4‚’°¢6öç7BöF—7F6‚¦GÒ²õ÷Vç6fU÷Vç&WF–æVBçe6†FW"¦72Ò2æ7&örbb&öuövWB…2æ7&ör’ò&öuövWB…2æ7&ör’Óç6…³UÒ¢æ–Ã°¢7FF–276…÷&ærw%´54…õ$åÓ²–çBçw"ÒÂFWF‚Ò°¢–b‚76…ö72†72ÂGÂw"Âfçw"ÂfFWF‚’’²uö76…÷7F÷³%Ò²³²'&V³²Ð¢–b†FWF‚bbGw&÷FR’²uö76…÷7F÷³uÒ²³²'&V³²Òòò—Bv÷VÆBÖ—72FWF‚F†RG&w2&Vf÷&R—Bw&÷FP¢–b†76…ö†—B‡w"Âçw"Â&BÂç&B’’²uö76…÷7F÷³5Ò²³²'&V³²ÒòòF†RG&w2&Vf÷&R—B&VBv†B—Bw&—FW0¢–b†uö76…öâãÒ54…ôÔ‚’²uö76…÷7F÷³EÒ²³²'&V³²Ð¢B‚$‚VÆÇRF—7F6‚W2Ö÷fVB†VBöbF†R72W2"Âu÷7FE÷72²Â72Óæ¶W’ÂFWF‚ò"‡6×ÆW2F†R72w2FWF‚Âv†–6‚æòG&r&Vf÷&R—Bw&÷FR’"¢""“°¢F—7F6‚†G“²uö76…ö÷¶uö76…öâ²µÒÒ²uö76…öÖ÷fVB²³²uö76…öFWF‚³ÒFWFƒ²Ö÷fVB²³°¢6öçF–çVS²Ð¢–b†‡æ÷ÓÒõõDeõ5DDR’²uö76…÷7F÷³UÒ²³²'&V³²Òòò6fW2÷&W7F÷&W26÷VçFW'2–âuRÖVÖ÷'¢–b‡7FFUö÷†‡æ÷Â’’6öçF–çVS°¢uö76…÷7F÷³UÒ²³²'&V³²òòç—F†–ærVÇ6RÖ’VæBF†R72Âw&—FRÖVÖ÷'’÷"FWVæBöâF†R÷&FW"†6÷–W2ÂWFFW2Â6ÆV'2ÂF&vWG2ÂVW&–W2¢Ð¢–b†7W"æ&B’uö76…÷7F÷³eÒ²³°¢ÖVÖ7’‚e2Âe76fRÂ6—¦Vöb2“²&w5öF—'G•öÆÂ‚“²"Ò'6fS²u÷6õöF—'G’Ò6õöF—'G’ÇÂÖ÷fVC²uöG75öF—'G’ÒG75öF—'G’ÇÂÖ÷fVC²&öu÷&W6öÇfR‚“°¢²7FF–2–çBöæ6S²–b†Ö÷fVBbböæ6R²²’ÔÄôr‚&6ö×WFRF—7F6†W2Ö÷fVB†VBöbF†R&VæFW"72F†W’v÷VÆB7Æ—B„DÕôådåô55ô„ô•5B“¢f—'7B66R72VÆÇRÂVBF—7F6†W2"Âu÷7FE÷72²ÂÖ÷fVB“²Ð§Ð§fö–B×FÅövVöÖW6…÷7FG2‡V–çCcE÷B¦âÂV–çCcE÷B§v—B’²¦âÒuövVõöÖW6…öã²§v—BÒuövVõöÖW6…÷v—C²Ð§fö–B×FÅövVö†ö—7E÷7FG2‡V–çCcE÷B¦Ö÷fVBÂV–çCcE÷B§'Vç2ÂV–çCcE÷B§7F÷2’²¦Ö÷fVBÒuövVõö†ö—7FVC²§'Vç2ÒuövVõö†ö—7E÷'Vç3²ÖV×6WB‡7F÷2ÂÂb¢6—¦Vöb§7F÷2“²7F÷5³5ÒÒuövVõöFWVæFVæ7•÷7Æ—G3²Ð§fö–B×FÅö76†ö—7E÷7FG2‡V–çCcE÷B¦Ö÷fVBÂV–çCcE÷B¦FWF‚ÂV–çCcE÷B§7F÷2’²¦Ö÷fVBÒuö76…öÖ÷fVC²¦FWF‚Òuö76…öFWFƒ²f÷"†–çB’Ò²’Âƒ²’²²’7F÷5¶•ÒÒuö76…÷7F÷¶•Ó²Ð¢òò‡—6–6Â6öÆ÷"GF6†ÖVçG2F†R6VÆV7FVB—VÆ–æR6âw&—FRâÖW&vVB72&VÖ2Æöv–6Â÷WGWG2æB6†ææVÂÖ6·3²7V6–Æ—¦VBg&vÖVçBw2÷WGWG2&RVæ¶æ÷vâ†W&Rà§7FF–2V–çC3%÷BG&uö6öÆ÷%÷w&—FW2„çe&ör§"’°¢çe6†FW"¦g2Ò"ò"Óç6…³Ò¢æ–Ã°¢–b‚g2ÇÂ2ç&7EöF—66&B’&WGW&â°¢V–çC3%÷B÷WGWG2Òu÷6õög6fâÇÂ×5ög5ö÷fW'&–FR†g2’ÇÂg2Óç&bæ6öÆ÷%ö¶æ÷vâòT”åC3%ôÔ‚¢g2Óç&bæ6öÆ÷%öÖ6³°¢V–çC3%÷BÖ6²Ò°¢f÷"†–çB’Ò²’Âƒ²’²²’°¢Vç6–væVBÂÒu÷6õöÆÖòu÷6õöÆÖ¶•Ò¢‡Vç6–væVB–“°¢–b†ÂÂ‚bb‚…2æ6†Ö6²æÖ6²ãâƒB¢Â’’b†b’bb†÷WGWG2ãâÂb’’Ö6²ÃÒRÃÂ“°¢Ð¢&WGW&âÖ6³°§Ð¢òòÒÒÒÒådã"ÖW6‚G&w2†çfä6öÖÖæD'VffW$G&tÖW6…F6·2Â&V6÷&FVB2G&röbådåõ$”ÕôÔU4‚“¢F†RF6²7FvR‡6Æ÷B2’2ÖWFÂw2ö&¦V7BgVæ7F–öâÂF†RÖW6‚7FvR‡6Æ÷BB’2—G0¢òòÖW6‚gVæ7F–öâ†F×6…öÖW6‡7FvRç’’ÂF†Rg&vÖVçB7FvR2W7VÂâG&uò÷Vç2F†R72æB6WG2F†R7FFR2f÷"ç’G&s²F†W6RF¶RF†R—VÆ–æRw2Æ6RæBF†RG&r6ÆÂw2à§7FF–2çfåö†Öu÷6õö×6‚Ò²æ¶ÆVâÒ6—¦Vöb‡6õö¶W’—Ó²7FF–2÷5÷Væf—%öÆö6²u÷6õö×6…öÆö6²Òõ5õTäd•%ôÄô4µô”ä•C°§7FF–2å4ö&¦V7B¦u÷6õö×6…ö'V–ÆF–æs°§7FF–2V–çCcE÷BuöÖW6…öG&w2ÂuöÖW6…÷v—C²òòÖW6‚G&w2G&vã²ÖW6‚G&w26¶—VBv†–ÆRF†V—"—VÆ–æRv2'V–Ç@§7FF–2–CÄÕDÅ&VæFW%—VÆ–æU7FFSâÖW6…÷6ò„çe&ör§"’°¢çe6†FW"§G2Ò"Óç6…³5ÒÂ¦×2Ò"Óç6…³EÒÂ¦g2Ò"Óç6…³Ó°¢6õö¶W’³²6õö¶W•öf–ÆÂ‚f²Â×2Âg2Âu÷72æ6f×BÂu÷72æFf×BÂu÷72ç6f×BÂu÷72ç6×ÆW2Â“°¢6õö¶W’¶BÒ³²²æg6fâÒ…õö'&–FvRfö–B¢—G3²òòF†RF6²7FvRFVÆÇ2—VÆ–æW2v—F‚F†R6ÖRÖW6‚æBg&vÖVçB7FvW2'B‡F†—2F&ÆR†öÆG2ÖW6‚—VÆ–æW2öæÇ’¢÷5÷Væf—%öÆö6µöÆö6²‚fu÷6õö×6…öÆö6²“²–B2Ò†ÕövWEö–B‚fu÷6õö×6‚Âf²“°¢–b‚2’²–b‚u÷6õö×6…ö'V–ÆF–ær’u÷6õö×6…ö'V–ÆF–ærÒ´å4ö&¦V7BæWuÓ²†Õ÷WEö–B‚fu÷6õö×6‚Âf²Âu÷6õö×6…ö'V–ÆF–ær“²Ð¢÷5÷Væf—%öÆö6µ÷VæÆö6²‚fu÷6õö×6…öÆö6²“°¢–b‚2’°¢ÕDÅ&VæFW%—VÆ–æTFW67&—F÷"¦BÒ6õöFW62‚f¶BÂ×2Âg2Âæ–ÂÂ“²òòF†Rg&vÖVçBgVæ7F–öâÂGF6†ÖVçG2æB&ÆVæF–ær2f÷"fW'FW‚—VÆ–æP¢ÕDÄÖW6…&VæFW%—VÆ–æTFW67&—F÷"¦ÒÒ´ÕDÄÖW6…&VæFW%—VÆ–æTFW67&—F÷"æWuÓ°¢Òæö&¦V7DgVæ7F–öâÒG2òG2Óæfâ¢æ–Ã²ÒæÖW6„gVæ7F–öâÒ×2Óæfã²Òæg&vÖVçDgVæ7F–öâÒBæg&vÖVçDgVæ7F–öã°¢–b‡G2’Òç–ÆöDÖVÖ÷'”ÆVæwF‚ÒG2Óç&bç–ÆöBòG2Óç&bç–ÆöB¢c3ƒC²òòF†RF6²w2–ÆöB†F×6…ö'V–ÆBç’’ÂVÇ6RÖWFÂw2Æ–Ö—@¢Òç&7FW%6×ÆT6÷VçBÒBç&7FW%6×ÆT6÷VçC²ÒæÇ†Fô6÷fW&vTVæ&ÆVBÒBæÇ†Fô6÷fW&vTVæ&ÆVC°¢f÷"†–çB’Ò²’Âƒ²’²²’Òæ6öÆ÷$GF6†ÖVçG5¶•ÒÒBæ6öÆ÷$GF6†ÖVçG5¶•Ó°¢ÒæFWF„GF6†ÖVçE—†VÄf÷&ÖBÒBæFWF„GF6†ÖVçE—†VÄf÷&ÖC²Òç7FVæ6–ÄGF6†ÖVçE—†VÄf÷&ÖBÒBç7FVæ6–ÄGF6†ÖVçE—†VÄf÷&ÖC°¢6öç7B6†"§F²ÒG2òG2Óæ¶W’¢"Ò"Â¦Ö²Ò×2Óæ¶W’Â¦f²Òg2òg2Óæ¶W’¢"Ò#²6õö¶W’¶²Ò³°¢F—7F6…ö7–æ2‡ö&u÷VWVR‚’Âç°¢å4W'&÷"¦RÒæ–Ã²V–çCcE÷BCÒ'öæ÷r‚“°¢–CÄÕDÅ&VæFW%—VÆ–æU7FFSâÒ¶uöFWbæWu&VæFW%—VÆ–æU7FFUv—F„ÖW6„FW67&—F÷#¦Ò÷F–öç3¤ÕDÅ—VÆ–æT÷F–öäæöæR&VfÆV7F–öã¦æ–ÂW'&÷#¢fUÓ°¢ÔÄôr‚"W6ÖW6‚—VÆ–æS¢F6²W2ÖW6‚W2g&vÖVçBW2ÂRãb×2–âF†R&6¶w&÷VæBW2W2"Âò""¢%µ$TEÒ"ÂF²ÂÖ²Âf²Â†'öæ÷r‚’ÒC’òSbÂò""¢#¢"Âò""¢RæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“°¢÷5÷Væf—%öÆö6µöÆö6²‚fu÷6õö×6…öÆö6²“²†Õ÷WEö–B‚fu÷6õö×6‚Âf¶²Âó¢å4çVÆÂæçVÆÂ“²÷5÷Væf—%öÆö6µ÷VæÆö6²‚fu÷6õö×6…öÆö6²“°¢Ò“°¢&WGW&âæ–Ã°¢Ð¢&WGW&â2ÓÒå4çVÆÂæçVÆÂÇÂ2ÓÒu÷6õö×6…ö'V–ÆF–æròæ–Â¢3°§Ð§7FF–2fö–BÖW6…öVæ6öFR†–CÄÕDÅ&VæFW$6öÖÖæDVæ6öFW#âVåòÂçe&ör§"Â6öç7BöG&r¦Gr’°¢–CÄÕDÃE&VæFW$6öÖÖæDVæ6öFW#âVâÒ†–CÄÕDÃE&VæFW$6öÖÖæDVæ6öFW#â–Våó°¢çe6†FW"§G2Ò"Óç6…³5ÒÂ¦×2Ò"Óç6…³EÓ°¢–b†GrÓæf—'7B’$TEôôä4RƒCBÂ&ÖW6‚G&rg&öÒv÷&¶w&÷WVC¢F†RF6²òÖW6‚6†FW'26÷VçBF†V—"v÷&¶w&÷W2g&öÒ†G&vâg&öÒ’"ÂGrÓæf—'7B“°¢¶Vâ6WD&wVÖVçEF&ÆS¦uö÷BE7FvW3¤ÕDÅ&VæFW%7FvTö&¦V7EÓ²¶Vâ6WD&wVÖVçEF&ÆS¦uö×BE7FvW3¤ÕDÅ&VæFW%7FvTÖW6…Ó°¢å5T–çFVvW"GBÒG2bbG2Óç&bçFu³ÒòG2Óç&bçFu³Ò¢Â×BÒ×2Óç&bçFu³Òò×2Óç&bçFu³Ò¢°¢¶VâG&tÖW6…F‡&VFw&÷W3¤ÕDÅ6—¦TÖ¶R‚„å5T–çFVvW"–GrÓæ6÷VçBÂÂ’F‡&VG5W$ö&¦V7EF‡&VFw&÷W¤ÕDÅ6—¦TÖ¶R‡GBÂÂ’F‡&VG5W$ÖW6…F‡&VFw&÷W¤ÕDÅ6—¦TÖ¶R†×BÂÂ•Ó°¢V–çCcE÷BâÒ²¶uöÖW6…öG&w3°¢–b†âÃÒ‚ÇÂ†âb†âÒ’’’ÔÄôr‚&ÖW6‚G&rVÆÇS¢F6²W2ÖW6‚W2g&vÖVçBW2ÂVBv÷&¶w&÷W2‡F‡&VG2VÇRòVÇR“²VÆÇR6¶—VB6òf"v†–ÆRF†V—"—VÆ–æRv2'V–ÇB"Â‡Vç6–væVBÆöærÆöær–âÀ¢G2òG2Óæ¶W’¢"Ò"Â×2Óæ¶W’Â"Óç6…³Òò"Óç6…³ÒÓæ¶W’¢"Ò"ÂGrÓæ6÷VçBÂ‡Vç6–væVBÆöær—GBÂ‡Vç6–væVBÆöær–×BÂ‡Vç6–væVBÆöærÆöær–uöÖW6…÷v—B“°§Ð§7FF–2fö–BG&uò†6öç7BöG&r¦GrÂ6öç7BöG&uö–G‚¦F’“°§7FF–2fö–BG&r†6öç7BöG&r¦GrÂ6öç7BöG&uö–G‚¦F’’²7÷7F×7F'BÒ7ö&Vv–â‚“²G&uò†GrÂF’“²7öVæBƒRÂ7F'B“²Ð§7FF–2fö–BG&uò†6öç7BöG&r¦GrÂ6öç7BöG&uö–G‚¦F’’°¢uöG&uö&–ç7BÒ‡V–çC3%÷B’†GròGrÓæ&6Uö–ç7Fæ6R¢F’òF’Óæ&6Uö–ç7Fæ6R¢“°¢6öç7B–çBÖW6‚ÒGrbbGrÓç&–ÒÓÒådåõ$”ÕôÔU4ƒ²òòâådã"ÖW6‚G&r†ÖW6…÷6òòÖW6…öVæ6öFR’òò3²ƒcCBöbF†R7FvW2F†B&VBF†RG&—fW"6VvÖVçB†Væ6öFUö&w2¢–b‡76GV×‚’’²çe&ör§ÒuövVõ÷&öw&ÒòuövVõ÷&öw&Ò¢2ç&örò&öuövWB…2ç&ör’¢æ–Ã°¢B‚&¢VÆÇRG&rÖVçG'’g3ÒW2g3ÒW2w3ÒW2&–ÓÒVB6÷VçCÒVB–ç7Fæ6W3ÒVB–æFW†VCÒVB–æF—&V7CÒR6ÆÇ‚F—66&CÒVBfVVF&6³ÒVB"Âu÷7FE÷72²uöVæ2À¢bbÓç6…³ÒòÓç6…³ÒÓæ¶W’¢"Ò"ÂbbÓç6…³ÒòÓç6…³ÒÓæ¶W’¢"Ò"ÂbbÓç6…³%ÒòÓç6…³%ÒÓæ¶W’¢"Ò"À¢GròGrÓç&–Ò¢F’Óç&–ÒÂGròGrÓæ6÷VçB¢F’Óæ6÷VçBÂGròGrÓæ–ç7Fæ6W2¢F’Óæ–ç7Fæ6W2ÂF’ÒåTÄÂÀ¢‡Vç6–væVBÆöærÆöær’†GròGrÓæ–æB¢F’Óæ–æB’Â2ç&7EöF—66&BÂ2çFeööâ“²Ð¢òòâ–æF—&V7BG&rw2&wVÖVçG2†æBâ–æFW†VBöæRw2f—'7B–æF–6W2’BVæ6öFRæB2F†RuRÆVgBF†VÒ‡&VBöæ6RF†R7V&Ö—76–öâ6ö×ÆWFVBÂgFW"ç’uR7VÆÆ–ærV&Æ–W"–â—@¢òòw&÷FRF†VÒ“¢–â72ÖGV×VBg&ÖW2ÂæBf÷"F†Rf—'7Bâ–æF—&V7BG&w2v—F‚DÕôådåô”äDÄôsÖà¢7FF–2Æöær–æFÆörÒÓ²–b†–æFÆörÂ’²6öç7B6†"§bÒvWFVçb‚$DÕôådåô”äDÄôr"“²–æFÆörÒbòFöÂ‡b’¢²Ð¢V–çCcE÷B–ÒGròGrÓæ–æB¢F’Óæ–æC°¢–b†–bb‡76GV×‚’ÇÂ–æFÆörâ’’²–b†–æFÆörâ’–æFÆörÒÓ°¢Vç6–væVBÆöærÆöær72Òu÷7FE÷72²uöVæ2Âg"Ò‡Vç6–væVBÆöærÆöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR“²V–çCcE÷B–"ÒF’òF’Óæ–'Vb¢²–çB—7¢ÒF’ò†F’Óæ—G—RÓÒò¢F’Óæ—G—RÓÒò"¢B’¢°¢V–çC3%÷BUõ³%ÒÒ³Ó²fÕ÷6—¦U÷BVrÒ²–çBVö²ÒfÕ÷&VEö÷fW'w&—FR†Ö6…÷F6µ÷6VÆb‚’Â‡fÕöFG&W75÷B––Â6—¦VöbUòÂ‡fÕöFG&W75÷B–UòÂfVr’ÓÒ´U$åõ5T44U53²V–çC3%÷BSÒUõ³ÒÂSÒUõ³Ó°¢¶uö6öÖÖ—Eö&Æö6·2FDö&¦V7C¥ç°¢V–çC3%÷B³eÒÒ³ÒÂ—…³eÒÒ³Ó²fÕ÷6—¦U÷Bv÷BÒ²–çBö²ÒfÕ÷&VEö÷fW'w&—FR†Ö6…÷F6µ÷6VÆb‚’Â‡fÕöFG&W75÷B––Â6—¦VöbÂ‡fÕöFG&W75÷B–Âfv÷B’ÓÒ´U$åõ5T44U53°¢V–çC…÷B&u³#EÒÒ³Ó²–çBöµö’Òö²bb–"bbfÕ÷&VEö÷fW'w&—FR†Ö6…÷F6µ÷6VÆb‚’Â‡fÕöFG&W75÷B’†–"²‡V–çCcE÷B–³%Ò¢‡V–çCcE÷B–—7¢’Â‡fÕ÷6—¦U÷B’ƒb¢—7¢’Â‡fÕöFG&W75÷B—&rÂfv÷B’ÓÒ´U$åõ5T44U53°¢f÷"†–çB²Ò²öµö’bb²Âc²²²²’—…¶µÒÒ—7¢ÓÒò&u¶µÒ¢—7¢ÓÒ"ò‚‡V–çCe÷B¢—&r•¶µÒ¢‚‡V–çC3%÷B¢—&r•¶µÓ°¢–b‚ö²’ÔÄôr‚%·75Ò–æF—&V7B&w2bVÆÇR72VÆÇRBR6ÆÇƒ¢Vç&VF&ÆR"Âg"Â72Â‡Vç6–væVBÆöærÆöær––“°¢VÇ6RÔÄôr‚%·75Ò–æF—&V7B&w2bVÆÇR72VÆÇRBR6ÆÇƒ¢WRWRWRVBWR†BVæ6öFRW2WRWR’W2–æF–6W2WRWRWRWRWRWS²v÷&G2RÓRW‚W‚W‚W‚W‚W‚W‚W‚W‚W‚W‚"Âg"Â72Â‡Vç6–væVBÆöærÆöær––À¢³ÒÂ³ÒÂ³%ÒÂ†–çB–³5ÒÂ³EÒÂVö²ò""¢'Vç&VF&ÆR"ÂSÂSÂöµö’ò""¢"†–æFW‚'VffW"Vç&VF&ÆR’"Â—…³ÒÂ—…³ÒÂ—…³%ÒÂ—…³5ÒÂ—…³EÒÂ—…³UÒÀ¢³UÒÂ³eÒÂ³uÒÂ³…ÒÂ³•ÒÂ³ÒÂ³ÒÂ³%ÒÂ³5ÒÂ³EÒÂ³UÒ“²ÕÓ²Ð¢–b‚uövVõ÷&öw&Òbb2ç&ör’°¢çe&ör¦vVöÖWG'•÷&öw&ÒÒ&öuövWB…2ç&ör“°¢–b†vVöÖWG'•÷&öw&ÒbbvVöÖWG'•÷&öw&ÒÓç6…³%ÒbbvVöÖWG'•÷&öw&ÒÓç6…³%ÒÓævVõ÷fW'F–6W2’°¢–b‚†GrbbGrÓæÖF•ö6çBbbÖF•öÆ—fR†GrÓæÖF•ö6çBÂGrÓæÖF•ö’’’ÇÂ†F’bbF’ÓæÖF•ö6çBbbÖF•öÆ—fR†F’ÓæÖF•ö6çBÂF’ÓæÖF•ö’’’’&WGW&ã°¢–ÖvU÷&W&R†vVöÖWG'•÷&öw&ÒÂƒRÃÂ’ÂƒRÃÂ"’“°¢vVõöG&r†vVöÖWG'•÷&öw&ÒÂGrÂF’Â“²&WGW&ã°¢Ð¢Ð¢–b†uöVæ2bbuöVæ5övVõöæö6öÆ÷"ÒuövVõööÖ—Eö6öÆ÷'2’'öVæB‚“°¢–çBÖG…ööâÒ†GrbbGrÓæÖF•ö6çB’ÇÂ†F’bbF’ÓæÖF•ö6çB’òÖG…÷vçFVB†GrÂF’’¢²òòâ–æF—&V7BÖ6÷VçB×VÇF’ÖG&rw2&V6÷&BÖ6¶VBöâF†RuR†ÖG…÷'Vâ’ÂVÇ6R&VBgFW"v—@¢–b‚ÖG…ööâbb‚†F’bbF’ÓæÖF•ö6çBbbÖF•öÆ—fR†F’ÓæÖF•ö6çBÂF’ÓæÖF•ö’’’ÇÂ†GrbbGrÓæÖF•ö6çBbbÖF•öÆ—fR†GrÓæÖF•ö6çBÂGrÓæÖF•ö’’’’’&WGW&ã²òò7BF†R×VÇF’ÖG&rw26÷Vç@¢–b†ÖG…ööâbbÖG…öæW‡B†GrÂF’’’&WGW&ã²òòF†RæW‡BG&röbÖ6¶VB×VÇF’ÖG&s¢öæÇ’F†RG&r—2Væ6öFV@¢uöÖG†2ææW‡BÒåTÄÃ°¢–b‚2ç&ör’²$TEôôä4Rƒ‚Â&æò&öw&Ò&÷VæB&Vf÷&RF†RG&r"“²B‚%¢VÆÇRG&r6¶—VC¢æò&öw&Ò&÷VæB"Âu÷7FE÷72²uöVæ2“²&WGW&ã²Ð¢õ÷Vç6fU÷Vç&WF–æVBçe&ör§"ÒuövVõ÷&öw&ÒòuövVõ÷&öw&Ò¢&öuövWB…2ç&ör“°¢–b†ÖW6‚bb‚"ÇÂ"Óç6…³EÒÇÂ"Óç6…³EÒÓæfâ’’²$TEôôä4RƒCRÂ&ÖW6‚G&rv—F‚æòÖW6‚6†FW"&÷VæB†÷"—Bf–ÆVBFòÆöB“¢6¶—VB"“²&WGW&ã²Ð¢–b‚ÖW6‚bb"bb"Óç6…³EÒbb"Óç6…³Ò’²B‚%¢VÆÇRG&r6¶—VC¢ÖW6‚&öw&Ò—2&÷VæBFòfW'FW‚G&r"Âu÷7FE÷72²uöVæ2“²&WGW&ã²Ð¢–ÖvU÷&W&R‡"Âƒb“°¢G&uöw5ö6†V6²‡"“²òòvVöÖWG'’7FvRÆVgBv—F‚æògVæ7F–öâ‡6VR&öuöföÆEöw2¢–b‚uövVõ÷&öw&Òbb2ç&7EöF—66&Bbb2çFeööâ’²FeöG&r‡"ÂGrÂF’“²&WGW&ã²Ð¢òòfVVF&6²6GW&W2&7FW&—¦VBG&w2Föò„Föæ¶W’¶öær6÷VçG'“¢G&÷–6Âg&VW¦Rw2gW"’à¢–b‚uövVõ÷&öw&Òbb2çFeööâbb"bb"Óç6…³Òbb"Óç6…³ÒÓçFeöfâ’FeöG&r‡"ÂGrÂF’“°¢òò&7FW&—¦W"F—66&B7W&W76W2g&vÖVçG2Âæ÷BfW'FW‚W†V7WF–öâ÷"—G27F÷&vRæB–ÖvRw&—FW2‡F†R'V–ÆBv—fW27V6‚fW'FW‚6†FW"Ö–ãöF—66&B’à¢òò†Væö&ÆFR6‡&öæ–6ÆW22W6W2fW'FW‚ÖöæÇ’v÷&²&Vf÷&R—G26¶–ææVBÖÖöFVÂG&w3²ÖWG&ö–B&–ÖR&VÖ7FW&VBf–ÆÇ2—G2förföÇVÖRv—F‚–ÖvR7F÷&W2g&öÒö–çG2à¢–b…2ç&7EöF—66&Bbb‚"Óç6…³ÒÇÂ"Óç6…³ÒÓæF—66&Eöfâ’’²B‚%¢VÆÇRG&r6¶—VC¢&7FW&—¦W"F—66&Bv—F‚æòfW'FW‚7F÷&vR÷"–ÖvRw&—FW2g3ÒW2"Âu÷7FE÷72²uöVæ2Â"Óç6…³Òò"Óç6…³ÒÓæ¶W’¢"Ò"“²&WGW&ã²Ð¢–b†uöçV&Bbb†uöVæ2ÇÂV&EöÆ§’‚’’bbV&Eö6öæfÆ–7B‡"ÂGrÂF’’’²u÷V&Eö6öæfÆ–7G2²³²'öVæB‚“²V&EöÆæB‚“²ÒòòÆæBF†RFVfW'&VB–æÆ–æRWFFW2&Vf÷&RF†—2G&r‡6VRV&Eö6öæfÆ–7B¢fW'FW…öGV×‡"Óç6…³ÒÂGrÂF’“°¢–b‡§6¶—ö6åöG&r‡"’’²òò¦W&ò×6÷W&6R6¶—¢F†R÷WGWB—2Çv—2Âw&—F–ær÷fW"Â6òF†—2G&r6†ævW2æò&W7VÇB‡6VR§6¶—ö6åöG&r¢u÷§6¶—ö6÷VçB²³²²7FF–2–çBöæ6S²–b‚öæ6R²²’ÔÄôr‚'¦W&ò×6÷W&6R6¶—¢f—'7B66R72VÆÇR"Âu÷7FE÷72²“²Ð¢B‚$âVÆÇR¦W&ò×6÷W&6RG&r6¶—VB"Âu÷7FE÷72²“²&WGW&ã²Ð¢–b†uöæÆ2bbuöVæ2’Æ5÷F¶R‡"“²òòv—F–ær6ÆV'3¢F†—272w2ÆöB7F–öç2Â÷"ÆæFVB&Vf÷&R—B‡6VRÆ5÷F¶R¢–b†—†5÷vçFVB†GrÂF’’bb—†5öf–æB†GrÂF’’’—†5÷'Vâ†GrÂF’“²òò–æF—&V7BVG2òfç2ò‚Ö&—B–æF–6W3¢6öçfW'FVBöâF†RuR&Vf÷&RF†R&VæFW"72ÂBF†V—"÷vâÆ6R–âF†R&WÆ’‡6VR—†5÷'Vâ¢V–çCcE÷BÖG‚Ò²òòF†RÖ6¶VB6÷’öbF†R&V6÷&B†ÖG…÷'Vâ“²G&r—B6÷VÆBæ÷B6÷’v—G2f÷"F†R6÷VçB–ç7FV@¢–b†ÖG…ööâ’²V–çCcE÷B–æBÒGròGrÓæ–æB¢F’Óæ–æBÂ6çBÒGròGrÓæÖF•ö6çB¢F’ÓæÖF•ö6çC²–çB’ÒGròGrÓæÖF•ö’¢F’ÓæÖF•ö“°¢–b‚ÖG…öf–æB†–æBÂ6çBÂfÖG‚’’²ÖG…÷'Vâ†GrÂF’“²–b‚ÖG…öf–æB†–æBÂ6çBÂfÖG‚’’ÖG‚Ò²Ð¢–b‚ÖG‚bbÖF•öÆ—fR†6çBÂ’’’&WGW&ã²Ð¢–b†F’bbv–FU÷vçFVB†F’’bbv–FUöf–æB†F’’’v–FU÷'Vâ†F’“²òò‚Ö&—B–æF–6W2w&—GFVâV&Æ–W"–âF†—27V&Ö—76–öã¢v–FVæVBöâF†RuR&Vf÷&RF†R&VæFW"72‡6VRv–FU÷'Vâ¢–b‚uöFVfW%÷&WÆ––ærbbuövVõ÷&öw&Ò’°¢–b†uöFVfW"æ7F—fRbbuöFVfW"æ–å÷bbuöFVfW"æE÷’²–b†FVfW%÷F¶R†GrÂF’’’&WGW&ã²Ð¢Ð¢òòG&w2F†B6ææ÷BG&rç—F†–ær†&÷F‚f6W27VÆÆVBÂV×G’66—76÷"’&WGW&â&Vf÷&RF†R&VæFW"72—2÷VæVC¢÷F†W'v—6RF†R&VæFW"72—2Ç&VG’÷VâæBöâD$E"F†Rv†öÆRGF6†ÖVçB—2ÆöFVBæB7F÷&VBf÷"æ÷F†–æp¢òò„Ö26†&7FW"6VÆV7BvS¢2&VæFW"76W2&R÷VæVBöæÇ’&V6W6Röb7V6‚G&w2ÂÆÂÆöFVBÂ¦W&òG&w2¢òòF–væ÷7F–27v—F6‚f÷"ÖV7W&–ærâWW"&÷VæBDÕôådåõ4´•ôe3ÓÆg&vÖVçB¶W“å²ÃÆg&vÖVçB¶W“âââåÒ†æöæR'’FVfVÇB“¢G&÷2ÆÂG&w2öbF†W6Rg&vÖVçB6†FW'2‡F†R–7GW&Rv–ÆÂ&Rw&öæs²öæÇ’W6VB–â&VÂÖFWf–6R$FòÖV7W&R&†÷r×V6‚uRF–ÖRF†—26Æ72öbG&w2—2v÷'F‚–âF÷FÂ"¢²7FF–26†"6µ³eÕ³#Ó²7FF–2–çBç6²ÒÓ°¢–b†ç6²Â’²ç6²Ò²f÷"†6öç7B6†"§ÒvWFVçb‚$DÕôådåõ4´•ôe2"“²bb§bbç6²Âc²Ò7G&6‡"‡ÂrÂr’ò7G&6‡"‡ÂrÂr’²¢²7G&ÆVâ‡’’²6—¦U÷BâÒ7G&77â‡Â"Â"“²–b†ââb’âÒc²ÖVÖ7’‡6µ¶ç6µÒÂÂâ“²6µ¶ç6²²µÕ¶åÒÒ²Ð¢–b†ç6²’ÔÄôr‚$DÕôådåõ4´•ôe3¢G&÷–ærF†RG&w2öbVBg&vÖVçB6†FW'2†F–væ÷7F–3²F†R–7GW&R—2w&öær’"Âç6²“²Ð¢–b†u÷V%öââbb"Óç6…³ÒbbuöFVfW%÷&WÆ––ærbbV%÷6¶—‡"Óç6…³ÒÓæ¶W’’’&WGW&ã²òò–â×&ö6W72ÇFW&æF–ærÖV7W&VÖVçC¢F†—2†6RG&÷2F†—26Æ70¢–b†ç6²bb"Óç6…³ÒbbuöFVfW%÷&WÆ––ær’f÷"†–çB’Ò²’Âç6³²’²²’–b‚7G&æ6×‡"Óç6…³ÒÓæ¶W’Â6µ¶•ÒÂ7G&ÆVâ‡6µ¶•Ò’’’&WGW&ã°¢–b†×5öG&r‡"’’&WGW&ã²Ð¢òò6–Ö–Æ"F–væ÷7F–2DÕôådåõ4´•õ4ÔÄÃÓÇv–GFƒã¢G&÷ÆÂG&w2v†÷6RF&vWBv–GF‚†'’F†RGF6†VBÆWfVÂ’—2ÃÒF†—2fÇVRÒÒÖV7W&W2†÷r×V6‚&G&÷–ærF–ç’&VæFW"76W2–â'VÆ²"—2v÷'F‚VæFW"F†R&ÆÆVÂ—VÆ–æR†f—†VBW"×72÷fW&†VB¢²7FF–2–çB6ÖÆÂÒÓ²–b‡6ÖÆÂÂ’²6öç7B6†"¦RÒvWFVçb‚$DÕôådåõ4´•õ4ÔÄÂ"“²6ÖÆÂÒRòFö’†R’¢²–b‡6ÖÆÂ’ÔÄôr‚$DÕôådåõ4´•õ4ÔÄÃ¢G&÷–ærG&w2v†÷6RF&vWBv–GF‚—2(šBVB†F–væ÷7F–3²F†R–7GW&R—2w&öær’"Â6ÖÆÂ“²Ð¢–b‡6ÖÆÂâbb"æ†5÷'BbbuöFVfW%÷&WÆ––ær’²–çB²Ò"ç'Bæâbb"ç'Bæ6öÆ÷%³Òò¢"ç'BæFWF‚ò‚¢Ó°¢–b†²ãÒ’²V–çCcE÷B5ó²–çBÇbÂ6Ã²'E÷7V"‚e"ç'BÂ²Âg5òÂfÇbÂg6Â“²6öç7BFW…÷B§BÒ²ÓÒ‚ò"ç'BæFWF‚¢"ç'Bæ6öÆ÷%³Ó²–b‚†–çB”Ô‚‡BÓæBçv–GF‚ãâÇbÂR’ÃÒ6ÖÆÂ’&WGW&ã²ÒÒÐ¢–b…2çöÇ’æ7VÆÂÓÒådåôd4Uôe$ôåEôäEô$4²bb2ç&7EöF—66&B’²B‚%¢VÆÇRG&r6¶—VB†&Vf÷&RF†R72&Vv–ç2“¢&÷F‚f6W27VÆÆVB"Âu÷7FE÷72²uöVæ2“²&WGW&ã²Ð¢òòF†R6öçF–çVVB&VæFW"72Ö’†fR&VVâ÷VæVBv—F‚fWvW"FWF‚ò7FVæ6–ÂGF6†ÖVçG2†÷VâÖöFW2"ò2òB“¢–bF†RÆæRF†—2G&ræVVG2—2æ÷BGF6†VBÂ7Æ—B&Vf÷&R—B†öâ&V÷VâÂGF6‚W"F†RæWrÆöö¶†VB¢–b†u÷'EöÖW&vVBbbuöVæ5öÖöFRÒbb‚‚…2æG2æGBÇÂ2æG2æGr’bbu÷72æ†B’ÇÂ…2æG2ç7Bbbu÷72æ‡2’’’°¢7FF–2–çBöæ6S#²–b‚öæ6S"²²’ÔÄôr‚'6ÖR×F&vWB6öçF–çVF–öã¢G&r–âF†R6öçF–çVVB72æVVG2FWF‚ò7FVæ6–ÂÆæRF†B—2æ÷BGF6†VC²7Æ—B†W&R†f—'7B66R72VÆÇR’"Âu÷7FE÷72“°¢B‚$ÒVÆÇRG&r–âF†R6öçF–çVVB72æVVG2FWF‚ò7FVæ6–ÂÆæRF†B—2æ÷BGF6†VC²7Æ—B†W&R"Âu÷7FE÷72“²'öVæB‚“²Ð¢–b†u÷'E÷&ævUö7F—fRbb‚'E÷&ævU÷%÷&ò‡"’ÇÂ2æG2æGBÇÂ2æG2æGrÇÂ2æG2ç7B’’'öVæB‚“²òòF†RW‡FW&æÂw&—FRf—6–&–Æ—G’BF†R÷&–v–æÂVæ6öFW"&÷VæF'’—2æ÷B&VÖ÷fV@¢u÷6×ÆW5öf%öÆVgBÒF&G%ööâ‚“°¢–çB'E÷6×ÆVBÒu÷'EöÖW&vVBbbG&u÷6×ÆW5÷'B‡"“²u÷6×ÆW5öf%öÆVgBÒ°¢–b‡'E÷6×ÆVB’²7FF–2–çBöæ6S²–b‚öæ6R²²’ÔÄôr‚'6ÖR×F&vWB6öçF–çVF–öã¢G&r–âF†R6öçF–çVVB726×ÆW2âGF6†ÖVçC²7Æ—B†W&R†f—'7B66R72VÆÇR’"Âu÷7FE÷72“°¢B‚$ÒVÆÇRG&r–âF†R6öçF–çVVB726×ÆW2âGF6†ÖVçC²7Æ—B†W&R"Âu÷7FE÷72“²'öVæB‚“²Ð¢÷&FW%öG&r‡"Âuög¢æ7F—fRÇÂu÷6Õ÷72ÇÂuöFVfW%÷&WÆ––ær“²òòw&—FRgFW"â÷&FW&–ær&'&–W"–âF†R70¢7FF–2–CÄÕDÄgVæ7F–öãâg¥öfã²òò7V6–Æ—¦VBg&vÖVçBgVæ7F–öâf÷"F†—2G&röbF†RÆFW"72–âÖW&vVB&VæFW"72†æ–ÂÒW6RF†R÷&–v–æÂfW'6–öâ¢–b†uög¢æ7F—fRbbuöVæ2bbuög¢ç†6RÓÒ"bb†Æ÷ööâ‚’ÇÂg¥÷%öG&r‡"ÂGrÂF’Âuög¢æÆÖ"Âfg¥öfâ’’’²òò6ææ÷B&R&÷fVâ†÷"Æöv–2÷W&F–öâÂv†–6‚æVVG2F†R6†FW"w2÷vâF&vWG3¢Æ÷ö¶W’“¢VæBF†RÖW&vVB&VæFW"72‡F†RV&Æ–W"72w2&W7VÇB—27F÷&VB2W7VÂ’æB&V÷Vâv—F‚F†RÆFW"72w2÷&–v–æÂF&vWG0¢7FF–2–çBöæ6S3²–b‚öæ6S2²²’ÔÄôr‚'&VF&ÆR×F–ÆR72ÖW&vS¢G&röbF†RÆFW"72—2æ÷B&÷f&ÆS²&6²Fò6W&FR76W2†f—'7B66R72VÆÇR’"Âu÷7FE÷72“°¢uög¥öfÆÆ&6²²³²'öVæB‚“²Ð¢–b†uöVæ2bbu÷f—5ö6÷VçF–ærbbu÷f—5öâãÒd•5õ4ÄõE2’'öVæB‚“²òòF†—2G&rw2VW'’æVVG26Æ÷BæBF†R'VffW"—2&÷VæBf÷"F†Rv†öÆR73¢F†RæW‡B72&Vv–ç2v—F‚æWröæR‡'ö&Vv–åöW‚¢–b†uöVæ2bbu÷'örbb‚"æ†5÷62ÇÂ†Æöær•"ç62ç‚²"ç62çrâ†Æöær–u÷'örÇÂ†Æöær•"ç62ç’²"ç62æ‚â†Æöær–u÷'ö‚’’'öVæB‚“²òòÆVfW2F†R72w2&VæFW"&V¢f%÷&W&R‡"“²òò6ÖR×72fVVF&6²6×Æ–æs¢6÷’F†R6æ6†÷Bf—'7B‡6VRf%÷&W&R¢–b…"æ†5÷62bbuöVæ2bb2ç&7EöF—66&B’°¢å5T–çFVvW"GsÒÂFƒÒ²'EöF–×2‚gGsÂgFƒ“°¢ÆöærƒÒ"ç62ç‚Âò¢"ç62ç‚Â“Ò"ç62ç’Âò¢"ç62ç’ÂƒÒ†Æöær•"ç62ç‚²"ç62çrÂ“Ò†Æöær•"ç62ç’²"ç62æƒ°¢–b‡ƒâ†Æöær—Gs’ƒÒGs²–b‡“â†Æöær—Fƒ’“ÒFƒ°¢–b‡ƒÃÒƒÇÂ“ÃÒ“’²B‚%¢VÆÇRG&r6¶—VB†&Vf÷&RF†R72&Vv–ç2“¢V×G’66—76÷"‚VBÂVBVG‚VBÂF&vWBVÇW‚VÇR’"Âu÷7FE÷72²Â"ç62ç‚Â"ç62ç’Â"ç62çrÂ"ç62æ‚Â‡Vç6–væVBÆöær—GsÂ‡Vç6–væVBÆöær—Fƒ“²&WGW&ã²Ð¢Ð¢6–fFVbådåõ%Eõ$ätUõDU5@¢–b‚uöVæ2’'%ö&Vv–â‚“°¢6VæF–`¢–çBFVfW%÷öG&rÒuöVæ2bbuöFVfW%÷&WÆ––ærbbuöFVfW%öæGâ²òòF†RG&rF†B÷Vç2w2&VæFW"73¢gFW"F†R&VæFW"72—2÷VæVB2W7VÂF‡&÷Vv‚†ö—7E÷G'’ògW6U÷G'’Â&WÆ’f—'7BæBF†VâG&rF†—2öæP¢–b‚uöVæ2bbF&G%ööâ‚’bbuövVõ÷&öw&Ò’†ö—7E÷G'’‚“²òòf—'7B†ö—7BæBW†V7WFRF†RW&RÖ6ÆV"&VæFW"76W26æGv–6†VB&WGvVVâF†—2&VæFW"72æB—G26öçF–çVF–öâ†&Vf÷&R'ö&Vv–åöW‚Âæ÷Bö67W––ærF†—2&VæFW"72¢òòF†RFVfW'&ÂFV6—6–öâ6öÖW2gFW"†ö—7E÷G'“¢†ö—7FVBW&RÖ6ÆV"&VæFW"76W2&R6¶—VB–âF†RÆöö¶†VBÂæBöæÇ’F†Vâ6âw26ÖR×F&vWB6öçF–çVF–öâ6VvÖVçB&R&V6övæ—¦V@¢–b‚uöVæ2bbF&G%ööâ‚’bbuövVõ÷&öw&ÒbbFVfW%ö†5ö–å÷‚’bbuöFVfW%÷&WÆ––ærbbuög¢æ7F—fRbbuög¥ö6†V6¶–ærbbuöW†V5öæW‡B’²FVfW%÷G'’†GrÂF’“²–b†uöFVfW%ö7W"Óæ7F—fRbbuöFVfW%ö7W"Óæ–å÷’&WGW&ã²Ð¢–b‚uöVæ2bbF&G%ööâ‚’bbuög¥ö6†V6¶–ærbbuövVõ÷&öw&Ò’gW6U÷G'’‚“²òò&VF&ÆRF–ÆR&VæFW"72ÖW&vR—2öâ'’FVfVÇC²v†VâF†R7G'V7GW&Rò6ÖR×—†VÂ&ööbFöW2æ÷B†öÆB—B7F–ÆÂfÆÇ2&6²Fòæ÷&ÖÂ&VæFW"72ÂæB—B—2æ÷B&RÖVçFW&VBGW&–ærF†R4„T4²&VG&p¢–b‚uöVæ2bbF&G%ööâ‚’bbuövVõ÷&öw&Òbb76†ö—7Eööâ‚’’75ö†ö—7E÷G'’‡"ÂGrÂF’“²òò6VRDÕôådåô55ô„ô•5@¢–çBæVVBÒuöVæ2ÇÂF&G%ööâ‚’ò2¢75öG5öæVVB‚’ÂÖöFRÒFVfW%÷öG&rò¢æVVBÓÒò"¢æVVBÓÒ"ò2¢æVVBÓÒòB¢²òò"GF6‚æV—F†W"Â2GF6‚7FVæ6–ÂöæÇ’ÂBGF6‚FWF‚öæÇ¢–çB6¶—öBÒÖöFRÒ°¢–b‚uöVæ2’²uöfEöGrÒGs²uöfEöF’ÒF“²Ð¢u÷'öÆ–W&VBÒ"Óç6…³Òbb"Óç6…³ÒÓæÆ–W%ö÷WC°¢–b†uöVæ2bbu÷72æÆ–W&VBÒu÷'öÆ–W&VB’'öVæB‚“²òòÆ–W&VB72öæÇ’f÷"Æ–W"×w&—F–ærfW'FW‚6†FW'2‡6VR'ö&Vv–åöW‚¢–b†uöVæ2bbu÷6Õ÷72bbG&u÷6×ÆW5÷'B‡"’’²B‚$ÒVÆÇRG&r–âF†R6Æ–6RÖÖW&vVB726×ÆW2—G2GF6†ÖVçC²7Æ—B†W&R"Âu÷7FE÷72“²'öVæB‚“²Ð¢–b‚uöVæ2bbÖöFRÓÒbbu÷'öÆ–W&VBbbFVfW%÷öG&rbbuöW†V5öæW‡BbbF&G%ööâ‚’bbuövVõ÷&öw&Òbb6Õööâ‚’’6Õ÷Æâ‡"“°¢f'Ve÷&W&R‡"Óç6…³Òòg"Óç6…³ÒÓç&b¢åTÄÂÂGrÂF’“²òòâVæÆ–væVBuR×w&—GFVâ7G&VÒÂ÷"öæRF†RG&r&VG27BÂæVVG2—G2÷&FW&VB6÷’&Vf÷&RF†R&VæFW"Væ6öFW ¢–CÄÕDÅ&VæFW$6öÖÖæDVæ6öFW#âVâÒ'ö&Vv–åöW‚†ÖöFR“°¢u÷'öÆ–W&VBÒ²u÷6Õ÷ÆâæöâÒ°¢uöfEöGrÒåTÄÃ²uöfEöF’ÒåTÄÃ°¢–b‚Vâ’²B‚%¢VÆÇRG&r6¶—VC¢F†R&VæFW"72F–Bæ÷B&Vv–âg3ÒRãg2g3ÒRãg2"Âu÷7FE÷72Â"bb"Óç6…³Òò"Óç6…³ÒÓæ¶W’¢"Ò"Â"bb"Óç6…³Òò"Óç6…³ÒÓæ¶W’¢"Ò"“²&WGW&ã²Òòò¢Æ–æW3¢v‡’G&rv2G&÷VBBF†W6RV&Ç’&WGW&ç0¢–b†FVfW%÷öG&r’°¢FVfW%÷&WÆ’‚“°¢òò&WÆ–VBG&rÖ’†fRVæFVBw2Væ6öFW"FòÆ–vâuR×w&—GFVâ7G&VÒâ&W7F÷&RF†—2G&rw27G&V×2æBVæ6öFW"à¢u÷'öÆ–W&VBÒ"Óç6…³Òbb"Óç6…³ÒÓæÆ–W%ö÷WC°¢–b†uöVæ2bbu÷72æÆ–W&VBÒu÷'öÆ–W&VB’'öVæB‚“°¢f'Ve÷&W&R‡"Óç6…³Òòg"Óç6…³ÒÓç&b¢åTÄÂÂGrÂF’“²uöfEöGrÒGs²uöfEöF’ÒF“²VâÒ'ö&Vv–åöW‚†ÖöFR“²uöfEöGrÒåTÄÃ²uöfEöF’ÒåTÄÃ²u÷'öÆ–W&VBÒ°¢–b‚Vâ’&WGW&ã°¢Ð¢–b‡6¶—öB’²u÷7FEöG5ööÖ—B²³²–b†ÖöFRÓÒ2’u÷7FE÷5ööæÇ’²³²VÇ6R–b†ÖöFRÓÒB’u÷7FEöEööæÇ’²³²Ð¢òòF†R—VÆ–æRFWVæG2öæÇ’öâ‡&öw&ÒÂ&VæFW"72F&vWG2Â&ÆVæBò6öÆ÷"òw&—FRÖ6²ò×VÇF—6×ÆR7FFRÂfW'FW‚Æ–÷WB“¢–bF†W6RF–Bæ÷B6†ævRÂ&WW6RF†R&Wf–÷W2öæP¢7FF–2–CÄÕDÅ&VæFW%—VÆ–æU7FFSâ7W%÷3²7FF–2çe&ör¦7W%÷#²7FF–2V–çCcE÷B7W%÷5÷73²7FF–2–çB7W%÷VÆÃ²òò7W%÷VÆÃ¢‡–'&–B‡VÆÆ–ær’—VÆ–æRÂ6¶VBf÷"v–âBF†RæW‡BG&p¢–b†uög¢æ7F—fRbbuöFVfW%÷&WÆ––ær’²u÷6õöÆÖÒuög¢ç†6RÓÒòuög¢æÆÖ¢uög¢æÆÖ#²u÷6õög6fâÒuög¢ç†6RÓÒòæ–Â¢g¥öfã²u÷6õög¢Òuög¢ç†6RÓÒò‡6õög¥öFW62—³Ò¢uög¥öFW63²u÷6õöF—'G’Ò²Òòò&VF&ÆRF–ÆRÖW&vS¢fWF6‚F†R—VÆ–æR'’F†R‡—6–6ÂÓâÆöv–6ÂÖ–æræBF†R7V6–Æ—¦F–öà¢VÇ6R–b†uöf%÷G6WÓÒuöG&u÷6Wbbuöf%÷FfâbbuöFVfW%÷&WÆ––ær’²u÷6õöÆÖÒuöf%÷FÆÖ²u÷6õög6fâÒuöf%÷Ffã²u÷6õög¢Òuöf%÷FFW63²u÷6õöF—'G’Ò²Òòò6ÖR×72fVVF&6²&VBg&öÒF†RF–ÆR†f%÷F–ÆU÷Æâ¢–b†ÖW6‚’²7W%÷2ÒÖW6…÷6ò‡"“²7W%÷VÆÂÒ²7W%÷"Òæ–Ã²u÷6õöF—'G’Ò²–b‚7W%÷2’²uöÖW6…÷v—B²³²B‚%¢VÆÇRÖW6‚G&r6¶—VC¢—G2—VÆ–æR—2&V–ær'V–ÇB"Âu÷7FE÷72“²&WGW&ã²ÒÒòòâådã"ÖW6‚G&s¢F†RæW‡BG&rfWF6†W2—G2÷vâ—VÆ–æP¢VÇ6R–b†uövVõöÖW6‚æöâ’²7W%÷2ÒvVõöÖW6…÷6ò‡"“²7W%÷VÆÂÒ²7W%÷"Òæ–Ã²–b‚7W%÷2’²uövVõöÖW6‚æÖ—72Ò²&WGW&ã²ÒÒòòÖW6‚G&r†vVõöÖW6…öG&r“¢VÆÇ2—G27G&V×3²F†RæW‡BG&rfWF6†W2—G2÷vâ—VÆ–æS²æöæR–WC¢F†R6ö×WFRF‚G&w0¢VÇ6R–b†u÷6õöF—'G’ÇÂ"Ò7W%÷"ÇÂ7W%÷5÷72Òu÷72ç6W&–Â’²u÷6õö×Òu÷6Õ÷73²7W%÷2Ò6õövWB‡"Âæ–Â“²u÷6õö×Ò²7W%÷VÆÂÒu÷6õ÷VÆÃ²7W%÷"Ò#²7W%÷5÷72Òu÷72ç6W&–Ã²u÷6õöF—'G’Ò7W%÷2ÇÂ7W%÷VÆÃ²Ð¢–b‚7W%÷2bbu÷6Õ÷72’²òòG&rv†÷6R×Æ–f–VB—VÆ–æR—2æ÷B'V–ÇB–WB‡6Õö×övWBVWVVB—B“¢F†R6Æ–6RÖÖW&vVB72VæG2æBF†RG&r÷Vç272öb—G2÷và¢u÷6Õöçv—B²³²B‚$ÒVÆÇR6Æ–6RÖÖW&vVB73¢æò×Æ–f–VB—VÆ–æR–WBf÷"fW'FW‚W2g&vÖVçBW3²7Æ—B†W&R"Âu÷7FE÷72Â"Óç6…³Òò"Óç6…³ÒÓæ¶W’¢"Ò"Â"Óç6…³Òò"Óç6…³ÒÓæ¶W’¢"Ò"“°¢'öVæB‚“²VâÒ'ö&Vv–åöW‚†ÖöFR“²–b‚Vâ’&WGW&ã°¢7W%÷2Ò6õövWB‡"Âæ–Â“²7W%÷VÆÂÒu÷6õ÷VÆÃ²7W%÷"Ò#²7W%÷5÷72Òu÷72ç6W&–Ã²u÷6õöF—'G’Ò7W%÷2ÇÂ7W%÷VÆÃ²Ð¢V–çC3%÷B&7FW%öÖ6²ÒG&uö6öÆ÷%÷w&—FW2‡"“²òò6GW&RF†R‡—6–6ÂÖ–ær&Vf÷&R&W6WGF–ærF†RÖW&vVB—VÆ–æR7FFRà¢–b†u÷6õöÆÖ’²u÷6õöÆÖÒåTÄÃ²u÷6õög6fâÒæ–Ã²ÖV×6WB‚fu÷6õög¢ÂÂ6—¦Vöbu÷6õög¢“²u÷6õöF—'G’Ò²ÒòòF†RæW‡BG&rfWF6†W2v–â66÷&F–ærFò—G2÷vâ6—GVF–öà¢õ÷Vç6fU÷Vç&WF–æVB–CÄÕDÅ&VæFW%—VÆ–æU7FFSâ2Ò7W%÷3°¢–b‚2’²B‚%¢VÆÇRG&r6¶—VC¢æò&VæFW"—VÆ–æR‡fW'FW‚W2g&vÖVçBW2’"Âu÷7FE÷72Â"Óç6…³Òò"Óç6…³ÒÓæ¶W’¢"Ò"Â"Óç6…³Òò"Óç6…³ÒÓæ¶W’¢"Ò"“²&WGW&ã²Ð¢–b‡2ÒuöVæ5÷6ò’²Væ5÷%÷6ò†VâÂ2“²uöVæ5÷6òÒ3²Ð¢–b†u÷6Õ÷72bbu÷6ÕöÆ–VBÒu÷6Õö7W"’²òòF†RÆ–W"F†—2G&rvöW2Fò‡6VRu÷6Õ÷Æâ¢ÕDÅfW'FW„×Æ–f–6F–öåf–WtÖ–ærfÒÒ³Â‡V–çC3%÷B’†u÷6Õö7W"Òu÷6ÕöÆò—Ó²²†–CÄÕDÃE&VæFW$6öÖÖæDVæ6öFW#â–Vâ6WEfW'FW„×Æ–f–6F–öä6÷VçC£f–WtÖ–æw3¢gfÕÓ²u÷6ÕöÆ–VBÒu÷6Õö7W#²Ð¢7FF–2–CÄÕDÄFWF…7FVæ6–Å7FFSâ7W%öG3²7FF–2V–çCcE÷B7W%öG5÷73°¢òòF–væ÷7F–2DÕôådåôDUD„õe#Ög&vÖVçB6†FW"¶W’&Vf—‚Âââã¢F†÷6RG&w26ö×&RFWF‚v—F‚Åt•2Âv†–6‚FVÆÇ2'F†RFWF‚FW7B&V¦V7G2WfW'’g&vÖVçB"g&öÒ&æ÷F†–ær&7FW&—¦W2"‡v—F‚DÕôådåô„”Ä•DR“²Vç6WBÒVæ6†ævV@¢7FF–26†"F¶õ³…Õ³#EÓ²7FF–2–çBæF¶òÒÓ°¢–b†æF¶òÂ’²æF¶òÒ²6öç7B6†"¦RÒvWFVçb‚$DÕôådåôDUD„õe""“°¢f÷"†6öç7B6†"§ÒS²bb§bbæF¶òÂƒ²Ò7G&6‡"‡ÂrÂr’ò7G&6‡"‡ÂrÂr’²¢²7G&ÆVâ‡’’²6—¦U÷BâÒ7G&77â‡Â"Â"“²–b†ââ#2’âÒ#3²ÖVÖ7’†F¶õ¶æF¶õÒÂÂâ“²F¶õ¶æF¶ò²µÕ¶åÒÒ²ÒÐ¢–b†æF¶ò’²–çB÷g"Ò²f÷"†–çB’Ò²’ÂæF¶òbb"Óç6…³Ó²’²²’÷g"ÃÒ7G&æ6×‡"Óç6…³ÒÓæ¶W’ÂF¶õ¶•ÒÂ7G&ÆVâ†F¶õ¶•Ò’“°¢Vç6–væVBFcÒ‡Vç6–væVB•2æG2æFc²–b†÷g"’2æG2æFbÒƒ²7W%öG2ÒG75övWB†u÷72æ†BÂu÷72æ‡2“²2æG2æFbÒFc²7W%öG5÷72Òu÷72ç6W&–Ã²uöG75öF—'G’Ò²Ð¢VÇ6R–b†uöG75öF—'G’ÇÂ7W%öG5÷72Òu÷72ç6W&–Â’²7W%öG2ÒG75övWB†u÷72æ†BÂu÷72æ‡2“²7W%öG5÷72Òu÷72ç6W&–Ã²uöG75öF—'G’Ò²Ð¢–b†7W%öG2ÒuöVæ5öG72’²Væ5÷%öG72†VâÂ7W%öG2“²uöVæ5öG72Ò7W%öG3²Ð¢²V–çC3%÷B7"Ò2ç7E÷&Ve³Ó²òòVæ5÷%÷7FVæ6–Å÷&Vb6VæG2F†Rg&öçBfÇVRöæÇ¢–b‚†uöVBæö²bTEõ5$Tb’ÇÂuöVBç7&VbÒ7"’²Væ5÷%÷7FVæ6–Å÷&Vb†VâÂ7"Â7"“²uöVBç7&VbÒ7#²uöVBæö²ÃÒTEõ5$Tc²ÒÐ¢òòf–Ww÷'Bò66—76÷"†6Æ×VBv—F†–âF†RF&vWB6—¦R¢å5T–çFVvW"GrÒu÷72çGrÂF‚Òu÷72çFƒ°¢²ÕDÅf–Ww÷'BbÒ2æ†5÷gò„ÕDÅf–Ww÷'B—µ2çg³Õ³ÒÂ2çg³Õ³ÒÂ2çg³Õ³%ÒÂ2çg³Õ³5ÒÂ2æG%³ÒÂ2æG%³×Ò¢„ÕDÅf–Ww÷'B—³ÂÂGrÂF‚ÂÂÓ°¢–b†FöÖ–5öÆöEöW‡Æ–6—B‚fuö÷&–v–åöÆÂÂÖVÖ÷'•ö÷&FW%÷&VÆ†VB’’²bæ÷&–v–å’³Òbæ†V–v‡C²bæ†V–v‡BÒ×bæ†V–v‡C²ÒòòÆ÷vW"ÖÆVgB÷&–v–â‡6VRuö÷&–v–åöÆÂ¢òòf–Ww÷'B7v—§¦ÆS¢äTtD•dUõ’æVvFW26Æ—’ÂF†R6ÖR2F†Rf–Ww÷'BfÆ—VB&÷WB—G26VçG&S²f6–ær—2FV6–FVBgFW"—BöâF†RuR2–âÖWFÂÂ6òF†Rv–æF–ær7F—2à¢òò÷F†W"7v—§¦ÆW2††—2W†6†ævW2ÂæVvFVB‚ò¢òr’&Ræòf–Ww÷'B6†ævRæB&RG&vâVç7v—§¦ÆV@¢²V–çC3%÷B7s²ÖVÖ7’‚g7rÂ2çg÷7u³ÒÂB“°¢–b‡7rÓÒƒ’²bæ÷&–v–å’³Òbæ†V–v‡C²bæ†V–v‡BÒ×bæ†V–v‡C²Ð¢VÇ6R–b‡7r’$TEôôä4Rƒ‚Â'f–Ww÷'B7v—§¦ÆRVBÂVBÂVBÂVBæ÷B–×ÆVÖVçFVB†G&vâVç7v—§¦ÆVB’"Â2çg÷7u³Õ³ÒÂ2çg÷7u³Õ³Òâ"Â2çg÷7u³Õ³%ÒâBÂ2çg÷7u³Õ³5Òâb“²Ð¢–b‚†uöVBæö²bTEõe’ÇÂÖVÖ6×‚fuöVBçgÂgbÂ6—¦Vöbb’’²Væ5÷%÷f–Ww÷'B†VâÂb“²uöVBçgÒc²uöVBæö²ÃÒTEõe²ÒÐ¢ÕDÅ66—76÷%&V7B62Ò³ÂÂGrÂF‡Ó°¢–b…"æ†5÷62bb2ç&7EöF—66&B’²òò&7FW"7FFR6ææ÷B7W&W72fW'FW‚7F÷&vRw&—FW0¢ÆöærƒÒ"ç62ç‚Âò¢"ç62ç‚Â“Ò"ç62ç’ÃÒò¢"ç62ç’ÂƒÒ†Æöær•"ç62ç‚²"ç62çrÂ“Ò†Æöær•"ç62ç’²"ç62æƒ°¢–b‡ƒâ†Æöær—Gr’ƒÒGs²–b‡“â†Æöær—F‚’“ÒFƒ°¢–b‡ƒÃÒƒÇÂ“ÃÒ“’²B‚%¢VÆÇRG&r6¶—VC¢V×G’66—76÷"‚VBÂVBVG‚VBÂF&vWBVÇW‚VÇR’"Âu÷7FE÷72Â"ç62ç‚Â"ç62ç’Â"ç62çrÂ"ç62æ‚Â‡Vç6–væVBÆöær—GrÂ‡Vç6–væVBÆöær—F‚“²&WGW&ã²Ð¢62Ò„ÕDÅ66—76÷%&V7B—²„å5T–çFVvW"—ƒÂ„å5T–çFVvW"—“Â„å5T–çFVvW"’‡ƒÒƒ’Â„å5T–çFVvW"’‡“Ò“—Ó°¢Ð¢–b‚†uöVBæö²bTEõ42’ÇÂÖVÖ6×‚fuöVBç62Âg62Â6—¦Vöb62’’²Væ5÷%÷66—76÷"†VâÂ62“²uöVBç62Ò63²uöVBæö²ÃÒTEõ43²Ð¢–b…2çöÇ’æ7VÆÂÓÒådåôd4Uôe$ôåEôäEô$4²bb2ç&7EöF—66&B’²B‚%¢VÆÇRG&r6¶—VC¢&÷F‚f6W27VÆÆVB"Âu÷7FE÷72“²&WGW&ã²Ð¢²–çB7RÒ2çöÇ’æ7VÆÂÓÒådåôd4Uôe$ôåBòÕDÄ7VÆÄÖöFTg&öçB¢2çöÇ’æ7VÆÂÓÒådåôd4Uô$4²òÕDÄ7VÆÄÖöFT&6²¢ÕDÄ7VÆÄÖöFTæöæS°¢–çBv’Ò…2çöÇ’æg&öçBÓÒådåôe$ôåEôd4Uô45r’Òu÷VWVUö÷&–v–åöÆÂòÕDÅv–æF–æt6÷VçFW$6Æö6·v—6R¢ÕDÅv–æF–æt6Æö6·v—6S²òòådâw2&7FW"fÆ——26GW&VBv†VâF†RVWVR—2–æ—F–Æ—¦VBÂ&F†W"F†âv†VâF†RFWf–6R÷&–v–â6†ævW2à¢–çBf’Ò2çöÇ’æÖöFRÓÒådåõôÅ”tôåôÔôDUôÄ”äRòÕDÅG&–ævÆTf–ÆÄÖöFTÆ–æW2¢ÕDÅG&–ævÆTf–ÆÄÖöFTf–ÆÂÂ6ÂÒ2æFWF…ö6Æ×²òòFWF‚6Æ—öfbÒ6WDFWF„6Æ× ¢fÆöB&•³5ÒÒ³ÂÂÓ²–b…2çöÇ’æöfg6WEöVâbB’²&•³ÒÒ2çòçVæ—G3²&•³ÒÒ2çòæf7F÷#²&•³%ÒÒ2çòæ6Æ×²Ð¢–b‚†uöVBæö²bTEô5TÄÂ’ÇÂuöVBæ7VÆÂÒ7R’²Væ5÷%ö7VÆÂ†VâÂ„ÕDÄ7VÆÄÖöFR–7R“²uöVBæ7VÆÂÒ7S²uöVBæö²ÃÒTEô5TÄÃ²Ð¢–b‚†uöVBæö²bTEõt”äB’ÇÂuöVBçv–æBÒv’’²Væ5÷%÷v–æF–ær†VâÂ„ÕDÅv–æF–ær—v’“²uöVBçv–æBÒv“²uöVBæö²ÃÒTEõt”äC²Ð¢–b‚†uöVBæö²bTEôd”ÄÂ’ÇÂuöVBæf–ÆÂÒf’’²Væ5÷%öf–ÆÂ†VâÂ„ÕDÅG&–ævÆTf–ÆÄÖöFR–f’“²uöVBæf–ÆÂÒf“²uöVBæö²ÃÒTEôd”ÄÃ²Ð¢–b‚†uöVBæö²bTEô$”2’ÇÂÖVÖ6×†uöVBæ&–2Â&’Â6—¦Vöb&’’’²Væ5÷%öFWF…ö&–2†VâÂ&•³ÒÂ&•³ÒÂ&•³%Ò“²ÖVÖ7’†uöVBæ&–2Â&’Â6—¦Vöb&’“²uöVBæö²ÃÒTEô$”3²Ð¢–b‚†uöVBæö²bTEô4Ä•’ÇÂuöVBæ6Æ—Ò6Â’²Væ5÷%öFWF…ö6Æ—†VâÂ6Â“²uöVBæ6Æ—Ò6Ã²uöVBæö²ÃÒTEô4Ä•²Ð¢–b‚†uöVBæö²bTEô$ÄTäB’ÇÂÖVÖ6×†uöVBæ&ÆVæBÂ2æ&ÆVæEö6öÆ÷"Â6—¦VöbuöVBæ&ÆVæB’’²Væ5÷%ö&ÆVæEö6öÆ÷"†VâÂ2æ&ÆVæEö6öÆ÷%³ÒÂ2æ&ÆVæEö6öÆ÷%³ÒÂ2æ&ÆVæEö6öÆ÷%³%ÒÂ2æ&ÆVæEö6öÆ÷%³5Ò“²ÖVÖ7’†uöVBæ&ÆVæBÂ2æ&ÆVæEö6öÆ÷"Â6—¦VöbuöVBæ&ÆVæB“²uöVBæö²ÃÒTEô$ÄTäC²ÒÐ¢–b†ÖW6‚’·ÒòòæòfW'FW‚7G&V×3¢F†RÖW6‚7FvRÖ¶W2—G2fW'F–6W0¢VÇ6R–b†7W%÷VÆÂ’²VÆÅö&–æB†Vâ“²uö‡•öG&w2³ÒuövVõöÖW6‚æöã²ÒòòVÆÆ–ær—VÆ–æR&VG2F†R7G&V×2—G6VÆb†æò6öçfW'6–öâV—F†W#¢FÕöfWF6‚FV6öFW2WfW'’f÷&ÖB¢VÇ6R°¢&–æE÷f'Vg2†Vâ“°¢–b‡"Óç6…³Ò’f6öçeö&–æB†VâÂ"Óç6…³ÒÂ2ç&7EöF—66&BÂuövVõ÷g&ævRòuövVõ÷g&ævR¢GrÂuövVõ÷g&ævRòåTÄÂ¢F’“²òò–çFVvW"–çWG2fVBfÆöB×G—Rf÷&ÖG2÷"‚òbÖ&—B–çFVvW'2öbF†R÷F†W"6–vâ‡6VRf6öçeöæ6ö×Âf6öçeö–çB“²ÆVâvVöÖWG'’G&s¢F†RfW'F–6W2—G2–æF–6W2æÖP¢Ð¢uö¶Eöf÷&6RÒ¶EöG&r‡"ÂGrÂF’“°¢–b†ÖW6‚’²–b‡"Óç6…³5Ò’Væ6öFUö&w2†VâÂ"Óç6…³5ÒÂ2Â“²Væ6öFUö&w2†VâÂ"Óç6…³EÒÂBÂ“²Ð¢VÇ6R–b‡"Óç6…³Ò’Væ6öFUö&w2†VâÂ"Óç6…³ÒÂÂ“°¢–b‡"Óç6…³Òbb2ç&7EöF—66&B’Væ6öFUö&w2†VâÂ"Óç6…³ÒÂÂ“°¢òò&–æF–ærFW‡GW&Rf–Wr6âWw&FR—G2FW‡GW&Rf—'7B‡FW…÷Ww&FS¢f÷&ÖBf–WröbFW‡GW&RÖFRv—F†÷WB—†VÄf÷&ÖEf–Wr’ÂæBF†RWw&FRw26÷’vöW2F‡&÷Vv‚&Æ—B‚’À¢òòv†–6‚VæG2F†R&VæFW"73¢Vâ—2F†VââVæFVBVæ6öFW"ÂæBG&v–æröâ—B7&6†VB–âF†RG&—fW"„u‚&VG2—G2&VÆV6VB6öçFW‡C¢æVVBf÷"7VVC¢†÷BW'7V—B&VÖ7FW&VBÀ¢òòC“gƒ#BFW‡GW&R6×ÆVBF‡&÷Vv‚æ÷F†W"f÷&ÖBw2f–WrB—G2f—'7BÆöF–ær67&VVâ’âF†RFW‡GW&R—2Ww&FVBæ÷rÂ6òF†RG&r—2Væ6öFVBv–â–âæWr70¢–b†uöVæ2ÒVâ’°¢7FF–2–çBv–ã²–b†v–â’²$TEôôä4Rƒ3Â&&–æF–ærG&rw2&wVÖVçG2VæFVB—G2&VæFW"72Gv–6S²G&r6¶—VB"“²&WGW&ã²Ð¢B‚$ÒVÆÇR&–æF–ærF†RG&rw2&wVÖVçG2VæFVBF†R&VæFW"72‡FW‡GW&RWw&FR“²Væ6öFVBv–â"Âu÷7FE÷72“°¢v–âÒ²G&r†GrÂF’“²v–âÒ²&WGW&ã°¢Ð¢òò&ÖWFW"&–æF–ær6GW&W2F†R7GVÂ&W6÷W&6W2VæFW"F†R&Vv—7G'’Æö6³²–bF†RfW'6–öâ6†ævVB&WGvVVâF†RVÆ–v–&–Æ—G’6†V6²æBF†R&–æF–ærÂ7Æ—BæB&WG'’&Vf÷&Rç’uRG&rà¢–b†u÷'E÷&ævUö7F—fRbb†u÷72ç&VuövVâÒu÷&VuövVâÇÂu÷72ç6W%övVâÒu÷6W%övVâÇÂu÷72ç&ævUöF—'G’ÒçfåöF—'G•öæ÷r‚’’’²'öVæB‚“²G&r†GrÂF’“²&WGW&ã²Ð¢–çB&–ÒÒGròGrÓç&–Ò¢F’Óç&–Ó°¢ÕDÅ&–Ö—F—fUG—R×°¢7v—F6‚‡&–Ò’°¢66R¢×ÒÕDÅ&–Ö—F—fUG—Uö–çC²'&V³²66R¢×ÒÕDÅ&–Ö—F—fUG—TÆ–æS²'&V³²66R3¢×ÒÕDÅ&–Ö—F—fUG—TÆ–æU7G&—²'&V³°¢66RC¢66Rc¢66Rs¢×ÒÕDÅ&–Ö—F—fUG—UG&–ævÆS²'&V³²66RS¢×ÒÕDÅ&–Ö—F—fUG—UG&–ævÆU7G&—²'&V³²òòbG&–ævÆRfâÂrVG3¢ÖWFÂ†2æV—F†W"ÂW‡æFVB–çFòG&–ævÆRÆ—7G0¢66Rådåõ$”ÕôÔU4ƒ¢×ÒÕDÅ&–Ö—F—fUG—UG&–ævÆS²'&V³²òòF†RÖW6‚7FvRw2÷vâF÷öÆöw’†ÖW6…öVæ6öFR¢FVfVÇC¢²7FF–2–çBã²–b†â²²Â‚’ÔÄôr‚%µ$TEÒ&–Ö—F—fRG—RVBæ÷B–×ÆVÖVçFVB‡fW'FW‚W2‡VÆÂW2FöÖ–âW2g&vÖVçBW3²G&r6¶—VB’"Â&–ÒÂ"Óç6…³Òò"Óç6…³ÒÓæ¶W’¢"Ò"À¢"Óç6…³5Òò"Óç6…³5ÒÓæ¶W’¢"Ò"Â"Óç6…³EÒò"Óç6…³EÒÓæ¶W’¢"Ò"Â"Óç6…³Òò"Óç6…³ÒÓæ¶W’¢"Ò"“²Ð¢B‚%¢VÆÇRG&r6¶—VC¢&–Ö—F—fRG—RVB"Âu÷7FE÷72Â&–Ò“²&WGW&ã°¢Ð¢–b†u÷f—5ö6÷VçF–ær’²òòö66ÇW6–öâVW'’6÷VçF–æs¢æWr6Æ÷Bf÷"F†—2&VæFW"72†Ç6ò7v—F6‚FòæWr6Æ÷BgFW"&W÷'Bò&W6WB¢–b‚uöVæ5ö†5÷f—2’$TEôôä4Rƒ3BÂ&ö66ÇW6–öâVW'’6÷VçF–ær&Vvâ–â72v—F†÷WBf—6–&–Æ—G’'VffW"‡F†Rf—'7B&W6WB†VæVBÖ–B×72“²F†B72w26×ÆW2&Ræ÷B6÷VçFVB"“°¢VÇ6R°¢–b†u÷f—5ö7W"ÂÇÂu÷f—5ö7W%÷72Òu÷72ç6W&–Â’°¢–b†u÷f—5öâãÒd•5õ4ÄõE2’$TEôôä4Rƒ3RÂ'f—6–&–Æ—G’6Æ÷G2W††W7FVB"“°¢VÇ6R²u÷f—5ö7W"Ò†–çB–u÷f—5öâ²³²u÷f—5ö7W%÷72Òu÷72ç6W&–Ã°¢–b†uöçg2ÓÒu÷g66’²u÷g66Òu÷g66òu÷g66¢"¢cC²u÷g2Ò&VÆÆö2†u÷g2Âu÷g66¢6—¦Vöb¦u÷g2“²Ð¢u÷g5¶uöçg2²µÒÒ‡f—5÷6Æ÷B—²‡fö–B¢”4d'&–Fv–æu&WF–â†u÷f—2’Â‡V–çC3%÷B–u÷f—5ö7W'Ó²Ð¢Ð¢–b†u÷f—5ö7W"ãÒbbuöVæ5÷f—2Òu÷f—5ö7W"²’²Væ5÷%÷f—6–&–Æ—G’†VâÂÕDÅf—6–&–Æ—G•&W7VÇDÖöFT6÷VçF–ærÂ„å5T–çFVvW"–u÷f—5ö7W"¢‚“²uöVæ5÷f—2Òu÷f—5ö7W"²²Ð¢Ð¢Ð¢u÷7FEöG&r²³°¢–b†Væ5öÆ&VÅööâ‚’bbuöVÅöG&w2²²bb"Óç6…³Ò’²ÖVÖ7’†uöVÅö¶W’Â"Óç6…³ÒÓæ¶W’Â‚“²uöVÅö¶W•³…ÒÒ²Ð¢–b‡76GV×‚’’²òò6†&7FW&—7F–72öb&VæFW"72w2f—'7BG&r‡Fò§VFvRv†WF†W"F†RÆöB6â&RöÖ—GFVC¢gVÆÂ67&VVâÂæò&ÆVæF–ærÂw&—FW2ÆÂ6†ææVÇ2ÂFöW2æ÷BFWVæBöâFWF‚“²v—F‚DÕôådåõ54ETÕôÄÂWfW'’G&r—2&V6÷&FV@¢7FF–2V–çCcE÷BÆ7E÷72ÒãVÆÃ²7FF–2–çBÆÂÒÓ²–b†ÆÂÂ’ÆÂÒvWFVçb‚$DÕôådåõ54ETÕôÄÂ"’ÒåTÄÃ°¢–b†ÆÂÇÂuö¶Eöf÷&6R’B‚&BVÆÇR'cÒVB&“ÒVBf—'7CÒVBg3ÒW2g3ÒW2f3ÒVB–ç7CÒVB&ÆVæCÒR7‚#ÒR6ÆÇ‚GCÒVBGsÒVBFcÒVB7VÆÃÒVB7CÒVB6cÒVBÂVB6÷ÒVBÂVBÂVBòVBÂVBÂVB6Ö6³ÒR7‚ÂR7‚òR7‚ÂR7‚7&VcÒR7‚ÂR7‚–#ÒR6ÆÇ‚—CÒVB6Ö6³ÒR7‚#ÒR6ÆÇ‚##ÒR6ÆÇ‚#3ÒR6ÆÇ‚óÒVBòVrÂVrÂVrF3ÒVB"Âu÷7FE÷72ÂF’òF’Óæ&6U÷fW'FW‚¢ÂF’òF’Óæ&6Uö–ç7Fæ6R¢ÂGròGrÓæf—'7B¢Â"Óç6…³Òò"Óç6…³ÒÓæ¶W’¢"Ò"Â"Óç6…³Òò"Óç6…³ÒÓæ¶W’¢"Ò"À¢GròGrÓæ6÷VçB¢F’Óæ6÷VçBÂGròGrÓæ–ç7Fæ6W2¢F’Óæ–ç7Fæ6W2Â2æ6öÆ÷"æ&ÆVæEöVâÂ‡Vç6–væVBÆöærÆöær’¢‡V–çCcE÷B¢’e2æ&ÆVæE³ÒÂ†–çB•2æG2æGBÂ†–çB•2æG2æGrÂ†–çB•2æG2æFbÂ†–çB•2çöÇ’æ7VÆÂÀ¢†–çB•2æG2ç7BÂ†–çB•2æG2ç6eöbÂ†–çB•2æG2ç6eö"Â†–çB•2æG2æf–ÅöbÂ†–çB•2æG2ç¦f–ÅöbÂ†–çB•2æG2ç§75öbÀ¢†–çB•2æG2æf–Åö"Â†–çB•2æG2ç¦f–Åö"Â†–çB•2æG2ç§75ö"Â2ç7E÷fÖ6µ³ÒÂ2ç7E÷fÖ6µ³ÒÂ2ç7E÷vÖ6µ³ÒÂ2ç7E÷vÖ6µ³ÒÂ2ç7E÷&Ve³ÒÂ2ç7E÷&Ve³ÒÀ¢F’ò‡Vç6–væVBÆöærÆöær–F’Óæ–'Vb¢VÆÂÂF’òF’Óæ—G—R¢Â2æ6†Ö6²æÖ6²Â‡Vç6–væVBÆöærÆöær’¢‡V–çCcE÷B¢’e2æ&ÆVæE³ÒÂ‡Vç6–væVBÆöærÆöær’¢‡V–çCcE÷B¢’e2æ&ÆVæE³%ÒÂ‡Vç6–væVBÆöærÆöær’¢‡V–çCcE÷B¢’e2æ&ÆVæE³5ÒÀ¢†–çB•2çöÇ’æöfg6WEöVâÂ2çòæf7F÷"Â2çòçVæ—G2Â2çòæ6Æ×Â2æFWF…ö6Æ×“°¢–b†ÆÂbb"Óç6…³Ò’²òò6öçFVçBf–ævW'&–çBöbV6‚6öç7FçB'VffW"öbF†RfW'FW‚7FvR‡Fò§VFvRv†WF†W"G&r—26ö×ÆWFVÇ’–FVçF–6Â7&÷72g&ÖW3¢W6VB'’F†R6†F÷r66†RWF2â“²v†VâVç&VF&ÆR†w&—FRVæF–ærW†V7WF–öâ–âF†—27V&Ö—B’&V6÷&B ¢6†"V%³CÓ²6—¦U÷BòÒ²çe6†FW"§bÒ"Óç6…³Ó°¢f÷"†–çB’Ò²’ÂbÓç&bææ"bbòÂ6—¦VöbV"ÒC²’²²’–b‡bÓç&bæ%¶•Òæ¶–æBÓÒådåõ$%ô4%TbbbbÓç&bæ%¶•Òç&VrÂb’°¢6öç7Bö&–æEö'Vb§RÒe2çV&õ³Õ·bÓç&bæ%¶•Òç&VuÓ²fö–B¦Ö#²V–çCcE÷BÖó²6öç7BV–çC…÷B¦&6S²V–çCcE÷B‚ÒCc“S“ƒ3“3CccSc7VÆÃ°¢–b‚RÓæFG"ÇÂRÓç6—¦R’²ò³Ò6ç&–çFb‡V"²òÂ6—¦VöbV"ÒòÂ""VCÒÒ"ÂbÓç&bæ%¶•Òç&Vr“²6öçF–çVS²Ð¢–b‡VæEö†—G2‡RÓæFG"ÂRÓç6—¦R’ÇÂ×FÅöFG"‡RÓæFG"ÂRÓç6—¦RÂfÖ"ÂfÖò’ÇÂ†&6RÒ²…õö'&–FvR–CÄÕDÄ'VffW#â–Ö"6öçFVçG5Ò’’²ò³Ò6ç&–çFb‡V"²òÂ6—¦VöbV"ÒòÂ""VCÕòVÆÇR"ÂbÓç&bæ%¶•Òç&VrÂ‡Vç6–væVBÆöærÆöær—RÓç6—¦R“²6öçF–çVS²Ð¢V–çCcE÷BƒcBÒ²f÷"‡V–çCcE÷BÒ²ÂRÓç6—¦S²²²’²‚Ò†‚â&6U¶Öò²Ò’¢““Sc#ƒ#VÆÃ²–b‡ÓÒc2’ƒcBÒƒ²ÒòòƒcC¢F†Rf—'7BcB'—FW2‡v÷&ÆBÖG&—‚fV3(	32öbF†Rö&¦V7B'VffW"¢²7FF–2V–çCcE÷B‡ÒãVÆÃ²7FF–2–çB†ã²–b†‡Òu÷7FE÷72’²‡Òu÷7FE÷73²†âÒ²Òòòf÷"F†Rf—'7BbG&w2öbV6‚&VæFW"72ÂFF—F–öæÆÇ’&–çB'VffW'2öbÃÒc‚'—FW2–â†W‚‡Fò6ö×&R'—FR'’'—FRv†–6‚f–VÆG26†ævVB7&÷72g&ÖW2¢–b††âÂ"bbRÓç6—¦RÃÒc‚bbbÓç&bæ%¶•Òç&VrÓÒ2’²†â²³²6†"‡…³3Ó²f÷"‡V–çCcE÷BÒ²ÂRÓç6—¦S²²²’6ç&–çFb†‡‚²"¢Â2Â"S'‚"Â&6U¶Öò²Ò“²B‚'‚VÆÇR"VBW2"Âu÷7FE÷72ÂbÓç&bæ%¶•Òç&VrÂ‡‚“²ÒÐ¢ò³Ò6ç&–çFb‡V"²òÂ6—¦VöbV"ÒòÂ""VCÒSfÆÇ‚òVÆÇRòSfÆÇ‚"ÂbÓç&bæ%¶•Òç&VrÂ‡Vç6–væVBÆöærÆöær–‚Â‡Vç6–væVBÆöærÆöær—RÓç6—¦RÂ‡Vç6–væVBÆöærÆöær–ƒcB“²Ð¢B‚'RVÆÇRW2"Âu÷7FE÷72ÂV"“²Ð¢–b†ÆÂ’²6†"f%³cÓ²6—¦U÷BòÒ²ò³Ò6ç&–çFb‡f"²òÂ6—¦Vöbf"ÒòÂ&æGG#ÒVB"Â2ææGG"“°¢f÷"†–çB’Ò²’Â2ææGG"bb’ÂbbbòÂ6—¦Vöbf"Ò²’²²’²–çB7BÒ2æGG%¶•Òç7G&VÓ²–b‚2æGG%¶•Òæf×B’6öçF–çVS²òòöæÇ’GG&–'WFW2v—F‚f÷&Ö@¢ò³Ò6ç&–çFb‡f"²òÂ6—¦Vöbf"ÒòÂ"VG¶f×CÒR7‚öfcÒVB3ÒVB7G&–FSÒWRF—cÒWRf#ÒR6ÆÇ‚òVÆÇWÒ"Â’Â2æGG%¶•Òæf×BÂ2æGG%¶•Òæöfg6WBÂ7BÀ¢7BÂbò2ç7G&VÕ·7EÒç7G&–FR¢Â7BÂbò2ç7G&VÕ·7EÒæF—f—6÷"¢Â7BÂbò‡Vç6–væVBÆöærÆöær•2çf'Ve·7EÒæFG&W72¢VÆÂÂ7BÂbò‡Vç6–væVBÆöærÆöær•2çf'Ve·7EÒç6—¦R¢VÆÂ“²Ð¢B‚'bVÆÇRW2"Âu÷7FE÷72Âf"“°¢òòDÕôådåô4$ETÕ²DÕôådåô4$ETÕô´U“×fW'FW‚6†FW"¶W“¢Ç6ò6fRF†BG&rw2fW'FW‚7G&V×2æB–æF–6W2„cÆg&ÖSå÷Ç73å÷f#Ç7G&VÓåóÆFG&W73âæ&–âòö–%óÆFG&W73âæ&–â¢6öç7B6†"¦6&BÒvWFVçb‚$DÕôådåô4$ETÕ"’Â¦6&²ÒvWFVçb‚$DÕôådåô4$ETÕô´U’"“°¢–b†6&Bbb6&²bb"Óç6…³Òbb7G&6×‡"Óç6…³ÒÓæ¶W’Â6&²’’°¢6†"g³#EÓ°¢f÷"†–çB7BÒ²7BÂc²7B²²’–b…2çf'Ve·7EÒæFG&W72bb2çf'Ve·7EÒç6—¦R’°¢6ç&–çFb†gÂ6—¦VöbgÂ"W2ôbVÆÇU÷VÆÇU÷f"VEòVÆÇ‚æ&–â"Â6&BÂ‡Vç6–væVBÆöærÆöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR’Âu÷7FE÷72Â7BÂ‡Vç6–væVBÆöærÆöær•2çf'Ve·7EÒæFG&W72“°¢d”ÄR¦bÒf÷Vâ†gÂ'v""“²–b†b’²gw&—FR‚†6öç7Bfö–B¢•2çf'Ve·7EÒæFG&W72ÂÂ2çf'Ve·7EÒç6—¦RÂb“²f6Æ÷6R†b“²ÒÐ¢–b†F’bbF’Óæ–'Vb’°¢6ç&–çFb†gÂ6—¦VöbgÂ"W2ôbVÆÇU÷VÆÇUö–"VEòVÆÇ‚æ&–â"Â6&BÂ‡Vç6–væVBÆöærÆöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR’Âu÷7FE÷72ÂF’Óæ—G—RÂ‡Vç6–væVBÆöærÆöær–F’Óæ–'Vb“°¢d”ÄR¦bÒf÷Vâ†gÂ'v""“²–b†b’²gw&—FR‚†6öç7Bfö–B¢–F’Óæ–'VbÂÂ‡6—¦U÷B–F’Óæ6÷VçBÃÂ†F’Óæ—G—RÓÒ"ò"¢F’Óæ—G—R’Âb“²f6Æ÷6R†b“²ÒÐ¢ÒÐ¢–b†Æ7E÷72Òu÷7FE÷72’°¢Æ7E÷72Òu÷7FE÷73°¢B‚$BVÆÇR&–ÓÒVBf3ÒVB–ç7CÒVB&ÆVæCÒR7‚vÓÒR7‚GCÒVBGsÒVBFcÒVB7CÒVB63ÒVÇRÂVÇRÂVÇRÂVÇRFwCÒVÇW‚VÇRgÒRãbÂRãbÂRãbÂRãbG#ÒVrÂVrÆóÒWR"Âu÷7FE÷72Â&–ÒÀ¢GròGrÓæ6÷VçB¢F’Óæ6÷VçBÂGròGrÓæ–ç7Fæ6W2¢F’Óæ–ç7Fæ6W2Â2æ6öÆ÷"æ&ÆVæEöVâÂ2æ6†Ö6²æÖ6²Â†–çB•2æG2æGBÂ†–çB•2æG2æGrÂ†–çB•2æG2æFbÂ†–çB•2æG2ç7BÀ¢‡Vç6–væVBÆöær—62ç‚Â‡Vç6–væVBÆöær—62ç’Â‡Vç6–væVBÆöær—62çv–GF‚Â‡Vç6–væVBÆöær—62æ†V–v‡BÂ‡Vç6–væVBÆöær—GrÂ‡Vç6–væVBÆöær—F‚À¢2æ†5÷gò2çg³Õ³Ò¢Â2æ†5÷gò2çg³Õ³Ò¢Â2æ†5÷gò2çg³Õ³%Ò¢†fÆöB—GrÂ2æ†5÷gò2çg³Õ³5Ò¢†fÆöB—F‚Â2æ†5÷gò2æG%³Ò¢æbÂ2æ†5÷gò2æG%³Ò¢æbÂ‡Vç6–væVB•2æ6öÆ÷"æÆöv–5ö÷“°¢B‚'2VÆÇRg&öçB6×ÒWRf–ÃÒWR¦f–ÃÒWR73ÒWR&VcÒR7‚&VCÒR7‚w&—FSÒR7‚&6²6×ÒWRf–ÃÒWR¦f–ÃÒWR73ÒWR&VcÒR7‚&VCÒR7‚w&—FSÒR7‚7VÆÃÒWRv–æF–æsÒWR"Âu÷7FE÷72À¢2æG2ç6eöbÂ2æG2æf–ÅöbÂ2æG2ç¦f–ÅöbÂ2æG2ç§75öbÂ2ç7E÷&Ve³ÒÂ2ç7E÷fÖ6µ³ÒÂ2ç7E÷vÖ6µ³ÒÀ¢2æG2ç6eö"Â2æG2æf–Åö"Â2æG2ç¦f–Åö"Â2æG2ç§75ö"Â2ç7E÷&Ve³ÒÂ2ç7E÷fÖ6µ³ÒÂ2ç7E÷vÖ6µ³ÒÂ2çöÇ’æ7VÆÂÂ2çöÇ’æg&öçB“°¢Ð¢Ð¢uö¶Eöf÷&6RÒ°¢–b‡'E÷&ævUööâ‚’bb'E÷&ævU÷%÷&ò‡"’’u÷72ç&ævUöW‡FW&æÃÓ°¢'E÷&ævUöæ÷FUöFWF‚‚“°¢6–fFVbådåõ%Eõ$ätUõDU5@¢'%÷6æ†GrÆF’“°¢6VæF–`¢VF5öæ÷FUöG&r‡"Â&7FW%öÖ6²“°¢7w&—FUöæ÷FR‡"“²÷&FW%öæ÷FR‡"“°¢–b†ÖW6‚’²ÖW6…öVæ6öFR†VâÂ"ÂGr“²&WGW&ã²Ð¢–b†uövVõöÖW6‚æöâ’²vVõöÖW6…öVæ6öFR†Vâ“²&WGW&ã²Ð¢–b†uövVõ÷&öw&Ò’°¢–b†uövVõö6÷VçB’Væ5÷%öG&uö–æFW†VB†VâÂ×ÂuövVõö6÷VçBÂÕDÄ–æFW…G—UT–çC3"ÂuövVõö–æF–6W2ÂuövVõö–æFW…ö'—FW2ÂÂÂ‡V–çC3%÷B–uöG&uö&–ç7B“²òòvVõöÆVå÷G&’w26Æ÷G0¢VÇ6RVæ5÷%öG&uö–æFW†VEö–æF—&V7B†VâÂ×ÂÕDÄ–æFW…G—UT–çC3"ÂuövVõö–æF–6W2ÂuövVõö–æFW…ö'—FW2ÂuövVõö&w2“°¢&WGW&ã°¢Ð¢òòG&rv—F‚&–Ö—F—fW2‡F†RvÖRFöW26VæB6÷VçCÓ’G&w2æò—†VÇ2Â6ò6¶—Væ6öF–æs²ÖWFÂ&WV—&W2F†R6÷VçBFò&Ræöâ×¦W&òÂ÷F†W'v—6RF†RfÆ–FF–öâÆ–W"&÷'G2öâF†R7÷B†ÖV7W&VBöâF†R6×òv–ÆFW&æW72ò6öÖ&BvW2¢–b‚†GrbbGrÓæ–æB’ÇÂ†F’bbF’Óæ–æB’’²òòG&t'&—4–æF—&V7BòG&tVÆVÖVçG4–æF—&V7C¢ådâw2&wVÖVçB&V6÷&G2¶6÷VçBÂ–ç7Fæ6W2Âf—'7BÂ&6R–ç7Fæ6WÒæB¶6÷VçBÂ–ç7Fæ6W2Âf—'7B–æFW‚À¢òò&6RfW'FW‚Â&6R–ç7Fæ6WÒ†fRF†RÆ–÷WG2öbÕDÄG&u&–Ö—F—fW4–æF—&V7D&wVÖVçG2æBÕDÄG&t–æFW†VE&–Ö—F—fW4–æF—&V7D&wVÖVçG3²VG2Âfç2æB‚Ö&—B–æF–6W2W6RF†R6öçfW'FVBöæW2†—†5÷'Vâ¢6öç7B—†5öVçB¦RÒ—†5÷vçFVB†GrÂF’’ò—†5öf–æB†GrÂF’’¢åTÄÃ²fö–B¦Ö"Â¦#²V–çCcE÷BÖòÂòÂ–s°¢V–çCcE÷B–æBÒGròGrÓæ–æB¢F’Óæ–æC°¢–b†—†5÷vçFVB†GrÂF’’’²–b†RbbRÓæ&w2’Væ5÷%öG&uö–æFW†VEö–æF—&V7B†VâÂ×ÂÕDÄ–æFW…G—UT–çC3"ÂRÓæ—†"ÂRÓæ—†ÆVâÂRÓæ&w2“²VÇ6R$TEôôä4Rƒƒ‚Â&–æF—&V7BG&r6¶—VC¢F†R&wVÖVçG2f÷"&–Ö—F—fRVBvW&Ræ÷B6öçfW'FVB†&wVÖVçBFG&W72R6ÆÇ‚—2–âæòÖVÖ÷'’ööÂÂ÷"F†R6öçfW'6–öâ—VÆ–æRv2æ÷B7&VFVB’"Â&–ÒÂ‡Vç6–væVBÆöærÆöær––æB“²Ð¢VÇ6R–b‚ÖG‚bb×FÅöFG"†–æBÂGròb¢#Âf"Âfò’’$TEôôä4Rƒƒ‚Â&–æF—&V7BG&r6¶—VC¢&wVÖVçBFG&W72R6ÆÇ‚—2–âæòÖVÖ÷'’ööÂ"Â‡Vç6–væVBÆöærÆöær––æB“°¢VÇ6R–b†Gr’²Væ5÷%öG&uö–æF—&V7B†VâÂ×ÂÖG‚òÖG‚¢‚…õö'&–FvR–CÄÕDÄ'VffW#â–"’æwTFG&W72²ò“²–b†ÖG‚’ÖG…ö6†–â†GrÓæÖF•ö6çBÂGrÓæÖF•ö’Â×ÂÕDÄ–æFW…G—UT–çC3"ÂÂ“²Ð¢VÇ6R–b‚×FÅöwUöFG"†F’Óæ–'VbÂÂfÖ"ÂfÖòÂf–r’’$TEôôä4Rƒ#"Â&–æFW‚'VffW"—2–âæòÖVÖ÷'’ööÂ"“°¢VÇ6R²ÕDÄ–æFW…G—RG’ÒF’Óæ—G—RÓÒòÕDÄ–æFW…G—UT–çCb¢ÕDÄ–æFW…G—UT–çC3#²V–çCcE÷B–ÂÒ‚…õö'&–FvR–CÄÕDÄ'VffW#â–Ö"’æÆVæwF‚ÒÖó°¢Væ5÷%öG&uö–æFW†VEö–æF—&V7B†VâÂ×ÂG’Â–rÂ–ÂÂÖG‚òÖG‚¢‚…õö'&–FvR–CÄÕDÄ'VffW#â–"’æwTFG&W72²ò“²–b†ÖG‚’ÖG…ö6†–â†F’ÓæÖF•ö6çBÂF’ÓæÖF•ö’Â×ÂG’Â–rÂ–Â“²Ð¢ÒVÇ6R–b†Grbb&–ÒÓÒr’²òòVG3¢6†&VB–æF–6W2ÃÃ"ÃÃ"Ã0¢VEö–%öÖ¶R‚“°¢–b†GrÓæ6÷VçBãÒB’Væ5÷%öG&uö–æFW†VB†VâÂ×Â†GrÓæ6÷VçBòB’¢bÂÕDÄ–æFW…G—UT–çC3"Âu÷VEöwRÂu÷VEö–"æÆVæwF‚ÂGrÓæ–ç7Fæ6W2âòGrÓæ–ç7Fæ6W2¢ÂGrÓæf—'7BÂGrÓæ&6Uö–ç7Fæ6R“°¢ÒVÇ6R–b†Grbb&–ÒÓÒb’²òòG&–ævÆRfç2ƒ26VçF–æVÇ2“¢6†&VB–æF–6W2ƒÂ’Â’³’Âf—'7B2F†R&6RfW'FW€¢fåö–%öÖ¶R‚“°¢–b†GrÓæ6÷VçBãÒ2’Væ5÷%öG&uö–æFW†VB†VâÂ×Â†GrÓæ6÷VçBÒ"’¢2ÂÕDÄ–æFW…G—UT–çC3"ÂuöfåöwRÂuöfåö–"æÆVæwF‚ÂGrÓæ–ç7Fæ6W2âòGrÓæ–ç7Fæ6W2¢ÂGrÓæf—'7BÂGrÓæ&6Uö–ç7Fæ6R“°¢ÒVÇ6R–b†Gr’°¢–b†GrÓæ6÷VçBbb†GrÓæ–ç7Fæ6W2âÇÂGrÓæ&6Uö–ç7Fæ6R’’Væ5÷%öG&uö–ç7Fæ6VB†VâÂ×ÂGrÓæf—'7BÂGrÓæ6÷VçBÂGrÓæ–ç7Fæ6W2âòGrÓæ–ç7Fæ6W2¢ÂGrÓæ&6Uö–ç7Fæ6R“°¢VÇ6R–b†GrÓæ6÷VçB’Væ5÷%öG&r†VâÂ×ÂGrÓæf—'7BÂGrÓæ6÷VçB“°¢ÒVÇ6R°¢–b‡&–ÒÓÒbÇÂ&–ÒÓÒr’°¢òò–æFW†VBG&–ævÆRfç2òVG2ƒ26VçF–æVÇ2“¢B&WÆ’F†R–æF–6W2&R&VBg&öÒwVW7BÖVÖ÷'’æBW‡æFVB–çFò3"Ö&—BG&–ævÆRÆ—7B–âF†—2g&ÖRw2&Væ†2f÷"‚Ö&—B–æF–6W2¢–çB—7¢ÒF’Óæ—G—RÓÒò¢F’Óæ—G—RÓÒò"¢C°¢V–çC3%÷BçBÒ&–ÒÓÒrò‡V–çC3%÷B’†F’Óæ6÷VçBòB’¢"¢F’Óæ6÷VçBãÒ2ò‡V–çC3%÷B–F’Óæ6÷VçBÒ"¢°¢–b‚çB’&WGW&ã°¢fö–B¦Ö#²V–çCcE÷BÖó²6öç7BV–çC…÷B§7&3°¢–b‚×FÅöFG"†F’Óæ–'VbÂ‡V–çCcE÷B–F’Óæ6÷VçB¢—7¢ÂfÖ"ÂfÖò’ÇÂ‡7&2Ò²…õö'&–FvR–CÄÕDÄ'VffW#â–Ö"6öçFVçG5Ò’’²$TEôôä4Rƒ#"Â&–æFW‚'VffW"—2–âæòÖVÖ÷'’ööÂ"“²&WGW&ã²Ð¢–b‡VæEö†—G2†F’Óæ–'VbÂ‡V–çCcE÷B–F’Óæ6÷VçB¢—7¢’’$TEôôä4Rƒs2Â'G&–ævÆRfâòVB–æFW‚'VffW"†2uR×6–FRw&—FW2–âF†—27V&Ö—76–öã¢W‡æFVBg&öÒF†R7W'&VçB6öçFVçB"“°¢7&2³ÒÖó°¢–CÄÕDÄ'VffW#â#²6—¦U÷BöfbÒ&VæöÆÆö2‚‚‡6—¦U÷B–çB¢"²R’bâ‡6—¦U÷B“RÂf"“°¢V–çC3%÷B¦BÒ‡V–çC3%÷B¢’‚‡V–çC…÷B¢–"æ6öçFVçG2²öfb“°¢òòv—F‚&–Ö—F—fR&W7F'BöâÂF†R&W7F'B–æFW‚†6ö×&VB&Vf÷&RF†R&6RfW'FW‚—2FFVB’VæG2fâ÷"VBæBF†RæW‡BöæR7F'G2gFW"—C¢F†W’vW&R¦ö–æVB7&÷72—BÀ¢òòæBF†RW‡G&G&–ævÆW26ÆV&VB'G2öbÇV–v’w2Öç6–öâ2w266†VB6†F÷rFWF‚‡7G&–v‡BÆ–v‡BVFvW2öâF†Rv&vRfÆö÷"Â"3’âBÖ÷7BçBG&–ævÆW26öÖR÷W@¢V–çC3%÷B6÷VçBÒçfåöW‡æE÷&–Ö—F—fR†BÂ7&2Â‡V–çC3%÷B–F’Óæ6÷VçBÂ‡Vç6–væVB–—7¢Â‡Vç6–væVB—&–ÒÂ2ç&W7F'BæöâÂ2ç&W7F'Bæ–æFW‚“°¢–b†6÷VçB’Væ5÷%öG&uö–æFW†VB†VâÂ×Â6÷VçBÂÕDÄ–æFW…G—UT–çC3"Âuö7W%ö&VæÓæwR²öfbÂ„å5T–çFVvW"–6÷VçB¢BÂF’Óæ–ç7Fæ6W2òF’Óæ–ç7Fæ6W2¢ÂF’Óæ&6U÷fW'FW‚ÂF’Óæ&6Uö–ç7Fæ6R“°¢–b†u÷'E÷&ævU÷VæF–ær’²B‚&¢'EöÆWfVÅ÷&ævR73ÒVÆÇRf3ÒWR"Âu÷7FE÷72Â‡Vç6–væVB–F’Óæ6÷VçB“²u÷'E÷&ævU÷VæF–ærÒ²Ð¢&WGW&ã°¢Ð¢–b†F’Óæ—G—RÓÒ’°¢òò‚Ö&—B–æF–6W3¢ÖWFÂöæÇ’†2bò3"Ö&—BâB&WÆ’F†W’&R&VBg&öÒwVW7BÖVÖ÷'’æBv–FVæVBFòbÖ&—B–âF†—2g&ÖRw2&Væ…4Ô%rw2gVÆÂ×67&VVâ76W2æBF†Rf–æÂ6÷’–çFòF†Rv–æF÷rÆÂW6R‚Ö&—B–æF–6W3²v—F†÷WBF†—2F†Rv†öÆR67&VVâ—2&Æ6²¢–b‚F’Óæ6÷VçB’&WGW&ã°¢–b‡&–ÒÓÒ2ÇÂ&–ÒÓÒRÇÂ&–ÒÓÒÇÂ&–ÒÓÒ2’²7FF–2–çBöæ6S²–b‚öæ6R²²’ÔÄôr‚#‚Ö&—B–æF–6W2–â7G&—‡&–Ö—F—fRVBÂf—'7B66R72VÆÇR“¢&–Ö—F—fR&W7F'B–æFW‚†fb7F—2ƒfbv†Vâv–FVæVBÂv†–6‚ÖWFÂFöW2æ÷B&W7F'Böâ"Â&–ÒÂu÷7FE÷72“²Ð¢6öç7Bv–FUöVçB§rÒv–FU÷vçFVB†F’’òv–FUöf–æB†F’’¢åTÄÃ°¢–b‡rbbrÓæwR’²òòw&—GFVâV&Æ–W"–âF†—27V&Ö—76–öã¢v–FVæVBöâF†RuR&Vf÷&RF†—272‡v–FU÷'Vâ¢Væ5÷%öG&uö–æFW†VB†VâÂ×ÂF’Óæ6÷VçBÂÕDÄ–æFW…G—UT–çCbÂrÓæwRÂ„å5T–çFVvW"–F’Óæ6÷VçB¢"ÂF’Óæ–ç7Fæ6W2òF’Óæ–ç7Fæ6W2¢ÂF’Óæ&6U÷fW'FW‚ÂF’Óæ&6Uö–ç7Fæ6R“°¢–b†u÷'E÷&ævU÷VæF–ær’²B‚&¢'EöÆWfVÅ÷&ævR73ÒVÆÇRf3ÒWR"Âu÷7FE÷72Â‡Vç6–væVB–F’Óæ6÷VçB“²u÷'E÷&ævU÷VæF–ærÒ²Ð¢&WGW&ã°¢Ð¢fö–B¦Ö#²V–çCcE÷BÖó²6öç7BV–çC…÷B§7&3°¢–b‚×FÅöFG"†F’Óæ–'VbÂF’Óæ6÷VçBÂfÖ"ÂfÖò’ÇÂ‡7&2Ò²…õö'&–FvR–CÄÕDÄ'VffW#â–Ö"6öçFVçG5Ò’’²$TEôôä4Rƒ#"Â&–æFW‚'VffW"—2–âæòÖVÖ÷'’ööÂ"“²&WGW&ã²Ð¢–b‡VæEö†—G2†F’Óæ–'VbÂF’Óæ6÷VçB’’$TEôôä4RƒC‚Â#‚Ö&—B–æFW‚'VffW"†2uR×6–FRw&—FW2–âF†—27V&Ö—76–öâæBF†RuRW‡ç6–öâf–ÆVC¢W‡æFVBg&öÒF†R7W'&VçB6öçFVçB"“°¢–CÄÕDÄ'VffW#â#²6—¦U÷BöfbÒ&VæöÆÆö2‚‚‡6—¦U÷B–F’Óæ6÷VçB¢"²R’bâ‡6—¦U÷B“RÂf"“°¢V–çCe÷B¦BÒ‡V–çCe÷B¢’‚‡V–çC…÷B¢–"æ6öçFVçG2²öfb“²7&2³ÒÖó°¢f÷"†–çC3%÷B’Ò²’ÂF’Óæ6÷VçC²’²²’E¶•ÒÒ7&5¶•Ó°¢Væ5÷%öG&uö–æFW†VB†VâÂ×ÂF’Óæ6÷VçBÂÕDÄ–æFW…G—UT–çCbÂuö7W%ö&VæÓæwR²öfbÂ„å5T–çFVvW"–F’Óæ6÷VçB¢"ÂF’Óæ–ç7Fæ6W2òF’Óæ–ç7Fæ6W2¢ÂF’Óæ&6U÷fW'FW‚ÂF’Óæ&6Uö–ç7Fæ6R“°¢–b†u÷'E÷&ævU÷VæF–ær’²B‚&¢'EöÆWfVÅ÷&ævR73ÒVÆÇRf3ÒWR"Âu÷7FE÷72Â‡Vç6–væVB–F’Óæ6÷VçB“²u÷'E÷&ævU÷VæF–ærÒ²Ð¢&WGW&ã°¢Ð¢fö–B¦Ö#²V–çCcE÷BÖó°¢V–çCcE÷BwS°¢–b‚×FÅöwUöFG"†F’Óæ–'VbÂÂfÖ"ÂfÖòÂfwR’’²$TEôôä4Rƒ#"Â&–æFW‚'VffW"—2–âæòÖVÖ÷'’ööÂ"“²&WGW&ã²Ð¢–b†F’Óæ6÷VçB’Væ5÷%öG&uö–æFW†VB†VâÂ×ÂF’Óæ6÷VçBÂF’Óæ—G—RÓÒòÕDÄ–æFW…G—UT–çCb¢ÕDÄ–æFW…G—UT–çC3"ÂwRÂ‚…õö'&–FvR–CÄÕDÄ'VffW#â–Ö"’æÆVæwF‚ÒÖòÀ¢F’Óæ–ç7Fæ6W2òF’Óæ–ç7Fæ6W2¢ÂF’Óæ&6U÷fW'FW‚ÂF’Óæ&6Uö–ç7Fæ6R“°¢Ð¢–b†u÷'E÷&ævU÷VæF–ær’²B‚&¢'EöÆWfVÅ÷&ævR73ÒVÆÇRf3ÒWR"Âu÷7FE÷72Â‡Vç6–væVB’†GròGrÓæ6÷VçB¢F’Óæ6÷VçB’“²u÷'E÷&ævU÷VæF–æsÓ²Ð§Ð§7FF–2çfåö†Öuö72Ò²æ¶ÆVâÒ6—¦Vöb‡fö–B¢—Ó°§7FF–2fö–Bv&Õö6Æ–Ò‡fö–B’°¢–b‚FöÖ–5öW†6†ævUöW‡Æ–6—B‚fu÷v&Õ÷&VG’ÂÂÖVÖ÷'•ö÷&FW%ö7÷&VÂ’’&WGW&ã°¢÷5÷Væf—%öÆö6µöÆö6²‚fu÷v&ÕöÆö6²“°¢å4'&“Åv&Ô—FVÒ£â¦—FV×2Ò¶u÷v&Õö—FV×26÷•Ó°¢¶u÷v&Õö—FV×2&VÖ÷fTÆÄö&¦V7G5Ó°¢÷5÷Væf—%öÆö6µ÷VæÆö6²‚fu÷v&ÕöÆö6²“°¢f÷"…v&Ô—FVÒ¦—FVÒ–â—FV×2’°¢–b†—FVÒÓæ¶–æBÓÒ’°¢–b‚u÷6†66†R’u÷6†66†RÒ´å4×WF&ÆTF–7F–öæ'’F–7F–öæ'•Ó°¢–b‚u÷6†66†U¶—FVÒÓæ¶W•Ò’²u÷6†66†U¶—FVÒÓæ¶W•ÒÒ—FVÒÓç6†FW#²u÷v&Õö6Æ–Õ÷6†FW'2²³°¢–b†—FVÒÓç&V6÷&B’²6õ÷&V6÷&B‚%5ÇBW5ÇBVEÆâ"Â—FVÒÓæ¶W’åUDc…7G&–ærÂ—FVÒÓç6†FW"Óç7FvR“²6õ÷W6W%÷&V6÷&B‚%5ÇBW5ÇBVEÆâ"Â—FVÒÓæ¶W’åUDc…7G&–ærÂ—FVÒÓç6†FW"Óç7FvR“²ÒÐ¢ÒVÇ6R–b†—FVÒÓæ¶–æBÓÒ’°¢–b‚u÷6†66†R’u÷6†66†RÒ´å4×WF&ÆTF–7F–öæ'’F–7F–öæ'•Ó°¢çe6†FW"§g2Òu÷6†66†U¶—FVÒÓçg5ö¶W•ÒÂ¦g2Ò—FVÒÓæg5ö¶W’æÆVæwF‚òu÷6†66†U¶—FVÒÓæg5ö¶W•Ò¢æ–Ã°¢–b‚g2’²g2Ò—FVÒÓçg3²u÷6†66†U¶—FVÒÓçg5ö¶W•ÒÒg3²Ð¢–b†—FVÒÓæg5ö¶W’æÆVæwF‚bbg2’²g2Ò—FVÒÓæg3²u÷6†66†U¶—FVÒÓæg5ö¶W•ÒÒg3²Ð¢—FVÒÓç6òçg2Ò…õö'&–FvRfö–B¢—g3²—FVÒÓç6òæg2Ò…õö'&–FvRfö–B¢–g3°¢–b‚†ÕövWEö–B‚fu÷6òÂf—FVÒÓç6ò’’²†Õ÷WEö–B‚fu÷6òÂf—FVÒÓç6òÂ—FVÒÓæö&¦V7B“²u÷v&Õö6Æ–Õ÷'2²³²Ð¢ÒVÇ6R°¢çe6†FW"¦72Òu÷6†66†U¶—FVÒÓæ¶W•Ó°¢–b‚72’²72Ò—FVÒÓç6†FW#²–b‚u÷6†66†R’u÷6†66†RÒ´å4×WF&ÆTF–7F–öæ'’F–7F–öæ'•Ó²u÷6†66†U¶—FVÒÓæ¶W•ÒÒ73²Ð¢6öç7Bfö–B¦¶W’Ò…õö'&–FvR6öç7Bfö–B¢–73°¢–b‚†ÕövWEö–B‚fuö72Âf¶W’’’²†Õ÷WEö–B‚fuö72Âf¶W’Â—FVÒÓæö&¦V7B“²u÷v&Õö6Æ–Õö72²³²Ð¢Ð¢Ð§Ð¢òò&öw&Òw2Æ–'&&–W2&RÆöFVB–âF†R&6¶w&÷VæBv†VâF†RvÖRÖ¶W2F†R&öw&Ò†çfå&öw&Õ6WE6†FW'2ÂÖ÷7BöbF†VÒv†–ÆR—BÆöG266VæR’ÂæBF†RæW‡B7V&Ö—76–öà¢òòF¶W2F†VÒ‡v&Õö6Æ–Ò’–ç7FVBöb—G2f—'7BG&rÆöF–ærV6‚öâF†R7V&Ö—BF‡&VC¢–âFöæ¶W’¶öær&æç¦ÆWfVÂw2f—'7Bg&ÖW2ÆöFVB2FòrÆ–'&&–W2F†W&RÂ#Ó“×0¢òòv—F‚F†RG&—fR–FÆRæBWFòB2v—F‚—B'W7’âG&rF†B6öÖW2&Vf÷&R—G2Æ–'&'’†2ÆöFVBÆöG2—B—G6VÆbÂ2&Vf÷&RâDÕôådåõ$TdUD4ƒÓGW&ç2—Böf`§7FF–2ôFöÖ–2V–çCcE÷Bu÷&öweöÆöFVC°§7FF–2–çB&VfWF6…ööâ‡fö–B’²7FF–2–çBbÒÓ²–b‡bÂ’²6öç7B6†"¦RÒvWFVçb‚$DÕôådåõ$TdUD4‚"“²bÒ†Rbb7G&6×†RÂ#"’“²Ò&WGW&âc²Ð§fö–B×FÅ÷&öu÷&VfWF6‚†6öç7B&öu÷B¦7’°¢&öu÷B§Ò‡&öu÷B¢–7°¢–b‚uöFWbÇÂ&VfWF6…ööâ‚’’&WGW&ã°¢6†Ö÷&VEööæ6R‚“²òòF†RÖöæÇ“¢F†R&Wv&Ò7F'G2öâF†R7V&Ö—BF‡&VB‡6†ÖöÆöB¢f÷"‡Vç6–væVB’Ò²’ÂÓæç7FvS²’²²’°¢V–çC…÷BÖE³…Ó²&öuö¶W’‡Â’ÂÖB“°¢å57G&–ær¦²Ò´å57G&–ær7G&–æuv—F„f÷&ÖC¤"S'‚S'‚S'‚S'‚S'‚S'‚S'‚S'‚"ÂÖE³ÒÂÖE³ÒÂÖE³%ÒÂÖE³5ÒÂÖE³EÒÂÖE³UÒÂÖE³eÒÂÖE³uÕÓ°¢å57G&–ær§7FvRÒu÷6†Ö¶µÓ²–çB7BÒ7FvRò7FvUööb‡7FvR’¢Ó°¢–b‡7BÂ’6öçF–çVS²òòæ÷B–âF†R6¶vS¢'V–ÇBÖ–â¶W&æVÂ‡&öuövWB¢å57G&–ær§2Ò¶²7G&–æt'”VæF–æu7G&–æs¤%ö"%Ó°¢–b‚&öweöf—'7B‡2’’6öçF–çVS°¢F—7F6…ö7–æ2‡ö&u÷VWVR‚’Âç²WF÷&VÆV6WööÂ°¢çe6†FW"§6‚Ò6†FW%öÆöB‡2Â7BÂ“°¢–b‡6‚bb7BÓÒbb6‚Óæ†5÷VÆÂbb6‚ÓæÆ–W%ö÷WBbb‡–'&–Eööâ‚’’VÆÅöÆöB‡6‚“²òò—G2VÆÆ–ærVçG'’FöòÂ2F†R&Wv&ÒFöW2‡v&Õ÷6†FW"¢–b‚6‚’&WGW&ã°¢v&Ô—FVÒ¦—FVÒÒµv&Ô—FVÒæWuÓ²—FVÒÓæ¶–æBÒ²—FVÒÓç&V6÷&BÒ²—FVÒÓæ¶W’Ò3²—FVÒÓç6†FW"Ò6ƒ²v&ÕöVçVWVR†—FVÒ“°¢–b‚u÷&öweöÆöFVB²²’ÔÄôr‚'6†FW"&VfWF6‚B&öw&Ò7&VF–öã¢f—'7BÆ–'&'’W2ÆöFVB–âF†R&6¶w&÷VæB„DÕôådåõ$TdUD4ƒÓGW&ç2—Böfb’"Â2åUDc…7G&–ær“°¢ÒÒ“°¢Ð§Ð¢òòF—7F6†W26†&RF†R÷Vâ6÷’Væ6öFW"†ÖWFÂB6ö×WFRVæ6öFW"’v—F‚F†R6÷–W2æBF—7F6†W2&÷VæBF†VÒÂV6‚÷&FW&VBgFW"WfW'—F†–ærF†RVæ6öFW"Ç&VG’†öÆG2'’&'&–W"–ç6–FR—BÀ¢òò2FW‡GW&R6÷’—2†&Æ—B’ââVæ6öFW"öbF†V—"÷vâÂv—F‚gVÆÂVWVR&'&–W'2öâ&÷F‚6–FW2Â6÷7BF†RuRG&–âW"F—7F6ƒ¢‡—'VÆRv'&–÷'3¢vRöb6ÆÖ—G’w2&GFÆW2gFW"F†Rf—'7B7WG66VæP¢òò'Vâ&÷WBÃc6ÖÆÂF—7F6†W2g&ÖRÂæBF†RÖ2uRvVçBg&öÒbãrFòb×2g&ÖRBF†R6ÖRG&w2ÂF†R•†öæRw2g&öÒRFò&÷WBSƒrg2’à¢òòDÕôådåôD•5D4…ôõtãÓv—fW2V6‚F—7F6‚—G2÷vâVæ6öFW"v–â‡6òFöW2DÕôådåõ54uRÂv†–6‚F–ÖW2V6‚öæR§7FF–2–çBF—7F6…ö÷vâ‡fö–B’°¢7FF–2–çBbÒÓ°¢–b‡bÂ’²6öç7B6†"¦RÒvWFVçb‚$DÕôådåôD•5D4…ôõtâ"“²bÒ†Rbb7G&6×†RÂ#"’’ÇÂuööâ‚“²–b‡b’ÔÄôr‚&6ö×WFRF—7F6†W3¢V6‚–ââVæ6öFW"öb—G2÷vâ„DÕôådåôD•5D4…ôõtãÓ÷"DÕôådåõ54uR’"“²Ð¢&WGW&âc°§Ð§7FF–2–CÄÕDÃD6ö×WFT6öÖÖæDVæ6öFW#âF—7F6…öVæ2‡fö–B’°¢–CÄÕDÃD6ö×WFT6öÖÖæDVæ6öFW#âRÒ&Æ—Eö÷Vâ‚“²uöwöF—7²³°¢–b†uö&Æ—Eö÷2’&Æ—Eö&'&–W"‚“°¢uö'uöÆõ³ÒÒ²uö'uö†•³ÒÒãVÆÃ²uö'uöâÒ²uö'%öâÒ²uö&Æ—Eö÷2Ò²òò&V6÷&FVB2&WfW'—F†–ærw&—GFVâ#¢F†RæW‡B6÷’÷"F—7F6‚vWG2&'&–W ¢–b‚uö&Æ—Eö7B’²¶R6WD&wVÖVçEF&ÆS¦uö7EÓ²uö&Æ—Eö7BÒ²Ð¢&w5öF—'G•öÆÂ‚“°¢&WGW&âS°§Ð§7FF–2fö–BF—7F6‚†6öç7BöF—7F6‚¦G’°¢§6¶—ö6ÆV"‚“°¢u÷w%öWö6‚²³²uöÖG…öWö6‚²³°¢–b‚2æ7&ör’²$TEôôä4Rƒ#rÂ&æò&öw&Ò&÷VæB&Vf÷&RF†RF—7F6‚"“²&WGW&ã²Ð¢òòF—7F6„6ö×WFT–æF—&V7C¢F†Rw&÷W6÷VçG2·‚Â’Â§ÒB–æB„ÕDÄF—7F6…F‡&VFw&÷W4–æF—&V7D&wVÖVçG2’&RW7VÆÇ’w&—GFVâ'’F†RuRV&Æ–W"–âF†Rg&ÖS²ÖWFÂ&VG2F†VÒv†VâF†RF—7F6‚'Vç0¢fö–B¦–"ÒåTÄÃ²V–çCcE÷B–òÒÂ–rÒ°¢–b†GÓæ–æBbb×FÅöwUöFG"†GÓæ–æBÂ"Âf–"Âf–òÂf–r’’²$TEôôä4Rƒ“BÂ&–æF—&V7BF—7F6‚6¶—VC¢&wVÖVçBFG&W72R6ÆÇ‚—2–âæòÖVÖ÷'’ööÂ"Â‡Vç6–væVBÆöærÆöær–GÓæ–æB“²&WGW&ã²Ð¢–b†GÓæ–æB’²7FF–2–çBöæ6S²–b‚öæ6R²²’ÔÄôr‚&–æF—&V7BF—7F6ƒ¢f—'7BF–ÖR‡F†RuR&VG2F†RF‡&VFw&÷W6÷VçG2g&öÒR6ÆÇ‚’"Â‡Vç6–væVBÆöærÆöær–GÓæ–æB“²Ð¢çe6†FW"¦72Ò&öuövWB…2æ7&ör’Óç6…³UÓ°¢–b‚72’²$TEôôä4Rƒ#‚Â'F†RF—7F6†VB&öw&Ò†2æò6ö×WFR6†FW"†÷"—B—2æ÷BÖVB’"“²&WGW&ã²Ð¢6öç7Bfö–B¦²Ò…õö'&–FvR6öç7Bfö–B¢–73°¢–CÄÕDÄ6ö×WFU—VÆ–æU7FFSâ2Ò†ÕövWEö–B‚fuö72Âf²“°¢–b‚†–B—2ÓÒ´å4çVÆÂçVÆÅÒ’&WGW&ã²òò—G2—VÆ–æR6÷VÆBæ÷B&RÖFR†&VÆ÷r“¢F†R6ÖRgVæ7F–öâf–Ç2F†R6ÖRv’Â6ò—B—2æ÷B6ö×–ÆVBv–âBWfW'’F—7F6€¢–b‚2’°¢å4W'&÷"¦RÒæ–Ã²V–çCcE÷B'BÒ'öæ÷r‚“²2Ò¶uöFWbæWt6ö×WFU—VÆ–æU7FFUv—F„gVæ7F–öã¦72ÓæfâW'&÷#¢fUÓ°¢uö'÷6õöâ²³²uö'÷6õöç2³Ò'öæ÷r‚’Ò'C°¢òò&VÖVÖ&W&VB2f–ÆVC¢Ö&–ò¶'Bv÷&ÆBw26ö×WFR6†FW'2F†BæVVBC‚´"öbF‡&VFw&÷WÖVÖ÷'’„ÖWFÂÆÆ÷w23"´"’vW&R6ö×–ÆVBv–âBV6‚öbF†V—"F—7F6†W2Â2Ãc‚f–ÆV@¢òò6ö×–ÆW2–â2Ö–çWFW2ÂæBF†RvÖR&âBfWrg&ÖW26V6öæBâ—G2F—7F6†W2&R6¶—VBÂ2&Vf÷&P¢–b‚2’²ÔÄôr‚%µ$TEÒf–ÆVBFò7&VFR6ö×WFR—VÆ–æRf÷"6†FW"W3¢W2†—G2F—7F6†W2&R6¶—VB’"Â72Óæ¶W’ÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“²†Õ÷WEö–B‚fuö72Âf²Â†–B•´å4çVÆÂçVÆÅÒ“²&WGW&ã²Ð¢6õ÷&V6÷&B‚$5ÇBW5Æâ"Â72Óæ¶W’“°¢–b†72Óæ¶W•³Ò’6õ÷W6W%÷&V6÷&B‚$5ÇBW5Æâ"Â72Óæ¶W’“°¢†Õ÷WEö–B‚fuö72Âf²Â2“°¢Ð¢–b†72Óæ'V–ÇF–â’²òò'V–ÇBÖ–â6ö×WFR¶W&æVÃ¢âÒVæ–f÷&Ò'VffW"Â7&2Ò7F÷&vR'VffW"ÂG7BÒ7F÷&vR'VffW"¢6öç7Bö&–æEö'Vb§RÒe2çV&õ³UÕ³ÒÂ¦Òe2ç76&õ³UÕ³ÒÂ¦Òe2ç76&õ³UÕ³Ó°¢fö–B§V"Â§6"Â¦F#²V–çCcE÷BVòÂ6òÂFöc°¢–b‚×FÅöFG"‡RÓæFG"ÂBÂgV"ÂgVò’ÇÂ×FÅöFG"†ÓæFG"Â‚Âg6"Âg6ò’ÇÂ×FÅöFG"†ÓæFG"Â‚ÂfF"ÂfFöb’’²$TEôôä4Rƒ3Â&VÖ&VFFVB6ö×WFR¶W&æVÂW2FöW2æ÷B†fRÆÂ—G2'VffW'2&÷VæB"Â72Óæ'V–ÇF–â“²&WGW&ã²Ð¢–b‚72ÓæÆövvVB²²’°¢ÖöGVÆU÷B¦ÒÒÖöGVÆUööb‚‡fö–B¢–GÓæ6ÆÆW"“°¢ÔÄôr‚&VÖ&VFFVB6ö×WFR¶W&æVÂW2f—'7BF—7F6ƒ¢6ÆÂ6—FRW2²R6ÆÇ‚ÂF‡&VFw&÷W2VG‚VG‚VBÂVÆVÖVçB6÷VçB†7W'&VçB5R×6–FRfÇVR’WR"Â72Óæ'V–ÇF–âÂÒòÒÓææÖR¢#ò"À¢Òò‡Vç6–væVBÆöærÆöær’†GÓæ6ÆÆW"Ò‡V–çCcE÷B–ÒÓæ&6R’¢‡Vç6–væVBÆöærÆöær–GÓæ6ÆÆW"ÂGÓç‚ÂGÓç’ÂGÓç¢Â¢‡V–çC3%÷B¢—RÓæFG"“°¢Ð¢'öVæB‚“²–b†uö&Æ—B’²Væ5ö%öVæB†uö&Æ—B“²uö&Æ—BÒæ–Ã²Ð¢–CÄÕDÄ6ö×WFT6öÖÖæDVæ6öFW#â6RÒuö6ö×WFR‚“°¢Væ5ö5÷6ò†6RÂ2“°¢Væ5ö5ö&w2†6RÂ‚…õö'&–FvR–CÄÕDÄ'VffW#â—V"’æwTFG&W72²VòÂ“²Væ5ö5ö&w2†6RÂ‚…õö'&–FvR–CÄÕDÄ'VffW#â—6"’æwTFG&W72²6òÂ“²Væ5ö5ö&w2†6RÂ‚…õö'&–FvR–CÄÕDÄ'VffW#â–F"’æwTFG&W72²FöbÂ"“²òò'V–ÇBÖ–â6ö×WFR¶W&æVÃ¢6öÆBF‚ÂF¶RF†RFG&W76W2F—&V7FÇ¢–b†GÓæ–æB’Væ5ö5öF—7F6…ö–æF—&V7B†6RÂ–rÂÕDÅ6—¦TÖ¶Rƒ3"ÂÂ’“²VÇ6RVæ5ö5öF—7F6‚†6RÂÕDÅ6—¦TÖ¶R†GÓç‚ÂGÓç’ÂGÓç¢’ÂÕDÅ6—¦TÖ¶Rƒ3"ÂÂ’“°¢Væ5ö5öVæB†6R“°¢&WGW&ã°¢Ð¢6öç7BV–çCe÷B§FrÒ72Óç&bçFs°¢–b‚Fu³Ò’²$TEôôä4Rƒ#’Â&6ö×WFR6†FW"Æ6·2F‡&VFw&÷W6—¦R&VfÆV7F–öâ"“²&WGW&ã²Ð¢òò7F÷&RF†R6†FW"Ö¶W2öâ—G2fW'’f—'7B–çfö6F–öâöæÇ’Â&Vf÷&RF†R÷F†W'2'VâöâF†R7v—F6ƒ¢F†R'V–ÆBF¶W2—B÷WBöbF†R6†FW"†F×6…ö'V–ÆBç’f—'7E÷7F÷&R’Â2ÆRw2uW27F'Bv÷&¶w&÷W2–âæò÷&FW ¢–b†GÓæ–æBÇÂ†GÓç‚bbGÓç’bbGÓç¢’’f÷"†–çB’Ò²’Â72Óç&bææf—'7C²’²²’²6öç7Bö&–æEö'Vb§RÒe2ç76&õ³UÕ¶72Óç&bæf—'7E¶•Òç&VuÓ²–b‡RÓæFG"’×FÅ÷7V%öf–ÆÂ‡RÓæFG"²GVÆÂ¢72Óç&bæf—'7E¶•Òçv÷&BÂBÂ72Óç&bæf—'7E¶•ÒçfÇVR“²Ð¢–b†GÓæ–æB’f÷"†–çB’Ò²’Â72Óç&bææ#²’²²’–b†72Óç&bæ%¶•Òæ¶–æBÓÒådåõ$%õ5”ÄÂ’²òò—G267&F6‚—26—¦VB'’F†RF‡&VFw&÷W6÷VçBÂv†–6‚öæÇ’F†RuR¶æ÷w2†W&P¢$TEôôä4RƒC2Â&–æF—&V7BF—7F6‚öb6†FW"W2Âv†÷6R6†&VBÖVÖ÷'’—27BÖWFÂw2Æ–Ö—B‡7–ÆÆVBFòFWf–6RÖVÖ÷'’“¢6¶—VB"Â72Óæ¶W’“²&WGW&ã²Ð¢–b†uöçV&BbbV&Eö6öæfÆ–7Eö72†72ÂGÓæ–æB’’²u÷V&Eö6öæfÆ–7G2²³²V&EöÆæB‚“²Òòò6VRV&Eö6öæfÆ–7Eö70¢'öfÇW6‚‚“²òògVÆÂ6ÆV"v—F‚æòG&rgFW"—Bv—G2f÷"F†RæW‡B72öâ—G2F&vWG2Â'WB—B6öÖW2&Vf÷&RF†—2F—7F6‚Âv†–6‚Ö’&VB÷"w&—FRF†VÒ2–ÖvW3¢ÆæB—Bf—'7BÂ26÷•÷C'BFöW0¢–ÖvU÷&W&R‡&öuövWB…2æ7&ör’Âƒ#“°¢–çB÷vâÒF—7F6…ö÷vâ‚“°¢–b†÷vâbbuö&Æ—B’²Væ5ö%öVæB†uö&Æ—B“²uö&Æ—BÒæ–Ã²Ð¢–CÄÕDÄ6ö×WFT6öÖÖæDVæ6öFW#â6RÒ÷vâòuö6ö×WFR‚’¢†–CÄÕDÄ6ö×WFT6öÖÖæDVæ6öFW#â–F—7F6…öVæ2‚“°¢Væ5ö5÷6ò†6RÂ2“°¢uöF—7F6…öF–Õ³ÒÒ‡V–çC3%÷B–GÓçƒ²uöF—7F6…öF–Õ³ÒÒ‡V–çC3%÷B–GÓç“²uöF—7F6…öF–Õ³%ÒÒ‡V–çC3%÷B–GÓç£²òò&VB&6²'’F†RG&—fW"6VvÖVçB3†Væ6öFUö&w5ò¢uö3öF–Õö'VbÒæ–Ã°¢Væ6öFUö&w2†6RÂ72ÂRÂ“°¢–b†GÓæ–æB’²òòF†R6†FW"&VG2F†Rw&÷W6÷VçG2g&öÒ3²ƒ#¢6÷’F†RuR×w&—GFVâöæW2F†W&Rf—'7B„ÖWFÂB÷&FW'2æ÷F†–ær–ç6–FRâVæ6öFW"v—F†÷WB&'&–W"¢–b†uö3öF–Õö'Vb’²²†–CÄÕDÃD6ö×WFT6öÖÖæDVæ6öFW#â–6R6÷”g&öÔ'VffW#¢…õö'&–FvR–CÄÕDÄ'VffW#â––"6÷W&6Töfg6WC¦–òFô'VffW#¦uö3öF–Õö'VbFW7F–æF–öäöfg6WC¦uö3öF–Õööfb6—¦S£%Ó°¢–b†FÕ÷6Fµö÷Væ6R‚’’²†–CÄÕDÃD6ö×WFT6öÖÖæDVæ6öFW#â–6R6÷”g&öÔ'VffW#¢…õö'&–FvR–CÄÕDÄ'VffW#â––"6÷W&6Töfg6WC¦–òFô'VffW#¦uö3öF–Õö'VbFW7F–æF–öäöfg6WC¦uö3öF–ÕööfbÒDÕ4…ô3ôD•5D4‚²DÕ4…ô3ôD•5D4…ôådã"6—¦S£%Ó°¢²†–CÄÕDÃD6ö×WFT6öÖÖæDVæ6öFW#â–6R&'&–W$gFW$Væ6öFW%7FvW3¤ÕDÅ7FvT&Æ—B&Vf÷&TVæ6öFW%7FvW3¤ÕDÅ7FvTF—7F6‚f—6–&–Æ—G”÷F–öç3§&W6÷W&6U÷f—2‚•Ó²Ð¢Væ5ö5öF—7F6…ö–æF—&V7B†6RÂ–rÂÕDÅ6—¦TÖ¶R‡Fu³ÒÂFu³ÒÂFu³%Ò’“°¢ÒVÇ6RVæ5ö5öF—7F6‚†6RÂÕDÅ6—¦TÖ¶R†GÓç‚ÂGÓç’ÂGÓç¢’ÂÕDÅ6—¦TÖ¶R‡Fu³ÒÂFu³ÒÂFu³%Ò’“°¢–b†÷vâ’Væ5ö5öVæB†6R“°§Ð¢òòuS¢FW‡GW&RÓâ'VffW"‡&÷r7G&–FRF¶Vâg&öÒ6WD6÷•&÷u7G&–FRÂÆ–W"7G&–FRg&öÒ6WD6÷”–ÖvU7G&–FRÂÒF–v‡FÇ’6¶VB¢òò6÷’BÆWfVÂF†R†÷7BFW‡GW&RÆ6·2†Ö÷&RÆWfVÇ2FV6Æ&VBF†â—G26—¦R†öÆG2ÂFW…öFW62’—26¶—VC¢ÖWFÂFöW2æ÷B6†V6²F†RÆWfVÂöb&Æ—@§7FF–2–çBFW…öÆWfVÅöö²†–CÄÕDÅFW‡GW&SâBÂå5T–çFVvW"ÆWfVÂÂ6öç7B6†"§v†B’°¢–b†ÆWfVÂÂBæÖ—ÖÆWfVÄ6÷VçB’&WGW&â°¢$TEôôä4RƒbÂ"W2BÆWfVÂVÇRöbFW‡GW&Rv—F‚VÇRÆWfVÇ3¢6¶—VB†f—'7B66R’"Âv†BÂ‡Vç6–væVBÆöær–ÆWfVÂÂ‡Vç6–væVBÆöær—BæÖ—ÖÆWfVÄ6÷VçB“°¢&WGW&â°§Ð¢òòF†R$t#ScRfÖ–Ç’w2&VBò&ÇVRW†6†ævR‡&%÷6¶VE÷6†–gB’öâF†RuRÂf÷"6÷–W2&WGvVVâwVW7BÖVÖ÷'’æB†÷7BFW‡GW&S¢F†R6÷’'Vç2öâF†RuRF–ÖVÆ–æRÂgFW"uRw&—FW2V&Æ–W"–âF†P¢òò7V&Ö—76–öâÂ6ò—G2'—FW26âöæÇ’&R6öçfW'FVBF†W&Râv"'—FW2öbV6‚öb&÷w2&÷w2–âç¢–ÖvW2Âg&öÒ7&2‡&÷rò–Ör'B’FòG7B†G&÷ròF–Ör'B“²7&2ÓÒG7B6öçfW'G2–âÆ6Rà¢òòF—7F6†VB–âF†R6÷’Væ6öFW"Â÷&FW&VB'’—G2&'&–W'2†F—7F6…öVæ2§7FF–2–çB&%÷7vöwR‡V–çCcE÷B7&2Â6—¦U÷B&÷rÂ6—¦U÷B–ÖrÂV–çCcE÷BG7BÂ6—¦U÷BG&÷rÂ6—¦U÷BF–ÖrÂ6—¦U÷Bv"Â6—¦U÷B&÷w2Â6—¦U÷Bç¢Â–çB6†–gB’°¢7FF–2–CÄÕDÄ6ö×WFU—VÆ–æU7FFSâ3²7FF–2–çBf–ÆVC°¢–b†f–ÆVB’&WGW&â°¢–b‚2’°¢å4W'&÷"¦RÒæ–Ã²–CÄÕDÄÆ–'&'“âÆ–"Ò†÷7EöÆ–"‚'&%÷7v"ÂfR“°¢–CÄÕDÄgVæ7F–öãâfâÒ¶Æ–"æWtgVæ7F–öåv—F„æÖS¤'&%÷7v%Ó°¢2Òfâò¶uöFWbæWt6ö×WFU—VÆ–æU7FFUv—F„gVæ7F–öã¦fâW'&÷#¢fUÒ¢æ–Ã°¢–b‚2’²f–ÆVBÒ²ÔÄôr‚%µ$TEÒ6ææ÷B7&VFRF†R&VBò&ÇVRW†6†ævRf÷"$t#ScRÖfÖ–Ç’6÷–W3¢W2"ÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“²&WGW&â²Ð¢Ð¢–CÄÕDÄ'VffW#â#²6—¦U÷BòÒ&VæöÆÆö2ƒC‚Âg"“°¢7G'V7B²V–çCcE÷B2ÂC²V–çC3%÷BrÂ‚Â&÷rÂ–ÖrÂG&÷rÂF–ÖrÂ6†–gC²Ò§Ò‡fö–B¢’‚‡V–çC…÷B¢—"æ6öçFVçG2²ò“°¢§Ò‡G—Vöb‚§’—·7&2ÂG7BÂ‡V–çC3%÷B’‡v"ò"’Â‡V–çC3%÷B—&÷w2Â‡V–çC3%÷B—&÷rÂ‡V–çC3%÷B––ÖrÂ‡V–çC3%÷B–G&÷rÂ‡V–çC3%÷B–F–ÖrÂ‡V–çC3%÷B—6†–gGÓ°¢–CÄÕDÃD6ö×WFT6öÖÖæDVæ6öFW#âRÒF—7F6…öVæ2‚“°¢¶uö7B6WDFG&W73§"æwTFG&W72²òD–æFWƒ£Ó²Væ5ö5÷6ò†RÂ2“°¢Væ5ö5öF—7F6‚†RÂÕDÅ6—¦TÖ¶R‚‡v"ò"²c2’òcBÂ&÷w2Âç¢’ÂÕDÅ6—¦TÖ¶RƒcBÂÂ’“°¢7FF–2–çBöæ6S²–b‚öæ6R²²’ÔÄôr‚%$t#ScRÖfÖ–Ç’6÷’&WGvVVâwVW7BÖVÖ÷'’æB†÷7BFW‡GW&S¢&VBæB&ÇVRW†6†ævVBöâF†RuR†f—'7B66RW§W‚W§W‚W§RFW†VÇ2’"Âv"ò"Â&÷w2Âç¢“°¢&WGW&â°§Ð¢òòv†WF†W"6÷’–çFò÷"÷WBöbF†—2†÷7BFW‡GW&R7&÷76W2&WGvVVâF†RwVW7Bw2÷&FW"æBÖWFÂw2f÷"F†R$t#ScRfÖ–Ç“¢F†R&VB&—B÷6—F–öâ‡&%÷6¶VE÷6†–gB’Âf÷"÷F†W"f÷&ÖG2æBf÷"¢òòFW‡GW&RöâööÂw2'VffW"Âv†–6‚†öÆG2F†RwVW7Bw2&—G2†Æ–æV"æB'VffW"FW‡GW&W2§7FF–2–çB&%ö6÷•÷6†–gB†6öç7BFW…÷B§BÂ–CÄÕDÅFW‡GW&Sâ×B’²&WGW&â×Bæ'VffW"ò¢&%÷6¶VE÷6†–gB‡BÓæBæf÷&ÖB“²Ð§7FF–2fö–B6÷•÷C&"†6öç7Bö6÷•ö#'B¦2’°¢'öfÇW6‚‚“²òòÖFW&–Æ—¦RVæF–ær6ÆV'2&Vf÷&R&VF–ærÂÖF6†–ær6÷•÷C'Bà¢u÷w%öWö6‚²³²uöÖG…öWö6‚²³°¢–b†2ÓçFW‚’‡eö6†V6²†2ÓçFW‚Óç6W&–ÂÂ2Óæ†5÷f–Wrbb2Óçf–Wrç6WEöÆWfVÇ2ò2Óçf–Wræ&6UöÆWfVÂ¢Â2Óæ†5÷f–Wrbb2Óçf–Wrç6WEöÆ–W'2ò2Óçf–WræÖ–åöÆ–W"¢Â'FW‡GW&R×FòÖ'VffW"6÷’"“°¢6ögEö†&FVâ†2ÓçFW‚“°¢–CÄÕDÅFW‡GW&SâBÒFW…ö×FÂ†2ÓçFW‚“°¢–b‚B’&WGW&ã°¢–çB'rÂ&‚Â'"Òçfåöf×Eö'"†2ÓçFW‚ÓæBæf÷&ÖBÂf'rÂf&‚“°¢Vç6–væVBÆWfVÂÒ2Óæ†5÷f–Wrbb2Óçf–Wrç6WEöÆWfVÇ2ò2Óçf–Wræ&6UöÆWfVÂ¢ÂÆ–W#Ò2Óæ†5÷f–Wrbb2Óçf–Wrç6WEöÆ–W'2ò2Óçf–WræÖ–åöÆ–W"¢°¢–b‚FW…÷6ögEöÆWfVÂ†2ÓçFW‚ÂfÆWfVÂ’’²$TEôôä4RƒCrÂ&6÷’÷WBöbÆWfVÂöb6ögBFW‡GW&S¢6¶—VB†f—'7B66R6W#ÒVÆÇR’"Â‡Vç6–væVBÆöærÆöær–2ÓçFW‚Óç6W&–Â“²&WGW&ã²Ð¢–b‚FW…öÆWfVÅöö²‡BÂÆWfVÂÂ'FW‡GW&R×FòÖ'VffW"6÷’"’’&WGW&ã°¢6—¦U÷B&÷rÒ2æ6÷•÷7G&–FRò‡6—¦U÷B•2æ6÷•÷7G&–FR¢‡6—¦U÷B’‚†2Óç"çv–GF‚²'rÒ’ò'r’¢'#°¢6—¦U÷B&÷w2Ò†2Óç"æ†V–v‡B²&‚Ò’ò&‚Â–ÖrÒ2æ6÷•ö–Öu÷7G&–FRò‡6—¦U÷B•2æ6÷•ö–Öu÷7G&–FR¢&÷r¢&÷w3²òòW"Æ–W#¢6WD6÷”–ÖvU7G&–FRÂVÇ6RF–v‡FÇ’6¶V@¢–çB—36BÒ2ÓçFW‚ÓæBçF&vWBÓÒ"Âç¢Ò2Óç"æFWF‚ò2Óç"æFWF‚¢°¢fö–B¦Ö#²V–çCcE÷BÖó°¢–b‚×FÅöFG"†2Óç7&2Â–Ör¢ç¢ÂfÖ"ÂfÖò’’²$TEôôä4Rƒ3Â'FW‡GW&^(i&'VffW"6÷’FW7F–æF–öâ—2–âæòÖVÖ÷'’ööÂ"“²&WGW&ã²Ð¢6'uöæ÷FR†2Óç7&2Â–Ör¢ç¢Â4%uôuR“°¢–CÄÕDÃD6ö×WFT6öÖÖæDVæ6öFW#â"Ò&Æ—B‚“°¢–b†—36B¢Væ5ö%ö6÷•÷FW‡GW&Uö'VffW"†"ÂBÂÂÆWfVÂÂÕDÄ÷&–v–äÖ¶R†2Óç"ç†öfg6WBÂ2Óç"ç–öfg6WBÂ2Óç"ç¦öfg6WB’ÂÕDÅ6—¦TÖ¶R†2Óç"çv–GF‚Â2Óç"æ†V–v‡BÂç¢’Â…õö'&–FvR–CÄÕDÄ'VffW#â–Ö"ÂÖòÂ&÷rÂ–Ör“°¢VÇ6P¢f÷"†–çB¢Ò²¢Âç£²¢²²¢Væ5ö%ö6÷•÷FW‡GW&Uö'VffW"†"ÂBÂÆ–W#²2Óç"ç¦öfg6WB²¢ÂÆWfVÂÂÕDÄ÷&–v–äÖ¶R†2Óç"ç†öfg6WBÂ2Óç"ç–öfg6WBÂ’ÂÕDÅ6—¦TÖ¶R†2Óç"çv–GF‚Â2Óç"æ†V–v‡BÂ’Â…õö'&–FvR–CÄÕDÄ'VffW#â–Ö"ÂÖò²¢¢–ÖrÂ&÷rÂ–Ör“°¢–çB6†–gBÒ&%ö6÷•÷6†–gB†2ÓçFW‚ÂB“²òò–çFòF†RwVW7Bw2÷&FW"Â–âF†R'VffW ¢–b‡6†–gB’²V–çCcE÷BÒ‚…õö'&–FvR–CÄÕDÄ'VffW#â–Ö"’æwTFG&W72²Öó²6—¦U÷Bv"Ò‡6—¦U÷B’‚†2Óç"çv–GF‚²'rÒ’ò'r’¢'#°¢–b‚&%÷7vöwR†Â&÷rÂ–ÖrÂÂ&÷rÂ–ÖrÂv"Â&÷w2Âç¢Â6†–gB’’$TEôôä4RƒbÂ%$t#ScRÖfÖ–Ç’FW‡GW&^(i&'VffW"6÷’ÆVgB–âÖWFÂw2÷&FW#¢&VBæB&ÇVRW†6†ævVB"“²Ð§Ð¢òò5RFW‡GW&R&VB…&VEFW†VÇ2“¢f÷"6¶vVBFW‡GW&W2F†RööÂÖVÖ÷'’—2F†Rw&÷VæBG'WF‚ÂFRÖÆ–÷WBæB&VBF—&V7FÇ“²F†R&W7B&R&VB&6²g&öÒF†RÕDÅFW‡GW&R'’&Æ—B‡F†R6ÆÆW"†2Ç&VG’v—FVBf÷"F†RuRW"ådâ'VÆW2§fö–B×FÅ÷&VE÷FW†VÇ2†6öç7BFW…÷B§BÂVç6–væVBÆWfVÂÂVç6–væVBÆ–W#Â6öç7Bådæ6÷•&Vv–öâ§"ÂV–çC…÷B¦G7BÂ6—¦U÷B&÷rÂ6—¦U÷B–Ör’°¢‡eö6†V6²‡BÓç6W&–ÂÂÆWfVÂÂÆ–W#Â$5R&VF&6²"“°¢B‚%R7U÷&VB6W#ÒVÆÇRÇbWR6ÂWR‡—£ÒVBÂVBÂVB6—¦SÒVBÂVBÂVBG7CÒW&÷sÒVÇR–ÖsÒVÇR"ÂBÓç6W&–ÂÂÆWfVÂÂÆ–W#Â"Óç†öfg6WBÂ"Óç–öfg6WBÂ"Óç¦öfg6WBÂ"Óçv–GF‚Â"Óæ†V–v‡BÂ"ÓæFWF‚ÂG7BÂ‡Vç6–væVBÆöær—&÷rÂ‡Vç6–væVBÆöær––Ör“°¢–çB'rÂ&‚Â'"Òçfåöf×Eö'"‡BÓæBæf÷&ÖBÂf'rÂf&‚“°¢6—¦U÷Bv"Ò‡6—¦U÷B’‚‡"Óçv–GF‚²'rÒ’ò'r’¢'"Â†"Ò‡"Óæ†V–v‡B²&‚Ò’ò&‚ÂƒÒ‡6—¦U÷B’‡"Óç†öfg6WBò'r’¢'"Â“Ò"Óç–öfg6WBò&ƒ°¢–çBç¢Ò"ÓæFWF‚ò"ÓæFWF‚¢Â—36BÒBÓæBçF&vWBÓÒ#°¢òòFW‡GW&Rv†÷6R6öçFVçBF†R†÷7BF¶W2g&öÒwVW7BÖVÖ÷'’‡FW…ög&öÕ÷7F÷&vR’—2&VBF†W&RÂöâv†FWfW"F‡&VB6·2Âv—F†÷WB&W6öÇf–ær†æBÖ–&RWÆöF–ær’F†R†÷7BFW‡GW&S ¢òòF†RWÆöB&F6‚&VÆöæw2FòF†R7V&Ö—BF‡&VBâFV'2öbF†R¶–ævFöÒ&VG2FW‡GW&W2öb—G27'6RööÂöâ—G2÷vâF‡&VG2†&¶S£¤ÖöGVÆTÖw"“²&W6öÇfVBF†W&RÂF†RFW‡GW&Rv0¢òò&R×WÆöFVBöâF†BF‡&VBæBWö÷væW%ö6†V6²&÷'FVBÂfWr6V6öæG2gFW"F†R6ÖR&6²g&öÒF†R&6¶w&÷VæB‚3Cs’à¢òòF†B–æ6ÇVFW2FW…öF—'G•÷WÆöBFW‡GW&RF†R5Rw&÷FR6–æ6R&Vv—7G&F–öâ'WBF†R7V&Ö—BF‡&VB†2æ÷B&W6öÇfVB–WC¢FV'2öbF†R¶–ævFöÒf–ÆÇ2#Gƒ#B$3BÖFW&–À¢òòFW‡GW&W2g&öÒF†R5RÂ&VG2F†VÒ&6²v—F‚&VEFW†VÇ2öâ—G2÷vâF‡&VB&Vf÷&Rç’G&rW6W2F†VÒÂæB6÷–W2F†R&Æö6·2–çFò—G2S#ƒS#&÷Vv†æW72FÆ2â&VBg&öÒF†P¢òòæWfW"×WÆöFVB†÷7BFW‡GW&RF†W’vW&R¦W&÷2ÂF†RFÆ27F–VBV×G’ÂæBF†R÷Væ–ær6fRw27FöæR&Æö6·2v÷B&÷Vv†æW72æBæòF÷&6‚Æ–v‡@¢–çBçfåögE÷w&—GFVâ†6öç7BFW…÷B¢“°¢–b‡FW…ög&öÕ÷7F÷&vR‡B’ÇÂ„DÕôtÔUõDU…ôD•%E•õUÄôBbbBÓæÖVÒbbçfåögE÷w&—GFVâ‡B’’’°¢òò7'6RööÂw2FW‡GW&R—2&VBF‡&÷Vv‚6÷’vF†W&VBg&öÒ—G2vRF&ÆRÂVæÖVBvW22Â2¶u÷WÆöEòWÆöG2—C¢v†W&Ræ÷F†–ær—2ÖVB—G2FG&W72&ævR—0¢òò–æ66W76–&ÆR‡&VF–ærF†R7F÷&vRF—&V7FÇ’7F÷VBF†R†÷7Bv—F‚4”t%U2¢6öç7BV–çC…÷B¦ÖVÒÒBÓæÖVÓ²V–çC…÷B§'BÒåTÄÃ°¢–b‡BÓçööÂbbçfå÷7'6U÷ööÂ‡BÓçööÂ’’°¢–b‚‡'BÒÖÆÆö2‡BÓäÂç6—¦RòBÓäÂç6—¦R¢’’’ÔD”R‚'7'6RFW‡GW&R6÷’öbVÆÇR'—FW2f–ÆVB"Â‡Vç6–væVBÆöærÆöær—BÓäÂç6—¦R“°¢çfå÷7'6Uö6÷’‡BÓçööÂÂ'BÂ‡V–çCcE÷B’‡V–çGG%÷B—BÓæÖVÒÂBÓäÂç6—¦R“²ÖVÒÒ'C°¢Ð¢V–çC3%÷BÇrÒÔ‚‡BÓæBçv–GF‚ãâÆWfVÂÂR’ÂÆ‚ÒÔ‚‡BÓæBæ†V–v‡BãâÆWfVÂÂR’ÂÇ¢Ò—36BòÔ‚‡BÓæBæFWF‚ãâÆWfVÂÂR’¢°¢6—¦U÷BÇÒ‡6—¦U÷B’‚†Çr²'rÒ’ò'r’¢'"ÂÇ"Ò†Æ‚²&‚Ò’ò&ƒ°¢V–çC…÷B¦Æ–âÒÖÆÆö2†Ç¢Ç"¢Ç¢“°¢f÷"†–çB¢Ò²¢Âç£²¢²²’°¢Vç6–væVBÆ–W"Ò—36Bò¢Æ–W#²"Óç¦öfg6WB²£°¢–b‚†—36Bbb¢ÓÒ’ÇÂ—36B’–b†çfåöFW7v—§¦ÆR‚gBÓæBÂÆWfVÂÂÆ–W"ÂÖVÒÂÆ–âÂÇ’’²$TEôôä4Rƒ3"Â%&VEFW†VÇ2FW7v—§¦ÆRf–ÆVB"“²'&V³²Ð¢6öç7BV–çC…÷B§7&2ÒÆ–â²†—36Bò‡6—¦U÷B’‡"Óç¦öfg6WB²¢’¢Ç¢Ç"¢“°¢f÷"‡6—¦U÷B’Ò²’Â†#²’²²’ÖVÖ7’†G7B²¢¢–Ör²’¢&÷rÂ7&2²‡“²’’¢Ç²ƒÂv"“°¢Ð¢g&VR†Æ–â“²g&VR‡'B“°¢&WGW&ã°¢Ð¢òòwVW7BF‡&VB6âÆ—fR–æFVf–æ—FVÇ’v—F†÷WBG&–æ–ær—G2÷WFW"ööÃ¢&VÆV6R&VF&6²FV×÷&&–W2†W&RÂgFW"F†RuRæB5R6÷–W2f–æ—6‚à¢òòf–æF–æs¢Fö72÷&VfW&Væ6RöÖVÖ÷'’æÖBÂuR&VF&6²v—F†÷WBâWF÷&VÆV6RööÂà¢WF÷&VÆV6WööÂ°¢6ögEö†&FVâ‡B“°¢–CÄÕDÅFW‡GW&Sâ×BÒFW…ö×FÂ‡B“°¢²Vç6–væVBÇbÒ‡Vç6–væVB–ÆWfVÃ²–b†×BbbFW…÷6ögEöÆWfVÂ‡BÂfÇb’’²ÖV×6WB†G7BÂÂ–Ör¢ç¢“²&WGW&ã²ÒÆWfVÂÒÇc²Ð¢–b‚×BÇÂFW…öÆWfVÅöö²†×BÂÆWfVÂÂ$5R&VF&6²"’’²ÖV×6WB†G7BÂÂ–Ör¢ç¢“²&WGW&ã²Ð¢×FÅöf–æ—6…öwUöB…u5õ$TEDU‚“°¢–CÄÕDÄ'VffW#â'VbÒ¶uöFWbæWt'VffW%v—F„ÆVæwFƒ§v"¢†"¢ç¢÷F–öç3¤ÕDÅ&W6÷W&6U7F÷&vTÖöFU6†&VEÓ°¢–CÄÕDÄ6öÖÖæD'VffW#â6"Òö6"‚“²–CÄÕDÄ&Æ—D6öÖÖæDVæ6öFW#â"ÒVæ5ö%ö&Vv–â†6"“°¢f÷"†–çB¢Ò²¢Âç£²¢²²¢¶"6÷”g&öÕFW‡GW&S¦×B6÷W&6U6Æ–6S¦—36Bò¢Æ–W#²"Óç¦öfg6WB²¢6÷W&6TÆWfVÃ¦ÆWfVÂ6÷W&6T÷&–v–ã¤ÕDÄ÷&–v–äÖ¶R‡"Óç†öfg6WBÂ"Óç–öfg6WBÂ—36Bò"Óç¦öfg6WB²¢¢¢6÷W&6U6—¦S¤ÕDÅ6—¦TÖ¶R‡"Óçv–GF‚Â"Óæ†V–v‡BÂ’Fô'VffW#¦'VbFW7F–æF–öäöfg6WC§¢¢v"¢†"FW7F–æF–öä'—FW5W%&÷s§v"FW7F–æF–öä'—FW5W$–ÖvS§v"¢†%Ó°¢Væ5ö%öVæB†"“²¶6"6öÖÖ—EÓ²¶6"v—EVçF–Ä6ö×ÆWFVEÓ°¢f÷"†–çB¢Ò²¢Âç£²¢²²’f÷"‡6—¦U÷B’Ò²’Â†#²’²²’ÖVÖ7’†G7B²¢¢–Ör²’¢&÷rÂ‡V–çC…÷B¢–'Vbæ6öçFVçG2²¢¢v"¢†"²’¢v"Âv"“°¢–çB6†–gBÒ&%ö6÷•÷6†–gB‡BÂ×B“²–b‡6†–gB’&%÷7v÷&÷w2†G7BÂv"Â†"Â‡6—¦U÷B–ç¢Â&÷rÂ–ÖrÂ6†–gB“²òò–çFòF†RwVW7Bw2÷&FW ¢Ð§Ð¢òòv†WF†W"FW‡GW&Rw27F÷&vR—2v†W&RF†R†÷7BF¶W2—G26öçFVçBg&öÒÂæ÷r÷"öæ6RF†R5Rw&—FW2—C¢FW…ö×FÅòw26¶vVBFW‡GW&W2†6¶vVBÆ–÷WBÂ5R×w&—GFVâÂWÆöFVB’æBF†÷6P¢òòFW…öfÇW6…÷WÆöBòFW…öF—'G•÷WÆöB&Vv—7FW&VBæB†2æ÷B&W6öÇfVB–WC²æ÷BöæRF†RuR÷vç2âuR6÷’–çFò7V6‚FW‡GW&R—2w&—GFVâ–çFò—G27F÷&vR2vVÆÂ†çfå÷FW…öÖ—'&÷%ö#'Bð¢òòçfå÷FW…öÖ—'&÷%÷C'B–â†ÆUöçfâæ2’Â2F†R†&Gv&Rw26÷’v÷VÆBÂ6òF†BâWÆöBg&öÒF†R7F÷&vR6'&–W2F†R6÷–VBÆWfVÇ2‡F†R&V6öç2&RF†W&R“¢öâF†R5Rg&öÒF†R6÷W&6Rw27F÷&vP¢òòv†VâF†B†öÆG2F†R6÷W&6Rw26öçFVçB‡FW…÷7F÷&vU÷7&2’ÂVÇ6R&VB&6²gFW"F†R6÷’†Ö—'&÷%÷&VF&6²§7FF–2–çBFW…÷7F÷&vUöG7B†6öç7BFW…÷B§B’°¢–b‡FW…÷6ögB‡B’’&WGW&â²òò—G2†÷7BÆWfVÇ2&Ræ÷BF†RvÖRw3¢æ÷F†–ær&VG26ögBFW‡GW&R&6²–çFò—G27F÷&vP¢–b‚BÓæÖVÒÇÂBÓçööÂÇÂ‡BÓçööÂÓæfÆw2b„ådåôÔTÔõ%•õôôÅôdÄu5ô5Uôäõô44U52ÂådåôÔTÔõ%•õôôÅôdÄu5õd•%ETÂ’’ÇÂBÓæBç6×ÆW2âÇÂBÓæBçF&vWBÓÒådåõDU…EU$UõD$tUEô%TddU ¢ÇÂ‡BÓæBæfÆw2b„ådåõDU…EU$UôdÄu5ôÄ”äT"ÂådåõDU…EU$UôdÄu5ôÄ”äT%õ$TäDU%õD$tUBÂådåõDU…EU$UôdÄu5ô4ôÕ$U54”$ÄRÂådåõDU…EU$UôdÄu5ô”ÔtR’’’&WGW&â°¢–b‡BÓæ†5öÆ–÷WBÇÂFW…ö7U÷w&—GFVâ‡BÓç6W&–Â’ÇÂçfåögE÷G&6¶VB‡BÓç6W&–Â’’&WGW&â×FÅ÷FW…öwUö÷væVB‡BÓç6W&–Â“°¢&WGW&â·FW…övWB‡BÓç6W&–Â’—4¶–æDöd6Æ73¥´çe¶r6Æ75ÕÓ°§Ð¢òò&VF&6·2VWVVBæBæ÷BÆæFVBÂ'’FW‡GW&R†Ö—'&÷%÷&VF&6²òÖ—'&÷%÷&%öÆæB’âVçF–ÂöæRÆæG2F†RFW‡GW&Rw27F÷&vRÆ6·2F†R6÷–VB'—FW3¢6÷’g&öÒF†RFW‡GW&R—2&VB&6²FöòÂæ@¢òò6ò—26÷’–çFò—BÂ6òF†B&÷F‚ÆæB–âF†R÷&FW"F†W’vW&R&V6÷&FVB†7V&Ö—76–öâw2÷7BÖ÷2'Vâ–â÷&FW"ÂæB7V&Ö—76–öç26ö×ÆWFR–â÷&FW"’âgVÆÂF&ÆR6÷VçG2F†R&W7B0¢òòVçG&6¶VBÂv†–6‚Ö¶W2WfW'’FW‡GW&RVæF–ærVçF–ÂF†W’Ææ@§7FF–2F‡&VEö×WFW…÷Buö×&%ö×RÒD…$TEôÕUDU…ô”ä•D”Ä•¤U#²7FF–27G'V7B²V–çCcE÷B6W&–Ã²–çBã²Òuö×&%³cEÓ²7FF–2–çBuö×&%öâÂuö×&%÷VçG&6¶VC°§7FF–2fö–B×&%öæ÷FR‡V–çCcE÷B6W&–ÂÂ–çBB’°¢F‡&VEö×WFW…öÆö6²‚fuö×&%ö×R“°¢–çB’Ò²v†–ÆR†’Âuö×&%öâbbuö×&%¶•Òç6W&–ÂÒ6W&–Â’’²³°¢–b†’ÓÒuö×&%öâbb†BÂÇÂuö×&%öâÓÒ†–çB’‡6—¦Vöbuö×&"ò6—¦Vöbuö×&%³Ò’’’uö×&%÷VçG&6¶VB³ÒC°¢VÇ6R²–b†’ÓÒuö×&%öâ’²uö×&%¶•Òç6W&–ÂÒ6W&–Ã²uö×&%¶•ÒæâÒ²uö×&%öâ²³²Ò–b‚†uö×&%¶•Òæâ³ÒB’’uö×&%¶•ÒÒuö×&%²ÒÖuö×&%öåÓ²Ð¢F‡&VEö×WFW…÷VæÆö6²‚fuö×&%ö×R“°§Ð§7FF–2–çB×&%÷VæF–ær‡V–çCcE÷B6W&–Â’°¢F‡&VEö×WFW…öÆö6²‚fuö×&%ö×R“²–çB"Òuö×&%÷VçG&6¶VBâ°¢f÷"†–çB’Ò²"bb’Âuö×&%öã²’²²’"Òuö×&%¶•Òç6W&–ÂÓÒ6W&–Ã°¢F‡&VEö×WFW…÷VæÆö6²‚fuö×&%ö×R“²&WGW&â#°§Ð¢òòF†RuRÖ’†öÆB6öçFVçBF†R7F÷&vRÆ6·3¢F†R†÷7BFW‡GW&RÂ÷"F†RöæR6¶vVB&V6÷&B†öÆG2†6ÆV"÷"G&vâ6÷’v—fW2—B&VæFW"×F&vWBW6S²×FÅ÷FW…öwUö÷væVB6VW2F†R&V6÷&BÀ¢òòæ÷BF†RFW‡GW&R–â—B’Â6â&R&VæFW&VB–çFò÷"w&—GFVâ'’6†FW'2Â÷"&VF&6²–çFò—B—2VæF–ærâ7V6‚FW‡GW&Rw27F÷&vR—2æò6÷W&6Rf÷"6÷’w25RÖ—'&÷ §7FF–2–çBFW…öwU÷F÷V6†VB‡V–çCcE÷B6W&–Â’°¢õ÷Vç6fU÷Vç&WF–æVB–BòÒFW…övWB‡6W&–Â“°¢–b…¶ò—4¶–æDöd6Æ73¥´çe¶r6Æ75ÕÒ’òÒ‚„çe¶r¢–ò’ÓçFWƒ°¢&WGW&âòbb¶ò6öæf÷&×5Fõ&÷Fö6öÃ¤&÷Fö6öÂ„ÕDÅFW‡GW&R•Òbb‚‚†–CÄÕDÅFW‡GW&Sâ–ò’çW6vRb„ÕDÅFW‡GW&UW6vU&VæFW%F&vWBÂÕDÅFW‡GW&UW6vU6†FW%w&—FR’’Ò°§Ð§7FF–2–çBFW…÷7F÷&vU÷7&2†6öç7BFW…÷B§B’²&WGW&âFW…÷7F÷&vUöG7B‡B’bbFW…öwU÷F÷V6†VB‡BÓç6W&–Â’bb×&%÷VæF–ær‡BÓç6W&–Â“²Ð§7FF–2fö–BÖ—'&÷%öæ÷FR†–çB&2Â6öç7BFW…÷B§BÂVç6–væVBÆWfVÂÂ6öç7B6†"¦g&öÒÂ–çB&"’²òòF†Rf—'7BfWröbV6‚¶–æBöb6÷W&6RæBv’‡&#¢&VB&6²’ÂæBF†Rf—'7BfWrf–ÇW&W0¢7FF–2–çBå³%Õ³%ÒÂæc²–çB²Òg&öÕ³ÒÓÒv"s°¢–b‡&2ÓÒ’²–b†å¶µÕ·&%Ò²²ÂB’ÔÄôr‚"W>(i'FW‡GW&R6÷’–çFò6W#ÒVÆÇRWW‚WRf÷&ÖBR7‚ÆWfVÂWS¢W2‡F†RFW‡GW&R—2WÆöFVBg&öÒ—G27F÷&vR’"Âg&öÒÂ‡Vç6–væVBÆöærÆöær—BÓç6W&–ÂÂBÓæBçv–GF‚ÂBÓæBæ†V–v‡BÂBÓæBæf÷&ÖBÂÆWfVÂÀ¢&"ò&—G2&Vv–öâ—2&VB&6²–çFòF†R7F÷&vRv†VâF†RuR6ö×ÆWFW2"¢'w&—GFVâ–çFòF†R7F÷&vR2vVÆÂ"“²Ð¢VÇ6R–b†æb²²ÂB’ÔÄôr‚%µ$TEÒW>(i'FW‡GW&R6÷’–çFò6W#ÒVÆÇRWW‚WW‚WRf÷&ÖBR7‚F&vWBVBÆWfVÂWRæ÷Bw&—GFVâ–çFò—G27F÷&vS¢F†RæW‡BWÆöBg&öÒF†W&RG&÷2—B"Âg&öÒÂ‡Vç6–væVBÆöærÆöær—BÓç6W&–ÂÂBÓæBçv–GF‚ÂBÓæBæ†V–v‡BÂBÓæBæFWF‚ÂBÓæBæf÷&ÖBÂBÓæBçF&vWBÂÆWfVÂ“°§Ð¢òò6÷W&6RF†RuRw&÷FR†2æò6öçFVçB–âwVW7BÖVÖ÷'’FòÖ—'&÷"öâF†R5S¢F†RFW7F–æF–öâ&Vv–öâ—2&VB&6²g&öÒF†R†÷7BFW‡GW&RgFW"F†R6÷’ÂöâF†RuRÂ–çFò6†&VB'VffW"Âæ@¢òòw&—GFVâ–çFòF†R7F÷&vRv†VâF†R7V&Ö—76–öâ6ö×ÆWFW2†÷7BÖ÷Â'Vâ&Vf÷&RF†RfVæ6R—2V&Æ—6†VBÂ6òF†RvÖR6ææ÷B†fR&VÆV6VBF†RööÂ–WC²—B'Vç2gFW"ç’WÆöBöbF†P¢òòFW‡GW&RVæ6öFVB&Vf÷&R—BÂ6òF†RvW2—BF—'F–W2&R6'&–VB'’F†RæW‡BöæR’âF†RFW‡GW&R&V6÷&B—26÷–VC¢F†RvÖRÖ’f–æÆ—¦RF†Rö&¦V7Böæ6R—B†2v—FV@§G—VFVb7G'V7B²FW…÷BC²Vç6–væVBÆWfVÂÂÆ–W#²ådæ6÷•&Vv–öâ#²6—¦U÷B&÷rÂ–Ös²fö–B¦'VbÂ§7&3²–çB6†–gC²ÒÖ—'&÷%÷&#²òò'Vc¢F†R&VF&6²'VffW"Â&WF–æVBf÷"F†R†æFÆW#²7&3¢&VæFW"72w2†÷7BFWƒ²6†–gC¢&%ö6÷•÷6†–g@§7FF–26—¦U÷B×FÅ÷eö&Æö6µö'—FW2„ÕDÅ—†VÄf÷&ÖBb’²òò'—FW2W"FW†VÂÂ÷"W"&Æö6²öb6ö×&W76VBf÷&ÖC²v†Vâæ÷B¶æ÷và¢–çB&"Ò&5ö&Æö6µö'—FW2‡b“²–b†&"’&WGW&â‡6—¦U÷B–&#°¢–b‡bãÒÕDÅ—†VÄf÷&ÖD5D5óGƒE÷5$t"bbbÃÒÕDÅ—†VÄf÷&ÖD5D5ó'ƒ%ô„E"’&WGW&âc°¢–b‡bãÒÕDÅ—†VÄf÷&ÖDT5õ#Væ÷&ÒbbbÃÒÕDÅ—†VÄf÷&ÖDUD3%õ$t#„÷5$t"’&WGW&âbÓÒÕDÅ—†VÄf÷&ÖDT5õ$sVæ÷&ÒÇÂbÓÒÕDÅ—†VÄf÷&ÖDT5õ$s6æ÷&ÒÇÂbÓÒÕDÅ—†VÄf÷&ÖDT5õ$t$‚ÇÂbÓÒÕDÅ—†VÄf÷&ÖDT5õ$t$…÷5$t"òb¢ƒ°¢&WGW&â×FÅ÷eö'—FW2‡b“°§Ð¢òò&VF&6²'VffW'2&R¶WBöæ6RF†W’†fRÆæFVBÂ'’÷vW"Ööb×Gvò6—¦S¢æWröæRV6‚F–ÖRvVçB–çFòF†R&W6–FVæ7’6WBæB÷WBv–âgFW"—G27V&Ö—76–öâÂ6öÖÖ—BV6‚v¢òò…FV'2öbF†R¶–ævFöÒ&VG2&6²&÷WBR&VæFW"76W2g&ÖRöâF†R•†öæS¢&W6–FVæ7’6öÖÖ—G2#SÓc×2WfW'’3g&ÖW2v–ç7B&÷WB#R&Vf÷&RÂv—D‡V"3“3’âBÖ÷7BbÔ"—2¶W@§7FF–2F‡&VEö×WFW…÷Buö×&'Veö×RÒD…$TEôÕUDU…ô”ä•D”Ä•¤U#²7FF–2õ÷7G&öær–CÄÕDÄ'VffW#âuö×&'Ve³eÓ²7FF–2–çBuö×&'Veöã²7FF–26—¦U÷Buö×&'Veö'—FW3°§7FF–2–CÄÕDÄ'VffW#â×&'VeövWB‡6—¦U÷Bâ’°¢6—¦U÷B2ÒC“c²v†–ÆR†2Ââ’2ÃÃÒ°¢–CÄÕDÄ'VffW#â"Òæ–Ã°¢F‡&VEö×WFW…öÆö6²‚fuö×&'Veö×R“°¢f÷"†–çB’Ò²’Âuö×&'Veöã²’²²’–b†uö×&'Ve¶•ÒæÆVæwF‚ÓÒ2’²"Òuö×&'Ve¶•Ó²uö×&'Ve¶•ÒÒuö×&'Ve²ÒÖuö×&'VeöåÓ²uö×&'Ve¶uö×&'VeöåÒÒæ–Ã²uö×&'Veö'—FW2ÓÒ3²'&V³²Ð¢F‡&VEö×WFW…÷VæÆö6²‚fuö×&'Veö×R“°¢–b‚"bb†"Ò¶uöFWbæWt'VffW%v—F„ÆVæwFƒ¦2÷F–öç3¤ÕDÅ&W6÷W&6U7F÷&vTÖöFU6†&VEÒ’’&W5öFB†"“°¢&WGW&â#°§Ð§7FF–2fö–B×&'Ve÷WB†–CÄÕDÄ'VffW#â"’²òòF†R&VF&6²†2ÆæFVC¢F†RuR—2FöæRv—F‚F†R'VffW ¢F‡&VEö×WFW…öÆö6²‚fuö×&'Veö×R“°¢–çB¶WBÒuö×&'VeöâÂbbbuö×&'Veö'—FW2²"æÆVæwF‚ÃÒƒgRÃÂ#“°¢–b†¶WB’²uö×&'Ve¶uö×&'Veöâ²µÒÒ#²uö×&'Veö'—FW2³Ò"æÆVæwFƒ²Ð¢F‡&VEö×WFW…÷VæÆö6²‚fuö×&'Veö×R“°¢–b‚¶WB’&W5÷&VÖ÷fR†"“°§Ð§7FF–2fö–BÖ—'&÷%÷&VF&6²†6öç7BFW…÷B§BÂ–CÄÕDÅFW‡GW&SâBÂVç6–væVBÆWfVÂÂVç6–væVBÆ–W#Â6öç7Bådæ6÷•&Vv–öâ§"Â6öç7B6†"¦g&öÒÂ–çB'6÷&"’²òò'6÷&#¢&VæFW"72w2&VF&6²Âv†–6‚F†RFW‡GW&RF¶W22—G2÷vâ‡¶uö'6÷&%öæ÷FR¢–çB'rÂ&‚Â'"Òçfåöf×Eö'"‡BÓæBæf÷&ÖBÂf'rÂf&‚’Âç¢Ò"ÓæFWF‚ò"ÓæFWF‚¢°¢–b‚'"ÇÂ"Óçv–GF‚ÃÒÇÂ"Óæ†V–v‡BÃÒÇÂBç6×ÆT6÷VçBâÇÂ×FÅöf×EöFWF‚†Bç—†VÄf÷&ÖB’ÇÂBç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖE7FVæ6–Ã‚ÇÂ×FÅ÷eö&Æö6µö'—FW2†Bç—†VÄf÷&ÖB’Ò‡6—¦U÷B–'"’²Ö—'&÷%öæ÷FR‚ÓÂBÂÆWfVÂÂg&öÒÂ“²&WGW&ã²Ð¢6—¦U÷B&÷rÒ‡6—¦U÷B’‚‡"Óçv–GF‚²'rÒ’ò'r’¢'"Â–ÖrÒ&÷r¢‡6—¦U÷B’‚‡"Óæ†V–v‡B²&‚Ò’ò&‚“°¢–CÄÕDÄ'VffW#â"Ò×&'VeövWB†–Ör¢ç¢“°¢–b‚"’²Ö—'&÷%öæ÷FR‚ÓÂBÂÆWfVÂÂg&öÒÂ“²&WGW&ã²Ð¢–CÄÕDÃD6ö×WFT6öÖÖæDVæ6öFW#âRÒ&Æ—B‚“°¢f÷"†–çB¢Ò²¢Âç£²¢²²’Væ5ö%ö6÷•÷FW‡GW&Uö'VffW"†RÂBÂÆ–W#²"Óç¦öfg6WB²¢ÂÆWfVÂÂÕDÄ÷&–v–äÖ¶R‡"Óç†öfg6WBÂ"Óç–öfg6WBÂ’ÂÕDÅ6—¦TÖ¶R‡"Óçv–GF‚Â"Óæ†V–v‡BÂ’Â"Â‡6—¦U÷B—¢¢–ÖrÂ&÷rÂ–Ör“°¢Ö—'&÷%÷&"¦ÒÒÖÆÆö2‡6—¦Vöb¦Ò“²¦ÒÒ†Ö—'&÷%÷&"—²§BÂÆWfVÂÂÆ–W#Â§"Â&÷rÂ–ÖrÂ‡fö–B¢”4d'&–Fv–æu&WF–â†"’Â'6÷&"ò‡fö–B¢”4d'&–Fv–æu&WF–â†Bç&VçEFW‡GW&RòBç&VçEFW‡GW&R¢B’¢åTÄÂÂ&%ö6÷•÷6†–gB‡BÂB—Ó°¢÷7B‚‡÷7Eö÷—³BÂåTÄÂÂÂÂÂÂÒÂÒ“°¢×&%öæ÷FR‡BÓç6W&–ÂÂ“°¢–b†ÒÓç7&2’¶uö'6÷&%÷VæF–ær‡BÓç6W&–ÂÂ“°¢Ö—'&÷%öæ÷FRƒÂBÂÆWfVÂÂg&öÒÂ“°§Ð§7FF–2fö–BÖ—'&÷%÷&%öÆæB‡fö–B§’²òòF†R7V&Ö—76–öâw26ö×ÆWF–öâ†æFÆW ¢Ö—'&÷%÷&"¦ÒÒ²–CÄÕDÄ'VffW#â"Ò4d'&–Fv–æu&VÆV6R†ÒÓæ'Vb“°¢–b†ÒÓç6†–gB’²–çB'rÂ&ƒ²çfåöf×Eö'"†ÒÓçBæBæf÷&ÖBÂf'rÂf&‚“²&%÷7v÷&÷w2†"æ6öçFVçG2ÂÒÓç&÷rÂ‡6—¦U÷B’‚†ÒÓç"æ†V–v‡B²&‚Ò’ò&‚’Â‡6—¦U÷B’†ÒÓç"æFWF‚òÒÓç"æFWF‚¢’ÂÒÓç&÷rÂÒÓæ–ÖrÂÒÓç6†–gB“²Òòò–çFòF†RwVW7Bw2÷&FW ¢–b†çfå÷FW…öÖ—'&÷%ö#'B‚fÒÓçBÂÒÓæÆWfVÂÂÒÓæÆ–W#ÂfÒÓç"Â"æ6öçFVçG2ÂÒÓç&÷rÂÒÓæ–Ör’’²7FF–2–çBã²–b†â²²ÂB’ÔÄôr‚%µ$TEÒuR6÷’&VB&6²–çFòF†R7F÷&vRöbFW‡GW&R6W#ÒVÆÇRÆWfVÂWS¢æ÷Bw&—GFVâ"Â‡Vç6–væVBÆöærÆöær–ÒÓçBç6W&–ÂÂÒÓæÆWfVÂ“°¢–b†ÒÓç7&2’4e&VÆV6R†ÒÓç7&2“²Ð¢VÇ6R–b†ÒÓç7&2’¶uö'6÷&%öæ÷FR‚fÒÓçBÂÒÓç7&2“°¢–b†ÒÓç7&2’¶uö'6÷&%÷VæF–ær†ÒÓçBç6W&–ÂÂÓ“²òògFW"F†Ræ÷FS¢F†RæW‡B&W6öÇWF–öâf–æG2—@¢×&%öæ÷FR†ÒÓçBç6W&–ÂÂÓ“°¢×&'Ve÷WB†"“°¢g&VR†Ò“°§Ð§7FF–2fö–B6÷•ö#'B†6öç7Bö6÷•ö#'B¦2’°¢–b†2ÓçFW‚’§6¶—öFVÂ†2ÓçFW‚Óç6W&–Â“°¢–CÄÕDÅFW‡GW&SâBÒFW…ö×FÂ†2ÓçFW‚“°¢–b‚B’²B‚$26W#ÒVÆÇR†æò†÷7BFW‡GW&R’"Â2ÓçFW‚Óç6W&–Â“²&WGW&ã²Ð¢–çB'rÂ&‚Â'"Òçfåöf×Eö'"†2ÓçFW‚ÓæBæf÷&ÖBÂf'rÂf&‚“°¢Vç6–væVBÆWfVÂÒ2Óæ†5÷f–Wrbb2Óçf–Wrç6WEöÆWfVÇ2ò2Óçf–Wræ&6UöÆWfVÂ¢ÂÆ–W#Ò2Óæ†5÷f–Wrbb2Óçf–Wrç6WEöÆ–W'2ò2Óçf–WræÖ–åöÆ–W"¢°¢–b‚FW…÷6ögEöÆWfVÂ†2ÓçFW‚ÂfÆWfVÂ’’&WGW&ã²òò–çFòÆWfVÂöb6ögBFW‡GW&RÂv†–6‚—2æ÷B¶W@¢–b‚FW…öÆWfVÅöö²‡BÂÆWfVÂÂ&'VffW"×Fò×FW‡GW&R6÷’"’’&WGW&ã°¢6—¦U÷B&÷rÒ2æ6÷•÷7G&–FRò‡6—¦U÷B•2æ6÷•÷7G&–FR¢‡6—¦U÷B’‚†2Óç"çv–GF‚²'rÒ’ò'r’¢'#°¢6—¦U÷B&÷w2Ò†2Óç"æ†V–v‡B²&‚Ò’ò&‚Â–ÖrÒ2æ6÷•ö–Öu÷7G&–FRò‡6—¦U÷B•2æ6÷•ö–Öu÷7G&–FR¢&÷r¢&÷w3²òòW"Æ–W#¢6WD6÷”–ÖvU7G&–FRÂVÇ6RF–v‡FÇ’6¶V@¢–çB—36BÒ2ÓçFW‚ÓæBçF&vWBÓÒ"Âç¢Ò2Óç"æFWF‚ò2Óç"æFWF‚¢°¢B‚$26W#ÒVÆÇRF&vWBVBWW‚WW‚WRf÷&ÖBR7‚ÇbWR6ÂWR&Vv–öâVBÂVBÂVBVG‚VG‚VBg&öÒR6ÆÇ‚&÷rW§R–ÖvRW§R"Â2ÓçFW‚Óç6W&–ÂÂ2ÓçFW‚ÓæBçF&vWBÂ2ÓçFW‚ÓæBçv–GF‚Â2ÓçFW‚ÓæBæ†V–v‡BÂ2ÓçFW‚ÓæBæFWF‚À¢2ÓçFW‚ÓæBæf÷&ÖBÂÆWfVÂÂÆ–W#Â2Óç"ç†öfg6WBÂ2Óç"ç–öfg6WBÂ2Óç"ç¦öfg6WBÂ2Óç"çv–GF‚Â2Óç"æ†V–v‡BÂ2Óç"æFWF‚Â‡Vç6–væVBÆöærÆöær–2Óç7&2Â&÷rÂ–Ör“°¢fö–B¦Ö#²V–çCcE÷BÖó°¢–b‚×FÅöFG"†2Óç7&2Â–Ör¢ç¢ÂfÖ"ÂfÖò’’²$TEôôä4Rƒ#2Â&'VffW.(i'FW‡GW&R6÷’6÷W&6R—2–âæòÖVÖ÷'’ööÂ"“²&WGW&ã²Ð¢òòF†R6÷’w2'—FW2&VÆöær–âF†RFW‡GW&Rw27F÷&vS¢w&—GFVâF†W&Ræ÷rg&öÒF†R'VffW"w2ÂVæÆW72F†RuRw&—FW2F†R'VffW"†6÷’Âf–ÆÂ÷"WFFRVWVVB–âF†—27V&Ö—76–öã¢VæEö†—G3²6†FW ¢òò÷"FW‡GW&^(i&'VffW"6÷’öb—C¢6'uö†—G3²öæRöb7V&Ö—76–öâ7F–ÆÂ'Vææ–æs¢wuöÆ—fR’÷"&VF&6²–çFòF†RFW‡GW&R—2VæF–ær‡&VB&6²gFW"F†R6÷’F†Vâ¢–çB&"Ò—36BbbFW…÷7F÷&vUöG7B†2ÓçFW‚“°¢–b‡&"bb–Ör¢ç¢bb×&%÷VæF–ær†2ÓçFW‚Óç6W&–Â’bbVæEö†—G2†2Óç7&2Â–Ör¢ç¢’bbu÷6'uö÷fW"bb6'uö†—G2†2Óç7&2Â2Óç7&2²–Ör¢ç¢’bbwuöÆ—fR†2Óç7&2Â2Óç7&2²–Ör¢ç¢¢bbçfåö†÷7E÷&VF&ÆR†2Óç7&2Â–Ör¢ç¢’bbçfå÷FW…öÖ—'&÷%ö#'B†2ÓçFW‚ÂÆWfVÂÂÆ–W#Âf2Óç"Â†6öç7BV–çC…÷B¢’‡V–çGG%÷B–2Óç7&2Â&÷rÂ–Ör’’²Ö—'&÷%öæ÷FRƒÂ2ÓçFW‚ÂÆWfVÂÂ&'VffW""Â“²&"Ò²Ð¢uö7WEö6÷’Ò°¢–b†—36Bbb76GV×‚’’²6öç7B6†"¦FBÒvWFVçb‚$DÕôådåô4$ETÕ"“²–b†FB’²òòF†R6÷W&6R'—FW2öb4BWÆöBÂ&W6–FRF†RVæ–f÷&ÒGV×2„cÆg&ÖSåö36EóÇ6W&–Ãâæ&–â¢6†"³#EÓ²6ç&–çFb‡Â6—¦VöbÂ"W2ôbVÆÇUö36EòVÆÇRæ&–â"ÂFBÂ‡Vç6–væVBÆöærÆöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR’Â‡Vç6–væVBÆöærÆöær–2ÓçFW‚Óç6W&–Â“°¢d”ÄR¦bÒf÷Vâ‡Â'v""“²–b†b’²gw&—FR‚†6öç7BV–çC…÷B¢’‚…õö'&–FvR–CÄÕDÄ'VffW#â–Ö"’æ6öçFVçG2²ÖòÂÂ–Ör¢ç¢Âb“²f6Æ÷6R†b“²ÒÒÐ¢òò–çFòÖWFÂw2÷&FW"‡&%ö6÷•÷6†–gB“¢6öçfW'FVBöâF†RuR–çFò7Fv–ær&VöbF†Rg&ÖR&VæÂv†–6‚F†R&Æ—B6÷–W2g&öÐ¢–CÄÕDÄ'VffW#â6"Ò…õö'&–FvR–CÄÕDÄ'VffW#â–Ö#²–çB6†–gBÒ&%ö6÷•÷6†–gB†2ÓçFW‚ÂB“°¢–b‡6†–gB’°¢6—¦U÷Bv"Ò‡6—¦U÷B’‚†2Óç"çv–GF‚²'rÒ’ò'r’¢'#²–CÄÕDÄ'VffW#â#²6—¦U÷BòÒ&VæöÆÆö2‡v"¢&÷w2¢ç¢Âf"“°¢–b‡&%÷7vöwR‡6"æwTFG&W72²ÖòÂ&÷rÂ–ÖrÂ"æwTFG&W72²òÂv"Âv"¢&÷w2Âv"Â&÷w2Âç¢Â6†–gB’’²6"Ò#²ÖòÒó²&÷rÒv#²–ÖrÒv"¢&÷w3²Ð¢VÇ6R$TEôôä4RƒrÂ%$t#ScRÖfÖ–Ç’'VffW.(i'FW‡GW&R6÷’ÆVgB–âF†RwVW7Bw2÷&FW#¢&VBæB&ÇVRW†6†ævVB"“°¢Ð¢–CÄÕDÃD6ö×WFT6öÖÖæDVæ6öFW#â"Ò&Æ—B‚“°¢–b†—36B¢Væ5ö%ö6÷•ö'VffW%÷FW‡GW&R†"Â6"ÂÖòÂ&÷rÂ–ÖrÂÕDÅ6—¦TÖ¶R†2Óç"çv–GF‚Â2Óç"æ†V–v‡BÂç¢’ÂBÂÂÆWfVÂÂÕDÄ÷&–v–äÖ¶R†2Óç"ç†öfg6WBÂ2Óç"ç–öfg6WBÂ2Óç"ç¦öfg6WB’“°¢VÇ6P¢f÷"†–çB¢Ò²¢Âç£²¢²²¢Væ5ö%ö6÷•ö'VffW%÷FW‡GW&R†"Â6"ÂÖò²¢¢–ÖrÂ&÷rÂ–ÖrÂÕDÅ6—¦TÖ¶R†2Óç"çv–GF‚Â2Óç"æ†V–v‡BÂ’ÂBÂÆ–W#²2Óç"ç¦öfg6WB²¢ÂÆWfVÂÂÕDÄ÷&–v–äÖ¶R†2Óç"ç†öfg6WBÂ2Óç"ç–öfg6WBÂ’“°¢–b‡&"’Ö—'&÷%÷&VF&6²†2ÓçFW‚ÂBÂÆWfVÂÂÆ–W#Âf2Óç"Â&'VffW""Â“°§Ð ¢òòFW‡GW&R6÷–W2&Æ—B6ææ÷BFòÒÒf÷&ÖB6†ævRÂ66ÆVB&Vv–öâ÷"Ö—'&÷&VBöæR„Ô•%$õ%õ‚òÔ•%$õ%õ’’„Æ–æ²w2v¶Væ–ær6öçfW'G2—G2C‡ƒs“"$s#b66VæRFò$t$dbWfW'’g&ÖS²7WW"Ö&–ò4Bv÷&ÆB66ÆW2“#ƒƒFò#ƒƒs# ¢òòv†VâÆVf–ærF†RF—FÆR“¢öæR&VæFW"72W"Æ–W"F†BG&w2F†R6÷W&6R&Vv–öâ–çFòF†RFW7F–æF–öâ&Vv–öâÂæV&W7B÷"„Ä”äT%ôd”ÅDU"’&–Æ–æV"âfÆöBÂVæ÷&ÒæB5$t"f÷&ÖG2öæÇ’Â$Bò$B'&’À¢òòFW7F–æF–öç2F†B6â&R&VæFW"F&vWG3²ç—F†–ærVÇ6R&WGW&ç2æB¶VW2F†R&VBÆ–æRâFWF‚„FWFƒ3$fÆöBòFWFƒ3$fÆöEõ7FVæ6–Ã‚–çFòF†R6ÖRf÷&ÖB’—2G&vâ'’w&—F–ærFWF‚g&öÒF†Rg&vÖVçB6†FW"À¢òòÇv—2æV&W7BÂæB7FVæ6–Â—2w&—GFVâg&öÒF†Rg&vÖVçB6†FW"Föò…µ·7FVæ6–ÅÕÒW‡÷'BÂ&VBF‡&÷Vv‚âƒ3%õ7FVæ6–Ã‚f–WröbF†R6÷W&6R’âÆ–æ²w2v¶Væ–ær†ÇfW2—G2FWF‚×7FVæ6–Â'VffW"Gv–6P¢òòƒ“#ƒƒ(i"“cƒSC(i"Cƒƒ#s’æB—G2FWF‚ÖöbÖf–VÆBÖ6²72öæÇ’w&—FW2v†W&R7FVæ6–Â&—B—26ÆV#¢v—F‚F†R7FVæ6–ÂÆVgB2F†RæWrFW‡GW&Rw2VæFVf–æVB6öçFVçBÂF†RÖ6²¶WB—G0¢òò6ÆV"fÇVRã†gVÆÂ&ÇW"’WfW'—v†W&RæBF†Rv†öÆR–7GW&Rv2÷WBöbfö7W2âv—F†÷WB7FVæ6–Âf–WrF†R7FVæ6–Â—26ÆV&VBFò–ç7FV@§7FF–2–çBf×Eö–çEö÷%öG2„ÕDÅ—†VÄf÷&ÖBb’°¢7v—F6‚†b’°¢66RÕDÅ—†VÄf÷&ÖE#…V–çC¢66RÕDÅ—†VÄf÷&ÖE#…6–çC¢66RÕDÅ—†VÄf÷&ÖE#eV–çC¢66RÕDÅ—†VÄf÷&ÖE#e6–çC¢66RÕDÅ—†VÄf÷&ÖE$s…V–çC¢66RÕDÅ—†VÄf÷&ÖE$s…6–çC ¢66RÕDÅ—†VÄf÷&ÖE#3%V–çC¢66RÕDÅ—†VÄf÷&ÖE#3%6–çC¢66RÕDÅ—†VÄf÷&ÖE$seV–çC¢66RÕDÅ—†VÄf÷&ÖE$se6–çC¢66RÕDÅ—†VÄf÷&ÖE$t$…V–çC¢66RÕDÅ—†VÄf÷&ÖE$t$…6–çC ¢66RÕDÅ—†VÄf÷&ÖE$t#%V–çC¢66RÕDÅ—†VÄf÷&ÖE$s3%V–çC¢66RÕDÅ—†VÄf÷&ÖE$s3%6–çC¢66RÕDÅ—†VÄf÷&ÖE$t$eV–çC¢66RÕDÅ—†VÄf÷&ÖE$t$e6–çC ¢66RÕDÅ—†VÄf÷&ÖE$t$3%V–çC¢66RÕDÅ—†VÄf÷&ÖE$t$3%6–çC¢66RÕDÅ—†VÄf÷&ÖDFWFƒeVæ÷&Ó¢66RÕDÅ—†VÄf÷&ÖDFWFƒ3$fÆöC¢66RÕDÅ—†VÄf÷&ÖE7FVæ6–Ãƒ ¢66RÕDÅ—†VÄf÷&ÖDFWFƒ3$fÆöEõ7FVæ6–Ãƒ¢66RÕDÅ—†VÄf÷&ÖEƒ3%õ7FVæ6–Ãƒ¢&WGW&â°¢FVfVÇC¢&WGW&â°¢Ð§Ð§7FF–2–çB6÷•÷C'EöG&r†–CÄÕDÅFW‡GW&Sâ2Â–CÄÕDÅFW‡GW&SâBÂVç6–væVB6ÂÂVç6–væVB72ÂVç6–væVBFÂÂVç6–væVBG2Â6öç7Bådæ6÷•&Vv–öâ§7"Â6öç7Bådæ6÷•&Vv–öâ¦G"Â–çBÆ–æV"Â–çBÖ—'&÷"’°¢7FF–2å4×WF&ÆTF–7F–öæ'“Äå4çVÖ&W"¢Â–CÄÕDÅ&VæFW%—VÆ–æU7FFSãâ§6ó²7FF–2–CÄÕDÄgVæ7F–öãâfbÂfbÂfBÂfG3²7FF–2–CÄÕDÅ6×ÆW%7FFSâ6×³%Ó²7FF–2–CÄÕDÃD&wVÖVçEF&ÆSâC²7FF–2–CÄÕDÄFWF…7FVæ6–Å7FFSâG72ÂG75÷3²7FF–2–çBf–ÆVC°¢–b†f–ÆVBÇÂ7"Óçv–GF‚ÃÒÇÂ7"Óæ†V–v‡BÃÒÇÂG"Óçv–GF‚ÃÒÇÂG"Óæ†V–v‡BÃÒ’&WGW&â°¢–çBFWF‚Ò‡2ç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖDFWFƒ3$fÆöBÇÂ2ç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖDFWFƒ3$fÆöEõ7FVæ6–Ã‚’bbBç—†VÄf÷&ÖBÓÒ2ç—†VÄf÷&ÖBÂ7FVæ6–ÂÒFWF‚bbBç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖDFWFƒ3$fÆöEõ7FVæ6–Ãƒ°¢–b‚FWF‚bb†f×Eö–çEö÷%öG2‡2ç—†VÄf÷&ÖB’ÇÂf×Eö–çEö÷%öG2†Bç—†VÄf÷&ÖB’’’&WGW&â°¢–b†×FÅöf×Eö&Æö6²†Bç—†VÄf÷&ÖB’ÇÂ†BçW6vRbÕDÅFW‡GW&UW6vU&VæFW%F&vWB’ÇÂ2ç6×ÆT6÷VçBâÇÂBç6×ÆT6÷VçBâ’&WGW&â°¢–b‚6ò’°¢å4W'&÷"¦RÒæ–Ã²–CÄÕDÄÆ–'&'“âÆ–"Ò†÷7EöÆ–"‚&6÷’"ÂfR“°¢fbÒ¶Æ–"æWtgVæ7F–öåv—F„æÖS¤'g2%Ó²fbÒ¶Æ–"æWtgVæ7F–öåv—F„æÖS¤&g2%Ó²fBÒ¶Æ–"æWtgVæ7F–öåv—F„æÖS¤&g6B%Ó²fG2Ò¶Æ–"æWtgVæ7F–öåv—F„æÖS¤&g6G2%Ó°¢ÕDÄFWF…7FVæ6–ÄFW67&—F÷"¦FBÒ´ÕDÄFWF…7FVæ6–ÄFW67&—F÷"æWuÓ²FBæFWF„6ö×&TgVæ7F–öâÒÕDÄ6ö×&TgVæ7F–öäÇv—3²FBæFWF…w&—FTVæ&ÆVBÒ”U3²G72Ò¶uöFWbæWtFWF…7FVæ6–Å7FFUv—F„FW67&—F÷#¦FEÓ°¢ÕDÅ7FVæ6–ÄFW67&—F÷"§6EòÒ´ÕDÅ7FVæ6–ÄFW67&—F÷"æWuÓ²6Eòç7FVæ6–Ä6ö×&TgVæ7F–öâÒÕDÄ6ö×&TgVæ7F–öäÇv—3²6EòæFWF…7FVæ6–Å74÷W&F–öâÒÕDÅ7FVæ6–Ä÷W&F–öå&WÆ6S²òò&WÆ6Rw&—FW2F†Rµ·7FVæ6–ÅÕÒ÷WGW@¢FBæg&öçDf6U7FVæ6–ÂÒFBæ&6´f6U7FVæ6–ÂÒ6Eó²G75÷2Ò¶uöFWbæWtFWF…7FVæ6–Å7FFUv—F„FW67&—F÷#¦FEÓ°¢ÕDÃD&wVÖVçEF&ÆTFW67&—F÷"¦BÒ´ÕDÃD&wVÖVçEF&ÆTFW67&—F÷"æWuÓ²BæÖ…FW‡GW&T&–æD6÷VçBÒ#²BæÖ…6×ÆW%7FFT&–æD6÷VçBÒ²BæÖ„'VffW$&–æD6÷VçBÒ²BÒ&V5÷F&ÆR…¶uöFWbæWt&wVÖVçEF&ÆUv—F„FW67&—F÷#¦BW'&÷#¢fUÒ“°¢f÷"†–çB’Ò²’Â#²’²²’²ÕDÅ6×ÆW$FW67&—F÷"§6BÒ´ÕDÅ6×ÆW$FW67&—F÷"æWuÓ²6BæÖ–äf–ÇFW"Ò6BæÖtf–ÇFW"Ò’òÕDÅ6×ÆW$Ö–äÖtf–ÇFW$Æ–æV"¢ÕDÅ6×ÆW$Ö–äÖtf–ÇFW$æV&W7C°¢6Bç4FG&W74ÖöFRÒ6BçDFG&W74ÖöFRÒÕDÅ6×ÆW$FG&W74ÖöFT6Æ×FôVFvS²6Bç7W÷'D&wVÖVçD'VffW'2Ò”U3²6×¶•ÒÒ¶uöFWbæWu6×ÆW%7FFUv—F„FW67&—F÷#§6EÓ²Ð¢–b‚fbÇÂfbÇÂfBÇÂfG2ÇÂG72ÇÂG75÷2ÇÂBÇÂ6×³ÒÇÂ6×³Ò’²f–ÆVBÒ²ÔÄôr‚%µ$TEÒ6ææ÷B7&VFRF†R—VÆ–æW2f÷"f÷&ÖBÖ6öçfW'F–ærò66Æ–ærFW‡GW&R6÷–W3¢W2"ÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“²&WGW&â²Ð¢6òÒ´å4×WF&ÆTF–7F–öæ'’F–7F–öæ'•Ó°¢Ð¢òò7FVæ6–Ã¢&VBF‡&÷Vv‚7FVæ6–ÂÖöæÇ’f–WröbF†R6÷W&6R†öæRW"Æ–W"&VÆ÷r“²v—F†÷WBöæR—B—26ÆV&VBFò ¢–CÄÕDÅFW‡GW&Sâ7GcÒ7FVæ6–Âbb2ç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖDFWFƒ3$fÆöEõ7FVæ6–Ã‚ò·2æWuFW‡GW&Uf–Wuv—F…—†VÄf÷&ÖC¤ÕDÅ—†VÄf÷&ÖEƒ3%õ7FVæ6–Ã‚FW‡GW&UG—S¤ÕDÅFW‡GW&UG—S$BÆWfVÇ3¤å4Ö¶U&ævR‡6ÂÂ’6Æ–6W3¤å4Ö¶U&ævR‡72²7"Óç¦öfg6WBÂ•Ò¢æ–Ã°¢–çB66÷’Ò7GcÒæ–Ã°¢å4çVÖ&W"§²Ò†Bç—†VÄf÷&ÖB¢"²66÷’“°¢–CÄÕDÅ&VæFW%—VÆ–æU7FFSâ2Ò6õ·µÓ°¢–b‚2’°¢ÕDÅ&VæFW%—VÆ–æTFW67&—F÷"§BÒ´ÕDÅ&VæFW%—VÆ–æTFW67&—F÷"æWuÓ²BçfW'FW„gVæ7F–öâÒfc²Bæg&vÖVçDgVæ7F–öâÒFWF‚ò‡66÷’òfG2¢fB’¢fc°¢–b†FWF‚’²BæFWF„GF6†ÖVçE—†VÄf÷&ÖBÒBç—†VÄf÷&ÖC²–b‡7FVæ6–Â’Bç7FVæ6–ÄGF6†ÖVçE—†VÄf÷&ÖBÒBç—†VÄf÷&ÖC²ÒVÇ6RBæ6öÆ÷$GF6†ÖVçG5³Òç—†VÄf÷&ÖBÒBç—†VÄf÷&ÖC°¢å4W'&÷"¦RÒæ–Ã²2Ò¶uöFWbæWu&VæFW%—VÆ–æU7FFUv—F„FW67&—F÷#§BW'&÷#¢fUÓ°¢–b‚2’²$TEôôä4Rƒƒ2Â&f÷&ÖBÖ6öçfW'F–ærò66Æ–ærFW‡GW&R6÷“¢6ææ÷B7&VFRF†R—VÆ–æRf÷"FW7F–æF–öâf÷&ÖBVÇS¢W2"Â‡Vç6–væVBÆöær–Bç—†VÄf÷&ÖBÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“²&WGW&â²Ð¢6õ·µÒÒ3°¢Ð¢Vç6–væVB7rÒÔ‚‡2çv–GF‚ãâ6ÂÂVÂ’Â6‚ÒÔ‚‡2æ†V–v‡Bãâ6ÂÂVÂ’ÂGrÒÔ‚†Bçv–GF‚ãâFÂÂVÂ’ÂF‚ÒÔ‚†Bæ†V–v‡BãâFÂÂVÂ“°¢–b‚‡Vç6–væVB’†G"Óç†öfg6WB²G"Óçv–GF‚’âGrÇÂ‡Vç6–væVB’†G"Óç–öfg6WB²G"Óæ†V–v‡B’âF‚’&WGW&â°¢–çBç¢Ò7"ÓæFWF‚ò7"ÓæFWF‚¢°¢'öVæB‚“²–b†uö&Æ—B’²Væ5ö%öVæB†uö&Æ—B“²uö&Æ—BÒæ–Ã²Ð¢f÷"†–çB¢Ò²¢Âç£²¢²²’°¢–CÄÕDÅFW‡GW&Sâ7bÒ·2æWuFW‡GW&Uf–Wuv—F…—†VÄf÷&ÖC§2ç—†VÄf÷&ÖBFW‡GW&UG—S¤ÕDÅFW‡GW&UG—S$BÆWfVÇ3¤å4Ö¶U&ævR‡6ÂÂ’6Æ–6W3¤å4Ö¶U&ævR‡72²7"Óç¦öfg6WB²¢Â•Ó°¢–CÄÕDÄ'VffW#â#²6—¦U÷BòÒ&VæöÆÆö2ƒ3"Âg"“²òò&ÖWFW'2–âF†Rg&ÖR&Væ‡&W6–FVçBÇ&VG’“¢g&W6‚3"Ö'—FR'VffW"W"6÷’v2&W6–FVæ7’FBÂ6öÖÖ—BæB&VÆV6R„Æ–æ²w2v¶Væ–ærÂ&÷WBg&ÖR¢–b‚7bÇÂ"’²$TEôôä4RƒƒBÂ&f÷&ÖBÖ6öçfW'F–ærò66Æ–ærFW‡GW&R6÷“¢6ææ÷B7&VFRF†R6÷W&6Rf–Wrò&wVÖVçB&Æö6²"“²&WGW&â¢â²Ð¢fÆöB§Ò†fÆöB¢’‚‡V–çC…÷B¢—"æ6öçFVçG2²ò“²³ÒÒ7"Óç†öfg6WC²³ÒÒ7"Óç–öfg6WC²³%ÒÒ†fÆöB—7"Óçv–GF‚òG"Óçv–GFƒ²³5ÒÒ†fÆöB—7"Óæ†V–v‡BòG"Óæ†V–v‡C°¢–b†Ö—'&÷"bådåô4õ•ôdÄu5ôÔ•%$õ%õ…ô$•B’²³Ò³Ò7"Óçv–GFƒ²³%ÒÒ×³%Ó²ÒòòÖ—'&÷&VC¢vÆ²F†R6÷W&6R&Vv–öâg&öÒ—G2f"VFvP¢–b†Ö—'&÷"bådåô4õ•ôdÄu5ôÔ•%$õ%õ•ô$•B’²³Ò³Ò7"Óæ†V–v‡C²³5ÒÒ×³5Ó²Ð¢³EÒÒG"Óç†öfg6WC²³UÒÒG"Óç–öfg6WC²³eÒÒ7s²³uÒÒ6ƒ°¢ÕDÃE&VæFW%74FW67&—F÷"§'Ò´ÕDÃE&VæFW%74FW67&—F÷"æWuÓ°¢ÕDÅ&VæFW%74GF6†ÖVçDFW67&—F÷"¦ÒFWF‚ò„ÕDÅ&VæFW%74GF6†ÖVçDFW67&—F÷"¢—'æFWF„GF6†ÖVçB¢„ÕDÅ&VæFW%74GF6†ÖVçDFW67&—F÷"¢—'æ6öÆ÷$GF6†ÖVçG5³Ó°¢–çBv†öÆRÒG"Óç†öfg6WBbbG"Óç–öfg6WBbb‡Vç6–væVB–G"Óçv–GF‚ÓÒGrbb‡Vç6–væVB–G"Óæ†V–v‡BÓÒFƒ²òòF†R6÷’G&w2WfW'’—†VÂöbF†RÆWfVÂ†æò&ÆVæF–ærÂæòF—66&B“¢—G2öÆB6öçFVçB—2æ÷BÆöFVB„Æ–æ²w2v¶Væ–ær6÷–W2—G266VæRF†—2v’WfW'’g&ÖR¢çFW‡GW&RÒC²æÆWfVÂÒFÃ²ç6Æ–6RÒG2²G"Óç¦öfg6WB²£²æÆöD7F–öâÒv†öÆRòÕDÄÆöD7F–öäFöçD6&R¢ÕDÄÆöD7F–öäÆöC²ç7F÷&T7F–öâÒÕDÅ7F÷&T7F–öå7F÷&S°¢–CÄÕDÅFW‡GW&Sâ7GbÒ66÷’ò‡¢ò·2æWuFW‡GW&Uf–Wuv—F…—†VÄf÷&ÖC¤ÕDÅ—†VÄf÷&ÖEƒ3%õ7FVæ6–Ã‚FW‡GW&UG—S¤ÕDÅFW‡GW&UG—S$BÆWfVÇ3¤å4Ö¶U&ævR‡6ÂÂ’6Æ–6W3¤å4Ö¶U&ævR‡72²7"Óç¦öfg6WB²¢Â•Ò¢7Gc’¢æ–Ã°¢–b‡7FVæ6–Â’²ÕDÅ&VæFW%757FVæ6–ÄGF6†ÖVçDFW67&—F÷"§7BÒ'ç7FVæ6–ÄGF6†ÖVçC²7BçFW‡GW&RÒC²7BæÆWfVÂÒFÃ²7Bç6Æ–6RÒç6Æ–6S²7BæÆöD7F–öâÒ66÷’ò‡v†öÆRòÕDÄÆöD7F–öäFöçD6&R¢ÕDÄÆöD7F–öäÆöB’¢ÕDÄÆöD7F–öä6ÆV#²7Bæ6ÆV%7FVæ6–ÂÒ²7Bç7F÷&T7F–öâÒÕDÅ7F÷&T7F–öå7F÷&S²Ð¢6W%övFR‚“²–CÄÕDÃE&VæFW$6öÖÖæDVæ6öFW#âVâÒVæ5÷%ö&Vv–â‡'“°¢–b†FWF‚’¶Vâ6WDFWF…7FVæ6–Å7FFS§66÷’òG75÷2¢G75Ó°¢¶Vâ6WD&wVÖVçEF&ÆS¦BE7FvW3¤ÕDÅ&VæFW%7FvTg&vÖVçEÓ°¢¶B6WEFW‡GW&S§7bæwU&W6÷W&6T”BD–æFWƒ£Ó²–b‡7Gb’¶B6WEFW‡GW&S§7GbæwU&W6÷W&6T”BD–æFWƒ£Ó²¶B6WE6×ÆW%7FFS§6×²Æ–æV"bbFWF…ÒæwU&W6÷W&6T”BD–æFWƒ£Ó²¶B6WDFG&W73§"æwTFG&W72²òD–æFWƒ£Ó°¢¶Vâ6WEf–Ww÷'C¢„ÕDÅf–Ww÷'B—¶G"Óç†öfg6WBÂG"Óç–öfg6WBÂG"Óçv–GF‚ÂG"Óæ†V–v‡BÂÂÕÓ°¢¶Vâ6WE66—76÷%&V7C¢„ÕDÅ66—76÷%&V7B—²„å5T–çFVvW"–G"Óç†öfg6WBÂ„å5T–çFVvW"–G"Óç–öfg6WBÂ„å5T–çFVvW"–G"Óçv–GF‚Â„å5T–çFVvW"–G"Óæ†V–v‡GÕÓ°¢¶Vâ6WE&VæFW%—VÆ–æU7FFS§5Ó²¶VâG&u&–Ö—F—fW3¤ÕDÅ&–Ö—F—fUG—UG&–ævÆRfW'FW…7F'C£fW'FW„6÷VçC£5Ó²Væ5÷%öVæB†Vâ“°¢FVfW%öö&¥÷Vç&Vb‡7b“²–b‡7Gb’FVfW%öö&¥÷Vç&Vb‡7Gb“²òò&VÆV6VBgFW"F†—27V&Ö—76–öâ†2'Và¢Ð¢&w5öF—'G•öÆÂ‚“²u÷6õöF—'G’ÒuöG75öF—'G’Ò°¢²7FF–2–çBöæ6U³%Ó²–b‚öæ6U¶FWF…Ò²²’ÔÄôr‚&f÷&ÖBÖ6öçfW'F–ærò66Æ–ærFW‡GW&R6÷“¢f—'7B66RVG‚VBf÷&ÖBVÇR(i"VG‚VBf÷&ÖBVÇRW2W2"Â7"Óçv–GF‚Â7"Óæ†V–v‡BÂ‡Vç6–væVBÆöær—2ç—†VÄf÷&ÖBÂG"Óçv–GF‚ÂG"Óæ†V–v‡BÂ‡Vç6–væVBÆöær–Bç—†VÄf÷&ÖBÀ¢Æ–æV"bbFWF‚ò"†&–Æ–æV"’"¢""Â7FVæ6–Âò‡66÷’ò"†FWF‚²7FVæ6–Â’"¢"†æò7FVæ6–Âf–Wr÷76–&ÆRÂ6ÆV&VBFò’"’¢""“²Ð¢&WGW&â°§Ð¢òòçfä6öÖÖæD'VffW$G&uFW‡GW&S¢F†R6÷W&6R&Vv–öâöbFW‡GW&R‡FW†VÇ2ÂF‡&÷Vv‚F†R†æFÆRw26×ÆW"’G&vâ–çFòF†RFW7F–æF–öâ&Vv–öâöbV6‚&÷VæB6öÆ÷W"F&vWB‡—†VÇ2–âF†Rv–æF÷rw0¢òò6ö÷&F–æFW2Âv†–6‚&RF†RF&vWG2r÷vâ&÷w2–âV—F†W"v–æF÷r÷&–v–âÂ6–æ6RF†R†÷7B¶VW2F†R7v—F6‚w2ÖVÖ÷'’Æ–÷WB’â7G&VWG2öb&vRBG&w2V6‚Ö÷f–Rg&ÖR‡F†R–çG&òÂF†P¢òòF—FÆRw2fÆÖW2’–çFò—G2“#ƒƒ66VæRF&vWBF†—2v’Âg&öÒ6V6öæBFW‡GW&RööÃ²v—F†÷WB—BF†R67&VVâ7F–VB&Æ6²âF†R†æFÆRw2”G2&RF†R†÷7BF&ÆRw2†Ö÷fVBFòF†P¢òò&÷VæBööÇ2v†Vâ&V6÷&FVBÂ†ÆUöçfâæ2’â2FW‡GW&R6÷’Â—G2÷vâ&VæFW"72W"F&vWBgFW"F†R72–â&öw&W72‡F†RæÇ—6W2G&VB—B2öæR’À¢òòv—F‚F†R66—76÷"æBF†R6†ææVÂÖ6³²fÆöB÷"Væ÷&ÒF&vWBöæÇ’ÂæB&ÆVæF–ær—2æ÷BÆ–VB†&VBÆ–æRv†Vâ—B—2öâ§7FF–2fö–BG&u÷FW‚†6öç7BöG&u÷FW‚¦2’°¢7FF–2å4×WF&ÆTF–7F–öæ'“Äå4çVÖ&W"¢Â–CÄÕDÅ&VæFW%—VÆ–æU7FFSãâ§6ó²7FF–2–CÄÕDÄgVæ7F–öãâfbÂfc²7FF–2–CÄÕDÃD&wVÖVçEF&ÆSâC²7FF–2–çBf–ÆVC°¢–b†f–ÆVBÇÂu÷&5÷6¶—ÇÂ"æ†5÷'B’&WGW&ã°¢fÆöBGrÒ2ÓæBçƒÒ2ÓæBçƒÂF‚Ò2ÓæBç“Ò2ÓæBç“°¢–b†GrÓÒÇÂF‚ÓÒ’&WGW&ã°¢–b‚6ò’°¢å4W'&÷"¦RÒæ–Ã²–CÄÕDÄÆ–'&'“âÆ–"Ò†÷7EöÆ–"‚&G&u÷FW‚"ÂfR“°¢fbÒ¶Æ–"æWtgVæ7F–öåv—F„æÖS¤'g2%Ó²fbÒ¶Æ–"æWtgVæ7F–öåv—F„æÖS¤&g2%Ó°¢ÕDÃD&wVÖVçEF&ÆTFW67&—F÷"¦BÒ´ÕDÃD&wVÖVçEF&ÆTFW67&—F÷"æWuÓ²BæÖ…FW‡GW&T&–æD6÷VçBÒ²BæÖ…6×ÆW%7FFT&–æD6÷VçBÒ²BæÖ„'VffW$&–æD6÷VçBÒ²BÒ&V5÷F&ÆR…¶uöFWbæWt&wVÖVçEF&ÆUv—F„FW67&—F÷#¦BW'&÷#¢fUÒ“°¢–b‚fbÇÂfbÇÂB’²f–ÆVBÒ²ÔÄôr‚%µ$TEÒ6ææ÷B7&VFRF†R—VÆ–æRf÷"çfä6öÖÖæD'VffW$G&uFW‡GW&S¢W2"ÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“²&WGW&ã²Ð¢6òÒ´å4×WF&ÆTF–7F–öæ'’F–7F–öæ'•Ó°¢Ð¢÷5÷Væf—%öÆö6µöÆö6²‚fu÷&×R“°¢–CÄÕDÅFW‡GW&SâBÒFW…ö'•ö–B‚†–çB’†2Óæ‚b†ffffb’“²–CÄÕDÅ6×ÆW%7FFSâ72Ò6×ö'•ö–B‚†–çB’†2Óæ‚ãâ#’“°¢÷5÷Væf—%öÆö6µ÷VæÆö6²‚fu÷&×R“°¢–b‚BÇÂ72’²$TEôôä4Rƒ#"Â$G&uFW‡GW&S¢FW‡GW&R†æFÆRR6ÆÇ‚†2æòFW‡GW&R÷"6×ÆW""Â‡Vç6–væVBÆöærÆöær–2Óæ‚“²&WGW&ã²Ð¢–b‡BçFW‡GW&UG—RÒÕDÅFW‡GW&UG—S$BÇÂf×Eö–çEö÷%öG2‡Bç—†VÄf÷&ÖB’’²$TEôôä4Rƒ#2Â$G&uFW‡GW&S¢6÷W&6RG—RVÇRf÷&ÖBVÇRæ÷BG&vâƒ$BfÆöBòVæ÷&ÒöæÇ’’"Â‡Vç6–væVBÆöær—BçFW‡GW&UG—RÂ‡Vç6–væVBÆöær—Bç—†VÄf÷&ÖB“²&WGW&ã²Ð¢–b…2æ6öÆ÷"æ&ÆVæEöVâ’$TEôôä4Rƒ#BÂ$G&uFW‡GW&Rv—F‚&ÆVæF–æröâ‡F&vWG2R7‚“¢G&vâv—F†÷WB—B"Â‡Vç6–væVB•2æ6öÆ÷"æ&ÆVæEöVâ“°¢'öfÇW6‚‚“²'öVæB‚“²–b†uö&Æ—B’²Væ5ö%öVæB†uö&Æ—B“²uö&Æ—BÒæ–Ã²ÒòòVæF–ær6ÆV'2öbF†RF&vWG2ÆæBf—'7BÂ2f÷"6÷•÷C'@¢f÷"†–çB²Ò²²Â"ç'Bæâbb²Âƒ²²²²’°¢6öç7BFW…÷B§'BÒ"ç'Bæ6öÆ÷%¶µÓ²–b‚'B’6öçF–çVS°¢V–çC3%÷BvÒÒ…2æ6†Ö6²æÖ6²ãâƒB¢²’’b†c²–b‚vÒ’6öçF–çVS°¢–CÄÕDÅFW‡GW&SâBÒFW…ö×FÂ‡'B“²V–çCcE÷B6W#²–çBÇbÂ6Ã²'E÷7V"‚e"ç'BÂ²Âg6W"ÂfÇbÂg6Â“°¢–b‚BÇÂf×Eö–çEö÷%öG2†Bç—†VÄf÷&ÖB’ÇÂ×FÅöf×Eö&Æö6²†Bç—†VÄf÷&ÖB’ÇÂ†BçW6vRbÕDÅFW‡GW&UW6vU&VæFW%F&vWB’’²$TEôôä4Rƒ#RÂ$G&uFW‡GW&S¢F&vWBVB†f÷&ÖBVÇR’æ÷BG&vâ"Â²ÂBò‡Vç6–væVBÆöær–Bç—†VÄf÷&ÖB¢VÂ“²6öçF–çVS²Ð¢å4çVÖ&W"§²Ò†Bç—†VÄf÷&ÖB¢b²vÒ“°¢–CÄÕDÅ&VæFW%—VÆ–æU7FFSâ2Ò6õ·µÓ°¢–b‚2’°¢ÕDÅ&VæFW%—VÆ–æTFW67&—F÷"§BÒ´ÕDÅ&VæFW%—VÆ–æTFW67&—F÷"æWuÓ²BçfW'FW„gVæ7F–öâÒfc²Bæg&vÖVçDgVæ7F–öâÒfc²Bæ6öÆ÷$GF6†ÖVçG5³Òç—†VÄf÷&ÖBÒBç—†VÄf÷&ÖC°¢Bæ6öÆ÷$GF6†ÖVçG5³Òçw&—FTÖ6²Ò‡vÒbòÕDÄ6öÆ÷%w&—FTÖ6µ&VB¢’Â‡vÒb"òÕDÄ6öÆ÷%w&—FTÖ6´w&VVâ¢’Â‡vÒbBòÕDÄ6öÆ÷%w&—FTÖ6´&ÇVR¢’Â‡vÒb‚òÕDÄ6öÆ÷%w&—FTÖ6´Ç†¢“°¢å4W'&÷"¦RÒæ–Ã²2Ò¶uöFWbæWu&VæFW%—VÆ–æU7FFUv—F„FW67&—F÷#§BW'&÷#¢fUÓ°¢–b‚2’²$TEôôä4Rƒ#bÂ$G&uFW‡GW&S¢6ææ÷B7&VFRF†R—VÆ–æRf÷"F&vWBf÷&ÖBVÇS¢W2"Â‡Vç6–væVBÆöær–Bç—†VÄf÷&ÖBÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“²6öçF–çVS²Ð¢6õ·µÒÒ3°¢Ð¢å5T–çFVvW"GrÒÔ‚†Bçv–GF‚ãâÇbÂVÂ’ÂF‚ÒÔ‚†Bæ†V–v‡BãâÇbÂVÂ“°¢òòF†RFW7F–æF–öâ&V7FævÆR†V—F†W"6÷&æW"÷&FW"’v—F†–âF†RF&vWBæBF†R66—76÷ ¢F÷V&ÆRƒÒÔ”â†2ÓæBçƒÂ2ÓæBçƒ’ÂƒÒÔ‚†2ÓæBçƒÂ2ÓæBçƒ’Â“ÒÔ”â†2ÓæBç“Â2ÓæBç“’Â“ÒÔ‚†2ÓæBç“Â2ÓæBç““°¢–b…"æ†5÷62’²ƒÒÔ‚‡ƒÂ†F÷V&ÆR•"ç62ç‚“²“ÒÔ‚‡“Â†F÷V&ÆR•"ç62ç’“²ƒÒÔ”â‡ƒÂ†F÷V&ÆR•"ç62ç‚²"ç62çr“²“ÒÔ”â‡“Â†F÷V&ÆR•"ç62ç’²"ç62æ‚“²Ð¢ƒÒÔ‚‡ƒÂã“²“ÒÔ‚‡“Âã“²ƒÒÔ”â‡ƒÂ†F÷V&ÆR—Gr“²“ÒÔ”â‡“Â†F÷V&ÆR—F‚“°¢å5T–çFVvW"7‚Ò„å5T–çFVvW"–6V–Â‡ƒÒãR’Â7’Ò„å5T–çFVvW"–6V–Â‡“ÒãR’ÂW‚Ò„å5T–çFVvW"–6V–Â‡ƒÒãR’ÂW’Ò„å5T–çFVvW"–6V–Â‡“ÒãR“²òò—†VÇ2v†÷6R6VçG&W2&R–ç6–FP¢–b†W‚ÃÒ7‚ÇÂW’ÃÒ7’’6öçF–çVS°¢–CÄÕDÄ'VffW#â#²6—¦U÷BòÒ&VæöÆÆö2ƒ3"Âg"“°¢–b‚"’²$TEôôä4Rƒ#rÂ$G&uFW‡GW&S¢æò&wVÖVçB&Æö6²"“²&WGW&ã²Ð¢fÆöB§Ò†fÆöB¢’‚‡V–çC…÷B¢—"æ6öçFVçG2²ò“°¢³ÒÒ2ÓæBçƒ²³ÒÒ2ÓæBç“²³%ÒÒ†2Óç2çƒÒ2Óç2çƒ’òGs²³5ÒÒ†2Óç2ç“Ò2Óç2ç“’òFƒ°¢³EÒÒ2Óç2çƒ²³UÒÒ2Óç2ç“²³eÒÒ†fÆöB—Bçv–GFƒ²³uÒÒ†fÆöB—Bæ†V–v‡C°¢ÕDÃE&VæFW%74FW67&—F÷"§'Ò´ÕDÃE&VæFW%74FW67&—F÷"æWuÓ²ÕDÅ&VæFW%746öÆ÷$GF6†ÖVçDFW67&—F÷"¦Ò'æ6öÆ÷$GF6†ÖVçG5³Ó°¢çFW‡GW&RÒC²æÆWfVÂÒÇc²ç6Æ–6RÒ6Ã²æÆöD7F–öâÒÕDÄÆöD7F–öäÆöC²ç7F÷&T7F–öâÒÕDÅ7F÷&T7F–öå7F÷&S°¢6W%övFR‚“²–CÄÕDÃE&VæFW$6öÖÖæDVæ6öFW#âVâÒVæ5÷%ö&Vv–â‡'“°¢¶Vâ6WD&wVÖVçEF&ÆS¦BE7FvW3¤ÕDÅ&VæFW%7FvTg&vÖVçEÓ°¢¶B6WEFW‡GW&S§BæwU&W6÷W&6T”BD–æFWƒ£Ó²¶B6WE6×ÆW%7FFS§72æwU&W6÷W&6T”BD–æFWƒ£Ó²¶B6WDFG&W73§"æwTFG&W72²òD–æFWƒ£Ó°¢¶Vâ6WEf–Ww÷'C¢„ÕDÅf–Ww÷'B—³ÂÂGrÂF‚ÂÂÕÓ°¢¶Vâ6WE66—76÷%&V7C¢„ÕDÅ66—76÷%&V7B—·7‚Â7’ÂW‚Ò7‚ÂW’Ò7—ÕÓ°¢¶Vâ6WE&VæFW%—VÆ–æU7FFS§5Ó²¶VâG&u&–Ö—F—fW3¤ÕDÅ&–Ö—F—fUG—UG&–ævÆRfW'FW…7F'C£fW'FW„6÷VçC£5Ó²Væ5÷%öVæB†Vâ“°¢§6¶—öFVÂ‡'BÓç6W&–Â“°¢B‚%BVÆÇRG&rFW‡GW&R–BVB(i"F&vWBVB6W#ÒVÆÇR&Vv–öâVrÂVrââVrÂVrg&öÒVrÂVrââVrÂVr"Âu÷7FE÷72Â†–çB’†2Óæ‚b†ffffb’Â²Â‡Vç6–væVBÆöærÆöær—'BÓç6W&–ÂÂ2ÓæBçƒÂ2ÓæBç“Â2ÓæBçƒÂ2ÓæBç“Â2Óç2çƒÂ2Óç2ç“Â2Óç2çƒÂ2Óç2ç““°¢²7FF–2–çBöæ6S²–b‚öæ6R²²’ÔÄôr‚$G&uFW‡GW&S¢f—'7B66RVw‚VrFW†VÇ2öbVÇW‚VÇRFW‡GW&R†f÷&ÖBVÇR’(i"Vw‚Vr—†VÇ2öbF&vWBVB‚VÇW‚VÇRÂf÷&ÖBVÇR’"Â2Óç2çƒÒ2Óç2çƒÂ2Óç2ç“Ò2Óç2ç“À¢‡Vç6–væVBÆöær—Bçv–GF‚Â‡Vç6–væVBÆöær—Bæ†V–v‡BÂ‡Vç6–væVBÆöær—Bç—†VÄf÷&ÖBÂGrÂF‚Â²Â‡Vç6–væVBÆöær—GrÂ‡Vç6–væVBÆöær—F‚Â‡Vç6–væVBÆöær–Bç—†VÄf÷&ÖB“²Ð¢Ð¢&w5öF—'G•öÆÂ‚“²u÷6õöF—'G’ÒuöG75öF—'G’Ò°§Ð¢òòuS¢FW‡GW&RÓâFW‡GW&R‡&Wf–÷W6Ç’Ö—6Æ—7FVB2æòÖ÷Â2F–ÖW2W"g&ÖRöâF†R6†&7FW"6VÆV7BvR’â6ÖR6—¦RÂæ÷BÖ—'&÷&VBÂ6ÖRf÷&ÖBÓâ&Æ—BÆ–W"'’Æ–W#²F†R÷F†W"6VÖçF–72‡66Æ–ærv—F‚f–ÇFW&–ærÂÖ—'&÷&–ærÂf÷&ÖB6†ævR’6ææ÷B&RW‡&W76VB'’&Æ—C¢&W÷'Bf–ÇW&RÂFò&R–×ÆVÖVçFV@§7FF–2fö–B6÷•÷C'B†6öç7Bö6÷•÷C'B¦2’°¢–b†2Óç7&2’‡eö6†V6²†2Óç7&2Óç6W&–ÂÂ2Óæ†5÷7bbb2Óç7bç6WEöÆWfVÇ2ò2Óç7bæ&6UöÆWfVÂ¢Â2Óæ†5÷7bbb2Óç7bç6WEöÆ–W'2ò2Óç7bæÖ–åöÆ–W"¢Â'FW‡GW&R6÷’6÷W&6R"“°¢'öfÇW6‚‚“²òòÆæBF†RVæF–ærW&RÖ6ÆV"&VæFW"72f—'7B‡¦W&ò×6÷W&6R6æF–FFW2öæÇ’VçFW"F†R6WBv†VâF†R&VæFW"72VæG2“²F†R&Æ—B‚’&VÆ÷rv÷VÆBVæBF†R&VæFW"72ç—v¢–b†2ÓæG7B’§6¶—öFVÂ†2ÓæG7BÓç6W&–Â“°¢–b†2Óç7&2bb2ÓæG7Bbb2Óç7"ç†öfg6WBÓÒbb2Óç7"ç–öfg6WBÓÒbb2ÓæG"ç†öfg6WBÓÒbb2ÓæG"ç–öfg6WBÓÒ ¢bb2Óç7"çv–GF‚ÓÒ†–çB–2Óç7&2ÓæBçv–GF‚bb2Óç7"æ†V–v‡BÓÒ†–çB–2Óç7&2ÓæBæ†V–v‡Bbb2ÓæG"çv–GF‚ÓÒ†–çB–2ÓæG7BÓæBçv–GF‚bb2ÓæG"æ†V–v‡BÓÒ†–çB–2ÓæG7BÓæBæ†V–v‡B’°¢–çB6ÂÒ2Óæ†5÷7bbb2Óç7bç6WEöÆWfVÇ2ò2Óç7bæ&6UöÆWfVÂ¢ÂFÂÒ2Óæ†5öGbbb2ÓæGbç6WEöÆWfVÇ2ò2ÓæGbæ&6UöÆWfVÂ¢°¢–çB72Ò2Óæ†5÷7bbb2Óç7bç6WEöÆ–W'2ò2Óç7bæÖ–åöÆ–W"¢ÂG2Ò2Óæ†5öGbbb2ÓæGbç6WEöÆ–W'2ò2ÓæGbæÖ–åöÆ–W"¢°¢–b‡§6¶—ö†2†2Óç7&2Óç6W&–ÂÂ6ÂÂ72’’§6¶—öFB†2ÓæG7BÓç6W&–ÂÂFÂÂG2“°¢Ð¢6ögEö†&FVâ†2Óç7&2“°¢–CÄÕDÅFW‡GW&Sâ2ÒFW…ö×FÂ†2Óç7&2’ÂBÒFW…ö×FÂ†2ÓæG7B“°¢B‚%B7&3ÒVÆÇRG7CÒVÆÇRVG‚VG‚VN(i"VG‚VG‚VBfÆw3ÒR7‚"Â2Óç7&2Óç6W&–ÂÂ2ÓæG7BÓç6W&–ÂÂ2Óç7"çv–GF‚Â2Óç7"æ†V–v‡BÂ2Óç7"æFWF‚Â2ÓæG"çv–GF‚Â2ÓæG"æ†V–v‡BÂ2ÓæG"æFWF‚Â2ÓæfÆw2“°¢–b‚2ÇÂB’&WGW&ã°¢Vç6–væVB6ÂÒ2Óæ†5÷7bbb2Óç7bç6WEöÆWfVÇ2ò2Óç7bæ&6UöÆWfVÂ¢ÂFÂÒ2Óæ†5öGbbb2ÓæGbç6WEöÆWfVÇ2ò2ÓæGbæ&6UöÆWfVÂ¢°¢Vç6–væVB7£Ò2Óæ†5÷7bbb2Óç7bç6WEöÆ–W'2ò2Óç7bæÖ–åöÆ–W"¢ÂG£Ò2Óæ†5öGbbb2ÓæGbç6WEöÆ–W'2ò2ÓæGbæÖ–åöÆ–W"¢°¢–b‚FW…÷6ögEöÆWfVÂ†2Óç7&2Âg6Â’ÇÂFW…÷6ögEöÆWfVÂ†2ÓæG7BÂfFÂ’’²$TEôôä4RƒCrÂ'FW‡GW&R6÷’v—F‚ÆWfVÂöb6ögBFW‡GW&S¢6¶—VBÂ—B—2æ÷B¶WB†f—'7B66R7&26W#ÒVÆÇRG7B6W#ÒVÆÇR’"Â‡Vç6–væVBÆöærÆöær–2Óç7&2Óç6W&–ÂÂ‡Vç6–væVBÆöærÆöær–2ÓæG7BÓç6W&–Â“²&WGW&ã²Ð¢–b‚FW…öÆWfVÅöö²‡2Â6ÂÂ'FW‡GW&R6÷’6÷W&6R"’ÇÂFW…öÆWfVÅöö²†BÂFÂÂ'FW‡GW&R6÷’FW7F–æF–öâ"’’&WGW&ã°¢–b†2ÓæfÆw2bDÕô4õ•õ$U4ôÅdR’²òòF÷vç6×ÆS¢×VÇF—6×ÆR&W6öÇfR–çFòF†R6–ævÆR×6×ÆRFW7F–æF–öâ†FWF‚F¶W26×ÆR“²v—F‚WVÂ6×ÆR6÷VçG2—B—2Æ–âv†öÆR6÷¢–b‡2ç6×ÆT6÷VçBÓÒBç6×ÆT6÷VçBbb2ç—†VÄf÷&ÖBÓÒBç—†VÄf÷&ÖBbb2çv–GF‚ÓÒBçv–GF‚bb2æ†V–v‡BÓÒBæ†V–v‡B’°¢Væ5ö%ö6÷•÷FW‡GW&R†&Æ—B‚’Â2Â7£Â6ÂÂÕDÄ÷&–v–äÖ¶RƒÂÂ’ÂÕDÅ6—¦TÖ¶R‡2çv–GF‚Â2æ†V–v‡BÂ’ÂBÂG£ÂFÂÂÕDÄ÷&–v–äÖ¶RƒÂÂ’“°¢&WGW&ã°¢Ð¢òò6–ævÆR×6×ÆR6÷W&6S¢f–ÇFW&VB66ÆRöbF†Rv†öÆRÆWfVÂ–çFòF†RFW7F–æF–öâÂ6öçfW'F–ærF†Rf÷&ÖB„&–öæWGF"Â4D²2ãc¢3#ƒsb(i"ƒƒC‚$t$‚WfW'’g&ÖRÂV'FW"ö`¢òò—G2#ƒƒs#g&ÖRæBV'FW"öbF†BÂ&÷VæFVBWFòc²æB—G2#ƒƒs#$u$‚g&ÖR–çFòâ$t$‚FW‡GW&RöbF†R6ÖR6—¦R’ÂG&vâ266Æ–ær6÷’v—F‚&–Æ–æV"f–ÇFW&–æp¢–b‡2ç6×ÆT6÷VçBÓÒbbBç6×ÆT6÷VçBÓÒbb2Óç7&2ÓæBçF&vWBÒ"bb2ÓæG7BÓæBçF&vWBÒ"’°¢–b‚†BçW6vRbÕDÅFW‡GW&UW6vU&VæFW%F&vWB’bb×FÅöf×Eö&Æö6²†Bç—†VÄf÷&ÖB’bbf×Eö–çEö÷%öG2†Bç—†VÄf÷&ÖB’’BÒFW…÷Ww&FR†2ÓæG7BÂBÂÕDÅFW‡GW&UW6vU&VæFW%F&vWBÂBç—†VÄf÷&ÖB“°¢ådæ6÷•&Vv–öâ7"Ò³ÂÂÂ†–çB”Ô‚‡2çv–GF‚ãâ6ÂÂR’Â†–çB”Ô‚‡2æ†V–v‡Bãâ6ÂÂR’ÂÒÂG"Ò³ÂÂÂ†–çB”Ô‚†Bçv–GF‚ãâFÂÂR’Â†–çB”Ô‚†Bæ†V–v‡BãâFÂÂR’ÂÓ°¢–b†6÷•÷C'EöG&r‡2ÂBÂ6ÂÂ7£ÂFÂÂG£Âg7"ÂfG"Âf×Eö–çEö÷%öG2‡2ç—†VÄf÷&ÖB’Â’’&WGW&ã°¢Ð¢–b‡2ç6×ÆT6÷VçBÃÒÇÂBç6×ÆT6÷VçBÒÇÂ2ç—†VÄf÷&ÖBÒBç—†VÄf÷&ÖBÇÂ2çv–GF‚ÒBçv–GF‚ÇÂ2æ†V–v‡BÒBæ†V–v‡B’°¢$TEôôä4Rƒs"Â$F÷vç6×ÆR66Ræ÷B–×ÆVÖVçFVC¢VÇW‚VÇRVÇR6×ÆW2f÷&ÖBVÇR(i"VÇW‚VÇRVÇR6×ÆW2f÷&ÖBVÇR"Â‡Vç6–væVBÆöær—2çv–GF‚Â‡Vç6–væVBÆöær—2æ†V–v‡BÂ‡Vç6–væVBÆöær—2ç6×ÆT6÷VçBÀ¢‡Vç6–væVBÆöær—2ç—†VÄf÷&ÖBÂ‡Vç6–væVBÆöær–Bçv–GF‚Â‡Vç6–væVBÆöær–Bæ†V–v‡BÂ‡Vç6–væVBÆöær–Bç6×ÆT6÷VçBÂ‡Vç6–væVBÆöær–Bç—†VÄf÷&ÖB“°¢&WGW&ã°¢Ð¢'öVæB‚“²–b†uö&Æ—B’²Væ5ö%öVæB†uö&Æ—B“²uö&Æ—BÒæ–Ã²Ð¢ÕDÃE&VæFW%74FW67&—F÷"§'Ò´ÕDÃE&VæFW%74FW67&—F÷"æWuÓ°¢ÕDÅ—†VÄf÷&ÖBbÒ2ç—†VÄf÷&ÖC°¢–çBFWF‚ÒbÓÒÕDÅ—†VÄf÷&ÖDFWFƒ3$fÆöBÇÂbÓÒÕDÅ—†VÄf÷&ÖDFWFƒeVæ÷&ÒÇÂbÓÒÕDÅ—†VÄf÷&ÖDFWFƒ3$fÆöEõ7FVæ6–Ãƒ°¢ÕDÅ&VæFW%74GF6†ÖVçDFW67&—F÷"¦ÒFWF‚ò'æFWF„GF6†ÖVçB¢'æ6öÆ÷$GF6†ÖVçG5³Ó°¢çFW‡GW&RÒ3²æÆWfVÂÒ6Ã²ç6Æ–6RÒ7£²ç&W6öÇfUFW‡GW&RÒC²ç&W6öÇfTÆWfVÂÒFÃ²ç&W6öÇfU6Æ–6RÒG£°¢æÆöD7F–öâÒÕDÄÆöD7F–öäÆöC²ç7F÷&T7F–öâÒÕDÅ7F÷&T7F–öå7F÷&TæD×VÇF—6×ÆU&W6öÇfS°¢–b†FWF‚’'æFWF„GF6†ÖVçBæFWF…&W6öÇfTf–ÇFW"ÒÕDÄ×VÇF—6×ÆTFWF…&W6öÇfTf–ÇFW%6×ÆS°¢6W%övFR‚“²Væ5÷%öVæB†Væ5÷%ö&Vv–â‡'’“°¢&WGW&ã°¢Ð¢B‚'B7&3ÒVÆÇRÇbWR6ÂWR‡—£ÒVBÂVBÂVBG7CÒVÆÇRÇbWR6ÂWR‡—£ÒVBÂVBÂVB6—¦SÒVBÂVBÂVB"Â2Óç7&2Óç6W&–ÂÂ6ÂÂ7£Â2Óç7"ç†öfg6WBÂ2Óç7"ç–öfg6WBÂ2Óç7"ç¦öfg6WBÀ¢2ÓæG7BÓç6W&–ÂÂFÂÂG£Â2ÓæG"ç†öfg6WBÂ2ÓæG"ç–öfg6WBÂ2ÓæG"ç¦öfg6WBÂ2Óç7"çv–GF‚Â2Óç7"æ†V–v‡BÂ2Óç7"æFWF‚“°¢–çB32Ò2Óç7&2ÓæBçF&vWBÓÒ"ÂC2Ò2ÓæG7BÓæBçF&vWBÓÒ"Âç¢Ò2Óç7"æFWF‚ò2Óç7"æFWF‚¢°¢òòF†R6÷’w2'—FW2&VÆöær–âF†RFW7F–æF–öâw27F÷&vS¢w&—GFVâF†W&Ræ÷rg&öÒF†R6÷W&6Rw27F÷&vRv†VâF†B†öÆG2—G26öçFVçBæBF†R6÷’—2Æ–âöæR‡F†R6ÖRÖWFÂf÷&ÖC¢¢òòf÷&ÖB6†ævR—2G&vâ&VÆ÷ræB6öçfW'G2F†R'—FW2’ÂVÇ6R&VB&6²gFW"F†R6÷’†FV6–FVB&Vf÷&RF†RG&rF‚v—fW2F†RFW7F–æF–öâ&VæFW"×F&vWBW6RÂv†–6‚Ö¶W2—BuRÖ÷væVBg&öÒF†Vâöâ¢–çB&"ÒC2bbFW…÷7F÷&vUöG7B†2ÓæG7B“°¢–b‡&"bb32bb2Óç7"çv–GF‚ÓÒ2ÓæG"çv–GF‚bb2Óç7"æ†V–v‡BÓÒ2ÓæG"æ†V–v‡Bbb†2ÓæfÆw2b„ådåô4õ•ôdÄu5ôÔ•%$õ%õ…ô$•BÂådåô4õ•ôdÄu5ôÔ•%$õ%õ•ô$•B’’bb2ç—†VÄf÷&ÖBÓÒBç—†VÄf÷&Ö@¢bb×&%÷VæF–ær†2ÓæG7BÓç6W&–Â’bbFW…÷7F÷&vU÷7&2†2Óç7&2’bbçfå÷FW…öÖ—'&÷%÷C'B†2Óç7&2Â6ÂÂ7£Âf2Óç7"Â2ÓæG7BÂFÂÂG£Âf2ÓæG"’’²Ö—'&÷%öæ÷FRƒÂ2ÓæG7BÂFÂÂ'FW‡GW&R"Â“²&"Ò²Ð¢òòF†RG&rF‚&VæFW'2–çFòF†RFW7F–æF–öã¢öæRÖFRv—F†÷WB&VæFW"×F&vWBW6R‡6¶vVB’vWG2—Bf—'7BÂ2v†öÆRÖÆWfVÂ6ÆV"FöW2…W"Ö&–ó¢F†RF†÷W6æBÕ–V"Fö÷"6÷–W2¢òò#‡ƒ#‚$t$‚&VæFW"F&vWB–çFòâ$s‚FW‡GW&RÂÖ—'&÷&VB–â’¢–b‚†2Óç7"çv–GF‚Ò2ÓæG"çv–GF‚ÇÂ2Óç7"æ†V–v‡BÒ2ÓæG"æ†V–v‡BÇÂ2ç—†VÄf÷&ÖBÒBç—†VÄf÷&ÖBÇÂ†2ÓæfÆw2b„ådåô4õ•ôdÄu5ôÔ•%$õ%õ…ô$•BÂådåô4õ•ôdÄu5ôÔ•%$õ%õ•ô$•B’’’bb32bbC0¢bb†BçW6vRbÕDÅFW‡GW&UW6vU&VæFW%F&vWB’bb×FÅöf×Eö&Æö6²†Bç—†VÄf÷&ÖB’bbBç6×ÆT6÷VçBÓÒbbf×Eö–çEö÷%öG2‡2ç—†VÄf÷&ÖB’bbf×Eö–çEö÷%öG2†Bç—†VÄf÷&ÖB’¢BÒFW…÷Ww&FR†2ÓæG7BÂBÂÕDÅFW‡GW&UW6vU&VæFW%F&vWBÂBç—†VÄf÷&ÖB“°¢–b‚†2Óç7"çv–GF‚Ò2ÓæG"çv–GF‚ÇÂ2Óç7"æ†V–v‡BÒ2ÓæG"æ†V–v‡BÇÂ2ç—†VÄf÷&ÖBÒBç—†VÄf÷&ÖBÇÂ†2ÓæfÆw2b„ådåô4õ•ôdÄu5ôÔ•%$õ%õ…ô$•BÂådåô4õ•ôdÄu5ôÔ•%$õ%õ•ô$•B’’’bb32bbC0¢bb†2Óç7"æFWF‚ò2Óç7"æFWF‚¢’ÓÒ†2ÓæG"æFWF‚ò2ÓæG"æFWF‚¢¢bb†2ÓæfÆw2bâ„ådåô4õ•ôdÄu5ôTät”äUó$Eô$•BÂådåô4õ•ôdÄu5ôÄ”äT%ôd”ÅDU%ô$•BÂådåô4õ•ôdÄu5ôÔ•%$õ%õ…ô$•BÂådåô4õ•ôdÄu5ôÔ•%$õ%õ•ô$•B’¢bb6÷•÷C'EöG&r‡2ÂBÂ6ÂÂ7£ÂFÂÂG£Âf2Óç7"Âf2ÓæG"Â†2ÓæfÆw2bådåô4õ•ôdÄu5ôÄ”äT%ôd”ÅDU%ô$•B’Â2ÓæfÆw2’’²òòf÷&ÖB6†ævRò66Æ–æròÖ—'&÷&–æs¢G&r—B‡6VR6÷•÷C'EöG&r¢–b‡&"’Ö—'&÷%÷&VF&6²†2ÓæG7BÂBÂFÂÂG£Âf2ÓæG"Â'FW‡GW&R"Â“°¢&WGW&ã°¢Ð¢òòF†RFWF‚ÆæRöbFWF‚×7FVæ6–ÂFW‡GW&R–çFòFWF‚ÖöæÇ’öæRÂ÷"&6²„T5õ%E2d2#rw2g&÷7F&—FR6÷–W2—G2“#ƒƒFWFƒ3$fÆöEõ7FVæ6–Ã‚F&vWB–çFòFWFƒ3$fÆö@¢òòFW‡GW&R“¢&Æ—B6÷–W2öæÇ’&WGvVVâWVÂf÷&ÖG2Â6òF†R3"Ö&—BfÆöBFWF‚vöW2F‡&÷Vv‚'VffW"ÂF¶Vâg&öÒæBw&—GFVâFòF†R6öÖ&–æVBFW‡GW&Rw2FWF‚ÆæP¢–b‚32bbC2bb2Óç7"çv–GF‚ÓÒ2ÓæG"çv–GF‚bb2Óç7"æ†V–v‡BÓÒ2ÓæG"æ†V–v‡Bbb†2Óç7"æFWF‚ò2Óç7"æFWF‚¢’ÓÒbb†2ÓæG"æFWF‚ò2ÓæG"æFWF‚¢’ÓÒ¢bb†2ÓæfÆw2bäådåô4õ•ôdÄu5ôTät”äUó$Eô$•Bbäådåô4õ•ôdÄu5ôÄ”äT%ôd”ÅDU%ô$•B’bb2ç6×ÆT6÷VçBÓÒbbBç6×ÆT6÷VçBÓÒ¢bb‚‡2ç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖDFWFƒ3$fÆöEõ7FVæ6–Ã‚bbBç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖDFWFƒ3$fÆöB’ÇÂ‡2ç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖDFWFƒ3$fÆöBbbBç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖDFWFƒ3$fÆöEõ7FVæ6–Ã‚’’’°¢7FF–2–CÄÕDÄ'VffW#âÆæS²òò¶WBg&öÒ6÷’Fò6÷’Â&WÆ6VB†FVfW'&VB’'’Æ&vW"öæP¢å5T–çFVvW"&÷rÒ„å5T–çFVvW"–2Óç7"çv–GF‚¢BÂ–ÖrÒ&÷r¢„å5T–çFVvW"–2Óç7"æ†V–v‡C°¢–b‚ÆæRÇÂÆæRæÆVæwF‚Â–Ör’°¢–b‡ÆæR’FVfW%öö&¥÷&VÆV6R‡ÆæR“°¢ÆæRÒ¶uöFWbæWt'VffW%v—F„ÆVæwFƒ¦–Ör÷F–öç3¤ÕDÅ&W6÷W&6U7F÷&vTÖöFU&—fFUÓ°¢–b‚ÆæR’²$TEôôä4Rƒ3"Â&FWF‚ÆæR6÷“¢6ææ÷B7&VFRR6Ç‚Ö'—FR7Fv–ær'VffW""Â‡Vç6–væVBÆöær––Ör“²&WGW&ã²Ð¢&W5öFB‡ÆæR“°¢Ð¢ÕDÄ&Æ—D÷F–öâ6òÒ2ç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖDFWFƒ3$fÆöEõ7FVæ6–Ã‚òÕDÄ&Æ—D÷F–öäFWF„g&öÔFWF…7FVæ6–Â¢ÕDÄ&Æ—D÷F–öäæöæRÂF÷ÒBç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖDFWFƒ3$fÆöEõ7FVæ6–Ã‚òÕDÄ&Æ—D÷F–öäFWF„g&öÔFWF…7FVæ6–Â¢ÕDÄ&Æ—D÷F–öäæöæS°¢Væ5ö%ö6÷•÷FW‡GW&Uö'VffW%ö÷B†&Æ—B‚’Â2Â7£Â6ÂÂÕDÄ÷&–v–äÖ¶R†2Óç7"ç†öfg6WBÂ2Óç7"ç–öfg6WBÂ’ÂÕDÅ6—¦TÖ¶R†2Óç7"çv–GF‚Â2Óç7"æ†V–v‡BÂ’ÂÆæRÂÂ&÷rÂ–ÖrÂ6ò“°¢Væ5ö%ö6÷•ö'VffW%÷FW‡GW&Uö÷B†&Æ—B‚’ÂÆæRÂÂ&÷rÂ–ÖrÂÕDÅ6—¦TÖ¶R†2Óç7"çv–GF‚Â2Óç7"æ†V–v‡BÂ’ÂBÂG£ÂFÂÂÕDÄ÷&–v–äÖ¶R†2ÓæG"ç†öfg6WBÂ2ÓæG"ç–öfg6WBÂ’ÂF÷“°¢7FF–2–çBöæ6S²–b‚öæ6R²²’ÔÄôr‚&FWF‚ÆæR6÷“¢VG‚VBÖWFÂf÷&ÖBVÇR(i"VÇRF‡&÷Vv‚7Fv–ær'VffW""Â2Óç7"çv–GF‚Â2Óç7"æ†V–v‡BÂ‡Vç6–væVBÆöær—2ç—†VÄf÷&ÖBÂ‡Vç6–væVBÆöær–Bç—†VÄf÷&ÖB“°¢–b‡&"’Ö—'&÷%÷&VF&6²†2ÓæG7BÂBÂFÂÂG£Âf2ÓæG"Â'FW‡GW&R"Â“°¢&WGW&ã°¢Ð¢–b†2Óç7"çv–GF‚Ò2ÓæG"çv–GF‚ÇÂ2Óç7"æ†V–v‡BÒ2ÓæG"æ†V–v‡BÇÂ†2Óç7"æFWF‚ò2Óç7"æFWF‚¢’Ò†2ÓæG"æFWF‚ò2ÓæG"æFWF‚¢’ÇÂ†2ÓæfÆw2bäådåô4õ•ôdÄu5ôTät”äUó$Eô$•Bbäådåô4õ•ôdÄu5ôÄ”äT%ôd”ÅDU%ô$•B¢ÇÂ2ç—†VÄf÷&ÖBÒBç—†VÄf÷&ÖBÇÂ32ÒC2’°¢$TEôôä4Rƒ32Â'FW‡GW&^(i'FW‡GW&R6÷’66Ræ÷B–×ÆVÖVçFVC¢VG‚VG‚VN(i"VG‚VG‚VBfÆw2R7‚f÷&ÖBVÇ^(i"VÇR4BVN(i"VB"Â2Óç7"çv–GF‚Â2Óç7"æ†V–v‡BÂ2Óç7"æFWF‚Â2ÓæG"çv–GF‚Â2ÓæG"æ†V–v‡BÂ2ÓæG"æFWF‚À¢2ÓæfÆw2Â‡Vç6–væVBÆöær—2ç—†VÄf÷&ÖBÂ‡Vç6–væVBÆöær–Bç—†VÄf÷&ÖBÂ32ÂC2“°¢&WGW&ã°¢Ð¢–CÄÕDÃD6ö×WFT6öÖÖæDVæ6öFW#â"Ò&Æ—B‚“°¢–b‡32¢Væ5ö%ö6÷•÷FW‡GW&R†"Â2ÂÂ6ÂÂÕDÄ÷&–v–äÖ¶R†2Óç7"ç†öfg6WBÂ2Óç7"ç–öfg6WBÂ2Óç7"ç¦öfg6WB’ÂÕDÅ6—¦TÖ¶R†2Óç7"çv–GF‚Â2Óç7"æ†V–v‡BÂç¢’ÂBÂÂFÂÂÕDÄ÷&–v–äÖ¶R†2ÓæG"ç†öfg6WBÂ2ÓæG"ç–öfg6WBÂ2ÓæG"ç¦öfg6WB’“°¢VÇ6P¢f÷"†–çB¢Ò²¢Âç£²¢²²¢Væ5ö%ö6÷•÷FW‡GW&R†"Â2Â7£²2Óç7"ç¦öfg6WB²¢Â6ÂÂÕDÄ÷&–v–äÖ¶R†2Óç7"ç†öfg6WBÂ2Óç7"ç–öfg6WBÂ’ÂÕDÅ6—¦TÖ¶R†2Óç7"çv–GF‚Â2Óç7"æ†V–v‡BÂ’ÂBÂG£²2ÓæG"ç¦öfg6WB²¢ÂFÂÂÕDÄ÷&–v–äÖ¶R†2ÓæG"ç†öfg6WBÂ2ÓæG"ç–öfg6WBÂ’“°¢–b‡&"’Ö—'&÷%÷&VF&6²†2ÓæG7BÂBÂFÂÂG£Âf2ÓæG"Â'FW‡GW&R"Â“°¢òòF–væ÷7F–2DÕôådåõC%EõE$4SÕw„ƒ¦f–ÆS¢gFW"V6‚6÷’–çFòâ$t$dbFW‡GW&RöbF†B6—¦RÂ6÷’F†R&W7VÇBöæ6RÖ÷&R–çFòG&6R'VffW"öâF†RuRF–ÖVÆ–æR†g&ÖRçVÖ&W'2¶WBöâF†R5R6–FR“²gFW"#C6÷–W2w&—FRF†Rf–ÆP¢²7FF–2–çBöâÒÓ²7FF–2V–çC3%÷BGrÂFƒ²7FF–26öç7B6†"§Fc²7FF–2–CÄÕDÄ'VffW#âF#²7FF–2–çBFã²7FF–2V–çCcE÷BFg%³#CÒÂg&öÓ°¢–b†öâÂ’²6öç7B6†"¦RÒvWFVçb‚$DÕôådåõC%EõE$4R"’Â¦fRÒvWFVçb‚$DÕôådåõC%Eôe$ôÒ"“²öâÒRbb766æb†RÂ"WW‚WR"ÂgGrÂgF‚’ÓÒ"bb‡FbÒ7G&6‡"†RÂs¢r’’bb¢²·Fc²g&öÒÒfRò7G'F÷VÆÂ†fRÂåTÄÂÂ’¢²Ð¢–b†öââbbFâÂ#CbbFöÖ–5öÆöB‚fuö7W%ög&ÖR’ãÒg&öÒbbBçv–GF‚ÓÒGrbbBæ†V–v‡BÓÒF‚bbBç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖE$t$dfÆöBbb32’°¢–b‚F"’²F"Ò¶uöFWbæWt'VffW%v—F„ÆVæwFƒ¢‡6—¦U÷B“#C¢Gr¢F‚¢‚÷F–öç3¤ÕDÅ&W6÷W&6U7F÷&vTÖöFU6†&VEÓ²&W5öFB‡F"“²Ð¢Væ5ö%ö6÷•÷FW‡GW&Uö'VffW"†&Æ—B‚’ÂBÂG£ÂFÂÂÕDÄ÷&–v–äÖ¶RƒÂÂ’ÂÕDÅ6—¦TÖ¶R‡GrÂF‚Â’ÂF"Â‡6—¦U÷B—Fâ¢Gr¢F‚¢‚Â‡6—¦U÷B—Gr¢‚Â‡6—¦U÷B—Gr¢F‚¢‚“°¢Fg%·Fâ²µÒÒFöÖ–5öÆöB‚fuö7W%ög&ÖR“°¢–b‡FâÓÒ#C’°¢'öVæB‚“²–b†uö&Æ—B’²Væ5ö%öVæB†uö&Æ—B“²uö&Æ—BÒæ–Ã²Ð¢×FÅöf–æ—6…öwR‚“°¢d”ÄR¦bÒf÷Vâ‡FbÂ'v""“°¢–b†b’²g&–çFb†bÂ"WRWRVEÆâ"ÂGrÂF‚ÂFâ“²gw&—FR‡Fg"Â‚Â#CÂb“²gw&—FR‡F"æ6öçFVçG2ÂÂ‡6—¦U÷B“#C¢Gr¢F‚¢‚Âb“²f6Æ÷6R†b“²Ð¢ÔÄôr‚%·C'B×G&6UÒVB6÷–W2(i"W2†g&ÖW2VÆÇ^(	2VÆÇR’"ÂFâÂFbÂ‡Vç6–væVBÆöærÆöær—Fg%³ÒÂ‡Vç6–væVBÆöærÆöær—Fg%³#3•Ò“°¢Ð¢Ð¢Ð§Ð ¢òòådã"w2DÅ52öâÖWFÄe‚w2FV×÷&Â66ÆW"ÂöæR66ÆW"W"æWGv÷&²æB6WBöbf÷&ÖG2æB6—¦W2††ÆUöçfâæ2$ådã"w2DÅ52"’âF†RfÆw2&Räu‚w2fVGW&RfÆw2‡F†RvÖR76W2ƒ#¢Ö÷F–öà¢òòfV7F÷'2BF†R–çWBw26—¦RÂ–çfW'FVBFWF‚Âv—F‚ƒCf÷"—G2DÅ54WFôW‡÷7W&R’âv†W&RÖWFÄe‚F¶W2æò66ÆW"f÷"F†Rf÷&ÖG2Â÷"FWF‚÷"Ö÷F–öâFW‡GW&R—2Ö—76–ærÂF†R–çWB—266ÆV@¢òò&–Æ–æV&Ç’–çFòF†R÷WGWBv—F‚&VBÆ–æS¢F†Rg&ÖR7F—2v†öÆRÂv—F†÷WBF†RFV×÷&Âv–à¦VçVÒ²DÅ55ôeôDUD…ô”ådU%DTBÒÃÂ2ÂDÅ55ôeôUDõôU…õ5U$RÒÃÂbÓ°§G—VFVb7G'V7B²V–çCcE÷BæWC²–CÄÕDÃDe…FV×÷&Å66ÆW#â63²å5T–çFVvW"¶W•³•Ó²–çBg&W6ƒ²ÒFÇ75÷66ÆW#°§7FF–2FÇ75÷66ÆW"uöFÇ75³EÓ°§7FF–2–CÄÕDÃD6ö×–ÆW#âuöFÇ75ö6ö×°§7FF–2FÇ75÷66ÆW"¦FÇ75÷66ÆW%öf÷"†6öç7BöFÇ72¦2Â–CÄÕDÅFW‡GW&Sâ6öÂÂ–CÄÕDÅFW‡GW&SâFWÂ–CÄÕDÅFW‡GW&SâÖ÷BÂ–CÄÕDÅFW‡GW&Sâ÷WBÂå5T–çFVvW"—rÂå5T–çFVvW"–‚’°¢å5T–çFVvW"¶W•³•ÒÒ¶6öÂç—†VÄf÷&ÖBÂFWç—†VÄf÷&ÖBÂÖ÷Bç—†VÄf÷&ÖBÂ÷WBç—†VÄf÷&ÖBÂ—rÂ–‚Â÷WBçv–GF‚Â÷WBæ†V–v‡BÂ2ÓæfÆw7Ó°¢FÇ75÷66ÆW"¦RÒåTÄÂÂ§7&RÒåTÄÃ°¢f÷"†–çB’Ò²’ÂBbbS²’²²’²–b†uöFÇ75¶•ÒææWBÓÒ2ÓææWB’RÒfuöFÇ75¶•Ó²VÇ6R–b‚uöFÇ75¶•ÒææWBbb7&R’7&RÒfuöFÇ75¶•Ó²Ð¢–b‚R’²RÒ7&Rò7&R¢fuöFÇ75³Ó²RÓææWBÒ2ÓææWC²RÓç62Òæ–Ã²ÖV×6WB†RÓæ¶W’ÂÂ6—¦VöbRÓæ¶W’“²Ð¢–b‚ÖVÖ6×†RÓæ¶W’Â¶W’Â6—¦Vöb¶W’’’&WGW&âRÓç62òR¢åTÄÃ°¢ÖVÖ7’†RÓæ¶W’Â¶W’Â6—¦Vöb¶W’“²RÓç62Òæ–Ã²RÓæg&W6‚Ò°¢å4W'&÷"¦W'"Òæ–Ã°¢–b‚uöFÇ75ö6ö×’uöFÇ75ö6ö×Ò¶uöFWbæWt6ö×–ÆW%v—F„FW67&—F÷#¥´ÕDÃD6ö×–ÆW$FW67&—F÷"æWuÒW'&÷#¢fW'%Ó°¢fÆöB66ÆRÒ†fÆöB–÷WBçv–GF‚ò†fÆöB–—s°¢ÕDÄe…FV×÷&Å66ÆW$FW67&—F÷"¦BÒ´ÕDÄe…FV×÷&Å66ÆW$FW67&—F÷"æWuÓ°¢Bæ6öÆ÷%FW‡GW&Tf÷&ÖBÒ6öÂç—†VÄf÷&ÖC²BæFWF…FW‡GW&Tf÷&ÖBÒFWç—†VÄf÷&ÖC²BæÖ÷F–öåFW‡GW&Tf÷&ÖBÒÖ÷Bç—†VÄf÷&ÖC²Bæ÷WGWEFW‡GW&Tf÷&ÖBÒ÷WBç—†VÄf÷&ÖC°¢Bæ–çWEv–GF‚Ò—s²Bæ–çWD†V–v‡BÒ–ƒ²Bæ÷WGWEv–GF‚Ò÷WBçv–GFƒ²Bæ÷WGWD†V–v‡BÒ÷WBæ†V–v‡C°¢BæWFôW‡÷7W&TVæ&ÆVBÒ†2ÓæfÆw2bDÅ55ôeôUDõôU…õ5U$R’Ò°¢Bæ–çWD6öçFVçE&÷W'F–W4Væ&ÆVBÒ”U3²Bæ–çWD6öçFVçDÖ–å66ÆRÒ66ÆS²Bæ–çWD6öçFVçDÖ…66ÆRÒ66ÆS°¢RÓç62ÒuöFÇ75ö6ö×bb´ÕDÄe…FV×÷&Å66ÆW$FW67&—F÷"7W÷'G4ÖWFÃDeƒ¦uöFWeÒò¶BæWuFV×÷&Å66ÆW%v—F„FWf–6S¦uöFWb6ö×–ÆW#¦uöFÇ75ö6ö×Ò¢æ–Ã°¢–b‚RÓç62’ÔÄôr‚%µ$TEÒDÅ53¢ÖWFÄe‚Ö¶W2æòFV×÷&Â66ÆW"f÷"6öÆ÷W"f÷&ÖBVÇRÂFWF‚VÇRÂÖ÷F–öâVÇRÂ÷WGWBVÇRÂVÇW‚VÇR(i"VÇW‚VÇS²66Æ–ær&–Æ–æV&Ç’"Â‡Vç6–væVBÆöær–6öÂç—†VÄf÷&ÖBÂ‡Vç6–væVBÆöær–FWç—†VÄf÷&ÖBÀ¢‡Vç6–væVBÆöær–Ö÷Bç—†VÄf÷&ÖBÂ‡Vç6–væVBÆöær–÷WBç—†VÄf÷&ÖBÂ‡Vç6–væVBÆöær–—rÂ‡Vç6–væVBÆöær––‚Â‡Vç6–væVBÆöær–÷WBçv–GF‚Â‡Vç6–væVBÆöær–÷WBæ†V–v‡B“°¢VÇ6RÔÄôr‚$DÅ53¢ÖWFÄe‚FV×÷&Â66ÆW"VÇW‚VÇR(i"VÇW‚VÇRÂ6öÆ÷W"f÷&ÖBVÇRÂFWF‚VÇRÂÖ÷F–öâVÇRÂ÷WGWBVÇR"Â‡Vç6–væVBÆöær–—rÂ‡Vç6–væVBÆöær––‚Â‡Vç6–væVBÆöær–÷WBçv–GF‚Â‡Vç6–væVBÆöær–÷WBæ†V–v‡BÀ¢‡Vç6–væVBÆöær–6öÂç—†VÄf÷&ÖBÂ‡Vç6–væVBÆöær–FWç—†VÄf÷&ÖBÂ‡Vç6–væVBÆöær–Ö÷Bç—†VÄf÷&ÖBÂ‡Vç6–væVBÆöær–÷WBç—†VÄf÷&ÖB“°¢&WGW&âRÓç62òR¢åTÄÃ°§Ð§7FF–2fö–BFÇ75öWfÂ†6öç7BöFÇ72¦2’°¢'öfÇW6‚‚“²òòF†R76W2&Vf÷&R—BÆæBf—'7C¢F†W’w&—FR—G2–çWG0¢–CÄÕDÅFW‡GW&Sâ6öÂÒ2ÓçFW…´DÅ55ô4ôÄõ%ÒòFW…ö×FÂ†2ÓçFW…´DÅ55ô4ôÄõ%Ò’¢æ–ÂÂFWÒ2ÓçFW…´DÅ55ôDUD…ÒòFW…ö×FÂ†2ÓçFW…´DÅ55ôDUD…Ò’¢æ–ÂÂÖ÷BÒ2ÓçFW…´DÅ55ôÔõD”ôåÒòFW…ö×FÂ†2ÓçFW…´DÅ55ôÔõD”ôåÒ’¢æ–ÂÀ¢÷WBÒ2ÓçFW…´DÅ55ôõUEUEÒòFW…ö×FÂ†2ÓçFW…´DÅ55ôõUEUEÒ’¢æ–Ã°¢–b‚6öÂÇÂ÷WB’²7FF–2–çBöæ6S²–b‚öæ6R²²’ÔÄôr‚%µ$TEÒDÅ52WfÇVFVBv—F†÷WBW2"Â6öÂò&â÷WGWB"¢&â–çWB6öÆ÷W""“²&WGW&ã²Ð¢§6¶—öFVÂ†2ÓçFW…´DÅ55ôõUEUEÒÓç6W&–Â“°¢å5T–çFVvW"—rÒ2Óæ—râò„å5T–çFVvW"–2Óæ—r¢6öÂçv–GF‚Â–‚Ò2Óæ–‚âò„å5T–çFVvW"–2Óæ–‚¢6öÂæ†V–v‡C°¢'öVæB‚“²–b†uö&Æ—B’²Væ5ö%öVæB†uö&Æ—B“²uö&Æ—BÒæ–Ã²Ð¢FÇ75÷66ÆW"¦RÒFWbbÖ÷BòFÇ75÷66ÆW%öf÷"†2Â6öÂÂFWÂÖ÷BÂ÷WBÂ—rÂ–‚’¢åTÄÃ°¢–b‚R’°¢ådæ6÷•&Vv–öâ7"Ò¶2Óæ&6U´DÅ55ô4ôÄõ%Õ³ÒÂ2Óæ&6U´DÅ55ô4ôÄõ%Õ³ÒÂÂ†–çB–—rÂ†–çB––‚ÂÒÂG"Ò¶2Óæ&6U´DÅ55ôõUEUEÕ³ÒÂ2Óæ&6U´DÅ55ôõUEUEÕ³ÒÂÂ†–çB–÷WBçv–GF‚Â†–çB–÷WBæ†V–v‡BÂÓ°¢–b‚†÷WBçW6vRbÕDÅFW‡GW&UW6vU&VæFW%F&vWB’bb×FÅöf×Eö&Æö6²†÷WBç—†VÄf÷&ÖB’bbf×Eö–çEö÷%öG2†÷WBç—†VÄf÷&ÖB’’÷WBÒFW…÷Ww&FR†2ÓçFW…´DÅ55ôõUEUEÒÂ÷WBÂÕDÅFW‡GW&UW6vU&VæFW%F&vWBÂ÷WBç—†VÄf÷&ÖB“°¢–b‚6÷•÷C'EöG&r†6öÂÂ÷WBÂÂÂÂÂg7"ÂfG"ÂÂ’’²7FF–2–çBöæ6S²–b‚öæ6R²²’ÔÄôr‚%µ$TEÒDÅ53¢F†R&–Æ–æV"7FæBÖ–â6÷VÆBæ÷BG&rf÷&ÖBVÇR–çFòf÷&ÖBVÇR"Â‡Vç6–væVBÆöær–6öÂç—†VÄf÷&ÖBÂ‡Vç6–væVBÆöær–÷WBç—†VÄf÷&ÖB“²Ð¢&WGW&ã°¢Ð¢–CÄÕDÃDe…FV×÷&Å66ÆW#â62ÒRÓç63°¢–b‚†6öÂçW6vRb62æ6öÆ÷%FW‡GW&UW6vR’Ò62æ6öÆ÷%FW‡GW&UW6vR’6öÂÒFW…÷Ww&FR†2ÓçFW…´DÅ55ô4ôÄõ%ÒÂ6öÂÂ62æ6öÆ÷%FW‡GW&UW6vRÂ6öÂç—†VÄf÷&ÖB“°¢–b‚†FWçW6vRb62æFWF…FW‡GW&UW6vR’Ò62æFWF…FW‡GW&UW6vR’FWÒFW…÷Ww&FR†2ÓçFW…´DÅ55ôDUD…ÒÂFWÂ62æFWF…FW‡GW&UW6vRÂFWç—†VÄf÷&ÖB“°¢–b‚†Ö÷BçW6vRb62æÖ÷F–öåFW‡GW&UW6vR’Ò62æÖ÷F–öåFW‡GW&UW6vR’Ö÷BÒFW…÷Ww&FR†2ÓçFW…´DÅ55ôÔõD”ôåÒÂÖ÷BÂ62æÖ÷F–öåFW‡GW&UW6vRÂÖ÷Bç—†VÄf÷&ÖB“°¢–b‚†÷WBçW6vRb62æ÷WGWEFW‡GW&UW6vR’Ò62æ÷WGWEFW‡GW&UW6vR’÷WBÒFW…÷Ww&FR†2ÓçFW…´DÅ55ôõUEUEÒÂ÷WBÂ62æ÷WGWEFW‡GW&UW6vRÂ÷WBç—†VÄf÷&ÖB“°¢–CÄÕDÅFW‡GW&SâW‡òÒ2ÓçFW…´DÅ55ôU…õ5U$UÒòFW…ö×FÂ†2ÓçFW…´DÅ55ôU…õ5U$UÒ’¢æ–Ã°¢–b†uö&Æ—B’²Væ5ö%öVæB†uö&Æ—B“²uö&Æ—BÒæ–Ã²ÒòòW6vRWw&FR6÷–W2F‡&÷Vv‚—@¢V&EöÆæEö6ö×WFR‚“°¢–CÄÕDÃD6ö×WFT6öÖÖæDVæ6öFW#â6RÒ¶uö6"6ö×WFT6öÖÖæDVæ6öFW%Ó²òòF†R66ÆW"w2Væ6öFW'26öÖRgFW"F†Rg&ÖRw2v÷&²6òf#¢ÖWFÄe‚÷Vç2Væ6öFW'2öb—G2÷vâÂv†–6‚F¶Ræò&'&–W ¢¶6R&'&–W$gFW%VWVU7FvW3¤ÕDÅ7FvTÆÂ&Vf÷&U7FvW3¤ÕDÅ7FvT&Æ—Bf—6–&–Æ—G”÷F–öç3§&W6÷W&6U÷f—2‚•Ó°¢¶6R&'&–W$gFW%7FvW3¤ÕDÅ7FvT&Æ—B&Vf÷&UVWVU7FvW3¤ÕDÅ7FvTÆÂf—6–&–Æ—G”÷F–öç3§&W6÷W&6U÷f—2‚•Ó°¢¶6RVæDVæ6öF–æuÓ°¢62æ6öÆ÷%FW‡GW&RÒ6öÃ²62æFWF…FW‡GW&RÒFW²62æÖ÷F–öåFW‡GW&RÒÖ÷C²62æ÷WGWEFW‡GW&RÒ÷WC²62æW‡÷7W&UFW‡GW&RÒW‡ó°¢62æ–çWD6öçFVçEv–GF‚Ò—s²62æ–çWD6öçFVçD†V–v‡BÒ–ƒ°¢–b„f–Æ&ÆR†Ö4õ2#rãÂ”õ2#rãÂ¢’’²òòF†R&V7FævÆW2r6÷&æW'3¢ÆÂBF†R÷&–v–âöâF†RvÖR6VVâÂ6òâöÆFW"7—7FVÒÆ÷6W2æ÷F†–ærF†W&P¢62æ6öÆ÷$6öçFVçDöfg6WE‚Ò„å5T–çFVvW"–2Óæ&6U´DÅ55ô4ôÄõ%Õ³Ó²62æ6öÆ÷$6öçFVçDöfg6WE’Ò„å5T–çFVvW"–2Óæ&6U´DÅ55ô4ôÄõ%Õ³Ó°¢62æFWF„6öçFVçDöfg6WE‚Ò„å5T–çFVvW"–2Óæ&6U´DÅ55ôDUD…Õ³Ó²62æFWF„6öçFVçDöfg6WE’Ò„å5T–çFVvW"–2Óæ&6U´DÅ55ôDUD…Õ³Ó°¢62æÖ÷F–öä6öçFVçDöfg6WE‚Ò„å5T–çFVvW"–2Óæ&6U´DÅ55ôÔõD”ôåÕ³Ó²62æÖ÷F–öä6öçFVçDöfg6WE’Ò„å5T–çFVvW"–2Óæ&6U´DÅ55ôÔõD”ôåÕ³Ó°¢62æ÷WGWDöfg6WE‚Ò„å5T–çFVvW"–2Óæ&6U´DÅ55ôõUEUEÕ³Ó²62æ÷WGWDöfg6WE’Ò„å5T–çFVvW"–2Óæ&6U´DÅ55ôõUEUEÕ³Ó°¢ÒVÇ6R–b†2Óæ&6U´DÅ55ô4ôÄõ%Õ³ÒÇÂ2Óæ&6U´DÅ55ô4ôÄõ%Õ³ÒÇÂ2Óæ&6U´DÅ55ôDUD…Õ³ÒÇÂ2Óæ&6U´DÅ55ôDUD…Õ³ÒÇÂ2Óæ&6U´DÅ55ôÔõD”ôåÕ³ÒÇÂ2Óæ&6U´DÅ55ôÔõD”ôåÕ³ÒÇÂ2Óæ&6U´DÅ55ôõUEUEÕ³ÒÇÂ2Óæ&6U´DÅ55ôõUEUEÕ³Ò’°¢7FF–2–çBöæ6S²–b‚öæ6R²²’ÔÄôr‚%µ$TEÒDÅ53¢&V7FævÆW2v’g&öÒF†R÷&–v–âæVVBÖ4õ2ò”õ2#rw2ÖWFÄeƒ²F¶Vâg&öÒF†R÷&–v–â"“°¢Ð¢62æÖ÷F–öåfV7F÷%66ÆU‚Ò2Óæ×e³Ó²62æÖ÷F–öåfV7F÷%66ÆU’Ò2Óæ×e³Ó°¢62æ¦—GFW$öfg6WE‚Ò2Óæ¦—GFW%³Ó²62æ¦—GFW$öfg6WE’Ò2Óæ¦—GFW%³Ó°¢62ç&TW‡÷7W&RÒ2Óç&UöW‡÷7W&S°¢62æFWF…&WfW'6VBÒ†2ÓæfÆw2bDÅ55ôeôDUD…ô”ådU%DTB’Ò°¢62ç&W6WBÒ2Óç&W6WBÇÂRÓæg&W6ƒ²RÓæg&W6‚Ò°¢·62Væ6öFUFô6öÖÖæD'VffW#¦uö6%Ó°¢–b†2ÓçFW…´DÅ55ô$”5Ò’²7FF–2–çBöæ6S²–b‚öæ6R²²’ÔÄôr‚%µ$TEÒDÅ53¢F†RvÖRw2&–2Ö6²†&V7F—fRÖ6²f÷"ÖWFÄe‚’—2æ÷B76VBöâ"“²Ð§Ð ¢òòF†R6öÆ÷W"6†ææVÇ2ÖWFÂf÷&ÖB†2Â26†ææVÂw&—FRÖ6³¢6ÆV"v†÷6RÖ6²6÷fW'2F†VÒÆÂw&—FW2WfW'’FW†VÂv†öÆR„d”äÂdåD5’D5D”52ÒF†R—fÆ–6R6‡&öæ–6ÆW26ÆV'2à¢òò#‚&VæFW"F&vWBv—F‚Ö6²ƒ§7FF–2Vç6–væVB×FÅ÷eö6†Ö6²„ÕDÅ—†VÄf÷&ÖBb’°¢7v—F6‚‚†–çB—b’°¢66R¢&WGW&âƒƒ²òò€¢66R¢66R¢66R#¢66R3¢66RC¢66R#¢66R##¢66R#3¢66R#C¢66R#S¢66RS3¢66RSC¢66RSS¢&WGW&âƒ²òò#‚ò#bò#3 ¢66R3¢66R3¢66R3#¢66R33¢66R3C¢66Rc¢66Rc#¢66Rc3¢66RcC¢66RcS¢66R3¢66RC¢66RS¢&WGW&âƒ3²òò$s‚ò$sbò$s3 ¢66RC¢66R“#¢66R“3¢&WGW&âƒs²òò#Tse#Rò$s#bò$t#”SP¢Ð¢&WGW&â†c°§Ð¢òòuS¢6ÆV"FW‡GW&R&Vv–öâv—F‚F†Rv—fVâfÇVR‡&Wf–÷W6Ç’Ö—6Æ—7FVB2æòÖ÷Â6ÆÆVBRF–ÖW2B–æ—F–Æ—¦F–öã²F†R–æ—F–ÂfÇVRöbÖWFÂ&—fFRFW‡GW&R—2VæFVf–æVBÂæBF†RvÖR&VÆ–W2öâF†R6ÆV&VB–æ—F–ÂfÇVR’à¢òòv†öÆRÆWfVÂæBv†öÆRf6RÂÆÂf÷W"6†ææVÇ2w&—GFVâÓâ÷VâöæR6ÆV"ÖöæÇ’&VæFW"72W"Æ–W"òFWF‚6Æ–6R‡F†R†&Gv&R6öçfW'G2F†R6ÆV"fÇVR'’F†RFW‡GW&Rf÷&ÖBÂ–æ6ÇVF–ær–çFVvW"æBFWF‚“°¢òòf–Wrv—F‚6†ævVBf÷&ÖC¢6ÆV"öâF†Bf–Wrâ'F–Â6öÆ÷W"&Vv–öç2ò6†ææVÂÖ6·2W6RG&–ævÆRæB&W6W'fRWfW'—F†–ærVÇ6Rà§7FF–2fö–B6ÆV%÷FW…÷G&’†–CÄÕDÅFW‡GW&SâBÂVç6–væVBÇbÂVç6–væVBÃÂ6öç7Bö6ÆV%÷FW‚¦2“°§7FF–2fö–B6ÆV%÷FW‚†6öç7Bö6ÆV%÷FW‚¦2’°¢–b†2ÓçFW‚’§6¶—öFVÂ†2ÓçFW‚Óç6W&–Â“°¢–CÄÕDÅFW‡GW&SâBÒFW…ö×FÂ†2ÓçFW‚“°¢B‚%‚6W#ÒVÆÇRf–Wrf÷&ÖBR7‚&Vv–öâVBÂVBÂVBVG‚VG‚VBÖ6²R7‚W2VrÂVrÂVrÂVr"Â2ÓçFW‚ò2ÓçFW‚Óç6W&–Â¢VÆÂÂ2Óæ†5÷f–Wrbb2Óçf–Wrç6WEöf÷&ÖBò2Óçf–Wræf÷&ÖB¢ÓÂ2Óç"ç†öfg6WBÂ2Óç"ç–öfg6WBÂ2Óç"ç¦öfg6WBÂ2Óç"çv–GF‚Â2Óç"æ†V–v‡BÂ2Óç"æFWF‚Â2ÓæÖ6²À¢2Óæ—5÷V–çBò'V’"¢&b"Â2Óæ—5÷V–çBò†F÷V&ÆR–2Óæ2çU³Ò¢2Óæ2æe³ÒÂ2Óæ—5÷V–çBò†F÷V&ÆR–2Óæ2çU³Ò¢2Óæ2æe³ÒÂ2Óæ—5÷V–çBò†F÷V&ÆR–2Óæ2çU³%Ò¢2Óæ2æe³%ÒÂ2Óæ—5÷V–çBò†F÷V&ÆR–2Óæ2çU³5Ò¢2Óæ2æe³5Ò“°¢–b‚B’&WGW&ã°¢Vç6–væVBÇbÒ2Óæ†5÷f–Wrbb2Óçf–Wrç6WEöÆWfVÇ2ò2Óçf–Wræ&6UöÆWfVÂ¢ÂÃÒ2Óæ†5÷f–Wrbb2Óçf–Wrç6WEöÆ–W'2ò2Óçf–WræÖ–åöÆ–W"¢°¢å5T–çFVvW"rÒÔ‚‡Bçv–GF‚ãâÇbÂR’Â‚ÒÔ‚‡Bæ†V–v‡BãâÇbÂR“°¢–çB—36BÒBçFW‡GW&UG—RÓÒÕDÅFW‡GW&UG—S4BÂç¢Ò2Óç"æFWF‚ò2Óç"æFWF‚¢°¢–b‚‡BçW6vRbÕDÅFW‡GW&UW6vU&VæFW%F&vWB’bb×FÅöf×Eö&Æö6²‡Bç—†VÄf÷&ÖB’bbBç6×ÆT6÷VçBÓÒ¢BÒFW…÷Ww&FR†2ÓçFW‚ÂBÂÕDÅFW‡GW&UW6vU&VæFW%F&vWBÂBç—†VÄf÷&ÖB“²òò6ÆV"öbFW‡GW&RÖFRv—F†÷WB&VæFW"×F&vWBW6R‡6¶vVB“¢v—fR—BF†BW6P¢–b†2Óæ†5÷f–Wrbb2Óçf–Wrç6WEöf÷&ÖB’²òòf–Wrv—F‚6†ævVBf÷&ÖC¢F†R6ÆV"fÇVR—2–çFW'&WFVB–âF†Rf–Wrf÷&ÖB†ÆÂRöbF†RvÖRw26ÆÇ2&RF†RV’fW'6–öâv—F‚â–çFVvW"f–Wr’ÂæBF†R6ÆV"—2FöæRöâf–WröbF†R6ÖRf÷&Ö@¢V–çC3%÷BgbÒ²V–çC…÷B7w¥³EÓ°¢–b‚çfåöf×E÷Fõö×FÂ†2Óçf–Wræf÷&ÖBÂggbÂ7w¢’ÇÂgb’²$TEôôä4RƒC"Â'FW‡GW&R6ÆV"f–Wrf÷&ÖBR7‚†2æòWV—fÆVçB"Â2Óçf–Wræf÷&ÖB“²&WGW&ã²Ð¢–b‡FW…öæVVE÷gb‡Bç—†VÄf÷&ÖBÂ„ÕDÅ—†VÄf÷&ÖB—gb’bb‡BçW6vRbÕDÅFW‡GW&UW6vU—†VÄf÷&ÖEf–Wr’’BÒFW…÷Ww&FU÷gb†2ÓçFW‚ÂBÂ„ÕDÅ—†VÄf÷&ÖB—gb“°¢–b‡FW…öæVVE÷gb‡Bç—†VÄf÷&ÖBÂ„ÕDÅ—†VÄf÷&ÖB—gb’’°¢–CÄÕDÅFW‡GW&SâbÒ·BæWuFW‡GW&Uf–Wuv—F…—†VÄf÷&ÖC¢„ÕDÅ—†VÄf÷&ÖB—geÓ°¢–b‚b’²$TEôôä4RƒC2Â'FW‡GW&R6ÆV#¢6ææ÷B7&VFRF†Rf–Ws¢f÷&ÖBVÇR(i"WR"Â‡Vç6–væVBÆöær—Bç—†VÄf÷&ÖBÂgb“²&WGW&ã²Ð¢BÒc°¢Ð¢Ð¢ÕDÅ—†VÄf÷&ÖBbÒBç—†VÄf÷&ÖC°¢–çBFWF‚ÒbÓÒÕDÅ—†VÄf÷&ÖDFWFƒ3$fÆöBÇÂbÓÒÕDÅ—†VÄf÷&ÖDFWFƒeVæ÷&ÒÇÂbÓÒÕDÅ—†VÄf÷&ÖDFWFƒ3$fÆöEõ7FVæ6–Ã‚Â7FVæ6–ÂÒbÓÒÕDÅ—†VÄf÷&ÖE7FVæ6–Ã‚ÇÂbÓÒÕDÅ—†VÄf÷&ÖDFWFƒ3$fÆöEõ7FVæ6–Ãƒ°¢–b†2Óç"ç†öfg6WBÇÂ2Óç"ç–öfg6WBÇÂ„å5T–çFVvW"–2Óç"çv–GF‚ÒrÇÂ„å5T–çFVvW"–2Óç"æ†V–v‡BÒ‚ÇÂ†2ÓæÖ6²b×FÅ÷eö6†Ö6²‡b’’Ò×FÅ÷eö6†Ö6²‡b¢ÇÂ‡BçW6vRbÕDÅFW‡GW&UW6vU&VæFW%F&vWB’’°¢–b‚FWF‚bb7FVæ6–Âbb‡BçW6vRbÕDÅFW‡GW&UW6vU&VæFW%F&vWB’’²6ÆV%÷FW…÷G&’‡BÂÇbÂÃÂ2“²&WGW&ã²Ð¢$TEôôä4RƒCBÂ'FW‡GW&R6ÆV"66Ræ÷B–×ÆVÖVçFVC¢&Vv–öâVBÂVBVG‚VB†ÆWfVÂ6—¦RVÇW‚VÇR’Ö6²R7‚&VæFW"×F&vWB6&ÆRVB†f÷&ÖBR7‚fÆw2R7‚’"Â2Óç"ç†öfg6WBÂ2Óç"ç–öfg6WBÂ2Óç"çv–GF‚Â2Óç"æ†V–v‡BÀ¢‡Vç6–væVBÆöær—rÂ‡Vç6–væVBÆöær–‚Â2ÓæÖ6²Â‡BçW6vRbÕDÅFW‡GW&UW6vU&VæFW%F&vWB’Â2ÓçFW‚ÓæBæf÷&ÖBÂ2ÓçFW‚ÓæBæfÆw2“°¢&WGW&ã°¢Ð¢'öVæB‚“²–b†uö&Æ—B’²Væ5ö%öVæB†uö&Æ—B“²uö&Æ—BÒæ–Ã²Ð¢f÷"†–çB¢Ò²¢Âç£²¢²²’°¢ÕDÃE&VæFW%74FW67&—F÷"§'Ò´ÕDÃE&VæFW%74FW67&—F÷"æWuÓ°¢ÕDÅ&VæFW%74GF6†ÖVçDFW67&—F÷"¦ÒFWF‚ò'æFWF„GF6†ÖVçB¢7FVæ6–Âò'ç7FVæ6–ÄGF6†ÖVçB¢'æ6öÆ÷$GF6†ÖVçG5³Ó°¢çFW‡GW&RÒC²æÆWfVÂÒÇc²æÆöD7F–öâÒÕDÄÆöD7F–öä6ÆV#²ç7F÷&T7F–öâÒÕDÅ7F÷&T7F–öå7F÷&S°¢–b†—36B’æFWF…ÆæRÒ2Óç"ç¦öfg6WB²£²VÇ6Rç6Æ–6RÒÃ²2Óç"ç¦öfg6WB²£°¢–b†FWF‚’²'æFWF„GF6†ÖVçBæ6ÆV$FWF‚ÒbÓÒÕDÅ—†VÄf÷&ÖDFWFƒeVæ÷&ÒòçfåöFWFƒeö6ÆV%÷fÇVR†2Óæ2æe³Ò’¢2Óæ2æe³Ó°¢–b‡7FVæ6–Â’²'ç7FVæ6–ÄGF6†ÖVçBçFW‡GW&RÒC²'ç7FVæ6–ÄGF6†ÖVçBæÆWfVÂÒÇc²'ç7FVæ6–ÄGF6†ÖVçBç6Æ–6RÒç6Æ–6S²'ç7FVæ6–ÄGF6†ÖVçBæÆöD7F–öâÒÕDÄÆöD7F–öä6ÆV#°¢'ç7FVæ6–ÄGF6†ÖVçBç7F÷&T7F–öâÒÕDÅ7F÷&T7F–öå7F÷&S²'ç7FVæ6–ÄGF6†ÖVçBæ6ÆV%7FVæ6–ÂÒ2Óæ2çU³Ó²ÒÐ¢VÇ6R–b‡7FVæ6–Â’'ç7FVæ6–ÄGF6†ÖVçBæ6ÆV%7FVæ6–ÂÒ2Óæ2çU³Ó°¢VÇ6R'æ6öÆ÷$GF6†ÖVçG5³Òæ6ÆV$6öÆ÷"Ò2Óæ—5÷V–çBòÕDÄ6ÆV$6öÆ÷$Ö¶R†2Óæ2çU³ÒÂ2Óæ2çU³ÒÂ2Óæ2çU³%ÒÂ2Óæ2çU³5Ò’¢ÕDÄ6ÆV$6öÆ÷$Ö¶R†2Óæ2æe³ÒÂ2Óæ2æe³ÒÂ2Óæ2æe³%ÒÂ2Óæ2æe³5Ò“°¢6W%övFR‚“²Væ5÷%öVæB†Væ5÷%ö&Vv–â‡'’“°¢Ð§Ð ¢òò6öçFW‡BFW67&—F–öâöböæR7V&Ö—B†×FÅ÷7V%ö6öÖÖ—BF¶W2öæRW"7V&Ö—BæB&–çG2—B–çFòF†Rf–ÇW&RÆ–æRv†VâF†R6öÖÖæB'VffW"W'&÷'2“¢f÷"F†RuRÖVÖ÷'’fVÇBöâF†Rf—'7Bg&ÖRVçFW&–ærG&f–æ6ÂÂF†Rf–ÇW&RÆ–æR†BöæÇ’F†R6WVVæ6RçVÖ&W"æBW'&÷"6öFRÂæB—Bv2–×÷76–&ÆRFòFVÆÂv†Bv2–âF†B7V&Ö—@§7FF–26öç7B6†"¦67G…÷6æ‡fö–B’°¢7FF–26†"%³3#Ó²7FF–2V–çCcE÷B÷72Â÷WÂ÷&WWÂ÷6W#²V–çCcE÷BWÒÂ&WWÒ²×FÅ÷¶u÷7FG2‚gWÂg&WW“°¢F‡&VEö×WFW…öÆö6²‚fuöFVfW'&VEö×R“²Vç6–væVBÆöæræBÒ‡Vç6–væVBÆöær–uöFVfW'&VBæ6÷VçC²F‡&VEö×WFW…÷VæÆö6²‚fuöFVfW'&VEö×R“°¢6ç&–çFb†"Â6—¦Vöb"Â&g&ÖRVÆÇS²6–æ6RF†RÆ7B7V&Ö—76–öã¢76W2²VÆÇRÂf—'7BFW‡GW&RWÆöG2²VÆÇRÂ&R×WÆöG2²VÆÇRÂ6W&–ÂF&ÆR6†ævW2²VÆÇS²FVfW'&VB×&VÆV6RVWVRVÇR"À¢‡Vç6–væVBÆöærÆöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR’Â‡Vç6–væVBÆöærÆöær’†u÷7FE÷72Ò÷72’Â‡Vç6–væVBÆöærÆöær’‡WÒ÷W’Â‡Vç6–væVBÆöærÆöær’‡&WWÒ÷&WW’À¢‡Vç6–væVBÆöærÆöær’†u÷6W%övVâÒ÷6W"’ÂæB“°¢÷72Òu÷7FE÷73²÷WÒW²÷&WWÒ&WW²÷6W"Òu÷6W%övVã²&WGW&â#°§Ð¢òòv†VâF†RvÖR—G6VÆb6·2f÷"3g2‡6–ær'’F†RvÖRw2&W6VçB–çFW'fÂÂ–çFW'fÂ"Â†ÆUöçfâæ2’F†RF—7Æ’Æ–æ²'Vç2B3‡¢Föó¢vÖW2v†÷6RÆöv–26÷VçG2g&ÖW2Â7V6‚2VævvRÂ&Ræ÷BVÆÆVBWFòcg2'’F†RF—7Æ’Æ–æ²öâ”õ0¥õöGG&–'WFUõò‚‡vV²’’–çBçfåövÖUög2‡fö–B’²&WGW&âc²Òòò†ÆUöçfâæ2&÷f–FW2F†R7G&öærFVf–æ—F–öà¦–çB×FÅög5÷F&vWB‡fö–B’²7FF–2–çBVçbÒÓ#²–b†VçbÓÒÓ"’²6öç7B6†"¦RÒvWFVçb‚$DÕôe2"“²VçbÒRòFö’†R’¢Ó²–b†VçbÒ3bbVçbÒc’VçbÒÓ²VÇ6RÔÄôr‚$DÕôe3ÒVC¢g&ÖR&FR–ææVB'’F†RFW7B7v—F6‚"ÂVçb“²Ð¢&WGW&âVçbâòVçb¢çfåövÖUög2‚’ÓÒ3ò3¢c²Ð¢òò†æB7V&Ö—76–öâFòF†RuR–â6VvÖVçG2„DÕôådåô5UCÖG&r6÷VçBÂÒöfc²FVfVÇBcB“¢v†VâF†RvÖR†öÆG2&6²F†RæW‡Bg&ÖRw27V&Ö—BVçF–ÂF†—2g&ÖRw2uRv÷&²—2FöæR…F†RÆVvVæBöb¦VÆF¢'&VF‚öbF†Rv–ÆC¢F†RÖ–âF‡&VBw2çfå7–æ5v—BöâF†RfVæ6RöbF†Rg&ÖR§W7B7V&Ö—GFVB’À¢òòF†RuR6—G2–FÆRf÷"F†RÖ–ÆÆ—6V6öæG2F†R†÷7B7VæG2G&ç6ÆF–ær–ç6–FRF†R7V&Ö—B†–âF†R6‡&–æRV6‚g&ÖR—2uR2×2²G&ç6ÆF–öâr×2&6²Fò&6²’âV6‚F–ÖRF†—2Öç’G&w2†fR67V×VÆFVBÂF†R6öÖÖæB'VffW"Væ6öFVB6òf"—26öÖÖ—GFVB&WGvVVâGvò76W3¢F†RuR'Vç2F†RV&Æ–W"76W2v†–ÆRF†R5RG&ç6ÆFW2F†RÆFW"öæW2à¢òò7WBöæÇ’v†VâæòVæ6öFW"—2÷VâæBæò72—2FVfW'&VBòÖW&vVBò†ö—7FVC²6öÖÖ—BF†—2&F6‚w2WÆöG2f—'7B†ÆFW"G&w2W6RF†VÒ’âF†RfVæ6R6–væÂæB6ö×ÆWF–öâv÷&²7F’öâF†RÆ7B6VvÖVçBöæÇ“¢F†RVWVRW†V7WFW2–â6öÖÖ—B÷&FW"æBWfW'’Væ6öFW"7F'G2v—F‚VWVR&'&–W ¢òò'VffW"(i"FW‡GW&R6÷’—27WBgFW"2vVÆÂÂBF†RæW‡B72&÷VæF'’v†FWfW"F†RG&r6÷VçB†uö7WEö6÷’Â6WB'’6÷•ö#'B“¢öâF†R7v—F6‚F†RuR'Vç27V&Ö—76–öâw26÷–W226ööâ2F†P¢òò7V&Ö—B&WGW&ç2ÂæBvÖRÖ’&WW6RF†R6÷’w26÷W&6R&–v‡BgFW"â6F†W&–æS¢gVÆÂ&öG’w2&W6÷W&6RF‡&VBw&—FW2F†RæW‡BFW‡GW&R–çFòF†R7Fv–ær&Æö6²cB×2gFW"F†RÖ–âF‡&V@¢òò&V6÷&FVB6÷’g&öÒ—B†Ö–â³ƒ3VScBÂg&ÖR7F'B’æB&Vf÷&RF†Bg&ÖRw27V&Ö—B†BWfVâ&WGW&æVC¢F†R†÷7Bv2'V–ÆF–ær—VÆ–æW2–ç6–FRF†R7V&Ö—Bf÷"3×2ÂæB6öÖÖ—GFVBF†P¢òò6÷’öæÇ’B—G2VæBÂ6òF†R7V'F—FÆR&æBw2w&F–VçB„$32Â3'ƒS"’6ÖR÷WB2F†RæW‡BFW‡GW&Rw2&Æö6·2â6öÖÖ—GFVBBF†Rf—'7B72&÷VæF'’ÂF†R6÷–W2'Vâv†–ÆRF†R†÷7BVæ6öFW2F†R&W7@§7FF–2–çB7WEöG&w2‡fö–B’²7FF–2–çBbÒÓ²–b‡bÂ’²6öç7B6†"¦RÒvWFVçb‚$DÕôådåô5UB"“²bÒRòFö’†R’¢cC²ÔÄôr‚'6VvÖVçFVB7V&Ö—76–öã¢6VvÖVçB—27V&Ö—GFVB&WGvVVâ76W2WfW'’VBG&w2„DÕôådåô5UBÂÒöfb’"Âb“²Ò&WGW&âc²Ð§7FF–2V–çCcE÷Buö7WEöG&sÂuö7WE÷7V"Âuö7WG2Âuö6÷•ö7WG3°§7FF–2fö–B7V%ö7WB‡fö–B’°¢–çBâÒ7WEöG&w2‚“°¢–b‚uö6"ÇÂuöVæ2’&WGW&ã°¢–b†uö7WE÷7V"Òu÷7V%÷6W&–Â’²uö7WE÷7V"Òu÷7V%÷6W&–Ã²uö7WEöG&sÒu÷7FEöG&s²Òòò6÷VçBg&öÒF†R7F'BöbV6‚7V&Ö—76–öà¢–b‚‚uö7WEö6÷’bb†âÃÒÇÂu÷7FEöG&rÒuö7WEöG&sÂ‡V–çCcE÷B–â’’ÇÂuö†ö—7EöâÇÂuöFVfW%öæ7BÇÂuöFVfW%÷VæBÇÂuög¢æ7F—fRÇÂu÷'EöÖW&vVB’&WGW&ã°¢–b†uö7WEö6÷’’uö6÷•ö7WG2²³°¢uö7WEö6÷’Ò°¢–b†uö&Æ—B’²Væ5ö%öVæB†uö&Æ—B“²uö&Æ—BÒæ–Ã²Ð¢WöfÇW6‚‚“°¢V–çCcE÷BbÒFöÖ–5öÆöB‚fuö7W%ög&ÖR“²wU÷Fuö&Vv–â†b“°¢V–çC3%÷B²Òu÷6Vr²³²6Vuö6Æ²§62ÒåTÄÃ²òò&W÷'G2Væ6öFVB6òf"¶VWF†—26VvÖVçBw2uRF–ÖW2‡6Vu÷G2¢–b‡6Vu÷G2‚’’²–b‚u÷66Æ²’²u÷66Æ²Ò6ÆÆö2ƒÂ6—¦Vöb¦u÷66Æ²“²FöÖ–5÷7F÷&R‚fu÷66Æ²Óç&Vg2Â“²Ò62Òu÷66Æ³²FöÖ–5öfWF6…öFB‚g62Óç&Vg2Â“²Ð¢ÕDÃD6öÖÖ—D÷F–öç2¦òÒ´ÕDÃD6öÖÖ—D÷F–öç2æWuÓ°¢¶òFDfVVF&6´†æFÆW#¥â†–CÄÕDÃD6öÖÖ—DfVVF&6³â6"’°¢wU÷FuöFöæR†bÂ6"äuU7F'EF–ÖRÂ6"äuTVæEF–ÖR“°¢–b‡62’²–b†²Â4Tuô4Äµôâ’²FöÖ–5÷7F÷&R‚g62ÓçE¶µÕ³ÒÂwU÷5öwVW7Eöç2†6"äuU7F'EF–ÖR’“²FöÖ–5÷7F÷&R‚g62ÓçE¶µÕ³ÒÂwU÷5öwVW7Eöç2†6"äuTVæEF–ÖR’“²Ò6Vuö6Æµ÷WB‡62“²Ð¢–b†6"æW'&÷"’²FöÖ–5÷7F÷&R‚fuöÆ7Eö6öÖÖ—EöW'&÷"Â“²7FF–2–çB³²–b†²²²Â‚’ÔÄôr‚%µ$TEÒ6VvÖVçB6öÖÖæB'VffW"W†V7WF–öâW'&÷#¢W2"Â6"æW'&÷"æFW67&—F–öâåUDc…7G&–ær“²Ð¢ÕÓ°¢&W5ö6öÖÖ—B‚“°¢¶uö6"VæD6öÖÖæD'VffW%Ó°¢&V5ö6öÖÖ—B†uö6"Âò“°¢–b‡&V5ööâ‚’’&V5÷V&Æ—6‚‚“²òòF†R6VvÖVçB&V6†W2F†RVæ6öFRF‡&VBæ÷rÂæ÷Bv†Vâ—G2&Æö6²öb&V6÷&G2f–ÆÇ0¢uö6"Ò&V5ö6%öæWr‚“°¢¶uö6"&Vv–ä6öÖÖæD'VffW%v—F„ÆÆö6F÷#¦uöÆÆö3E¶uöÆÆö5ö7W%ÕÓ²òòöæRÆÆö6F÷"Væ6öFW26WfW&Â6öÖÖæB'VffW'2–â6WVVæ6S²—B—2&W6WBöæÇ’öæ6R—G2fVæ6R‡F†RÆ7B6VvÖVçB’†276VBÂv†VâÆFW"7V&Ö—76–öâ7F'G0¢uö7WEöG&sÒu÷7FEöG&s°¢–b‚uö7WG2²²’ÔÄôr‚'6VvÖVçFVB7V&Ö—76–öã¢f—'7B6VvÖVçB7V&Ö—GFVBBg&ÖRVÆÇR72VÆÇR"Â‡Vç6–væVBÆöærÆöær–bÂ‡Vç6–væVBÆöærÆöær–u÷7FE÷72“°§Ð§V–çCcE÷B×FÅö7WEö6÷VçB‡fö–B’²&WGW&âuö7WG3²Ð§V–çCcE÷B×FÅö6÷•ö7WEö6÷VçB‡fö–B’²&WGW&âuö6÷•ö7WG3²Ð ¢òòÒÒÒÒÒÒÒÒÒÒ6ÆV'2ÆöB7F–öâ6ææ÷BW‡&W72ÒÒÒÒÒÒÒÒÒÐ¢òò6öÆ÷W"6ÆV"—77VVB–ç6–FRâ÷Vâ&VæFW"72Âv—F‚'F–Â6†ææVÂÖ6²÷"v—F‚66—76÷"6ÖÆÆW"F†âF†RF&vWBÂæBFWF‚ò7FVæ6–Â6ÆV"v—F‚66—76÷"–ç6–FRà¢òò÷Vâ72…FV'2öbF†R¶–ævFöÒ—77VW2ÆÂöbF†VÒ’&RG&vâ2öæRG&–ævÆR÷fW"F†Rv†öÆR72–âF†R&VæFW"Væ6öFW"â—Bw&—FW2F†R6ÆV"fÇVR–çFòF†RÖ6¶VB6†ææVÇ2ö`¢òòF†BGF6†ÖVçBÂ÷"F†RFWF‚òF†R7FVæ6–Â&—G2öbF†R6ÆV"w2w&—FRÖ6²Â–ç6–FRF†R66—76÷"Âv—F‚F†RFWF‚FW7BÅt•2æBæòf—6–&–Æ—G’6÷VçF–ærâF†RfÇVR—2&VBg&öÐ¢òòF†Rg&ÖR&VæF‡&÷Vv‚F†RG&–ævÆRw2÷vâ&wVÖVçBF&ÆW2‡F†RG&w2ru÷gBòuögB&RWB&6²gFW"—B’Â6òF†R—VÆ–æW2FWVæBöæÇ’öâF†R72f÷&ÖG2ÂF†R6Æ÷BæBF†P¢òòÖ6²â6ÆV'2F†RÆöB7F–öâW‡&W76W2¶VW—BâFFVBg&öÒÖ4æF—fTvÖW2r6ÆV%÷VBò6ÆV%öG5÷VB†v—F‡V"æ6öÒöÇ“SC’ôÖ4æF—fTvÖW2ÂÔ•B’à§7FF–2–çBeö–çEö¶–æB„ÕDÅ—†VÄf÷&ÖBb’²òòF†Rg&vÖVçB÷WGWBG—R6öÆ÷W"GF6†ÖVçBF¶W3¢fÆöB†Ç6òæ÷&ÖÆ—¦VB’ÂV–çBÂ"–ç@¢7v—F6‚†b’°¢66RÕDÅ—†VÄf÷&ÖE#…V–çC¢66RÕDÅ—†VÄf÷&ÖE#eV–çC¢66RÕDÅ—†VÄf÷&ÖE#3%V–çC¢66RÕDÅ—†VÄf÷&ÖE$s…V–çC¢66RÕDÅ—†VÄf÷&ÖE$seV–çC¢66RÕDÅ—†VÄf÷&ÖE$s3%V–çC ¢66RÕDÅ—†VÄf÷&ÖE$t$…V–çC¢66RÕDÅ—†VÄf÷&ÖE$t#%V–çC¢66RÕDÅ—†VÄf÷&ÖE$t$eV–çC¢66RÕDÅ—†VÄf÷&ÖE$t$3%V–çC¢&WGW&â°¢66RÕDÅ—†VÄf÷&ÖE#…6–çC¢66RÕDÅ—†VÄf÷&ÖE#e6–çC¢66RÕDÅ—†VÄf÷&ÖE#3%6–çC¢66RÕDÅ—†VÄf÷&ÖE$s…6–çC¢66RÕDÅ—†VÄf÷&ÖE$se6–çC¢66RÕDÅ—†VÄf÷&ÖE$s3%6–çC ¢66RÕDÅ—†VÄf÷&ÖE$t$…6–çC¢66RÕDÅ—†VÄf÷&ÖE$t$e6–çC¢66RÕDÅ—†VÄf÷&ÖE$t$3%6–çC¢&WGW&â#°¢FVfVÇC¢&WGW&â°¢Ð§Ð§7FF–2–CÄÕDÄÆ–'&'“â6Ç%öÆ–"‡fö–B’²òò6Ç%÷g2‡F†RG&–ævÆRÂFWF‚g&öÒ'VffW"’æB6Ç%ög5óÇ6Æ÷CåóÆfÆöGÇV–çGÆ–çCâ‡F†B6öÆ÷W"6Æ÷Bw2fÇVRg&öÒ'VffW"“¢FÒövg‚ö×6Âö6ÆV"æÖWFÀ¢7FF–2–CÄÕDÄÆ–'&'“âÆ–#²7FF–2–çBf–ÆVC°¢–b†Æ–"ÇÂf–ÆVB’&WGW&âÆ–#°¢å4W'&÷"¦RÒæ–Ã²Æ–"Ò†÷7EöÆ–"‚&6ÆV""ÂfR“°¢–b‚Æ–"’²f–ÆVBÒ²ÔÄôr‚%µ$TEÒ6ÆV"×G&–ævÆR6†FW'3¢æòÆ–'&'“¢W2"ÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“²Ð¢&WGW&âÆ–#°§Ð§G—VFVb7G'V7B²–çC3%÷B6Æ÷BÂ¶–æBÂÖ6³²V–çC3%÷B6f×E³…ÒÂFf×BÂ6f×BÂ6×ÆW3²Ò6Ç%ö¶W“²òò6Æ÷BÓ¢FWF‚ò7FVæ6–Â6ÆV"†æòg&vÖVçBgVæ7F–öâ§7FF–2–CÄÕDÅ&VæFW%—VÆ–æU7FFSâ6Ç%÷6ò†6öç7B6Ç%ö¶W’¦²’°¢7FF–2å4×WF&ÆTF–7F–öæ'“Äå4FF¢Â–CÄÕDÅ&VæFW%—VÆ–æU7FFSãâ¦66†S²–b‚66†R’66†RÒ´å4×WF&ÆTF–7F–öæ'’F–7F–öæ'•Ó°¢å4FF¦¶BÒ´å4FFFFv—F„'—FW3¦²ÆVæwFƒ§6—¦Vöb¦µÓ²–CÄÕDÅ&VæFW%—VÆ–æU7FFSâ2Ò66†U¶¶EÓ²–b‡2’&WGW&â3°¢–CÄÕDÄÆ–'&'“âÆ–"Ò6Ç%öÆ–"‚“²–b‚Æ–"’&WGW&âæ–Ã°¢7FF–26öç7B6†"§G•³5ÒÒ²&fÆöB"Â'V–çB"Â&–çB'Ó°¢ÕDÅ&VæFW%—VÆ–æTFW67&—F÷"¦BÒ´ÕDÅ&VæFW%—VÆ–æTFW67&—F÷"æWuÓ°¢BçfW'FW„gVæ7F–öâÒ¶Æ–"æWtgVæ7F–öåv—F„æÖS¤&6Ç%÷g2%Ó²–b†²Óç6Æ÷BãÒ’Bæg&vÖVçDgVæ7F–öâÒ¶Æ–"æWtgVæ7F–öåv—F„æÖS¥´å57G&–ær7G&–æuv—F„f÷&ÖC¤&6Ç%ög5òVEòW2"Â²Óç6Æ÷BÂG•¶²Óæ¶–æEÕÕÓ°¢f÷"†–çB’Ò²’Âƒ²’²²’–b†²Óæ6f×E¶•Ò’²Bæ6öÆ÷$GF6†ÖVçG5¶•Òç—†VÄf÷&ÖBÒ„ÕDÅ—†VÄf÷&ÖB–²Óæ6f×E¶•Ó²Bæ6öÆ÷$GF6†ÖVçG5¶•Òçw&—FTÖ6²ÒÕDÄ6öÆ÷%w&—FTÖ6´æöæS²ÒòòF†R÷F†W"GF6†ÖVçG27F’2F†W’&P¢–b†²Óç6Æ÷BãÒ’Bæ6öÆ÷$GF6†ÖVçG5¶²Óç6Æ÷EÒçw&—FTÖ6²Ò„ÕDÄ6öÆ÷%w&—FTÖ6²’‚†²ÓæÖ6²bòÕDÄ6öÆ÷%w&—FTÖ6µ&VB¢’Â†²ÓæÖ6²b"òÕDÄ6öÆ÷%w&—FTÖ6´w&VVâ¢’Â†²ÓæÖ6²bBòÕDÄ6öÆ÷%w&—FTÖ6´&ÇVR¢’Â†²ÓæÖ6²b‚òÕDÄ6öÆ÷%w&—FTÖ6´Ç†¢’“°¢BæFWF„GF6†ÖVçE—†VÄf÷&ÖBÒ„ÕDÅ—†VÄf÷&ÖB–²ÓæFf×C²Bç7FVæ6–ÄGF6†ÖVçE—†VÄf÷&ÖBÒ„ÕDÅ—†VÄf÷&ÖB–²Óç6f×C²Bç&7FW%6×ÆT6÷VçBÒ²Óç6×ÆW2ò²Óç6×ÆW2¢°¢å4W'&÷"¦RÒæ–Ã²2Ò¶uöFWbæWu&VæFW%—VÆ–æU7FFUv—F„FW67&—F÷#¦BW'&÷#¢fUÓ°¢–b‚2’²7FF–2–çBã²–b†â²²ÂB’ÔÄôr‚%µ$TEÒ6ææ÷B7&VFRF†R6ÆV"×G&–ævÆR—VÆ–æR‡6Æ÷BVB6öÆ÷"f÷&ÖBWRFWF‚WR7FVæ6–ÂWR6×ÆW2WR“¢W3²F†—26ÆV"—26¶—VB"Â²Óç6Æ÷BÂ²Óç6Æ÷BãÒò²Óæ6f×E¶²Óç6Æ÷EÒ¢Â²ÓæFf×BÂ²Óç6f×BÂ²Óç6×ÆW2ÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“²&WGW&âæ–Ã²Ð¢66†U¶¶EÒÒ3²&WGW&â3°§Ð§7FF–2–CÄÕDÄFWF…7FVæ6–Å7FFSâ6Ç%öG72†–çBGrÂVç6–væVB6Ò’²òòFWF‚Åt•2Âw&—GFVâv†VâGs²7FVæ6–ÂÅt•2(i"$UÄ4RF‡&÷Vv‚w&—FRÖ6²6Ò†æò7FVæ6–ÂFW7Bv†Vâ¢7FF–2–CÄÕDÄFWF…7FVæ6–Å7FFSâ5³%Õ³#SeÓ°¢–b†5¶GuÕ·6ÕÒ’&WGW&â5¶GuÕ·6ÕÓ°¢ÕDÄFWF…7FVæ6–ÄFW67&—F÷"¦BÒ´ÕDÄFWF…7FVæ6–ÄFW67&—F÷"æWuÓ²BæFWF„6ö×&TgVæ7F–öâÒÕDÄ6ö×&TgVæ7F–öäÇv—3²BæFWF…w&—FTVæ&ÆVBÒGrÒ°¢–b‡6Ò’²ÕDÅ7FVæ6–ÄFW67&—F÷"§7BÒ´ÕDÅ7FVæ6–ÄFW67&—F÷"æWuÓ²7Bç7FVæ6–Ä6ö×&TgVæ7F–öâÒÕDÄ6ö×&TgVæ7F–öäÇv—3²7Bç7FVæ6–Äf–ÇW&T÷W&F–öâÒ7BæFWF„f–ÇW&T÷W&F–öâÒÕDÅ7FVæ6–Ä÷W&F–öä¶VW°¢7BæFWF…7FVæ6–Å74÷W&F–öâÒÕDÅ7FVæ6–Ä÷W&F–öå&WÆ6S²7Bç&VDÖ6²Ò†fc²7Bçw&—FTÖ6²Ò6Ó²Bæg&öçDf6U7FVæ6–ÂÒBæ&6´f6U7FVæ6–ÂÒ7C²Ð¢&WGW&â5¶GuÕ·6ÕÒÒ¶uöFWbæWtFWF…7FVæ6–Å7FFUv—F„FW67&—F÷#¦EÓ°§Ð§7FF–2–çB6Ç%÷66—76÷"„ÕDÅ66—76÷%&V7B§62’²òòF†R6ÆV"w266—76÷"–âF†R÷Vâ72Â6Æ×VB2f÷"G&w3²ÒV×G¢å5T–çFVvW"GrÒu÷72çGrÂF‚Òu÷72çFƒ²§62Ò„ÕDÅ66—76÷%&V7B—³ÂÂGrÂF‡Ó°¢–b‚"æ†5÷62’&WGW&âGrbbFƒ°¢ÆöærƒÒ"ç62ç‚Âò¢"ç62ç‚Â“Ò"ç62ç’Âò¢"ç62ç’ÂƒÒ†Æöær•"ç62ç‚²"ç62çrÂ“Ò†Æöær•"ç62ç’²"ç62æƒ°¢–b‡ƒâ†Æöær—Gr’ƒÒ†Æöær—Gs²–b‡“â†Æöær—F‚’“Ò†Æöær—Fƒ°¢–b‡ƒÃÒƒÇÂ“ÃÒ“’&WGW&â°¢§62Ò„ÕDÅ66—76÷%&V7B—²„å5T–çFVvW"—ƒÂ„å5T–çFVvW"—“Â„å5T–çFVvW"’‡ƒÒƒ’Â„å5T–çFVvW"’‡“Ò“—Ó²&WGW&â°§Ð§7FF–2fö–B6Ç%÷G&’†–CÄÕDÃE&VæFW$6öÖÖæDVæ6öFW#âVâÂ–CÄÕDÅ&VæFW%—VÆ–æU7FFSâ2Â–CÄÕDÄFWF…7FVæ6–Å7FFSâG2ÂÕDÅ66—76÷%&V7B62Â6öç7BV–çC3%÷BfÅ³EÒÂfÆöB¢ÂV–çC3%÷B7&VbÂå5T–çFVvW"GrÂå5T–çFVvW"F‚’°¢7FF–2–CÄÕDÃD&wVÖVçEF&ÆSâgBÂgC°¢–b‚gB’²å4W'&÷"¦RÒæ–Ã²ÕDÃD&wVÖVçEF&ÆTFW67&—F÷"¦BÒ´ÕDÃD&wVÖVçEF&ÆTFW67&—F÷"æWuÓ²BæÖ„'VffW$&–æD6÷VçBÒ°¢gBÒ&V5÷F&ÆR…¶uöFWbæWt&wVÖVçEF&ÆUv—F„FW67&—F÷#¦BW'&÷#¢fUÒ“²gBÒgBò&V5÷F&ÆR…¶uöFWbæWt&wVÖVçEF&ÆUv—F„FW67&—F÷#¦BW'&÷#¢fUÒ’¢æ–Ã°¢–b‚gB’²gBÒæ–Ã²7FF–2–çBã²–b‚â²²’ÔÄôr‚%µ$TEÒ6ææ÷B7&VFRF†R6ÆV"×G&–ævÆR&wVÖVçBF&ÆS¢W3²F†—26ÆV"—26¶—VB"ÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“²&WGW&ã²ÒÐ¢–CÄÕDÄ'VffW#â#²6—¦U÷BòÒ&VæöÆÆö2ƒ3"Âf"“²V–çC…÷B§Ò‡V–çC…÷B¢–"æ6öçFVçG2²ó²ÖVÖ7’‡ÂfÂÂb“²ÖVÖ7’‡²bÂg¢ÂB“²òò6öÆ÷W"BÂFWF‚B`¢¶gB6WDFG&W73¦"æwTFG&W72²òD–æFWƒ£Ó²·gB6WDFG&W73¦"æwTFG&W72²ò²bD–æFWƒ£Ó°¢¶Vâ6WD&wVÖVçEF&ÆS§gBE7FvW3¤ÕDÅ&VæFW%7FvUfW'FW…Ó²¶Vâ6WD&wVÖVçEF&ÆS¦gBE7FvW3¤ÕDÅ&VæFW%7FvTg&vÖVçEÓ°¢Væ5÷%÷6ò†VâÂ2“²uöVæ5÷6òÒæ–Ã²Væ5÷%öG72†VâÂG2“²uöVæ5öG72Òæ–Ã²òòF†RæW‡BG&r6WG2—G2÷vâv–à¢–b†uöVæ5÷f—2’²Væ5÷%÷f—6–&–Æ—G’†VâÂÕDÅf—6–&–Æ—G•&W7VÇDÖöFTF—6&ÆVBÂ“²uöVæ5÷f—2Ò²Òòòæ÷B6÷VçFVB'’âö66ÇW6–öâVW'“²F†RæW‡B6÷VçFVBG&rGW&ç26÷VçF–ær&6²öà¢Væ5÷%÷f–Ww÷'B†VâÂ„ÕDÅf–Ww÷'B—³ÂÂ†F÷V&ÆR—GrÂ†F÷V&ÆR—F‚ÂÂÒ“²Væ5÷%÷66—76÷"†VâÂ62“°¢Væ5÷%ö7VÆÂ†VâÂÕDÄ7VÆÄÖöFTæöæR“²Væ5÷%öf–ÆÂ†VâÂÕDÅG&–ævÆTf–ÆÄÖöFTf–ÆÂ“²Væ5÷%öFWF…ö&–2†VâÂÂÂ“²Væ5÷%öFWF…ö6Æ—†VâÂ”U2“²Væ5÷%÷7FVæ6–Å÷&Vb†VâÂ7&VbÂ7&Vb“²uöVBæö²Ò°¢Væ5÷%öG&r†VâÂÕDÅ&–Ö—F—fUG—UG&–ævÆRÂÂ2“°¢¶Vâ6WD&wVÖVçEF&ÆS¦u÷gBE7FvW3¤ÕDÅ&VæFW%7FvUfW'FW…Ó²¶Vâ6WD&wVÖVçEF&ÆS¦uögBE7FvW3¤ÕDÅ&VæFW%7FvTg&vÖVçEÓ°§Ð§7FF–2fö–B6ÆV%÷FW…÷G&’†–CÄÕDÅFW‡GW&SâBÂVç6–væVBÇbÂVç6–væVBÃÂ6öç7Bö6ÆV%÷FW‚¦2’°¢–çBÖ6²Ò2ÓæÖ6²b×FÅ÷eö6†Ö6²‡Bç—†VÄf÷&ÖB“²–b‚Ö6²’&WGW&ã°¢å5T–çFVvW"rÒÔ‚‡Bçv–GF‚ãâÇbÂR’Â‚ÒÔ‚‡Bæ†V–v‡BãâÇbÂR“°¢ÆöærƒÒÔ‚†2Óç"ç†öfg6WBÂ’Â“ÒÔ‚†2Óç"ç–öfg6WBÂ“°¢ÆöærƒÒÔ”â‚†Æöær–2Óç"ç†öfg6WB²2Óç"çv–GF‚Â†Æöær—r’Â“ÒÔ”â‚†Æöær–2Óç"ç–öfg6WB²2Óç"æ†V–v‡BÂ†Æöær–‚“°¢–b‡ƒÃÒƒÇÂ“ÃÒ“’&WGW&ã°¢–çB¶–æBÒeö–çEö¶–æB‡Bç—†VÄf÷&ÖB“°¢6Ç%ö¶W’²Ò³Ó²²æ¶–æBÒ¶–æC²²æÖ6²ÒÖ6³²²æ6f×E³ÒÒBç—†VÄf÷&ÖC²²ç6×ÆW2ÒBç6×ÆT6÷VçC°¢–CÄÕDÅ&VæFW%—VÆ–æU7FFSâ2Ò6Ç%÷6ò‚f²“²–CÄÕDÄFWF…7FVæ6–Å7FFSâG2Ò6Ç%öG72ƒÂ“²–b‚2ÇÂG2’&WGW&ã°¢V–çC3%÷Be³EÓ°¢f÷"†–çB’Ò²’ÂC²’²²’°¢–b†2Óæ—5÷V–çBbb¶–æB’²e¶•ÒÒ2Óæ2çU¶•Ó²6öçF–çVS²Ð¢fÆöBbÒ2Óæ—5÷V–çBò†fÆöB–2Óæ2çU¶•Ò¢2Óæ2æe¶•Ó²–çC3%÷B6“°¢–b†¶–æBÓÒ’e¶•ÒÒbâò†bÂC#“C“csCæbò‡V–çC3%÷B–b¢†fffffffgR’¢°¢VÇ6R–b†¶–æBÓÒ"’²6’Ò†bÓÒb’ò¢bãÒ#CsCƒ3S#æbò”åC3%ôÔ‚¢bÃÒÓ#CsCƒ3cC‚æbò”åC3%ôÔ”â¢†–çC3%÷B–c²ÖVÖ7’‚ge¶•ÒÂg6’ÂB“²Ð¢VÇ6RÖVÖ7’‚ge¶•ÒÂfbÂB“°¢Ð¢'öVæB‚“²–b†uö&Æ—B’²Væ5ö%öVæB†uö&Æ—B“²uö&Æ—BÒæ–Ã²Ð¢f÷"†–çB¢ÒÂç¢Ò2Óç"æFWF‚ò2Óç"æFWF‚¢²¢Âç£²¢²²’°¢ÕDÃE&VæFW%74FW67&—F÷"§'Ò´ÕDÃE&VæFW%74FW67&—F÷"æWuÓ°¢'æ6öÆ÷$GF6†ÖVçG5³ÒçFW‡GW&RÒC²'æ6öÆ÷$GF6†ÖVçG5³ÒæÆWfVÂÒÇc°¢–b‡BçFW‡GW&UG—RÓÒÕDÅFW‡GW&UG—S4B’'æ6öÆ÷$GF6†ÖVçG5³ÒæFWF…ÆæRÒ2Óç"ç¦öfg6WB²£°¢VÇ6R'æ6öÆ÷$GF6†ÖVçG5³Òç6Æ–6RÒÃ²2Óç"ç¦öfg6WB²£°¢'æ6öÆ÷$GF6†ÖVçG5³ÒæÆöD7F–öâÒÕDÄÆöD7F–öäÆöC²'æ6öÆ÷$GF6†ÖVçG5³Òç7F÷&T7F–öâÒÕDÅ7F÷&T7F–öå7F÷&S°¢6W%övFR‚“²–CÄÕDÃE&VæFW$6öÖÖæDVæ6öFW#âVâÒVæ5÷%ö&Vv–â‡'“°¢6Ç%÷G&’†VâÂ2ÂG2Â„ÕDÅ66—76÷%&V7B—·ƒÇ“Çƒ×ƒÇ“×“ÒÂbÂÂÂrÂ‚“²Væ5÷%öVæB†Vâ“°¢Ð§Ð§7FF–2fö–B6Ç%öæ÷FR†–çB6Æ÷BÂ–çBFWF‚Â–çB7FVæ6–Â’²òò2VF5öæ÷FUöG&rf÷"G&s¢F†RGF6†ÖVçBv2w&—GFVâ‡¦W&ò×6÷W&6R6¶—ÂVçw&—GFVâÖGF6†ÖVçB7F÷&R¢–b‡6Æ÷BãÒbb"ç'Bæ6öÆ÷%·6Æ÷EÒ’²§6¶—öFVÂ…"ç'Bæ6öÆ÷%·6Æ÷EÒÓç6W&–Â“²u÷§6¶—ö6æEööµ·6Æ÷EÒÒ²Ð¢f÷"†–çB’Ò²’Âuöç6C²’²²’²6EöGB§BÒfu÷6E¶•Ó°¢–b‚‡BÓæFWF‚ÓÒbb6Æ÷BãÒbbBÓæ–G‚ÓÒ6Æ÷B’ÇÂ‡BÓæFWF‚ÓÒbbFWF‚’ÇÂ‡BÓæFWF‚ÓÒ"bb7FVæ6–Â’’²BÓçw&÷FRÒBÓç&7BÒ²BÓçwv‡’Ò#²ÒÐ§Ð§7FF–2V–çCcE÷Buö6Ç%÷G&•³%Ó²òò6ÆV'2G&vâ2G&–ævÆS¢6öÆ÷W"ÂFWF‚ò7FVæ6–À§7FF–2fö–B6ÆV%ö6öÆ÷%÷G&’†6öç7Bö6ÆV%ö6öÆ÷"¦2’°¢–çB6Æ÷BÒ2Óæ–æFW‚ÂÖ6²Ò2ÓæÖ6²b†c°¢–b‚Ö6²ÇÂ6Æ÷BÂÇÂ6Æ÷BãÒ‚ÇÂ6Æ÷BãÒ"ç'BæâÇÂ"ç'Bæ6öÆ÷%·6Æ÷EÒ’&WGW&ã°¢–CÄÕDÃE&VæFW$6öÖÖæDVæ6öFW#âVâÒ'ö&Vv–â‚“²òò6ÆV"÷WG6–FR72÷Vç2—B2G&rv÷VÆB‡VæF–ær6ÆV'2ÆæBf—'7B¢–b†Vâbbu÷72æ6f×E·6Æ÷EÒbbuöVæ5öÖöFR’²'öVæB‚“²VâÒ'ö&Vv–â‚“²Òòò÷VæVBv—F†÷WBF†—2GF6†ÖVçB†6ÆV"ÖöæÇ’òD$E"ÖöFW2“¢&V÷Vâv—F‚ÆÂöbF†VÐ¢–b‚Vâ’&WGW&ã°¢–çBBÒ6Æ÷C°¢–b‚u÷72æ6f×E¶EÒbb'EöGW÷6Æ÷B‚e"ç'BÂ6Æ÷B’’òò6Æ÷BGF6†VBGv–6R—2GF6†VBöæÇ’öæ6R‡'EöGW÷6Æ÷B“¢6ÆV"F†Rf—'7BöæP¢f÷"†–çB¢Ò²¢Â6Æ÷C²¢²²’²6öç7BFW…÷B¦Ò"ç'Bæ6öÆ÷%¶¥ÒÂ¦"Ò"ç'Bæ6öÆ÷%·6Æ÷EÓ²–b‚ÇÂÓç6W&–ÂÒ"Óç6W&–ÂÇÂu÷72æ6f×E¶¥Ò’6öçF–çVS°¢6öç7Bf–Wu÷B§fÒe"ç'Bæ7f–Wu¶¥ÒÂ§f"Òe"ç'Bæ7f–Wu·6Æ÷EÓ²–çB†Ò"ç'Bæ†5ö7f–Wu¶¥ÒÂ†"Ò"ç'Bæ†5ö7f–Wu·6Æ÷EÓ°¢–b‚††bbfÓç6WEöÆWfVÇ2òfÓæ&6UöÆWfVÂ¢’ÓÒ††"bbf"Óç6WEöÆWfVÇ2òf"Óæ&6UöÆWfVÂ¢’bb††bbfÓç6WEöÆ–W'2òfÓæÖ–åöÆ–W"¢’ÓÒ††"bbf"Óç6WEöÆ–W'2òf"ÓæÖ–åöÆ–W"¢’’²BÒ£²'&V³²ÒÐ¢–b‚u÷72æ6f×E¶EÒ’²B‚&2VÆÇR6öÆ÷"Ö6ÆV"G&–ævÆR6¶—VC¢6Æ÷BVB—2æ÷B–âF†—272"Âu÷7FE÷72Â6Æ÷B“²&WGW&ã²Ð¢ÕDÅ66—76÷%&V7B63²–b‚6Ç%÷66—76÷"‚g62’’²B‚&2VÆÇR6öÆ÷"Ö6ÆV"G&–ævÆR6¶—VC¢V×G’66—76÷"6Æ÷BVB"Âu÷7FE÷72Â6Æ÷B“²&WGW&ã²Ð¢–çB¶–æBÒeö–çEö¶–æB‚„ÕDÅ—†VÄf÷&ÖB–u÷72æ6f×E¶EÒ“°¢6Ç%ö¶W’³²ÖV×6WB‚f²ÂÂ6—¦Vöb²“²²ç6Æ÷BÒC²²æ¶–æBÒ¶–æC²²æÖ6²ÒÖ6³²ÖVÖ7’†²æ6f×BÂu÷72æ6f×BÂ6—¦Vöb²æ6f×B“²²æFf×BÒu÷72æFf×C²²ç6f×BÒu÷72ç6f×C²²ç6×ÆW2Òu÷72ç6×ÆW3°¢–CÄÕDÅ&VæFW%—VÆ–æU7FFSâ2Ò6Ç%÷6ò‚f²“²–CÄÕDÄFWF…7FVæ6–Å7FFSâG2Ò6Ç%öG72ƒÂ“²–b‚2ÇÂG2’&WGW&ã°¢V–çC3%÷Be³EÓ°¢f÷"†–çB’Ò²’ÂC²’²²’²fÆöBbÒ2Óç&v&¶•Ó²–çC3%÷B6“²òò–çFVvW"GF6†ÖVçG2F¶RF†RfÇVR6öçfW'FVBÂ2ÆöBÖ7F–öâ6ÆV"FöW0¢–b†¶–æBÓÒ’e¶•ÒÒbâò†bÂC#“C“csCæbò‡V–çC3%÷B–b¢†fffffffgR’¢°¢VÇ6R–b†¶–æBÓÒ"’²6’Ò†bÓÒb’ò¢bãÒ#CsCƒ3S#æbò”åC3%ôÔ‚¢bÃÒÓ#CsCƒ3cC‚æbò”åC3%ôÔ”â¢†–çC3%÷B–c²ÖVÖ7’‚ge¶•ÒÂg6’ÂB“²Ð¢VÇ6RÖVÖ7’‚ge¶•ÒÂfbÂB“²Ð¢6Ç%÷G&’†VâÂ2ÂG2Â62ÂbÂÂÂu÷72çGrÂu÷72çF‚“²6Ç%öæ÷FR†BÂÂ“°¢–b‚uö6Ç%÷G&•³Ò²²’ÔÄôr‚&Ö–B×72ò'F–ÂÖÖ6²ò'F–Â×66—76÷"6öÆ÷"6ÆV#¢G&vâ2gVÆÂ×72G&–ævÆR†f—'7B66R72VÆÇR6Æ÷BVBÖ6²R7‚66—76÷"VÇRÂVÇRVÇW‚VÇR’"Âu÷7FE÷72ÂBÂÖ6²À¢‡Vç6–væVBÆöær—62ç‚Â‡Vç6–væVBÆöær—62ç’Â‡Vç6–væVBÆöær—62çv–GF‚Â‡Vç6–væVBÆöær—62æ†V–v‡B“°¢B‚&2VÆÇR6öÆ÷"6ÆV"G&vâ2G&–ævÆR6Æ÷BVB6W#ÒVÆÇRÖ6³ÒR7‚66—76÷#ÒVÇRÂVÇRÂVÇRÂVÇR6öÆ÷#ÒVrÂVrÂVrÂVr"Âu÷7FE÷72ÂBÂ‡Vç6–væVBÆöærÆöær•"ç'Bæ6öÆ÷%¶EÒÓç6W&–ÂÂÖ6²À¢‡Vç6–væVBÆöær—62ç‚Â‡Vç6–væVBÆöær—62ç’Â‡Vç6–væVBÆöær—62çv–GF‚Â‡Vç6–væVBÆöær—62æ†V–v‡BÂ2Óç&v&³ÒÂ2Óç&v&³ÒÂ2Óç&v&³%ÒÂ2Óç&v&³5Ò“°§Ð§7FF–2fö–B6ÆV%öG5÷G&’†6öç7Bö6ÆV%öG2¦2’°¢Vç6–væVBvçE÷2Ò‡Vç6–væVB–2Óç7FVæ6–ÅöÖ6²b†fc°¢–b‚2ÓæFWF…öÖ6²bbvçE÷2’&WGW&ã°¢–CÄÕDÃE&VæFW$6öÖÖæDVæ6öFW#âVâÒ'ö&Vv–â‚“°¢–b†VâbbuöVæ5öÖöFRbb‚†2ÓæFWF…öÖ6²bbu÷72æ†B’ÇÂ‡vçE÷2bbu÷72æ‡2’’’²'öVæB‚“²VâÒ'ö&Vv–â‚“²Òòò÷VæVBv—F†÷WBF†RFWF‚÷"7FVæ6–ÂÆæR…D$E"Æöö¶†VB“¢&V÷Vâv—F‚ÆÂGF6†ÖVçG0¢–b‚Vâ’&WGW&ã°¢–çBGrÒ2ÓæFWF…öÖ6²bbu÷72æ†C²Vç6–væVB6ÒÒu÷72æ‡2òvçE÷2¢°¢–b‚Grbb6Ò’&WGW&ã°¢ÕDÅ66—76÷%&V7B63²–b‚6Ç%÷66—76÷"‚g62’’²B‚&2VÆÇRFWF‚×7FVæ6–Â6ÆV"G&–ævÆR6¶—VC¢V×G’66—76÷""Âu÷7FE÷72“²&WGW&ã²Ð¢6Ç%ö¶W’³²ÖV×6WB‚f²ÂÂ6—¦Vöb²“²²ç6Æ÷BÒÓ²ÖVÖ7’†²æ6f×BÂu÷72æ6f×BÂ6—¦Vöb²æ6f×B“²²æFf×BÒu÷72æFf×C²²ç6f×BÒu÷72ç6f×C²²ç6×ÆW2Òu÷72ç6×ÆW3°¢–CÄÕDÅ&VæFW%—VÆ–æU7FFSâ2Ò6Ç%÷6ò‚f²“²–CÄÕDÄFWF…7FVæ6–Å7FFSâG2Ò6Ç%öG72†GrÂ6Ò“²–b‚2ÇÂG2’&WGW&ã°¢fÆöB¢Ò2ÓæFWF‚âò†2ÓæFWF‚Âò2ÓæFWF‚¢’¢²òò26ÆV$FWFƒ¢v—F†–â³ÂÒ„æâ(i"¢6öç7BV–çC3%÷Be³EÒÒ³Ó°¢6Ç%÷G&’†VâÂ2ÂG2Â62ÂbÂ¢Â‡V–çC3%÷B–2Óç7FVæ6–Âb†fbÂu÷72çGrÂu÷72çF‚“²6Ç%öæ÷FR‚ÓÂGrÂ6ÒÒ“°¢–b‚uö6Ç%÷G&•³Ò²²’ÔÄôr‚&Ö–B×7266—76÷&VBFWF‚×7FVæ6–Â6ÆV#¢G&vâ2gVÆÂ×72G&–ævÆR†f—'7B66R72VÆÇRFWF‚VB7FVæ6–ÂÖ6²R7‚66—76÷"VÇRÂVÇRVÇW‚VÇR’"Âu÷7FE÷72ÂGrÂ6ÒÀ¢‡Vç6–væVBÆöær—62ç‚Â‡Vç6–væVBÆöær—62ç’Â‡Vç6–væVBÆöær—62çv–GF‚Â‡Vç6–væVBÆöær—62æ†V–v‡B“°¢B‚&2VÆÇRFWF‚×7FVæ6–Â6ÆV"G&vâ2G&–ævÆR6W#ÒVÆÇRFWFƒÒVr‚VB’7FVæ6–ÃÒVBÖ6³ÒR7‚66—76÷#ÒVÇRÂVÇRÂVÇRÂVÇR"Âu÷7FE÷72Â"ç'BæFWF‚ò‡Vç6–væVBÆöærÆöær•"ç'BæFWF‚Óç6W&–Â¢VÆÂÂ¢ÂGrÂ2Óç7FVæ6–ÂÂ6ÒÀ¢‡Vç6–væVBÆöær—62ç‚Â‡Vç6–væVBÆöær—62ç’Â‡Vç6–væVBÆöær—62çv–GF‚Â‡Vç6–væVBÆöær—62æ†V–v‡B“°§Ð §fö–B×FÅöW†V2†6öç7BV–çC…÷B§’°¢–b‚u÷7FFUö–æ—FVB’7FFU÷&W6WB‚“°¢7FF–2–çBGV×öÆÂÒÓ²–b†GV×öÆÂÂ’GV×öÆÂÒvWFVçb‚$DÕôådåõ54ETÕôÄÂ"’ÒåTÄÃ°¢–çBG&6RÒGV×öÆÂbb76GV×‚“²òòæ÷&ÖÂÖöFRFöW2æ÷B6†V6²F†Rg&ÖR&ævRW"÷W&F–öã²F†RgVÆÂ6öÖÖæB÷6—F–öâ—2öæÇ’W6VBf÷"F†RFW6–væFVBF–væ÷7F–2g&ÖW0¢f÷"ƒ³²’°¢÷ö†G"ƒ²ÖVÖ7’‚f‚ÂÂ6—¦Vöb‚“°¢uöW†V5ö7W"Ò²òò66â7F'Bö–çBöbF†R7F÷&RÆöö¶†VC¢F†R÷W&F–öâF†BVæG2F†R&VæFW"72†–æ6ÇW6—fR¢–b‡ÓÒuöFVfW%÷VæB’²f÷"†–çBv’Ò²v’ÂC²v’²²’–b†uöFVfW%÷6Æ÷G5¶v•Òæ7F—fRbbuöFVfW%÷6Æ÷G5¶v•ÒçöVæEö÷ÓÒ’uöFVfW%÷6Æ÷G5¶v•Òæ–å÷Ò²uöFVfW%÷VæBÒåTÄÃ²Ð¢6G…÷6W‡“°¢–b‡G&6R’B‚&ò6ÖCÒW73ÒVÆÇR÷ÒWR6—¦SÒWR"ÂÂu÷7FE÷72Â‡Vç6–væVB–‚æ÷Â‡Vç6–væVB–‚ç6—¦R“°¢òòF–væ÷7F–2DÕôådåôÄÄô5$ôcÓ¢w&÷wF‚öbF†RuRÆÆö6FVB6—¦R—26†&vVBFòF†R&Wf–÷W2÷W&F–öâw2G—R‡&÷Vvƒ¢ÆÆö6F–öç2öâ÷F†W"F‡&VG2&R6÷VçFVBFöò“²F†RÖVÖ÷'’&W÷'BÆ–æRVæG2F†RF÷@¢–b†uöÆÆö7&öbÂ’uöÆÆö7&öbÒvWFVçb‚$DÕôådåôÄÄô5$ôb"’ÒåTÄÃ°¢–b†uöÆÆö7&öb’²7FF–2V–çCcE÷BÆ7C²7FF–2–çB&WbÒÓ²V–çCcE÷Bæ÷rÒuöFWbæ7W'&VçDÆÆö6FVE6—¦S²–b‡&WbãÒbb&WbÂcBbbæ÷râÆ7B’uöö'—FW5·&WeÒ³Òæ÷rÒÆ7C²Æ7BÒæ÷s²&WbÒ‚æ÷²Ð¢–b†uö†ö—7Eöâbb†ö—7E÷F¶R‡’’²–b†uög¢æ7F—fRbbÓÒuög¢ç'C%ö÷’²uög¢ç†6RÒ#²u÷'EöÖW&vVBÒ²B‚$bVÆÇRÖW&vVB72VçFW'2F†RÆFW"72w2G&w2"Âu÷7FE÷72“²Ò³Ò‚ç6—¦S²6öçF–çVS²ÒòòW&RÖ6ÆV"&VæFW"72Ç&VG’W†V7WFVBV&Ç’òF†R7F'BöbÆFW"72Ç&VG’ÖW&vVB–âòòVçFW&–ærF†RÆFW"73¢F†R6ÖR×F&vWB6öçF–çVF–öâöbF†RV&Æ–W"72VæG2†W&S²F†RÆFW"72&VF–ærF†RV&Æ–W"72w2GF6†ÖVçB—2&÷fVâFò&R6ÖR×—†VÂ'’g¥÷%öG&rÂæB—2æòÆöævW"7Æ—B'’F†R6öçF–çVF–öâ'VÆP¢6öç7Bfö–B¦Ò²ƒ°¢–b†uöæÆ2bbÆ5ö¶VW2†‚æ÷’bb†‚æ÷ÓÒõôTäBbbu÷&WEöâ’’Æ5öÆæB‚“²òò6VRÆ5ö¶VW2†7V"ÖÆ—7Bw2VæB&WGW&ç2Fò—G26ÆÆW"¢7v—F6‚†‚æ÷’°¢66RõôTäC¢uöW†V5ö7W"ÒåTÄÃ°¢–b†uö†ö—7Eöâbbu÷&WEöâbbuöç&W7B’°¢–b†uög¢æ7F—fR’g¥÷Vç6¶—‚“°¢–b†uö†ö—7Eöâ’²ÔÄôr‚%µ$TEÒW&RÖ6ÆV"72†ö—7F–æs¢VB†ö—7FVB÷W&F–öç2vW&Ræ÷B&V6†VB'’F†RVæBöbF†R7V&Ö—76–öã²6ÆV&VB‡6†÷VÆBæ÷B†Vâ’"Âuö†ö—7Eöâ“²uö†ö—7EöâÒ²Ð¢Ð¢&WGW&ã²òòâVæBÖöb×72G&–vvW&VB÷WG6–FRÆ—7B‡&W6VçBò5R&VF&6²WF2â’×W7Bæ÷B66âW‡—&VB6öÖÖæBÖVÖ÷'“¢çVÆÂ7F'Bö–çBÖVç27F÷&RWfW'—F†–æp¢66Rõô$Tt”ã¢'&V³°¢66Rõô¥TÕ¢²6öç7BV–çC…÷B§C²ÖVÖ7’‚gBÂÂ‚“²ÒC²6öçF–çVS²Ð¢66Rõô4ÄÃ¢²V–çCcE÷BC²ÖVÖ7’‚gBÂÂ‚“°¢–b†u÷&WEöâÂb’u÷&WE¶u÷&WEöåÒÒ²‚ç6—¦S²u÷&WEöâ²³²òò&WGW&â7F6²f÷"Æöö¶†VC¢gFW"7V"ÖÆ—7BVæG2Â6öçF–çVRÆöö¶–ærBF†R6ÆÆW ¢×FÅöW†V2‚†6öç7BV–çC…÷B¢—B“²u÷&WEöâÒÓ²'&V³²Ð¢66Rõõ4”täÅôUdTåC¢²föÆF–ÆRV–çC3%÷B§fÃ²V–çC3%÷BÒÂc²ÖVÖ7’‚gfÂÂÂ‚“²ÖVÖ7’‚fÒÂ†6öç7BV–çC…÷B¢–²‚ÂB“²ÖVÖ7’‚gbÂ†6öç7BV–çC…÷B¢–²"ÂB“°¢B‚%’6–væÅöWfVçB6ÖCÒWG7CÒWÖöFSÒWRfÇVSÒWR"ÂÂfÂÂÒÂb“²×FÅ÷7V%öWfVçB‡fÂÂÒÂb“²'&V³²Ð¢66Rõõ$Uõ%Eô4õTåDU#¢²V–çCcE÷BBÂC²ÖVÖ7’‚gBÂÂ‚“²ÖVÖ7’‚fBÂ†6öç7BV–çC…÷B¢–²‚Â‚“²uö6çE÷&W÷'E·Bb3Ò²³°¢B‚%&W÷'BG—SÒVÆÇR6ÖCÒWG7CÒR6ÆÇ‚"Â‡Vç6–væVBÆöærÆöær—BÂÂ‡Vç6–væVBÆöærÆöær–B“°¢–b„DÕôtÔUõ$TäDU%ô4ôäB’&5öæ÷FR†BÂBÓÒÇÂBÓÒƒò¢"“°¢–b‡BÓÒådåô4õTåDU%õE•Uõ4ÕÄU5õ54TB’²÷7B‚‡÷7Eö÷—³2ÂåTÄÂÂÂÂBÂ‡V–çC3%÷B—BÂf—5÷6æ6†÷B‚—Ò“²u÷f—5ö7W"ÒÓ²Ð¢VÇ6R×FÅ÷7V%ö6÷VçFW"†BÂ‡V–çC3%÷B—B“°¢'&V³²Ð¢66Rõõ$TäDU%ô4ôäC¢²6öç7B÷&VæFW%ö6öæB§&2Ò²òò&V6÷&FVBöæÇ’v—F‚F†R&öf–ÆR¶W’&VæFW%ö6öæB††ÆUöçfâæ2¢–b‡&2ÓæÖöFRãÒ"’²u÷&5÷6¶—Ò&2ÓæÖöFRÓÒ3²B‚$RVÆÇR&VæFW"Væ&ÆRVB"Âu÷7FE÷72Â&2ÓæÖöFRÓÒ"“²Ð¢VÇ6R²V–çCcE÷B‚ÒÂ’Ò²–çB¶æ÷vâÒ&5÷fÇVR‡&2ÓæFG"Âg‚’b&5÷fÇVR‡&2ÓæFG"²bÂg’“°¢u÷&5÷6¶—Ò¶æ÷vâbb‡&2ÓæÖöFRò‚ÓÒ’¢‚Ò’“°¢B‚$RVÆÇR6öæF—F–öæÂ&VæFW"W2FG#ÒR6ÆÇ‚fÇVW2VÆÇRòVÆÇRW2W2"Âu÷7FE÷72Â&2ÓæÖöFRò&G&r–bæ÷BWVÂ"¢&G&r–bWVÂ"Â‡Vç6–væVBÆöærÆöær—&2ÓæFG"Â‡Vç6–væVBÆöærÆöær—‚Â‡Vç6–væVBÆöærÆöær—’Â¶æ÷vâò""¢"‡Væ¶æ÷vâÂG&vâ’"Âu÷&5÷6¶—ò"(i"æ÷BG&vâ"¢""“²Ð¢'&V³²Ð¢66RõôdTä4Uõ5”ä3¢²7–æ5÷B§3²ÖVÖ7’‚g2ÂÂ‚“²2ÓæfVæ6RÒuöfVæ6R²²2ÓæfVæ6VBÒ²òòF†RfVæ6RfÇVRF†—27V&Ö—76–öâ‡6VvÖVçB’6öÖÖ—G2v—F€¢B‚%’fVæ6U÷7–æ26ÖCÒW7–æ3ÒWfVæ6SÒVÆÇR"ÂÂ‡fö–B¢—2Â‡Vç6–væVBÆöærÆöær—2ÓæfVæ6R“²'&V³²Ð¢66Rõõ$U4UEô4õTåDU#¢²V–çCcE÷BC²ÖVÖ7’‚gBÂÂ‚“²uö6çE÷&W6WE·Bb3Ò²³²B‚%&W6WBG—SÒVÆÇR6ÖCÒW"Â‡Vç6–væVBÆöærÆöær—BÂ“°¢–b‡BÓÒådåô4õTåDU%õE•Uõ4ÕÄU5õ54TB’°¢6–fFVbådåõ%Eõ$ätUõDU5@¢'%÷&W6WB‚“°¢6VæF–`¢u÷f—5ööâÒ²u÷f—5ö6÷VçF–ærÒ²f—5ö6ÆV"‚“°¢òòF†Rf—'7B&W6WB6â6öÖR–ç6–FR&VæFW"72F†Bv2&VwVâv—F†÷WBf—6–&–Æ—G’'VffW"„ö7F÷F‚G&fVÆW"¢Vç&VÂw2ö66ÇW6–öâVW&–W2Âöæ6RB&ö÷B“¢7Æ—BF†R70¢òò†W&RÂ6òF†R6öçF–çVF–öâ—2&VwVâv—F‚F†R'VffW"æB—G26×ÆW2&R6÷VçFVB‡F†W’W6VBæ÷BFò&S¢$TB&ö66ÇW6–öâVW'’6÷VçF–ær&Vvâ–â72v—F†÷WBf—6–&–Æ—G’'VffW""¢–b†uöVæ2bbuöVæ5ö†5÷f—2’²B‚$ÒVÆÇRö66ÇW6–öâVW'’6÷VçF–ær7F'G3¢&VæFW"727Æ—B†W&R"Âu÷7FE÷72“²'öVæB‚“²ÒÐ¢'&V³²Ð¢66Rõô4ÄT%ô%Tc¢²6öç7Bö6ÆV%ö'Vb¦6"Ò²×FÅ÷7V%öf–ÆÂ†6"ÓæFG"Â6"Óç6—¦RÂ6"ÓçfÇVR“²'&V³²Ð¢66Rõô4õ•ô#$#¢²V–çCcE÷B5òÂBÂã²ÖVÖ7’‚g5òÂÂ‚“²ÖVÖ7’‚fBÂ†6öç7BV–çC…÷B¢–²‚Â‚“²ÖVÖ7’‚fâÂ†6öç7BV–çC…÷B¢–²bÂ‚“°¢B‚%’6÷•ö#&"6ÖCÒW7&3ÒR6ÆÇ‚G7CÒR6ÆÇ‚'—FW3ÒVÆÇR"ÂÂ‡Vç6–væVBÆöærÆöær—5òÂ‡Vç6–væVBÆöærÆöær–BÂ‡Vç6–væVBÆöærÆöær–â“²×FÅ÷7V%ö6÷’‡5òÂBÂâ“²'&V³²Ð¢66RõõUDDUõT#¢²V–çCcE÷BBÂã²ÖVÖ7’‚fBÂÂ‚“²ÖVÖ7’‚fâÂ†6öç7BV–çC…÷B¢–²‚Â‚“°¢–b‡ÓÒuöÖG…÷7F÷’²uöÖG…öWö6‚²³²uöÖG…÷7F÷ÒåTÄÃ²Òòò—Bw&—FW2&V6÷&B÷"6÷VçBF†BÖ6¶VB6÷–W2vW&RÖFRöb†ÖG…÷'Vâ¢B‚%’WFFU÷V"6ÖCÒW7&3ÒR6ÆÇ‚G7CÒR6ÆÇ‚'—FW3ÒVÆÇRcÒVrÂVrÂVrÂVr"ÂÂ‡Vç6–væVBÆöærÆöær’‚†6öç7BV–çC…÷B¢–²b’Â‡Vç6–væVBÆöærÆöær–BÂ‡Vç6–væVBÆöærÆöær–âÀ¢âãÒBò‚†6öç7BfÆöB¢’‚†6öç7BV–çC…÷B¢–²b’•³Ò¢æbÂâãÒ‚ò‚†6öç7BfÆöB¢’‚†6öç7BV–çC…÷B¢–²b’•³Ò¢æbÂâãÒ"ò‚†6öç7BfÆöB¢’‚†6öç7BV–çC…÷B¢–²b’•³%Ò¢æbÂâãÒbò‚†6öç7BfÆöB¢’‚†6öç7BV–çC…÷B¢–²b’•³5Ò¢æb“²×FÅ÷7V%÷WFFR‚†6öç7BV–çC…÷B¢–²bÂBÂâ“²'&V³²Ð¢66Rõõ4UEõ%C¢°¢6öç7B÷6WE÷'B§"Ò°¢–b†FVfW%ö&Vv–å÷‡’’'&V³°¢–çBVæBÒ"æ6ÆV%öBÇÂ"æ6ÆV%÷3²f÷"†–çB’Ò²’Âƒ²’²²’VæBÃÒ"æ6ÆV%ö5¶•Ó°¢–b†uöVæ2bbVæBbb"æ†5÷'BbbF&G%ööâ‚’bb'E÷&ævU÷6ÖR‚e"ç'BÇ"’’°¢u÷'EöÖW&vVBÒu÷'E÷&ævUö7F—fRÒu÷'E÷&ævU÷VæF–ærÒ²"ç'BÒ§#²'&V³²Ð¢–b†uöVæ2bbuög¢æ7F—fRbbuög¢ç†6RÓÒbbg¥÷&WW6Uö÷‡’bbVæBbbuöVæ5öÖöFRÒbb"æ†5÷'BbbF&G%ööâ‚’bb'E÷6ÖR‚fuög¢ç'CÂ"’’°¢u÷'EöÖW&vVBÒ²–b‚u÷'EöÖW&vUöâ²²’ÔÄôr‚'6ÖR×F&vWB6öçF–çVF–öã¢f—'7B66R72VÆÇR"Âu÷7FE÷72“²B‚$ÒVÆÇR6ÖR×F&vWB6öçF–çVF–öâ72"Âu÷7FE÷72“²'&V³²Ð¢òòFòæ÷B6öçF–çVR&VæFW"72÷VæVBöæÇ’f÷"6ÆV#²öæRv—F‚fWvW"FWF‚ò7FVæ6–ÂGF6†ÖVçG26â&R6öçF–çVVB†6†V6²F†RÆæW2&Vf÷&RG&v–ær¢–b†uöVæ2bbuöVæ5öÖöFRÒbbVæBbb"æ†5÷'Bbb‡F&G%ööâ‚’ÇÂ‡'F5ööâ‚’bb"Óæâ’’bb'E÷6ÖR‚e"ç'BÂ"’’²u÷'EöÖW&vVBÒ²–b‚u÷'EöÖW&vUöâ²²’ÔÄôr‚'6ÖR×F&vWB6öçF–çVF–öã¢f—'7B66R72VÆÇR"Âu÷7FE÷72“²B‚$ÒVÆÇR6ÖR×F&vWB6öçF–çVF–öâ72"Âu÷7FE÷72“²"ç'BÒ§#²'&V³²Ð¢–b†uöVæ2bbu÷6Õ÷72bbVæBbbuövÖUö&'&–W"’²–çB6Ã²òòæ÷F†W"Æ–W"öbF†R6Æ–6RÖÖW&vVB72w2ÆWfVÂ‡6Õ÷66âÆææVB—B¢–b‡6Õ÷'E÷6Æ–6R‚e"ç'BÂ"Âg6Â’bb6ÂãÒu÷6ÕöÆòbb6ÂÃÒu÷6Õö†’’²"ç'BÒ§#²u÷6Õö7W"Ò6Ã²u÷6Õöæ6öçB²³°¢²fö–BFEöæ÷FUöÆ–W"†–CÄÕDÅFW‡GW&SâÂV–çCcE÷BÂå5T–çFVvW"Âå5T–çFVvW"“²FEöæ÷FUöÆ–W"‡FW…ö×FÂ‡"Óæ6öÆ÷%³Ò’Â"Óæ6öÆ÷%³ÒÓç6W&–ÂÂ"Óæ7f–Wu³Òç6WEöÆWfVÇ2ò"Óæ7f–Wu³Òæ&6UöÆWfVÂ¢Â„å5T–çFVvW"—6Â“²ÒB‚$ÒVÆÇR6Æ–6RÖÖW&vVB72vöW2öâ–âÆ–W"VB"Âu÷7FE÷72Â6Â“²'&V³²ÒÐ¢–b‡VæBbbuöVæ2bbuö6"bbÆ5ööâ‚’’Æ5÷W6‚‚“²VÇ6R'öfÇW6‚‚“²òò6ÆV'2v—F‚æòG&rv—Bf÷"F†RæW‡B72F†BGF6†W2F†V—"F&vWG2†Æ5÷W6‚¢7V%ö7WB‚“²"ç'BÒ§#²"æ†5÷'BÒ²'&V³°¢Ð¢66Rõô4ÄT%ô4ôÄõ#¢°¢6öç7Bö6ÆV%ö6öÆ÷"¦2Ò°¢–b†u÷&5÷6¶—’²B‚&2VÆÇR6öÆ÷"6ÆV"6¶—VB'’6öæF—F–öæÂ&VæFW&–ær6Æ÷BVB"Âu÷7FE÷72Â2Óæ–æFW‚“²'&V³²Ð¢òò66—76÷&VB6ÆV"Fò÷WG6–FR72v—F‚ÆÂ6†ææVÇ2…4Ô%r6ÆV'2—&Ö–BÆWfVÇ227V"×&V7FævÆW2öbCƒ9s#s"æB÷F†W"F&vWG2“¢6÷’g&öÒF†R¦W&ò'VffW"–çFòF†R66—76÷"&V7FævÆR–âF†R&WÆ’7G&VÐ¢–b‚uöVæ2bb2ÓæÖ6²ÓÒ†bbb2Óç&v&³Òbb2Óç&v&³Òbb2Óç&v&³%Òbb2Óç&v&³5Òbb"æ†5÷62bb2Óæ–æFW‚ãÒbb2Óæ–æFW‚Â"ç'Bæâbb"ç'Bæ6öÆ÷%¶2Óæ–æFW…Ð¢bb66—76÷%ögVÆÂ…"ç'Bæ6öÆ÷%¶2Óæ–æFW…ÒÂe"ç'Bæ7f–Wu¶2Óæ–æFW…ÒÂ"ç'Bæ†5ö7f–Wu¶2Óæ–æFW…Ò’bb6ÆV%÷&V7E÷¦W&ò†2Óæ–æFW‚’’'&V³°¢–b†u÷'EöÖW&vVBÇÂu÷6Õ÷72’'öVæB‚“²òò6ÆV"'&—fVB–â6öçF–çVVB&VæFW"73¢VæB—B†W&RÂæBF†R6ÆV"—2GF6†VBFòF†R7F'BöbF†RæW‡B&VæFW"722W7VÀ¢–b†2Óæ–æFW‚ÂÇÂ2Óæ–æFW‚ãÒ"ç'BæâÇÂ"ç'Bæ6öÆ÷%¶2Óæ–æFW…Ò’²$TEôôä4RƒRÂ&6öÆ÷"6ÆV"F&vWBVB–çfÆ–B"Â2Óæ–æFW‚“²'&V³²Ð¢–b†uöVæ2ÇÂ2ÓæÖ6²Ò†bÇÂ66—76÷%ögVÆÂ…"ç'Bæ6öÆ÷%¶2Óæ–æFW…ÒÂe"ç'Bæ7f–Wu¶2Óæ–æFW…ÒÂ"ç'Bæ†5ö7f–Wu¶2Óæ–æFW…Ò’’°¢òò–ç6–FR72Âv—F‚'F–ÂÖ6²÷"'F–Â66—76÷#¢G&vâ2G&–ævÆR†6ÆV%ö6öÆ÷%÷G&’¢7FF–2V–çCcE÷Bã²7FF–2–çB6ÆörÒÓ²–b†6ÆörÂ’6ÆörÒvWFVçb‚$DÕôådåô4ÄT$Äôr"’ÒåTÄÃ²–b†6Æörbbâ²²ÂC’ÔÄôr‚&6öÆ÷"6ÆV"G&vâ2G&–ævÆRg&ÖRVÆÇR72VÆÇRF&vWBVB6W#ÒVÆÇRWW‚WRÖ–CÒVBÖ6³ÒR7‚66—76÷#ÒVBÂVBÂVBÂVB6öÆ÷#ÒVrÂVrÂVrÂVr"Â‡Vç6–væVBÆöærÆöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR’Âu÷7FE÷72Â2Óæ–æFW‚À¢‡Vç6–væVBÆöærÆöær•"ç'Bæ6öÆ÷%¶2Óæ–æFW…ÒÓç6W&–ÂÂ"ç'Bæ6öÆ÷%¶2Óæ–æFW…ÒÓæBçv–GF‚Â"ç'Bæ6öÆ÷%¶2Óæ–æFW…ÒÓæBæ†V–v‡BÂuöVæ2Òæ–ÂÂ2ÓæÖ6²Â"ç62ç‚Â"ç62ç’Â"ç62çrÂ"ç62æ‚Â2Óç&v&³ÒÂ2Óç&v&³ÒÂ2Óç&v&³%ÒÂ2Óç&v&³5Ò“°¢6ÆV%ö6öÆ÷%÷G&’†2“²'&V³²Ð¢"æ6ÆV%ö5¶2Óæ–æFW…ÒÒ²"æ65¶2Óæ–æFW…ÒÒÕDÄ6ÆV$6öÆ÷$Ö¶R†2Óç&v&³ÒÂ2Óç&v&³ÒÂ2Óç&v&³%ÒÂ2Óç&v&³5Ò“°¢B‚&2VÆÇR6öÆ÷"6ÆV"&V6÷&FVB6Æ÷BVB6W#ÒVÆÇRÖ6³ÒR7‚66—76÷#ÒVC¢VBÂVBÂVBÂVB"Âu÷7FE÷72Â2Óæ–æFW‚Â‡Vç6–væVBÆöærÆöær•"ç'Bæ6öÆ÷%¶2Óæ–æFW…ÒÓç6W&–ÂÂ2ÓæÖ6²Â"æ†5÷62Â"ç62ç‚Â"ç62ç’Â"ç62çrÂ"ç62æ‚“°¢'&V³°¢Ð¢66Rõô4ÄT%ôE3¢°¢6öç7Bö6ÆV%öG2¦2Ò°¢–b†u÷&5÷6¶—’²B‚&2VÆÇRFWF‚×7FVæ6–Â6ÆV"6¶—VB'’6öæF—F–öæÂ&VæFW&–ær"Âu÷7FE÷72“²'&V³²Ð¢–b†u÷'EöÖW&vVBÇÂu÷6Õ÷72’'öVæB‚“°¢òòÆöBÖ7F–öâ6ÆV"6÷fW'2F†Rv†öÆRGF6†ÖVçBæBWfW'’7FVæ6–Â&—Bâ6ÆV"v†÷6R66—76÷"FöW2æ÷B6÷fW"F†RFWF‚F&vWBÂ÷"v†÷6R7FVæ6–ÂÖ6²—2'F–ÂÂ—2G&vâ2G&–ævÆP¢òò†6ÆV%öG5÷G&’’Â–â÷"&Vf÷&R73¢&Vf÷&RF†Rf—'7BG&ræBf÷"'F–Â7FVæ6–ÂÖ6²–â72—BW6VBFò6ÆV"F†Rv†öÆRF&vWB…"3‚¢Vç6–væVB6ÒÒ‡Vç6–væVB–2Óç7FVæ6–ÅöÖ6²b†fc°¢–b…"ç'BæFWF‚bb‚66—76÷%ögVÆÂ…"ç'BæFWF‚Âe"ç'BæGf–WrÂ"ç'Bæ†5öGf–Wr’ÇÂ‡6Òbb6ÒÒ†fb’’’²6ÆV%öG5÷G&’†2“²'&V³²Ð¢òògVÆÂ6ÆV"–âF†RÖ–FFÆRöb73¢VæBF†R72æBGF6‚F†R6ÆV"FòF†R7F'BöbF†RæW‡BöæR†6öÆ÷"GF6†ÖVçG2ÆöB2W7VÂÂ26öçF–çVVB72FöW2&÷fRv†Vâ6ÆV"'&—fW2’à¢òò—BW6VBFò&R6¶—VBÂæBÆFW"G&w2FWF‚×FW7FVBv–ç7B7FÆRFWF€¢–b†uöVæ2’°¢²7FF–2–çBöæ6S²–b‚öæ6R²²’ÔÄôr‚&Ö–B×72FWF‚×7FVæ6–Â6ÆV#¢VæF–ærF†—273²F†R6ÆV"Ö÷fW2FòF†R7F'BöbF†RæW‡B72†f—'7B66R72VÆÇR’"Âu÷7FE÷72“²Ð¢'öVæB‚“°¢Ð¢–b†2ÓæFWF…öÖ6²’²"æ6ÆV%öBÒ²"æ6BÒ2ÓæFWFƒ²Ð¢–b†2Óç7FVæ6–ÅöÖ6²’²"æ6ÆV%÷2Ò²"æ72Ò‡V–çC3%÷B–2Óç7FVæ6–Ã²Ð¢B‚&2VÆÇRFWF‚×7FVæ6–Â6ÆV"&V6÷&FVB6W#ÒVÆÇRFWFƒÒVrFWF…öÖ6³ÒVB7FVæ6–ÃÒVB7FVæ6–ÅöÖ6³ÒR7‚"Âu÷7FE÷72Â"ç'BæFWF‚ò‡Vç6–væVBÆöærÆöær•"ç'BæFWF‚Óç6W&–Â¢VÆÂÂ2ÓæFWF‚Â2ÓæFWF…öÖ6²Â2Óç7FVæ6–ÂÂ2Óç7FVæ6–ÅöÖ6²“°¢'&V³°¢Ð¢66RõôE$s¢²–b†u÷&5÷6¶—’'&V³²öG&rC²uöW†V5ö7W"Ò²uöW†V5öæW‡BÒ²‚ç6—¦S²G&r†G&u÷&V2†Â‚ç6—¦RÂgB’ÂåTÄÂ“²uöW†V5öæW‡BÒåTÄÃ²'&V³²ÒòòÆöö¶†VB7F'Bö–ç@¢66RõôE$uô”Eƒ¢²–b†u÷&5÷6¶—’'&V³²öG&uö–G‚C²uöW†V5ö7W"Ò²uöW†V5öæW‡BÒ²‚ç6—¦S²G&r„åTÄÂÂG&uö–G…÷&V2†Â‚ç6—¦RÂgB’“²uöW†V5öæW‡BÒåTÄÃ²'&V³²Ð¢òòFòæ÷BÖ&²F—'G’–bF†R7FFRF–Bæ÷B6†ævR‡F†RvÖRögFVâ&R×6VæG2F†R6ÖR&öw&Òò7FFR&Æö6²òfW'FW‚f÷&ÖB&Vf÷&RV6‚G&s²F†RF—'G’fÆröæÇ’FV6–FW2v†WF†W"Fò&RÖfWF6‚F†R—VÆ–æRòFWF‚×7FVæ6–Âö&¦V7BÂæBWVÂfÇVW2ÖVâF†R6ÖRö&¦V7B¢66Rõô$”äEõ$ôu$Ó¢²6öç7Bö&–æE÷&öw&Ò¦"Ò²7FFUö÷†‚æ÷Æ“°¢–b‡76GV×‚’’²çe&ör§"Ò2ç&örò&öuövWB…2ç&ör’¢æ–Ã°¢B‚%’&–æE÷&öw&Ò6ÖCÒW&öw&ÓÒW7FvW3ÒR7‚g3ÒW2g3ÒW2w3ÒW2"ÇÂ‡fö–B¢–"Óç&örÆ"Óç7FvW2À¢"bb"Óç6…³Òò"Óç6…³ÒÓæ¶W’¢"Ò"Â"bb"Óç6…³Òò"Óç6…³ÒÓæ¶W’¢"Ò"Ç"bb"Óç6…³%Òò"Óç6…³%ÒÓæ¶W’¢"Ò"“²Ð¢'&V³²Ð¢66Rõõ44•54õ#¢66Rõô$”äEõ5DDS¢66Rõô$”äEõdEE#¢66Rõô$”äEõe5E$TÓ¢66Rõô$”äEõd%Tc¢66Rõô$”äEõd%Tc ¢66Rõô$”äEõT$ó¢66Rõô$”äEõT$õõ4ÔÄÃ¢66Rõô$”äEõ54$ó¢66Rõô$”äEõDUƒ¢66Rõô$”äEõDUƒ¢66Rõô$”äEõ4Õ¢66Rõô$”äEô”Ôs¢66Rõõ5DTä4”Ã ¢66RõõôÅ•ôôde4UC¢66RõôDUD…õ$ätS¢66RõôDUD…ô4ÄÕ¢66Rõõeõ5t•¥¤ÄS¢66Rõô$ÄTäEô4ôÄõ#¢66RõôÅ„õ$Tc¢66RõõU4…ô4ôå5C¢66Rõô4õ•õ5E$”DS ¢7FFUö÷†‚æ÷Â“²'&V³°¢òò54ETÕòÆ–æW3¢F†Rf–Ww÷'G2F†RvÖR6WG2Âv—F‚F†R72F†W’fÆÂ–â†öæR6WB&Vf÷&R—G272÷Vç2F¶W2F†B72w2çVÖ&W"Â2F†RÆ–æW2Fò’âF†RBÆ–æP¢òò&–çG2F†RöæRG&rW6VBÂ6òf–Ww÷'BF†R726WB—2FöÆBg&öÒöæR6'&–VB÷fW"â&–çFVB†W&RÂæ÷B–â7FFUö÷¢Æöö²Ö†VB66ç2&WÆ’7FFUö÷Föð¢66Rõõd”Uuõ%E3¢²6öç7B÷f–Ww÷'G2¦"Ò²7FFUö÷†‚æ÷Â“°¢–b‡76GV×‚’’²Vç6–væVBÆöærÆöærâÒuöVæ2òu÷7FE÷72¢u÷7FE÷72²²B‚$òVÆÇR6WE÷f–Ww÷'G2f—'7CÒVBãÒVB"ÂâÂ"Óæf—'7BÂ"Óæâ“°¢f÷"†–çB’Ò²’Â"Óæâbb’ÂC²’²²’B‚$òVÆÇRg²VEÒÒRãbÂRãbÂRãbÂRãb"ÂâÂ"Óæf—'7B²’Â"Óçe¶•Õ³ÒÂ"Óçe¶•Õ³ÒÂ"Óçe¶•Õ³%ÒÂ"Óçe¶•Õ³5Ò“²Ð¢'&V³²Ð¢66RõõDeô$”äC¢²6öç7B÷Feö&–æB¦"Ò²B‚%’Fb&–æBVBFG#ÒR6ÆÇ‚6—¦SÒVÆÇR"Â"Óæ–æFW‚Â‡Vç6–væVBÆöærÆöær–"ÓæFG"Â‡Vç6–væVBÆöærÆöær–"Óç6—¦R“²7FFUö÷†‚æ÷Â“²'&V³²Ð¢66RõõDeõ5DDS¢²6öç7B÷Fe÷7FFR¦"Ò²7FF–26öç7B6†"¦6öç7Bu³EÒÒ²&&Vv–â"Â&VæB"Â'W6R"Â'&W7VÖR'Ó²B‚%’FbW27FÃÒR6ÆÇ‚"Âu¶"Óçv†Bb5ÒÂ‡Vç6–væVBÆöærÆöær–"Óæ7FÂ“°¢7FFUö÷†‚æ÷Â“²Fe÷7FFUöwR†"“²'&V³²Ð¢66Rõõ$”Õõ$U5D%C¢7FFUö÷†‚æ÷Â“²'&V³°¢66Rõõ$5EôD•44$C¢B‚%’&7FW&—¦W"F—66&BVB"Â‚†6öç7B÷&7EöF—66&B¢–’Óæöâ“²7FFUö÷†‚æ÷Â“²'&V³°¢66Rõô4õ•ô#%C¢6÷•ö#'B†“²'&V³°¢66Rõô4õ•õC$#¢6÷•÷C&"†“²'&V³°¢66Rõô4õ•õC%C¢6÷•÷C'B†“²'&V³°¢66RõôE$uõDUƒ¢G&u÷FW‚†“²'&V³°¢66Rõô4ÄT%õDUƒ¢6ÆV%÷FW‚†“²'&V³°¢66RõôDÅ53¢FÇ75öWfÂ†“²'&V³°¢66RõôD•5D4ƒ¢–b†uö76…öâbb76…÷F¶R‡’’²B‚$‚VÆÇRF—7F6‚Ç&VG’'Vâ†VBöbF†R72"Âu÷7FE÷72“²'&V³²Ð¢F—7F6‚†“²'&V³°¢66Rõô$%$”U#¢²–çC3%÷B&—G3²ÖVÖ7’‚f&—G2ÂÂB“²òò÷&FW&VBBF†RæW‡BG&r72†&'&–W%övÖR“²–ç6–FR72v—F‚6†FW"w&—FW2ÂF†R72VæG2f—'7B‡7w&—FUöæ÷FR“²–âöæRv—F†÷WBÂ&Vf÷&R—G2æW‡Bw&—FR‡&öu÷7w&—FW2¢–çB÷&FW"ÒuöVæ2bb†&—G2b„ådåô$%$”U%ôõ$DU%õ$”Ô•D•dU5ô$•BÂådåô$%$”U%ôõ$DU%ôe$tÔTåE5ô$•BÂådåô$%$”U%ôõ$DU%ôe$tÔTåE5õD”ÄTEô$•B’’Â7Æ—BÒ÷&FW"bbu÷75÷7w&—FS°¢B‚$’VÆÇR&'&–W"&—G2R7‚W2"Âu÷7FE÷72Â&—G2Â7Æ—Bò#¢VæG2F†R72‡6†FW"w&—FW2’"¢÷&FW"ò"‡&VæFW"72÷Vã²ÆFW"6†FW"w&—FR—2÷&FW&VBgFW"—G2G&w2’"¢uöVæ2ò"‡&VæFW"72÷Vâ’"¢""“°¢–b‡7Æ—B’²'öVæB‚“²7FF–2–çBöæ6S²–b‚öæ6R²²’ÔÄôr‚&vÖR&'&–W"†&—G2R7‚’gFW"6†FW"w&—FW2–âG&r73¢F†R72VæG2F†W&R†f—'7B66R72VÆÇR’"Â&—G2Â‡Vç6–væVBÆöærÆöær–u÷7FE÷72“²Ð¢VÇ6R–b†÷&FW"’u÷75ö÷&FW"ÃÒ†&—G2bådåô$%$”U%ôõ$DU%õ$”Ô•D•dU5ô$•Bò¢’Â†&—G2b„ådåô$%$”U%ôõ$DU%ôe$tÔTåE5ô$•BÂådåô$%$”U%ôõ$DU%ôe$tÔTåE5õD”ÄTEô$•B’ò"¢“°¢–b†&—G2’²uövÖUö&'&–W"Ò²uöf%ö&"²³²Ò'&V³²Òòòuöf%ö&#¢6æ6†÷G2F¶Vâ&Vf÷&R—B&R7FÆR‡6VRf%÷&W&R¢66Rõõ4UEõDU…õôôÃ¢66Rõõ4UEõ4ÕõôôÃ¢7FFUö÷†‚æ÷Â“²'&V³°¢FVfVÇC¢'&V³°¢Ð¢³Ò‚ç6—¦S°¢Ð§Ð ¢òòÒÒÒÒÒÒÒÒÒÒ&W6VçC¢G&rF†R7v6†–âFW‡GW&RFòF†R4ÖWFÄÆ–W#²÷F–öæÂ67&VVç6†÷BÒÒÒÒÒÒÒÒÒÐ§7FF–2–CÄÕDÅ&VæFW%—VÆ–æU7FFSâuö&Æ—E÷6ó²7FF–2ÕDÅ—†VÄf÷&ÖBuö&Æ—Eöf×C°§7FF–2–CÄÕDÅ6×ÆW%7FFSâuö&Æ—E÷6×°§7FF–2V–çCcE÷Bu÷&W6VçEöã°§V–çCcE÷B×FÅ÷&W6VçEö6÷VçB‡fö–B’²&WGW&âu÷&W6VçEöã²Òòò7GVÂ&W6VçB6÷VçB††VBGG&–'WF–öâDÕô„TB6ö×WFW2F†R&W6VçBg&ÖR&FR§fö–B×FÅ÷&W5÷6æ6†÷B†×FÅ÷&W5ö6÷VçB¦÷WB’°¢ÖV×6WB†÷WBÂÂ6—¦Vöb¦÷WB“°¢÷WBÓç6òÒu÷6òæã°¢÷WBÓæ72Òuö72æã°¢÷WBÓæG72ÒuöG72æã°¢÷WBÓç6†FW'2Ò‡V–çC3%÷B’†u÷6†66†Ròu÷6†66†Ræ6÷VçB¢“°¢÷5÷Væf—%öÆö6µöÆö6²‚fuö3öÆö6²“°¢÷WBÓæ3ö'Vg2Ò‡V–çC3%÷B’†uö3ö&æ·2æ6÷VçB²uö3öW†6ÇW6—fR“²÷WBÓæ3öÆÆö2Òuö3öÆÆö5ö'—FW3²÷WBÓæ3öFFÒuö3öFFö'—FW3°¢÷5÷Væf—%öÆö6µ÷VæÆö6²‚fuö3öÆö6²“°¢F‡&VEö×WFW…öÆö6²‚fu÷6×R“°¢÷WBÓæÆ—fU÷FW‚Òuö'•÷6W&–Âæã°¢F‡&VEö×WFW…÷VæÆö6²‚fu÷6×R“°¢F‡&VEö×WFW…öÆö6²‚fuö×R“°¢÷WBÓçööÇ2Ò‡V–çC3%÷B–uöçööÃ°¢÷WBÓçööÅöFG2Òu÷ööÅöFG3°¢F‡&VEö×WFW…÷VæÆö6²‚fuö×R“°¢÷5÷Væf—%öÆö6µöÆö6²‚fu÷&×R“°¢÷WBÓçf–Wu÷6Æ÷G2Ò‡V–çC3%÷B–uöçG&Vs°¢f÷"†–çB’Ò²’ÂuöçG&Vs²’²²’–b†u÷G&Vu¶•Òçf–Wr’÷WBÓçf–WuöæöæçVÆÂ²³°¢f÷"†–çB’Ò²’Âuöç7&Vs²’²²’–b†u÷7f–Wu¶•Ò’÷WBÓç7f–WuöæöæçVÆÂ²³°¢÷5÷Væf—%öÆö6µ÷VæÆö6²‚fu÷&×R“°§Ð¢òò&W6÷W&6R6÷VçG2‡&VB'’Æöær×'Vâ7F&–Æ—G’Ã2“¢F†R†÷7B6ÆÇ2—BW"vÖRF–6²òWfVçB†WfW'’#SF–6·2ÂÆöF–ær67&VVâF—6Ö—76VBÂW†—B’Âæ÷B6×ÆVB'’vÆÂ6Æö6²ò&W6VçB6÷VçBà¢òòÆ–æRf÷&ÖC¢¶çfåÒ&W6÷W&6R6÷VçG2F–6²ÇF–6²çVÖ&W#âÇG&–vvW#ã¢ÆæÖSâÇfÇVSââââ†æÖW26öçF–âæò76W2Â6–ævÆR×76R6W&FVB“²æWrf–VÆG2&RVæFVBFòF†RVæBöbF†—2Æ–æP§fö–B×FÅ÷&W5ö6÷VçG2‡V–çC3%÷BF–6²Â6öç7B6†"§v‡’’°¢×FÅ÷&W5ö6÷VçB3°¢×FÅ÷&W5÷6æ6†÷B‚f2“°¢–çBFÂÂGS²V–çCcE÷BF²çfåöF—'G•÷6Æ÷E÷7FG2‚fFÂÂfGRÂfF“°¢Äôr‚%¶çfåÒ&W6÷W&6R6÷VçG2F–6²WRW3¢&VæFW%÷—VÆ–æW2WR6ö×WFU÷—VÆ–æW2WR6†FW'2WRFWF…÷7FVæ6–Å÷7FFW2WRf–Ww5öæöæçVÆÂWRf–Wu÷6Æ÷G2WRÆ—fU÷FW‡GW&W2WRööÇ5öÆ—fRWRööÇ5÷F÷FÂVÆÇR6×ÆW%÷f–Ww5öæöæçVÆÂWRF—'G•÷&ævW5öÆ—fRVBF—'G•÷&ævW5÷F÷FÂVÆÇR3ö'VffW'2WR3ö'—FW2VÆÇR3÷W6VBVÆÇR"À¢F–6²Âv‡’Â2ç6òÂ2æ72Â2ç6†FW'2Â2æG72Â2çf–WuöæöæçVÆÂÂ2çf–Wu÷6Æ÷G2Â2æÆ—fU÷FW‚À¢2çööÇ2Â‡Vç6–væVBÆöærÆöær–2çööÅöFG2Â2ç7f–WuöæöæçVÆÂÂFÂÂ‡Vç6–væVBÆöærÆöær–FÂ2æ3ö'Vg2Â‡Vç6–væVBÆöærÆöær–2æ3öÆÆö2Â‡Vç6–væVBÆöærÆöær–2æ3öFF“°§Ð¢òò67&VVç6†÷Böâ&WVW7B‡W6VB'’DÕôådåõ4„õEõT’“¢FÕövg…÷6†÷EöB†²ÂF‚’ÒF¶R67&VVç6†÷BBF†R·F‚g&ÖRgFW"F†R7W'&VçB&W6VçBg&ÖS²GVR&WVW7G2&R6†V6¶VBBWfW'’g&ÖRw2&W6Vç@¢6FVf–æR4„õEôÔ‚€§7FF–27G'V7B²V–çCcE÷Bg&ÖS²6†"F…³#EÓ²Òu÷6†÷Gµ4„õEôÔ…Ó²7FF–2–çBuöç6†÷G°§7FF–2F‡&VEö×WFW…÷Bu÷6†÷Gö×RÒD…$TEôÕUDU…ô”ä•D”Ä•¤U#°§7FF–2–çB&W6VçEöVæ6öFR†–CÄÕDÄ6öÖÖæD'VffW#â6"Â–CÄÕDÅFW‡GW&Sâ7&2Â–CÄÕDÅFW‡GW&SâG7BÂ6öç7B–çB7&÷³EÒ“°¦VçVÒ²$U4TåEôd”ÄTBÂ$U4TåEõ5DäD$BÂ$U4TåEôÔUDÄe‚Ó°§7FF–2–çB&W6VçEööfg67&VVåöÓB†–CÄÕDÅFW‡GW&Sâ7&2Â6öç7B–çB7&÷³EÒÂ–CÄÕDÅFW‡GW&Sâ÷WBÂfö–B…æFöæR’†–çB’“°§7FF–2–çB&W6VçEööfg67&VVå÷v—B†–CÄÕDÅFW‡GW&Sâ7&2Â6öç7B–çB7&÷³EÒÂ–CÄÕDÅFW‡GW&Sâ÷WB“°¢òòvÖR–7GW&RæVVFVB'’F†Rv†öÆR×v–æF÷r67&VVç6†÷B†g&öçBVæBv–æF÷u÷6†÷C²æòÆöævW"FöW2G&uf–Wt†–W&&6‡’öâÖWFÂÆ–W"v—F‚F†RF—7Æ’Æ–æ²GF6†VB“¢BF†RæW‡B&W6VçBÂF†R&W6VçBF‡&VBG&w2F†R6ÖRg&ÖP¢òòv—F‚F†R6ÖRVæ6öF–ær2F†R&W6VçB–çFòâöfg67&VVâFW‡GW&RöbF†Rf–Wrw26—¦RÂæBgFW"F†RuRf–æ—6†W2†æG2÷WB4t–ÖvR–âF†R6ö×ÆWF–öâ6ÆÆ&6²âwV&FVB'’u÷6†÷Gö×P§7FF–2fö–B…æuöw&%ö6"’„4t–ÖvU&Vb–Ör“°§fö–B×FÅöw&%öæW‡E÷&W6VçB‡fö–B…æ6"’„4t–ÖvU&Vb–Ör’’²F‡&VEö×WFW…öÆö6²‚fu÷6†÷Gö×R“²uöw&%ö6"Ò¶6"6÷•Ó²F‡&VEö×WFW…÷VæÆö6²‚fu÷6†÷Gö×R“²Ð§7FF–2fö–Bw&%÷&W6VçB†–CÄÕDÅFW‡GW&Sâ7&2Â6öç7B–çB7&÷³EÒÂ4u6—¦R7¢’°¢F‡&VEö×WFW…öÆö6²‚fu÷6†÷Gö×R“²fö–B…æ6"’„4t–ÖvU&Vb’Òuöw&%ö6#²uöw&%ö6"Òæ–Ã²F‡&VEö×WFW…÷VæÆö6²‚fu÷6†÷Gö×R“°¢–b‚6"’&WGW&ã°¢6—¦U÷BrÒ‡6—¦U÷B—7¢çv–GF‚Â‚Ò‡6—¦U÷B—7¢æ†V–v‡BÂ&÷rÒr¢C°¢ÕDÅFW‡GW&TFW67&—F÷"¦BÒ´ÕDÅFW‡GW&TFW67&—F÷"FW‡GW&S$DFW67&—F÷%v—F…—†VÄf÷&ÖC¤ÕDÅ—†VÄf÷&ÖD$u$…Væ÷&Òv–GFƒ§r†V–v‡C¦‚Ö—ÖVC¤äõÓ°¢BçW6vRÒÕDÅFW‡GW&UW6vU&VæFW%F&vWBÂÕDÅFW‡GW&UW6vU6†FW%&VC²Bç7F÷&vTÖöFRÒÕDÅ7F÷&vTÖöFU&—fFS°¢–CÄÕDÅFW‡GW&Sâ÷WBÒrbb‚ò¶uöFWbæWuFW‡GW&Uv—F„FW67&—F÷#¦EÒ¢æ–Ã²–CÄÕDÄ'VffW#â'VbÒ÷WBò¶uöFWbæWt'VffW%v—F„ÆVæwFƒ§&÷r¢‚÷F–öç3¤ÕDÅ&W6÷W&6U7F÷&vTÖöFU6†&VEÒ¢æ–Ã°¢–b‚'Vb’²ÔÄôr‚%µ$TEÒgVÆÂ×v–æF÷r67&VVç6†÷C¢6ææ÷B7&VFR÷WGWB‚W§W‚W§R’"ÂrÂ‚“²6"„åTÄÂ“²&WGW&ã²Ð¢òòuR6ö×ÆWF–öâ7F'G2F†R&VF&6²Â¶VW–ærF†R&W6VçBF‡&VBg&VRâV6‚6GW&R÷vç2—G266ÆW"æBFW‡GW&W2VçF–Â6ö×ÆWF–öâà¢fö–B…ç&VF&6²’†–çB’Òâ†–çBö²’°¢–b‚ö²’²6"„åTÄÂ“²&WGW&ã²Ð¢–CÄÕDÄ6öÖÖæD'VffW#â6#"Òö6"‚“²–CÄÕDÄ&Æ—D6öÖÖæDVæ6öFW#â"Ò¶6#"&Æ—D6öÖÖæDVæ6öFW%Ó°¢¶"6÷”g&öÕFW‡GW&S¦÷WB6÷W&6U6Æ–6S£6÷W&6TÆWfVÃ£6÷W&6T÷&–v–ã¤ÕDÄ÷&–v–äÖ¶RƒÂÂ’6÷W&6U6—¦S¤ÕDÅ6—¦TÖ¶R‡rÂ‚Â’Fô'VffW#¦'VbFW7F–æF–öäöfg6WC£FW7F–æF–öä'—FW5W%&÷s§&÷rFW7F–æF–öä'—FW5W$–ÖvS§&÷r¢…Ó°¢¶"VæDVæ6öF–æuÓ°¢¶6#"FD6ö×ÆWFVD†æFÆW#¥â†–CÄÕDÄ6öÖÖæD'VffW#â2’²‡fö–B–3°¢4t6öÆ÷%76U&Vb72Ò4t6öÆ÷%76T7&VFUv—F„æÖR†´4t6öÆ÷%76U5$t"“°¢4t6öçFW‡E&Vb7G‚Ò4t&—FÖ6öçFW‡D7&VFR†'Vbæ6öçFVçG2ÂrÂ‚Â‚Â&÷rÂ72Â´4t–ÖvTÇ†æöæU6¶—f—'7BÂ´4t&—FÖ'—FT÷&FW#3$Æ—GFÆR“°¢4t–ÖvU&Vb–ÖrÒ7G‚ò4t&—FÖ6öçFW‡D7&VFT–ÖvR†7G‚’¢åTÄÃ²6"†–Ör“°¢4t–ÖvU&VÆV6R†–Ör“²4t6öçFW‡E&VÆV6R†7G‚“²4t6öÆ÷%76U&VÆV6R†72“²ÕÓ°¢¶6#"6öÖÖ—EÓ°¢Ó°¢–b†u÷B’²–b‚&W6VçEööfg67&VVåöÓB‡7&2Â7&÷Â÷WBÂ&VF&6²’’6"„åTÄÂ“²Ð¢VÇ6R°¢–CÄÕDÄ6öÖÖæD'VffW#âG&rÒö6"‚“°¢–b‚&W6VçEöVæ6öFR†G&rÂ7&2Â÷WBÂ7&÷’’²6"„åTÄÂ“²&WGW&ã²Ð¢¶G&rFD6ö×ÆWFVD†æFÆW#¥â†–CÄÕDÄ6öÖÖæD'VffW#â2’²&VF&6²†2ç7FGW2ÓÒÕDÄ6öÖÖæD'VffW%7FGW46ö×ÆWFVB“²ÕÓ²¶G&r6öÖÖ—EÓ°¢Ð§Ð§fö–BFÕövg…÷6†÷EöB†Æöær²Â6öç7B6†"§F‚’°¢F‡&VEö×WFW…öÆö6²‚fu÷6†÷Gö×R“°¢–b†uöç6†÷GÂ4„õEôÔ‚’²u÷6†÷G¶uöç6†÷GÒæg&ÖRÒu÷&W6VçEöâ²‡V–çCcE÷B’†²âò²¢“²7G&Æ7’†u÷6†÷G¶uöç6†÷GÒçF‚ÂF‚Â6—¦Vöbu÷6†÷G³ÒçF‚“²uöç6†÷G²³²Ð¢VÇ6RÔÄôr‚%µ$TEÒÖ÷&RF†âVB67&VVç6†÷B&WVW7G3²G&÷–ærW2"Â4„õEôÔ‚ÂF‚“°¢F‡&VEö×WFW…÷VæÆö6²‚fu÷6†÷Gö×R“°§Ð¢òòDÕôådåõ4„õEôõUCÕt”ED‡„„T”t…B†f÷"–7GW&RVÆ—G’ÖWG&–72“¢F†R&W6VçBÖg&ÖR67&VVç6†÷B–ç7FVB6GW&W2F†R&W7VÇB'W66ÆVBFòF†—2&W6öÇWF–öâ'’F†R&W6VçB72"Â’æRâF†R–7GW&RF†RÆ–W"6VW2öâ67&VVà¢òòöbF†B6—¦R‡v—F‚DÕõ$U4TåEõU44ÄSÖÖWFÆg‚F‡&÷Vv‚ÖWFÄe‚ÂöâF†RÖWFÂBVWVS¢&W6VçEööfg67&VVåöÓB“²67&VVãÓ‡FW‡GW&RGV×’—2VæffV7FV@§7FF–2fö–B6†÷B†–CÄÕDÅFW‡GW&Sâ7&2Â6öç7B6†"§F‚Â–çB67&VVâÂ6öç7B–çB7&÷³EÒ’°¢7FF–2–çB÷WE÷rÒÓÂ÷WEöƒ°¢–b†÷WE÷rÂ’°¢÷WE÷rÒ°¢6öç7B6†"¦RÒvWFVçb‚$DÕôådåõ4„õEôõUB"“²–çBrÒÂ‚Ò²6†"6WÒÂW‡G&Ò°¢–b†Rbb766æb†RÂ"VBV2VBV2"ÂgrÂg6WÂf‚ÂfW‡G&’ÓÒ2bb6WÓÒw‚rbbrâbb‚â’²÷WE÷rÒs²÷WEö‚Òƒ²Ð¢Ð¢–CÄÕDÅFW‡GW&Sâ&VBÒ7&3°¢–b†÷WE÷rbb67&VVâ’°¢ÕDÅFW‡GW&TFW67&—F÷"¦BÒ´ÕDÅFW‡GW&TFW67&—F÷"FW‡GW&S$DFW67&—F÷%v—F…—†VÄf÷&ÖC¤ÕDÅ—†VÄf÷&ÖD$u$…Væ÷&Òv–GFƒ¦÷WE÷r†V–v‡C¦÷WEö‚Ö—ÖVC¤äõÓ°¢BçW6vRÒÕDÅFW‡GW&UW6vU&VæFW%F&vWBÂÕDÅFW‡GW&UW6vU6†FW%&VC²Bç7F÷&vTÖöFRÒÕDÅ7F÷&vTÖöFU&—fFS°¢–CÄÕDÅFW‡GW&Sâ÷WBÒ¶uöFWbæWuFW‡GW&Uv—F„FW67&—F÷#¦EÓ°¢–b‚÷WB’²ÔÄôr‚%µ$TEÒ6ææ÷B7&VFRF†R67&VVç6†÷B÷WGWBFW‡GW&RVG‚VB"Â÷WE÷rÂ÷WEö‚“²&WGW&ã²Ð¢–b†u÷B’²–b‚&W6VçEööfg67&VVå÷v—B‡7&2Â7&÷Â÷WB’’&WGW&ã²Ð¢VÇ6R²–CÄÕDÄ6öÖÖæD'VffW#â6"Òö6"‚“²–b‚&W6VçEöVæ6öFR†6"Â7&2Â÷WBÂ7&÷’’&WGW&ã²¶6"6öÖÖ—EÓ²¶6"v—EVçF–Ä6ö×ÆWFVEÓ²Ð¢&VBÒ÷WC°¢Ð¢–b‡&VBç—†VÄf÷&ÖBÒÕDÅ—†VÄf÷&ÖE$t$…Væ÷&Òbb&VBç—†VÄf÷&ÖBÒÕDÅ—†VÄf÷&ÖE$t$…Væ÷&Õ÷5$t"b`¢&VBç—†VÄf÷&ÖBÒÕDÅ—†VÄf÷&ÖD$u$…Væ÷&Òbb&VBç—†VÄf÷&ÖBÒÕDÅ—†VÄf÷&ÖD$u$…Væ÷&Õ÷5$t"’²ÔÄôr‚%µ$TEÒ67&VVç6†÷BFöW2æ÷B7W÷'Bf÷&ÖBVÇR"Â‡Vç6–væVBÆöær—&VBç—†VÄf÷&ÖB“²&WGW&ã²Ð¢6—¦U÷BrÒ&VBçv–GF‚Â‚Ò&VBæ†V–v‡BÂ&÷rÒr¢C°¢–CÄÕDÄ'VffW#â'VbÒ¶uöFWbæWt'VffW%v—F„ÆVæwFƒ§&÷r¢‚÷F–öç3¤ÕDÅ&W6÷W&6U7F÷&vTÖöFU6†&VEÓ°¢–CÄÕDÄ6öÖÖæD'VffW#â6"Òö6"‚“°¢–CÄÕDÄ&Æ—D6öÖÖæDVæ6öFW#â"Ò¶6"&Æ—D6öÖÖæDVæ6öFW%Ó°¢¶"6÷”g&öÕFW‡GW&S§&VB6÷W&6U6Æ–6S£6÷W&6TÆWfVÃ£6÷W&6T÷&–v–ã¤ÕDÄ÷&–v–äÖ¶RƒÂÂ’6÷W&6U6—¦S¤ÕDÅ6—¦TÖ¶R‡rÂ‚Â¢Fô'VffW#¦'VbFW7F–æF–öäöfg6WC£FW7F–æF–öä'—FW5W%&÷s§&÷rFW7F–æF–öä'—FW5W$–ÖvS§&÷r¢…Ó°¢¶"VæDVæ6öF–æuÓ²¶6"6öÖÖ—EÓ²¶6"v—EVçF–Ä6ö×ÆWFVEÓ°¢–b‡67&VVâbb&VBÓÒ7&2bbFöÖ–5öÆöEöW‡Æ–6—B‚fuö÷&–v–åöÆÂÂÖVÖ÷'•ö÷&FW%÷&VÆ†VB’’²òòÆ÷vW"ÖÆVgB÷&–v–ã¢F†Rv–æF÷r–ÖvR—27F÷&VB&÷GFöÒ×W‡6VRuö÷&–v–åöÆÂ¢V–çC…÷B§Ò'Vbæ6öçFVçG2Â§F×ÒÖÆÆö2‡&÷r“°¢–b‡F×’²f÷"‡6—¦U÷B’Ò²’Â‚ò#²’²²’²ÖVÖ7’‡F×Â²’¢&÷rÂ&÷r“²ÖVÖ7’‡²’¢&÷rÂ²†‚ÒÒ’’¢&÷rÂ&÷r“²ÖVÖ7’‡²†‚ÒÒ’’¢&÷rÂF×Â&÷r“²Òg&VR‡F×“²ÒÐ¢–çB&w&Ò&VBç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖD$u$…Væ÷&ÒÇÂ&VBç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖD$u$…Væ÷&Õ÷5$t#°¢4t6öÆ÷%76U&Vb72Ò4t6öÆ÷%76T7&VFUv—F„æÖR†´4t6öÆ÷%76U5$t"“°¢4t&—FÖ–æfò&’Ò&w&ò†´4t–ÖvTÇ†æöæU6¶—f—'7BÂ´4t&—FÖ'—FT÷&FW#3$Æ—GFÆR’¢„4t&—FÖ–æfò–´4t–ÖvTÇ†æöæU6¶—Æ7C°¢4t6öçFW‡E&Vb7G‚Ò4t&—FÖ6öçFW‡D7&VFR†'Vbæ6öçFVçG2ÂrÂ‚Â‚Â&÷rÂ72Â&’“°¢4t–ÖvU&Vb–ÖrÒ7G‚ò4t&—FÖ6öçFW‡D7&VFT–ÖvR†7G‚’¢åTÄÃ°¢4eU$Å&VbW&ÂÒ4eU$Ä7&VFTg&öÔf–ÆU7—7FVÕ&W&W6VçFF–öâ„åTÄÂÂ†6öç7BT–çC‚¢—F‚Â7G&ÆVâ‡F‚’ÂfÇ6R“°¢4t–ÖvTFW7F–æF–öå&VbG7BÒW&Âò4t–ÖvTFW7F–æF–öä7&VFUv—F…U$Â‡W&ÂÂ…õö'&–FvR4e7G&–æu&Vb•UEG—Uäræ–FVçF–f–W"ÂÂåTÄÂ’¢åTÄÃ°¢òò–bç’7FW––VÆG2çVÆÂ†F—&V7F÷'’FöW2æ÷BW†—7Bòæ÷Bw&—F&ÆRÂ&—FÖ6öçFW‡B6ææ÷B&R7&VFVB’Â&W÷'Bf–ÇW&RæB&WGW&ã²&Wf–÷W6Ç’—BvVçB7G&–v‡BFò4e&VÆV6R„åTÄÂ’æB7&6†VB–â4e&VÆV6Ræ6öÆB††—BGW&–ær“S"×v–FRDU„ETÕ¢–çBö²Ò°¢–b‚7G‚ÇÂ–ÖrÇÂG7B’ÔÄôr‚%µ$TEÒ67&VVç6†÷BW2f–ÆVC¢W2"ÂF‚Â7G‚ò&6ææ÷B7&VFRF†R&—FÖ6öçFW‡B"¢–Örò&6ææ÷BvWBF†R–ÖvR"¢&6ææ÷Bw&—FR†F—&V7F÷'’Ö—76–ær÷"æ÷Bw&—F&ÆR’"“°¢VÇ6R²4t–ÖvTFW7F–æF–öäFD–ÖvR†G7BÂ–ÖrÂåTÄÂ“²ö²Ò4t–ÖvTFW7F–æF–öäf–æÆ—¦R†G7B“²Ð¢–b†G7B’4e&VÆV6R†G7B“²–b‡W&Â’4e&VÆV6R‡W&Â“²4t–ÖvU&VÆV6R†–Ör“²4t6öçFW‡E&VÆV6R†7G‚“²4t6öÆ÷%76U&VÆV6R†72“°¢ÔÄôr‚'67&VVç6†÷BW§W‚W§Rf÷&ÖBVÇR(i"W2W2"ÂrÂ‚Â‡Vç6–væVBÆöær—&VBç—†VÄf÷&ÖBÂF‚Âö²ò&ö²"¢&f–ÆVB"“°¢–b†uöFVfW%ö"ãÒbbvWFVçb‚$DÕôådåôDTdU%ô""’bb7G&6×†vWFVçb‚$DÕôådåôDTdU%ô""’Â#"’bbö²’²7FF–2–çB6†÷G3²–b‚²·6†÷G2ÓÒ"’²uöFVfW%ö%ööâÒ²ÔÄôr‚$DÕôådåôDTdU%ô#¢6V6öæB67&VVç6†÷Bw&—GFVã²FVfW'&VB72ÖW&v–æröâ"“²ÒÐ¢–b†u÷VF5ö"ãÒbbvWFVçb‚$DÕôådåõTD5ô""’bb7G&6×†vWFVçb‚$DÕôådåõTD5ô""’Â#"’bbö²’°¢7FF–2–çB6†÷G3°¢–b‚²·6†÷G2ÓÒ"’²u÷VF5ö%ööâÒ²ÔÄôr‚$DÕôådåõTD5ô#¢6V6öæB67&VVç6†÷Bw&—GFVã²DÕôådåõTåu$•EDTåôD2öâ"“²Ð¢Ð§Ð¢òòÒÒÒÒÒÒÒÒÒÒDÕôådåõDU„ETÕÖg&ÖRçVÖ&W#§F‚‚VB–âF†RF‚—2&WÆ6VBv—F‚&–æFW‚£¶Ö—ÆWfVÂ"“¢6fRV6‚6öÆ÷"&VæFW"F&vWBW6VB'’F†Bg&ÖR2â–ÖvRÒÒÒÒÒÒÒÒÒÐ¢òòF–væ÷7F–3¢&Vf÷&RF†Rf—'7B&VæFW"72öbF†RæW‡Bg&ÖR—2÷VæVBÂv—Bf÷"F†RuRFòf–æ—6‚ÂF†VâG&rV6‚öæR‡F†RÖ—ÆWfVÂW6VB'’F†Bg&ÖR’'’gVÆÂ×67&VVâ6×Æ–ær–çFò$t$‚æB6fR2är†fÆöBf÷&ÖG2&R6Æ×VBFò³ÃÒ§7FF–2Æöæru÷FEög&ÖRÒÓ#²7FF–26öç7B6†"¦u÷FE÷Fƒ²7FF–2å4×WF&ÆT'&’¦u÷FEöÆ—7C²7FF–2–çBu÷FEöFöæS°¢òòDÕôådåõDU„ETÕÖÆ–W'3ÓÆãã§Fƒ¢F†Rf—'7Bg&ÖRv—F‚BÆV7BâG&w2–çFò6–ævÆRÆ–W'2öb6ÖÆÂ†BÖ÷7BcGƒcB’'&’÷"7V&RFW‡GW&RÆWfVÇ2Â6÷VçFVBW"72÷"6Æ–6RÖÖW&vVBÆ–W ¢òò„'&VF‚öbF†Rv–ÆBw27V&RÖf–ÇFW"g&ÖRÂWfW'’FVçFƒ¢&÷WBSv–ç7B#“¢—G2F&vWG2&R¶WBg&ÖR'’g&ÖRæBGV×VBv†Vâ7V6‚g&ÖRVæG2ÂF†R6ÖRg&ÖRv—F‚÷"v—F†÷WBDÕôådåõ4Ä”4UôÔU$tP§7FF–2–çBu÷FEöÆ–W'2Âu÷FEöÆ6÷VçC²7FF–2Æöæru÷FEöÆg&ÖRÒÓ°§7FF–2fö–BFEöÆ–W%ö6÷VçB†–CÄÕDÅFW‡GW&SâBÂå5T–çFVvW"Çb’²–b‡Bbb‡BçFW‡GW&UG—RÓÒÕDÅFW‡GW&UG—S$D'&’ÇÂBçFW‡GW&UG—RÓÒÕDÅFW‡GW&UG—T7V&R’bbÔ‚‡Bçv–GF‚ãâÇbÂVÂ’ÃÒcB’u÷FEöÆ6÷VçB²³²Ð§7FF–2fö–BFEöGV×‡fö–B“°¢òòDÕôådåõDU„ETÕ×6—¦SÕw„ƒ§F‚–ç7FVB–6·2F†Rg&ÖR'’6öçFVçC¢F†Rf—'7Bg&ÖRF†B÷Vç2&VæFW"72v†÷6Rf—'7B6öÆ÷"GF6†ÖVçBÆWfVÂ—2w„‚†f÷"öæRÖöfbv÷&²v†÷6Rg&ÖRF–ffW'2&WGvVVâ'Vç2§7FF–2Vç6–væVBu÷FE÷rÂu÷FEöƒ°§fö–BFEöæ÷FR„ÕDÃE&VæFW%74FW67&—F÷"§BÂå5T–çFVvW"æ2’°¢–b†u÷FEög&ÖRÓÒÓ"’²6öç7B6†"¦RÒvWFVçb‚$DÕôådåõDU„ETÕ"’Â¦2ÒRò7G&6‡"†RÂs¢r’¢åTÄÃ²u÷FEög&ÖRÒ2òFöÂ†R’¢Ó²u÷FE÷F‚Ò2ò2²¢åTÄÃ²u÷FEöÆ—7BÒ´å4×WF&ÆT'&’'&•Ó°¢–b†2bb7G&æ6×†RÂ'6—¦SÒ"ÂR’bb766æb†R²RÂ"WW‚WR"Âfu÷FE÷rÂfu÷FEö‚’ÓÒ"’u÷FEög&ÖRÒÄôäuôÔƒ°¢–b†2bb7G&æ6×†RÂ&Æ–W'3Ò"Âr’’²u÷FEöÆ–W'2ÒFö’†R²r“²u÷FEög&ÖRÒ†Æöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR“²ÒÐ¢–b†u÷FEög&ÖRÂÇÂu÷FEöFöæR’&WGW&ã°¢ÆöærbÒ†Æöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR“°¢–b†u÷FEöÆ–W'2’°¢–b†bÒu÷FEöÆg&ÖR’²–b†u÷FEöÆg&ÖRãÒbbu÷FEöÆ6÷VçBãÒu÷FEöÆ–W'2’²ÔÄôr‚%·FW†GV×Òg&ÖRVÆC¢VBG&w2–çFò6ÖÆÂ'&’ò7V&RÆ–W'2"Âu÷FEöÆg&ÖRÂu÷FEöÆ6÷VçB“²u÷FEög&ÖRÒu÷FEöÆg&ÖS²FEöGV×‚“²&WGW&ã²Ð¢u÷FEöÆg&ÖRÒc²u÷FEög&ÖRÒc²u÷FEöÆ6÷VçBÒ²¶u÷FEöÆ—7B&VÖ÷fTÆÄö&¦V7G5Ó²Ð¢–b†æ2’FEöÆ–W%ö6÷VçB‡Bæ6öÆ÷$GF6†ÖVçG5³ÒçFW‡GW&RÂBæ6öÆ÷$GF6†ÖVçG5³ÒæÆWfVÂ“²Ð¢–b†u÷FEög&ÖRÓÒÄôäuôÔ‚’²–CÄÕDÅFW‡GW&SâCÒæ2òBæ6öÆ÷$GF6†ÖVçG5³ÒçFW‡GW&R¢æ–Ã²å5T–çFVvW"ÇbÒBæ6öÆ÷$GF6†ÖVçG5³ÒæÆWfVÃ°¢–b‡CbbÔ‚‡Cçv–GF‚ãâÇbÂVÂ’ÓÒu÷FE÷rbbÔ‚‡Cæ†V–v‡BãâÇbÂVÂ’ÓÒu÷FEö‚’²u÷FEög&ÖRÒc²ÔÄôr‚%·FW†GV×Òf—'7BWW‚WR&VæFW"F&vWBBg&ÖRVÆC¢GV×–ærF†—2g&ÖR"Âu÷FE÷rÂu÷FEö‚Âb“²ÒÐ¢–b†bâu÷FEög&ÖR’²FEöGV×‚“²&WGW&ã²Ð¢–b†bÒu÷FEög&ÖR’&WGW&ã°¢f÷"„å5T–çFVvW"’Ò²’Âæ2bb’Âƒ²’²²’°¢–CÄÕDÅFW‡GW&SâBÒBæ6öÆ÷$GF6†ÖVçG5¶•ÒçFW‡GW&S²–b‚B’6öçF–çVS°¢V–çCcE÷B6W"Ò’Â„å5T–çFVvW"•"ç'Bæâbb"ç'Bæ6öÆ÷%¶•Òò"ç'Bæ6öÆ÷%¶•ÒÓç6W&–Â¢°¢å5T–çFVvW"ÇbÒBæ6öÆ÷$GF6†ÖVçG5¶•ÒæÆWfVÂÂ6ÂÒBçFW‡GW&UG—RÓÒÕDÅFW‡GW&UG—S$D'&’òBæ6öÆ÷$GF6†ÖVçG5¶•Òç6Æ–6R¢°¢$ôôÂ6VVâÒäó²f÷"„å4'&’¦R–âu÷FEöÆ—7B’–b†Ræ6÷VçBÂRbb¶U³ÒVç6–væVDÆöætÆöæufÇVUÒÓÒ6W"bb¶U³%ÒVç6–væVD–çFVvW%fÇVUÒÓÒÇbbb¶U³5ÒVç6–væVD–çFVvW%fÇVUÒÓÒ6Â’6VVâÒ”U3°¢–b‚6VVâ’¶u÷FEöÆ—7BFDö&¦V7C¤·BÂ‡6W"’Â†Çb’Â‡6Â•ÕÓ°¢Ð§Ð§fö–BFEöæ÷FUöÆ–W"†–CÄÕDÅFW‡GW&SâBÂV–çCcE÷B6W"Âå5T–çFVvW"ÇbÂå5T–çFVvW"6Â’²òò6Æ–6RÖÖW&vVB72w2÷F†W"Æ–W'2‡6VRu÷6Õ÷Æâ¢–b†u÷FEög&ÖRÂÇÂu÷FEöFöæRÇÂu÷FEög&ÖRÓÒÄôäuôÔ‚ÇÂ†Æöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR’Òu÷FEög&ÖRÇÂB’&WGW&ã°¢–b†u÷FEöÆ–W'2’FEöÆ–W%ö6÷VçB‡BÂÇb“°¢–b‡BçFW‡GW&UG—RÒÕDÅFW‡GW&UG—S$D'&’’6ÂÒ°¢f÷"„å4'&’¦R–âu÷FEöÆ—7B’–b†Ræ6÷VçBÂRbb¶U³ÒVç6–væVDÆöætÆöæufÇVUÒÓÒ6W"bb¶U³%ÒVç6–væVD–çFVvW%fÇVUÒÓÒÇbbb¶U³5ÒVç6–væVD–çFVvW%fÇVUÒÓÒ6Â’&WGW&ã°¢¶u÷FEöÆ—7BFDö&¦V7C¤·BÂ‡6W"’Â†Çb’Â‡6Â•ÕÓ°§Ð¢òòF†RæW‡Bg&ÖRw2F&vWG2Âöâ&WVW7B‡F†RÖ2w2DÕô5DÂ'FW†GV×ÇFƒâ"Â†ÆUö†–Bæ2“¢F†RF‚—2¶WB‡7G&GW’ÂF†Rg&ÖRçVÖ&W"—2F†RöæRgFW"F†R7W'&Vç@¢òòF†RÖ2w2DÕô5DÂ'FW‡&rÇ6W&–ÃâÇFƒâ"††ÆUö†–Bæ2“¢ÆWfVÂöbWfW'’6Æ–6R†7V&Rw26—‚f6W2’öbF†R†÷7BFW‡GW&R&V†–æB6W&–ÂÂ&rÂgFW"F†RuRf–æ—6†W2††VFW"Æ–æS¢v–GF‚†V–v‡BÖWFÂf÷&ÖB'—FW2W"&÷rÂF†VâF†R6Æ–6R6÷VçBv†Vâ&÷fRöæR“°¢òòf÷"FW‡GW&RF†B—2æò&VæFW"F&vWB†FVfVÇBFW‡GW&RÂÆöö·WF&ÆR’v†÷6R6öçFVçB6†FW"FWVæG2öà§fö–BFÕövg…÷FW‡&r‡V–çCcE÷B6W&–ÂÂ6öç7B6†"§F‚’°¢õ÷Vç6fU÷Vç&WF–æVB–BòÒFW…övWB‡6W&–Â“°¢–b…¶ò—4¶–æDöd6Æ73¥´çe¶r6Æ75ÕÒ’òÒ‚„çe¶r¢–ò’ÓçFWƒ²òòFW‡GW&Rg&öÒF†RvÖRw2f–ÆW0¢–b‚òÇÂ¶ò6öæf÷&×5Fõ&÷Fö6öÃ¤&÷Fö6öÂ„ÕDÅFW‡GW&R•Ò’²ÔÄôr‚%·FW‡&uÒ6W#ÒVÆÇS¢æò†÷7BFW‡GW&R"Â‡Vç6–væVBÆöærÆöær—6W&–Â“²&WGW&ã²Ð¢–CÄÕDÅFW‡GW&SâBÒó²×FÅöf–æ—6…öwR‚“°¢–b‡Bç7F÷&vTÖöFRÓÒÕDÅ7F÷&vTÖöFTÖVÖ÷'–ÆW72ÇÂBç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖD–çfÆ–B’&WGW&ã°¢å5T–çFVvW"rÒBçv–GF‚Â‚ÒBæ†V–v‡BÂ&÷rÒr¢bÂ7¢Ò&÷r¢ƒ²òòb'—FW2FW†VÃ¢×VÇF—ÆRöbWfW'’Væ6ö×&W76VBFW†VÂ6—¦P¢å5T–çFVvW"ç2Ò‡BçFW‡GW&UG—RÓÒÕDÅFW‡GW&UG—T7V&RÇÂBçFW‡GW&UG—RÓÒÕDÅFW‡GW&UG—T7V&T'&’òb¢’¢Ô‚‡Bæ'&”ÆVæwF‚ÂR“²òòWfW'’6Æ–6R†7V&Rw26—‚f6W2’ÂöæRgFW"æ÷F†W ¢–CÄÕDÄ'VffW#â'VbÒ¶uöFWbæWt'VffW%v—F„ÆVæwFƒ§7¢¢ç2÷F–öç3¤ÕDÅ&W6÷W&6U7F÷&vTÖöFU6†&VEÓ°¢–CÄÕDÄ6öÖÖæD'VffW#â6"Òö6"‚“²–CÄÕDÄ&Æ—D6öÖÖæDVæ6öFW#â"Ò¶6"&Æ—D6öÖÖæDVæ6öFW%Ó°¢ÕDÄ&Æ—D÷F–öâ&òÒBç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖDFWFƒ3$fÆöEõ7FVæ6–Ã‚òÕDÄ&Æ—D÷F–öäFWF„g&öÔFWF…7FVæ6–Â¢ÕDÄ&Æ—D÷F–öäæöæS°¢f÷"„å5T–çFVvW"2Ò²2Âç3²2²²¢¶"6÷”g&öÕFW‡GW&S§B6÷W&6U6Æ–6S§26÷W&6TÆWfVÃ£6÷W&6T÷&–v–ã¤ÕDÄ÷&–v–äÖ¶RƒÂÂ’6÷W&6U6—¦S¤ÕDÅ6—¦TÖ¶R‡rÂ‚Â’Fô'VffW#¦'VbFW7F–æF–öäöfg6WC§7¢¢2FW7F–æF–öä'—FW5W%&÷s§&÷rFW7F–æF–öä'—FW5W$–ÖvS§7¢÷F–öç3¦&õÓ°¢¶"VæDVæ6öF–æuÓ²¶6"6öÖÖ—EÓ²¶6"v—EVçF–Ä6ö×ÆWFVEÓ°¢d”ÄR¦bÒf÷Vâ‡F‚Â'v""“²–b†b’²g&–çFb†bÂ"VÇRVÇRVÇRVÇR"Â‡Vç6–væVBÆöær—rÂ‡Vç6–væVBÆöær–‚Â‡Vç6–væVBÆöær—Bç—†VÄf÷&ÖBÂ‡Vç6–væVBÆöær—&÷r“²–b†ç2â’g&–çFb†bÂ"VÇR"Â‡Vç6–væVBÆöær–ç2“²gWF2‚uÆârÂb“²gw&—FR†'Vbæ6öçFVçG2ÂÂ7¢¢ç2Âb“²f6Æ÷6R†b“²Ð¢6öç7BV–çC…÷B§Ò'Vbæ6öçFVçG3°¢ÔÄôr‚%·FW‡&uÒ6W#ÒVÆÇRVÇW‚VÇRf÷&ÖBVÇRG—RVÇR(i"W3²f—'7B'—FW2S'‚S'‚S'‚S'‚S'‚S'‚S'‚S'‚"Â‡Vç6–væVBÆöærÆöær—6W&–ÂÂ‡Vç6–væVBÆöær—rÂ‡Vç6–væVBÆöær–‚Â‡Vç6–væVBÆöær—Bç—†VÄf÷&ÖBÂ‡Vç6–væVBÆöær—BçFW‡GW&UG—RÂF‚Â³ÒÂ³ÒÂ³%ÒÂ³5ÒÂ³EÒÂ³UÒÂ³eÒÂ³uÒ“°§Ð¢òòF†RÖ2w2DÕô5DÂ'FW†Æ—7BÄådâF&vWCâ"††ÆUö†–Bæ2“¢F†R&Vv—7FW&VBFW‡GW&W2öbF†BF&vWBƒ"Ò4B’Âv—F‚F†V—"7F÷&vR†wVW7BFG&W72Â6—¦R’æB&Æö6²ÖÆ–æV"Æ–÷WBÂf÷"&ÖVÖGV× §fö–BFÕövg…÷FW†Æ—7B†–çBF&vWB’°¢÷5÷Væf—%öÆö6µöÆö6²‚fu÷&×R“°¢f÷"†–çB’Ò²’ÂuöçG&Vs²’²²’²6öç7BFW…÷B§BÒfu÷G&Vu¶•ÒçC²–b‚u÷G&Vu¶•Òæö²ÇÂBÓæBçF&vWBÒF&vWB’6öçF–çVS°¢ÔÄôr‚%·FW†Æ—7EÒF–CÒVB6W#ÒVÆÇRF&vWBVBWW‚WW‚WRf÷&ÖBR7‚ÆWfVÇ2WRfÆw2R7‚7F÷&vRW6—¦RR6ÆÇ‚&Æö6²†V–v‡BöFWF‚Æös"WRòWR"Â’Â‡Vç6–væVBÆöærÆöær—BÓç6W&–ÂÂBÓæBçF&vWBÀ¢BÓæBçv–GF‚ÂBÓæBæ†V–v‡BÂBÓæBæFWF‚ÂBÓæBæf÷&ÖBÂBÓæBæÆWfVÇ2ÂBÓæBæfÆw2Â‡fö–B¢—BÓæÖVÒÂ‡Vç6–væVBÆöærÆöær—BÓäÂç6—¦RÂBÓäÂçF–ÆT‚ÂBÓäÂçF–ÆTB“²Ð¢÷5÷Væf—%öÆö6µ÷VæÆö6²‚fu÷&×R“°§Ð§fö–BFÕövg…÷FW†GV×öæW‡B†6öç7B6†"§F‚’°¢–b†u÷FEög&ÖRÓÒÓ"’FEöæ÷FR†æ–ÂÂ“²òò&VBDÕôådåõDU„ETÕf—'7BÂ6ò—BFöW2æ÷B÷fW'&–FRF†R&WVW7@¢u÷FE÷F‚Ò7G&GW‡F‚“²u÷FEöÆ—7BÒ´å4×WF&ÆT'&’'&•Ó²u÷FEöFöæRÒ²u÷FEög&ÖRÒ†Æöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR’²°§Ð§7FF–2fö–BFEöGV×‡fö–B’°¢u÷FEöFöæRÒ°¢'öVæB‚“²–b†uö&Æ—B’²Væ5ö%öVæB†uö&Æ—B“²uö&Æ—BÒæ–Ã²Ð¢×FÅöf–æ—6…öwR‚“°¢7FF–2–CÄÕDÅ&VæFW%—VÆ–æU7FFSâ3²7FF–2–CÄÕDÅ6×ÆW%7FFSâ7°¢–b‚2’°¢7FF–26öç7B6†"§7&5ö×6ÂÐ¢"6–æ6ÇVFRÆÖWFÅ÷7FFÆ–#åÆçW6–æræÖW76RÖWFÃµÆâ ¢'7G'V7Bb²fÆöCBµ·÷6—F–öåÕÓ²fÆöC"Wc²ÓµÆâ ¢'fW'FW‚bg2‡V–çB–Bµ·fW'FW…ö–EÕÒ’²fÆöC"WbÒfÆöC"‚†–BÃÂ’b"Â–Bb"“²bó²òçÒfÆöCB‡Wb¢fÆöC"ƒ"ÂÓ"’²fÆöC"‚ÓÂ’ÂÂ“²òçWbÒWc²&WGW&âó²ÕÆâ ¢&g&vÖVçBfÆöCBg2…b–âµ·7FvUö–åÕÒÂFW‡GW&S&CÆfÆöCâBµ·FW‡GW&Rƒ•ÕÒÂ6×ÆW"2µ·6×ÆW"ƒ•ÕÒ’²fÆöCB2ÒBç6×ÆR‡2Â–âçWb“²&WGW&âfÆöCB†2ç&v"Â“²ÕÆâ#°¢å4W'&÷"¦RÒæ–Ã²–CÄÕDÄÆ–'&'“âÆ–"Ò¶uöFWbæWtÆ–'&'•v—F…6÷W&6S¤‡7&5ö×6Â’÷F–öç3¦æ–ÂW'&÷#¢fUÓ°¢ÕDÅ&VæFW%—VÆ–æTFW67&—F÷"¦BÒ´ÕDÅ&VæFW%—VÆ–æTFW67&—F÷"æWuÓ°¢BçfW'FW„gVæ7F–öâÒ¶Æ–"æWtgVæ7F–öåv—F„æÖS¤'g2%Ó²Bæg&vÖVçDgVæ7F–öâÒ¶Æ–"æWtgVæ7F–öåv—F„æÖS¤&g2%Ó°¢Bæ6öÆ÷$GF6†ÖVçG5³Òç—†VÄf÷&ÖBÒÕDÅ—†VÄf÷&ÖE$t$…Væ÷&Ó°¢2Ò¶uöFWbæWu&VæFW%—VÆ–æU7FFUv—F„FW67&—F÷#¦BW'&÷#¢fUÓ°¢ÕDÅ6×ÆW$FW67&—F÷"§6BÒ´ÕDÅ6×ÆW$FW67&—F÷"æWuÓ²7Ò¶uöFWbæWu6×ÆW%7FFUv—F„FW67&—F÷#§6EÓ°¢–b‚2’²ÔÄôr‚%µ$TEÒDU„ETÕ—VÆ–æRf–ÆVBW2"ÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“²&WGW&ã²Ð¢Ð¢ÔÄôr‚%·FW†GV×Òg&ÖRVÆC¢VÇR6öÆ÷"F&vWG2"Âu÷FEög&ÖRÂ‡Vç6–væVBÆöær–u÷FEöÆ—7Bæ6÷VçB“°¢òòDÕôådåõDU…$sÖ–æFW‚Âââã¢6fR&r'—FW2'’FW‡GW&R–æFW‚†–æ6ÇVF–ær4C²f–ÆS¢VB–âF†RF‚—2&WÆ6VBv—F‚F†R–æFW‚Â7Vff—‚ç&rÂf—'7BÖÆ–æRFW‡B†VFW#¢v–GF‚†V–v‡BFWF‚f÷&ÖB'—FW2×W"×&÷r¢f÷"†6öç7B6†"§ÒvWFVçb‚$DÕôådåõDU…$r"“²bb§²Ò7G&6‡"‡ÂrÂr’ò7G&6‡"‡ÂrÂr’²¢²7G&ÆVâ‡’’°¢–b‚7G&æ6×‡Â&–æfò"ÂB’’²òòDÕôådåõDU…$sÖ–æfó¢F†RFW‡GW&W2&VÖVÖ&W&VB'’DÕôådåõDU„”ädð¢7FF–26†"'Ve³#SeÓ²6—¦U÷BòÒ²f÷"†–çB’Ò²’Âu÷FW†–æfõöã²’²²’ò³Ò6ç&–çFb†'Vb²òÂ6—¦Vöb'VbÒòÂ"W2VÆÇR"Â’ò"Â"¢""Â‡Vç6–væVBÆöærÆöær–u÷FW†–æfõ÷6W%¶•Ò“°¢–b†ò’²7FF–26†"Vçe³3Ó²6ç&–çFb†VçbÂ6—¦VöbVçbÂ"W2"Â'Vb“²6WFVçb‚$DÕôådåõDU…$r"ÂVçbÂ“²ÒvWFVçb‚$DÕôådåõDU…$r"“²ÔÄôr‚%·FW‡&uÒ–æfò(i"W2"Â“²ÒVÇ6R6öçF–çVS°¢Ð¢V–çCcE÷B6W"Ò‡V–çCcE÷B–FöÆÂ‡“²–BòÒFW…övWB‡6W"“°¢–CÄÕDÅFW‡GW&SâBÒ¶ò—4¶–æDöd6Æ73¥´çe¶r6Æ75ÕÒò‚„çe¶r¢–ò’ÓçFW‚¢ó°¢–b‚B’²ÔÄôr‚%·FW‡&uÒ6W#ÒVÆÇRæ÷B–âF†R6W&–ÂF&ÆR"Â6W"“²6öçF–çVS²Ð¢–çB'rÂ&ƒ²å5T–çFVvW"rÒBçv–GF‚Â‚ÒBæ†V–v‡BÂG¢ÒBæFWFƒ°¢6—¦U÷B'ÒBç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖE$t$dfÆöBò‚¢Bç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖE$t$3$fÆöBòb¢C²‡fö–B–'s²‡fö–B–&ƒ°¢6—¦U÷B&÷rÒr¢'Â–ÖrÒ&÷r¢ƒ°¢–CÄÕDÄ'VffW#â'VbÒ¶uöFWbæWt'VffW%v—F„ÆVæwFƒ¦–Ör¢G¢÷F–öç3¤ÕDÅ&W6÷W&6U7F÷&vTÖöFU6†&VEÓ°¢–CÄÕDÄ6öÖÖæD'VffW#â6"Òö6"‚“²–CÄÕDÄ&Æ—D6öÖÖæDVæ6öFW#â"Ò¶6"&Æ—D6öÖÖæDVæ6öFW%Ó°¢ÕDÄ&Æ—D÷F–öâ&òÒBç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖDFWFƒ3$fÆöEõ7FVæ6–Ã‚òÕDÄ&Æ—D÷F–öäFWF„g&öÔFWF…7FVæ6–Â¢ÕDÄ&Æ—D÷F–öäæöæS²òòFWF‚×7FVæ6–Ã¢F†R3"Ö&—BfÆöBFWF‚ÆæRöæÇ¢¶"6÷”g&öÕFW‡GW&S§B6÷W&6U6Æ–6S£6÷W&6TÆWfVÃ£6÷W&6T÷&–v–ã¤ÕDÄ÷&–v–äÖ¶RƒÂÂ’6÷W&6U6—¦S¤ÕDÅ6—¦TÖ¶R‡rÂ‚ÂG¢’Fô'VffW#¦'VbFW7F–æF–öäöfg6WC£FW7F–æF–öä'—FW5W%&÷s§&÷rFW7F–æF–öä'—FW5W$–ÖvS¦–Ör÷F–öç3¦&õÓ°¢¶"VæDVæ6öF–æuÓ²¶6"6öÖÖ—EÓ²¶6"v—EVçF–Ä6ö×ÆWFVEÓ°¢6†"F…³#EÓ²6ç&–çFb‡F‚Â6—¦VöbF‚Âu÷FE÷F‚Â†–çB—6W"“²7G&Æ6B‡F‚Â"ç&r"Â6—¦VöbF‚“°¢d”ÄR¦bÒf÷Vâ‡F‚Â'v""“²–b†b’²g&–çFb†bÂ"VÇRVÇRVÇRVÇRW§UÆâ"Â‡Vç6–væVBÆöær—rÂ‡Vç6–væVBÆöær–‚Â‡Vç6–væVBÆöær–G¢Â‡Vç6–væVBÆöær—Bç—†VÄf÷&ÖBÂ&÷r“²gw&—FR†'Vbæ6öçFVçG2ÂÂ–Ör¢G¢Âb“²f6Æ÷6R†b“²Ð¢ÔÄôr‚%·FW‡&uÒ6W#ÒVÆÇRVÇW‚VÇW‚VÇRf÷&ÖBVÇR(i"W2"Â6W"Â‡Vç6–væVBÆöær—rÂ‡Vç6–væVBÆöær–‚Â‡Vç6–væVBÆöær–G¢Â‡Vç6–væVBÆöær—Bç—†VÄf÷&ÖBÂF‚“°¢Ð¢f÷"„å4'&’¦R–âu÷FEöÆ—7B’°¢–CÄÕDÅFW‡GW&SâBÒU³Ó²V–çCcE÷B6W"Ò¶U³ÒVç6–væVDÆöætÆöæufÇVUÓ²å5T–çFVvW"ÇbÒ¶U³%ÒVç6–væVD–çFVvW%fÇVUÒÂ6ÂÒ¶U³5ÒVç6–væVD–çFVvW%fÇVUÓ°¢òò$B'&—2&R6fVB'’F†RGF6†VBÆ–W"†f–ÆR–æFW‚6W"£²Æ–W"£²ÆWfVÃ²Æ–W"—2æÖVBÆ–¶R$B¢–b‡BçFW‡GW&UG—RÒÕDÅFW‡GW&UG—S$BbbBçFW‡GW&UG—RÒÕDÅFW‡GW&UG—S$D'&’’²ÔÄôr‚%·FW†GV×Ò6W#ÒVÆÇRG—RVÇR6¶—VB"Â6W"Â‡Vç6–væVBÆöær—BçFW‡GW&UG—R“²6öçF–çVS²Ð¢–b†vWFVçb‚$DÕôådåõ4ä$r"’’²òòÇ6ò6fRF†R&r'—FW2†6öçfW'F–ærâ„E"F&vWBFòâ‚Ö&—B–ÖvRG'Væ6FW3²f—'7BÖÆ–æRFW‡B†VFW#¢v–GF‚†V–v‡Bf÷&ÖB'—FW2×W"×&÷r¢å5T–çFVvW"'rÒÔ‚‡Bçv–GF‚ãâÇbÂR’Â&‚ÒÔ‚‡Bæ†V–v‡BãâÇbÂR“°¢6—¦U÷B'ÒBç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖE$t$dfÆöBò‚¢Bç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖE$t$3$fÆöBòb¢Bç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖE#dfÆöBÇÂBç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖE#eVæ÷&Òò"¢C°¢–CÄÕDÄ'VffW#â&"Ò¶uöFWbæWt'VffW%v—F„ÆVæwFƒ§'r¢&‚¢'÷F–öç3¤ÕDÅ&W6÷W&6U7F÷&vTÖöFU6†&VEÓ°¢–CÄÕDÄ6öÖÖæD'VffW#â&2Òö6"‚“²–CÄÕDÄ&Æ—D6öÖÖæDVæ6öFW#â&"Ò·&2&Æ—D6öÖÖæDVæ6öFW%Ó°¢¶&"6÷”g&öÕFW‡GW&S§B6÷W&6U6Æ–6S£6÷W&6TÆWfVÃ¦Çb6÷W&6T÷&–v–ã¤ÕDÄ÷&–v–äÖ¶RƒÂÂ’6÷W&6U6—¦S¤ÕDÅ6—¦TÖ¶R‡'rÂ&‚Â’Fô'VffW#§&"FW7F–æF–öäöfg6WC£FW7F–æF–öä'—FW5W%&÷s§'r¢'FW7F–æF–öä'—FW5W$–ÖvS§'r¢&‚¢'Ó°¢¶&"VæDVæ6öF–æuÓ²·&26öÖÖ—EÓ²·&2v—EVçF–Ä6ö×ÆWFVEÓ°¢6†"'õ³#EÓ²6ç&–çFb‡'òÂ6—¦Vöb'òÂu÷FE÷F‚Â†–çB’‡6W"¢²Çb’“°¢–b†Ræ6÷VçBâB’²6†"F–Å³cEÓ²6ç&–çFb‡F–ÂÂ6—¦VöbF–ÂÂ%÷2S6B"Â¶U³EÒ–çEfÇVUÒ“²7G&Æ6B‡'òÂF–ÂÂ6—¦Vöb'ò“²ÒòòDU…4ä6÷–W2öböæRF&vWC¢öæRf–ÆRW"70¢7G&Æ6B‡'òÂ"ç&r"Â6—¦Vöb'ò“°¢d”ÄR¦bÒf÷Vâ‡'òÂ'v""“²–b†b’²g&–çFb†bÂ"VÇRVÇRVÇRW§UÆâ"Â‡Vç6–væVBÆöær—'rÂ‡Vç6–væVBÆöær—&‚Â‡Vç6–væVBÆöær—Bç—†VÄf÷&ÖBÂ'r¢'“²gw&—FR‡&"æ6öçFVçG2ÂÂ'r¢&‚¢'Âb“²f6Æ÷6R†b“²Ð¢Ð¢–CÄÕDÅFW‡GW&SâbÒ·BæWuFW‡GW&Uf–Wuv—F…—†VÄf÷&ÖC§Bç—†VÄf÷&ÖBFW‡GW&UG—S¤ÕDÅFW‡GW&UG—S$BÆWfVÇ3¤å4Ö¶U&ævR†ÇbÂ’6Æ–6W3¤å4Ö¶U&ævR‡6ÂÂ•Ó°¢å5T–çFVvW"rÒÔ‚‡Bçv–GF‚ãâÇbÂR’Â‚ÒÔ‚‡Bæ†V–v‡BãâÇbÂR“°¢ÕDÅFW‡GW&TFW67&—F÷"¦öBÒ´ÕDÅFW‡GW&TFW67&—F÷"FW‡GW&S$DFW67&—F÷%v—F…—†VÄf÷&ÖC¤ÕDÅ—†VÄf÷&ÖE$t$…Væ÷&Òv–GFƒ§r†V–v‡C¦‚Ö—ÖVC¤äõÓ°¢öBçW6vRÒÕDÅFW‡GW&UW6vU&VæFW%F&vWBÂÕDÅFW‡GW&UW6vU6†FW%&VC²öBç7F÷&vTÖöFRÒÕDÅ7F÷&vTÖöFU&—fFS°¢–CÄÕDÅFW‡GW&SâòÒ¶uöFWbæWuFW‡GW&Uv—F„FW67&—F÷#¦öEÓ°¢–CÄÕDÄ6öÖÖæD'VffW#â6"Òö6"‚“°¢ÕDÅ&VæFW%74FW67&—F÷"§'Ò´ÕDÅ&VæFW%74FW67&—F÷"&VæFW%74FW67&—F÷%Ó°¢'æ6öÆ÷$GF6†ÖVçG5³ÒçFW‡GW&RÒó²'æ6öÆ÷$GF6†ÖVçG5³ÒæÆöD7F–öâÒÕDÄÆöD7F–öäFöçD6&S²'æ6öÆ÷$GF6†ÖVçG5³Òç7F÷&T7F–öâÒÕDÅ7F÷&T7F–öå7F÷&S°¢–CÄÕDÅ&VæFW$6öÖÖæDVæ6öFW#âVâÒ¶6"&VæFW$6öÖÖæDVæ6öFW%v—F„FW67&—F÷#§'Ó°¢¶Vâ6WE&VæFW%—VÆ–æU7FFS§5Ó²¶Vâ6WDg&vÖVçEFW‡GW&S§bD–æFWƒ£Ó²¶Vâ6WDg&vÖVçE6×ÆW%7FFS§7D–æFWƒ£Ó°¢¶VâG&u&–Ö—F—fW3¤ÕDÅ&–Ö—F—fUG—UG&–ævÆRfW'FW…7F'C£fW'FW„6÷VçC£5Ó²¶VâVæDVæ6öF–æuÓ²¶6"6öÖÖ—EÓ²¶6"v—EVçF–Ä6ö×ÆWFVEÓ°¢6†"F…³#EÓ²6ç&–çFb‡F‚Â6—¦VöbF‚Âu÷FE÷F‚Â†–çB’‡6W"¢²6Â¢²Çb’“°¢–b†Ræ6÷VçBâB’²6ç&–çFb‡F‚Â6—¦VöbF‚Âu÷FE÷F‚Â“²6†"¦F÷BÒ7G'&6‡"‡F‚Ârâr“²6†"W‡E³eÒÒ"#²–b†F÷B’²7G&Æ7’†W‡BÂF÷BÂ6—¦VöbW‡B“²¦F÷BÒ²Ð¢6†"F–Å³cEÓ²6ç&–çFb‡F–ÂÂ6—¦VöbF–ÂÂ%÷2S6EòVÆÇUòVÇRW2"Â¶U³EÒ–çEfÇVUÒÂ‡Vç6–væVBÆöærÆöær—6W"Â‡Vç6–væVBÆöær–ÇbÂW‡B“²7G&Æ6B‡F‚ÂF–ÂÂ6—¦VöbF‚“²ÒòòDU…4ä6÷¢ÔÄôr‚%·FW†GV×Ò6W#ÒVÆÇRÇcÒVÇRÆ–W"VÇRf÷&ÖBVÇRVÇW‚VÇR"Â6W"Â‡Vç6–væVBÆöær–ÇbÂ‡Vç6–væVBÆöær—6ÂÂ‡Vç6–væVBÆöær—Bç—†VÄf÷&ÖBÂ‡Vç6–væVBÆöær—rÂ‡Vç6–væVBÆöær–‚“°¢6†÷B†òÂF‚ÂÂåTÄÂ“°¢Ð¢¶u÷FEöÆ—7B&VÖ÷fTÆÄö&¦V7G5Ó°§Ð¢òòDÕôådåõDU…4äÖ–âÖg&ÖR–æFW‚Ââââ‡FövWF†W"v—F‚F†Rg&ÖRçVÖ&W"æBF‚öbDÕôådåõDU„ETÕ“¢26ööâ2F†W6R&VæFW"76W2VæBÂ6÷’F†V—"6öÆ÷"GF6†ÖVçG2æB6fRF†VÒFövWF†W"v—F‚F†RVæBÖöbÖg&ÖRGV× ¢òò‡F†R6ÖRF&vWB—2ögFVâ6ÆV&VBæB&WW6VBv—F†–âg&ÖRÂ6òF†RVæBÖöbÖg&ÖR7FFR6ææ÷B&VfÆV7BF†R÷WGWBöb–çFW&ÖVF–FR76W2“²6fVB2ÅDU„ETÕF‚v—F‚VBÒå÷3Æ–âÖg&ÖR–æFWƒåóÇ6W&–ÃåóÆÆWfVÃâçæp§7FF–2–çBu÷G5ö–G…³cEÒÂu÷G5öâÒÓÂu÷G5ööã²7FF–2V–çCcE÷Bu÷G5öf—'7BÂu÷G5ö7W&bÒãVÆÃ°§7FF–2õ÷7G&öær–CÄÕDÅFW‡GW&Sâu÷G5öGE³…Ó²7FF–2V–çCcE÷Bu÷G5÷6W%³…Ó²7FF–2å5T–çFVvW"u÷G5öÇe³…Ó²7FF–2–çBu÷G5öæ°§fö–BG5öæ÷FR„ÕDÃE&VæFW%74FW67&—F÷"§BÂå5T–çFVvW"æ2’°¢–b†u÷G5öâÂ’²u÷G5öâÒ²6öç7B6†"¦RÒvWFVçb‚$DÕôådåõDU…4ä"“°¢f÷"†6öç7B6†"§ÒS²bb§bbu÷G5öâÂcC²Ò7G&6‡"‡ÂrÂr’ò7G&6‡"‡ÂrÂr’²¢²7G&ÆVâ‡’’u÷G5ö–G…¶u÷G5öâ²µÒÒFö’‡“²Ð¢u÷G5ööâÒ²–b‚u÷G5öâÇÂu÷FEög&ÖRÂÇÂu÷FEöFöæR’&WGW&ã°¢V–çCcE÷BbÒFöÖ–5öÆöB‚fuö7W%ög&ÖR“²–b†bÒu÷G5ö7W&b’²u÷G5ö7W&bÒc²u÷G5öf—'7BÒu÷7FE÷73²Ð¢–b‚†Æöær–bÒu÷FEög&ÖR’&WGW&ã°¢–çB–G‚Ò†–çB’†u÷7FE÷72Òu÷G5öf—'7B“²f÷"†–çB’Ò²’Âu÷G5öã²’²²’–b†u÷G5ö–G…¶•ÒÓÒ–G‚’u÷G5ööâÒ–G‚²°¢u÷G5öæÒ²–b‚u÷G5ööâ’&WGW&ã°¢f÷"„å5T–çFVvW"’Ò²’Âæ2bb’Âƒ²’²²’²–CÄÕDÅFW‡GW&SâBÒBæ6öÆ÷$GF6†ÖVçG5¶•ÒçFW‡GW&S²–b‚B’6öçF–çVS°¢u÷G5öGE¶u÷G5öæÒÒC²u÷G5÷6W%¶u÷G5öæÒÒ"ç'Bæ6öÆ÷%¶•Òò"ç'Bæ6öÆ÷%¶•ÒÓç6W&–Â¢²u÷G5öÇe¶u÷G5öæÒÒBæ6öÆ÷$GF6†ÖVçG5¶•ÒæÆWfVÃ²u÷G5öæ²³²Ð§Ð§fö–BG5ögFW"‡fö–B’°¢–b‚u÷G5ööâ’&WGW&ã°¢òòF†R6÷’vöW2–çFòF†R&WÆ’7G&VÒ„ÖWFÂB6÷’Væ6öFW"ÂgFW"F†R72F†B§W7BVæFVB’â6öÖÖæB'VffW"öbF†R6Æ76–2VWVRÂ6öÖÖ—GFVB†W&RÂ&â&Vf÷&RF†P¢òòÖWFÂB6öÖÖæB'VffW"†öÆF–ærF†R72v2WfVâ6öÖÖ—GFVBÂ6òWfW'’6æ6†÷B6†÷vVBF†RGF6†ÖVçG22F†W’vW&R&Vf÷&RF†R72†f÷VæBöâFV'2öbF†R¶–ævFöÒ¢f÷"†–çB’Ò²’Âu÷G5öæ²’²²’°¢–CÄÕDÅFW‡GW&SâBÒu÷G5öGE¶•Ó²–b‡BçFW‡GW&UG—RÒÕDÅFW‡GW&UG—S$BbbBçFW‡GW&UG—RÒÕDÅFW‡GW&UG—S$D'&’’6öçF–çVS²òò$B'&—2F¶R6Æ–6R ¢ÕDÅFW‡GW&TFW67&—F÷"¦BÒ´ÕDÅFW‡GW&TFW67&—F÷"FW‡GW&S$DFW67&—F÷%v—F…—†VÄf÷&ÖC§Bç—†VÄf÷&ÖBv–GFƒ§Bçv–GF‚†V–v‡C§Bæ†V–v‡BÖ—ÖVC§BæÖ—ÖÆWfVÄ6÷VçBâÓ°¢BçW6vRÒÕDÅFW‡GW&UW6vU6†FW%&VC²Bç7F÷&vTÖöFRÒÕDÅ7F÷&vTÖöFU&—fFS²BæÖ—ÖÆWfVÄ6÷VçBÒBæÖ—ÖÆWfVÄ6÷VçC°¢–CÄÕDÅFW‡GW&Sâ2Ò¶uöFWbæWuFW‡GW&Uv—F„FW67&—F÷#¦EÓ²&W5öFB†2“°¢å5T–çFVvW"ÇbÒu÷G5öÇe¶•ÒÂrÒÔ‚‚„å5T–çFVvW"“ÂBçv–GF‚ãâÇb’Â‚ÒÔ‚‚„å5T–çFVvW"“ÂBæ†V–v‡BãâÇb“°¢Væ5ö%ö6÷•÷FW‡GW&R†&Æ—B‚’ÂBÂÂÇbÂÕDÄ÷&–v–äÖ¶RƒÂÂ’ÂÕDÅ6—¦TÖ¶R‡rÂ‚Â’Â2ÂÂÇbÂÕDÄ÷&–v–äÖ¶RƒÂÂ’“°¢¶u÷FEöÆ—7BFDö&¦V7C¤¶2Â†u÷G5÷6W%¶•Ò’Â†u÷G5öÇe¶•Ò’ÂÂ†u÷G5ööâÒ•ÕÓ²òòF†RVæBÖöbÖg&ÖRf–VÆG2‡FW‡GW&RÂ6W&–ÂÂÆWfVÂÂÆ–W"’ÇW2F†R–âÖg&ÖR72–æFWƒ¢6fVB2ÇF‚v—F‚VBÒå÷3Æ–æFWƒåóÇ6W&–ÃåóÆÆWfVÃâçæp¢Ð¢u÷G5ööâÒ°§Ð¢òò&–æröb7GVÂ&W6VçFF–öâF–ÖW3¢&W6VçB6WVVæ6RçVÖ&W"Óâ&W6VçFVEF–ÖR‡6V6öæG3²ÓÒG&÷VB'’F†R7—7FVÒÂæ÷B&W6VçFVB¢6FVf–æR$”är#@§G—VFVb7G'V7B²V–çCcE÷Bã²F÷V&ÆRC²Ò&W5öVçC°§7FF–2&W5öVçBu÷&W5µ$”äuÓ°¢òòvWBF†R7GVÂ&W6VçFF–öâF–ÖRöbF†RçF‚&W6VçB‡&WGW&ç2–bæ÷B&VG’ÂÓ–bG&÷VB¦F÷V&ÆR×FÅ÷&W6VçFVE÷F–ÖR‡V–çCcE÷Bâ’²&W5öVçBRÒu÷&W5¶âR$”äuÓ²&WGW&âRæâÓÒâòRçB¢²Ð¢òò&–æröbF†RF—7Æ’ÖÆ–æ²F–6²F†BFöö²V6‚&W6VçB‡w&—GFVâöâF†RF—7Æ’ÖÆ–æ²6ÆÆ&6²F‡&VBÂ&VB‚&W6VçG2ÆFW"f÷"DÕôådåõE5bÂ2u÷&W2“¢—G2F&vWBF—7Æ’F–ÖP¢òò‡6V6öæG2Â47W'&VçDÖVF–F–ÖR&6R’Â†÷rÆöær&Vf÷&R—BF†R6ÆÆ&6²&âÂF†R6ÆÆ&6·26–æ6RF†R&Wf–÷W2&W6VçBw2ƒÒæòF–6²v—F†÷WBg&ÖR’æBF†RÆöævW7B–çFW'fÂ&WGvVVâF†VÐ§G—VFVb7G'V7B²V–çCcE÷Bã²F÷V&ÆRF&vWBÂÆVEö×2Âvö×3²–çBF–6·3²ÒÆ–æµöVçC°§7FF–2Æ–æµöVçBuöÆ–æµ÷&–æuµ$”äuÓ°¦–çB×FÅöÆ–æµ÷F–6²‡V–çCcE÷BâÂF÷V&ÆR§F&vWBÂF÷V&ÆR¦ÆVEö×2Â–çB§F–6·2ÂF÷V&ÆR¦vö×2’°¢Æ–æµöVçBRÒuöÆ–æµ÷&–æu¶âR$”äuÓ²–b†RæâÒâ’&WGW&â°¢§F&vWBÒRçF&vWC²¦ÆVEö×2ÒRæÆVEö×3²§F–6·2ÒRçF–6·3²¦vö×2ÒRævö×3²&WGW&â°§Ð§7FF–2ôFöÖ–2V–çCcE÷BuöG&v&ÆUöç3²òò67V×VÆFVBvÆÂ6Æö6²v—F–ærf÷"æW‡DG&v&ÆR–â&W6VçB‡F–Ö–ær–ç7G'VÖVçFF–öâ§V–çCcE÷B×FÅöG&v&ÆUöç5÷F¶R‡fö–B’²&WGW&âFöÖ–5öW†6†ævR‚fuöG&v&ÆUöç2Â“²Ð¢òòÖ†–×VÒ&Vg&W6‚&FR„‡¢’öbF†R67&VVâv†VâF†W&R—2v–æF÷s²&WGW&ç2v—F†÷WBv–æF÷râÖ2W6W26÷&Tw&†–72F—7Æ’ÖöFW2‡F‡&VB×6fR2–çFW&f6S²F†R&W6VçB6öæf–wW&F–öâ—2&VBöâF†RvÖRF‡&VB¦–çB×FÅöF—7Æ•ö‡¢‡fö–B’°¢–b‚g&öçEöÖWFÅöÆ–W"‚’’&WGW&â°¢6–bD$tUEôõ5ôõ5€¢4tF—7Æ”ÖöFU&VbÒÒ4tF—7Æ”6÷”F—7Æ”ÖöFR„4tÖ–äF—7Æ””B‚’“°¢F÷V&ÆR‡¢ÒÒò4tF—7Æ”ÖöFTvWE&Vg&W6…&FR†Ò’¢²–b†Ò’4tF—7Æ”ÖöFU&VÆV6R†Ò“°¢&WGW&â†–çB’†‡¢²ãR“°¢6VÇ6P¢6Æ722Òå46Æ74g&öÕ7G&–ær„%T•67&VVâ"“²–B2Ò2ò¶2fÇVTf÷$¶W“¤&Ö–å67&VVâ%Ò¢æ–Ã°¢&WGW&â2òµ·2fÇVTf÷$¶W“¤&Ö†–×VÔg&ÖW5W%6V6öæB%Ò–çEfÇVUÒ¢°¢6VæF–`§Ð§7FF–2ôFöÖ–2V–çCcE÷Bu÷&W6VçE÷7FÆÅöç3²òòf÷"Væ—BFW7G3¢âW‡G&W6R&Vf÷&RV6‚&W6VçBöâF†R&W6VçBF‡&VB‡6–×VÆF–ærG&v&ÆRF†B7F—2Væf–Æ&ÆR§7FF–24ÖWFÄÆ–W"¦u÷&W6VçEöÆ–W%÷&W6–FVçC°§fö–B×FÅ÷&W6VçE÷FW7E÷7FÆÂ‡V–çCcE÷Bç2’²u÷&W6VçE÷7FÆÅöç2Òç3²Ð§7FF–2fö–B&W6VçE÷—VÆ–æR„ÕDÅ—†VÄf÷&ÖBf÷&ÖB’°¢–b‚uö&Æ—E÷6òÇÂuö&Æ—Eöf×BÒf÷&ÖB’°¢å4W'&÷"¦RÒæ–Ã°¢–CÄÕDÄÆ–'&'“âÆ–"Ò†÷7EöÆ–"‚'&W6VçB"ÂfR“°¢–b‚Æ–"’ÔD”R‚'&W6VçB6†FW"6ö×–ÆRf–ÆVBW2"ÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“°¢ÕDÅ&VæFW%—VÆ–æTFW67&—F÷"§BÒ´ÕDÅ&VæFW%—VÆ–æTFW67&—F÷"æWuÓ°¢BçfW'FW„gVæ7F–öâÒ¶Æ–"æWtgVæ7F–öåv—F„æÖS¤'g2%Ó²Bæg&vÖVçDgVæ7F–öâÒ¶Æ–"æWtgVæ7F–öåv—F„æÖS¤&g2%Ó°¢Bæ6öÆ÷$GF6†ÖVçG5³Òç—†VÄf÷&ÖBÒf÷&ÖC°¢uö&Æ—E÷6òÒ¶uöFWbæWu&VæFW%—VÆ–æU7FFUv—F„FW67&—F÷#§BW'&÷#¢fUÓ°¢–b‚uö&Æ—E÷6ò’ÔD”R‚'&W6VçB—VÆ–æRf–ÆVBW2"ÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“°¢uö&Æ—Eöf×BÒf÷&ÖC°¢ÕDÅ6×ÆW$FW67&—F÷"§6BÒ´ÕDÅ6×ÆW$FW67&—F÷"æWuÓ²6BæÖ–äf–ÇFW"Ò6BæÖtf–ÇFW"ÒÕDÅ6×ÆW$Ö–äÖtf–ÇFW$Æ–æV#²6Bç7W÷'D&wVÖVçD'VffW'2Ò”U3°¢uö&Æ—E÷6×Ò¶uöFWbæWu6×ÆW%7FFUv—F„FW67&—F÷#§6EÓ°¢Ð§Ð¢òò&W6VçB6×ÆW2öæÇ’F†RfÆ–B6÷W&6R&V7FævÆRFV6Æ&VB'’ådã²¶WB6W&FRg&öÒF†R&r7v6†–â&VF&6²67&VVç6†÷BâVFvW2&R6Æ×VBFòF†RFW†VÂ6VçFW'2öbF†RfÆ–B&VÂ6ò&–Æ–æV"FöW2æ÷B&VBF†R&Æ6²FW†VÇ2÷WG6–FRà¢òò&W6VçB6×ÆW2F†R&r'—FW27F÷&VB–âF†R7v6†–âFW‡GW&R‡F†R7v—F6‚w266âÖ÷WB6VæG2F†R7F÷&VB'—FW22F†W’&R“¢â5$t"6÷W&6R—2&WÆ6VB'’æöâ×5$t"f–WröbF†R6ÖR7F÷&vRÀ¢òò÷F†W'v—6R6×Æ–ærFV6öFW2FòÆ–æV"f—'7BÂF†R$u$…Væ÷&ÒÆ–W"FöW2æ÷BVæ6öFRv–âÂæBF†Rv†öÆR–7GW&R6öÖW2÷WBF&²â÷F†W"f÷&ÖG2&R&WGW&æVB2F†W’&R„Ö&–òw27v6†–â—2$t$‚æBFöW2æ÷B6öÖR†W&R’à¢òòf–Wr66†RöbBVçG&–W3¢7v6†–âFW‡GW&W2Æ—fRÆöæræB&RfWs²f–Wr†öÆG2—G2&VçBÂ6òâWVÂ&VçBö–çFW"ÖVç2F†R6ÖRö&¦V7Bâ&WÆ6VBf–Wr—2&VÆV6VBgFW"F†R7V&Ö—76–öç2F†BW6VB—B6ö×ÆWFR„ÖWFÂBFöW2æ÷B&WF–â&W6÷W&6W2§7FF–2–CÄÕDÅFW‡GW&Sâ&W6VçE÷&u÷7&2†–CÄÕDÅFW‡GW&Sâ7&2’°¢ÕDÅ—†VÄf÷&ÖBbÒ7&2ç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖE$t$…Væ÷&Õ÷5$t"òÕDÅ—†VÄf÷&ÖE$t$…Væ÷&Ò¢7&2ç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖD$u$…Væ÷&Õ÷5$t"òÕDÅ—†VÄf÷&ÖD$u$…Væ÷&Ò¢ÕDÅ—†VÄf÷&ÖD–çfÆ–C°¢–b‡bÓÒÕDÅ—†VÄf÷&ÖD–çfÆ–B’&WGW&â7&3°¢7FF–2õ÷7G&öær–CÄÕDÅFW‡GW&Sâf–Ww5³EÓ²7FF–2–çBæW‡C²7FF–2F‡&VEö×WFW…÷B×RÒD…$TEôÕUDU…ô”ä•D”Ä•¤U#°¢F‡&VEö×WFW…öÆö6²‚f×R“°¢–CÄÕDÅFW‡GW&SâbÒæ–Ã°¢f÷"†–çB’Ò²’ÂBbbc²’²²’–b‡f–Ww5¶•Òbbf–Ww5¶•Òç&VçEFW‡GW&RÓÒ7&2’bÒf–Ww5¶•Ó°¢–b‚bbb‡bÒ·7&2æWuFW‡GW&Uf–Wuv—F…—†VÄf÷&ÖC§eÒ’’²–b‡f–Ww5¶æW‡EÒ’FVfW%öö&¥÷Vç&Vb‡f–Ww5¶æW‡EÒ“²f–Ww5¶æW‡EÒÒc²æW‡BÒ†æW‡B²’b3²Ð¢F‡&VEö×WFW…÷VæÆö6²‚f×R“°¢–b‚b’²$TEôôä4Rƒc’Â'&W6VçC¢6ææ÷B7&VFRæöâ×5$t"f–WröbF†R5$t"7v6†–âFW‡GW&R†f÷&ÖBVÇR“²&W6VçF–ærF†RFV6öFVBfÇVW2"Â‡Vç6–væVBÆöær—7&2ç—†VÄf÷&ÖB“²&WGW&â7&3²Ð¢&WGW&âc°§Ð¢òòF†R&V7FævÆRöbF†R6çf2†Gr‚F‚’F†R–7GW&R†7r‚6‚’—2G&vâ–çFó¢F†R–7GW&Rw27V7B&F–ò¶WBæB6VçG&VBÂ&Æ6²&'2–âF†R&W7B†æò7G&WF6†–ærv†VâF†R67&VVâw2&F–ð¢òòF–ffW'2g&öÒF†RvÖRw2ÂRærâ•†öæR&÷WB’ãS£’v–ç7BF†RvÖRw2c£’“²v†VâF†R&F–÷2F–ffW"'’VæFW"R‡F†R&VÖ–æFW"g&öÒvÖRv–GF‚F†B—2×VÇF—ÆRöb‚–âF†P¢òògVÆÂ×67&VVâ6WGWÂRærâ“S'ƒ“v–ç7B#ƒc‡ƒ3#F–ffW'2'’ãrR’F†Rv†öÆR6çf2ÂÆVf–æræò&÷&FW"öbfWr—†VÇ0§7FF–2ÕDÅf–Ww÷'B&W6VçE÷f–Ww÷'B†F÷V&ÆR7rÂF÷V&ÆR6‚ÂF÷V&ÆRGrÂF÷V&ÆRF‚’°¢F÷V&ÆR2ÒfÖ–â†Grò7rÂF‚ò6‚“°¢ÕDÅf–Ww÷'BgÒ¶fÆö÷"‚†GrÒ7r¢2’ò"’ÂfÆö÷"‚†F‚Ò6‚¢2’ò"’Â&÷VæB†7r¢2’Â&÷VæB†6‚¢2’ÂÂÓ°¢–b†f'2‚†GròF‚’ò†7rò6‚’Ò’Âã’gÒ„ÕDÅf–Ww÷'B—³ÂÂGrÂF‚ÂÂÓ°¢&WGW&âg°§Ð¢òòF†RfÆ–B6÷W&6R&V7FævÆRöb&W6VçB‡‚Â’Âv–GF‚Â†V–v‡C²F†R7&÷öbçfåv–æF÷u6WD7&÷Â¦W&òÒF†Rv†öÆRFW‡GW&R“²v†Vâ—BÆ–W2÷WG6–FRF†RFW‡GW&P§7FF–2–çB&W6VçEö7&÷÷&V7B†–CÄÕDÅFW‡GW&Sâ7&2Â6öç7B–çB7&÷³EÒÂå5T–çFVvW"%³EÒ’°¢%³ÒÒ²%³ÒÒ²%³%ÒÒ7&2çv–GFƒ²%³5ÒÒ7&2æ†V–v‡C°¢–b‚7&÷ÇÂ†7&÷³%ÒÇÂ7&÷³5Ò’’&WGW&â°¢–b†7&÷³ÒÂÇÂ7&÷³ÒÂÇÂ7&÷³%ÒÃÒÇÂ7&÷³5ÒÃÒÇÀ¢‡V–çCcE÷B–7&÷³Ò²7&÷³%Òâ7&2çv–GF‚ÇÂ‡V–çCcE÷B–7&÷³Ò²7&÷³5Òâ7&2æ†V–v‡B’°¢ÔÄôr‚%µ$TEÒ&W6VçB7&÷÷WBöb&÷VæG3¢VBÂVBVG‚VBò6÷W&6RFW‡GW&RVÇW‚VÇR"Â7&÷³ÒÂ7&÷³ÒÂ7&÷³%ÒÂ7&÷³5ÒÂ‡Vç6–væVBÆöær—7&2çv–GF‚Â‡Vç6–væVBÆöær—7&2æ†V–v‡B“²&WGW&â°¢Ð¢%³ÒÒ7&÷³Ó²%³ÒÒ7&÷³Ó²%³%ÒÒ7&÷³%Ó²%³5ÒÒ7&÷³5Ó²&WGW&â°§Ð§7FF–2–çB&W6VçEöVæ6öFR†–CÄÕDÄ6öÖÖæD'VffW#â6"Â–CÄÕDÅFW‡GW&Sâ7&2Â–CÄÕDÅFW‡GW&SâG7BÂ6öç7B–çB7&÷³EÒ’°¢å5T–çFVvW"%³EÓ²–b‚&W6VçEö7&÷÷&V7B‡7&2Â7&÷Â"’’&WGW&â°¢å5T–çFVvW"7‚Ò%³ÒÂ7’Ò%³ÒÂ7rÒ%³%ÒÂ6‚Ò%³5Ó°¢7FF–2V–çCcE÷BÆ7Eö7&÷³…Ó²V–çCcE÷BÖWF³…ÒÒ·7&2çv–GF‚Â7&2æ†V–v‡BÂ7‚Â7’Â7rÂ6‚ÂG7Bçv–GF‚ÂG7Bæ†V–v‡GÓ°¢–b†ÖVÖ6×†ÖWFÂÆ7Eö7&÷Â6—¦VöbÖWF’’²ÖVÖ7’†Æ7Eö7&÷ÂÖWFÂ6—¦VöbÖWF“°¢ÔÄôr‚'&W6VçB7&÷¢6÷W&6RFW‡GW&RVÇW‚VÇRÂfÆ–BVÇRÂVÇRVÇW‚VÇR(i"F&vWBVÇW‚VÇR"Â‡Vç6–væVBÆöær—7&2çv–GF‚Â‡Vç6–væVBÆöær—7&2æ†V–v‡BÀ¢‡Vç6–væVBÆöær–7‚Â‡Vç6–væVBÆöær–7’Â‡Vç6–væVBÆöær–7rÂ‡Vç6–væVBÆöær–6‚Â‡Vç6–væVBÆöær–G7Bçv–GF‚Â‡Vç6–væVBÆöær–G7Bæ†V–v‡B“²Ð¢&W6VçE÷—VÆ–æR†G7Bç—†VÄf÷&ÖB“°¢ÕDÅ&VæFW%74FW67&—F÷"§'Ò´ÕDÅ&VæFW%74FW67&—F÷"&VæFW%74FW67&—F÷%Ó°¢F÷V&ÆRGrÒG7Bçv–GF‚ÂF‚ÒG7Bæ†V–v‡C°¢ÕDÅf–Ww÷'BgÒ&W6VçE÷f–Ww÷'B†7rÂ6‚ÂGrÂF‚“°¢–çBgVÆÂÒgæ÷&–v–å‚ÓÒbbgæ÷&–v–å’ÓÒbbgçv–GF‚ÓÒGrbbgæ†V–v‡BÓÒFƒ°¢7FF–2F÷V&ÆRÆ7E÷g³EÓ²òòÆöröæRÆ–æRv†VâF†Rf–Ww÷'B6†ævW2‡W6VB'’F†R&VÂÖFWf–6R67&VVâÖÆWfVÂ66WFæ6R67&VVåö6†V6²ç’Fò6†V6²F†R7V7B&F–ò¢–b†ÖVÖ6×†Æ7E÷gÂggÂ6—¦VöbÆ7E÷g’’²ÖVÖ7’†Æ7E÷gÂggÂ6—¦VöbÆ7E÷g“°¢ÔÄôr‚'&W6VçBf–Ww÷'BRãbÂRãbRãg‚Rãbò6çf2Rãg‚Rãb†vÖR–ÖvRVÇW‚VÇR’"Âgæ÷&–v–å‚Âgæ÷&–v–å’Âgçv–GF‚Âgæ†V–v‡BÂGrÂF‚Â‡Vç6–væVBÆöær—7&2çv–GF‚Â‡Vç6–væVBÆöær—7&2æ†V–v‡B“²Ð¢'æ6öÆ÷$GF6†ÖVçG5³ÒçFW‡GW&RÒG7C²'æ6öÆ÷$GF6†ÖVçG5³ÒæÆöD7F–öâÒgVÆÂòÕDÄÆöD7F–öäFöçD6&R¢ÕDÄÆöD7F–öä6ÆV#°¢'æ6öÆ÷$GF6†ÖVçG5³Òæ6ÆV$6öÆ÷"ÒÕDÄ6ÆV$6öÆ÷$Ö¶RƒÂÂÂ“°¢'æ6öÆ÷$GF6†ÖVçG5³Òç7F÷&T7F–öâÒÕDÅ7F÷&T7F–öå7F÷&S°¢–CÄÕDÅ&VæFW$6öÖÖæDVæ6öFW#âVâÒ¶6"&VæFW$6öÖÖæDVæ6öFW%v—F„FW67&—F÷#§'Ó°¢¶Vâ6WEf–Ww÷'C§gÓ°¢¶Vâ6WE&VæFW%—VÆ–æU7FFS¦uö&Æ—E÷6õÓ²¶Vâ6WDg&vÖVçEFW‡GW&S§&W6VçE÷&u÷7&2‡7&2’D–æFWƒ£Ó²¶Vâ6WDg&vÖVçE6×ÆW%7FFS¦uö&Æ—E÷6×D–æFWƒ£Ó°¢fÆöBWe³…ÒÒ²†fÆöB–7‚ò7&2çv–GF‚Â†fÆöB–7’ò7&2æ†V–v‡BÂ†fÆöB–7rò7&2çv–GF‚Â†fÆöB–6‚ò7&2æ†V–v‡BÀ¢†7‚²ãVb’ò7&2çv–GF‚Â†7’²ãVb’ò7&2æ†V–v‡BÂ†7‚²7rÒãVb’ò7&2çv–GF‚Â†7’²6‚ÒãVb’ò7&2æ†V–v‡GÓ°¢–b†FöÖ–5öÆöEöW‡Æ–6—B‚fuö÷&–v–åöÆÂÂÖVÖ÷'•ö÷&FW%÷&VÆ†VB’’²We³Ò³ÒWe³5Ó²We³5ÒÒ×We³5Ó²ÒòòÆ÷vW"ÖÆVgB÷&–v–ã¢F†Rv–æF÷r–ÖvR—27F÷&VB&÷GFöÒ×W ¢¶Vâ6WDg&vÖVçD'—FW3§WbÆVæwFƒ§6—¦VöbWbD–æFWƒ£Ó°¢¶VâG&u&–Ö—F—fW3¤ÕDÅ&–Ö—F—fUG—UG&–ævÆRfW'FW…7F'C£fW'FW„6÷VçC£5Ó°¢¶VâVæDVæ6öF–æuÓ°¢&WGW&â°§Ð¢òòÒÒÒÒÒÒÒÒÒÒW66Æ–ær–âF†R&W6VçB„DÕõ$U4TåEõU44ÄR’ÒÒÒÒÒÒÒÒÒÐ¢òò÷BÖ–â7F–Â66Æ–æröbF†RfÆ–B6öçFVçBFò—G2ÆWGFW&&÷†VBf–Ww÷'Bâ6÷’F†R7&÷Fòâ÷&–v–âÖÆ–væVBFW‡GW&RÂ66ÆR–âW&6WGVÂÖöFR‡&W6VçE÷&u÷7&27WÆ–W2vÖÖ¢òò'—FW2’ÂF†VâG&r£v—F‚F†RW7VÂ÷&–v–âæB÷WGWBf÷&ÖBâVWVR&'&–W'26W&–Æ—¦R&÷F‚F†Rg&ÖRw2&VæFW&–æræB&WW6RöbF†R66ÆW"w2FW‡GW&W27&÷72&W6VçG2à¢òò&WÆ6–ærF†÷6RFW‡GW&W2v—G2f÷"V&Æ–W"&W6VçG3²F‡&÷GFÆR&WÆ6VÖVçBGW&–ær&W6—¦–ærÂæB¶VW7FæF&B66Æ–ærf÷"&VgW6VB6—¦W2âöfg67&VVâ6GW&W2÷vâ6W&FR7FFRà¢6FVf–æRe…õ$TÔ´Uõ$U4TåE23 §G—VFVb7G'V7B²–CÄÕDÃDe…7F–Å66ÆW#â63²–CÄÕDÅFW‡GW&Sâ–âÂ÷WC²V–çCcE÷BÖFUöC²–çBÖFS²å5T–çFVvW"&VgW6VE³eÓ²Òg…÷7FFS°§7FF–2g…÷7FFRuög…÷&W6VçC²òòöæÇ’F†R&W6VçBF‡&VBVæ6öFW2÷"&WÆ6W2F†R66ÆW §7FF–2–CÄÕDÃD6ö×–ÆW#âuög…ö6ö×°§7FF–2–çBuög…öÖöFRÒÓ²òòÒÖWFÆg‚ÂÒöf`§7FF–2–çBg…öÖöFR‡fö–B’°¢–b†uög…öÖöFRãÒ’&WGW&âuög…öÖöFS°¢6öç7B6†"¦RÒvWFVçb‚$DÕõ$U4TåEõU44ÄR"“²–çBbÒ°¢–b†Rbb‚7G&6×†RÂ&ÖWFÆg‚"’ÇÂ7G&6×†RÂ#"’’’bÒ°¢VÇ6R–b†Rbb‚7G&6×†RÂ&öfb"’ÇÂ7G&6×†RÂ#"’’’bÒ°¢VÇ6R–b†Rbb¦R’ÔÄôr‚%µ$TEÒDÕõ$U4TåEõU44ÄSÒW3¢ÖWFÆg‚÷"öfc²¶VW–ærW2"ÂRÂbò&ÖWFÆg‚"¢&öfb"“°¢–b‡bbb†u÷Bbb´ÕDÄe…7F–Å66ÆW$FW67&—F÷"7W÷'G4ÖWFÃDeƒ¦uöFWeÒ’’²ÔÄôr‚'&W6VçBW66Æ–æs¢F†—2uR†2æòÖWFÂBÖWFÄe‚7F–Â66ÆW#²F†R&W6VçB66ÆW2&–Æ–æV&Ç’"“²bÒ²Ð¢ÔÄôr‚'&W6VçBW66Æ–ær„DÕõ$U4TåEõU44ÄR“¢W2"Âbò&ÖWFÆg‚ÂÖWFÄe‚w27F–Â66ÆW"v†W&RF†R–7GW&R—26†÷vâÆ&vW"F†âF†RvÖR&VæFW'2—B"¢&öfbÂ&–Æ–æV""“°¢&WGW&âuög…öÖöFRÒc°§Ð¢òòÖWFÄe‚66ÆW2F†R6öçFVçB†7r‚6‚’Fò—G2&V7FævÆRöâF†R6çf2‡g’v†VâF†R&V7FævÆR—2BÆV7B2Æ&vR–â&÷F‚F—&V7F–öç2æBÆ&vW"–âöæP§7FF–2–çBg…÷vçFVB„ÕDÅf–Ww÷'BgÂå5T–çFVvW"7rÂå5T–çFVvW"6‚’°¢å5T–çFVvW"rÒ„å5T–çFVvW"—gçv–GF‚Â‚Ò„å5T–çFVvW"—gæ†V–v‡C°¢&WGW&â7rbb6‚bbrãÒ7rbb‚ãÒ6‚bb‡râ7rÇÂ‚â6‚“°§Ð¢òòVæ6öFW2F†RW66ÆRöbF†R&V7FævÆR"öb&r–çFò6"‡6VR&÷fR“²&WGW&ç2F†R66ÆVB–7GW&RÂv†–6‚F†R&W6VçB72G&w2£–çFògÂ÷"æ–Â‡F†R&–Æ–æV"72’à¢òò&VÖ¶U÷v—C¢g‚—2F†R&W6VçBF‡&VBw2Âv†÷6RFW‡GW&W2&W6VçG27F–ÆÂöâF†RuRÖ’&VC¢æWr66ÆW"v—G2f÷"F†VÒæBf÷"e…õ$TÔ´Uõ$U4TåE2&W6VçG26–æ6RF†RÆ7@§7FF–2–CÄÕDÅFW‡GW&Sâg…öVæ6öFR†g…÷7FFR¦g‚Â–CÄÕDÃD6öÖÖæD'VffW#â6"Â–CÄÕDÅFW‡GW&Sâ&rÂ6öç7Bå5T–çFVvW"%³EÒÂÕDÅf–Ww÷'BgÂÕDÅ—†VÄf÷&ÖB÷WE÷bÂ–çB&VÖ¶U÷v—B’°¢å5T–çFVvW"7rÒ%³%ÒÂ6‚Ò%³5ÒÂ÷rÒ„å5T–çFVvW"—gçv–GF‚Âö‚Ò„å5T–çFVvW"—gæ†V–v‡C°¢–b†g…öÖöFR‚’ÒÇÂg…÷vçFVB‡gÂ7rÂ6‚’’&WGW&âæ–Ã°¢–b‚g‚Óç62ÇÂg‚Óæ–âçv–GF‚Ò7rÇÂg‚Óæ–âæ†V–v‡BÒ6‚ÇÂg‚Óæ–âç—†VÄf÷&ÖBÒ&rç—†VÄf÷&ÖBÇÂg‚Óæ÷WBçv–GF‚Ò÷rÇÂg‚Óæ÷WBæ†V–v‡BÒö‚ÇÂg‚Óæ÷WBç—†VÄf÷&ÖBÒ÷WE÷b’°¢å5T–çFVvW"¶W•³eÒÒ¶7rÂ6‚Â&rç—†VÄf÷&ÖBÂ÷rÂö‚Â÷WE÷gÓ°¢–b‚ÖVÖ6×†¶W’Âg‚Óç&VgW6VBÂ6—¦Vöb¶W’’’&WGW&âæ–Ã°¢–b‡&VÖ¶U÷v—Bbbg‚ÓæÖFRbbu÷&W6VçEöâÂg‚ÓæÖFUöB²e…õ$TÔ´Uõ$U4TåE2’&WGW&âæ–Ã°¢–b‡&VÖ¶U÷v—B’²F‡&VEö×WFW…öÆö6²‚fu÷ö×R“²v†–ÆR†u÷ö–æfÆ–v‡B’F‡&VEö6öæE÷v—B‚fu÷ö7bÂfu÷ö×R“²F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“²Ð¢&W5÷&VÖ÷fR†g‚Óæ–â“²&W5÷&VÖ÷fR†g‚Óæ÷WB“²g‚Óç62Òæ–Ã²g‚Óæ–âÒg‚Óæ÷WBÒæ–Ã°¢V–çCcE÷BCÒæ÷uöç2‚“²å4W'&÷"¦RÒæ–Ã°¢–b‚uög…ö6ö×’uög…ö6ö×Ò¶uöFWbæWt6ö×–ÆW%v—F„FW67&—F÷#¥´ÕDÃD6ö×–ÆW$FW67&—F÷"æWuÒW'&÷#¢fUÓ°¢ÕDÄe…7F–Å66ÆW$FW67&—F÷"¦BÒ´ÕDÄe…7F–Å66ÆW$FW67&—F÷"æWuÓ°¢Bæ6öÆ÷%FW‡GW&Tf÷&ÖBÒ&rç—†VÄf÷&ÖC²Bæ÷WGWEFW‡GW&Tf÷&ÖBÒ÷WE÷c²Bæ–çWEv–GF‚Ò7s²Bæ–çWD†V–v‡BÒ6ƒ²Bæ÷WGWEv–GF‚Ò÷s²Bæ÷WGWD†V–v‡BÒöƒ°¢Bæ6öÆ÷%&ö6W76–ætÖöFRÒÕDÄe…7F–Å66ÆW$6öÆ÷%&ö6W76–ætÖöFUW&6WGVÃ°¢–CÄÕDÃDe…7F–Å66ÆW#â62Òuög…ö6ö×ò¶BæWu7F–Å66ÆW%v—F„FWf–6S¦uöFWb6ö×–ÆW#¦uög…ö6ö×Ò¢æ–Ã°¢g‚ÓæÖFUöBÒu÷&W6VçEöã²g‚ÓæÖFR²³°¢–b‚62’²ÖVÖ7’†g‚Óç&VgW6VBÂ¶W’Â6—¦Vöb¶W’“°¢ÔÄôr‚'&W6VçBW66Æ–æs¢ÖWFÄe‚Ö¶W2æò7F–Â66ÆW"f÷"VÇW‚VÇRf÷&ÖBVÇR(i"VÇW‚VÇRf÷&ÖBVÇRW3²F†—26—¦R—266ÆVB&–Æ–æV&Ç’"Â‡Vç6–væVBÆöær–7rÂ‡Vç6–væVBÆöær–6‚Â‡Vç6–væVBÆöær—&rç—†VÄf÷&ÖBÀ¢‡Vç6–væVBÆöær–÷rÂ‡Vç6–væVBÆöær–ö‚Â‡Vç6–væVBÆöær–÷WE÷bÂuög…ö6ö×ò""¢"†æòÖWFÂB6ö×–ÆW"’"“²&WGW&âæ–Ã²Ð¢ÕDÅFW‡GW&TFW67&—F÷"§FBÒ´ÕDÅFW‡GW&TFW67&—F÷"FW‡GW&S$DFW67&—F÷%v—F…—†VÄf÷&ÖC§&rç—†VÄf÷&ÖBv–GFƒ¦7r†V–v‡C¦6‚Ö—ÖVC¤äõÓ°¢FBç7F÷&vTÖöFRÒÕDÅ7F÷&vTÖöFU&—fFS²FBçW6vRÒ62æ6öÆ÷%FW‡GW&UW6vRÂÕDÅFW‡GW&UW6vU6†FW%&VC°¢–CÄÕDÅFW‡GW&Sâ–âÒ¶uöFWbæWuFW‡GW&Uv—F„FW67&—F÷#§FEÓ°¢FBÒ´ÕDÅFW‡GW&TFW67&—F÷"FW‡GW&S$DFW67&—F÷%v—F…—†VÄf÷&ÖC¦÷WE÷bv–GFƒ¦÷r†V–v‡C¦ö‚Ö—ÖVC¤äõÓ°¢FBç7F÷&vTÖöFRÒÕDÅ7F÷&vTÖöFU&—fFS²FBçW6vRÒ62æ÷WGWEFW‡GW&UW6vRÂÕDÅFW‡GW&UW6vU6†FW%&VC°¢–CÄÕDÅFW‡GW&Sâ÷WBÒ¶uöFWbæWuFW‡GW&Uv—F„FW67&—F÷#§FEÓ°¢–b‚–âÇÂ÷WB’²ÖVÖ7’†g‚Óç&VgW6VBÂ¶W’Â6—¦Vöb¶W’“²ÔÄôr‚%µ$TEÒ&W6VçBW66Æ–æs¢6ææ÷B7&VFRF†R66ÆW"w2FW‡GW&W2‚VÇW‚VÇRÂVÇW‚VÇR’"Â‡Vç6–væVBÆöær–7rÂ‡Vç6–væVBÆöær–6‚Â‡Vç6–væVBÆöær–÷rÂ‡Vç6–væVBÆöær–ö‚“²&WGW&âæ–Ã²Ð¢–âæÆ&VÂÒ'&W6VçBW66ÆR–çWB#²÷WBæÆ&VÂÒ'&W6VçBW66ÆR÷WGWB#°¢&W5öFB†–â“²&W5öFB†÷WB“²&W5ö6öÖÖ—B‚“°¢g‚Óç62Ò63²g‚Óæ–âÒ–ã²g‚Óæ÷WBÒ÷WC°¢ÔÄôr‚'&W6VçBW66Æ–æs¢ÖWFÄe‚7F–Â66ÆW"VÇW‚VÇR(i"VÇW‚VÇR†f÷&ÖBVÇR(i"VÇRÂ66ÆW"2VBÂÖFR–âRãb×2’"Â‡Vç6–væVBÆöær–7rÂ‡Vç6–væVBÆöær–6‚Â‡Vç6–væVBÆöær–÷rÂ‡Vç6–væVBÆöær–ö‚À¢‡Vç6–væVBÆöær—&rç—†VÄf÷&ÖBÂ‡Vç6–væVBÆöær–÷WE÷bÂg‚ÓæÖFRÂ†æ÷uöç2‚’ÒC’òSb“°¢Ð¢–CÄÕDÃD6ö×WFT6öÖÖæDVæ6öFW#â6RÒ¶6"6ö×WFT6öÖÖæDVæ6öFW%Ó°¢¶6R&'&–W$gFW%VWVU7FvW3¤ÕDÅ7FvTÆÂ&Vf÷&U7FvW3¤ÕDÅ7FvT&Æ—Bf—6–&–Æ—G”÷F–öç3§&W6÷W&6U÷f—2‚•Ó²òògFW"F†Rg&ÖRw2v÷&°¢¶6R6÷”g&öÕFW‡GW&S§&r6÷W&6U6Æ–6S£6÷W&6TÆWfVÃ£6÷W&6T÷&–v–ã¤ÕDÄ÷&–v–äÖ¶R‡%³ÒÂ%³ÒÂ’6÷W&6U6—¦S¤ÕDÅ6—¦TÖ¶R†7rÂ6‚Â’FõFW‡GW&S¦g‚Óæ–âFW7F–æF–öå6Æ–6S£FW7F–æF–öäÆWfVÃ£FW7F–æF–öä÷&–v–ã¤ÕDÄ÷&–v–äÖ¶RƒÂÂ•Ó°¢¶6R&'&–W$gFW%7FvW3¤ÕDÅ7FvT&Æ—B&Vf÷&UVWVU7FvW3¤ÕDÅ7FvTÆÂf—6–&–Æ—G”÷F–öç3§&W6÷W&6U÷f—2‚•Ó²òòF†R66ÆW"w2Væ6öFW'26öÖRgFW"F†R6÷¢¶6RVæDVæ6öF–æuÓ°¢g‚Óç62æ6öÆ÷%FW‡GW&RÒg‚Óæ–ã²g‚Óç62æ–çWD6öçFVçEv–GF‚Ò7s²g‚Óç62æ–çWD6öçFVçD†V–v‡BÒ6ƒ²g‚Óç62æ÷WGWEFW‡GW&RÒg‚Óæ÷WC°¢¶g‚Óç62Væ6öFUFô6öÖÖæD'VffW#¦6%Ó°¢&WGW&âg‚Óæ÷WC°§Ð¢òòF†R&W6VçB73¢7&2w2fÆ–B&V7FævÆRG&vâ–çFòG7B‡F†RG&v&ÆRÂ÷"67&VVç6†÷Bw2FW‡GW&R’2&W6VçEöVæ6öFRG&w2—BÂF‡&÷Vv‚ÖWFÄe‚f—'7Bv†W&Rg…öVæ6öFRF¶W2—@¢òò†gƒ¢F†R66ÆW"7FFRFòW6RÂæ–ÂÒ&–Æ–æV"’âWf#¢F†R72w27&÷6öç7FçG2†öæR'VffW"W"&W6VçBÆÆö6F÷"Âw&—GFVâ'’F†R5R§7FF–2–çB&W6VçEöVæ6öFUöÓB†–CÄÕDÃD6öÖÖæD'VffW#â6"Â–CÄÕDÅFW‡GW&Sâ7&2Â–CÄÕDÅFW‡GW&SâG7BÂ6öç7B–çB7&÷³EÒÂ–CÄÕDÄ'VffW#âWf"Âg…÷7FFR¦g‚’°¢å5T–çFVvW"%³EÓ²–b‚&W6VçEö7&÷÷&V7B‡7&2Â7&÷Â"’’&WGW&â°¢&W6VçE÷—VÆ–æR†G7Bç—†VÄf÷&ÖB“°¢F÷V&ÆRGrÒG7Bçv–GF‚ÂF‚ÒG7Bæ†V–v‡C°¢ÕDÅf–Ww÷'BgÒ&W6VçE÷f–Ww÷'B‡%³%ÒÂ%³5ÒÂGrÂF‚“°¢–çBgVÆÂÒgæ÷&–v–å‚ÓÒbbgæ÷&–v–å’ÓÒbbgçv–GF‚ÓÒGrbbgæ†V–v‡BÓÒFƒ°¢–CÄÕDÅFW‡GW&SâFW‚Ò&W6VçE÷&u÷7&2‡7&2’ÂWÒg‚òg…öVæ6öFR†g‚Â6"ÂFW‚Â"ÂgÂG7Bç—†VÄf÷&ÖBÂg‚ÓÒfuög…÷&W6VçB’¢æ–Ã°¢–b‡W’²FW‚ÒW²%³ÒÒ%³ÒÒ²%³%ÒÒWçv–GFƒ²%³5ÒÒWæ†V–v‡C²ÒòòF†R66ÆVB–7GW&RÂF†Rf–Ww÷'Bw26—¦S¢G&vâ£¢ÕDÃE&VæFW%74FW67&—F÷"§'Ò´ÕDÃE&VæFW%74FW67&—F÷"æWuÓ°¢'æ6öÆ÷$GF6†ÖVçG5³ÒçFW‡GW&RÒG7C°¢'æ6öÆ÷$GF6†ÖVçG5³ÒæÆöD7F–öâÒgVÆÂòÕDÄÆöD7F–öäFöçD6&R¢ÕDÄÆöD7F–öä6ÆV#°¢'æ6öÆ÷$GF6†ÖVçG5³Òç7F÷&T7F–öâÒÕDÅ7F÷&T7F–öå7F÷&S°¢'æ6öÆ÷$GF6†ÖVçG5³Òæ6ÆV$6öÆ÷"ÒÕDÄ6ÆV$6öÆ÷$Ö¶RƒÂÂÂ“°¢–CÄÕDÃE&VæFW$6öÖÖæDVæ6öFW#âVâÒ¶6"&VæFW$6öÖÖæDVæ6öFW%v—F„FW67&—F÷#§'Ó°¢–b‚Vâ’²ÔÄôr‚%µ$TEÒf–ÆVBFò7&VFRF†RÓB&W6VçB&VæFW"Væ6öFW""“²&WGW&â²Ð¢¶Vâ&'&–W$gFW%VWVU7FvW3¤ÕDÅ7FvTÆÂ&Vf÷&U7FvW3¤ÕDÅ7FvTÆÂf—6–&–Æ—G”÷F–öç3§&W6÷W&6U÷f—2‚•Ó°¢¶Vâ6WD&wVÖVçEF&ÆS¦u÷&W6VçEö&w2E7FvW3¤ÕDÅ&VæFW%7FvTg&vÖVçEÓ°¢¶u÷&W6VçEö&w26WEFW‡GW&S§FW‚æwU&W6÷W&6T”BD–æFWƒ£Ó°¢¶u÷&W6VçEö&w26WE6×ÆW%7FFS¦uö&Æ—E÷6×æwU&W6÷W&6T”BD–æFWƒ£Ó°¢fÆöB§WbÒWf"æ6öçFVçG3²å5T–çFVvW"GrÒFW‚çv–GF‚ÂF‚ÒFW‚æ†V–v‡BÂ7‚Ò%³ÒÂ7’Ò%³ÒÂ7rÒ%³%ÒÂ6‚Ò%³5Ó°¢We³ÒÒ†fÆöB–7‚òGs²We³ÒÒ†fÆöB–7’òFƒ²We³%ÒÒ†fÆöB–7ròGs²We³5ÒÒ†fÆöB–6‚òFƒ°¢We³EÒÒ†7‚²ãVb’òGs²We³UÒÒ†7’²ãVb’òFƒ²We³eÒÒ†7‚²7rÒãVb’òGs²We³uÒÒ†7’²6‚ÒãVb’òFƒ°¢–b†FöÖ–5öÆöEöW‡Æ–6—B‚fuö÷&–v–åöÆÂÂÖVÖ÷'•ö÷&FW%÷&VÆ†VB’’²We³Ò³ÒWe³5Ó²We³5ÒÒ×We³5Ó²ÒòòÆ÷vW"ÖÆVgB÷&–v–ã¢F†Rv–æF÷r–ÖvR—27F÷&VB&÷GFöÒ×W ¢¶u÷&W6VçEö&w26WDFG&W73§Wf"æwTFG&W72D–æFWƒ£Ó°¢¶Vâ6WEf–Ww÷'C§gÓ²¶Vâ6WE&VæFW%—VÆ–æU7FFS¦uö&Æ—E÷6õÓ°¢¶VâG&u&–Ö—F—fW3¤ÕDÅ&–Ö—F—fUG—UG&–ævÆRfW'FW…7F'C£fW'FW„6÷VçC£5Ó°¢ÓEöVæB†Vâ“°¢&WGW&âWò$U4TåEôÔUDÄe‚¢$U4TåEõ5DäD$C°§Ð¢òòF†R&W6VçBw2–7GW&RG&vâ–çFòâöfg67&VVâFW‡GW&R†÷WC¢&—fFRÂ&VæFW"F&vWB’F†Rv’&W6VçEöæ÷rG&w2—B–çFòF†RG&v&ÆRÂÖWFÄe‚–æ6ÇVFVBÂöâF†RÖWFÂBVWVS°¢òò&WGW&ç2v†WF†W"—Bv27V&Ö—GFVBâ6ö×ÆWF–öâ&W÷'G2F†RVæ6öF–ær¶–æBÂ÷"$U4TåEôd”ÄTBöâuRW'&÷"â&W6÷W&6W2&VÖ–â&W6–FVçBæB&WF–æVBVçF–ÂuR6ö×ÆWF–öâà§7FF–2–çB&W6VçEööfg67&VVåöÓB†–CÄÕDÅFW‡GW&Sâ7&2Â6öç7B–çB7&÷³EÒÂ–CÄÕDÅFW‡GW&Sâ÷WBÂfö–B…æFöæR’†–çB’’°¢–CÄÕDÃD6öÖÖæDÆÆö6F÷#âÂÒ¶uöFWbæWt6öÖÖæDÆÆö6F÷%Ó²–CÄÕDÃD6öÖÖæD'VffW#â6"Ò¶uöFWbæWt6öÖÖæD'VffW%Ó°¢–CÄÕDÄ'VffW#âWf"Ò¶uöFWbæWt'VffW%v—F„ÆVæwFƒ§6—¦Vöb†fÆöB’¢‚÷F–öç3¤ÕDÅ&W6÷W&6U7F÷&vTÖöFU6†&VEÓ°¢–b‚ÂÇÂ6"ÇÂWf"’²ÔÄôr‚%µ$TEÒöfg67&VVâ&W6VçC¢æò6öÖÖæBÆÆö6F÷"÷"'VffW""“²&WGW&â²Ð¢–CÄÕDÅFW‡GW&Sâ&rÒ&W6VçE÷&u÷7&2‡7&2“°¢g…÷7FFRg‚Ò³Ó°¢&W5öFB†÷WB“²&W5öFB‡Wf"“²&W5ö6öÖÖ—B‚“°¢¶6"&Vv–ä6öÖÖæD'VffW%v—F„ÆÆö6F÷#¦ÅÓ°¢–çB¶–æBÒ&W6VçEöVæ6öFUöÓB†6"Â&rÂ÷WBÂ7&÷ÂWf"Âfg‚“°¢¶6"VæD6öÖÖæD'VffW%Ó°¢–b‚¶–æB’²&W5÷&VÖ÷fR†÷WB“²&W5÷&VÖ÷fR‡Wf"“²&W5÷&VÖ÷fR†g‚æ–â“²&W5÷&VÖ÷fR†g‚æ÷WB“²&WGW&â²Ð¢ÕDÃD6öÖÖ—D÷F–öç2¦÷F–öç2Ò´ÕDÃD6öÖÖ—D÷F–öç2æWuÓ°¢¶÷F–öç2FDfVVF&6´†æFÆW#¥â†–CÄÕDÃD6öÖÖ—DfVVF&6³âb’°¢‡fö–B–Ã²‡fö–B–6#²‡fö–B—&s²‡fö–B–g‚ç63²òò¶VWF†RÖWFÂB6öÖÖæG2Â6÷W&6Rf–WræB66ÆW"Æ—fRFöð¢&W5÷&VÖ÷fR†÷WB“²&W5÷&VÖ÷fR‡Wf"“²&W5÷&VÖ÷fR†g‚æ–â“²&W5÷&VÖ÷fR†g‚æ÷WB“°¢–b†bæW'&÷"’ÔÄôr‚%µ$TEÒöfg67&VVâ&W6VçC¢W2"ÂbæW'&÷"æFW67&—F–öâåUDc…7G&–ær“°¢FöæR†bæW'&÷"ò$U4TåEôd”ÄTB¢¶–æB“°¢ÕÓ°¢¶u÷B6öÖÖ—C¢f6"6÷VçC£÷F–öç3¦÷F–öç5Ó°¢&WGW&â°§Ð¢òòF–væ÷7F–2&VF&6²æBFW7G2v—Bf÷"F†R6ÖR7–æ6‡&öæ÷W26GW&RW6VB'’v†öÆR×v–æF÷r67&VVç6†÷G2âF–ÖV÷WBÆVfW26ÆVçWv—F‚F†RuR6ö×ÆWF–öâ†æFÆW"à§7FF–2–çB&W6VçEööfg67&VVå÷v—B†–CÄÕDÅFW‡GW&Sâ7&2Â6öç7B–çB7&÷³EÒÂ–CÄÕDÅFW‡GW&Sâ÷WB’°¢F—7F6…÷6VÖ†÷&U÷B6VÒÒF—7F6…÷6VÖ†÷&Uö7&VFRƒ“²õö&Æö6²–çB¶–æBÒ$U4TåEôd”ÄTC°¢–b‚&W6VçEööfg67&VVåöÓB‡7&2Â7&÷Â÷WBÂâ†–çB²’²¶–æBÒ³²F—7F6…÷6VÖ†÷&U÷6–væÂ‡6VÒ“²Ò’’&WGW&â°¢–b†F—7F6…÷6VÖ†÷&U÷v—B‡6VÒÂF—7F6…÷F–ÖR„D•5D4…õD”ÔUôäõrÂ¢å4T5õU%õ4T2’’’²ÔÄôr‚%µ$TEÒöfg67&VVâ&W6VçC¢F†RuRF–Bæ÷Bf–æ—6‚v—F†–â2"“²&WGW&â²Ð¢&WGW&â¶–æC°§Ð¢òòF†R&W6VçB6öÖÖæB'VffW"w2uRF–ÖR†—G26öÖÖ—BfVVF&6³¢g&öÒ—G2f—'7BFò—G2Æ7B6öÖÖæBÂF†RÖWFÄe‚66ÆW"æBF†R&W6VçB72’ÂWfW'’c&W6VçG22Sò“RÂ'@¢òòf÷"&W6VçG2F‡&÷Vv‚ÖWFÄe‚æBF‡&÷Vv‚F†R&–Æ–æV"72âVæFW"u÷ö×R‡F†RfVVF&6²†æFÆW"§7FF–2fö–B&W6VçEöwUöæ÷FR†F÷V&ÆR2Â–çBg‚’°¢7FF–2F÷V&ÆRe³%Õ³cÓ²7FF–2–çBå³%Ó°¢–b‚‡2âbb2Â’’&WGW&ã°¢e¶g…Õ¶å¶g…Ò²µÒÒ2¢S3°¢–b†å¶g…ÒÂc’&WGW&ã°¢F÷V&ÆR¦Òe¶g…Ó²f÷"†–çB’Ò²’Âc²’²²’²F÷V&ÆR‚Ò¶•Ó²–çB¢Ò’Ò²v†–ÆR†¢ãÒbb¶¥Òâ‚’²¶¢²ÒÒ¶¥Ó²¢ÒÓ²Ò¶¢²ÒÒƒ²Ð¢ÔÄôr‚'&W6VçB6öÖÖæB'VffW"uRF–ÖR÷fW"c&W6VçG2W3¢SRã6b×2Â“RRã6b×2ÂÖ‚Rã6b×2"Âg‚ò'F‡&÷Vv‚ÖWFÄe‚"¢'F‡&÷Vv‚F†R&–Æ–æV"72"Â³#“•ÒÂ³Sc•ÒÂ³S“•Ò“°¢å¶g…ÒÒ°§Ð§7FF–2F÷V&ÆRuöÆ–æµö—E÷F–ã²òòVçVWVRF–ÖRöbF†Rg&ÖRF†RF—7Æ’ÖÆ–æ²6ÆÆ&6²—2&W6VçF–ær‡&VBæBw&—GFVâöæÇ’öâF†RF—7Æ’ÖÆ–æ²F‡&VB§7FF–2–çB&W6VçEöæ÷r†6öç7BFW…÷B§BÂF÷V&ÆRW&–öBÂV–çCcE÷Bg&ÖRÂV–çCcE÷BF–6¶WBÂ6öç7B–çB7&÷³EÒÂ–CÄ4ÖWFÄG&v&ÆSâ7WÆ–VBÂF÷V&ÆRF&vWE÷G2’°¢–b†u÷&W6VçE÷7FÆÅöç2’²7G'V7BF–ÖW7V2G2Ò³Â†Æöær–u÷&W6VçE÷7FÆÅöç7Ó²ææ÷6ÆVW‚gG2ÂåTÄÂ“²Ð¢u÷&W6VçEöâ²³°¢–b†&ö÷G&öb‚’bb†u÷&W6VçEöâÓÒÇÂu÷&W6VçEöâR#ÓÒ’’°¢–b†u÷&W6VçEöâÓÒ’uö'öf—'7E÷&W6VçBÒ'öæ÷r‚’Òuö'÷C°¢ÔÄôr‚%¶&ö÷G&öeÒCÒRã6b2&W6VçB2VÆÇS¢ÖWFÆÆ–'2ÆöFVBVÆÇRòRãb×3²—VÆ–æW27&VFVBVÆÇRòRãb×2†ÆÂ7–æ6‡&öæ÷W2öâF†R7V&Ö—BF‡&VB“²&Wv&ÖVBæBW6VC¢6†FW'2VÆÇRò&VæFW"—VÆ–æW2VÆÇRò6ö×WFR—VÆ–æW2VÆÇS² ¢'FW‡GW&RWÆöG2RãbÔ"òVÆÇRf—'7BWÆöG2²VÆÇR&R×WÆöG2òRãb×3²f—'7B&W6VçBCÒRã6b2"Â†'öæ÷r‚’Òuö'÷C’òS’Âu÷&W6VçEöâÀ¢uö'öÆ–%öâÂuö'öÆ–%öç2òSbÂuö'÷6õöâÂuö'÷6õöç2òSbÂu÷v&Õö6Æ–Õ÷6†FW'2Âu÷v&Õö6Æ–Õ÷'2Âu÷v&Õö6Æ–Õö72À¢uö'÷Wö'—FW2òCƒSsbãÂu÷¶u÷WÆöG2Âu÷¶u÷&WWÆöG2À¢u÷W÷F÷FÅöç2òSbÂuö'öf—'7E÷&W6VçBòS’“°¢Ð¢B‚%‚6W#ÒVÆÇR"ÂBÓç6W&–Â“°¢–CÄÕDÅFW‡GW&Sâ7&2ÒFW…ö×FÂ‡B“°¢–b‚7&2’²$TEôôä4Rƒ‚Â'&W6VçBFW‡GW&R6÷VÆBæ÷BvWBâÕDÅFW‡GW&R"“²&WGW&â²Ð¢òòFW‡GW&RÖFRöâf—'7B&W6öÇWF–öâ¦ö–ç2F†R&W6–FVæ7’6WB‡&W5öFB’Âv†–6‚7V&Ö—76–öâ6öÖÖ—G3²v–æF÷rFW‡GW&R&W6VçFVB&Vf÷&Rç’7V&Ö—76–öâ†2W6VB—Bv27F–ÆÀ¢òò÷WG6–FRF†R6öÖÖ—GFVB6WBv†VâF†RF—7Æ’6öÖÖæB'VffW"6×ÆVB—Bâ7WW"6Ö6‚'&÷2âVÇF–ÖFR&W6VçG2—G2v–æF÷rFW‡GW&R–â—G2f—'7Bg&ÖW2&Vf÷&RG&v–ær–çFò—C ¢òòF†RF—7Æ’VWVRvRÖfVÇFVB†´”ôuT6öÖÖæD'VffW$6ÆÆ&6´W'&÷%vTfVÇB’æBÖ4õ2F†Vâ–væ÷&VBWfW'’ÆFW"6öÖÖæB'VffW"öbF†R&ö6W72Â6òF†Rv–æF÷r7F–VB–æ°¢òò††VFÆW72'Vç2æWfW"&W6VçBÂæBWfW'’vÖR&Vf÷&RG&Wr–çFòv–æF÷rFW‡GW&R&Vf÷&R&W6VçF–ær—B’â6öÖÖ—Bv—F‚æ÷F†–æræWr6÷7G2æ÷F†–æp¢&W5ö6öÖÖ—B‚“°¢òòDÕôådåõ$TtÅTÓ×ƒÇ“ÇƒÇ“¦f–ÆR†æ÷&ÖÆ—¦VB6ö÷&F–æFW2“¢&VB&6²F†—2&Vv–öâöâWfW'’&W6VçBæB&V6÷&B'&W6VçB6WVVæ6RçVÖ&W"Â&W6VçBF–ÖRç2Â&Vv–öâÖVâÇVÖ–ææ6RÂ7V'6×ÆVBw&—66ÆRF‡VÖ&æ–Â††W‚’"†F–væ÷7F–3¢fW&–g’F†Bæ–ÖF–öâ7VVB—2–æFWVæFVçBöbg&ÖR&FR’à¢òò6W&FR6öÖÖæB'VffW"—2VWVVBgFW"F†—2g&ÖRw2&VæFW&–ær‡F†R6ÖRVWVRW†V7WFW2–â7V&Ö—B÷&FW"“²FöW2æ÷Bv—Bf÷"F†RuRæBFöW2æ÷BF—7GW&"g&ÖR6–æp¢7FF–2–çB&ÂÒÓ²7FF–2fÆöB'%³EÓ²7FF–2d”ÄR§&Æc²7FF–2F‡&VEö×WFW…÷B&Æ×RÒD…$TEôÕUDU…ô”ä•D”Ä•¤U#°¢–b‡&ÂÂ’²6öç7B6†"¦RÒvWFVçb‚$DÕôådåõ$TtÅTÒ"’Â¦2ÒRò7G&6‡"†RÂs¢r’¢åTÄÃ°¢&ÂÒ2bb766æb†RÂ"VbÂVbÂVbÂVb"Âg'%³ÒÂg'%³ÒÂg'%³%ÒÂg'%³5Ò’ÓÒBbb‡&ÆbÒf÷Vâ†2²Â'r"’’ÒåTÄÃ°¢–b†Rbb&Â’ÔÄôr‚%µ$TEÒDÕôådåõ$TtÅTÒ&wVÖVçB÷"f–ÆR–çfÆ–C¢W2"ÂR“²Ð¢–b‡&Âbb‡7&2ç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖE$t$…Væ÷&ÒÇÂ7&2ç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖD$u$…Væ÷&ÒÇÂ7&2ç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖE$t$…Væ÷&Õ÷5$t"ÇÂ7&2ç—†VÄf÷&ÖBÓÒÕDÅ—†VÄf÷&ÖD$u$…Væ÷&Õ÷5$t"’’°¢å5T–çFVvW"ƒÒ„å5T–çFVvW"’‡'%³Ò¢7&2çv–GF‚’Â“Ò„å5T–çFVvW"’‡'%³Ò¢7&2æ†V–v‡B’ÂrÒ„å5T–çFVvW"’‡'%³%Ò¢7&2çv–GF‚’ÒƒÂ‚Ò„å5T–çFVvW"’‡'%³5Ò¢7&2æ†V–v‡B’Ò“°¢–CÄÕDÄ'VffW#â&"Ò¶uöFWbæWt'VffW%v—F„ÆVæwFƒ§r¢‚¢B÷F–öç3¤ÕDÅ&W6÷W&6U7F÷&vTÖöFU6†&VEÓ°¢–CÄÕDÄ6öÖÖæD'VffW#â6"Òö6"‚“²–CÄÕDÄ&Æ—D6öÖÖæDVæ6öFW#â"Ò¶6"&Æ—D6öÖÖæDVæ6öFW%Ó°¢¶"6÷”g&öÕFW‡GW&S§7&26÷W&6U6Æ–6S£6÷W&6TÆWfVÃ£6÷W&6T÷&–v–ã¤ÕDÄ÷&–v–äÖ¶R‡ƒÂ“Â’6÷W&6U6—¦S¤ÕDÅ6—¦TÖ¶R‡rÂ‚Â’Fô'VffW#§&"FW7F–æF–öäöfg6WC£FW7F–æF–öä'—FW5W%&÷s§r¢BFW7F–æF–öä'—FW5W$–ÖvS§r¢‚¢EÓ°¢¶"VæDVæ6öF–æuÓ°¢Vç6–væVBÆöærÆöærâÒu÷&W6VçEöâÂFç2Ò6Æö6µövWGF–ÖUöç6V5öç„4Äô4µõUD”ÔUõ$r“°¢¶6"FD6ö×ÆWFVD†æFÆW#¥â†–CÄÕDÄ6öÖÖæD'VffW#â2’²‡fö–B–3²6öç7BV–çC…÷B§Ò&"æ6öçFVçG3²V–çCcE÷B7VÒÒ°¢V–çC…÷Bu³C“eÓ²å5T–çFVvW"ærÒ²òò7V'6×ÆVBw&—66ÆRF‡VÖ&æ–Â†WfW'’GF‚&÷rò6öÇVÖâÂBÖ÷7BC“bö–çG2’Â&V6÷&FVBÆöærv—F‚F†RÖVâÂf÷"W"Ög&ÖR–ÖvR6÷'&VÆF–öâæÇ—6—0¢f÷"„å5T–çFVvW"’Ò²’Âƒ²’³ÒB’f÷"„å5T–çFVvW"‚Ò²‚ÂrbbærÂ6—¦Vöbs²‚³ÒB’²6öç7BV–çC…÷B§‚Ò²B¢‡’¢r²‚“²u¶ær²µÒÒ‡V–çC…÷B’‚‡…³Ò²…³Ò²…³%Ò’ò2“²Ð¢f÷"„å5T–çFVvW"²Ò²²Âr¢ƒ²²²²’7VÒ³Ò³B¢µÒ²³B¢²²Ò²³B¢²²%Ó°¢F‡&VEö×WFW…öÆö6²‚g&Æ×R“²g&–çFb‡&ÆbÂ"VÆÇUÇBVÆÇUÇBRãFeÇB"ÂâÂFç2Â7VÒòƒ2ã¢r¢‚’“°¢f÷"„å5T–çFVvW"²Ò²²Âæs²²²²’g&–çFb‡&ÆbÂ"S'‚"Âu¶µÒ“°¢gWF2‚uÆârÂ&Æb“²ffÇW6‚‡&Æb“²F‡&VEö×WFW…÷VæÆö6²‚g&Æ×R“²ÕÓ°¢¶6"6öÖÖ—EÓ°¢Ð¢òòDÕôådåõ4„õCÖg&ÖRçVÖ&W%²Æg&ÖRçVÖ&W"ââåÓ§F‚‚VB–âF†RF‚—2&WÆ6VBv—F‚F†Rg&ÖRçVÖ&W"¢7FF–2Æöær6†÷G5³C“eÓ²7FF–2–çBç6†÷BÒÓ²7FF–26öç7B6†"§6†÷E÷Fƒ²òòWFòC“bg&ÖW2†'W'7B÷fW"g&ÖR&ævRÂf÷"66÷&–ærg&ÖR'’g&ÖR¢–b†ç6†÷BÂ’°¢ç6†÷BÒ²6öç7B6†"¦RÒvWFVçb‚$DÕôådåõ4„õB"’Â¦2ÒRò7G&6‡"†RÂs¢r’¢åTÄÃ°¢–b†2’²6†÷E÷F‚Ò2²²f÷"†6öç7B6†"§ÒS²Â2bbç6†÷BÂC“c²Ò7G&6‡"‡ÂrÂr’ò7G&6‡"‡ÂrÂr’²¢2’6†÷G5¶ç6†÷B²µÒÒFöÂ‡“²Ð¢Ð¢f÷"†–çB’Ò²’Âç6†÷C²’²²’–b‚†Æöær–u÷&W6VçEöâÓÒ6†÷G5¶•Ò’²6†"F…³#EÓ²6ç&–çFb‡F‚Â6—¦VöbF‚Â6†÷E÷F‚Â†–çB–u÷&W6VçEöâ“²×FÅöf–æ—6…öwR‚“²6†÷B‡7&2ÂF‚ÂÂ7&÷“²Ð¢F‡&VEö×WFW…öÆö6²‚fu÷6†÷Gö×R“°¢f÷"†–çB’Ò²’Âuöç6†÷G²’–b†u÷&W6VçEöâãÒu÷6†÷G¶•Òæg&ÖR’²×FÅöf–æ—6…öwR‚“²6†÷B‡7&2Âu÷6†÷G¶•ÒçF‚ÂÂ7&÷“²u÷6†÷G¶•ÒÒu÷6†÷G²ÒÖuöç6†÷GÓ²ÒVÇ6R’²³°¢F‡&VEö×WFW…÷VæÆö6²‚fu÷6†÷Gö×R“° ¢4ÖWFÄÆ–W"¦Æ–W"Ò…õö'&–FvR4ÖWFÄÆ–W"¢–g&öçEöÖWFÅöÆ–W"‚“°¢–b‚Æ–W"’&WGW&â°¢–b†u÷&W6VçEöÆ–W%÷&W6–FVçBÒÆ–W"’²¶u÷BFE&W6–FVæ7•6WC¦Æ–W"ç&W6–FVæ7•6WEÓ²u÷&W6VçEöÆ–W%÷&W6–FVçBÒÆ–W#²Ð¢6–bD$tUEôõ5ôõ5€¢–b‚Æ–W"æF—7Æ•7–æ4Væ&ÆVB’Æ–W"æF—7Æ•7–æ4Væ&ÆVBÒ”U3²òòv–æF÷vVBÖöFRföÆÆ÷w2F†RF—7Æ’w2g7–æ2Fò&WfVçBFV&–æp¢6VæF–`¢&W6VçE÷—VÆ–æR†Æ–W"ç—†VÄf÷&ÖB“²òò6ÖR2F†R÷&–v–æÂFƒ¢F†Rf—'7Bg&ÖR6ö×–ÆW2F†R—VÆ–æR&Vf÷&RF¶–ærF†RG&v&ÆP¢–b†uöw&%ö6"’w&%÷&W6VçB‡7&2Â7&÷ÂÆ–W"æG&v&ÆU6—¦R“²òòvÖR–7GW&RæVVFVB'’F†Rv†öÆR×v–æF÷r67&VVç6†÷B‡VV²v—F†÷WBF†RÆö6²f—'7C²F†R&VÂF¶RÆö6·2–ç6–FRF†RgVæ7F–öâ¢V–çCcE÷BGCÒæ÷uöç2‚“°¢–CÄ4ÖWFÄG&v&ÆSâG"Ò7WÆ–VBò7WÆ–VB¢¶Æ–W"æW‡DG&v&ÆUÓ°¢–b‚7WÆ–VB’uöG&v&ÆUöç2³Òæ÷uöç2‚’ÒGC°¢–b‚G"’&WGW&â°¢–çBÆÆö5ö’Ò†–çB’‡F–6¶WBR2“°¢–b†u÷&W6VçEöÆÆö5öfVæ6U¶ÆÆö5ö•Ò’°¢F‡&VEö×WFW…öÆö6²‚fu÷ö×R“°¢v†–ÆR†u÷&W6VçEöÆÆö5öFöæU¶ÆÆö5ö•ÒÂu÷&W6VçEöÆÆö5öfVæ6U¶ÆÆö5ö•Ò’F‡&VEö6öæE÷v—B‚fu÷ö7bÂfu÷ö×R“°¢F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“°¢Ð¢¶u÷&W6VçEöÆÆö3E¶ÆÆö5ö•Ò&W6WEÓ°¢¶u÷Bv—Df÷$G&v&ÆS¦G%Ó°¢–CÄÕDÃD6öÖÖæD'VffW#â6"Òu÷&W6VçEö6%¶ÆÆö5ö•Ó°¢¶6"&Vv–ä6öÖÖæD'VffW%v—F„ÆÆö6F÷#¦u÷&W6VçEöÆÆö3E¶ÆÆö5ö•ÕÓ°¢–çB¶–æBÒ&W6VçEöVæ6öFUöÓB†6"Â7&2ÂG"çFW‡GW&RÂ7&÷Âu÷&W6VçE÷We¶ÆÆö5ö•ÒÂfuög…÷&W6VçB“°¢–b‚¶–æB’²¶6"VæD6öÖÖæD'VffW%Ó²¶u÷B6–væÄG&v&ÆS¦G%Ó²&WGW&â²Ð¢–çBg…ö¶–æBÒ¶–æBÓÒ$U4TåEôÔUDÄeƒ°¢¶6"VæD6öÖÖæD'VffW%Ó°¢ÕDÃD6öÖÖ—D÷F–öç2¦òÒ´ÕDÃD6öÖÖ—D÷F–öç2æWuÓ°¢¶òFDfVVF&6´†æFÆW#¥â†–CÄÕDÃD6öÖÖ—DfVVF&6³âb’°¢–b†bæW'&÷"’ÔÄôr‚%µ$TEÒF—7Æ’6öÖÖæB'VffW"W'&÷#¢W2"ÂbæW'&÷"æFW67&—F–öâåUDc…7G&–ær“°¢F‡&VEö×WFW…öÆö6²‚fu÷ö×R“²–b‡F–6¶WBâu÷öFöæR’u÷öFöæRÒF–6¶WC²–b‡F–6¶WBâu÷&W6VçEöÆÆö5öFöæU¶ÆÆö5ö•Ò’u÷&W6VçEöÆÆö5öFöæU¶ÆÆö5ö•ÒÒF–6¶WC²u÷ö–æfÆ–v‡BÒÓ²F‡&VEö6öæEö'&öF67B‚fu÷ö7b“°¢&W6VçEöwUöæ÷FR†bäuTVæEF–ÖRÒbäuU7F'EF–ÖRÂg…ö¶–æB“²F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“°¢ÕÓ°¢–CÄÕDÃD6öÖÖæD'VffW#â'VffW'5µÒÒ¶6'Ó°¢F‡&VEö×WFW…öÆö6²‚fu÷ö×R“²u÷ö–æfÆ–v‡B²³²F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“°¢¶u÷B6öÖÖ—C¦'VffW'26÷VçC£÷F–öç3¦õÓ°¢–b‡F&vWE÷G2â’°¢7FF–2F÷V&ÆRÆVG5³3ÒÂS&U³3Ó²òòS&S¢vÖRVçVWVRÓâ7—7FVÒ×66†VGVÆVB&W6VçB†–æ6ÇVF–ærF†RF–ÖRv—F–ær–âF†RVWVRf÷"F†RF—7Æ’ÖÆ–æ²6ÆÆ&6²’ÂF†RgVÆÂ7âöb–çWBÆFVæ7’F†B÷W"6–FR6â&V@¢7FF–2V–çCcE÷BÆVEöã°¢ÆVG5¶ÆVEöâR3ÒÒ‡F&vWE÷G2Ò47W'&VçDÖVF–F–ÖR‚’’¢ã²S&U¶ÆVEöâR3ÒÒuöÆ–æµö—E÷F–ââò‡F&vWE÷G2ÒuöÆ–æµö—E÷F–â’¢ã¢°¢ÆVEöâ²³°¢–b†ÆVEöâR3ÓÒ’°¢F÷V&ÆR³3Ó²ÖVÖ7’†ÂÆVG2Â6—¦Vöb“°¢f÷"†–çB’Ò²’Â3²’²²’²F÷V&ÆRbÒ¶•Ó²–çB¢Ò’Ò²v†–ÆR†¢ãÒbb¶¥Òâb’²¶¢²ÒÒ¶¥Ó²¢ÒÓ²Ò¶¢²ÒÒc²Ð¢F÷V&ÆR%³3Ó²ÖVÖ7’†"ÂS&RÂ6—¦Vöb"“°¢f÷"†–çB’Ò²’Â3²’²²’²F÷V&ÆRbÒ%¶•Ó²–çB¢Ò’Ò²v†–ÆR†¢ãÒbb%¶¥Òâb’²%¶¢²ÒÒ%¶¥Ó²¢ÒÓ²Ò%¶¢²ÒÒc²Ð¢ÔÄôr‚&F—7Æ’Æ–æ²WFòF–6²VÆÇS¢F–6·23Â&W6VçG2VÆÇRÂV×G’F–6·2VÆÇR††VÆBf÷"F†R&W6VçB–çFW'fÂVÆÇR’ÂöÆBg&ÖW2G&÷VB'’ÖW&v–ærVÆÇS²7V&Ö—76–öâ†VBöbF&vWBF—7Æ’SRã6b×2òÖ–âRã6b×3²vÖRVçVWVRFòF&vWBF—7Æ’SRã6b×2òÖ–âRã6bòÖ‚Rã6b×2"À¢‡Vç6–væVBÆöærÆöær–uöÆ–æµ÷F–6µöâÂ‡Vç6–væVBÆöærÆöær–u÷&W6VçEöâÂ‡Vç6–væVBÆöærÆöær–uöÆ–æµöV×G’Â‡Vç6–væVBÆöærÆöær–uöÆ–æµö†VÆBÀ¢‡Vç6–væVBÆöærÆöær–uöÆ–æµöG&÷VBÂ³C•ÒÂ³ÒÂ%³C•ÒÂ%³ÒÂ%³#“•Ò“°¢Ð¢Ð¢¶u÷B6–væÄG&v&ÆS¦G%Ó°¢u÷&W6VçEöÆÆö5öfVæ6U¶ÆÆö5ö•ÒÒF–6¶WC°¢6–bD$tUEôõ5õ4”ÕTÄDõ"òòF†R”õ26–×VÆF÷"4D²†2æò&W6VçFF–öâ×F–ÖR6ÆÆ&6²æBæòÖ–æ–×VÒF—7Æ’GW&F–öâ‡W6VBöæÇ’f÷"–7GW&R66WFæ6RÂF—7Æ’ÖÆ–æ²F–6·2&Ræ÷B6÷VçFVB¢‡fö–B—W&–öC²¶G"&W6VçEÓ°¢6VÇ6P¢V–çCcE÷BâÒF–6¶WC²òò&V6÷&BF†R&W6VçBF–ÖR'’F†RvÖRw2&W6VçB6WVVæ6RçVÖ&W"‡F–6¶WBçVÖ&W"“¢†ÆRw2&W6VçB7FF—7F–72&RF¶Vâ'’'F†RvÖRw2çF‚&W6VçB#²–âF†RF—7Æ’Æ–æ²F‚F†R7GVÂ&W6VçB6÷VçBÒF†RvÖRw2&W6VçB6÷VçB†öÆBg&ÖW2vWBÖW&vVBæBG&÷VB¢¶G"FE&W6VçFVD†æFÆW#¥â†–CÄÕDÄG&v&ÆSâB’²òò7GVÂ&W6VçFF–öâF–ÖR‡F†RF–ÖRF†—2g&ÖR—2&VÆÇ’WBöâ67&VVâÂ47W'&VçDÖVF–F–ÖR&6R¢4eF–ÖT–çFW'fÂGBÒBç&W6VçFVEF–ÖS°¢u÷&W5·âR$”äuÒÒ‡&W5öVçB—·âÂGBâòGB¢ÓÓ°¢ÕÓ°¢òòV6‚g&ÖR—2F—7Æ–VBf÷"BÆV7BöæR&W6VçBW&–öC¢v†VâF†RuR6ö×ÆWF–öâF–ÖR¦—GFW'2&÷VæB#‡¢&Vg&W6‚&÷VæF'’ÂF†R—&VB##R×2²‚ã32×2"7GWGFW"æòÆöævW"V'0¢–b‡F&vWE÷G2â’¶G"&W6VçEÓ°¢VÇ6R–b‡W&–öBâ’¶G"&W6VçDgFW$Ö–æ–×VÔGW&F–öã§W&–öEÓ²VÇ6R¶G"&W6VçEÓ°¢6VæF–`¢‡fö–B–g&ÖS°¢&WGW&â°§Ð¢òòÒÒÒÒÒÒÒÒÒÒ&W6VçBF‡&VBÒÒÒÒÒÒÒÒÒÐ¢òòådâ6VÖçF–73¢VWVU&W6VçEFW‡GW&RöæÇ’VWVW2F†R&W6VçBæB&WGW&ç2–ÖÖVF–FVÇ’‡F†RvÖRögFVâ6ÆÇ2—Bv†–ÆR†öÆF–ærÆö6³²”õ2&VÂFWf–6S¢7–æ6‡&öæ÷W6Ç’v—F–ærf÷"æW‡DG&v&ÆR–ç6–FR—@¢òòF¶W2&÷WB3×2ÂF†RÆö6²—2†VÆB&÷WB“RöbF†RF–ÖRÂæBF†RÆöF–ærF‡&VB7F'fW2’âF¶–ærF†RG&v&ÆRÂ&Æ—BæB&W6VçB&RÆÂFöæRöâF†—2F‡&VB–âVçVWVR÷&FW"à¢òòF–6¶WBçVÖ&W"Òv†–6‚VçVWVR—B—2‡7F'F–ærg&öÒ“²u÷öFöæRÒF†RF–6¶WBçVÖ&W"F†B†2&VVâ&ö6W76VB†&Æ—B6öÖÖ—GFVBFòF†RuRVWVR’à¢òò7v6†–âFW‡GW&R&WGW&ã¢&Vf÷&R7V—&UFW‡GW&RvWG2FW‡GW&R—Bv—G2f÷"F†RF–6¶WBöbF†BFW‡GW&Rw2&Wf–÷W2&W6VçBFò&R&ö6W76VB†×FÅ÷&W6VçE÷v—B’ÂgFW"v†–6‚F†R6öÖÖæG2F†RvÖRG&w2–çFò—@¢òò&RæV6W76&–Ç’÷&FW&VBgFW"F†B&Æ—BöâF†RuRVWVRà¢6FVf–æR€§G—VFVb7G'V7B²FW…÷BC²F÷V&ÆRW&–öC²V–çCcE÷Bg&ÖS²–çB7&÷³EÓ²F÷V&ÆREö–ã²V–çCcE÷BWe÷fÂÂ6æ÷c²Òö—FVÓ²òò6æ÷c¢F†R6æ6†÷B6÷’F†—2&W6VçB6†÷w2‡6æö6öÖÖ—E÷v—B’òòEö–ã¢F†RF–ÖRF†RvÖR†æFVBF†—2g&ÖR–çFòF†R&W6VçBVWVR„47W'&VçDÖVF–F–ÖR“²We÷fÃ¢6VRuöW‡EöW`¢òòW‡FW&æÂ&öGV6W"‡F†RgVÆ¶âÆ–W"ÂFÒövg‚÷f²’&VæFW'2öâ—G2÷vâVWVRæB6–væÇ2F†—2WfVçBv†Vâg&ÖRw2&VæFW&–ær—2FöæS¢F†R&W6VçBöbF†Bg&ÖRv—G2f÷"F†RfÇVRöâu÷@§7FF–2–CÄÕDÅ6†&VDWfVçCâuöW‡EöWc°§7FF–2fö–BW‡E÷v—B‡V–çCcE÷Bb’²–b‡bbbuöW‡EöWb’¶u÷Bv—Df÷$WfVçC¦uöW‡EöWbfÇVS§eÓ²Ð§7FF–2fö–B6æö6öÖÖ—E÷v—B‡V–çCcE÷Bb“°§7FF–2ö—FVÒu÷µÓ°§7FF–2V–çCcE÷Bu÷ö–ã°§7FF–2–çB&W6VçEöæ÷r†6öç7BFW…÷B¢ÂF÷V&ÆRÂV–çCcE÷BÂV–çCcE÷BÂ6öç7B–çE³EÒÂ–CÄ4ÖWFÄG&v&ÆSâÂF÷V&ÆR“°§7FF–2F÷V&ÆRuöÆ–æµ÷gW&–öBÂuöÆ–æµ÷f†Æc²òòf—'GVÂF–6·2†Æ–æµ÷&ævR“¢6V6öæG2W"6÷VçFVBF–6²ƒÒWfW'’6ÆÆ&6²—2F–6²’æB†Æb&Vg&W6‚öbF†R67&VVà ¤–çFW&f6RçdÖWFÄF—7Æ”Æ–æ´FVÆVvFR¢å4ö&¦V7BÄ4ÖWFÄF—7Æ”Æ–æ´FVÆVvFSà¤Væ@ ¢òòfÆÆ&6²f÷"F†RF—7Æ’Æ–æ²vö–ær6–ÆVçBv†–ÆR–âF†Rf÷&Vw&÷VæB‡Gvò6Öö¶R×FW7B†æw2&Vf÷&RÖW&vS¢F†RF—7Æ’Æ–æ²7F÷VB6ÆÆ–ær&6²v†–ÆRF†Rv2–âF†Rf÷&Vw&÷VæBÂæö&öG’F¶W2F†RVWVVBg&ÖW2ÂæBF†RvÖRv—G2f÷&WfW"f÷"—G2&Wf–÷W2&W6VçBFò&R&ö6W76VB&Vf÷&R&7V—&–ærF†RæW‡B7v6†–âFW‡GW&R"“ ¢òòv†Vâ×FÅ÷&W6VçE÷v—B†2v—FVBÆöævW"F†â5DÄÅôå2æBF†RF—7Æ’Æ–æ²†2æ÷B6ÆÆVB&6²f÷"F†BÆöærÂG&÷F†Rv—F–ærg&ÖRæBV&Æ–W"VWVVBg&ÖW22v†VâVçFW&–ærF†R&6¶w&÷VæB†66÷VçFVB2G&÷VBg&ÖW2’æBÆWBF†RvÖR¶VW'Vææ–æs²v†VâF†RF—7Æ’Æ–æ²6öÖW2&6²F†–æw2&WGW&âFòæ÷&ÖÂWFöÖF–6ÆÇ’à¢òòöæR7FÆÂÆöw2öæR†–çBÆ–æRÂæBöæRÆ–æRöâ&V6÷fW'’†æV—F†W"—2f–ÇW&R“²öæÇ’v†VâF†R6ÖR7FÆÂÆ7G2ÆöævW"F†â5DÄÅõ$TEôå2—2—Bf–ÇW&RâÆÂVæFW"u÷ö×P¢6FVf–æR5DÄÅôå2VÆÀ¢6FVf–æR5DÄÅõ$TEôå2#VÆÀ§7FF–2V–çCcE÷Bu÷7FÆÅ÷CÂu÷7FÆÅöG&÷Âu÷7FÆÅöã²7FF–2–çBu÷7FÆÅ÷&VBÂu÷7FÆÅöÆövvVC°¢òòv†VâF†RF—7Æ’Æ–æ²7F—26–ÆVçBf÷"Ä”äµõ$U45TUôå2v†–ÆRF†R—27F—fRÂ&W6VçG27F÷v—F–ærf÷"—C¢ÖöFR‡F†R&W6VçBF‡&VBF¶W2G&v&ÆW2—G6VÆb’Â6òF†P¢òò–7GW&R¶VW2Ö÷f–ær–ç7FVBöb7FæF–ær7F–ÆÂöâG&÷VBg&ÖW2†C'"ÖÖö&–ÆRfCcV2Âf–Ö4æF—fTvÖW26cSc#“¢â•†öæRw2Æ–æ²7F÷VBv†–ÆR7F—fRæBF†R–7GW&P¢òòg&÷¦Rf÷"ƒ3²Ö4õ2Ç6òW6W2F†RÆ–æ²öbv–æF÷r—BF¶W2f÷"6÷fW&VB’âF†RÆ–æ²—2G&–VBv–âgFW"u÷&W67VUö&6µöç2†Æ–æµ÷7–æ2’Âv†–6‚F÷V&ÆW2öâV6‚æWp¢òò7FÆÂWFòc2âF†R&WVW7FVBuöFÕ÷&W6VçEöÆ–æ²7F—2Væ6†ævVC²ÆÂ&W67VR7FFR6†ævW2F¶Ru÷ö×RÂæBÆ–æµ÷7–æ2†æFÆW2F†RÆ–æ²ö&¦V7BöâF†RVçVWVRF‡&VBà¢6FVf–æRÄ”äµõ$U45TUôå2SVÆÀ§7FF–2ôFöÖ–2–çBu÷&W67VUöÖöFS²òòÇ6ò&VB'’Æ–æµööâv—F†÷WBu÷ö×RÂFò7F÷6–æröâ6–ÆVçBF–6·2–ÖÖVF–FVÇ§7FF–2V–çCcE÷Bu÷&W67VUö&6µöBÂu÷&W67VUö&6µöç2Ò3VÆÂÂu÷&W67VUöã°¢òòF†RF—7Æ’Æ–æ²vF6†FöröæÇ’&V6övæ—¦W2F†R&7F—fR"7FFS¢v†VâF†RÆ÷6W27F—fR7FGW2‡VÆÆ–ærF÷vâ6öçG&öÂ6VçFW"òæ÷F–f–6F–öç2Â–æ6öÖ–ær6ÆÂÂ7—7FVÒ÷W’æB–âF†R&6¶w&÷VæBF†R7—7FVÒFöW2æ÷B6ÆÂ&6²ç—v’ÂæBF†R–7GW&R—26÷fW&VBFöòÒÒF†Rg&ÖRÖG&÷fÆÆ&6²7F—22&Vf÷&RÀ¢òò'WB—B—2æ÷B6÷VçFVB27GV6²æBæòf–ÇW&R—2&W÷'FVC²F†R7FÆÂ—2ÖV7W&VBg&öÒF†RÆFW"öb'F†RÖ÷7B&V6VçB6ÆÆ&6²òF†RÖ÷7B&V6VçB7F—fF–öâ"„bÆöw3¢Ž(	3r6V6öæG2–æ7F—fRæBr6V6öæG2–âF†R&6¶w&÷VæBvW&R&÷F‚&W÷'FVB2'7GV6²–âf÷&Vw&÷VæB"¥õöGG&–'WFUõò‚‡vV²’’–çBFÕöfö7W5öæ÷r‡fö–B’²&WGW&â²ÒòòF†R†÷7Bw2†ÆUööRæ2&÷f–FW2F†R7G&öærFVf–æ—F–öã²v†Vâw&†–72Væ—BFW7G2Fòæ÷BÆ–æ²F†R†÷7B—B—2G&VFVB2Çv—27F—fP¥õöGG&–'WFUõò‚‡vV²’’V–çCcE÷BFÕöfö7W5ö–å÷6–æ6R‡fö–B’²&WGW&â²Ð¤–×ÆVÖVçFF–öâçdÖWFÄF—7Æ”Æ–æ´FVÆVvFP¢Ò‡fö–B–ÖWFÄF—7Æ”Æ–æ³¢„4ÖWFÄF—7Æ”Æ–æ²¢–Æ–æ²æVVG5WFFS¢„4ÖWFÄF—7Æ”Æ–æµWFFR¢—WFFR°¢‡fö–B–Æ–æ³°¢F÷V&ÆR6"Ò47W'&VçDÖVF–F–ÖR‚“°¢F‡&VEö×WFW…öÆö6²‚fu÷ö×R“°¢–b†u÷7FÆÅ÷C’²V–çCcE÷B6–æ6RÒFÕöfö7W5ö–å÷6–æ6R‚’ÂBÒæ÷uöç2‚“°¢ÔÄôr‚&F—7Æ’Æ–æ²&V6÷fW&VC¢7FÆÆVBRãb×2‚Rãb×2öb—B7F—fR’ÂfÆÆ&6²G&÷VBVÆÇRg&ÖW2‡7FÆÂ2VÆÇR’"Â‡BÒu÷7FÆÅ÷C’òSbÂ‡BÒ†u÷7FÆÅ÷Câ6–æ6Ròu÷7FÆÅ÷C¢6–æ6R’’òSbÂ‡Vç6–væVBÆöærÆöær–u÷7FÆÅöG&÷Â‡Vç6–væVBÆöærÆöær–u÷7FÆÅöâ“²u÷7FÆÅ÷CÒ²Ð¢F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“°¢F÷V&ÆRgÒuöÆ–æµ÷gW&–öBÂFöÂÒã#²òòFöÃ¢†÷rV&Ç’F–6²Ö’6öÖRæB7F–ÆÂVæB†öÆB†&VÆ÷r¢–b‡gâ’²òòf—'GVÂF–6·3¢öæÇ’F†R&Vg&W6‚æV&W7BV6‚6Æ÷Böb7FVG’6Æö6²BF†RvçFVB&FR6÷VçG2†Æ–æµ÷&ævR¢7FF–2F÷V&ÆRæW‡BÂg÷v3²F÷V&ÆRBÒWFFRçF&vWE&W6VçFF–öåF–ÖW7F×°¢–b‡gÒg÷v2ÇÂBâæW‡B²g’²æW‡BÒC²g÷v2Òg²ÒòòF†Rf—'7B6ÆÆ&6²ÂæWr&FRÂ÷"gFW"7FÆÃ¢F†R6Æö6²7F'G2v–â†W&P¢–b‡BÂæW‡BÒuöÆ–æµ÷f†Æb’&WGW&ã°¢æW‡B³Òg²FöÂ³Ò"¢uöÆ–æµ÷f†Æc²òòF–6²Æ–W2WFò†Æb&Vg&W6‚öfb—G26Æ÷BÂ6òGvòöbF†VÒWFòöæR&Vg&W6‚æV&W"V6‚÷F†W"F†âF†R6Æ÷G0¢Ð¢7FF–2F÷V&ÆR6%÷&WbÂvöÖƒ²7FF–2–çBçF–6·3²òòf÷"uöÆ–æµ÷&–æs¢6ÆÆ&6·26÷VçFVB2F–6·26–æ6RF†R&Wf–÷W2&W6VçBÂæBF†RÆöævW7B–çFW'fÂ&WGvVVâF†VÐ¢–b†6%÷&Wbâbb6"Ò6%÷&WbâvöÖ‚’vöÖ‚Ò6"Ò6%÷&Wc°¢6%÷&WbÒ6#²çF–6·2²³°¢F‡&VEö×WFW…öÆö6²‚fuöÆ–æµö×R“°¢õöFöÖ–5÷7F÷&Uöâ‚fuöÆ–æµ÷F–6µöç2Âæ÷uöç2‚’ÂõôDôÔ”5õ$TÄ„TB“°¢uöÆ–æµ÷F–6µöâ²³°¢F‡&VEö6öæEö'&öF67B‚fuöÆ–æµö7b“°¢F‡&VEö×WFW…÷VæÆö6²‚fuöÆ–æµö×R“°¢F‡&VEö×WFW…öÆö6²‚fu÷ö×R“°¢–b‚uöFÕ÷&W6VçEöÆ–æ²ÇÂu÷&W67VUöÖöFRÇÂu÷÷&VBÓÒu÷ö–â’²–b†uöFÕ÷&W6VçEöÆ–æ²bbu÷&W67VUöÖöFR’uöÆ–æµöV×G’²³²F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“²&WGW&ã²Ð¢òòF†R&Wf–÷W2g&ÖRw2&W6VçB–çFW'fÂv2ãÒ"‡W&–öBâ#R×2“¢¶VW—Böâ67&VVâf÷"—G2v†öÆRW&–öB&Vf÷&RF¶–ærF†RæW‡BöæRÂ2F†R7v—F6‚fÆ—2öæÇ’WfW'’âfW'F–6Â7–æ72âF†—2Ç6ò†öÆG2v†–ÆRF†RÆ–æ²'Vç2Bc‡¢„Ö&–òã"ã6WG2"f÷"6–ævÆRg&ÖW2Â6VRçfåövÖUög2–â†ÆUöçfâæ2“°¢òò&Vf÷&RÂWfW'’F–6²Föö²F†RæWvW7Bg&ÖRÂ6òF†Bg&ÖRÂ6''––ærF†RW‡G&F–ÖR7FWF†RvÖRGfæ6VBf÷"—BÂv26†÷vâf÷"öæÇ’bãr×2æBF†RvÖRÆöö¶VBÆ–¶R—B§V×VB†V@¢–b†uöÆ–æµö†öÆE÷VçF–ÂâbbWFFRçF&vWE&W6VçFF–öåF–ÖW7F×ÂuöÆ–æµö†öÆE÷VçF–ÂÒFöÂ’²uöÆ–æµö†VÆB²³²F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“²&WGW&ã²Ð¢V–çCcE÷Bf—'7BÒu÷÷&VB²ÂF–6¶WBÒu÷ö–ã°¢òò–â÷&FW"ÂöæRg&ÖRW"F–6²æBBÖ÷7BöæR&V†–æBF†RæWvW7C¢Gvòv—F–ærÂF†RöÆFW"æ÷ræBF†RæWvW"öâF†RæW‡BF–6³²F‡&VR÷"Ö÷&RÂF†RöÆFW"öæW2&RG&÷VBâF¶–ærF†RæWvW7@¢òòG&÷VBF†RöæR&Vf÷&Rv†VæWfW"Gvòg&ÖW2ÆæFVB–âF†R6ÖR&Vg&W6‚Â2g&ÖW2öbcg2vÖRF†BF¶RbÓrãR×2FòÂæBF†RæW‡Bg&ÖRF†Vâ7F–VBW32×0¢òò„F&²6÷VÇ2&VÖ7FW&VBÂg&öÒ"33¢&W6VçG2v—F†–âBãrÓ‚ãr×2“‚ã"Óâ“’ã“‚RÂ&÷WB"×2Ö÷&Rg&öÒVçVWVRFòF—7Æ’’âDÕõ$U4TåEôd”dóÓF¶W2F†RæWvW7BÂ2&Vf÷&P¢7FF–2–çBf–fòÒÓ²–b†f–fòÂ’²6öç7B6†"¦RÒvWFVçb‚$DÕõ$U4TåEôd”dò"“²f–fòÒ†Rbb7G&6×†RÂ#"’“²ÔÄôr‚&F—7Æ’Æ–æ³¢W2"Âf–fòò&g&ÖW2–â÷&FW"ÂBÖ÷7BöæR&V†–æB"¢'F†RæWvW7Bg&ÖRV6‚F–6²„DÕõ$U4TåEôd”dóÓ’"“²Ð¢–b†f–fòbbF–6¶WBâf—'7B’F–6¶WBÒF–6¶WBÒf—'7BãÒ"òu÷ö–âÒ¢f—'7C°¢ö—FVÒ—BÒu÷²‡F–6¶WBÒ’RÓ²uöÆ–æµö—E÷F–âÒ—BçEö–ã°¢u÷÷&VBÒF–6¶WC°¢–b‡F–6¶WBâf—'7B’²uöÆ–æµöG&÷VB³ÒF–6¶WBÒf—'7C²f÷"‡V–çCcE÷B²Òf—'7C²²ÂF–6¶WC²²²²’u÷&W5¶²R$”äuÒÒ‡&W5öVçB—¶²ÂÓÓ²ÒòòöÆBg&ÖW2ÖW&vVBæBG&÷VB&R&V6÷&FVB2&æ÷B&W6VçFVB"Â6òF†R&W6VçB7FF—7F–72†öæW7FÇ’6÷VçBF†VÒ2G&÷VBg&ÖW0¢–b‡F–6¶WBâbbu÷öFöæRÂF–6¶WBÒ’u÷öFöæRÒF–6¶WBÒ°¢uöÆ–æµ÷&–æu·F–6¶WBR$”äuÒÒ†Æ–æµöVçB—·F–6¶WBÂWFFRçF&vWE&W6VçFF–öåF–ÖW7F×Â‡WFFRçF&vWE&W6VçFF–öåF–ÖW7F×Ò6"’¢S2ÂvöÖ‚¢S2ÂçF–6·7Ó²çF–6·2Ò²vöÖ‚Ò°¢F‡&VEö6öæEö'&öF67B‚fu÷ö7b“°¢F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“°¢–çB7V&Ö—GFVC°¢W‡E÷v—B†—BæWe÷fÂ“²6æö6öÖÖ—E÷v—B†—Bç6æ÷b“°¢WF÷&VÆV6WööÂ²7V&Ö—GFVBÒ&W6VçEöæ÷r‚f—BçBÂ—BçW&–öBÂ—Bæg&ÖRÂF–6¶WBÂ—Bæ7&÷ÂWFFRæG&v&ÆRÂWFFRçF&vWE&W6VçFF–öåF–ÖW7F×“²Ð¢uöÆ–æµö†öÆE÷VçF–ÂÒ7V&Ö—GFVBbb—BçW&–öBâã#RòWFFRçF&vWE&W6VçFF–öåF–ÖW7F×²—BçW&–öB¢°¢–b‚7V&Ö—GFVB’°¢F‡&VEö×WFW…öÆö6²‚fu÷ö×R“°¢–b‡F–6¶WBâu÷öFöæR’u÷öFöæRÒF–6¶WC°¢F‡&VEö6öæEö'&öF67B‚fu÷ö7b“°¢F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“°¢Ð§Ð¤Væ@ §7FF–24ÖWFÄF—7Æ”Æ–æ²¦uöÆ–æ³²7FF–24e'VäÆö÷&VbuöÆ–æµ÷&Ã²7FF–2–çBuöÆ–æµ÷7F'FVC°§7FF–2–çBuöÆ–æµö÷vç5öÆ–W"Âuö6Æ76–5ö'W7“²òòVæFW"u÷ö×S¢æW‡DG&v&ÆRF‡&÷w2v†–ÆRF†RÆ–W"†2F†RF—7Æ’Æ–æ²ö&¦V7BGF6†VBÂ6òF†R&W6VçBF‡&VBF†BF¶W2—G2÷vâG&v&ÆRæBF†RF—7Æ’Æ–æ²ö&¦V7B&R×WGVÆÇ’W†6ÇW6—fP§7FF–2–çBÆ–æµööâ‡fö–B’²&WGW&âuöFÕ÷&W6VçEöÆ–æ²bbFöÖ–5öÆöB‚fu÷&W67VUöÖöFR’bbuöÆ–æ²Òæ–Ã²Ð¢òò67&VVç2v†÷6R&Vg&W6‚&FR—2æ÷B×VÇF—ÆRöbF†RÆ–æ²w2ƒCB÷"cR‡¢v–ç7Bc÷"3“¢6¶VBf÷"c‡¢ÂF†R7—7FVÒf—&W2F†RÆ–æ²öâWfW'’F†—&B&Vg&W6‚öbcR‡¢67&VVâÀ¢òòSR‡¢ÂæBvÖR6VB'’F†RÆ–æ²'Vç2BSRg2ƒ#rãRB3’âF†W&RF†RÆ–æ²'Vç2BF†R67&VVâw2&FRæBöæÇ’F†R&Vg&W6†W2æV&W7B7FVG’6Æö6²öbF†RvçFVB&FR6÷VçB0¢òòF–6·2†uöÆ–æµ÷gW&–öBÂçdÖWFÄF—7Æ”Æ–æ´FVÆVvFR“¢c6V6öæBöâfW&vRÂV6‚g&ÖR6†÷vâöâF†R&Vg&W6†W2æV&W7B—G2óc26Æ÷BâDÕôÄ”äµõeD”4³Ó¶VW2F†R7—7FVÒw26†ö–6Rà¢òò–âgVÆÇ67&VVâF†RÖ2g&öçBVæB7v—F6†W27V6‚67&VVâFò&FRF†B—2×VÇF—ÆRv†W&R—B†2öæR†g&öçE÷67&VVåöÖF6‚’ÂæBF†R&ævR—26WBv–âf÷"—B†Æ–æµ÷7–æ2¥õöGG&–'WFUõò‚‡vV²’’–çBg&öçE÷67&VVåöÖ…ög2‡fö–B’²&WGW&â²Òòòg&öçEöÖ2æÓ²w&†–72Væ—BFW7G2æB”õ2†fRæöæP¥õöGG&–'WFUõò‚‡vV²’’fö–Bg&öçE÷67&VVåöÖF6‚†–çBg2’²‡fö–B–g3²Òòòg&öçEöÖ2æÓ²w&†–72Væ—BFW7G2æB”õ2†fRæöæP§7FF–24g&ÖU&FU&ævRÆ–æµ÷&ævR†–çBÖöFR’°¢7FF–2–çBöâÒÓ²–b†öâÂ’²6öç7B6†"¦RÒvWFVçb‚$DÕôÄ”äµõeD”4²"“²öâÒ†Rbb7G&6×†RÂ#"’“²Ð¢–çB&FRÒÖöFRÓÒ2ò3¢ÖöFRÓÒ"ò#¢cÂ‡¢Òöâòg&öçE÷67&VVåöÖ…ög2‚’¢°¢–b†ÖöFRÒ"bb‡¢â&FRbb‡¢R&FR’°¢–b†uöÆ–æµ÷gW&–öBÒãò&FR’ÔÄôr‚&F—7Æ’Æ–æ²BF†R67&VVâw2VB‡¢Â6÷VçF–ærF–6²WfW'’òVB2‡F†R67&VVâw2&FR—2æ÷B×VÇF—ÆRöb—B’"Â‡¢Â&FR“°¢uöÆ–æµ÷f†ÆbÒãRò‡£²uöÆ–æµ÷gW&–öBÒãò&FS°¢&WGW&â4g&ÖU&FU&ævTÖ¶R‡&FRÂ‡¢Â‡¢“°¢Ð¢uöÆ–æµ÷gW&–öBÒ°¢&WGW&âÖöFRÓÒ2ò4g&ÖU&FU&ævTÖ¶Rƒ3Â3Â3’¢ÖöFRÓÒ"ò4g&ÖU&FU&ævTÖ¶RƒcÂ#Â#’¢4g&ÖU&FU&ævTÖ¶RƒcÂcÂc“°§Ð¢òòF†RF—7Æ’Æ–æ²w2g&ÖRÆFVæ7“¢"–âf&–&ÆR×&Vg&W6‚gVÆÇ67&VVâÂ÷F†W'v—6R„DÕõ$U4TåEôÄDTä5’÷fW'&–FW2&÷F‚’â–âv–æF÷rF†R7—7FVÒ6ÆÇ2&6²&÷WBS×2&Vf÷&RF†RF&vWBF—7Æ¢òòF–ÖS²–âgVÆÇ67&VVâ—B6†÷w2F†RÆ–W"F—&V7FÇ’æBÂBÆFVæ7’Â6ÆÇ2&6²öæÇ’öæRc‡¢W&–öBƒb×2’†VBâF†W&RÂöâ&ôÖ÷F–öâ67&VVç2„ÓB&òÂÓRÖ‚’Â—B6†÷w26öÖP¢òò&W6VçG2öæR#‡¢&Vg&W6‚ÆFRæBF†RÆ–æ²6¶—2—G2æW‡BF–6²ÂF†÷Vv‚F†R6ÆÆ&6·2'VâöâF–ÖRæBæòuRv÷&²—2–âF†Rv“²F†Rg&ÖR7FW&VBF†RÆFRg&ÖW22Æö@¢òòæB7FWVBFò#g2öâg&ÖW26÷7F–ærãÓãbg7–æ72ÂæBF†RvÖR&W6VçFVBRg&ÖW26V6öæB†—77VW233CrÂ33C‚’âÆFVæ7’"v—fW232×2æB&VÖ÷fVBÖ÷7BöbF†RÖ—76W2öà¢òò&÷F‚Ö6†–æW2Âf÷"öæRÖ÷&Rg&ÖRöb–çWBÆr–âf&–&ÆR×&Vg&W6‚gVÆÇ67&VVâöæÇ’âf—†VB×&FR67&VVç2Fòæ÷BæVVBF†—2v÷&¶&÷VæB‚3ƒ#"’âv‡’F†R7—7FVÒÖ—76W2Bb×2À¢òòæBv‡’f—†VB×&FR67&VVâ7F–ÆÂ†2&÷WB#×2Ö÷&R&W6VçFF–öâFVÆ’v—F‚F†RÆ–æ²BV—F†W"ÆFVæ7’Â&VÖ–â÷Vâ†Fö72÷&VfW&Væ6Rög&ÖR×&FRæÖB¥õöGG&–'WFUõò‚‡vV²’’–çBg&öçEögVÆÇ67&VVâ‡fö–B’²&WGW&â²Òòòg&öçEöÖ2æÓ²w&†–72Væ—BFW7G2æB”õ2†fRæöæP¥õöGG&–'WFUõò‚‡vV²’’–çBg&öçE÷67&VVå÷f&–&ÆU÷&Vg&W6‚‡fö–B’²&WGW&â²Òòòg&öçEöÖ2æÓ²w&†–72Væ—BFW7G2æB”õ2†fRæöæP¢òò4ÖWFÄF—7Æ”Æ–æ²66WG2öæÇ’÷""â6ö×&R—G2&VF&6²v—F‚F†R7W÷'FVB&WVW7BÂ÷"â÷fW'6—¦VB÷fW'&–FR&Ww&—FW2æBÆöw2WfW'’g&ÖRà§7FF–2–çBÆ–æµöÆFVæ7’‡fö–B’²&WGW&âu÷&W6VçEöÆFVæ7’âòÔ”â†u÷&W6VçEöÆFVæ7’Â"’¢g&öçEögVÆÇ67&VVâ‚’bbg&öçE÷67&VVå÷f&–&ÆU÷&Vg&W6‚‚’ò"¢²Ð§7FF–26öç7B6†"¦Æ–æµöÆFVæ7•÷v‡’‡fö–B’²&WGW&âu÷&W6VçEöÆFVæ7’âò$DÕõ$U4TåEôÄDTä5’"¢g&öçEögVÆÇ67&VVâ‚’ò'v–æF÷r"¢g&öçE÷67&VVå÷f&–&ÆU÷&Vg&W6‚‚’ò&gVÆÇ67&VVâÂf&–&ÆR&Vg&W6‚"¢&gVÆÇ67&VVâÂf—†VB&Vg&W6‚#²Ð§7FF–2–çBuöÆ–æµöÖöFRÒ²òò7W'&VçBÖöFRöbF†RF—7Æ’Æ–æ²ö&¦V7C¢ò"6ÖR2uöFÕ÷&W6VçEöÆ–æ²Â2Ò÷vW"×6f–ærÖöFR3‡ §7FF–2fö–B¦Æ–æµöÖ–â‡fö–B¦&r’²òòF—7Æ’Æ–æ²F‡&VC¢7F'FVBv†Vâ7v—F6†–ærFòF—7Æ’ÖÆ–æ²ÖöFRÂW†—G2v†Vâ7v—F6†–ær&6²Fò ¢‡fö–B–&s°¢F‡&VE÷6WFæÖUöç‚&çfâ×&W6VçBÖÆ–æ²"“°¢F‡&VE÷6WE÷÷5ö6Æ75÷6VÆeöç…õ5ô4Ä55õU4U%ô”åDU$5D•dRÂ“°¢4ÖWFÄÆ–W"¦Æ–W"Ò…õö'&–FvR4ÖWFÄÆ–W"¢–g&öçEöÖWFÅöÆ–W"‚“°¢F‡&VEö×WFW…öÆö6²‚fu÷ö×R“²uöÆ–æµö÷vç5öÆ–W"Ò²v†–ÆR†uö6Æ76–5ö'W7’’F‡&VEö6öæE÷v—B‚fu÷ö7bÂfu÷ö×R“²F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“²òòv—Bf÷"F†R&W6VçBF‡&VBFòf–æ—6‚†æF–ær÷fW"F†Rg&ÖR—B†öÆG0¢çdÖWFÄF—7Æ”Æ–æ´FVÆVvFR¦FVÆVvFRÒ´çdÖWFÄF—7Æ”Æ–æ´FVÆVvFRæWuÓ°¢4ÖWFÄF—7Æ”Æ–æ²¦Æ–æ²Òµ´4ÖWFÄF—7Æ”Æ–æ²ÆÆö5Ò–æ—Ev—F„ÖWFÄÆ–W#¦Æ–W%Ó°¢Æ–æ²ç&VfW'&VDg&ÖU&FU&ævRÒÆ–æµ÷&ævR†uöÆ–æµöÖöFR“°¢Æ–æ²ç&VfW'&VDg&ÖTÆFVæ7’ÒÆ–æµöÆFVæ7’‚“²ÔÄôr‚&F—7Æ’Æ–æ³¢g&ÖRÆFVæ7’VB‚W2’"ÂÆ–æµöÆFVæ7’‚’ÂÆ–æµöÆFVæ7•÷v‡’‚’“°¢Æ–æ²æFVÆVvFRÒFVÆVvFS°¢¶Æ–æ²FEFõ'VäÆö÷¥´å5'VäÆö÷7W'&VçE'VäÆö÷Òf÷$ÖöFS¤å4FVfVÇE'VäÆö÷ÖöFUÓ°¢uöÆ–æµ÷&ÂÒ4e'VäÆö÷vWD7W'&VçB‚“²uöÆ–æ²ÒÆ–æ³°¢4e'VäÆö÷'Vâ‚“²òòv†Vâ7v—F6†–ær&6²FòÖöFRF†RF—7Æ’Æ–æ²ö&¦V7B—2–çfÆ–FFVBæBF†—2F‡&VBW†—G2†æW‡DG&v&ÆR6ææ÷BvWBG&v&ÆRv†–ÆRF†RÆ–W"7F–ÆÂ†2F†RF—7Æ’Æ–æ²ö&¦V7BGF6†VBÂ6òW6–ærÆöæR—2æ÷BVæ÷Vv‚¢uöÆ–æµ÷7F'FVBÒ°¢&WGW&âåTÄÃ°§Ð§7FF–2fö–BÆ–æµ÷7–æ2‡fö–B’²òò6ÆÆVBWfW'’g&ÖR'’F†RvÖRw2&W6VçBF‡&VC¢Æ–vç2F†RF—7Æ’Æ–æ²ö&¦V7Bw27FFRFòuöFÕ÷&W6VçEöÆ–æ²æBF†R÷vW"×6f–ærÖöFR†6â6†ævRB'VâF–ÖR¢7FF–2–çBÆ–VC°¢F‡&VEö×WFW…öÆö6²‚fu÷ö×R“°¢–çBvçBÒuöFÕ÷&W6VçEöÆ–æ³°¢–b‡vçBÂÇÂvçBâ"’²uöFÕ÷&W6VçEöÆ–æ²ÒÆ–VBÓÒ2ò¢Æ–VC²F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“²&WGW&ã²Ð¢–b†u÷&W67VUöÖöFRbb‡vçBÒu÷&W67VUöÖöFRÇÂæ÷uöç2‚’ãÒu÷&W67VUö&6µöB’’°¢u÷&W67VUöÖöFRÒ²u÷7FÆÅ÷CÒ²õöFöÖ–5÷7F÷&Uöâ‚fuöÆ–æµ÷F–6µöç2Âæ÷uöç2‚’ÂõôDôÔ”5õ$TÄ„TB“°¢ÔÄôr‚&F—7Æ’Æ–æ³¢&W67VRVæFVBÂG'––ær6–ærÖöFRVB†g&ÖRVÆÇR’"ÂvçBÂ‡Vç6–væVBÆöærÆöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR’“°¢Ð¢–b†u÷&W67VUöÖöFR’vçBÒ°¢F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“°¢–b‡vçBbb×FÅög5÷F&vWB‚’ÓÒ3’vçBÒ3²òò÷vW"×6f–ærÖöFS¢F—7Æ’Æ–æ²B3‡ ¢–b‡vçB’g&öçE÷67&VVåöÖF6‚‡vçBÓÒ2ò3¢vçBÓÒ"ò#¢c“°¢òòF†R67&VVâw2&FR6†ævVB‡7v—F6†VB'’g&öçE÷67&VVåöÖF6‚Â÷"æ÷F†W"67&VVâ“¢F†R&ævRv–âÂf—'GVÂF–6·2÷"æöæRâF¶VâöæÇ’öæ6RF†RÆ–æ²ö&¦V7BW†—7G2Â6ò6†ævP¢òòv†–ÆR—G2F‡&VB7F'G2†gFW"Æ–æµöÖ–â&VBF†R&FR’—2æ÷BÆ÷7@¢7FF–2–çB‡¥÷v3²–çB‡¢Òg&öçE÷67&VVåöÖ…ög2‚“²4ÖWFÄF—7Æ”Æ–æ²¦Æ²ÒuöÆ–æ³°¢–b†Æ²bb‡¢Ò‡¥÷v2’²‡¥÷v2Ò‡£²–b‡vçBbbvçBÓÒÆ–VB’²Æ²ç&VfW'&VDg&ÖU&FU&ævRÒÆ–æµ÷&ævR‡vçB“²ÔÄôr‚&F—7Æ’Æ–æ³¢F†R67&VVâ'Vç2BVB‡¢æ÷rÂ&ævR6WBv–â†g&ÖRVÆÇR’"Â‡¢Â‡Vç6–væVBÆöærÆöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR’“²ÒÐ¢–çBÆBÒÆ–æµöÆFVæ7’‚“²òògVÆÇ67&VVâÂ67&VVâ÷"&Vg&W6‚ÖöFR6†ævVC¢Æ–æ²7&VFVB6–æ6RFöö²F†R7W'&VçBöÆ–7¢–b†Æ²bbÆ²ç&VfW'&VDg&ÖTÆFVæ7’ÒÆB’²Æ²ç&VfW'&VDg&ÖTÆFVæ7’ÒÆC²ÔÄôr‚&F—7Æ’Æ–æ³¢g&ÖRÆFVæ7’VB‚W3²g&ÖRVÆÇR’"ÂÆBÂÆ–æµöÆFVæ7•÷v‡’‚’Â‡Vç6–væVBÆöærÆöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR’“²Ð¢–b‡vçBÓÒÆ–VB’&WGW&ã°¢uöÆ–æµöÖöFRÒvçC°¢–b‚uöÆ–æµ÷7F'FVB’°¢–b‚vçB’²Æ–VBÒ²&WGW&ã²Ð¢–b‚g&öçEöÖWFÅöÆ–W"‚’’&WGW&ã°¢uöÆ–æµ÷7F'FVBÒ²F‡&VE÷BFƒ²–b‡F‡&VEö7&VFR‚gF‚ÂåTÄÂÂÆ–æµöÖ–âÂåTÄÂ’’ÔD”R‚&6ææ÷B7F'BF†RF—7Æ’Æ–æ²F‡&VB"“²F‡&VEöFWF6‚‡F‚“°¢&WGW&ã²òòF†RF—7Æ’Æ–æ²ö&¦V7B—27&VFVBöâæWrF‡&VC²Æ–vâv–âæW‡Bg&ÖP¢Ð¢4ÖWFÄF—7Æ”Æ–æ²¦Æ–æ²ÒuöÆ–æ³²–b‚Æ–æ²’&WGW&ã°¢–b‡vçB’Æ–æ²ç&VfW'&VDg&ÖU&FU&ævRÒÆ–æµ÷&ævR‡vçB“°¢VÇ6R²4e'VäÆö÷&Vb&ÂÒuöÆ–æµ÷&Ã²uöÆ–æ²Òæ–Ã²4e'VäÆö÷W&f÷&Ô&Æö6²‡&ÂÂ´4e'VäÆö÷FVfVÇDÖöFRÂç²¶Æ–æ²–çfÆ–FFUÓ²F‡&VEö×WFW…öÆö6²‚fu÷ö×R“²uöÆ–æµö÷vç5öÆ–W"Ò²F‡&VEö6öæEö'&öF67B‚fu÷ö7b“²F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“²4e'VäÆö÷7F÷‡&Â“²Ò“²4e'VäÆö÷v¶UW‡&Â“²Ð¢ÔÄôr‚'&W6VçB6–ærÖöFRVB(i"VB†g&ÖRVÆÇR’"ÂÆ–VBÂvçBÂ‡Vç6–væVBÆöærÆöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR’“°¢Æ–VBÒvçC°¢F‡&VEö×WFW…öÆö6²‚fu÷ö×R“²F‡&VEö6öæEö'&öF67B‚fu÷ö7b“²F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“°§Ð¢òò&Vf÷&Rvö–ærFòF†R&6¶w&÷VæB†×FÅöG&–â“¢v†VâF†RF—7Æ’Æ–æ²G&—fW2&W6VçG2Âg&ÖW2VWVVB'WBæ÷B–WB&W6VçFVB6âöæÇ’&RF¶Vâ'’F†RF—7Æ’ÖÆ–æ²6ÆÆ&6²ÂæB–âF†R&6¶w&÷VæBF†R7—7FVÒæòÆöævW"6ÆÇ2&6²ÒÐ¢òòv—F–ær2Ö—2f÷"'F†R&W6VçBVWVRFòV×G’"v÷VÆBv—Bf÷&WfW"ÂF†RVçFW"Ö&6¶w&÷VæB6ÆÆ&6²v÷VÆBæ÷B&WGW&âÂæBF†R7—7FVÒv÷VÆB¶–ÆÂF†R&ö6W72‡F†R&ö6W72F—6V'2v—F†–â#"6V6öæG2gFW"7v—F6†–ærFòæ÷F†W"’à¢òòF†W6Rg&ÖW2v–ÆÂæ÷B&R6VVâç—v“¢F—66&BF†VÒF—&V7FÇ’†66÷VçFVB2G&÷VBg&ÖW2’æBöæÇ’v—Bf÷"F†RöæR&W6VçBÇ&VG’7V&Ö—GFVBFòF†RuRFòf–æ—6‚à§7FF–2fö–B&W6VçEöG&÷öf÷%÷7W7VæB‡fö–B’°¢F‡&VEö×WFW…öÆö6²‚fu÷ö×R“°¢–b†uöÆ–æµö÷vç5öÆ–W"’°¢V–çCcE÷BâÒu÷ö–âÒu÷÷&VC°¢f÷"‡V–çCcE÷B²Òu÷÷&VB²²²ÃÒu÷ö–ã²²²²’u÷&W5¶²R$”äuÒÒ‡&W5öVçB—¶²ÂÓÓ°¢uöÆ–æµöG&÷VB³Òã²u÷÷&VBÒu÷ö–ã°¢v†–ÆR†u÷ö–æfÆ–v‡B’F‡&VEö6öæE÷v—B‚fu÷ö7bÂfu÷ö×R“°¢–b†u÷öFöæRÂu÷ö–â’u÷öFöæRÒu÷ö–ã°¢F‡&VEö6öæEö'&öF67B‚fu÷ö7b“°¢–b†â’ÔÄôr‚&VçFW&–ærF†R&6¶w&÷VæC¢G&÷–ærVÆÇRVWVVBg&ÖW2æ÷B–WBF—7Æ–VB‡F†RF—7Æ’Æ–æ²FöW2æ÷Bf—&R–âF†R&6¶w&÷VæB’"Â‡Vç6–væVBÆöærÆöær–â“°¢Ð¢F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“°§Ð§7FF–2fö–B§&W6VçEöÖ–â‡fö–B¦&r’°¢‡fö–B–&s°¢F‡&VE÷6WFæÖUöç‚&çfâ×&W6VçB"“°¢F‡&VE÷6WE÷÷5ö6Æ75÷6VÆeöç…õ5ô4Ä55õU4U%ô”åDU$5D•dRÂ“°¢÷5ö6Æ75÷B2Òõ5ô4Ä55õTå5T4”d”TC²–çB'Ò²F‡&VEövWE÷÷5ö6Æ75öç‡F‡&VE÷6VÆb‚’Âg2Âg'“²FöÖ–5÷7F÷&R‚fu÷&W6VçE÷÷2Â‡V–çC3%÷B—2“°¢f÷"ƒ³²’°¢F‡&VEö×WFW…öÆö6²‚fu÷ö×R“°¢v†–ÆR†u÷÷&VBÓÒu÷ö–âÇÂuöÆ–æµö÷vç5öÆ–W"’F‡&VEö6öæE÷v—B‚fu÷ö7bÂfu÷ö×R“²òò–âF—7Æ’ÖÆ–æ²ÖöFRF†RF—7Æ’ÖÆ–æ²6ÆÆ&6²F¶W2F†VÓ²âV×G’VWVR—2§VFvVB'’'F¶Vâ#²&FöæR"†2Fòv—Bf÷"F†RfVVF&6²öbF†R&W6VçB7V&Ö—BæBÆw2&V†–æBF¶Và¢ö—FVÒ—BÒu÷¶u÷÷&VBRÓ°¢V–çCcE÷BF–6¶WBÒ²¶u÷÷&VC²uö6Æ76–5ö'W7’Ò°¢F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“°¢–çB7V&Ö—GFVC°¢W‡E÷v—B†—BæWe÷fÂ“²6æö6öÖÖ—E÷v—B†—Bç6æ÷b“°¢WF÷&VÆV6WööÂ²7V&Ö—GFVBÒ&W6VçEöæ÷r‚f—BçBÂ—BçW&–öBÂ—Bæg&ÖRÂF–6¶WBÂ—Bæ7&÷Âæ–ÂÂ“²Ð¢F‡&VEö×WFW…öÆö6²‚fu÷ö×R“²uö6Æ76–5ö'W7’Ò²F‡&VEö6öæEö'&öF67B‚fu÷ö7b“²F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“°¢–b‚7V&Ö—GFVB’°¢F‡&VEö×WFW…öÆö6²‚fu÷ö×R“°¢–b‡F–6¶WBâu÷öFöæR’u÷öFöæRÒF–6¶WC°¢F‡&VEö6öæEö'&öF67B‚fu÷ö7b“°¢F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“°¢Ð¢Ð¢&WGW&âåTÄÃ°§Ð§V–çC3%÷B×FÅ÷&W6VçE÷÷2‡fö–B’²&WGW&âFöÖ–5öÆöB‚fu÷&W6VçE÷÷2“²Ð§7FF–2fö–B&W6VçE÷7F'B‡fö–B’²F‡&VE÷BFƒ²–b‡F‡&VEö7&VFR‚gF‚ÂåTÄÂÂ&W6VçEöÖ–âÂåTÄÂ’’ÔD”R‚&6ææ÷B7F'BF†R&W6VçBF‡&VB"“²F‡&VEöFWF6‚‡F‚“²Ð§7FF–2ôFöÖ–2–çBu÷&W6VçEöF—66&C°§fö–B×FÅöÆ–æµ÷W6R†–çBöâ“°¢òòF†R6†÷'BW&–öBv†VâF†Rg&öçBVæBvöW2FòF†R&6¶w&÷VæBæBv—G2f÷"F†RvÖRFò6fS¢&W6VçG2&RG&÷VBF—&V7FÇ’‡F†RF—7Æ’Æ–æ²FöW2æ÷B6ÆÂ&6²–âF†R&6¶w&÷VæBÂVçVWVV–ærv÷VÆB&Æö6²F†RvÖRF‡&VC²æö&öG’6VW2F†R–7GW&Rç—v’’à¢òòv†VâGW&æVBöâÂÇ6ò6ÆV"F†Rg&ÖW2Ç&VG’VWVVC¢F†R7—7FVÒf—'7B7F÷2F†RF—7Æ’ÖÆ–æ²6ÆÆ&6²æBöæÇ’gFW'v&G2æ÷F–f–W2&÷WBvö–ærFòF†R&6¶w&÷VæBÂæB–âF†BvF†RvÖR†2Ç&VG’f–ÆÆVBF†R7v6†–âæB—27GV6²B&&Vf÷&R7V—&–ærF†RæW‡BFW‡GW&RÂv—Bf÷"—G2&Wf–÷W2&W6VçBFò&R&ö6W76VB ¢òò††ÆUöçfâæ2çfåv–æF÷t7V—&UFW‡GW&RÓâ×FÅ÷&W6VçE÷v—B’âv—F†÷WB6ÆV&–ærÂF†RvÖR6ææ÷BGfæ6R6–ævÆRF–6²GW&–ærF†÷6R"ãR6V6öæG2öbv—F–ærf÷"F†R6fRÂæBF†R6fR—2æ÷Bw&—GFVâ‡F†R6ÖRGFW&âF‡&VRF–ÖW2–âÆ–W"Æöw3¢2g&ÖW2VWVVBÂ7V&Ö—G2§fö–B×FÅ÷&W6VçEöF—66&B†–çBöâ’²FöÖ–5÷7F÷&R‚fu÷&W6VçEöF—66&BÂöâÒ“²–b†öâ’&W6VçEöG&÷öf÷%÷7W7VæB‚“²Ð§7FF–2V–çCcE÷B&W6VçEöVçVWVR†6öç7BFW…÷B§BÂF÷V&ÆRW&–öBÂ6öç7B–çB7&÷³EÒÂV–çCcE÷BWe÷fÂ“°§V–çCcE÷B×FÅ÷&W6VçB†6öç7BFW…÷B§BÂF÷V&ÆRW&–öBÂ6öç7B–çB7&÷³EÒ’²&WGW&â&W6VçEöVçVWVR‡BÂW&–öBÂ7&÷Â“²Ð§fö–B¦×FÅöFWf–6R‡fö–B’²×FÅö–æ—B‚“²&WGW&â…õö'&–FvRfö–B¢–uöFWc²Ð§fö–B¦×FÅöFWf–6Uö6Æ76–2‡fö–B’²uö6Æ76–5öö²Ò²&WGW&â×FÅöFWf–6R‚“²ÒòòF†RtÂæBgVÆ¶âÆ–W'3¢FWf–6Rv—F†÷WBÖWFÂB—2F¶VâFöòÂv—F‚&W6VçG2G&÷VB†×FÅö–æ—B¢òòâW‡FW&æÂ&öGV6W"w2FW‡GW&R‡F†RgVÆ¶âÆ–W"w27v6†–â–ÖvW2“¢6W&–ÂF†R&W6VçBF‚&W6öÇfW2‡6W&–Å÷WB’Â&W6–FVçBf÷"F†RÖWFÂBVWVRÂæWfW"6öÆÆV7FVB'’F†R7&÷72×7V&Ö—BÆVFvW §V–çCcE÷B×FÅöW‡E÷FW‚‡fö–B§FW‚’°¢7FF–2ôFöÖ–2V–çCcE÷Bã²–CÄÕDÅFW‡GW&SâBÒ…õö'&–FvR–CÄÕDÅFW‡GW&Sâ—FWƒ²V–çCcE÷B6W"ÒƒVÆÂÃÂc"’²FöÖ–5öfWF6…öFB‚fâÂ’²°¢&W5öFB‡B“²6W&–Å÷WB‡6W"ÂB“²×FÅ÷‡e÷v–æF÷u÷FW‚‡6W"“²&WGW&â6W#°§Ð¢òò&W6VçBâW‡FW&æÂFW‡GW&RgFW"F†R&öGV6W"w2WfVçB&V6†W2We÷fÂ†—G2&VæFW&–æröbF†Rg&ÖR—2FöæR“²&WGW&ç2F†RF–6¶WBf÷"×FÅ÷&W6VçE÷v—@§V–çCcE÷B×FÅ÷&W6VçEöW‡B‡V–çCcE÷B6W&–ÂÂF÷V&ÆRW&–öBÂfö–B¦WbÂV–çCcE÷BWe÷fÂÂ6öç7B–çB7&÷³EÒ’°¢FW…÷BC²ÖV×6WB‚gBÂÂ6—¦VöbB“²BæÖv–2ÒÔt”2‚uBrÂtRrÂu‚rÂ“²Bç6W&–ÂÒ6W&–Ã°¢–b†WbbbuöW‡EöWb’uöW‡EöWbÒ…õö'&–FvR–CÄÕDÅ6†&VDWfVçCâ–Wc°¢&WGW&â&W6VçEöVçVWVR‚gBÂW&–öBÂ7&÷ÂWbòWe÷fÂ¢“°§Ð¦W‡FW&âôFöÖ–2V–çCcE÷BFÕ÷&W6VçEöç3²òò†ÆU÷F‡&VBæ2„DÕô„•D4…ôETÕ“²F†RådâÂgVÆ¶âæBtÂ&W6VçG2ÆÂ6öÖRF‡&÷Vv‚&W6VçEöVçVWVP¢òò&W6VçB6æ6†÷G2„DÕõ$U4TåEõ4ä’âvÖRv—F‚Gvò7v6†–âFW‡GW&W2„'&VF‚öbF†Rv–ÆB’vWG2öæR&6²öæÇ’v†VâF†RF—7Æ’Æ–æ²†2F¶Vâ—G2&Wf–÷W2&W6VçBöâ¢òòF–6²Â6òF†RæW‡Bg&ÖRw25Rv÷&²7F'G2öâF†RF–6²æBF†RuR–FÆW2g&öÒF†RVæBöböæRg&ÖRFòF†Rf—'7Bv÷&²öbF†RæW‡BâÖ4õ2&VG2F†B–FÆRF–ÖR2&ööÒæB†öÆG2F†RuP¢òòB—G2Æ÷vW7B6Æö6²ƒCCBÔ‡¢öââÓ"Ö‚’WfVâv†–ÆRg&ÖW2Ö—72F†R67&VVâÂæBF†W&R"×2öbuRv÷&²g&ÖRÇW2F†RF–ÖRFòF†Rg&ÖRw2f—'7B7V&Ö—76–öâögFVâÖ—76VBbãr×2à¢òòv—F‚6æ6†÷G2V6‚&W6VçB—26÷–VBöâF†RvÖRw2VWVRÂ&–v‡BgFW"F†Rg&ÖRw2v÷&²Â–çFòFW‡GW&RöbF†R†÷7Bw2÷vâ†&–æröbu÷6æöâ’ÂF†RF—7Æ’Æ–æ²6†÷w2F†R6÷’Âæ@¢òòF†RvÖRw2FW‡GW&R—2g&VRBöæ6S¢F†RvÖR'Vç2WFòg&ÖR†VBöbF†R67&VVâÂæBg&ÖRv†÷6RuRv÷&²'Vç2Æöær—2'6÷&&VB–ç7FVBöb6†÷vâÆFRâ&–ær6Æ÷B—2&WW6V@¢òòöæ6R—G2&W6VçB†2&VVâ&ö6W76VBæB—G26÷’†26ö×ÆWFVC²7V—&UFW‡GW&Rv—G2f÷"F†B66—G’æB6÷VçG2—B26–ærÂæ÷B2F†Rg&ÖRw26÷7Bà¢òò&W6VçBF†B&V6†W2'W7’6Æ÷B‡6WfW&ÂFW‡GW&W27V—&VB†VBöbF–ÖR’¶VW2F†R÷&–v–æÂFW‡GW&RæB—G2F–6¶WB–ç7FVBöbv—F–ærv†–ÆRF†RvÖRÖ’†öÆBÆö6²à¢òòv—F‚ÖWFÂVæ6öF–æröâ—G2÷vâF‡&VB„DÕôådåôTä5õD…$TB’F†R6÷’—2&Æö6²öbF†RVæ6öFRF‡&VBw2VWVR‡&V5ö÷vå÷W6‚“¢—B'Vç2&V†–æBWfW'—F†–ærF†RvÖR7V&Ö—GFVB&Vf÷&P¢òòF†R&W6VçBÂF†R&W6VçF–ærF‡&VBFöW2æ÷Bv—Bf÷"F†RVæ6öFRF‡&VBÂæBF†Rg&ÖRw2&W6VçBv—G2öæÇ’VçF–ÂF†B&Æö6²—2Væ6öFVB‡6æö6öÖÖ—E÷v—B’Â&Vf÷&R—G2&Æ—Bà¢òòF†R&–ærw2ÆVæwF‚—2F†RçVÖ&W"öbg&ÖW2F†RvÖRÖ’'Vâ†VBöbF†R67&VVã¢DÕõ$U4TåEõ4äÓÆãâƒ"Fò4äôÔƒ²Ò"’âöfb'’FVfVÇC²F†R”õ2ÆVæ6†W"w2g&ÖR'VffW&–ærFövvÆRVæ&ÆW2—Bà¢6FVf–æR4äôÔ‚@¢òòöÆC¢F†R6Æ÷Bw2FW‡GW&R&Vf÷&RF†R&W6VçFVB6—¦R÷"f÷&ÖBÆ7B6†ævVBÂ&W6–FVçBVçF–ÂF†R6Æ÷Bw2æW‡BW6R‡6æ÷F¶R§7FF–27G'V7B²–CÄÕDÅFW‡GW&SâFW‚ÂöÆC²V–çCcE÷B6W&–Ã²ôFöÖ–2V–çCcE÷BF–6¶WBÂWc²Òu÷6æµ4äôÔ…Ó°§7FF–2–CÄÕDÃD6öÖÖæDÆÆö6F÷#âu÷6æöÆÆö5µ4äôÔ…Ó²7FF–2–CÄÕDÃD6öÖÖæD'VffW#âu÷6æö6%µ4äôÔ…Ó°§7FF–2–CÄÕDÅ&W6–FVæ7•6WCâu÷6æ÷&W3²7FF–2–CÄÕDÅ6†&VDWfVçCâu÷6æöWc²7FF–2V–çCcE÷Bu÷6æ÷c°§7FF–2ôFöÖ–2V–çCcE÷Bu÷6æöæW‡BÂu÷6æö6×C²òòu÷6æö6×C¢F†RÆ7B6÷’F†RVæ6öFRF‡&VB†26öÖÖ—GFVB†—G2u÷6æ÷b§7FF–2–çBu÷6æöâÒ#°¦–çB×FÅ÷6æööâ‡fö–B’°¢7FF–2–çBbÒÓ²–b‡bãÒ’&WGW&âc°¢6öç7B6†"¦RÒvWFVçb‚$DÕõ$U4TåEõ4ä"“°¢–çBâÒRòFö’†R’¢DÕôtÔUõ$U4TåEõ4äò"¢²òòF†R&öf–ÆRw2&W6VçE÷6æ†&–æröb"¢bÒââ²u÷6æöâÒâÂ"ò"¢ââ4äôÔ‚ò4äôÔ‚¢ã°¢–b‡b’ÔÄôr‚'&W6VçB6æ6†÷G2„DÕõ$U4TåEõ4äÒVB“¢V6‚&W6VçB—26÷–VB–çFò&–æröbVB†÷7BFW‡GW&W2æBF†RvÖRw2FW‡GW&R—2g&VRBöæ6R"ÂâÂu÷6æöâ“°¢&WGW&âc°§Ð§7FF–2fö–B6æö6öÖÖ—E÷v—B‡V–çCcE÷Bb’²–b‡b’&V5öæ÷FU÷v—B‚fu÷6æö6×BÂb“²Ð§7FF–2fö–B6æöG&–â‡fö–B’°¢òòF—7Æ’F–6¶WG2Ö’†fR&VVâG&÷VBGW&–ær7W7VæBÂ'WBF†V—"6÷–W27F–ÆÂ&VÆöærFòF†RÖWFÂBVWVRà¢V–çCcE÷BbÒu÷6æ÷c²–b‚b’&WGW&ã°¢6æö6öÖÖ—E÷v—B‡b“°¢–b‚¶u÷6æöWbv—EVçF–Å6–væÆVEfÇVS§bF–ÖV÷WDÕ3£Ò’ÔD”R‚'&W6VçB6æ6†÷G3¢uRF–Bæ÷BG&–âv—F†–â2"“°§Ð§V–çCcE÷B×FÅ÷6æö7V—&U÷v—B‡fö–B’°¢–b‚×FÅ÷6æööâ‚’ÇÂFöÖ–5öÆöB‚fu÷&W6VçEöF—66&B’’&WGW&â°¢–çB²Ò†–çB’†FöÖ–5öÆöB‚fu÷6æöæW‡B’R‡V–çCcE÷B–u÷6æöâ“°¢V–çCcE÷BF–6¶WBÒFöÖ–5öÆöB‚fu÷6æ¶µÒçF–6¶WB’ÂWbÒFöÖ–5öÆöB‚fu÷6æ¶µÒæWb’ÂsÒw5÷C‚“°¢F‡&VEö×WFW…öÆö6²‚fu÷ö×R“²–çB&Æö6¶VBÒF–6¶WBâu÷öFöæS²F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“°¢–b‡F–6¶WB’×FÅ÷&W6VçE÷v—B‡F–6¶WB“°¢–b†WbbbFöÖ–5öÆöB‚fu÷&W6VçEöF—66&B’bbu÷6æöWbç6–væÆVEfÇVRÂWb’°¢&Æö6¶VBÒ°¢òòöâF–ÖV÷WB&W6VçBfÆÇ2&6²Fò—G2÷&–v–æÂFW‡GW&S²F†R'W7’ÆÆö6F÷"—2æWfW"&W6WBà¢–b‚¶u÷6æöWbv—EVçF–Å6–væÆVEfÇVS¦WbF–ÖV÷WDÕ3£#Ò’$TEôôä4RƒÂ'&W6VçB6æ6†÷G3¢F†R&Wf–÷W26÷’–çFò&–ær6Æ÷BF–Bæ÷Bf–æ—6‚v—F†–â"2"“°¢Ð¢–b‡w5ööâ‚’’w5÷WB…u5õ4äÂsÂ&Æö6¶VB“°¢&WGW&âw5÷C‚’Òs°§Ð¢òò6÷’F†R&W6VçFVBFW‡GW&R–çFòF†RæW‡B&–ær6Æ÷C²&WGW&ç2F†R6Æ÷Bw26W&–ÂƒÒæò6æ6†÷C¢F†RvÖRw2FW‡GW&R—2&W6VçFVB2&Vf÷&R§7FF–2V–çCcE÷B6æ÷F¶R†6öç7BFW…÷B§B’°¢–çB²Ò†–çB’†FöÖ–5öÆöB‚fu÷6æöæW‡B’R‡V–çCcE÷B–u÷6æöâ“°¢F‡&VEö×WFW…öÆö6²‚fu÷ö×R“²–çB'W7’ÒFöÖ–5öÆöB‚fu÷6æ¶µÒçF–6¶WB’âu÷öFöæS²F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“°¢V–çCcE÷BWbÒFöÖ–5öÆöB‚fu÷6æ¶µÒæWb“°¢–b†'W7’ÇÂ†Wbbbu÷6æöWbç6–væÆVEfÇVRÂWb’’&WGW&â°¢òòF†RFW‡GW&RF†—26Æ÷B†VÆB&Vf÷&R—Bv2Æ7B&WÆ6VC¢WfW'’&W6VçBF†B&VB—B†B&VVâ6öÖÖ—GFVBv†VâF†R&WÆ6–ær6÷’v2‡F†R'W7’6†V6²F†Vâ’ÂæBF†RVWVP¢òò6–væÇ2u÷6æöWböæÇ’gFW"ÆÂv÷&²6öÖÖ—GFVB&Vf÷&RF†R6–væÂÂ6òF†÷6R&W6VçG2†fR'VâæB—B6âÆVfRF†R&W6–FVæ7’6W@¢–b†u÷6æ¶µÒæöÆB’²¶u÷6æ÷&W2&VÖ÷fTÆÆö6F–öã¦u÷6æ¶µÒæöÆEÓ²¶u÷6æ÷&W26öÖÖ—EÓ²u÷6æ¶µÒæöÆBÒæ–Ã²Ð¢–CÄÕDÅFW‡GW&Sâ7&2ÒFW…ö×FÂ‡B“²–b‚7&2’&WGW&â°¢òòF†R6÷’&VG2F†R6÷W&6R&Vf÷&R&W6VçEöæ÷r'Vç2Â6ò—BæVVG2F†R6ÖR6öÖÖ—B2&W6VçEöæ÷s¢v–æF÷rFW‡GW&R&W6VçFVB&Vf÷&Rç’7V&Ö—76–öâ†2W6VB—Bv2ÖFR§W7@¢òòæ÷ræB—2÷WG6–FRF†R6öÖÖ—GFVB&W6–FVæ7’6WBâE$tôâTU5B„’2&W6VçG2F‡&VRF–ÖW2&Vf÷&R—G2f—'7B7V&Ö—76–öã²öââr&òF†R6÷’fVÇFVB6–ÆVçFÇ’†—G26öÖÖæB'VffW ¢òò†2æòW'&÷"†æFÆW"’æBWfW'’ÆFW"6öÖÖæB'VffW"öbF†R&ö6W72v2–væ÷&VB„v—D‡V"—77VR3c#b’â6öÖÖ—Bv—F‚æ÷F†–æræWr6÷7G2æ÷F†–æp¢&W5ö6öÖÖ—B‚“°¢–b‚u÷6æöWb’°¢u÷6æöWbÒ¶uöFWbæWu6†&VDWfVçEÓ°¢ÕDÅ&W6–FVæ7•6WDFW67&—F÷"§&BÒ´ÕDÅ&W6–FVæ7•6WDFW67&—F÷"æWuÓ²&Bæ–æ—F–Ä66—G’Ò4äôÔƒ²å4W'&÷"¦RÒæ–Ã°¢u÷6æ÷&W2Ò¶uöFWbæWu&W6–FVæ7•6WEv—F„FW67&—F÷#§&BW'&÷#¢fUÓ²–b‚u÷6æ÷&W2’ÔD”R‚'&W6VçB6æ6†÷G3¢æò&W6–FVæ7’6WC¢W2"ÂRæÆö6Æ—¦VDFW67&—F–öâåUDc…7G&–ær“°¢¶u÷BFE&W6–FVæ7•6WC¦u÷6æ÷&W5Ó°¢f÷"†–çB’Ò²’Âu÷6æöã²’²²’°¢u÷6æöÆÆö5¶•ÒÒ¶uöFWbæWt6öÖÖæDÆÆö6F÷%Ó²u÷6æö6%¶•ÒÒ¶uöFWbæWt6öÖÖæD'VffW%Ó°¢–b‚u÷6æöÆÆö5¶•ÒÇÂu÷6æö6%¶•Ò’ÔD”R‚'&W6VçB6æ6†÷G3¢æò6öÖÖæBÆÆö6F÷"ò'VffW""“°¢Ð¢Ð¢–CÄÕDÅFW‡GW&SâG7BÒu÷6æ¶µÒçFWƒ°¢–b‚G7BÇÂG7Bçv–GF‚Ò7&2çv–GF‚ÇÂG7Bæ†V–v‡BÒ7&2æ†V–v‡BÇÂG7Bç—†VÄf÷&ÖBÒ7&2ç—†VÄf÷&ÖB’°¢ÕDÅFW‡GW&TFW67&—F÷"¦BÒ´ÕDÅFW‡GW&TFW67&—F÷"FW‡GW&S$DFW67&—F÷%v—F…—†VÄf÷&ÖC§7&2ç—†VÄf÷&ÖBv–GFƒ§7&2çv–GF‚†V–v‡C§7&2æ†V–v‡BÖ—ÖVC¤äõÓ°¢Bç7F÷&vTÖöFRÒÕDÅ7F÷&vTÖöFU&—fFS²BçW6vRÒÕDÅFW‡GW&UW6vU6†FW%&VC°¢òòF†RöÆBFW‡GW&R7F—2&W6–FVçB†æB&VfW&Væ6VB’VçF–ÂF†R6Æ÷Bw2æW‡BW6S¢F†R'W7’6†V6²&÷fRÖVç2—G2Æ7B&W6VçBv26öÖÖ—GFVBÂæ÷BF†BF†RuR†2'Vâ—BÀ¢òòæBFW‡GW&R&VÖ÷fVB†W&Rv26×ÆVB÷WG6–FRF†R&W6–FVæ7’6WB'’F—7Æ’6öÖÖæB'VffW"7F–ÆÂVWVV@¢u÷6æ¶µÒæöÆBÒG7C°¢G7BÒ¶uöFWbæWuFW‡GW&Uv—F„FW67&—F÷#¦EÓ²–b‚G7B’²$TEôôä4Rƒ"Â'&W6VçB6æ6†÷G3¢æòFW‡GW&R"“²&WGW&â²Ð¢¶u÷6æ÷&W2FDÆÆö6F–öã¦G7EÓ²¶u÷6æ÷&W26öÖÖ—EÓ°¢–b‚u÷6æ¶µÒç6W&–Â’²7FF–2ôFöÖ–2V–çCcE÷Bã²u÷6æ¶µÒç6W&–ÂÒƒVÆÂÃÂc’²FöÖ–5öfWF6…öFB‚fâÂ’²²Ð¢6W&–Å÷WB†u÷6æ¶µÒç6W&–ÂÂG7B“²u÷6æ¶µÒçFW‚ÒG7C°¢ÔÄôr‚'&W6VçB6æ6†÷G3¢&–ær6Æ÷BVBVÇW‚VÇRf÷&ÖBVÇR"Â²Â‡Vç6–væVBÆöær–G7Bçv–GF‚Â‡Vç6–væVBÆöær–G7Bæ†V–v‡BÂ‡Vç6–væVBÆöær–G7Bç—†VÄf÷&ÖB“°¢Ð¢¶u÷6æöÆÆö5¶µÒ&W6WEÓ°¢–CÄÕDÃD6öÖÖæD'VffW#â6"Òu÷6æö6%¶µÓ°¢¶6"&Vv–ä6öÖÖæD'VffW%v—F„ÆÆö6F÷#¦u÷6æöÆÆö5¶µÕÓ°¢–CÄÕDÃD6ö×WFT6öÖÖæDVæ6öFW#â6RÒ¶6"6ö×WFT6öÖÖæDVæ6öFW%Ó°¢¶6R&'&–W$gFW%VWVU7FvW3¤ÕDÅ7FvTÆÂ&Vf÷&U7FvW3¤ÕDÅ7FvT&Æ—Bf—6–&–Æ—G”÷F–öç3§&W6÷W&6U÷f—2‚•Ó²òògFW"F†Rg&ÖRw2v÷&²æBF†R6Æ÷Bw2Æ7B&Æ—BFòF†R67&VVà¢¶6R6÷”g&öÕFW‡GW&S§7&26÷W&6U6Æ–6S£6÷W&6TÆWfVÃ£6÷W&6T÷&–v–ã¤ÕDÄ÷&–v–äÖ¶RƒÂÂ’6÷W&6U6—¦S¤ÕDÅ6—¦TÖ¶R‡7&2çv–GF‚Â7&2æ†V–v‡BÂ’FõFW‡GW&S¦G7BFW7F–æF–öå6Æ–6S£FW7F–æF–öäÆWfVÃ£FW7F–æF–öä÷&–v–ã¤ÕDÄ÷&–v–äÖ¶RƒÂÂ•Ó°¢¶6R&'&–W$gFW%7FvW3¤ÕDÅ7FvT&Æ—B&Vf÷&UVWVU7FvW3¤ÕDÅ7FvTÆÂf—6–&–Æ—G”÷F–öç3§&W6÷W&6U÷f—2‚•Ó²òòF†RvÖRw2æW‡Bw&—FW2Fò—G2FW‡GW&R6öÖRgFW"F†R6÷¢¶6RVæDVæ6öF–æuÓ²¶6"VæD6öÖÖæD'VffW%Ó°¢V–çCcE÷BbÒ²¶u÷6æ÷c°¢–b‡&V5ööâ‚’bbu÷&%÷7F'FVB’²òò&Æö6²öbF†RVæ6öFRF‡&VBw2VWVRÂ&V†–æBF†Rg&ÖRw2v÷&²‡6VR&÷fR¢–b†öå÷7V%÷F‡&VB‚’’&V5÷V&Æ—6‚‚“²òòv†BF†R7V&Ö—GF–ærF‡&VB&V6÷&FVB÷WG6–FR7V&Ö—76–öâvöW2f—'7@¢&V5ö&Æ²¦"Ò&V5ö÷våö&Vv–â‚“°¢&V5÷B§"Ò&V5ö÷vå÷WB†"Â$õô4ôÔÔ•Eõ$TÂÂ"“²"Óæ³ÒÒ²"Óæ³ÒÒ&V5öö&¢†6"“°¢"Ò&V5ö÷vå÷WB†"Â$õõ4”täÂÂ“²"Óæ³ÒÒ&V5öö&¢†u÷6æöWb“²"Óæ³ÒÒc°¢"Ò&V5ö÷vå÷WB†"Â$õôäõDRÂ“²"Óæ³ÒÒ‡V–çCcE÷B’‡V–çGG%÷B’fu÷6æö6×C²"Óæ³ÒÒc°¢&V5ö÷vå÷W6‚†"“°¢ÒVÇ6R°¢F‡&VEö×WFW…öÆö6²‚fu÷&W5ö×R“²¶u÷B6öÖÖ—C¢f6"6÷VçC£Ó²¶u÷B6–væÄWfVçC¦u÷6æöWbfÇVS§eÓ²F‡&VEö×WFW…÷VæÆö6²‚fu÷&W5ö×R“°¢FöÖ–5÷7F÷&R‚fu÷6æö6×BÂb“°¢Ð¢u÷6æ¶µÒæWbÒc°¢&WGW&âu÷6æ¶µÒç6W&–Ã°§Ð§7FF–2V–çCcE÷B&W6VçEöVçVWVR†6öç7BFW…÷B§BÂF÷V&ÆRW&–öBÂ6öç7B–çB7&÷³EÒÂV–çCcE÷BWe÷fÂ’°¢FöÖ–5÷7F÷&R‚fFÕ÷&W6VçEöç2Âæ÷uöç2‚’“°¢–b†uö6Æ76–2ÇÂFöÖ–5öÆöB‚fu÷&W6VçEöF—66&B’’&WGW&â²òòF–6¶WBçVÖ&W"ÒæòVæF–ær&W6VçC²F†R7v6†–âFW‡GW&R7V—&RFöW2æ÷Bv—B†uö6Æ76–3¢æòÖWFÂBVWVRFò&W6VçBöâ¢–b‡B’²‡eö6†V6²‡BÓç6W&–ÂÂÂÂ'&W6VçB"“²–b†u÷‡eöç&W6VçBÂ#‚’u÷‡e÷&W6VçE¶u÷‡eöç&W6VçB²µÒÒBÓç6W&–Ã²Ð¢òòFW‡GW&Rv†÷6R6öçFVçB6öÖW2g&öÒ—G27F÷&vR—2WÆöFVBv†Vâ—B—2f—'7B&W6öÇfVB†æBv–âv†Vâ—G2vW26†ævR’Â–ââWÆöB&F6‚öæÇ’F†R7V&Ö—BF‡&VBÖ’÷Vâà¢òò&7F–öâf–ÆÇ2&÷F‚v–æF÷rFW‡GW&W2v—F‚w&—FUFW†VÇ2B&ö÷B†Ö–â³ƒ“3sB’æB&W6VçG2öæR&Vf÷&RG&v–ær–çFò—C¢&W6öÇfVBöâF†R&W6VçBF‡&VBÂF†RWÆöB&÷'FVBF†R†÷7Bà¢òò&W6öÇfR—B†W&RÂv†W&RF†RvÖRVWVW2F†R&W6VçBÂæB6öÖÖ—BF†R&F6‚Â6òF†RF—7Æ’6öÖÖæB'VffW"6öÖW2gFW"F†RWÆöBöâF†RVWVRæBF†R&W6VçBF‡&VBf–æG2—BWÆöFV@¢–b‡BbbWe÷fÂbböå÷7V%÷F‡&VB‚’bbFW…ög&öÕ÷7F÷&vR‡B’’²FW…ö×FÂ‡B“²WöfÇW6‚‚“²Ð¢7FF–2F‡&VEööæ6U÷Böæ6RÒD…$TEôôä4Uô”ä•C²F‡&VEööæ6R‚föæ6RÂ&W6VçE÷7F'B“°¢Æ–æµ÷7–æ2‚“°¢²òòFW7B7v—F6‚DÕõDU5EôÄ”äµõ5DÄÃ×7F'B6V6öæG2ÆGW&F–öâ6V6öæG3¢6÷VçF–ærg&öÒF†Rf—'7B&W6VçBÂW6RF†RF—7Æ’Æ–æ²BF†Bö–çB†–Ö—FF–ærF†RF—7Æ’Æ–æ²æ÷B6ÆÆ–ær&6²v†–ÆR–âF†Rf÷&Vw&÷VæB’æB&VÆV6R—Bv†VâF†RGW&F–öâ—2Wâf÷"fW&–g––ærF†RfÆÆ&6³²æ÷B6WB–âF†R&öGV7@¢7FF–2F÷V&ÆRCÂBÒÓÂGW#²7FF–2–çB7C²F÷V&ÆRæ÷rÒ47W'&VçDÖVF–F–ÖR‚“°¢–b†BÂ’²BÒ²6öç7B6†"¦RÒvWFVçb‚$DÕõDU5EôÄ”äµõ5DÄÂ"“²–b†Rbb766æb†RÂ"VÆbÂVÆb"ÂfBÂfGW"’ÓÒ"bbBâbbGW"â’²CÒæ÷s²7BÒ²ÒVÇ6RBÒ²Ð¢–b‡7BÓÒbbÆ–æµööâ‚’bbæ÷rÒCãÒB’²7BÒ#²×FÅöÆ–æµ÷W6Rƒ“²ÔÄôr‚$DÕõDU5EôÄ”äµõ5DÄÃ¢W6–ærF†RF—7Æ’Æ–æ²f÷"Rãb2†g&ÖRVÆÇR’"ÂGW"Â‡Vç6–væVBÆöærÆöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR’“²Ð¢VÇ6R–b‡7BÓÒ"bbæ÷rÒCãÒB²GW"’²7BÒ3²×FÅöÆ–æµ÷W6Rƒ“²ÔÄôr‚$DÕõDU5EôÄ”äµõ5DÄÃ¢&W7VÖ–ærF†RF—7Æ’Æ–æ²†g&ÖRVÆÇR’"Â‡Vç6–væVBÆöærÆöær–FöÖ–5öÆöB‚fuö7W%ög&ÖR’“²Ð¢Ð¢–b‡w5ööâ‚’’²7FF–2V–çCcE÷Bc²V–çCcE÷BbÒFöÖ–5öÆöB‚fuö7W%ög&ÖR“²–b‚c’cÒc²VÇ6R–b†bãÒc²3’²w5÷&W÷'B†bÂbÒc“²cÒc²ÒÐ¢FW…÷B6æ²V–çCcE÷B72Ò²–çB6Æ÷BÒ°¢–b‡BbbWe÷fÂbb×FÅ÷6æööâ‚’’°¢6Æ÷BÒ†–çB’†FöÖ–5öÆöB‚fu÷6æöæW‡B’R‡V–çCcE÷B–u÷6æöâ“°¢–b‚‡72Ò6æ÷F¶R‡B’’’²ÖV×6WB‚g6æÂÂ6—¦Vöb6æ“²6ææÖv–2ÒÔt”2‚uBrÂtRrÂu‚rÂ“²6æç6W&–ÂÒ73²BÒg6æ²Ð¢Ð¢F‡&VEö×WFW…öÆö6²‚fu÷ö×R“°¢v†–ÆR†u÷ö–âÒu÷öFöæRãÒ’F‡&VEö6öæE÷v—B‚fu÷ö7bÂfu÷ö×R“²òòv—BöæÇ’v†VâF†RVWVR—2gVÆÃ²æ÷&ÖÆÇ’7V—&UFW‡GW&R&Æö6·2f—'7@¢ö—FVÒ¦—BÒfu÷¶u÷ö–âRÓ²¦—BÒ‡ö—FVÒ—²§BÂW&–öBÂFöÖ–5öÆöB‚fuö7W%ög&ÖR’Â³ÒÂ47W'&VçDÖVF–F–ÖR‚’ÂWe÷fÂÂ72òu÷6æ·6Æ÷EÒæWb¢Ó°¢–b†7&÷’ÖVÖ7’†—BÓæ7&÷Â7&÷Â6—¦Vöb—BÓæ7&÷“²òòF†R7W'&VçBg&ÖRw27FFR—2VçVWVVBv—F‚F†RF–6¶WBçVÖ&W"Â&F†W"F†â&VF–ærF†R×WF&ÆRv–æF÷rÆFRöâF†R&W6VçBF‡&V@¢V–çCcE÷BF²Ò²¶u÷ö–ã°¢–b‡72’²u÷6æ·6Æ÷EÒçF–6¶WBÒF³²FöÖ–5öfWF6…öFB‚fu÷6æöæW‡BÂ“²Ð¢F‡&VEö6öæEö'&öF67B‚fu÷ö7b“°¢F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“°¢&WGW&â72ò¢F³²òò6æ6†÷C¢F†RvÖRw2FW‡GW&R—2g&VRæ÷r†—G2ÆFW"G&w26öÖRgFW"F†R6÷’öâF†RVWVR§Ð¢òòFòæ÷Bv—BGW&–ærF—66&C¢F†Rg&ÖRVçVWVVBöæÇ’gFW"F†RfÆrv26WB‡F†R&W6VçBF‡&VB†B§W7B76VBF†RfÆr6†V6²’†2æö&öG’FòF¶R—BÂæBv—F–ærf÷"—Bv÷VÆB7FÆÂv–ã²—B—2ÆVgBf÷"F†RföÆÆ÷v–ær×FÅöG&–âFò6ÆV ¢òòFòæ÷Bv—B–æFVf–æ—FVÇ’–âF—7Æ’ÖÆ–æ²ÖöFS¢6VRF†Ræ÷FRB5DÄÅôå2âöæÇ’g&ÖW2&æ÷B–WBF¶Vâ'’F†RF—7Æ’Æ–æ²"&RG&÷VC²F†RöæRÇ&VG’öâ—G2v’FòF†R67&VVâ—2v—FVBf÷"VçF–Â†æFVBFòF†RuR‡6ÖRG&÷–ær2v†VâVçFW&–ærF†R&6¶w&÷VæB§fö–B×FÅ÷&W6VçE÷v—B‡V–çCcE÷BF²’°¢F‡&VEö×WFW…öÆö6²‚fu÷ö×R“°¢v†–ÆR†u÷öFöæRÂF²bbFöÖ–5öÆöB‚fu÷&W6VçEöF—66&B’’°¢–b‚uöÆ–æµö÷vç5öÆ–W"’²F‡&VEö6öæE÷v—B‚fu÷ö7bÂfu÷ö×R“²6öçF–çVS²Ð¢–b†u÷÷&VBãÒF²’²F‡&VEö6öæE÷v—B‚fu÷ö7bÂfu÷ö×R“²6öçF–çVS²ÒòòF†—2öæR†2Ç&VG’&VVâF¶Vâ'’F†RF—7Æ’Æ–æ²æB—2&V–ær&W6VçFVC¢v—Bf÷"—BFò&R†æFVBFòF†RuR‡F†RuR6ö×ÆWF–öâ6ÆÆ&6²v–ÆÂv¶RW2¢7G'V7BF–ÖW7V2G3²6Æö6µövWGF–ÖR„4Äô4µõ$TÅD”ÔRÂgG2“²G2çGeöç6V2³Ò†Æöær•5DÄÅôå3²–b‡G2çGeöç6V2ãÒÂ’²G2çGe÷6V2²³²G2çGeöç6V2ÓÒÃ²Ð¢òòÇ&VG’FWFW&Ö–æVBFò&R–â7FÆÂ‡v†VâF†RF—7Æ’Æ–æ²6öÖW2&6²F†R6ÆÆ&6²f—'7B6ÆV'2u÷7FÆÅ÷CÂF†Vâ&V6÷&G2F†RF—7Æ’ÖÆ–æ²F–6²F–ÖR“¢æòÆöævW"v—B×2–âf–âWfW'’g&ÖRÂG&÷F—&V7FÇ’ÂæBF†RvÖR&ö6VVG2'’—G2÷vâF—7Æ’ÖÆ–æ²v—BF–ÖV÷WB†&÷WBGvòF–6·2¢–b‚u÷7FÆÅ÷Cbb‡F‡&VEö6öæE÷F–ÖVGv—B‚fu÷ö7bÂfu÷ö×RÂgG2’ÒUD”ÔTDõUBÇÂu÷öFöæRãÒF²’’6öçF–çVS°¢V–çCcE÷Bæ÷rÒæ÷uöç2‚’ÂÆ7BÒõöFöÖ–5öÆöEöâ‚fuöÆ–æµ÷F–6µöç2ÂõôDôÔ”5õ$TÄ„TB“°¢–b†u÷÷&VBãÒF²’6öçF–çVS°¢–b†æ÷rÒÆ7BÂ5DÄÅôå2’²F‡&VEö6öæE÷F–ÖVGv—B‚fu÷ö7bÂfu÷ö×RÂgG2“²6öçF–çVS²ÒòòF†RF—7Æ’Æ–æ²—27F–ÆÂ'Vææ–ærÂ—B—2§W7Bæ÷BF†—2öæRw2GW&â–WC¢¶VWv—F–ær†&÷VæFVBÂæò'W7’7–ææ–ær¢V–çCcE÷BâÒF²Òu÷÷&VC°¢f÷"‡V–çCcE÷B²Òu÷÷&VB²²²ÃÒF³²²²²’u÷&W5¶²R$”äuÒÒ‡&W5öVçB—¶²ÂÓÓ°¢uöÆ–æµöG&÷VB³Òã²u÷÷&VBÒF³°¢v†–ÆR†u÷ö–æfÆ–v‡B’F‡&VEö6öæE÷v—B‚fu÷ö7bÂfu÷ö×R“°¢–b†u÷öFöæRÂF²’u÷öFöæRÒF³°¢F‡&VEö6öæEö'&öF67B‚fu÷ö7b“°¢–b‚u÷7FÆÅ÷C’²u÷7FÆÅ÷CÒÆ7BòÆ7B¢æ÷s²u÷7FÆÅöG&÷Ò²u÷7FÆÅ÷&VBÒ²u÷7FÆÅöÆövvVBÒ²u÷7FÆÅöâ²³²Ð¢u÷7FÆÅöG&÷³Òã°¢–çB7F—fRÒFÕöfö7W5öæ÷r‚’ÓÒ²V–çCcE÷B6–æ6RÒFÕöfö7W5ö–å÷6–æ6R‚’Âg&öÒÒu÷7FÆÅ÷Câ6–æ6Ròu÷7FÆÅ÷C¢6–æ6S°¢V–çCcE÷BVÆ6VBÒæ÷rãÒg&öÒòæ÷rÒg&öÒ¢²òòfö7W26â&WGW&âgFW"F†RV&Æ–W"æ÷r6×ÆRÂ–æ6ÇVF–ærv†–ÆRv—F–ærf÷"â–âÖfÆ–v‡B&W6Vç@¢–b†7F—fRbbu÷7FÆÅöÆövvVBbbVÆ6VBãÒ5DÄÅôå2’²u÷7FÆÅöÆövvVBÒ°¢ÔÄôr‚&F—7Æ’Æ–æ²6–ÆVçBf÷"Rãb×2†7F—fR“¢G&÷–ærVÆÇRVWVVBg&ÖW2æ÷B–WBF—7Æ–VBÂF†RvÖR6öçF–çVW2‡7FÆÂ2VÆÇS²&6²Fòæ÷&ÖÂv†VâF†RÆ–æ²&WGW&ç2’"Â†æ÷rÒg&öÒ’òSbÂ‡Vç6–væVBÆöærÆöær–âÂ‡Vç6–væVBÆöærÆöær–u÷7FÆÅöâ“²Ð¢–b†7F—fRbbu÷&W67VUöÖöFRbbuöFÕ÷&W6VçEöÆ–æ²bbVÆ6VBãÒÄ”äµõ$U45TUôå2’°¢u÷&W67VUöÖöFRÒuöFÕ÷&W6VçEöÆ–æ³²u÷&W67VUöâ²³²u÷&W67VUö&6µöBÒæ÷r²u÷&W67VUö&6µöç3°¢ÔÄôr‚&F—7Æ’Æ–æ²6–ÆVçBf÷"Rãb×2v†–ÆR7F—fS¢F†R&W6VçBF‡&VBF¶W2G&v&ÆW2—G6VÆb‡&W67VR2VÆÇR’ÂF†RÆ–æ²—2G&–VBv–â–âRãb2"Â†æ÷rÒg&öÒ’òSbÂ‡Vç6–væVBÆöærÆöær–u÷&W67VUöâÂu÷&W67VUö&6µöç2òS’“°¢u÷&W67VUö&6µöç2Òu÷&W67VUö&6µöç2Â3VÆÂòu÷&W67VUö&6µöç2¢"¢cVÆÃ°¢Ð¢–b†7F—fRbbu÷7FÆÅ÷&VBbbVÆ6VBãÒ5DÄÅõ$TEôå2’²u÷7FÆÅ÷&VBÒ²ÔÄôr‚%µ$TEÒF—7Æ’Æ–æ²6–ÆVçBf÷"Rãb2v†–ÆR7F—fS²&W6VçF–ær&VÆ–W2VçF—&VÇ’öâfÆÆ&6²g&ÖRG&÷2ÂF†R–7GW&R—2g&÷¦Vâ"ÂVÆ6VBòS’“²Ð¢Ð¢F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“°§Ð¢òòv—BVçF–ÂÆÂVWVVB&W6VçG2†fR&VVâ&ö6W76VB‡W6VB'’×FÅöf–æ—6‚ÒçfåVWVTf–æ—6‚WF2â“¢vöW2F‡&÷Vv‚×FÅ÷&W6VçE÷v—BÂ6òF†W&R—27F–ÆÂfÆÆ&6²v†VâF†RF—7Æ’Æ–æ²7F÷2‡&VÂFWf–6S¢F†RvÖR6ÆÆVBVWVTf–æ—6‚öæ6Rv†–ÆRF†RF—7Æ’Æ–æ²v27F÷VBÀ¢òòæBF†—2W6VBFòv—B†W&Rv—F†÷WBF–ÖV÷WC²æö&öG’Föö²F†R2VWVVBg&ÖW2æBF†RvÖR7FÆÆVC²F†÷6RGvò'Vç2öâÖ2F–Bæ÷B†—BF†—26ÆÂ§fö–B×FÅ÷&W6VçEöfÇW6‚‡fö–B’²F‡&VEö×WFW…öÆö6²‚fu÷ö×R“²V–çCcE÷B‚Òu÷ö–ã²F‡&VEö×WFW…÷VæÆö6²‚fu÷ö×R“²×FÅ÷&W6VçE÷v—B†‚“²Ð§fö–B×FÅ÷7FG2‡V–çCcE÷B§76W2ÂV–çCcE÷B¦G&w2’²§76W2Òu÷7FE÷73²¦G&w2Òu÷7FEöG&s²7÷&W÷'B‚“²Ð§fö–B×FÅö&w5÷7FG2‡V–çCcE÷B¦Væ2ÂV–çCcE÷B§&WW6R’²¦Væ2Òu÷7FEö&w5öVæ3²§&WW6RÒu÷7FEö&w5÷&WW6S²Ð§fö–B×FÅ÷f'Ve÷7FG2‡V–çCcE÷B§&W6öÇfRÂV–çCcE÷B§&WW6RÂV–çCcE÷B§6WBÂV–çCcE÷B§6¶—ÂV–çCcE÷B¦6÷–W2ÂV–çCcE÷B¦6÷•ö'—FW2’°¢§&W6öÇfRÒu÷f'Ve÷&W6öÇfS²§&WW6RÒu÷f'Ve÷&WW6S²§6WBÒu÷f'Ve÷6WC²§6¶—Òu÷f'Ve÷6¶—²¦6÷–W2Òu÷f'Veö6÷–W3²¦6÷•ö'—FW2Òu÷f'Veö6÷•ö'—FW3°§Ð§fö–B×FÅ÷W6U÷7FG2‡V–çCcE÷B¦6ÆÂÂV–çCcE÷B§6¶—’²¦6ÆÂÒ²§6¶—Ò²Ð§V–çCcE÷B×FÅöG5ööÖ—B‡fö–B’²&WGW&âu÷7FEöG5ööÖ—C²Ð§fö–B×FÅö'r‡V–çCcE÷B¦ÆöBÂV–çCcE÷B§7F÷&RÂV–çCcE÷B¦gVÆÂ’²¦ÆöBÒuö'uöÆöC²§7F÷&RÒuö'u÷7F÷&S²¦gVÆÂÒu÷75ögVÆÅ÷'s²Ð§V–çCcE÷B×FÅ÷VF5÷6fVEö'—FW2‡fö–B’²&WGW&âu÷VF5÷6fVEö'—FW3²Ð¦–çB×FÅö†5÷v–æF÷r‡fö–B’²&WGW&âg&öçEöÖWFÅöÆ–W"‚’ÒåTÄÃ²Ð¦–çB×FÅöÆ–æµö7F—fR‡fö–B’²&WGW&âÆ–æµööâ‚“²Ð¦–çB×FÅöÆ–æµ÷7FÆÆVB‡fö–B’²&WGW&âõöFöÖ–5öÆöEöâ‚fu÷7FÆÅ÷CÂõôDôÔ”5õ$TÄ„TB’Ò²ÒòòF†RF—7Æ’Æ–æ²†2vöæR6–ÆVçBæB×FÅ÷&W6VçE÷v—BG&÷2g&ÖW2‡6VR5DÄÅôå2’VçF–Â—B6ÆÇ2&6°§fö–B×FÅöÆ–æµ÷W6R†–çBöâ’²4ÖWFÄF—7Æ”Æ–æ²¦ÂÒuöÆ–æ³²–b†Â’ÂçW6VBÒöâÒ²Òòòf÷"FW7F–ær„DÕõDU5Eô$r“¢–Ö—FFW2F†R7—7FVÒ7F÷–ærF†RF—7Æ’ÖÆ–æ²6ÆÆ&6²–âF†R&6¶w&÷Væ@¦–çB×FÅöÆ–æµ÷v—E÷F–6²†–çBg7–æ72ÂV–çCcE÷BF–ÖV÷WEöç2’°¢–b‚×FÅöÆ–æµö7F—fR‚’’&WGW&â°¢òòv—Bf÷"F†RF—7Æ’Æ–æ²öæÇ’v†Vâ'Vææ–ærf7FW"F†âF†RF—7Æ“¢–bF—7Æ’ÖÆ–æ²F–6²†2Ç&VG’76VB6–æ6RF†R&Wf–÷W2g&ÖR7F'FVBv÷&²‡F†—2g&ÖR—2Ç&VG’&V†–æB’Â7F'B–ÖÖVF–FVÇ’–ç7FVBöbv—F–ær–âf–âf÷"F†RæW‡BF—7Æ’ÖÆ–æ²F–6²ÒÐ¢òò÷F†W'v—6Rv†VâF†Rg&ÖRF–ÖR—26Æ–v‡FÇ’Ö÷&RF†âöæRF—7Æ’ÖÆ–æ²W&–öB—Bv÷VÆB&Ö—72F†RF–6²Óâv—Bæ÷F†W"v†öÆRF–6²"ÂÆö6¶–ærF†Rg&ÖR&FR7G&–v‡BFò†Æb†ÖV7W&VBöâÖ2VæFW"v–æF÷vVBÆöC¢'Vç2öbG&÷VBg&ÖW2¢òò†÷rÖç’F—7Æ’ÖÆ–æ²F–6·2öæRg&ÖRF¶W3¢F†RvÖRw2&W6VçB–çFW'fÂ‡g7–æ72Âc‡¢’F–ÖW2v—F‚c‡¢F—7Æ’Æ–æ²ÂF–ÖW2"v—F‚#‡¢öæRÂ†ÇfVBæB&÷VæFVBWB3‡¢à¢òòg&ÖR&W6VçFVBv—F‚–çFW'fÂâ&Æö6·2F†RvÖRf÷"âg7–æ72öâF†R7v—F6‚Â2F†RF—7Æ’6ÆÆ&6²†öÆG2—BâF–6·3¢vÖW2F†B66ÆRF†V—"F–ÖR7FW'’F†R–çFW'fÂ„&÷Erw2de#¢–çFW'fÂò"¢òò&âBF÷V&ÆR7VVBv†–ÆRF†W’&öGV6VB–çFW'fÂÓ"g&ÖW2WfW'’c‡¢F–6²†•†öæRÆös¢Æ–æ²7&–çF–ærv—F‚"ÂF†R–çFW'fÂfÆ–ær(iB"¢–b‡g7–æ72Â’g7–æ72Ò°¢&WGW&â×FÅöÆ–æµ÷v—E÷F–6·2†uöÆ–æµöÖöFRÓÒ2ò‡g7–æ72²’ò"¢g7–æ72¢†uöÆ–æµöÖöFRÓÒ"ò"¢’ÂF–ÖV÷WEöç2“°§Ð¢òòF†RF—7Æ’Æ–æ²w2&FS¢#‡¢„DÕõ$U4TåEôÄ”ä³Ó"’ÂcÂ÷"3–â÷vW"×6f–ærÖöFP¦–çB×FÅöÆ–æµ÷&FR‡fö–B’²&WGW&âuöÆ–æµöÖöFRÓÒ2ò3¢uöÆ–æµöÖöFRÓÒ"ò#¢c²Ð¢òòF†R6ÖRv—B6÷VçFVB–âF†RF—7Æ’Æ–æ²w2÷vâF–6·3¢öæRW"g&ÖRf÷"g&ÖR&FR2†–v‚2F†RÆ–æ²w2‡&W6VçEö–çFW'fÂF—7Æ’BDÕõ$U4TåEô…£Ó#öâ#‡¢Æ–æ²¦–çB×FÅöÆ–æµ÷v—E÷F–6·2†–çBF–6·2ÂV–çCcE÷BF–ÖV÷WEöç2’°¢–b‚×FÅöÆ–æµö7F—fR‚’’&WGW&â°¢7FF–2V–çCcE÷B7F'FVE÷F–6³²òòF—7Æ’ÖÆ–æ²F–6²çVÖ&W"v†VâF†R&Wf–÷W2g&ÖR7F'FVBv÷&²†öæÇ’F†RvÖRw2&W6VçBF‡&VB6ÆÇ2F†—2gVæ7F–öâ¢6öç7BV–çCcE÷B7FWÒF–6·2Âò¢‡V–çCcE÷B—F–6·3°¢F‡&VEö×WFW…öÆö6²‚fuöÆ–æµö×R“°¢–b†uöÆ–æµ÷F–6µöâãÒ7F'FVE÷F–6²²7FW’²7F'FVE÷F–6²ÒuöÆ–æµ÷F–6µöã²F‡&VEö×WFW…÷VæÆö6²‚fuöÆ–æµö×R“²&WGW&â²Ð¢7G'V7BF–ÖW7V2G3²6Æö6µövWGF–ÖR„4Äô4µõ$TÅD”ÔRÂgG2“°¢G2çGe÷6V2³ÒF–ÖV÷WEöç2òVÆÃ²G2çGeöç6V2³ÒF–ÖV÷WEöç2RVÆÃ°¢–b‡G2çGeöç6V2ãÒÂ’²G2çGe÷6V2²³²G2çGeöç6V2ÓÒÃ²Ð¢v†–ÆR†uöÆ–æµ÷F–6µöâÂ7F'FVE÷F–6²²7FW’–b‡F‡&VEö6öæE÷F–ÖVGv—B‚fuöÆ–æµö7bÂfuöÆ–æµö×RÂgG2’ÓÒUD”ÔTDõUB’²F‡&VEö×WFW…÷VæÆö6²‚fuöÆ–æµö×R“²&WGW&â²Ð¢7F'FVE÷F–6²ÒuöÆ–æµ÷F–6µöã°¢F‡&VEö×WFW…÷VæÆö6²‚fuöÆ–æµö×R“²&WGW&â°§Ð ¢òòF†RF—7Æ’w2g7–æ2f÷"f“£¤vWDF—7Æ•g7–æ4WfVçB††ÆU÷6F³bæ2“¢&WGW&ç2BF†Rf—'7BF—7Æ’ÖÆ–æ²F–6²öbF†RæW‡Bc‡¢g7–æ2gFW"§6VVâ‡F†RF–6²6÷VçB—BÆ7B&WGW&æVBÂWFFVB’À¢òòv†VâF†RÆ–æ²FöW2æ÷B'VâÂ'Vç2B3‡¢‡÷vW"6f–æs¢F†Rc‡¢WfVçB6ææ÷B6öÖRg&öÒ—B’÷"v—fW2æòF–6²v—F†–âF†RF–ÖV÷WC²F†R6ÆÆW"F†Vâ¶VW2—G2÷vâ6Æö6°¦–çB×FÅöÆ–æµ÷v—E÷g7–æ2‡V–çCcE÷B§6VVâÂV–çCcE÷BF–ÖV÷WEöç2’°¢–b‚×FÅöÆ–æµö7F—fR‚’ÇÂuöÆ–æµöÖöFRÓÒ2’&WGW&â°¢6öç7BV–çCcE÷B7FWÒuöÆ–æµöÖöFRÓÒ"ò"¢°¢F‡&VEö×WFW…öÆö6²‚fuöÆ–æµö×R“°¢–b‚§6VVââuöÆ–æµ÷F–6µöâÇÂuöÆ–æµ÷F–6µöââ§6VVâ²B¢7FW’§6VVâÒuöÆ–æµ÷F–6µöã²òòf—'7B6ÆÂÂ÷"f"&V†–æB†7FÆÂ“¢g&öÒF†R7W'&VçBF–6²öâÂæò'W'7@¢7G'V7BF–ÖW7V2G3²6Æö6µövWGF–ÖR„4Äô4µõ$TÅD”ÔRÂgG2“°¢G2çGe÷6V2³ÒF–ÖV÷WEöç2òVÆÃ²G2çGeöç6V2³ÒF–ÖV÷WEöç2RVÆÃ°¢–b‡G2çGeöç6V2ãÒÂ’²G2çGe÷6V2²³²G2çGeöç6V2ÓÒÃ²Ð¢v†–ÆR†uöÆ–æµ÷F–6µöâÂ§6VVâ²7FW’–b‡F‡&VEö6öæE÷F–ÖVGv—B‚fuöÆ–æµö7bÂfuöÆ–æµö×RÂgG2’ÓÒUD”ÔTDõUB’²F‡&VEö×WFW…÷VæÆö6²‚fuöÆ–æµö×R“²&WGW&â²Ð¢§6VVâ³Ò7FW°¢F‡&VEö×WFW…÷VæÆö6²‚fuöÆ–æµö×R“²&WGW&â°§Ð §7FF–2–CÄ4ÖWFÄG&v&ÆSâu÷&VfWF6†VEöG&v&ÆS²òòW6VBf÷"6ö×÷6—F–ær7G&–v‡BöçFòF†RG&v&ÆP¦–CÄ4ÖWFÄG&v&ÆSâ×FÅ÷&W6VçE÷F¶U÷&VfWF6†VB‡fö–B’²&WGW&âu÷&VfWF6†VEöG&v&ÆS²Ð ¢6–fFVbådåõ%Eõ$ätUõDU5@¢6–æ6ÇVFR''E÷&ævUö6†V6²æ–æ2 ¢6VæF–` ¢òò6ÆÆVB'’F†RvF6†För†æò&öw&W72“¢w&—FW2F†R7FFRöbÆÂ&Æö6¶–ærv—Bö–çG2öâF†Rw&†–726–FRFòF†RÆörÒÒv†ò—2v—F–ærf÷"v†öÒ—2f—6–&ÆRBvÆæ6R†gFW"7v—F6†–ærFòF†RÖWFÂB7V&Ö—BÖöFVÂF†W&Rv2öæ6R66Röb'&âFò“g&ÖW2F†Vâ7F÷VBÂæòf–ÇW&RÆ–æR"§fö–B×FÅ÷v—E÷7FFUöGV×‡fö–B’°¢ÔÄôr‚'v—Bö–çG3¢Ö–âfVæ6R7V&Ö—GFVBVÆÇRò6ö×ÆWFVBVÆÇS²Ö–âÆÆö6F÷"fVæ6W2²VÆÇRVÆÇRVÆÇUÓ²WÆöG2VæF–ærVBÂÆÆö6F÷'2'W7’²VBVBVEÒÂ6WWRÂVæfÇW6†VB&F6‚VC²&W6VçBVWVR–âVÆÇRòFöæRVÆÇS²Ö–â6öÖÖæB'VffW"Væ6öF–ærVBÂ7V&Ö—BF‡&VB6WBVB"À¢‡Vç6–væVBÆöærÆöær–uöfVæ6RÂ‡Vç6–væVBÆöærÆöær–uöFöæRÂ‡Vç6–væVBÆöærÆöær–uöÆÆö5öfVæ6U³ÒÂ‡Vç6–væVBÆöærÆöær–uöÆÆö5öfVæ6U³ÒÂ‡Vç6–væVBÆöærÆöær–uöÆÆö5öfVæ6U³%ÒÀ¢FöÖ–5öÆöB‚fu÷W÷VæF–ær’ÂFöÖ–5öÆöB‚fu÷WöÆÆö5ö'W7•³Ò’ÂFöÖ–5öÆöB‚fu÷WöÆÆö5ö'W7•³Ò’ÂFöÖ–5öÆöB‚fu÷WöÆÆö5ö'W7•³%Ò’Âu÷W÷6WÂu÷Wö6"Òæ–ÂÀ¢‡Vç6–væVBÆöærÆöær–u÷ö–âÂ‡Vç6–væVBÆöærÆöær–u÷öFöæRÂuö6"Òæ–ÂÂFöÖ–5öÆöB‚fu÷7V%÷F‡&VE÷6WB’“°§Ð§fö–B×FÅöÆÆö5÷7FG2‡V–çCcE÷B§v—G2ÂV–çCcE÷B¦w&÷vâÂ–çB¦â’²§v—G2ÒuöÆÆö5÷v—G3²¦w&÷vâÒuöÆÆö5öw&÷vã²¦âÒuöÆÆö5öã²ÒòòÖ–â7V&Ö—BÆÆö6F÷"ööÃ¢†÷rÖç’F–ÖW2—Bv—FVBÂ†÷rÖç’F–ÖW2—Bw&WrÂ†÷rÖç’W†—7Bæ÷p¢6–æ6ÇVFR&çfåö6GW&Ræ–æ2"òòDÕôådåô4EU$R†öæRg&ÖRf÷"FÒövg‚÷FW7G2öçfå÷&WÆ’¢7&vÖ6ÆærGG&–'WFR÷ 