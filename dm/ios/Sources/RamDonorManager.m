// Host side of the optional MeloNX-style shared-memory donor.
// This is an experimental per-process footprint workaround; the system still
// accounts for the physical pages globally and can reclaim/terminate processes.
#import "RamDonorManager.h"
#import "RamDonorProtocol.h"

#import <mach/mach.h>
#import <xpc/xpc.h>
#import <pthread.h>
#import <signal.h>
#import <sys/mman.h>
#import <unistd.h>
#import <string.h>
#import <stdlib.h>
#import <stdio.h>
#import <stdarg.h>
#import <errno.h>
#import <dispatch/dispatch.h>
#import <mach/mach_error.h>

extern kern_return_t mach_vm_remap(vm_map_t target, mach_vm_address_t *address,
                                   mach_vm_size_t size, mach_vm_offset_t mask,
                                   int flags, vm_map_t source,
                                   mach_vm_address_t sourceAddress,
                                   boolean_t copy, vm_prot_t *curProtection,
                                   vm_prot_t *maxProtection,
                                   vm_inherit_t inheritance);

#ifndef VM_FLAGS_OVERWRITE
#define VM_FLAGS_OVERWRITE 0x4000
#endif

// NSExtension's request launcher and process controls are private iOS APIs.
// The donor is opt-in and failures always leave the original file mapping intact.
@interface NSExtension : NSObject
@property (nonatomic, copy) NSArray<NSString *> *preferredLanguages;
+ (instancetype)extensionWithIdentifier:(NSString *)identifier error:(NSError **)error;
- (void)beginExtensionRequestWithInputItems:(NSArray *)items
                                 completion:(void (^)(NSUUID *requestIdentifier))completion;
- (int)pidForRequestIdentifier:(NSUUID *)requestIdentifier;
- (void)_kill:(int)signal;
@end

typedef struct {
    uintptr_t base;
    size_t size;
    xpc_object_t shmem;
    uint8_t *livePages;
    size_t pageCount;
    size_t liveCount;
} DMRamMappedChunk;

static DMRamMappedChunk gMapped[DM_RAM_DONOR_MAX_CHUNKS];
static size_t gMappedCount;
static pthread_mutex_t gLock = PTHREAD_MUTEX_INITIALIZER;
static BOOL gAttempted;
static xpc_connection_t gListener;
static xpc_connection_t gDonorConnection;
static dispatch_semaphore_t gConnected;
static NSExtension *gExtension;
static NSUUID *gRequestIdentifier;

static void DMRamLog(const char *format, ...) {
    va_list args;
    va_start(args, format);
    fputs("[mem] RAM donor: ", stderr);
    vfprintf(stderr, format, args);
    fputc('\n', stderr);
    va_end(args);
}

static void DMRamDonorAccept(xpc_connection_t connection) {
    pthread_mutex_lock(&gLock);
    if (!gDonorConnection) gDonorConnection = xpc_retain(connection);
    pthread_mutex_unlock(&gLock);

    xpc_connection_set_event_handler(connection, ^(xpc_object_t event) {
        if (xpc_get_type(event) == XPC_TYPE_ERROR) {
            DMRamLog("helper connection ended");
            return;
        }
        if (xpc_get_type(event) == XPC_TYPE_DICTIONARY) {
            const char *op = xpc_dictionary_get_string(event, DM_RAM_DONOR_KEY_OP);
            if (op && !strcmp(op, DM_RAM_DONOR_OP_HELLO) && gConnected) {
                dispatch_semaphore_signal(gConnected);
            }
        }
    });
    xpc_connection_resume(connection);
}

static NSString *DMRamDonorBundleIdentifier(void) {
    NSURL *plugIns = NSBundle.mainBundle.builtInPlugInsURL;
    if (!plugIns) return nil;
    NSArray<NSURL *> *urls = [[NSFileManager defaultManager]
        contentsOfDirectoryAtURL:plugIns
        includingPropertiesForKeys:nil
        options:NSDirectoryEnumerationSkipsHiddenFiles
        error:nil];
    for (NSURL *url in urls) {
        if (![url.pathExtension isEqualToString:@"appex"]) continue;
        NSBundle *bundle = [NSBundle bundleWithURL:url];
        NSDictionary *extension = bundle.infoDictionary[@"NSExtension"];
        if ([extension[@"NSExtensionPrincipalClass"] isEqualToString:@"DMRamDonorHandler"]) {
            return bundle.bundleIdentifier;
        }
    }
    return nil;
}

static BOOL DMRamDonorStart(void) {
    if (gDonorConnection) return YES;
    if (!gListener) {
        gConnected = dispatch_semaphore_create(0);
        gListener = xpc_connection_create(NULL,
            dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
        if (!gListener) {
            DMRamLog("could not create XPC listener");
            return NO;
        }
        xpc_connection_set_event_handler(gListener, ^(xpc_object_t event) {
            if (xpc_get_type(event) == XPC_TYPE_CONNECTION) {
                DMRamDonorAccept((xpc_connection_t)event);
            } else if (xpc_get_type(event) == XPC_TYPE_ERROR) {
                DMRamLog("XPC listener failed");
            }
        });
        xpc_connection_resume(gListener);
    }

    NSString *identifier = DMRamDonorBundleIdentifier();
    if (!identifier.length) {
        DMRamLog("embedded donor extension was not found");
        return NO;
    }

    NSError *error = nil;
    NSExtension *extension = [NSExtension extensionWithIdentifier:identifier error:&error];
    if (!extension) {
        DMRamLog("could not create extension %@: %@", identifier, error.localizedDescription ?: @"unknown error");
        return NO;
    }
    extension.preferredLanguages = @[];

    xpc_endpoint_t endpoint = xpc_endpoint_create(gListener);
    if (!endpoint) {
        DMRamLog("could not create listener endpoint");
        return NO;
    }
    NSXPCListenerEndpoint *wrapped = DMRamDonorWrapEndpoint(endpoint);

    NSExtensionItem *item = [NSExtensionItem new];
    item.userInfo = @{ @"endpoint": wrapped };

    dispatch_semaphore_t started = dispatch_semaphore_create(0);
    __block NSUUID *requestIdentifier = nil;
    [extension beginExtensionRequestWithInputItems:@[item] completion:^(NSUUID *identifier) {
        requestIdentifier = identifier;
        dispatch_semaphore_signal(started);
    }];

    if (dispatch_semaphore_wait(started, dispatch_time(DISPATCH_TIME_NOW, 12 * NSEC_PER_SEC))) {
        DMRamLog("donor extension did not start within 12 seconds");
        return NO;
    }
    if (!requestIdentifier) {
        DMRamLog("donor extension returned no request identifier");
        return NO;
    }
    if (dispatch_semaphore_wait(gConnected, dispatch_time(DISPATCH_TIME_NOW, 12 * NSEC_PER_SEC))) {
        DMRamLog("donor extension did not establish its XPC connection");
        [extension _kill:SIGTERM];
        return NO;
    }

    gExtension = extension;
    gRequestIdentifier = requestIdentifier;
    DMRamLog("helper extension connected (%@)", identifier);
    return YES;
}

static xpc_object_t DMRamDonorSend(const char *operation, uint64_t size,
                                   NSTimeInterval timeout) {
    xpc_connection_t connection = gDonorConnection;
    if (!connection) return NULL;

    xpc_object_t request = xpc_dictionary_create(NULL, NULL, 0);
    xpc_dictionary_set_string(request, DM_RAM_DONOR_KEY_OP, operation);
    if (size) xpc_dictionary_set_uint64(request, DM_RAM_DONOR_KEY_SIZE, size);

    dispatch_semaphore_t replied = dispatch_semaphore_create(0);
    __block xpc_object_t response = NULL;
    xpc_connection_send_message_with_reply(connection, request,
        dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^(xpc_object_t reply) {
            if (xpc_get_type(reply) == XPC_TYPE_DICTIONARY) response = xpc_retain(reply);
            dispatch_semaphore_signal(replied);
        });
    xpc_release(request);

    if (dispatch_semaphore_wait(replied,
        dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeout * NSEC_PER_SEC)))) {
        DMRamLog("timed out waiting for helper operation %s", operation);
        if (response) xpc_release(response);
        return NULL;
    }
    return response;
}

static BOOL DMRamDonorMapChunk(uintptr_t target, size_t size) {
    xpc_object_t reply = DMRamDonorSend(DM_RAM_DONOR_OP_CREATE, size, 30.0);
    if (!reply) return NO;

    xpc_object_t shmem = xpc_dictionary_get_value(reply, DM_RAM_DONOR_KEY_SHMEM);
    if (!shmem || xpc_get_type(shmem) != XPC_TYPE_SHMEM) {
        const char *error = xpc_dictionary_get_string(reply, DM_RAM_DONOR_KEY_ERROR);
        DMRamLog("helper could not allocate a %zu MB chunk: %s",
                 size >> 20, error ?: "no shared-memory handle");
        xpc_release(reply);
        return NO;
    }
    xpc_retain(shmem);
    xpc_release(reply);

    pthread_mutex_lock(&gLock);
    BOOL tableHasRoom = gMappedCount < DM_RAM_DONOR_MAX_CHUNKS;
    pthread_mutex_unlock(&gLock);
    if (!tableHasRoom) {
        xpc_release(shmem);
        DMRamLog("mapped-chunk table is full");
        return NO;
    }

    size_t pages = size / 4096;
    size_t bitmapBytes = (pages + 7) / 8;
    uint8_t *live = calloc(1, bitmapBytes);
    if (!live) {
        xpc_release(shmem);
        DMRamLog("could not allocate donor page bitmap");
        return NO;
    }
    memset(live, 0xff, bitmapBytes);
    if (pages & 7) live[bitmapBytes - 1] = (uint8_t)((1u << (pages & 7)) - 1u);

    void *scratch = NULL;
    size_t mappedSize = xpc_shmem_map(shmem, &scratch);
    if (!scratch || mappedSize < size) {
        if (scratch && mappedSize) mach_vm_deallocate(mach_task_self(), (mach_vm_address_t)scratch, mappedSize);
        free(live);
        xpc_release(shmem);
        DMRamLog("xpc_shmem_map returned only %zu of %zu bytes", mappedSize, size);
        return NO;
    }

    mach_vm_address_t destination = (mach_vm_address_t)target;
    vm_prot_t currentProtection = VM_PROT_NONE;
    vm_prot_t maximumProtection = VM_PROT_NONE;
    kern_return_t kr = mach_vm_remap(mach_task_self(), &destination, size, 0,
        VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, mach_task_self(),
        (mach_vm_address_t)scratch, false, &currentProtection,
        &maximumProtection, VM_INHERIT_SHARE);
    mach_vm_deallocate(mach_task_self(), (mach_vm_address_t)scratch, mappedSize);
    if (kr != KERN_SUCCESS || destination != (mach_vm_address_t)target) {
        free(live);
        xpc_release(shmem);
        DMRamLog("mach_vm_remap failed for guest heap chunk: %s", mach_error_string(kr));
        return NO;
    }

    pthread_mutex_lock(&gLock);
    if (gMappedCount >= DM_RAM_DONOR_MAX_CHUNKS) {
        pthread_mutex_unlock(&gLock);
        free(live);
        // This cannot occur during the single startup mapping pass. If it
        // does, leave the mapped region in place rather than replacing it with
        // an anonymous range that would break the existing file-backed heap.
        DMRamLog("mapped-chunk table filled after mapping a chunk");
        return NO;
    }
    gMapped[gMappedCount++] = (DMRamMappedChunk){
        .base = target, .size = size, .shmem = shmem,
        .livePages = live, .pageCount = pages, .liveCount = pages
    };
    pthread_mutex_unlock(&gLock);
    return YES;
}

size_t dm_ram_donor_map_heap(void *target, size_t requested) {
    if (!target || requested < DM_RAM_DONOR_CHUNK_SIZE) return 0;

    pthread_mutex_lock(&gLock);
    if (gAttempted) {
        size_t existing = 0;
        for (size_t i = 0; i < gMappedCount; ++i) existing += gMapped[i].size;
        pthread_mutex_unlock(&gLock);
        return existing;
    }
    gAttempted = YES;
    pthread_mutex_unlock(&gLock);

    if (!DMRamDonorStart()) {
        DMRamLog("helper unavailable; keeping file-backed guest memory");
        return 0;
    }

    xpc_object_t availableReply = DMRamDonorSend(DM_RAM_DONOR_OP_AVAILABLE, 0, 10.0);
    uint64_t available = availableReply
        ? xpc_dictionary_get_uint64(availableReply, DM_RAM_DONOR_KEY_AVAILABLE) : 0;
    if (availableReply) xpc_release(availableReply);

    uint64_t budget = available > DM_RAM_DONOR_HEADROOM
        ? available - DM_RAM_DONOR_HEADROOM : 0;
    uint64_t wanted = requested;
    uint64_t cap = DM_RAM_DONOR_CHUNK_SIZE * DM_RAM_DONOR_MAX_CHUNKS;
    if (wanted > cap) wanted = cap;
    if (wanted > budget) wanted = budget;
    wanted = (wanted / DM_RAM_DONOR_CHUNK_SIZE) * DM_RAM_DONOR_CHUNK_SIZE;

    if (wanted < DM_RAM_DONOR_CHUNK_SIZE) {
        DMRamLog("helper has insufficient safe headroom (%llu MB available); keeping file-backed guest memory",
                 (unsigned long long)(available >> 20));
        [gExtension _kill:SIGTERM];
        return 0;
    }

    size_t mapped = 0;
    while (mapped < wanted) {
        size_t chunk = (size_t)DM_RAM_DONOR_CHUNK_SIZE;
        if (!DMRamDonorMapChunk((uintptr_t)target + mapped, chunk)) break;
        mapped += chunk;
    }

    if (mapped) {
        int pid = [gExtension pidForRequestIdentifier:gRequestIdentifier];
        if (pid > 0) [gExtension _kill:SIGKILL];
        DMRamLog("mapped %zu MB of shared guest heap; donor process %d terminated; remaining heap stays file-backed",
                 mapped >> 20, pid);
    } else {
        [gExtension _kill:SIGTERM];
        DMRamLog("no shared guest memory was mapped; keeping file-backed guest memory");
    }
    return mapped;
}

void dm_ram_donor_restore_range(void *address, size_t size,
                                int memory_file_fd, void *guest_as_base) {
    if (!address || !size || !guest_as_base) return;

    uintptr_t requestedLo = (uintptr_t)address;
    uintptr_t requestedHi = requestedLo + size;
    if (requestedHi < requestedLo) return;
    size_t hostPage = (size_t)getpagesize();
    uintptr_t asBase = (uintptr_t)guest_as_base;

    pthread_mutex_lock(&gLock);
    for (size_t i = 0; i < gMappedCount;) {
        DMRamMappedChunk *chunk = &gMapped[i];
        uintptr_t chunkHi = chunk->base + chunk->size;
        uintptr_t lo = requestedLo > chunk->base ? requestedLo : chunk->base;
        uintptr_t hi = requestedHi < chunkHi ? requestedHi : chunkHi;
        if (lo >= hi) { ++i; continue; }

        lo &= ~(uintptr_t)(hostPage - 1);
        hi = (hi + hostPage - 1) & ~(uintptr_t)(hostPage - 1);
        if (lo < chunk->base) lo = chunk->base;
        if (hi > chunkHi) hi = chunkHi;
        size_t length = (size_t)(hi - lo);
        void *restored;
        if (memory_file_fd >= 0) {
            off_t offset = (off_t)(lo - asBase);
            restored = mmap((void *)lo, length, PROT_NONE,
                            MAP_SHARED | MAP_FIXED, memory_file_fd, offset);
        } else {
            restored = mmap((void *)lo, length, PROT_NONE,
                            MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
        }
        if (restored == MAP_FAILED) {
            DMRamLog("could not restore released guest range %p+%zu (errno %d); retaining donor mapping",
                     (void *)lo, length, errno);
            ++i;
            continue;
        }

        size_t firstPage = (size_t)(lo - chunk->base) / 4096;
        size_t lastPage = (size_t)(hi - chunk->base) / 4096;
        if (lastPage > chunk->pageCount) lastPage = chunk->pageCount;
        for (size_t page = firstPage; page < lastPage; ++page) {
            uint8_t mask = (uint8_t)(1u << (page & 7));
            uint8_t *byte = &chunk->livePages[page >> 3];
            if (*byte & mask) {
                *byte &= (uint8_t)~mask;
                if (chunk->liveCount) --chunk->liveCount;
            }
        }

        if (chunk->liveCount == 0) {
            xpc_release(chunk->shmem);
            free(chunk->livePages);
            DMRamLog("released a fully unused %zu MB donor chunk", chunk->size >> 20);
            gMapped[i] = gMapped[--gMappedCount];
            memset(&gMapped[gMappedCount], 0, sizeof(gMapped[gMappedCount]));
        } else {
            ++i;
        }
    }
    pthread_mutex_unlock(&gLock);
}
