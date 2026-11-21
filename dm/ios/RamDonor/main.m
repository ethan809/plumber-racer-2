// Experimental iOS RAM-donor extension. The host app maps these shared-memory
// regions into its guest heap, then may terminate this helper process. This is
// a per-process footprint workaround, not additional physical RAM.
@import Foundation;

#import <mach/mach.h>
#import <mach/mach_error.h>
#import <objc/runtime.h>
#import <os/proc.h>
#import <xpc/xpc.h>
#import <pthread.h>
#import <signal.h>
#import <stdlib.h>
#import <string.h>

#import "../Sources/RamDonorProtocol.h"

extern kern_return_t mach_vm_allocate(vm_map_t target, mach_vm_address_t *address,
                                      mach_vm_size_t size, int flags);
extern kern_return_t mach_vm_deallocate(vm_map_t target, mach_vm_address_t address,
                                        mach_vm_size_t size);

#ifndef VM_FLAGS_PURGABLE
#define VM_FLAGS_PURGABLE 0x2
#endif

typedef struct {
    mach_vm_address_t address;
    mach_vm_size_t size;
} DMRamDonorRegion;

static DMRamDonorRegion gRegions[DM_RAM_DONOR_MAX_CHUNKS];
static size_t gRegionCount;
static pthread_mutex_t gRegionLock = PTHREAD_MUTEX_INITIALIZER;

static void DMRamDonorReleaseAll(void) {
    pthread_mutex_lock(&gRegionLock);
    for (size_t i = 0; i < gRegionCount; ++i) {
        if (gRegions[i].address && gRegions[i].size) {
            mach_vm_deallocate(mach_task_self(), gRegions[i].address, gRegions[i].size);
        }
    }
    memset(gRegions, 0, sizeof(gRegions));
    gRegionCount = 0;
    pthread_mutex_unlock(&gRegionLock);
}

static xpc_object_t DMRamDonorCreateRegion(uint64_t size, NSString **error) {
    if (!size || size > DM_RAM_DONOR_CHUNK_SIZE) {
        *error = @"Invalid donor region size";
        return NULL;
    }

    pthread_mutex_lock(&gRegionLock);
    if (gRegionCount >= DM_RAM_DONOR_MAX_CHUNKS) {
        pthread_mutex_unlock(&gRegionLock);
        *error = @"Donor chunk limit reached";
        return NULL;
    }
    pthread_mutex_unlock(&gRegionLock);

    mach_vm_address_t address = 0;
    kern_return_t kr = mach_vm_allocate(mach_task_self(), &address, size,
                                        VM_FLAGS_ANYWHERE | VM_FLAGS_PURGABLE);
    if (kr != KERN_SUCCESS) {
        *error = [NSString stringWithFormat:@"mach_vm_allocate failed: %s",
                  mach_error_string(kr)];
        return NULL;
    }

    xpc_object_t shmem = xpc_shmem_create((void *)(uintptr_t)address, (size_t)size);
    if (!shmem) {
        mach_vm_deallocate(mach_task_self(), address, size);
        *error = @"xpc_shmem_create returned NULL";
        return NULL;
    }

    pthread_mutex_lock(&gRegionLock);
    if (gRegionCount >= DM_RAM_DONOR_MAX_CHUNKS) {
        pthread_mutex_unlock(&gRegionLock);
        mach_vm_deallocate(mach_task_self(), address, size);
        *error = @"Donor chunk limit reached";
        return NULL;
    }
    gRegions[gRegionCount++] = (DMRamDonorRegion){address, size};
    pthread_mutex_unlock(&gRegionLock);
    return shmem;
}

static void DMRamDonorHandleMessage(xpc_connection_t channel, xpc_object_t message) {
    if (xpc_get_type(message) != XPC_TYPE_DICTIONARY) return;

    const char *op = xpc_dictionary_get_string(message, DM_RAM_DONOR_KEY_OP);
    if (!op) return;

    xpc_object_t reply = xpc_dictionary_create_reply(message);
    if (!reply) return;

    if (!strcmp(op, DM_RAM_DONOR_OP_CREATE)) {
        NSString *error = nil;
        uint64_t size = xpc_dictionary_get_uint64(message, DM_RAM_DONOR_KEY_SIZE);
        xpc_object_t shmem = DMRamDonorCreateRegion(size, &error);
        if (shmem) {
            xpc_dictionary_set_value(reply, DM_RAM_DONOR_KEY_SHMEM, shmem);
        } else {
            xpc_dictionary_set_string(reply, DM_RAM_DONOR_KEY_ERROR,
                                      error.UTF8String ?: "unknown allocation error");
            NSLog(@"RAM donor allocation failed: %@", error);
        }
    } else if (!strcmp(op, DM_RAM_DONOR_OP_AVAILABLE)) {
        xpc_dictionary_set_uint64(reply, DM_RAM_DONOR_KEY_AVAILABLE,
                                  (uint64_t)os_proc_available_memory());
    } else if (!strcmp(op, DM_RAM_DONOR_OP_RELEASE)) {
        DMRamDonorReleaseAll();
    } else if (!strcmp(op, DM_RAM_DONOR_OP_HELLO)) {
        xpc_dictionary_set_bool(reply, "ready", true);
    } else {
        xpc_dictionary_set_string(reply, DM_RAM_DONOR_KEY_ERROR, "Unknown operation");
    }

    xpc_connection_send_message(channel, reply);
}

static void DMRamDonorDisableEndpointClassCheck(id self, SEL command,
                                                 Class cls, NSString *key,
                                                 BOOL allowingInvocations) {
    (void)self; (void)command; (void)cls; (void)key; (void)allowingInvocations;
}

__attribute__((constructor))
static void DMRamDonorPrepareEndpointDecoding(void) {
    Class decoder = NSClassFromString(@"NSXPCDecoder");
    SEL selector = NSSelectorFromString(@"_validateAllowedClass:forKey:allowingInvocations:");
    Method method = decoder ? class_getInstanceMethod(decoder, selector) : NULL;
    if (method) {
        method_setImplementation(method, (IMP)DMRamDonorDisableEndpointClassCheck);
        NSLog(@"RAM donor: endpoint decoder compatibility hook installed");
    } else {
        NSLog(@"RAM donor: NSXPCDecoder compatibility hook unavailable");
    }
}

@interface DMRamDonorHandler : NSObject <NSExtensionRequestHandling>
@end

@implementation DMRamDonorHandler {
    xpc_connection_t _channel;
}

- (void)beginRequestWithExtensionContext:(NSExtensionContext *)context {
    NSExtensionItem *item = (NSExtensionItem *)context.inputItems.firstObject;
    NSDictionary *info = item.userInfo;
    id wrapped = info[@"endpoint"];
    if (![wrapped isKindOfClass:NSXPCListenerEndpoint.class]) {
        NSLog(@"RAM donor: request did not contain an NSXPCListenerEndpoint");
        return;
    }

    xpc_endpoint_t endpoint = DMRamDonorUnwrapEndpoint((NSXPCListenerEndpoint *)wrapped);
    if (!endpoint) {
        NSLog(@"RAM donor: request contained no usable XPC endpoint");
        return;
    }

    _channel = xpc_connection_create_from_endpoint(endpoint);
    if (!_channel) {
        NSLog(@"RAM donor: could not create the XPC connection");
        return;
    }

    xpc_connection_set_event_handler(_channel, ^(xpc_object_t event) {
        if (xpc_get_type(event) == XPC_TYPE_ERROR) {
            NSLog(@"RAM donor: host connection ended");
            exit(0);
        }
        DMRamDonorHandleMessage(self->_channel, event);
    });
    xpc_connection_resume(_channel);

    xpc_object_t hello = xpc_dictionary_create(NULL, NULL, 0);
    xpc_dictionary_set_string(hello, DM_RAM_DONOR_KEY_OP, DM_RAM_DONOR_OP_HELLO);
    xpc_connection_send_message(_channel, hello);
    NSLog(@"RAM donor: helper ready (pid %d)", getpid());
}

@end
