#pragma once

#import <Foundation/Foundation.h>
#import <mach/mach.h>
#import <xpc/xpc.h>

#define DM_RAM_DONOR_CHUNK_SIZE (256ull * 1024ull * 1024ull)
#define DM_RAM_DONOR_MAX_CHUNKS 4
#define DM_RAM_DONOR_HEADROOM (256ull * 1024ull * 1024ull)

#define DM_RAM_DONOR_KEY_OP "op"
#define DM_RAM_DONOR_KEY_SIZE "size"
#define DM_RAM_DONOR_KEY_SHMEM "shmem"
#define DM_RAM_DONOR_KEY_ERROR "error"
#define DM_RAM_DONOR_KEY_AVAILABLE "available"

#define DM_RAM_DONOR_OP_HELLO "hello"
#define DM_RAM_DONOR_OP_CREATE "create"
#define DM_RAM_DONOR_OP_AVAILABLE "available"
#define DM_RAM_DONOR_OP_RELEASE "release"

@interface NSXPCListenerEndpoint (DMRamDonorPrivate)
@property (nonatomic, strong, setter=_setEndpoint:) xpc_object_t _endpoint;
@end

static inline NSXPCListenerEndpoint *DMRamDonorWrapEndpoint(xpc_endpoint_t endpoint) {
    NSXPCListenerEndpoint *wrapper = [NSXPCListenerEndpoint new];
    wrapper._endpoint = endpoint;
    return wrapper;
}

static inline xpc_endpoint_t DMRamDonorUnwrapEndpoint(NSXPCListenerEndpoint *wrapper) {
    if (!wrapper) return NULL;
    xpc_object_t endpoint = wrapper._endpoint;
    if (!endpoint || xpc_get_type(endpoint) != XPC_TYPE_ENDPOINT) return NULL;
    return (xpc_endpoint_t)endpoint;
}
