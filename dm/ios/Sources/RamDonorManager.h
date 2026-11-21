#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maps up to 1 GiB of shared donor memory over the start of the guest heap.
// Returns the number of bytes successfully mapped. A zero result is a safe
// fallback to the existing file-backed guest memory.
size_t dm_ram_donor_map_heap(void *target, size_t requested);

// Restores any donor-backed portion of a released guest range to the original
// file/anonymous mapping, then drops fully released shared-memory chunks.
void dm_ram_donor_restore_range(void *address, size_t size,
                                int memory_file_fd, void *guest_as_base);

#ifdef __cplusplus
}
#endif
