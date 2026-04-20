//===-- libkfd/ioctl.h - Ioctl protocol definitions -------------*- C++ -*-===//
//
// C++ wrappers around <linux/kfd_ioctl.h>, <drm/drm.h>, and
// <drm/amdgpu_drm.h>. The kernel headers are the source of truth for struct
// layouts, enums, and constants.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_IOCTL_H
#define LIBKFD_IOCTL_H

#include "libkfd/error.h"

#include <cerrno>

#include <drm/amdgpu_drm.h>
#include <drm/drm.h>
#include <linux/kfd_ioctl.h>
#include <sys/ioctl.h>

namespace kfd::ioctl {

// KFD ioctls (kfd::ioctl::kfd::*)
namespace kfd {

using version_args = ::kfd_ioctl_get_version_args;
using acquire_vm_args = ::kfd_ioctl_acquire_vm_args;
using set_memory_policy_args = ::kfd_ioctl_set_memory_policy_args;
using process_device_apertures = ::kfd_process_device_apertures;
using get_process_apertures_new_args =
    ::kfd_ioctl_get_process_apertures_new_args;
using alloc_memory_of_gpu_args = ::kfd_ioctl_alloc_memory_of_gpu_args;
using free_memory_of_gpu_args = ::kfd_ioctl_free_memory_of_gpu_args;
using map_memory_to_gpu_args = ::kfd_ioctl_map_memory_to_gpu_args;
using unmap_memory_from_gpu_args = ::kfd_ioctl_unmap_memory_from_gpu_args;
using create_queue_args = ::kfd_ioctl_create_queue_args;
using update_queue_args = ::kfd_ioctl_update_queue_args;
using destroy_queue_args = ::kfd_ioctl_destroy_queue_args;
using create_event_args = ::kfd_ioctl_create_event_args;
using destroy_event_args = ::kfd_ioctl_destroy_event_args;
using set_event_args = ::kfd_ioctl_set_event_args;
using reset_event_args = ::kfd_ioctl_reset_event_args;
using wait_events_args = ::kfd_ioctl_wait_events_args;
using event_data = ::kfd_event_data;
using set_scratch_backing_va_args = ::kfd_ioctl_set_scratch_backing_va_args;
using svm_args = ::kfd_ioctl_svm_args;
using svm_attribute = ::kfd_ioctl_svm_attribute;
using set_xnack_mode_args = ::kfd_ioctl_set_xnack_mode_args;
using set_trap_handler_args = ::kfd_ioctl_set_trap_handler_args;
using runtime_enable_args = ::kfd_ioctl_runtime_enable_args;

inline constexpr unsigned long GET_VERSION = AMDKFD_IOC_GET_VERSION;
inline constexpr unsigned long ACQUIRE_VM = AMDKFD_IOC_ACQUIRE_VM;
inline constexpr unsigned long SET_MEMORY_POLICY = AMDKFD_IOC_SET_MEMORY_POLICY;
inline constexpr unsigned long GET_PROCESS_APERTURES_NEW =
    AMDKFD_IOC_GET_PROCESS_APERTURES_NEW;
inline constexpr unsigned long ALLOC_MEMORY_OF_GPU =
    AMDKFD_IOC_ALLOC_MEMORY_OF_GPU;
inline constexpr unsigned long FREE_MEMORY_OF_GPU =
    AMDKFD_IOC_FREE_MEMORY_OF_GPU;
inline constexpr unsigned long MAP_MEMORY_TO_GPU = AMDKFD_IOC_MAP_MEMORY_TO_GPU;
inline constexpr unsigned long UNMAP_MEMORY_FROM_GPU =
    AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU;
inline constexpr unsigned long CREATE_QUEUE = AMDKFD_IOC_CREATE_QUEUE;
inline constexpr unsigned long UPDATE_QUEUE = AMDKFD_IOC_UPDATE_QUEUE;
inline constexpr unsigned long DESTROY_QUEUE = AMDKFD_IOC_DESTROY_QUEUE;
inline constexpr unsigned long CREATE_EVENT = AMDKFD_IOC_CREATE_EVENT;
inline constexpr unsigned long DESTROY_EVENT = AMDKFD_IOC_DESTROY_EVENT;
inline constexpr unsigned long SET_EVENT = AMDKFD_IOC_SET_EVENT;
inline constexpr unsigned long RESET_EVENT = AMDKFD_IOC_RESET_EVENT;
inline constexpr unsigned long WAIT_EVENTS = AMDKFD_IOC_WAIT_EVENTS;
inline constexpr unsigned long SET_SCRATCH_BACKING_VA =
    AMDKFD_IOC_SET_SCRATCH_BACKING_VA;
inline constexpr unsigned long SVM = AMDKFD_IOC_SVM;
inline constexpr unsigned long SET_XNACK_MODE = AMDKFD_IOC_SET_XNACK_MODE;
inline constexpr unsigned long SET_TRAP_HANDLER = AMDKFD_IOC_SET_TRAP_HANDLER;
inline constexpr unsigned long RUNTIME_ENABLE = AMDKFD_IOC_RUNTIME_ENABLE;

} // namespace kfd

// DRM ioctls (kfd::ioctl::drm::*)
namespace drm {

using version_args = ::drm_version;
using info_args = ::drm_amdgpu_info;
using ctx_args = ::drm_amdgpu_ctx;

inline constexpr unsigned long GET_VERSION = DRM_IOCTL_VERSION;
inline constexpr unsigned long AMDGPU_INFO = DRM_IOCTL_AMDGPU_INFO;
inline constexpr unsigned long AMDGPU_CTX = DRM_IOCTL_AMDGPU_CTX;

} // namespace drm

// Extra size handling used for SVM requests.
template <unsigned long Request, typename Args>
std::expected<void, Error> call(int fd, Args &args,
                                std::size_t extra_size = 0) {
  unsigned long cmd = Request + (extra_size << _IOC_SIZESHIFT);
  int ret;
  do {
    ret = ::ioctl(fd, cmd, &args);
  } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
  if (ret == -1)
    return unexpected(errno);
  return {};
}

} // namespace kfd::ioctl

#endif // LIBKFD_IOCTL_H
