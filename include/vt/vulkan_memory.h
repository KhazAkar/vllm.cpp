#pragma once

#include <cstdint>

#include "vulkan/vulkan_core.h"

namespace vt::vulkan {

struct MemorySelection {
  int type_index = -1;
  uint32_t heap_index = 0;
  VkDeviceSize heap_size = 0;
  bool unified = false;
};

// Vulkan allocations are always host-visible, host-coherent, and persistently
// mapped in this backend. The device type is retained in the seam because the
// diagnostic distinction matters for discrete BAR-backed heaps, even though
// reference-tier eligibility follows the host-mapped allocation property rather
// than DEVICE_LOCAL.
inline MemorySelection SelectMemoryType(
    const VkPhysicalDeviceMemoryProperties& properties,
    VkPhysicalDeviceType /*device_type*/) {
  constexpr VkMemoryPropertyFlags kHostFlags =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  MemorySelection result;
  for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    const auto flags = properties.memoryTypes[i].propertyFlags;
    if ((flags & (kHostFlags | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) ==
        (kHostFlags | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      result.type_index = static_cast<int>(i);
      break;
    }
  }
  if (result.type_index < 0) {
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
      if ((properties.memoryTypes[i].propertyFlags & kHostFlags) == kHostFlags) {
        result.type_index = static_cast<int>(i);
        break;
      }
    }
  }
  if (result.type_index >= 0) {
    result.heap_index =
        properties.memoryTypes[static_cast<uint32_t>(result.type_index)].heapIndex;
    result.heap_size = properties.memoryHeaps[result.heap_index].size;
    result.unified = true;
  }
  return result;
}

}  // namespace vt::vulkan
