#pragma once

#include <cstdint>

#include "vulkan/vulkan_core.h"

namespace vt::vulkan {

struct MemorySelection {
  int type_index = -1;
  uint32_t heap_index = 0;
  VkDeviceSize heap_size = 0;
  bool unified = false;
  bool bar_backed_discrete = false;
};

// Vulkan allocations are always host-visible, host-coherent, and persistently
// mapped in this backend. A discrete selection with DEVICE_LOCAL identifies the
// BAR-backed case that matters for the allocation ceiling diagnostic, while
// reference-tier eligibility follows the host-mapped property rather than
// DEVICE_LOCAL.
inline MemorySelection SelectMemoryType(
    const VkPhysicalDeviceMemoryProperties& properties,
    VkPhysicalDeviceType device_type) {
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
    const auto flags =
        properties.memoryTypes[static_cast<uint32_t>(result.type_index)].propertyFlags;
    result.heap_index =
        properties.memoryTypes[static_cast<uint32_t>(result.type_index)].heapIndex;
    result.heap_size = properties.memoryHeaps[result.heap_index].size;
    result.unified = true;
    result.bar_backed_discrete =
        device_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
        (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
  }
  return result;
}

}  // namespace vt::vulkan
