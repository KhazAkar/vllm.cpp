#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "vllm/v1/sample/device_scratch.h"
#include "vt/backend.h"
#include "vt/vulkan_memory.h"

namespace {

VkPhysicalDeviceMemoryProperties Properties(
    std::initializer_list<VkMemoryPropertyFlags> flags,
    std::initializer_list<VkDeviceSize> heaps) {
  VkPhysicalDeviceMemoryProperties p{};
  p.memoryTypeCount = static_cast<uint32_t>(flags.size());
  uint32_t i = 0;
  for (VkMemoryPropertyFlags f : flags) p.memoryTypes[i++].propertyFlags = f;
  p.memoryHeapCount = static_cast<uint32_t>(heaps.size());
  i = 0;
  for (VkDeviceSize size : heaps) p.memoryHeaps[i++].size = size;
  for (uint32_t type = 0; type < p.memoryTypeCount; ++type)
    p.memoryTypes[type].heapIndex = type < p.memoryHeapCount ? type : 0;
  return p;
}

class FakeBackend final : public vt::Backend {
 public:
  void* Alloc(size_t bytes) override {
    owned.resize(bytes);
    return owned.data();
  }
  void Free(void*) override {}
  void Memset(vt::Queue&, void* p, int value, size_t bytes) override {
    std::memset(p, value, bytes);
  }
  void Copy(vt::Queue&, void* dst, const void* src, size_t bytes) override {
    copied = true;
    std::memcpy(dst, src, bytes);
  }
  vt::Queue CreateQueue() override { return vt::Queue{vt::Device{vt::DeviceType::kXPU, 0}}; }
  bool UnifiedMemory() const override { return true; }
  bool HostPointerCanBeBoundAsDeviceMemory() const override { return false; }
  bool copied = false;
  std::vector<uint8_t> owned;
};

}  // namespace

TEST_CASE("Vulkan memory selection prefers device-local host-visible memory") {
  constexpr auto host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  constexpr auto preferred = host | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

  auto integrated = Properties({preferred, host}, {8u << 30, 2u << 30});
  auto selected = vt::vulkan::SelectMemoryType(
      integrated, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
  CHECK(selected.type_index == 0);
  CHECK(selected.heap_index == 0);
  CHECK(selected.heap_size == 8u << 30);
  CHECK(selected.unified);
  CHECK_FALSE(selected.bar_backed_discrete);

  auto rebar = Properties({host, preferred}, {2u << 30, 16u << 30});
  selected = vt::vulkan::SelectMemoryType(rebar,
                                           VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
  CHECK(selected.type_index == 1);
  CHECK(selected.heap_index == 1);
  CHECK(selected.heap_size == 16u << 30);
  CHECK(selected.unified);
  CHECK(selected.bar_backed_discrete);

  auto no_rebar = Properties({host}, {256u << 20});
  selected = vt::vulkan::SelectMemoryType(
      no_rebar, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
  CHECK(selected.type_index == 0);
  CHECK(selected.heap_index == 0);
  CHECK(selected.heap_size == 256u << 20);
  CHECK(selected.unified);
  CHECK_FALSE(selected.bar_backed_discrete);
}

TEST_CASE("host pointer binding defaults to unified memory and scratch copies") {
  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  CHECK(cpu.HostPointerCanBeBoundAsDeviceMemory() == cpu.UnifiedMemory());

  FakeBackend fake;
  vt::RegisterBackend(vt::DeviceType::kXPU, &fake);
  vt::Queue q = fake.CreateQueue();
  std::vector<uint8_t> source{1, 2, 3, 4};
  {
    vllm::v1::DeviceScratch scratch(vt::Device{vt::DeviceType::kXPU, 0}, q,
                                    source.data(), vt::DType::kI8, {4});
    CHECK(scratch.tensor().data != source.data());
    CHECK(fake.copied);
    CHECK(std::memcmp(scratch.tensor().data, source.data(), source.size()) == 0);
  }
}

#ifdef VLLM_CPP_VULKAN
TEST_CASE("Vulkan does not bind arbitrary host pointers as device memory") {
  if (vt::TryGetBackend(vt::DeviceType::kVULKAN) == nullptr) {
    MESSAGE("no Vulkan loader or conformant device; Vulkan override unverified");
    return;
  }
  vt::Backend& vulkan = vt::GetBackend(vt::DeviceType::kVULKAN);
  CHECK_FALSE(vulkan.HostPointerCanBeBoundAsDeviceMemory());
  CHECK(vulkan.DeviceMemoryIsHostAddressable());
}
#endif
