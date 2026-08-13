# Vulkan host-mapped memory and reference-tier eligibility

## Issue and scope

This follow-up addresses the Vulkan memory-property split tracked under
`BACKEND-VULKAN`, especially issues
[#125](https://github.com/mudler/vllm.cpp/issues/125) and
[#203](https://github.com/mudler/vllm.cpp/issues/203). It is intentionally separate from model work and is
limited to two backend seams:

1. whether a plain host pointer may be bound as a device tensor; and
2. whether the portable CPU reference tier may operate directly on Vulkan
   allocations.

Every Vulkan storage allocation is `HOST_VISIBLE|HOST_COHERENT` and remains
persistently mapped. Vulkan `Copy` and `Memset` are host operations over those
mapped pointers, so reference-tier execution is safe even when the selected
memory type is not `DEVICE_LOCAL`.

## Design

Add a backend capability meaning "a host pointer may be bound as device
memory." Its default is `UnifiedMemory()` to preserve CPU/CUDA/ROCm/Metal
behavior, while Vulkan overrides it to false: Vulkan device allocations are
host-mapped, but arbitrary `std::vector` pointers are not Vulkan allocations
and cannot be descriptor-bound. `DeviceScratch` will consult this predicate
instead of `UnifiedMemory()`.

On Vulkan, make `UnifiedMemory()` represent the property actually required by
the reference tier: the backend's allocations are host-visible, host-coherent,
persistently mapped, and directly accessed by host reference kernels. Keep the
existing memory-type selection order (prefer device-local host-visible, then
host-visible coherent), and expose the selected type and heap size for
diagnostics. The non-ReBAR discrete allocation ceiling remains an owed
performance limitation; staging/device-local allocations are out of scope.

Extract memory-type selection and the unified flag into a pure function over
synthetic `VkPhysicalDeviceMemoryProperties` plus device type, so integrated
shared-heap, discrete-with-ReBAR, and discrete-without-ReBAR cases can run
without a Vulkan device.

## Tests and evidence

RED-first CPU tests will cover:

- chosen memory type and unified result for integrated, discrete-with-ReBAR,
  and discrete-without-ReBAR properties;
- default host-pointer-binding capability matching `UnifiedMemory()` for
  backends that use the default;
- Vulkan's explicit false override; and
- `DeviceScratch` materializing and copying host data when binding is not
  allowed.

If the test harness cannot register a fake backend for `DeviceScratch`, that
coverage will be reported as unavailable rather than silently skipped.

The CPU Release build and applicable sampling, op-provider, backend, and
Vulkan tests are required. If Vulkan headers and loader are available, a
Vulkan-enabled build will compile the changed Vulkan translation units; no
Vulkan runtime device is assumed.

## Confirmed implementation anchors

- `VulkanContext::AllocBuffer` selects host-visible coherent memory and maps it.
- `VulkanBackend::Copy` and `Memset` operate on mapped pointers.
- `op_provider.cpp` gates the portable reference tier on `UnifiedMemory()`.
- `DeviceScratch` currently aliases arbitrary host pointers when
  `UnifiedMemory()` is true.

## Limitations

No staging path or device-local-only allocation is added. A discrete board
without a BAR-backed host-visible device-local heap will retain the existing
host-visible allocation ceiling. The user's RX 7700S memory topology remains
pending their Vulkan capability report.

## Outcome

The supplied code claims were confirmed: Vulkan allocations are persistently
mapped host-visible/coherent storage, `Copy`/`Memset` are host operations,
`DeviceMemoryIsHostAddressable()` is true, `DeviceScratch` was the
host-pointer aliasing site, and reference-tier eligibility uses
`UnifiedMemory()`. The fix keeps preferred memory-type selection and makes the
unified result true for both the preferred and fallback host-mapped choices.

CPU tests cover all three synthetic memory layouts and the copy-vs-alias seam.
The Vulkan Release build compiled the changed backend, but this host has no
conformant Vulkan device, so the runtime Vulkan override test and backend
device tests were unverified/skipped. BAR/ReBAR performance and the RX 7700S
heap topology remain hardware-pending.
