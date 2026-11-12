#include <c10/core/impl/CPUAllocatorOverride.h>

namespace c10::impl {

namespace {
thread_local Allocator* cpu_allocator_override = nullptr;
} // namespace

Allocator* getCPUAllocatorOverride() noexcept {
  return cpu_allocator_override;
}

CPUAllocatorOverrideGuard::CPUAllocatorOverrideGuard(
    Allocator* allocator) noexcept
    : prev_(cpu_allocator_override) {
  cpu_allocator_override = allocator;
}

CPUAllocatorOverrideGuard::~CPUAllocatorOverrideGuard() {
  cpu_allocator_override = prev_;
}

} // namespace c10::impl
