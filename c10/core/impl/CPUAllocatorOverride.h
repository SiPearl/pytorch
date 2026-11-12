#pragma once

#include <c10/core/Allocator.h>
#include <c10/macros/Export.h>

namespace c10::impl {

// Calling-thread override of the allocator used for non-pinned CPU tensor
// storage: at::detail::empty_cpu, at::detail::empty_strided_cpu, and resize_
// of a tensor that has no storage yet.
//
// - Scope is the calling thread only. Other threads, including intra-op workers
//   and at::launch tasks, do not see the override.
// - Precedence: the innermost live guard, then c10::GetCPUAllocator(). A guard
//   constructed with nullptr selects the global allocator inside an outer
//   guard.
// - pin_memory=True requests are not affected.
// - A storage keeps the allocator that created it. Tensors allocated inside a
//   scope are resized and freed through that allocator after the scope exits,
//   and tensors allocated outside are not re-homed when resized inside one.
// - Ownership is not taken. As with SetAllocator, the caller must keep the
//   allocator alive until every storage it created is gone.
C10_API Allocator* getCPUAllocatorOverride() noexcept;

class C10_API CPUAllocatorOverrideGuard {
 public:
  explicit CPUAllocatorOverrideGuard(Allocator* allocator) noexcept;
  ~CPUAllocatorOverrideGuard();

  CPUAllocatorOverrideGuard(const CPUAllocatorOverrideGuard&) = delete;
  CPUAllocatorOverrideGuard(CPUAllocatorOverrideGuard&&) = delete;
  CPUAllocatorOverrideGuard& operator=(const CPUAllocatorOverrideGuard&) =
      delete;
  CPUAllocatorOverrideGuard& operator=(CPUAllocatorOverrideGuard&&) = delete;

 private:
  Allocator* prev_;
};

} // namespace c10::impl
