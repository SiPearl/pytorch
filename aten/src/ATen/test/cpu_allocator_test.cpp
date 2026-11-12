#include <gtest/gtest.h>

#include <c10/core/CPUAllocator.h>
#include <c10/core/impl/CPUAllocatorOverride.h>
#include <c10/core/impl/alloc_cpu.h>
#include <ATen/ATen.h>
#include <ATen/DeviceAccelerator.h>

#include <ATen/test/allocator_clone_test.h>

#include <memory>
#include <thread>
#include <vector>

TEST(AllocatorTestCPU, test_clone) {
  test_allocator_clone(c10::GetDefaultCPUAllocator());
}

namespace {

using c10::impl::CPUAllocatorOverrideGuard;
using c10::impl::getCPUAllocatorOverride;
using Sizes = std::vector<size_t>;

// The DataPtr context records the size, so the deleter reports what was freed
// and through which allocator.
struct CountingAllocator final : at::Allocator {
  struct Block {
    CountingAllocator* owner;
    void* data;
    size_t nbytes;
  };

  at::DataPtr allocate(size_t nbytes) override {
    void* data = c10::alloc_cpu(nbytes);
    allocated.push_back(nbytes);
    live_bytes += nbytes;
    return {data, new Block{this, data, nbytes}, &free_block, at::Device(at::kCPU)};
  }

  static void free_block(void* ctx) {
    std::unique_ptr<Block> block(static_cast<Block*>(ctx));
    block->owner->freed.push_back(block->nbytes);
    block->owner->live_bytes -= block->nbytes;
    c10::free_cpu(block->data);
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    default_copy_data(dest, src, count);
  }

  Sizes allocated;
  Sizes freed;
  size_t live_bytes = 0;
};

at::Allocator* allocator_of(const at::Tensor& t) {
  return t.storage().allocator();
}

} // namespace

TEST(CPUAllocatorOverrideTest, NoOverride) {
  ASSERT_EQ(getCPUAllocatorOverride(), nullptr);
  ASSERT_EQ(allocator_of(at::empty({4})), c10::GetCPUAllocator());
}

TEST(CPUAllocatorOverrideTest, SingleScope) {
  CountingAllocator alloc;
  auto input = at::ones({2, 3});
  auto weight = at::ones({4, 3});
  auto bias = at::ones({4});
  at::Tensor empty, strided, sum, linear;
  {
    CPUAllocatorOverrideGuard guard(&alloc);
    ASSERT_EQ(getCPUAllocatorOverride(), &alloc);
    empty = at::empty({2, 3});
    strided = at::empty_strided({2, 3}, {1, 2});
    sum = input + input;
    linear = at::linear(input, weight, bias);
  }
  ASSERT_EQ(getCPUAllocatorOverride(), nullptr);
  for (const auto& t : {empty, strided, sum, linear}) {
    ASSERT_EQ(allocator_of(t), &alloc);
  }
  ASSERT_EQ(allocator_of(input), c10::GetCPUAllocator());
  ASSERT_EQ(allocator_of(at::empty({4})), c10::GetCPUAllocator());
  ASSERT_TRUE(sum.equal(at::full({2, 3}, 2.0)));
  ASSERT_TRUE(linear.equal(at::full({2, 4}, 4.0)));

  empty = strided = sum = linear = at::Tensor();
  ASSERT_EQ(alloc.live_bytes, 0u);
  ASSERT_EQ(alloc.freed.size(), alloc.allocated.size());
}

TEST(CPUAllocatorOverrideTest, NestedScopesRestoreOuter) {
  CountingAllocator outer, inner;
  {
    CPUAllocatorOverrideGuard outer_guard(&outer);
    ASSERT_EQ(allocator_of(at::empty({1})), &outer);
    {
      CPUAllocatorOverrideGuard inner_guard(&inner);
      ASSERT_EQ(allocator_of(at::empty({2})), &inner);
    }
    ASSERT_EQ(getCPUAllocatorOverride(), &outer);
    ASSERT_EQ(allocator_of(at::empty({3})), &outer);
  }
  ASSERT_EQ(getCPUAllocatorOverride(), nullptr);
  ASSERT_EQ(outer.allocated, Sizes({4, 12}));
  ASSERT_EQ(outer.freed, outer.allocated);
  ASSERT_EQ(inner.allocated, Sizes({8}));
  ASSERT_EQ(inner.freed, inner.allocated);
}

TEST(CPUAllocatorOverrideTest, NullInnerScopeSelectsGlobal) {
  CountingAllocator alloc;
  CPUAllocatorOverrideGuard guard(&alloc);
  {
    CPUAllocatorOverrideGuard no_override(nullptr);
    ASSERT_EQ(getCPUAllocatorOverride(), nullptr);
    ASSERT_EQ(allocator_of(at::empty({4})), c10::GetCPUAllocator());
  }
  ASSERT_TRUE(alloc.allocated.empty());
  ASSERT_EQ(allocator_of(at::empty({4})), &alloc);
}

TEST(CPUAllocatorOverrideTest, RestoredAfterException) {
  CountingAllocator alloc;
  auto a = at::ones({2});
  auto b = at::ones({3});
  auto throws_in_scope = [&] {
    CPUAllocatorOverrideGuard guard(&alloc);
    auto kept = at::empty({8});
    at::add(a, b);
  };
  ASSERT_THROW(throws_in_scope(), c10::Error);
  ASSERT_EQ(getCPUAllocatorOverride(), nullptr);
  ASSERT_FALSE(alloc.allocated.empty());
  ASSERT_EQ(alloc.live_bytes, 0u);
  ASSERT_EQ(allocator_of(at::empty({4})), c10::GetCPUAllocator());
}

TEST(CPUAllocatorOverrideTest, PinnedRequestIgnoresOverride) {
  if (!at::globalContext().hasCUDA() && !at::getAccelerator(false).has_value()) {
    GTEST_SKIP() << "No pinned memory allocator available";
  }
  CountingAllocator alloc;
  CPUAllocatorOverrideGuard guard(&alloc);
  auto pinned = at::empty({4}, at::TensorOptions().pinned_memory(true));
  ASSERT_NE(allocator_of(pinned), &alloc);
  ASSERT_TRUE(alloc.allocated.empty());
}

TEST(CPUAllocatorOverrideTest, ViewOutlivingScopeFreesOnceThroughCreator) {
  CountingAllocator alloc;
  at::Tensor view;
  {
    CPUAllocatorOverrideGuard guard(&alloc);
    view = at::empty({4, 8}).narrow(0, 1, 2);
  }
  ASSERT_EQ(alloc.live_bytes, 128u);
  ASSERT_TRUE(alloc.freed.empty());
  view = at::Tensor();
  ASSERT_EQ(alloc.freed, Sizes({128}));
  ASSERT_EQ(alloc.live_bytes, 0u);
}

TEST(CPUAllocatorOverrideTest, ResizeAfterScopeUsesCreatingAllocator) {
  CountingAllocator alloc;
  at::Tensor t;
  {
    CPUAllocatorOverrideGuard guard(&alloc);
    t = at::empty({4});
  }
  t.resize_({64});
  ASSERT_EQ(allocator_of(t), &alloc);
  ASSERT_EQ(alloc.allocated, Sizes({16, 256}));
  ASSERT_EQ(alloc.freed, Sizes({16}));
}

TEST(CPUAllocatorOverrideTest, DefaultTensorResizedInScopeKeepsDefault) {
  CountingAllocator alloc;
  auto t = at::empty({4});
  {
    CPUAllocatorOverrideGuard guard(&alloc);
    t.resize_({64});
  }
  ASSERT_EQ(allocator_of(t), c10::GetCPUAllocator());
  ASSERT_TRUE(alloc.allocated.empty());
}

TEST(CPUAllocatorOverrideTest, StoragelessResizeInScopeIsRouted) {
  CountingAllocator alloc;
  auto t = at::detail::make_tensor<at::TensorImpl>(
      c10::DispatchKeySet(c10::DispatchKey::CPU), caffe2::TypeMeta::Make<float>(), at::Device(at::kCPU));
  ASSERT_FALSE(t.has_storage());
  {
    CPUAllocatorOverrideGuard guard(&alloc);
    t.resize_({8});
  }
  ASSERT_EQ(allocator_of(t), &alloc);
  ASSERT_EQ(alloc.allocated, Sizes({32}));
}

TEST(CPUAllocatorOverrideTest, OtherThreadsAreUnaffected) {
  CountingAllocator alloc, worker_alloc;
  CPUAllocatorOverrideGuard guard(&alloc);

  at::Allocator* worker_override = &alloc;
  at::Allocator* worker_used = nullptr;
  std::thread([&] {
    worker_override = getCPUAllocatorOverride();
    worker_used = allocator_of(at::empty({4}));
  }).join();
  ASSERT_EQ(worker_override, nullptr);
  ASSERT_EQ(worker_used, c10::GetCPUAllocator());
  ASSERT_TRUE(alloc.allocated.empty());

  std::thread([&] {
    CPUAllocatorOverrideGuard worker_guard(&worker_alloc);
    worker_used = allocator_of(at::empty({4}));
  }).join();
  ASSERT_EQ(worker_used, &worker_alloc);
  ASSERT_EQ(getCPUAllocatorOverride(), &alloc);
  ASSERT_EQ(allocator_of(at::empty({4})), &alloc);
}
