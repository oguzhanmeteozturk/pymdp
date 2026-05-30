// Resize-up-only scratch primitives shared by the kernels.
//
// Two abstractions:
//   - `ensure_at_least(std::vector<T>&, int64_t)` — grow-only resize. The
//     primitive both kernels' scratch types use today.
//   - `ScratchBuffer<T>` — thin owning wrapper exposing the same surface
//     (`.data()`, `.size()`, `.operator[]`) with built-in `.ensure(n)`.
//
// Both keep capacity across calls, so pymdp's repeated-shape call pattern
// pays zero allocations in steady state.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pymdp_ffi {

// Grow-only resize. `vector::resize` would shrink, which destroys elements
// beyond the new size; FFI scratch buffers use max-of-shapes-seen sizing,
// so growing is the only direction intended. Takes int64_t (matching the
// kernels' native size type) so callers don't decorate every call site
// with `static_cast<size_t>(...)`; non-positive n is treated as a no-op.
template <class T> inline void ensure_at_least(std::vector<T>& v, int64_t n) {
  if (n <= 0) return;
  const std::size_t target = static_cast<std::size_t>(n);
  if (v.size() < target) v.resize(target);
}

// Resize-up-only typed slot. Forwards the read interface of std::vector
// (`.data()`, `.size()`, `.operator[]`) and adds `.ensure(n)`; intended to
// drop into existing scratch structs without touching consumer call sites.
//
// Stays header-only and templated so the FFI library doesn't need a
// dedicated TU per element type.
template <class T> class ScratchBuffer {
public:
  void ensure(int64_t n) { ensure_at_least(storage_, n); }

  T*       data() noexcept { return storage_.data(); }
  const T* data() const noexcept { return storage_.data(); }

  std::size_t size() const noexcept { return storage_.size(); }
  bool        empty() const noexcept { return storage_.empty(); }

  T&       operator[](std::size_t i) noexcept { return storage_[i]; }
  const T& operator[](std::size_t i) const noexcept { return storage_[i]; }

  // Escape hatch when a caller needs the raw vector (e.g., to swap, clear,
  // or hand to APIs that take a std::vector). Use sparingly.
  std::vector<T>&       storage() noexcept { return storage_; }
  const std::vector<T>& storage() const noexcept { return storage_; }

private:
  std::vector<T> storage_;
};

}  // namespace pymdp_ffi
