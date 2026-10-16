#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <utility>
#include <vector>
#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace strata::core {

#if defined(__linux__)
// Segment storage: anonymous mappings, so an evicted conversation can go back to the kernel.  The physical-RAM
// admission gate (generate.cpp) evicts parked conversations and then reads /proc/meminfo again, which only works
// when the evicted bytes leave the process.  With glibc's malloc they did not: after the first free of a chunk
// above 128 KiB the dynamic mmap threshold rises to that chunk's size (up to 32 MiB), every later 16 MiB segment
// comes from the brk heap, and a freed one stays resident until the heap's end is trimmed.  Measured on 0.1.43
// (2x RTX 3090, Ubuntu 24.04, glibc 2.39): evicting 9.3 GiB of parked conversations gave 1.4 GiB back, the
// engine's heap held 10.6 GiB with nothing parked, and the gate refused the snapshot it had just evicted
// everything for.
//
// Fresh pages cost a fault and the kernel's zeroing each (a 3.2 GiB park: ~2.1 s fresh against ~0.6 s on pages
// the heap already held), so a freed full segment is kept, still mapped, in a small spare pool and the next
// capture takes it first: budget evictions run just before the snapshot they make room for.  The pool holds at
// most segment_pool_limit bytes and release_conversation_segments() unmaps all of it; everything that evicts to
// get physical RAM back calls it before it measures again.  Tails and other sizes are mapped and unmapped
// directly.  New mappings are populated up front (MAP_POPULATE: one call instead of a fault per page, ~25%
// faster).  No madvise / huge pages (#771).  Linux only: this is the glibc behaviour; the other platforms keep
// std::allocator and nothing changes there.
inline constexpr size_t segment_pool_block = 16 * 1024 * 1024;     // ConversationBuffer::segment_bytes
inline constexpr size_t segment_pool_limit = 4ull << 30;           // ~ one 210K-token Flash-Next snapshot
namespace detail {
// A fixed table, no heap: releasing must not allocate (session_save_reclaim runs when RAM is short), and a
// trivially destructible static stays usable while other statics free their segments at exit.
struct SegmentPool {
    static constexpr size_t slots = segment_pool_limit / segment_pool_block;
    std::mutex lock;
    void* spare[slots] = {};
    size_t count = 0;
    static SegmentPool& get() noexcept {
        static SegmentPool pool;
        return pool;
    }
    void* take() noexcept {
        std::lock_guard<std::mutex> hold(lock);
        return count ? spare[--count] : nullptr;
    }
    bool put(void* p) noexcept {
        std::lock_guard<std::mutex> hold(lock);
        if (count == slots) return false;
        spare[count++] = p;
        return true;
    }
};
} // namespace detail

template<class T> struct MappedAllocator {
    using value_type = T;
    static_assert(alignof(T) <= 4096, "a mapping is page-aligned; over-aligned types are not supported");
    MappedAllocator() noexcept = default;
    template<class U> MappedAllocator(const MappedAllocator<U>&) noexcept {}
    T* allocate(size_t n) {
        if (n > std::numeric_limits<size_t>::max() / sizeof(T)) throw std::bad_array_new_length();
        const size_t size = bytes(n);
        if (size == segment_pool_block)
            if (void* p = detail::SegmentPool::get().take()) return static_cast<T*>(p);
        void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        if (p == MAP_FAILED) throw std::bad_alloc();
        return static_cast<T*>(p);
    }
    void deallocate(T* p, size_t n) noexcept {
        const size_t size = bytes(n);
        if (size == segment_pool_block && detail::SegmentPool::get().put(p)) return;
        munmap(p, size);   // the kernel rounds the length up to pages
    }
private:
    // allocate(0) is unspecified for an allocator; one byte maps one page and keeps the pointer distinct
    static size_t bytes(size_t n) noexcept { return n ? n * sizeof(T) : 1; }
};
template<class T, class U> bool operator==(const MappedAllocator<T>&, const MappedAllocator<U>&) noexcept { return true; }
template<class T, class U> bool operator!=(const MappedAllocator<T>&, const MappedAllocator<U>&) noexcept { return false; }

// Unmaps every spare segment; returns the bytes given back to the kernel.  Allocates nothing.
inline size_t release_conversation_segments() noexcept {
    auto& pool = detail::SegmentPool::get();
    void* spare[detail::SegmentPool::slots];
    size_t n = 0;
    {
        std::lock_guard<std::mutex> hold(pool.lock);
        n = pool.count;
        std::copy(pool.spare, pool.spare + n, spare);
        pool.count = 0;
    }
    for (size_t i = 0; i < n; ++i) munmap(spare[i], segment_pool_block);
    return n * segment_pool_block;
}
#else
template<class T> using MappedAllocator = std::allocator<T>;
inline size_t release_conversation_segments() noexcept { return 0; }
#endif

// Snapshot-owned host storage. Appending never relocates existing payloads;
// only the small segment directory can move. No sharing or copy-on-write is
// needed because a parked image and its retained active buffers have one owner.
class ConversationBuffer {
public:
    static constexpr size_t segment_bytes = 16 * 1024 * 1024;
    ConversationBuffer() = default;
    ConversationBuffer(std::initializer_list<uint8_t> values) {
        resize(values.size());
        visit(0, size(), [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, values.begin() + at, n);
            return true;
        });
    }
    ConversationBuffer(const ConversationBuffer&) = default;
    ConversationBuffer& operator=(const ConversationBuffer&) = default;
    ConversationBuffer(ConversationBuffer&& other) noexcept
        : segments_(std::move(other.segments_)), size_(std::exchange(other.size_, 0)) {}
    ConversationBuffer& operator=(ConversationBuffer&& other) noexcept {
        if (this == &other) return *this;
        segments_ = std::move(other.segments_);
        size_ = std::exchange(other.size_, 0);
        return *this;
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    size_t bytes() const {
        size_t n = segments_.capacity() * sizeof(Segment);
        for (const auto& segment : segments_) n += segment.capacity();
        return n;
    }
    // Includes the old directory while reserve allocates its replacement.
    // Saturation makes an overflowing estimate fail admission before resize.
    size_t allocation_peak(size_t n) const {
        size_t total = bytes();
        if (n <= size_) return total;
        const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
        const size_t extra = n - size_ - std::min(n - size_, spare);
        const size_t full = extra / segment_bytes, tail = extra % segment_bytes;
        if (!add(total, full * segment_bytes)) return SIZE_MAX;
        if (tail && !add(total, segment_capacity(tail, full ? segment_bytes :
                segments_.empty() ? 0 : segments_.back().capacity(), size_ != 0))) return SIZE_MAX;
        size_t count = segments_.size();
        if (!add(count, full + (tail != 0))) return SIZE_MAX;
        if (count > segments_.capacity() &&
            (count > SIZE_MAX / sizeof(Segment) || !add(total, count * sizeof(Segment)))) return SIZE_MAX;
        return total;
    }
    void resize(size_t n, uint8_t value = 0) {
        if (n > size_) {
            const bool growing = size_ != 0;
            const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
            const size_t extend = std::min(n - size_, spare);
            const size_t extra = n - size_ - extend;
            segments_.reserve(segments_.size() + extra / segment_bytes + (extra % segment_bytes != 0));
            if (extend) {
                segments_.back().resize(segments_.back().size() + extend, value);
                size_ += extend;
            }
            while (size_ < n) {
                const size_t count = std::min(segment_bytes, n - size_);
                const size_t previous = segments_.empty() ? 0 : segments_.back().capacity();
                Segment segment;
                segment.reserve(segment_capacity(count, previous, growing));
                segment.resize(count, value);
                segments_.push_back(std::move(segment));
                size_ += count;
            }
        } else {
            while (!segments_.empty() && size_ - segments_.back().size() >= n) {
                size_ -= segments_.back().size();
                segments_.pop_back();
            }
            if (size_ > n) segments_.back().resize(segments_.back().size() - (size_ - n));
            size_ = n;
        }
    }
    void pop_back() { resize(size_ - 1); }

    // fn(pointer, count, absolute_offset); false aborts a transfer immediately.
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) {
        return visit_impl(*this, offset, count, fn);
    }
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) const {
        return visit_impl(*this, offset, count, fn);
    }
    bool read(void* target, size_t offset, size_t count) const {
        return visit(offset, count, [&](const uint8_t* p, size_t n, size_t at) {
            std::memcpy(static_cast<uint8_t*>(target) + at - offset, p, n);
            return true;
        });
    }
    bool operator==(const ConversationBuffer& other) const {
        if (size_ != other.size_) return false;
        size_t a = 0, b = 0, x = 0, y = 0;
        while (a < segments_.size() && b < other.segments_.size()) {
            const auto& left = segments_[a];
            const auto& right = other.segments_[b];
            const size_t n = std::min(left.size() - x, right.size() - y);
            if (std::memcmp(left.data() + x, right.data() + y, n)) return false;
            x += n; y += n;
            if (x == left.size()) { ++a; x = 0; }
            if (y == right.size()) { ++b; y = 0; }
        }
        return true;
    }

private:
    using Segment = std::vector<uint8_t, MappedAllocator<uint8_t>>;   // mapped, full ones pooled (see above)
    // A fresh capture allocates exactly its payload. Later appends reserve
    // geometrically growing segments, bounded by 16 MiB. Small chat turns then
    // extend the final allocation instead of adding one transfer per turn.
    static size_t segment_capacity(size_t count, size_t previous, bool growing) {
        if (!growing) return count;
        return std::max(count, std::min(segment_bytes, std::max<size_t>(65536, std::min(previous, segment_bytes/2)*2)));
    }
    static bool add(size_t& n, size_t extra) {
        if (extra > SIZE_MAX - n) return false;
        n += extra;
        return true;
    }
    template<class Self, class Fn> static bool visit_impl(Self& self, size_t offset, size_t count, Fn fn) {
        if (offset > self.size_ || count > self.size_ - offset) return false;
        size_t begin = 0;
        for (auto& segment : self.segments_) {
            if (!count) break;
            const size_t end = begin + segment.size();
            if (offset < end) {
                const size_t n = std::min(count, end - offset);
                if (!fn(segment.data() + offset - begin, n, offset)) return false;
                offset += n; count -= n;
            }
            begin = end;
        }
        return count == 0;
    }
    std::vector<Segment> segments_;
    size_t size_ = 0;
};

#if defined(__linux__)
static_assert(segment_pool_block == ConversationBuffer::segment_bytes, "the pool recycles full segments");
#endif

} // namespace strata::core
