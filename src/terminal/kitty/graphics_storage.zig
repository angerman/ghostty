const std = @import("std");
const assert = @import("../../quirks.zig").inlineAssert;
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

const fastmem = @import("../../fastmem.zig");
const terminal = @import("../main.zig");
const point = @import("../point.zig");
const size = @import("../size.zig");
const animation = @import("graphics_animation.zig");
const command = @import("graphics_command.zig");
const PageList = @import("../PageList.zig");
const Screen = @import("../Screen.zig");
const LoadingImage = @import("graphics_image.zig").LoadingImage;
const Image = @import("graphics_image.zig").Image;
const Rect = @import("graphics_image.zig").Rect;
const Animation = animation.Animation;
const Command = command.Command;

const log = std.log.scoped(.kitty_gfx);

/// The clock that drives animation playback, as milliseconds since an
/// epoch established the first time it is read.
///
/// Every animation timestamp must come from one of these: the times
/// stamped when a command changes the current frame, and the deadlines
/// the renderer computes, are compared directly against each other, so
/// mixing in wall-clock time or a second epoch would produce nonsense
/// deadlines. Each ImageStorage owns exactly one.
///
/// Note that needing a monotonic clock at all is why the whole protocol
/// is compiled out on freestanding targets (see terminal/build_options),
/// so std.time.Instant is always available to us here.
const AnimationClock = struct {
    epoch: ?std.time.Instant = null,
    unavailable: bool = false,

    fn nowMs(self: *AnimationClock) ?u64 {
        if (self.unavailable) return null;

        const now = std.time.Instant.now() catch {
            // Log once rather than on every frame. Animation state still
            // updates, but nothing ever advances it.
            log.warn("no monotonic clock, kitty graphics animation disabled", .{});
            self.unavailable = true;
            return null;
        };

        const epoch = self.epoch orelse epoch: {
            self.epoch = now;
            break :epoch now;
        };

        return now.since(epoch) / std.time.ns_per_ms;
    }
};

/// Process-global counter backing all generation stamps (see
/// ImageStorage.generation and Image.generation). This is global rather
/// than per-storage so that stamps are unique across every storage in
/// the process: two mutation events never produce the same value, even
/// across separate screens (main vs. alt), storage resets, or separate
/// terminals. This lets consumers use a generation value alone as a
/// cache key without any ambiguity.
///
/// Thread-safe because separate terminals may mutate their storages
/// from different threads. On single-threaded targets this lowers to
/// plain operations.
var generation_counter: GenerationCounter = .{};

/// Returns the next generation stamp. Stamps are unique and strictly
/// monotonically increasing process-wide, starting at 1 (0 is reserved
/// to mean "never stamped").
pub fn nextGeneration() u64 {
    return generation_counter.next();
}

/// Backing implementation for the generation counter. We use a
/// lock-free atomic counter where we can, but not all targets support
/// 64-bit atomic operations (e.g. 32-bit ARM Android), so we fall back
/// to a mutex-protected counter on those. This is a cold path (only
/// invoked on content mutations) so the mutex cost is irrelevant.
///
/// The pointer-width check is a conservative proxy for 64-bit atomic
/// support: every 64-bit target supports 64-bit atomics, while 32-bit
/// targets may not (per the compiler's atomic operand validation).
const GenerationCounter = if (@bitSizeOf(usize) >= 64) struct {
    value: std.atomic.Value(u64) = .init(0),

    fn next(self: *@This()) u64 {
        return self.value.fetchAdd(1, .monotonic) + 1;
    }
} else struct {
    mutex: std.Thread.Mutex = .{},
    value: u64 = 0,

    fn next(self: *@This()) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.value += 1;
        return self.value;
    }
};

/// An image storage is associated with a terminal screen (i.e. main
/// screen, alt screen) and contains all the transmitted images and
/// placements.
pub const ImageStorage = struct {
    const ImageMap = std.AutoHashMapUnmanaged(u32, Image);
    const PlacementMap = std.AutoHashMapUnmanaged(PlacementKey, Placement);

    /// Layout is dirty: the set of images or placements changed, or the
    /// geometry they sit at moved. The renderer must rebuild and re-sort
    /// its placement list. Scrolling, resizing and screen switches set
    /// this from outside this struct, because they move placement pins
    /// even though the image set itself is unchanged.
    ///
    /// This is purely informational for the renderer and doesn't affect
    /// the correctness of the program. The renderer must set this to false
    /// if it cares about this value.
    ///
    /// Invariant: layout_dirty is always set when the generation changes
    /// for a layout reason (markLayoutMutated sets both); set without a
    /// generation change it means a geometry-only event.
    layout_dirty: bool = false,

    /// Pixels are dirty: some image's currently displayed pixels changed
    /// while its dimensions stayed the same. The renderer must re-upload
    /// the affected images' textures, and must NOT rebuild placements --
    /// nothing about the geometry changed.
    ///
    /// This is the animation steady state, so it is deliberately the
    /// cheapest signal: an advancing frame sets only this.
    pixel_dirty: bool = false,

    /// The schedule is dirty: animation timing changed (play/stop, a gap,
    /// a loop count, the current frame). The renderer must recompute when
    /// its next frame is due, but has no pixel or placement work to do.
    schedule_dirty: bool = false,

    /// Generation stamp of the last content mutation to this storage:
    /// any image transmit/replace, placement add, or delete of either.
    /// Zero means the storage has never been mutated (and is therefore
    /// empty).
    ///
    /// Unlike dirty, this is NOT updated by scrolling/resizing, so an
    /// unchanged generation means the placement set and all image data
    /// are identical; only placement geometry (pins) may have moved.
    /// Values come from a process-global monotonic counter, so a value
    /// observed from any storage never recurs for different content,
    /// even across screen switches or storage resets.
    ///
    /// This field must only be written via markLayoutMutated.
    generation: u64 = 0,

    /// This is the next automatically assigned image ID. We start mid-way
    /// through the u32 range to avoid collisions with buggy programs.
    /// TODO: This isn't good enough, it's perfectly legal for programs
    ///       to use IDs in the latter half of the range and collisions
    ///       are not gracefully handled.
    next_image_id: u32 = 2147483647,

    /// This is the next automatically assigned placement ID. This is never
    /// user-facing so we can start at 0. This is 32-bits because we use
    /// the same space for external placement IDs. We can start at zero
    /// because any number is valid.
    next_internal_placement_id: u32 = 0,

    /// The set of images that are currently known.
    images: ImageMap = .{},

    /// The set of placements for loaded images.
    placements: PlacementMap = .{},

    /// Non-null if there is an in-progress loading image.
    loading: ?*LoadingImage = null,

    /// The limits of what medium types are allowed for image loading.
    image_limits: LoadingImage.Limits = .direct,

    /// The total bytes of image data that have been loaded and the limit.
    /// If the limit is reached, the oldest images will be evicted to make
    /// space. Unused images take priority.
    ///
    /// Animation frames are charged against this same limit. Kitty gives
    /// frames a separate quota five times this size, but it can spill them
    /// to a disk cache and we can't, so we deliberately keep the single
    /// in-RAM budget.
    total_bytes: usize = 0,
    total_limit: usize = 320 * 1000 * 1000, // 320MB

    /// The number of images that have animation state. This lets playback
    /// scans early-out in O(1) for the overwhelmingly common case of no
    /// animations at all, without a flag that could drift out of sync.
    animation_count: usize = 0,

    /// The clock for animation playback. Read it via animationNowMs.
    animation_clock: AnimationClock = .{},

    pub fn deinit(
        self: *ImageStorage,
        alloc: Allocator,
        s: *terminal.Screen,
    ) void {
        if (self.loading) |loading| loading.destroy(alloc);

        var it = self.images.iterator();
        while (it.next()) |kv| kv.value_ptr.deinit(alloc);
        self.images.deinit(alloc);

        self.clearPlacements(s);
        self.placements.deinit(alloc);
    }

    /// Kitty image protocol is enabled if we have a non-zero limit.
    pub fn enabled(self: *const ImageStorage) bool {
        return self.total_limit != 0;
    }

    /// Record a content mutation: marks the storage dirty and assigns a
    /// fresh generation stamp. Must be called by anything that changes
    /// the set of images or placements (or image contents).
    ///
    /// Do NOT call this for geometry-only events (scrolling, resizing,
    /// screen switches); those must set only layout_dirty directly.
    /// Bumping the generation for geometry changes would break the
    /// contract that an unchanged generation means unchanged contents.
    ///
    /// Nor for the animation domains: see markPixelsMutated and
    /// markScheduleMutated, which exist so that an advancing frame does
    /// not drag a placement rebuild along with it.
    pub fn markLayoutMutated(self: *ImageStorage) void {
        self.layout_dirty = true;
        self.generation = nextGeneration();
    }

    /// Record that a stored image's pixels changed with its dimensions
    /// unchanged.
    ///
    /// `visible` says whether the buffer that changed is the one currently
    /// on screen. Only then does the renderer have any work to do, and
    /// only then is the image's own stamp bumped: stamping it otherwise
    /// would re-upload an identical texture, and a stored-only change (a
    /// frame that isn't showing) must cost nothing until it is displayed.
    ///
    /// This never invalidates layout. An animation advancing a frame, or a
    /// client editing pixels in place, cannot move a placement.
    pub fn markPixelsMutated(
        self: *ImageStorage,
        img: *Image,
        visible: bool,
        damage: animation.Damage,
    ) void {
        self.generation = nextGeneration();
        if (!visible) return;
        img.generation = self.generation;
        self.pixel_dirty = true;

        // Accumulate rather than replace: the renderer may not have
        // uploaded the previous change yet, and it must not upload a
        // region that omits it.
        switch (damage) {
            .none => {},
            .full => img.damage.addFull(),
            .rect => |r| img.damage.add(r),
        }
    }

    /// Record that animation timing changed. This has no pixel or
    /// placement consequences: the renderer only needs to work out when it
    /// is next due.
    pub fn markScheduleMutated(self: *ImageStorage) void {
        self.schedule_dirty = true;
    }

    /// Sets the limit in bytes for the total amount of image data that
    /// can be loaded. If this limit is lower, this will do an eviction
    /// if necessary. If the value is zero, then Kitty image protocol will
    /// be disabled.
    pub fn setLimit(
        self: *ImageStorage,
        alloc: Allocator,
        s: *terminal.Screen,
        limit: usize,
    ) !void {
        // Special case disabling by quickly deleting all
        if (limit == 0) {
            const image_limits = self.image_limits;
            self.deinit(alloc, s);
            self.* = .{ .image_limits = image_limits };
            self.markLayoutMutated();
        }

        // If we re lowering our limit, check if we need to evict.
        if (limit < self.total_bytes) {
            const req_bytes = self.total_bytes - limit;
            log.info("evicting images to lower limit, evicting={}", .{req_bytes});
            if (!try self.evictImage(alloc, s, 0, req_bytes)) {
                log.warn("failed to evict enough images for required bytes", .{});
            }
        }

        self.total_limit = limit;
    }

    /// Add an already-loaded image to the storage. This will automatically
    /// free any existing image with the same ID.
    pub fn addImage(
        self: *ImageStorage,
        alloc: Allocator,
        s: *terminal.Screen,
        img: Image,
    ) Allocator.Error!void {
        // If the image itself is over the limit, then error immediately
        if (img.byteSize() > self.total_limit) return error.OutOfMemory;

        // Replacing an image frees everything the old one owned, so only
        // the *net* growth has to be found. Charging the full new size
        // would evict unrelated images to make room for bytes we are about
        // to release, and the overcount is severe when the image being
        // replaced carries a long animation.
        const replaced: usize = if (self.images.get(img.id)) |old|
            old.byteSize()
        else
            0;
        const growth = img.byteSize() -| replaced;

        // Plan the eviction before touching anything, and keep the
        // replacement target out of the plan: it is often the oldest and
        // least-used candidate, so evicting it to make room for its own
        // replacement would silently drop its placements.
        var reservation = self.prepareReservation(alloc, growth, img.id) catch |err| switch (err) {
            // addImage's callers only distinguish "it didn't fit".
            error.OutOfSpace, error.OutOfMemory => return error.OutOfMemory,
            else => unreachable,
        };
        defer reservation.deinit(alloc);

        // Reserve the map slot so the commit below cannot fail partway.
        try self.images.ensureUnusedCapacity(alloc, 1);

        // --- Commit. Nothing from here on may fail. ---

        reservation.commit(self, alloc, s);
        const gop = self.images.getOrPutAssumeCapacity(img.id);

        log.debug("addImage image={}", .{img: {
            var copy = img;
            copy.data = "";
            break :img copy;
        }});

        // Write our new image
        if (gop.found_existing) {
            // Retransmitting an ID replaces the image wholesale, which
            // drops any animation it had (both the frames and the playback
            // state), matching Kitty.
            self.total_bytes -= gop.value_ptr.byteSize();
            if (gop.value_ptr.anim != null) self.animation_count -= 1;
            gop.value_ptr.deinit(alloc);
        }

        gop.value_ptr.* = img;
        self.total_bytes += img.byteSize();

        // Stamp the stored image with a fresh generation. This gives
        // every add/replace a unique stamp even when the same image ID
        // is retransmitted with identical dimensions, so consumers
        // (e.g. renderer texture caches) can detect content changes.
        self.markLayoutMutated();
        gop.value_ptr.generation = self.generation;
    }

    /// Add a placement for a given image. The caller must verify in advance
    /// the image exists to prevent memory corruption.
    pub fn addPlacement(
        self: *ImageStorage,
        alloc: Allocator,
        image_id: u32,
        placement_id: u32,
        p: Placement,
    ) !void {
        assert(self.images.get(image_id) != null);
        log.debug("placement image_id={} placement_id={} placement={}\n", .{
            image_id,
            placement_id,
            p,
        });

        // The important piece here is that the placement ID needs to
        // be marked internal if it is zero. This allows multiple placements
        // to be added for the same image. If it is non-zero, then it is
        // an external placement ID and we can only have one placement
        // per (image id, placement id) pair.
        const key: PlacementKey = .{
            .image_id = image_id,
            .placement_id = if (placement_id == 0) .{
                .tag = .internal,
                .id = id: {
                    defer self.next_internal_placement_id +%= 1;
                    break :id self.next_internal_placement_id;
                },
            } else .{
                .tag = .external,
                .id = placement_id,
            },
        };

        const gop = try self.placements.getOrPut(alloc, key);
        gop.value_ptr.* = p;

        self.markLayoutMutated();
    }

    fn clearPlacements(self: *ImageStorage, s: *terminal.Screen) void {
        var it = self.placements.iterator();
        while (it.next()) |entry| entry.value_ptr.deinit(s);
        self.placements.clearRetainingCapacity();
    }

    /// Get an image by its ID. If the image doesn't exist, null is returned.
    pub fn imageById(self: *const ImageStorage, image_id: u32) ?Image {
        return self.images.get(image_id);
    }

    /// Get an image by its number. If the image doesn't exist, return null.
    pub fn imageByNumber(self: *const ImageStorage, image_number: u32) ?Image {
        var newest: ?Image = null;

        var it = self.images.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.number == image_number) {
                if (newest == null or
                    kv.value_ptr.generation > newest.?.generation)
                {
                    newest = kv.value_ptr.*;
                }
            }
        }

        return newest;
    }

    /// The current animation time in milliseconds, or null if this target
    /// has no monotonic clock and animation therefore never advances.
    ///
    /// Both command execution and the renderer must timestamp through this
    /// one function, while holding the terminal state mutex, so that every
    /// animation timestamp shares a single clock domain.
    pub fn animationNowMs(self: *ImageStorage) ?u64 {
        return self.animation_clock.nowMs();
    }

    /// How an animation command's "i"/"I" identifiers resolved. Every
    /// animation action identifies its image the same way, but they don't
    /// all report a failure the same way, so this reports what happened
    /// and lets the caller decide.
    pub const ResolvedImage = union(enum) {
        /// The image exists. Mutation must go through this pointer, never
        /// through an Image copied out of the map by value.
        found: *Image,

        /// Both "i" and "I" were given, which is invalid.
        conflict,

        /// Neither "i" nor "I" was given.
        no_identifier,

        /// An identifier was given but names no image.
        not_found,
    };

    /// Resolve the image an animation command names.
    pub fn resolveAnimationImagePtr(
        self: *ImageStorage,
        image_id: u32,
        image_number: u32,
    ) ResolvedImage {
        if (image_id > 0 and image_number > 0) return .conflict;
        if (image_id == 0 and image_number == 0) return .no_identifier;

        const id = if (image_id > 0) image_id else id: {
            const img = self.imageByNumber(image_number) orelse return .not_found;
            break :id img.id;
        };

        return .{ .found = self.images.getPtr(id) orelse return .not_found };
    }

    /// The outcome of an animation tick.
    pub const TickResult = struct {
        /// True if any image's visible pixels changed, meaning the
        /// renderer has to re-upload and redraw.
        dirtied: bool = false,

        /// When the next frame is due, on the same clock as the tick's
        /// now_ms, or null if nothing is waiting to be shown. Callers must
        /// use this even when nothing was dirtied, or an animation that
        /// isn't due yet would never be looked at again.
        next_due_ms: ?u64 = null,
    };

    /// True if this image's animation could show a new frame at some
    /// point. This is Kitty's animatable check, and the reasons to say no
    /// are all cheap to test.
    fn animatable(_: *const ImageStorage, img: *const Image) bool {
        const anim = img.anim orelse return false;
        if (anim.state == .stopped) return false;

        // Nothing to advance through.
        if (anim.frames.items.len == 0) return false;

        // Every frame is gapless, so there is no frame to stop on. This
        // also guards advance() against spinning forever.
        if (anim.nonzero_gap_count == 0) return false;

        // The animation has played out its loops.
        if (anim.max_loops != 0 and anim.current_loop >= anim.max_loops) return false;

        // Only what the renderer actually draws is scheduled. This is a
        // field read, so the scan costs O(animated images) rather than
        // O(animated images x placements), and an image that is stored but
        // off screen keeps no timer armed and wakes nobody.
        return img.drawn;
    }

    /// Record which images the renderer draws.
    ///
    /// Called by the renderer during layout synchronization while it holds
    /// the terminal state lock. Scheduling reads the result, so an image
    /// that scrolls out of the viewport stops being scheduled at the next
    /// settling layout update, and one that scrolls back in resumes.
    pub fn setDrawn(self: *ImageStorage, drawn: anytype) void {
        var it = self.images.iterator();
        while (it.next()) |kv| kv.value_ptr.drawn = drawn.contains(kv.key_ptr.*);
    }

    /// Advance any animations that are due, and report when the next one
    /// is. This is Kitty's scan_active_animations.
    ///
    /// Note that this mutates terminal state, so the renderer may only
    /// call it while holding the state mutex.
    pub fn animationTick(self: *ImageStorage, now_ms: u64) TickResult {
        var result: TickResult = .{};

        // The overwhelmingly common case: nothing here animates.
        if (self.animation_count == 0) return result;

        var it = self.images.iterator();
        while (it.next()) |kv| {
            const img = kv.value_ptr;
            if (!self.animatable(img)) continue;
            const anim = img.anim.?;

            var next_at = anim.last_frame_ms +| anim.gapOf(anim.current_frame);
            if (now_ms >= next_at) {
                // Only ever advance one frame per tick: Kitty doesn't try
                // to catch up on missed frames either, it just lets the
                // timing drift under load.
                if (!anim.advance()) continue;

                anim.last_frame_ms = now_ms;

                // The heart of the fast path: advancing a frame changes
                // pixels only. It must never invalidate layout, or every
                // tick would rebuild and re-sort every placement.
                // A switch between two materialized frames changes
                // everything: nothing here knows how they differ. The
                // tiled-surface work would derive this from tile
                // identity; until then it is a full upload.
                self.markPixelsMutated(img, true, .full);
                result.dirtied = true;

                next_at = now_ms +| anim.gapOf(anim.current_frame);
            }

            if (next_at > now_ms) {
                result.next_due_ms = @min(result.next_due_ms orelse next_at, next_at);
            }
        }

        return result;
    }

    /// When the next animation frame is due, without advancing anything.
    /// The renderer uses this to arm its timer.
    ///
    /// The deadline is absolute, on the same clock as animationNowMs, and
    /// may be in the past if an animation is already overdue; the caller
    /// clamps it against its own idea of now.
    pub fn nextAnimationDeadline(self: *const ImageStorage) ?u64 {
        if (self.animation_count == 0) return null;

        var next: ?u64 = null;
        var it = self.images.iterator();
        while (it.next()) |kv| {
            const img = kv.value_ptr;
            if (!self.animatable(img)) continue;
            const anim = img.anim.?;

            const due = anim.last_frame_ms +| anim.gapOf(anim.current_frame);
            next = @min(next orelse due, due);
        }

        return next;
    }

    /// The RGBA canvas size of an image, with checked arithmetic.
    ///
    /// Dimensions are already bounded when an image is loaded, but the
    /// product is computed on every frame operation and a wrap here would
    /// undersize a buffer that pixel loops then run past.
    fn canvasLen(width: u32, height: u32) error{OutOfSpace}!usize {
        const px = std.math.mul(usize, width, height) catch return error.OutOfSpace;
        return std.math.mul(usize, px, 4) catch return error.OutOfSpace;
    }

    /// The most transient memory one frame operation may allocate on top of
    /// what it persists.
    ///
    /// The persistent quota alone is not enough protection. A transactional
    /// edit composes into a fresh canvas (and sometimes a freshly widened
    /// root) before swapping it in, so it can allocate two full canvases
    /// while its *persistent* delta is zero -- nothing else would bound it,
    /// and peak RSS could reach several times the nominal quota. Deriving
    /// the transient bound from the persistent limit means configuring one
    /// configures both.
    fn transientLimit(self: *const ImageStorage) usize {
        return self.total_limit;
    }

    /// Reject an operation whose transient allocations would exceed the
    /// transient budget, before any of them are made.
    fn checkTransient(self: *const ImageStorage, bytes: usize) AnimationError!void {
        if (bytes > self.transientLimit()) {
            log.warn("kitty animation operation needs {} transient bytes, limit is {}", .{
                bytes,
                self.transientLimit(),
            });
            return error.OutOfSpace;
        }
    }

    /// Errors from the animation frame operations below. graphics_exec.zig
    /// maps these onto the protocol's error responses.
    pub const AnimationError = error{
        /// We can't make room for the frame within the storage limit.
        OutOfSpace,
        /// The base frame for a new frame doesn't exist.
        BaseFrameNotFound,
        /// A source or destination frame doesn't exist.
        FrameNotFound,
        /// A rectangle doesn't fit in the image, or a same-frame
        /// composition would overlap itself.
        InvalidRect,
    } || Allocator.Error;

    /// A planned eviction that would make room for some number of bytes.
    ///
    /// Preparing and committing are separate so that an animation frame
    /// operation can do every fallible step -- planning the eviction,
    /// allocating buffers, growing the frame list -- before anything
    /// becomes visible. Committing then cannot fail, so the operation as a
    /// whole is atomic: it either happens completely, or it leaves the
    /// storage entirely untouched.
    ///
    /// A plan is only valid while the terminal state mutex is held, since
    /// that is what keeps the candidate set stable underneath it.
    const Reservation = struct {
        /// The images to evict, best candidate first. Empty when the new
        /// bytes already fit and nothing has to go.
        victims: []const EvictionCandidate = &.{},

        /// The number of bytes the eviction has to free.
        required: usize = 0,

        fn deinit(self: *const Reservation, alloc: Allocator) void {
            alloc.free(self.victims);
        }

        /// Evict the planned images. This cannot fail.
        fn commit(
            self: *const Reservation,
            storage: *ImageStorage,
            alloc: Allocator,
            s: *terminal.Screen,
        ) void {
            var evicted: usize = 0;
            for (self.victims) |c| {
                _ = storage.removePlacementsForImage(s, c.id);

                if (storage.images.getEntry(c.id)) |entry| {
                    log.info("evicting image id={} bytes={}", .{ c.id, c.bytes });
                    evicted += storage.removeImage(alloc, entry);
                    storage.markLayoutMutated();
                }

                if (evicted >= self.required) break;
            }
        }
    };

    /// Plan how to fit `delta` more bytes of image data, evicting images
    /// other than `exclude_image_id` if necessary. Does not mutate
    /// anything; see Reservation.
    ///
    /// Returns OutOfSpace if the other images could not free enough, in
    /// which case the storage is left completely unchanged.
    fn prepareReservation(
        self: *const ImageStorage,
        alloc: Allocator,
        delta: usize,
        exclude_image_id: u32,
    ) AnimationError!Reservation {
        // A frame canvas can be hundreds of megabytes, so this is checked.
        const total = std.math.add(usize, self.total_bytes, delta) catch
            return error.OutOfSpace;
        if (total <= self.total_limit) return .{};

        const required = total - self.total_limit;

        const candidates = try self.evictionCandidates(alloc, exclude_image_id);
        errdefer alloc.free(candidates);

        // Take candidates until they cover what we need. Freeing exactly
        // the required bytes is success, not failure.
        var evictable: usize = 0;
        var count: usize = 0;
        for (candidates) |c| {
            evictable += c.bytes;
            count += 1;
            if (evictable >= required) break;
        }

        if (evictable < required) {
            alloc.free(candidates);
            return error.OutOfSpace;
        }

        return .{
            .victims = try alloc.realloc(candidates, count),
            .required = required,
        };
    }

    /// The outcome of storing an animation frame.
    pub const FrameResult = struct {
        /// The 1-based frame number that was created or edited. The
        /// protocol requires reporting this back, and it can't be derived
        /// from the request because "r" may have been clamped or appended.
        frame: u32,

        /// True if the pixels currently on screen changed, i.e. the frame
        /// we touched happens to be the one being displayed.
        visible: bool,
    };

    /// Store an animation frame, per "a=f".
    ///
    /// `src` is the transmitted rectangle's pixels, already decompressed
    /// and PNG-decoded, `rect_width` x `rect_height` in `src_format`. It
    /// is composed onto either a newly created frame or an existing one,
    /// per the rules in `params`.
    ///
    /// This is transactional: on any error nothing at all changes, and on
    /// success the caller only has to deal with the response.
    pub fn addAnimationFrame(
        self: *ImageStorage,
        alloc: Allocator,
        s: *terminal.Screen,
        image_id: u32,
        params: command.AnimationFrameLoading,
        src: []const u8,
        src_format: command.Transmission.Format,
        rect_width: u32,
        rect_height: u32,
        now_ms: u64,
    ) AnimationError!FrameResult {
        const img = self.images.getPtr(image_id).?;

        // The transmitted rectangle has to fit the image. Note that unlike
        // "a=c" the offset is not checked: an off-canvas rectangle simply
        // clips, and may clip away entirely.
        if (rect_width > img.width or rect_height > img.height) {
            return error.InvalidRect;
        }

        const canvas_len = try canvasLen(img.width, img.height);
        const frame_count: u32 = @intCast(if (img.anim) |a| a.frameCount() else 1);

        // Kitty clamps an out-of-range or absent "r" to one past the last
        // frame, which means "append a new frame".
        const is_new = params.edit_frame == 0 or params.edit_frame > frame_count;
        const frame: u32 = if (is_new) frame_count + 1 else params.edit_frame;

        // The base canvas for a new frame must exist if one was named.
        if (is_new and params.create_frame > 0 and
            params.create_frame > frame_count)
        {
            return error.BaseFrameNotFound;
        }

        // Composition is RGBA-on-RGBA, so the root is widened when this
        // operation actually involves it: as the frame being edited, or as
        // the base a new frame is copied from. An operation that touches
        // neither -- a new frame from a background, or from another frame
        // -- must leave an RGB root alone rather than growing it to RGBA
        // for nothing. Editing the root also needs a fresh buffer even if
        // it already is RGBA, so a failure can't leave it half-composed.
        const root_is_target = !is_new and frame == 1;
        const root_is_base = is_new and params.create_frame == 1;

        // An edit composes into a copy so that a failure part way through
        // cannot leave a frame half written. But when nothing *can* fail,
        // that copy buys nothing and costs a full canvas allocated,
        // copied and freed for every frame -- at 1080p, most of a frame
        // budget, which is exactly the live VNC/video case.
        //
        // Nothing can fail once the target already is RGBA (no widening
        // to allocate) and the source already is RGBA (borrowed, not
        // converted), because an edit persists no extra bytes and so can
        // neither evict nor grow the frame list.
        const target_is_rgba = if (root_is_target) img.format == .rgba else true;
        const in_place = !is_new and target_is_rgba and src_format == .rgba;

        const need_root = !in_place and
            (root_is_target or (root_is_base and img.format != .rgba));
        const root_growth: usize = if (need_root) canvas_len - img.data.len else 0;

        // Preflight the transient cost before allocating any of it: the
        // widened root and the composed canvas are both held at once, and
        // an edit's persistent delta is zero so the quota below would not
        // bound them. The transmitted rectangle is only copied when it
        // needs widening.
        const rect_rgba: usize = if (animation.formatIsOpaque(src_format) or src_format != .rgba)
            try canvasLen(rect_width, rect_height)
        else
            0;
        const new_root_len: usize = if (need_root) canvas_len else 0;
        const target_len: usize = if (root_is_target or in_place) 0 else canvas_len;
        try self.checkTransient(std.math.add(usize, new_root_len, target_len) catch
            return error.OutOfSpace);
        try self.checkTransient(rect_rgba);

        // Reserve space before touching anything. Only a new frame and a
        // widened root grow the persistent total; replacing an existing
        // frame with a same-sized canvas doesn't.
        const delta = std.math.add(usize, root_growth, if (is_new) canvas_len else 0) catch
            return error.OutOfSpace;
        var reservation = try self.prepareReservation(alloc, delta, image_id);
        defer reservation.deinit(alloc);

        // --- Fallible work. Nothing below is visible until we commit. ---

        const new_root: ?[]u8 = if (need_root)
            try animation.allocRGBA(alloc, img.data, img.format)
        else
            null;
        errdefer if (new_root) |r| alloc.free(r);

        // The composed pixels of a 1-based frame, preferring the widened
        // root we're about to install so the base is always RGBA.
        const frameData = struct {
            fn f(i: *const Image, root: ?[]u8, n: u32) []const u8 {
                if (n == 1) return if (root) |r| r else i.data;
                return i.anim.?.frames.items[n - 2].data;
            }
        }.f;

        // The buffer we compose into: a new canvas, or a copy of the frame
        // being edited. Editing the root composes into its widened copy.
        const target: []u8 = target: {
            // Compose straight into the stored pixels when nothing can
            // fail; there is nothing to roll back from.
            if (in_place) break :target if (frame == 1)
                @constCast(img.data)
            else
                img.anim.?.frames.items[frame - 2].data;

            if (root_is_target) break :target new_root.?;

            const buf = try alloc.alloc(u8, canvas_len);
            errdefer alloc.free(buf);

            if (is_new) {
                // A new frame starts as a copy of its base frame, or as a
                // flat fill of the background color if it has no base.
                if (params.create_frame > 0) {
                    fastmem.copy(u8, buf, frameData(img, new_root, params.create_frame));
                } else {
                    animation.fill(buf, params.background);
                }
            } else {
                fastmem.copy(u8, buf, img.anim.?.frames.items[frame - 2].data);
            }

            break :target buf;
        };
        errdefer if (!root_is_target and !in_place) alloc.free(target);

        // View the transmitted rectangle as RGBA and compose it. Data
        // that already is RGBA is borrowed rather than duplicated, which
        // is every frame of a client streaming RGBA video.
        const src_rgba = try animation.rgbaView(alloc, src, src_format);
        defer src_rgba.deinit(alloc);

        // A source that cannot be translucent composes identically under
        // alpha blending and overwrite, so take the row-copy path. Kitty
        // makes the same call (`is_opaque = data_fmt == RGB`).
        const mode: command.CompositionMode = if (animation.formatIsOpaque(src_format))
            .overwrite
        else
            params.composition_mode;

        animation.composeTransmitted(
            target,
            img.width,
            img.height,
            src_rgba.data,
            rect_width,
            rect_height,
            params.x,
            params.y,
            mode,
        );

        const anim_is_new = img.anim == null;
        const anim: *Animation = img.anim orelse try alloc.create(Animation);
        errdefer if (anim_is_new) alloc.destroy(anim);
        if (anim_is_new) anim.* = .{};

        // Grow the frame list up front so appending below cannot fail.
        if (is_new) try anim.frames.ensureUnusedCapacity(alloc, 1);

        // --- Commit. Nothing from here on may fail. ---

        reservation.commit(self, alloc, s);

        if (anim_is_new) {
            img.anim = anim;
            self.animation_count += 1;
        }

        if (new_root) |root| {
            self.total_bytes = self.total_bytes - img.data.len + root.len;
            alloc.free(img.data);
            img.data = root;
            img.format = .rgba;
        }

        if (is_new) {
            // A new frame is never the current one, so it can't be
            // visible however it was composed.
            anim.appendFrameAssumeCapacity(.{
                .data = target,
                .gap_ms = animation.resolveNewGap(params.gap),
            });
            self.total_bytes += target.len;
        } else if (in_place) {
            // Already composed into the stored pixels: nothing to swap in.
        } else if (!root_is_target) {
            const old = anim.frames.items[frame - 2].data;
            assert(old.len == target.len);
            alloc.free(old);
            anim.frames.items[frame - 2].data = target;
        }

        // A zero gap on an edit means "leave it alone".
        if (!is_new and params.gap != 0) {
            anim.setGap(frame - 1, animation.resolveGap(params.gap));
        }

        // Frame pixels changed, never geometry: an added or edited frame
        // has the image's dimensions, so this must not invalidate layout.
        // Only a change to the frame on screen costs the renderer an
        // upload; a stored-only change costs nothing until it displays.
        const visible = !is_new and frame - 1 == anim.current_frame;
        self.markPixelsMutated(img, visible, damage: {
            // We know exactly what we composed: the transmitted rectangle
            // at its destination, clipped the same way composition
            // clipped it. This is the case a VNC or video client hits on
            // every frame, so uploading the whole canvas would throw away
            // the very thing that makes those clients cheap.
            if (params.x >= img.width or params.y >= img.height) break :damage .none;
            break :damage .{ .rect = .{
                .x = params.x,
                .y = params.y,
                .width = @min(rect_width, img.width - params.x),
                .height = @min(rect_height, img.height - params.y),
            } };
        });
        if (visible) {
            // Editing the visible frame restarts its gap interval, so it
            // stays up for the full new gap rather than a leftover slice.
            anim.last_frame_ms = now_ms;
            self.markScheduleMutated();
        }

        return .{ .frame = frame, .visible = visible };
    }

    /// Compose one frame's rectangle onto another, per "a=c".
    ///
    /// Transactional in the same way as addAnimationFrame. Returns true if
    /// the composition changed the pixels currently on screen.
    pub fn composeAnimationFrames(
        self: *ImageStorage,
        alloc: Allocator,
        s: *terminal.Screen,
        image_id: u32,
        params: command.AnimationFrameComposition,
        now_ms: u64,
    ) AnimationError!bool {
        const img = self.images.getPtr(image_id).?;

        // Remember that the names lie: "r" is the source, "c" is the
        // destination, X/Y offset into the source and x/y into the dest.
        const src_frame = params.edit_frame;
        const dst_frame = params.frame;
        const frame_count: u32 = @intCast(if (img.anim) |a| a.frameCount() else 1);
        if (src_frame == 0 or src_frame > frame_count) return error.FrameNotFound;
        if (dst_frame == 0 or dst_frame > frame_count) return error.FrameNotFound;

        // An absent width/height means the whole image.
        const w = if (params.width == 0) img.width else params.width;
        const h = if (params.height == 0) img.height else params.height;

        // Both rectangles must fit inside the image.
        if (@as(u64, params.left_edge) + w > img.width or
            @as(u64, params.top_edge) + h > img.height or
            @as(u64, params.x) + w > img.width or
            @as(u64, params.y) + h > img.height)
        {
            return error.InvalidRect;
        }

        // Composing a frame onto itself is only meaningful if the source
        // and destination rectangles are disjoint.
        if (src_frame == dst_frame and animation.rectsOverlap(
            params.left_edge,
            params.top_edge,
            params.x,
            params.y,
            w,
            h,
        )) {
            return error.InvalidRect;
        }

        const canvas_len = try canvasLen(img.width, img.height);

        // As in addAnimationFrame, all composition is RGBA-on-RGBA, and
        // the destination is composed as a fresh buffer so that a failure
        // can't leave a frame half written. When the destination is the
        // root frame, that fresh buffer is also its widened copy.
        // As above: widen the root only when this composition reads or
        // writes it.
        const root_is_target = dst_frame == 1;
        const root_is_source = src_frame == 1;
        const need_root = root_is_target or (root_is_source and img.format != .rgba);
        const root_growth: usize = if (need_root and img.format != .rgba)
            canvas_len - img.data.len
        else
            0;
        // Preflight the transient cost, as in addAnimationFrame.
        try self.checkTransient(std.math.add(
            usize,
            if (need_root) canvas_len else 0,
            if (root_is_target) 0 else canvas_len,
        ) catch return error.OutOfSpace);

        var reservation = try self.prepareReservation(alloc, root_growth, image_id);
        defer reservation.deinit(alloc);

        // --- Fallible work. Nothing below is visible until we commit. ---

        const new_root: ?[]u8 = if (need_root)
            try animation.allocRGBA(alloc, img.data, img.format)
        else
            null;
        errdefer if (new_root) |r| alloc.free(r);

        // The composed pixels of a 1-based frame, preferring the widened
        // root we're about to install so the source is always RGBA.
        const frameData = struct {
            fn f(i: *const Image, root: ?[]u8, n: u32) []const u8 {
                if (n == 1) return if (root) |r| r else i.data;
                return i.anim.?.frames.items[n - 2].data;
            }
        }.f;

        const target: []u8 = target: {
            if (root_is_target) break :target new_root.?;
            const buf = try alloc.alloc(u8, canvas_len);
            errdefer alloc.free(buf);
            fastmem.copy(u8, buf, img.anim.?.frames.items[dst_frame - 2].data);
            break :target buf;
        };
        errdefer if (!root_is_target) alloc.free(target);

        // Note that composing a frame onto itself reads and writes the
        // same buffer, which is safe only because the rectangles were
        // checked to be disjoint above.
        animation.composeFrames(
            target,
            frameData(img, new_root, src_frame),
            img.width,
            w,
            h,
            params.left_edge,
            params.top_edge,
            params.x,
            params.y,
            params.composition_mode,
        );

        // --- Commit. Nothing from here on may fail. ---

        reservation.commit(self, alloc, s);

        if (new_root) |root| {
            self.total_bytes = self.total_bytes - img.data.len + root.len;
            alloc.free(img.data);
            img.data = root;
            img.format = .rgba;
        }

        if (!root_is_target) {
            const old = img.anim.?.frames.items[dst_frame - 2].data;
            assert(old.len == target.len);
            alloc.free(old);
            img.anim.?.frames.items[dst_frame - 2].data = target;
        }

        // Note that an image with no animation state at all can only be
        // composing its root onto itself, which is always what's on
        // screen. There's nothing to play, so no state is created for it.
        const visible = if (img.anim) |anim|
            dst_frame - 1 == anim.current_frame
        else
            true;
        self.markPixelsMutated(img, visible, .{
            .rect = .{
                // The destination rectangle is validated to be inside the
                // image, so it is exactly the damage.
                .x = params.x,
                .y = params.y,
                .width = w,
                .height = h,
            },
        });
        if (visible) {
            if (img.anim) |anim| {
                anim.last_frame_ms = now_ms;
                self.markScheduleMutated();
            }
        }

        return visible;
    }

    /// The outcome of a frame deletion. Deleting never responds on
    /// success or when the image is simply missing, so the only thing the
    /// caller needs to distinguish is a malformed command.
    pub const FrameDeleteResult = enum {
        changed,
        no_op_or_missing,
        invalid_identifiers,
    };

    /// Delete a single animation frame, per "d=f" / "d=F".
    ///
    /// This isn't in the published spec at all; it follows Kitty's
    /// implementation. Deleting the root frame promotes the next frame
    /// into its place, and the uppercase form deletes the whole image if
    /// there are no frames to delete.
    pub fn deleteAnimationFrame(
        self: *ImageStorage,
        alloc: Allocator,
        t: *terminal.Terminal,
        v: @FieldType(command.Delete, "animation_frames"),
        now_ms: u64,
    ) FrameDeleteResult {
        // Unlike the other animation actions, a delete that names no
        // image only logs: it never sends a response.
        const img = switch (self.resolveAnimationImagePtr(v.image_id, v.image_number)) {
            .found => |img| img,
            .conflict => return .invalid_identifiers,
            .no_identifier => {
                log.warn("delete animation frame requires an image id or number", .{});
                return .no_op_or_missing;
            },
            .not_found => {
                log.warn("delete animation frame: no image id={} number={}", .{
                    v.image_id,
                    v.image_number,
                });
                return .no_op_or_missing;
            },
        };
        const image_id = img.id;

        // With no frames to delete, the uppercase form deletes the whole
        // image and the lowercase form does nothing.
        const frame_count: usize = if (img.anim) |a| a.frameCount() else 1;
        if (frame_count == 1) {
            if (!v.delete) return .no_op_or_missing;
            self.deleteById(alloc, t.screens.active, image_id, 0, true);
            self.markLayoutMutated();
            return .changed;
        }

        const anim = img.anim.?;

        // The frame number defaults to the first and is clamped into range.
        const frame: usize = if (v.frame == 0) 1 else @min(v.frame, frame_count);

        // Compare the buffer on screen before and after rather than
        // reasoning about indices: promoting a frame to root can leave the
        // very same pixels displayed even though the index changed. No
        // allocation happens in between, so the "before" address can't be
        // reused and a match really does mean nothing moved.
        const before = @intFromPtr(img.renderData().ptr);

        // The 0-based slot in `frames` that goes away. Deleting the root
        // consumes frame 2, so its slot is the one removed.
        const removed: usize = if (frame == 1) 0 else frame - 2;

        if (frame == 1) {
            // Frame 2 becomes the new root. We take over its buffer, so
            // only the old root's bytes go away.
            self.total_bytes -= img.data.len;
            alloc.free(img.data);
            img.data = anim.frames.items[0].data;
            anim.setGap(0, anim.frames.items[0].gap_ms);
            _ = anim.removeFrame(0);
        } else {
            const dead = anim.removeFrame(removed);
            self.total_bytes -= dead.data.len;
            alloc.free(dead.data);
        }

        // Kitty's rules for where the current frame lands.
        const len = anim.frames.items.len;
        if (anim.current_frame > len) {
            anim.current_frame = @intCast(len);
        } else if (removed < anim.current_frame) {
            anim.current_frame -= 1;
        }

        // The stored frames changed either way, but dimensions cannot
        // have, so this is the pixel domain. Only stamp the image when
        // what is actually on screen changed.
        const changed = before != @intFromPtr(img.renderData().ptr);
        self.markPixelsMutated(img, changed, .full);
        if (changed) anim.last_frame_ms = now_ms;
        self.markScheduleMutated();

        return .changed;
    }

    /// Delete placements, images.
    pub fn delete(
        self: *ImageStorage,
        alloc: Allocator,
        t: *terminal.Terminal,
        cmd: command.Delete,
    ) void {
        // Deletes only ever remove placements/images, so comparing counts
        // before and after tells us whether anything actually changed.
        // Only then do we mark a mutation. This matters because a
        // delete-all runs on every screen clear (e.g. `ESC [ 2 J`), and
        // we don't want empty clears to dirty the image state or bump
        // the generation.
        const placements_before = self.placements.count();
        const images_before = self.images.count();
        defer if (self.placements.count() != placements_before or
            self.images.count() != images_before) self.markLayoutMutated();

        switch (cmd) {
            .all => |delete_images| {
                var it = self.placements.iterator();
                while (it.next()) |entry| {
                    // Skip virtual placements
                    switch (entry.value_ptr.location) {
                        .pin => {},
                        .virtual => continue,
                    }

                    // Deinit the placement and remove it
                    const image_id = entry.key_ptr.image_id;
                    entry.value_ptr.deinit(t.screens.active);
                    self.placements.removeByPtr(entry.key_ptr);
                    if (delete_images) self.deleteIfUnused(alloc, image_id);
                }

                if (delete_images) {
                    var image_it = self.images.iterator();
                    while (image_it.next()) |kv| self.deleteIfUnused(alloc, kv.key_ptr.*);
                }
            },

            .id => |v| self.deleteById(
                alloc,
                t.screens.active,
                v.image_id,
                v.placement_id,
                v.delete,
            ),

            .newest => |v| newest: {
                const img = self.imageByNumber(v.image_number) orelse break :newest;
                self.deleteById(
                    alloc,
                    t.screens.active,
                    img.id,
                    v.placement_id,
                    v.delete,
                );
            },

            .intersect_cursor => |delete_images| {
                self.deleteIntersecting(
                    alloc,
                    t,
                    .{ .active = .{
                        .x = t.screens.active.cursor.x,
                        .y = t.screens.active.cursor.y,
                    } },
                    delete_images,
                    {},
                    null,
                );
            },

            .intersect_cell => |v| intersect_cell: {
                if (v.x <= 0 or v.y <= 0) {
                    log.warn("delete intersect cell coords must be at least 1", .{});
                    break :intersect_cell;
                }

                self.deleteIntersecting(
                    alloc,
                    t,
                    .{ .active = .{
                        .x = std.math.cast(size.CellCountInt, v.x - 1) orelse break :intersect_cell,
                        .y = std.math.cast(size.CellCountInt, v.y - 1) orelse break :intersect_cell,
                    } },
                    v.delete,
                    {},
                    null,
                );
            },

            .intersect_cell_z => |v| intersect_cell_z: {
                if (v.x <= 0 or v.y <= 0) {
                    log.warn("delete intersect cell coords must be at least 1", .{});
                    break :intersect_cell_z;
                }

                self.deleteIntersecting(
                    alloc,
                    t,
                    .{ .active = .{
                        .x = std.math.cast(size.CellCountInt, v.x - 1) orelse break :intersect_cell_z,
                        .y = std.math.cast(size.CellCountInt, v.y - 1) orelse break :intersect_cell_z,
                    } },
                    v.delete,
                    v.z,
                    struct {
                        fn filter(ctx: i32, p: Placement) bool {
                            return p.z == ctx;
                        }
                    }.filter,
                );
            },

            .column => |v| column: {
                if (v.x <= 0) {
                    log.warn("delete column must be greater than zero", .{});
                    break :column;
                }

                const x = v.x - 1;
                var it = self.placements.iterator();
                while (it.next()) |entry| {
                    const img = self.imageById(entry.key_ptr.image_id) orelse continue;
                    const rect = entry.value_ptr.rect(img, t) orelse continue;
                    if (rect.top_left.x <= x and rect.bottom_right.x >= x) {
                        entry.value_ptr.deinit(t.screens.active);
                        self.placements.removeByPtr(entry.key_ptr);
                        if (v.delete) self.deleteIfUnused(alloc, img.id);
                    }
                }
            },

            .row => |v| row: {
                if (v.y <= 0) {
                    log.warn("delete row must be greater than zero", .{});
                    break :row;
                }

                // v.y is in active coords so we want to convert it to a pin
                // so we can compare by page offsets.
                const target_pin = t.screens.active.pages.pin(.{ .active = .{
                    .y = std.math.cast(size.CellCountInt, v.y - 1) orelse break :row,
                } }) orelse break :row;

                var it = self.placements.iterator();
                while (it.next()) |entry| {
                    const img = self.imageById(entry.key_ptr.image_id) orelse continue;
                    const rect = entry.value_ptr.rect(img, t) orelse continue;

                    // We need to copy our pin to ensure we are at least at
                    // the top-left x.
                    var target_pin_copy = target_pin;
                    target_pin_copy.x = rect.top_left.x;
                    if (target_pin_copy.isBetween(rect.top_left, rect.bottom_right)) {
                        entry.value_ptr.deinit(t.screens.active);
                        self.placements.removeByPtr(entry.key_ptr);
                        if (v.delete) self.deleteIfUnused(alloc, img.id);
                    }
                }
            },

            .z => |v| {
                var it = self.placements.iterator();
                while (it.next()) |entry| {
                    switch (entry.value_ptr.location) {
                        .pin => {},

                        // Virtual placeholders cannot delete by z according
                        // to the spec.
                        .virtual => continue,
                    }

                    if (entry.value_ptr.z == v.z) {
                        const image_id = entry.key_ptr.image_id;
                        entry.value_ptr.deinit(t.screens.active);
                        self.placements.removeByPtr(entry.key_ptr);
                        if (v.delete) self.deleteIfUnused(alloc, image_id);
                    }
                }
            },

            .range => |v| range: {
                if (v.first <= 0 or v.last <= 0) {
                    log.warn("delete range values must be greater than zero", .{});
                    break :range;
                }
                if (v.first > v.last) {
                    log.warn("delete range 'x' ({}) must be less than or equal to 'y' ({})", .{ v.first, v.last });
                    break :range;
                }

                var it = self.placements.iterator();
                while (it.next()) |entry| {
                    if (entry.key_ptr.image_id >= v.first or entry.key_ptr.image_id <= v.last) {
                        const image_id = entry.key_ptr.image_id;
                        entry.value_ptr.deinit(t.screens.active);
                        self.placements.removeByPtr(entry.key_ptr);
                        if (v.delete) self.deleteIfUnused(alloc, image_id);
                    }
                }
            },

            // Frame deletion can report a malformed command, which this
            // void API can't express, so graphics_exec.zig calls the
            // helper directly. Delegate for any other caller.
            .animation_frames => |v| _ = self.deleteAnimationFrame(
                alloc,
                t,
                v,
                self.animationNowMs() orelse 0,
            ),
        }
    }

    fn deleteById(
        self: *ImageStorage,
        alloc: Allocator,
        s: *terminal.Screen,
        image_id: u32,
        placement_id: u32,
        delete_unused: bool,
    ) void {
        // If no placement, we delete all placements with the ID
        if (placement_id == 0) {
            var it = self.placements.iterator();
            while (it.next()) |entry| {
                if (entry.key_ptr.image_id == image_id) {
                    entry.value_ptr.deinit(s);
                    self.placements.removeByPtr(entry.key_ptr);
                }
            }
        } else {
            if (self.placements.getEntry(.{
                .image_id = image_id,
                .placement_id = .{ .tag = .external, .id = placement_id },
            })) |entry| {
                entry.value_ptr.deinit(s);
                self.placements.removeByPtr(entry.key_ptr);
            }
        }

        // If this is specified, then we also delete the image
        // if it is no longer in use.
        if (delete_unused) self.deleteIfUnused(alloc, image_id);
    }

    /// Delete an image if it is unused.
    fn deleteIfUnused(self: *ImageStorage, alloc: Allocator, image_id: u32) void {
        var it = self.placements.iterator();
        while (it.next()) |kv| {
            if (kv.key_ptr.image_id == image_id) {
                return;
            }
        }

        // If we get here, we can delete the image.
        if (self.images.getEntry(image_id)) |entry| {
            _ = self.removeImage(alloc, entry);
        }
    }

    /// Remove every placement of an image, releasing the pins they track.
    /// Returns true if any placement was removed.
    ///
    /// This is the only correct way to drop placements. A placement removed
    /// from the map without being deinitialized leaves its pin registered
    /// in the PageList forever, so evicting placed images leaks pins and
    /// makes every subsequent page update more expensive.
    fn removePlacementsForImage(
        self: *ImageStorage,
        s: *terminal.Screen,
        image_id: u32,
    ) bool {
        var removed = false;
        var it = self.placements.iterator();
        while (it.next()) |entry| {
            if (entry.key_ptr.image_id != image_id) continue;
            entry.value_ptr.deinit(s);
            self.placements.removeByPtr(entry.key_ptr);
            removed = true;
        }
        return removed;
    }

    /// Remove an image from the map, freeing it and keeping every piece of
    /// accounting (byte total, animation count) in step. Returns the bytes
    /// freed, which includes any animation frames the image owned.
    ///
    /// This is the only way images should be removed: doing it by hand is
    /// how the byte total drifts. Note this does not touch placements; the
    /// caller must have removed them via removePlacementsForImage.
    fn removeImage(
        self: *ImageStorage,
        alloc: Allocator,
        entry: ImageMap.Entry,
    ) usize {
        const bytes = entry.value_ptr.byteSize();
        self.total_bytes -= bytes;
        if (entry.value_ptr.anim != null) self.animation_count -= 1;
        entry.value_ptr.deinit(alloc);
        self.images.removeByPtr(entry.key_ptr);
        return bytes;
    }

    /// Deletes all placements intersecting a screen point.
    fn deleteIntersecting(
        self: *ImageStorage,
        alloc: Allocator,
        t: *terminal.Terminal,
        p: point.Point,
        delete_unused: bool,
        filter_ctx: anytype,
        comptime filter: ?fn (@TypeOf(filter_ctx), Placement) bool,
    ) void {
        // Convert our target point to a pin for comparison.
        const target_pin = t.screens.active.pages.pin(p) orelse return;

        var it = self.placements.iterator();
        while (it.next()) |entry| {
            const img = self.imageById(entry.key_ptr.image_id) orelse continue;
            const rect = entry.value_ptr.rect(img, t) orelse continue;
            if (target_pin.isBetween(rect.top_left, rect.bottom_right)) {
                if (filter) |f| if (!f(filter_ctx, entry.value_ptr.*)) continue;
                entry.value_ptr.deinit(t.screens.active);
                self.placements.removeByPtr(entry.key_ptr);
                if (delete_unused) self.deleteIfUnused(alloc, img.id);
            }
        }
    }

    /// Evict image to make space. This will evict the oldest image,
    /// prioritizing unused images first, as recommended by the published
    /// Kitty spec.
    ///
    /// This will evict as many images as necessary to make space for
    /// req bytes.
    fn evictImage(
        self: *ImageStorage,
        alloc: Allocator,
        s: *terminal.Screen,
        exclude_image_id: u32,
        req: usize,
    ) !bool {
        assert(req <= self.total_limit);

        const candidates = try self.evictionCandidates(alloc, exclude_image_id);
        defer alloc.free(candidates);

        // Evicting anything is a content mutation. This matters for the
        // setLimit path in particular, which doesn't otherwise mark it.
        var any_evicted = false;
        defer if (any_evicted) self.markLayoutMutated();

        // They're in order of best to evict.
        var evicted: usize = 0;
        for (candidates) |c| {
            // Delete all the placements for this image and the image.
            if (self.removePlacementsForImage(s, c.id)) any_evicted = true;

            if (self.images.getEntry(c.id)) |entry| {
                log.info("evicting image id={} bytes={}", .{ c.id, entry.value_ptr.byteSize() });

                evicted += self.removeImage(alloc, entry);
                any_evicted = true;

                // Freeing exactly the requested bytes is success.
                if (evicted >= req) return true;
            }
        }

        return false;
    }

    /// An image that could be evicted to make space, along with what we
    /// order eviction by.
    const EvictionCandidate = struct {
        id: u32,
        generation: u64,
        used: bool,
        bytes: usize,
    };

    /// Build the list of images that could be evicted, best candidate
    /// first: unused images before used ones, oldest before newest.
    ///
    /// `exclude_image_id` is never a candidate. Animation frame operations
    /// use this so that an image can't be evicted to make room for its own
    /// new frame. Image IDs are always nonzero, so 0 excludes nothing.
    ///
    /// Ironically we allocate to evict. We should probably redesign the
    /// data structures to avoid this but for now allocating a little
    /// bit is fine compared to the megabytes we're looking to save.
    /// The caller owns the returned slice.
    fn evictionCandidates(
        self: *const ImageStorage,
        alloc: Allocator,
        exclude_image_id: u32,
    ) Allocator.Error![]EvictionCandidate {
        var candidates: std.ArrayList(EvictionCandidate) = .empty;
        defer candidates.deinit(alloc);

        var it = self.images.iterator();
        while (it.next()) |kv| {
            const img = kv.value_ptr;
            if (img.id == exclude_image_id) continue;

            // This is a huge waste. See comment above about redesigning
            // our data structures to avoid this. Eviction should be very
            // rare though and we never have that many images/placements
            // so hopefully this will last a long time.
            const used = used: {
                var p_it = self.placements.iterator();
                while (p_it.next()) |p_kv| {
                    if (p_kv.key_ptr.image_id == img.id) {
                        break :used true;
                    }
                }

                break :used false;
            };

            try candidates.append(alloc, .{
                .id = img.id,
                .generation = img.generation,
                .used = used,
                .bytes = img.byteSize(),
            });
        }

        std.mem.sortUnstable(
            EvictionCandidate,
            candidates.items,
            {},
            struct {
                fn lessThan(
                    ctx: void,
                    lhs: EvictionCandidate,
                    rhs: EvictionCandidate,
                ) bool {
                    _ = ctx;

                    // If their usage matches, then it's based on the
                    // generation stamp, which orders by transmit time.
                    // (Stamps are unique but tie-break by ID anyway to
                    // stay deterministic for hand-built test images.)
                    if (lhs.used == rhs.used) return if (lhs.generation == rhs.generation)
                        lhs.id < rhs.id
                    else
                        lhs.generation < rhs.generation;

                    // If not used, then its a better candidate
                    return !lhs.used;
                }
            }.lessThan,
        );

        return try candidates.toOwnedSlice(alloc);
    }

    /// Every placement is uniquely identified by the image ID and the
    /// placement ID. If an image ID isn't specified it is assumed to be 0.
    /// Likewise, if a placement ID isn't specified it is assumed to be 0.
    pub const PlacementKey = struct {
        image_id: u32,
        placement_id: packed struct {
            tag: enum(u1) { internal, external },
            id: u32,
        },
    };

    pub const Placement = struct {
        /// The location where this placement should be drawn.
        location: Location,

        /// Offset of the x/y from the top-left of the cell.
        x_offset: u32 = 0,
        y_offset: u32 = 0,

        /// Source rectangle for the image to pull from
        source_x: u32 = 0,
        source_y: u32 = 0,
        source_width: u32 = 0,
        source_height: u32 = 0,

        /// The columns/rows this image occupies.
        columns: u32 = 0,
        rows: u32 = 0,

        /// The z-index for this placement.
        z: i32 = 0,

        pub const Location = union(enum) {
            /// Exactly placed on a screen pin.
            pin: *PageList.Pin,

            /// Virtual placement (U=1) for unicode placeholders.
            virtual: void,
        };

        pub fn deinit(
            self: *const Placement,
            s: *terminal.Screen,
        ) void {
            switch (self.location) {
                .pin => |p| s.pages.untrackPin(p),
                .virtual => {},
            }
        }

        /// Returns the size of this placement's image in pixels,
        /// taking into account the source rectangle, specified
        /// rows/columns, and aspect ratio.
        pub fn pixelSize(
            self: Placement,
            image: Image,
            t: *const terminal.Terminal,
        ) struct {
            width: u32,
            height: u32,
        } {
            // Height / width of the image in px.
            const width = if (self.source_width > 0) self.source_width else image.width;
            const height = if (self.source_height > 0) self.source_height else image.height;

            // If we don't have any specified cols or rows then the placement
            // should be the native size of the image, and doesn't need to be
            // re-scaled.
            if (self.columns == 0 and self.rows == 0) return .{
                .width = width,
                .height = height,
            };

            // We calculate the size of a cell so that we can multiply
            // it by the specified cols/rows to get the correct px size.
            //
            // We assume that the width is divided evenly by the column
            // count and the height by the row count, because it should be.
            const cell_width: u32 = t.width_px / t.cols;
            const cell_height: u32 = t.height_px / t.rows;

            const width_f64: f64 = @floatFromInt(width);
            const height_f64: f64 = @floatFromInt(height);

            // If we have a specified cols AND rows then we calculate
            // the width and height from them directly, we don't need
            // to adjust for aspect ratio.
            if (self.columns > 0 and self.rows > 0) {
                const calc_width = cell_width * self.columns;
                const calc_height = cell_height * self.rows;

                return .{
                    .width = calc_width,
                    .height = calc_height,
                };
            }

            // Either the columns or the rows were specified, but not both,
            // so we need to calculate the other one based on the aspect ratio.

            // If only the columns were specified, we determine
            // the height of the image based on the aspect ratio.
            if (self.columns > 0) {
                const aspect = height_f64 / width_f64;
                const calc_width: u32 = cell_width * self.columns;
                const calc_height: u32 = @intFromFloat(@round(
                    @as(f64, @floatFromInt(calc_width)) * aspect,
                ));

                return .{
                    .width = calc_width,
                    .height = calc_height,
                };
            }

            // Otherwise, only the rows were specified, so we
            // determine the width based on the aspect ratio.
            {
                const aspect = width_f64 / height_f64;
                const calc_height: u32 = cell_height * self.rows;
                const calc_width: u32 = @intFromFloat(@round(
                    @as(f64, @floatFromInt(calc_height)) * aspect,
                ));

                return .{
                    .width = calc_width,
                    .height = calc_height,
                };
            }
        }

        /// Returns the size in grid cells that this placement takes up.
        pub fn gridSize(
            self: Placement,
            image: Image,
            t: *const terminal.Terminal,
        ) struct {
            cols: u32,
            rows: u32,
        } {
            // If we have a specified columns and rows then this is trivial.
            if (self.columns > 0 and self.rows > 0) return .{
                .cols = self.columns,
                .rows = self.rows,
            };

            // Otherwise we calculate the pixel size, divide by
            // cell size, and round up to the nearest integer.
            const calc_size = self.pixelSize(image, t);
            return .{
                .cols = std.math.divCeil(
                    u32,
                    calc_size.width + self.x_offset,
                    t.width_px / t.cols,
                ) catch 0,
                .rows = std.math.divCeil(
                    u32,
                    calc_size.height + self.y_offset,
                    t.height_px / t.rows,
                ) catch 0,
            };
            // NOTE: Above `divCeil`s can only fail if the cell size is 0,
            //       in such a case it seems safe to return 0 for this.
        }

        /// Returns a selection of the entire rectangle this placement
        /// occupies within the screen. This can return null if the placement
        /// doesn't have an associated rect (i.e. a virtual placement).
        pub fn rect(
            self: Placement,
            image: Image,
            t: *const terminal.Terminal,
        ) ?Rect {
            const grid_size = self.gridSize(image, t);
            const pin = switch (self.location) {
                .pin => |p| p,
                .virtual => return null,
            };

            var br = switch (pin.downOverflow(grid_size.rows - 1)) {
                .offset => |v| v,
                .overflow => |v| v.end,
            };
            br.x = @min(
                // We need to sub one here because the x value is
                // one width already. So if the image is width "1"
                // then we add zero to X because X itself is width 1.
                pin.x + (grid_size.cols - 1),
                t.cols - 1,
            );

            return .{
                .top_left = pin.*,
                .bottom_right = br,
            };
        }
    };
};

// Our pin for the placement
fn trackPin(
    t: *terminal.Terminal,
    pt: point.Coordinate,
) !*PageList.Pin {
    return try t.screens.active.pages.trackPin(t.screens.active.pages.pin(.{
        .active = pt,
    }).?);
}

test "storage: add placement with zero placement id" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .cols = 100, .rows = 100 });
    defer t.deinit(alloc);
    t.width_px = 100;
    t.height_px = 100;

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try s.addImage(alloc, t.screens.active, .{ .id = 1, .width = 50, .height = 50 });
    try s.addImage(alloc, t.screens.active, .{ .id = 2, .width = 25, .height = 25 });
    try s.addPlacement(alloc, 1, 0, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 25, .y = 25 }) } });
    try s.addPlacement(alloc, 1, 0, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 25, .y = 25 }) } });

    try testing.expectEqual(@as(usize, 2), s.placements.count());
    try testing.expectEqual(@as(usize, 2), s.images.count());

    // verify the placement is what we expect
    try testing.expect(s.placements.get(.{
        .image_id = 1,
        .placement_id = .{ .tag = .internal, .id = 0 },
    }) != null);
    try testing.expect(s.placements.get(.{
        .image_id = 1,
        .placement_id = .{ .tag = .internal, .id = 1 },
    }) != null);
}

test "storage: delete all placements and images" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);
    const tracked = t.screens.active.pages.countTrackedPins();

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try s.addImage(alloc, t.screens.active, .{ .id = 1 });
    try s.addImage(alloc, t.screens.active, .{ .id = 2 });
    try s.addImage(alloc, t.screens.active, .{ .id = 3 });
    try s.addPlacement(alloc, 1, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });
    try s.addPlacement(alloc, 2, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });

    s.layout_dirty = false;
    s.delete(alloc, &t, .{ .all = true });
    try testing.expect(s.layout_dirty);
    try testing.expectEqual(@as(usize, 0), s.images.count());
    try testing.expectEqual(@as(usize, 0), s.placements.count());
    try testing.expectEqual(tracked, t.screens.active.pages.countTrackedPins());
}

test "storage: delete all placements and images preserves limit" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);
    const tracked = t.screens.active.pages.countTrackedPins();

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    s.total_limit = 5000;
    try s.addImage(alloc, t.screens.active, .{ .id = 1 });
    try s.addImage(alloc, t.screens.active, .{ .id = 2 });
    try s.addImage(alloc, t.screens.active, .{ .id = 3 });
    try s.addPlacement(alloc, 1, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });
    try s.addPlacement(alloc, 2, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });

    s.layout_dirty = false;
    s.delete(alloc, &t, .{ .all = true });
    try testing.expect(s.layout_dirty);
    try testing.expectEqual(@as(usize, 0), s.images.count());
    try testing.expectEqual(@as(usize, 0), s.placements.count());
    try testing.expectEqual(@as(usize, 5000), s.total_limit);
    try testing.expectEqual(tracked, t.screens.active.pages.countTrackedPins());
}

test "storage: delete all placements" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);
    const tracked = t.screens.active.pages.countTrackedPins();

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try s.addImage(alloc, t.screens.active, .{ .id = 1 });
    try s.addImage(alloc, t.screens.active, .{ .id = 2 });
    try s.addImage(alloc, t.screens.active, .{ .id = 3 });
    try s.addPlacement(alloc, 1, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });
    try s.addPlacement(alloc, 2, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });

    s.layout_dirty = false;
    s.delete(alloc, &t, .{ .all = false });
    try testing.expect(s.layout_dirty);
    try testing.expectEqual(@as(usize, 0), s.placements.count());
    try testing.expectEqual(@as(usize, 3), s.images.count());
    try testing.expectEqual(tracked, t.screens.active.pages.countTrackedPins());
}

test "storage: delete all placements by image id" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);
    const tracked = t.screens.active.pages.countTrackedPins();

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try s.addImage(alloc, t.screens.active, .{ .id = 1 });
    try s.addImage(alloc, t.screens.active, .{ .id = 2 });
    try s.addImage(alloc, t.screens.active, .{ .id = 3 });
    try s.addPlacement(alloc, 1, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });
    try s.addPlacement(alloc, 2, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });

    s.layout_dirty = false;
    s.delete(alloc, &t, .{ .id = .{ .image_id = 2 } });
    try testing.expect(s.layout_dirty);
    try testing.expectEqual(@as(usize, 1), s.placements.count());
    try testing.expectEqual(@as(usize, 3), s.images.count());
    try testing.expectEqual(tracked + 1, t.screens.active.pages.countTrackedPins());
}

test "storage: delete all placements by image id and unused images" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);
    const tracked = t.screens.active.pages.countTrackedPins();

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try s.addImage(alloc, t.screens.active, .{ .id = 1 });
    try s.addImage(alloc, t.screens.active, .{ .id = 2 });
    try s.addImage(alloc, t.screens.active, .{ .id = 3 });
    try s.addPlacement(alloc, 1, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });
    try s.addPlacement(alloc, 2, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });

    s.layout_dirty = false;
    s.delete(alloc, &t, .{ .id = .{ .delete = true, .image_id = 2 } });
    try testing.expect(s.layout_dirty);
    try testing.expectEqual(@as(usize, 1), s.placements.count());
    try testing.expectEqual(@as(usize, 2), s.images.count());
    try testing.expectEqual(tracked + 1, t.screens.active.pages.countTrackedPins());
}

test "storage: delete placement by specific id" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);
    const tracked = t.screens.active.pages.countTrackedPins();

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try s.addImage(alloc, t.screens.active, .{ .id = 1 });
    try s.addImage(alloc, t.screens.active, .{ .id = 2 });
    try s.addImage(alloc, t.screens.active, .{ .id = 3 });
    try s.addPlacement(alloc, 1, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });
    try s.addPlacement(alloc, 1, 2, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });
    try s.addPlacement(alloc, 2, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });

    s.layout_dirty = false;
    s.delete(alloc, &t, .{ .id = .{
        .delete = true,
        .image_id = 1,
        .placement_id = 2,
    } });
    try testing.expect(s.layout_dirty);
    try testing.expectEqual(@as(usize, 2), s.placements.count());
    try testing.expectEqual(@as(usize, 3), s.images.count());
    try testing.expectEqual(tracked + 2, t.screens.active.pages.countTrackedPins());
}

test "storage: delete intersecting cursor" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 100, .cols = 100 });
    defer t.deinit(alloc);
    t.width_px = 100;
    t.height_px = 100;
    const tracked = t.screens.active.pages.countTrackedPins();

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try s.addImage(alloc, t.screens.active, .{ .id = 1, .width = 50, .height = 50 });
    try s.addImage(alloc, t.screens.active, .{ .id = 2, .width = 25, .height = 25 });
    try s.addPlacement(alloc, 1, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 0, .y = 0 }) } });
    try s.addPlacement(alloc, 1, 2, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 25, .y = 25 }) } });

    t.screens.active.cursorAbsolute(12, 12);

    s.layout_dirty = false;
    s.delete(alloc, &t, .{ .intersect_cursor = false });
    try testing.expect(s.layout_dirty);
    try testing.expectEqual(@as(usize, 1), s.placements.count());
    try testing.expectEqual(@as(usize, 2), s.images.count());
    try testing.expectEqual(tracked + 1, t.screens.active.pages.countTrackedPins());

    // verify the placement is what we expect
    try testing.expect(s.placements.get(.{
        .image_id = 1,
        .placement_id = .{ .tag = .external, .id = 2 },
    }) != null);
}

test "storage: delete intersecting cursor plus unused" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 100, .cols = 100 });
    defer t.deinit(alloc);
    t.width_px = 100;
    t.height_px = 100;
    const tracked = t.screens.active.pages.countTrackedPins();

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try s.addImage(alloc, t.screens.active, .{ .id = 1, .width = 50, .height = 50 });
    try s.addImage(alloc, t.screens.active, .{ .id = 2, .width = 25, .height = 25 });
    try s.addPlacement(alloc, 1, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 0, .y = 0 }) } });
    try s.addPlacement(alloc, 1, 2, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 25, .y = 25 }) } });

    t.screens.active.cursorAbsolute(12, 12);

    s.layout_dirty = false;
    s.delete(alloc, &t, .{ .intersect_cursor = true });
    try testing.expect(s.layout_dirty);
    try testing.expectEqual(@as(usize, 1), s.placements.count());
    try testing.expectEqual(@as(usize, 2), s.images.count());
    try testing.expectEqual(tracked + 1, t.screens.active.pages.countTrackedPins());

    // verify the placement is what we expect
    try testing.expect(s.placements.get(.{
        .image_id = 1,
        .placement_id = .{ .tag = .external, .id = 2 },
    }) != null);
}

test "storage: delete intersecting cursor hits multiple" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 100, .cols = 100 });
    defer t.deinit(alloc);
    t.width_px = 100;
    t.height_px = 100;
    const tracked = t.screens.active.pages.countTrackedPins();

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try s.addImage(alloc, t.screens.active, .{ .id = 1, .width = 50, .height = 50 });
    try s.addImage(alloc, t.screens.active, .{ .id = 2, .width = 25, .height = 25 });
    try s.addPlacement(alloc, 1, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 0, .y = 0 }) } });
    try s.addPlacement(alloc, 1, 2, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 25, .y = 25 }) } });

    t.screens.active.cursorAbsolute(26, 26);

    s.layout_dirty = false;
    s.delete(alloc, &t, .{ .intersect_cursor = true });
    try testing.expect(s.layout_dirty);
    try testing.expectEqual(@as(usize, 0), s.placements.count());
    try testing.expectEqual(@as(usize, 1), s.images.count());
    try testing.expectEqual(tracked, t.screens.active.pages.countTrackedPins());
}

test "storage: delete by column" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 100, .cols = 100 });
    defer t.deinit(alloc);
    t.width_px = 100;
    t.height_px = 100;
    const tracked = t.screens.active.pages.countTrackedPins();

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try s.addImage(alloc, t.screens.active, .{ .id = 1, .width = 50, .height = 50 });
    try s.addImage(alloc, t.screens.active, .{ .id = 2, .width = 25, .height = 25 });
    try s.addPlacement(alloc, 1, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 0, .y = 0 }) } });
    try s.addPlacement(alloc, 1, 2, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 25, .y = 25 }) } });

    s.layout_dirty = false;
    s.delete(alloc, &t, .{ .column = .{
        .delete = false,
        .x = 60,
    } });
    try testing.expect(s.layout_dirty);
    try testing.expectEqual(@as(usize, 1), s.placements.count());
    try testing.expectEqual(@as(usize, 2), s.images.count());
    try testing.expectEqual(tracked + 1, t.screens.active.pages.countTrackedPins());

    // verify the placement is what we expect
    try testing.expect(s.placements.get(.{
        .image_id = 1,
        .placement_id = .{ .tag = .external, .id = 1 },
    }) != null);
}

test "storage: delete by column 1x1" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 100, .cols = 100 });
    defer t.deinit(alloc);
    t.width_px = 100;
    t.height_px = 100;

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try s.addImage(alloc, t.screens.active, .{ .id = 1, .width = 1, .height = 1 });
    try s.addPlacement(alloc, 1, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 0, .y = 0 }) } });
    try s.addPlacement(alloc, 1, 2, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 0 }) } });
    try s.addPlacement(alloc, 1, 3, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 2, .y = 0 }) } });

    s.delete(alloc, &t, .{ .column = .{
        .delete = false,
        .x = 2,
    } });
    try testing.expectEqual(@as(usize, 2), s.placements.count());
    try testing.expectEqual(@as(usize, 1), s.images.count());

    // verify the placement is what we expect
    try testing.expect(s.placements.get(.{
        .image_id = 1,
        .placement_id = .{ .tag = .external, .id = 1 },
    }) != null);
    try testing.expect(s.placements.get(.{
        .image_id = 1,
        .placement_id = .{ .tag = .external, .id = 3 },
    }) != null);
}

test "storage: delete by row" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 100, .cols = 100 });
    defer t.deinit(alloc);
    t.width_px = 100;
    t.height_px = 100;
    const tracked = t.screens.active.pages.countTrackedPins();

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try s.addImage(alloc, t.screens.active, .{ .id = 1, .width = 50, .height = 50 });
    try s.addImage(alloc, t.screens.active, .{ .id = 2, .width = 25, .height = 25 });
    try s.addPlacement(alloc, 1, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 0, .y = 0 }) } });
    try s.addPlacement(alloc, 1, 2, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 25, .y = 25 }) } });

    s.layout_dirty = false;
    s.delete(alloc, &t, .{ .row = .{
        .delete = false,
        .y = 60,
    } });
    try testing.expect(s.layout_dirty);
    try testing.expectEqual(@as(usize, 1), s.placements.count());
    try testing.expectEqual(@as(usize, 2), s.images.count());
    try testing.expectEqual(tracked + 1, t.screens.active.pages.countTrackedPins());

    // verify the placement is what we expect
    try testing.expect(s.placements.get(.{
        .image_id = 1,
        .placement_id = .{ .tag = .external, .id = 1 },
    }) != null);
}

test "storage: delete by row 1x1" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 100, .cols = 100 });
    defer t.deinit(alloc);
    t.width_px = 100;
    t.height_px = 100;

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try s.addImage(alloc, t.screens.active, .{ .id = 1, .width = 1, .height = 1 });
    try s.addPlacement(alloc, 1, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .y = 0 }) } });
    try s.addPlacement(alloc, 1, 2, .{ .location = .{ .pin = try trackPin(&t, .{ .y = 1 }) } });
    try s.addPlacement(alloc, 1, 3, .{ .location = .{ .pin = try trackPin(&t, .{ .y = 2 }) } });

    s.delete(alloc, &t, .{ .row = .{
        .delete = false,
        .y = 2,
    } });
    try testing.expectEqual(@as(usize, 2), s.placements.count());
    try testing.expectEqual(@as(usize, 1), s.images.count());

    // verify the placement is what we expect
    try testing.expect(s.placements.get(.{
        .image_id = 1,
        .placement_id = .{ .tag = .external, .id = 1 },
    }) != null);
    try testing.expect(s.placements.get(.{
        .image_id = 1,
        .placement_id = .{ .tag = .external, .id = 3 },
    }) != null);
}

test "storage: delete images by range 1" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);
    const tracked = t.screens.active.pages.countTrackedPins();

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try s.addImage(alloc, t.screens.active, .{ .id = 1 });
    try s.addImage(alloc, t.screens.active, .{ .id = 2 });
    try s.addImage(alloc, t.screens.active, .{ .id = 3 });
    try s.addPlacement(alloc, 1, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });
    try s.addPlacement(alloc, 2, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });
    try testing.expectEqual(@as(usize, 3), s.images.count());
    try testing.expectEqual(@as(usize, 2), s.placements.count());

    s.layout_dirty = false;
    s.delete(alloc, &t, .{ .range = .{ .delete = false, .first = 1, .last = 2 } });
    try testing.expect(s.layout_dirty);
    try testing.expectEqual(@as(usize, 3), s.images.count());
    try testing.expectEqual(@as(usize, 0), s.placements.count());
    try testing.expectEqual(tracked, t.screens.active.pages.countTrackedPins());
}

test "storage: delete images by range 2" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);
    const tracked = t.screens.active.pages.countTrackedPins();

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try s.addImage(alloc, t.screens.active, .{ .id = 1 });
    try s.addImage(alloc, t.screens.active, .{ .id = 2 });
    try s.addImage(alloc, t.screens.active, .{ .id = 3 });
    try s.addPlacement(alloc, 1, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });
    try s.addPlacement(alloc, 2, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });
    try testing.expectEqual(@as(usize, 3), s.images.count());
    try testing.expectEqual(@as(usize, 2), s.placements.count());

    s.layout_dirty = false;
    s.delete(alloc, &t, .{ .range = .{ .delete = true, .first = 1, .last = 2 } });
    try testing.expect(s.layout_dirty);
    try testing.expectEqual(@as(usize, 1), s.images.count());
    try testing.expectEqual(@as(usize, 0), s.placements.count());
    try testing.expectEqual(tracked, t.screens.active.pages.countTrackedPins());
}

test "storage: delete images by range 3" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);
    const tracked = t.screens.active.pages.countTrackedPins();

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try s.addImage(alloc, t.screens.active, .{ .id = 1 });
    try s.addImage(alloc, t.screens.active, .{ .id = 2 });
    try s.addImage(alloc, t.screens.active, .{ .id = 3 });
    try s.addPlacement(alloc, 1, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });
    try s.addPlacement(alloc, 2, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });
    try testing.expectEqual(@as(usize, 3), s.images.count());
    try testing.expectEqual(@as(usize, 2), s.placements.count());

    s.layout_dirty = false;
    s.delete(alloc, &t, .{ .range = .{ .delete = false, .first = 1, .last = 1 } });
    try testing.expect(s.layout_dirty);
    try testing.expectEqual(@as(usize, 3), s.images.count());
    try testing.expectEqual(@as(usize, 0), s.placements.count());
    try testing.expectEqual(tracked, t.screens.active.pages.countTrackedPins());
}

test "storage: delete images by range 4" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);
    const tracked = t.screens.active.pages.countTrackedPins();

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try s.addImage(alloc, t.screens.active, .{ .id = 1 });
    try s.addImage(alloc, t.screens.active, .{ .id = 2 });
    try s.addImage(alloc, t.screens.active, .{ .id = 3 });
    try s.addPlacement(alloc, 1, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });
    try s.addPlacement(alloc, 2, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });
    try testing.expectEqual(@as(usize, 3), s.images.count());
    try testing.expectEqual(@as(usize, 2), s.placements.count());

    s.layout_dirty = false;
    s.delete(alloc, &t, .{ .range = .{ .delete = true, .first = 1, .last = 1 } });
    try testing.expect(s.layout_dirty);
    try testing.expectEqual(@as(usize, 1), s.images.count());
    try testing.expectEqual(@as(usize, 0), s.placements.count());
    try testing.expectEqual(tracked, t.screens.active.pages.countTrackedPins());
}

test "storage: aspect ratio calculation when only columns or rows specified" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try terminal.Terminal.init(alloc, .{ .cols = 100, .rows = 100 });
    defer t.deinit(alloc);
    t.width_px = 1000; // 10 px per col
    t.height_px = 2000; // 20 px per row

    // Case 1: Only columns specified
    {
        const image = Image{ .id = 1, .width = 16, .height = 9 };
        var placement = ImageStorage.Placement{
            .location = .{ .virtual = {} },
            .columns = 10,
            .rows = 0,
        };

        // Image is 16x9, set to a width of 10 columns, at 10px per column
        // that's 100px width. 100px * (9 / 16) = 56.25, which should round
        // to a height of 56px.

        const calc_size = placement.pixelSize(image, &t);
        try testing.expectEqual(@as(u32, 100), calc_size.width);
        try testing.expectEqual(@as(u32, 56), calc_size.height);
    }

    // Case 2: Only rows specified
    {
        const image = Image{ .id = 2, .width = 16, .height = 9 };
        var placement = ImageStorage.Placement{
            .location = .{ .virtual = {} },
            .columns = 0,
            .rows = 5,
        };

        // Image is 16x9, set to a height of 5 rows, at 20px per row that's
        // 100px height. 100px * (16 / 9) = 177.77..., which should round to
        // a width of 178px.

        const calc_size = placement.pixelSize(image, &t);
        try testing.expectEqual(@as(u32, 178), calc_size.width);
        try testing.expectEqual(@as(u32, 100), calc_size.height);
    }
}

test "storage: generation stamps on image add and replace" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);

    // Fresh storage has generation zero (never mutated).
    try testing.expectEqual(@as(u64, 0), s.generation);

    try s.addImage(alloc, t.screens.active, .{ .id = 1, .width = 1, .height = 1 });
    const gen1 = s.generation;
    try testing.expect(gen1 > 0);

    const img1 = s.imageById(1).?;
    try testing.expectEqual(gen1, img1.generation);

    // A second image gets a strictly greater stamp.
    try s.addImage(alloc, t.screens.active, .{ .id = 2, .width = 1, .height = 1 });
    const gen2 = s.generation;
    try testing.expect(gen2 > gen1);
    try testing.expectEqual(gen2, s.imageById(2).?.generation);

    // Retransmitting the same image ID (identical dimensions) gets a
    // fresh stamp: this is what makes same-sized retransmissions
    // detectable by renderers.
    try s.addImage(alloc, t.screens.active, .{ .id = 1, .width = 1, .height = 1 });
    const gen3 = s.generation;
    try testing.expect(gen3 > gen2);
    try testing.expectEqual(gen3, s.imageById(1).?.generation);

    // Image 2 kept its stamp.
    try testing.expectEqual(gen2, s.imageById(2).?.generation);
}

test "storage: generation bumps on placement and delete" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try s.addImage(alloc, t.screens.active, .{ .id = 1 });
    const gen_add = s.generation;

    try s.addPlacement(alloc, 1, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });
    const gen_place = s.generation;
    try testing.expect(gen_place > gen_add);

    // Reads don't change the generation.
    _ = s.imageById(1);
    _ = s.imageByNumber(1);
    try testing.expectEqual(gen_place, s.generation);

    s.delete(alloc, &t, .{ .all = true });
    try testing.expect(s.generation > gen_place);
}

test "storage: generation bumps when setLimit evicts or disables" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);

    const data = try alloc.dupe(u8, "1234");
    try s.addImage(alloc, t.screens.active, .{ .id = 1, .width = 1, .height = 1, .data = data });
    const gen_add = s.generation;

    // Lowering the limit evicts the image and must mark a mutation.
    s.layout_dirty = false;
    try s.setLimit(alloc, t.screens.active, 1);
    try testing.expect(s.layout_dirty);
    try testing.expect(s.generation > gen_add);
    try testing.expectEqual(@as(usize, 0), s.images.count());
    const gen_evict = s.generation;

    // Disabling (limit=0) resets the storage and must mark a mutation.
    s.layout_dirty = false;
    try s.setLimit(alloc, t.screens.active, 0);
    try testing.expect(s.layout_dirty);
    try testing.expect(s.generation > gen_evict);
}

test "storage: imageByNumber returns most recently transmitted" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);

    // Two images sharing a number: the newest transmission wins,
    // regardless of insertion order or clock resolution.
    try s.addImage(alloc, t.screens.active, .{ .id = 1, .number = 7 });
    try s.addImage(alloc, t.screens.active, .{ .id = 2, .number = 7 });
    try testing.expectEqual(@as(u32, 2), s.imageByNumber(7).?.id);

    // Retransmit the first: it becomes the newest.
    try s.addImage(alloc, t.screens.active, .{ .id = 1, .number = 7 });
    try testing.expectEqual(@as(u32, 1), s.imageByNumber(7).?.id);
}

test "storage: nextGeneration is unique and monotonic" {
    const testing = std.testing;
    const a = nextGeneration();
    const b = nextGeneration();
    try testing.expect(b > a);
    try testing.expect(a > 0);
}

test "storage: no-op delete does not mark a mutation" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);

    // A delete-all on an empty storage (this runs on every screen
    // clear) must not dirty the state or bump the generation.
    s.delete(alloc, &t, .{ .all = true });
    try testing.expect(!s.layout_dirty);
    try testing.expectEqual(@as(u64, 0), s.generation);

    // Same for a delete that matches nothing.
    try s.addImage(alloc, t.screens.active, .{ .id = 1 });
    try s.addPlacement(alloc, 1, 1, .{ .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) } });
    const gen = s.generation;
    s.layout_dirty = false;
    s.delete(alloc, &t, .{ .id = .{ .image_id = 42 } });
    try testing.expect(!s.layout_dirty);
    try testing.expectEqual(gen, s.generation);

    // But a delete that removes something does mark a mutation.
    s.delete(alloc, &t, .{ .id = .{ .image_id = 1 } });
    try testing.expect(s.layout_dirty);
    try testing.expect(s.generation > gen);
}

/// Add an image whose pixel data is `fill` repeated, for animation tests.
fn testAddImage(
    s: *ImageStorage,
    alloc: Allocator,
    screen: *terminal.Screen,
    id: u32,
    width: u32,
    height: u32,
    format: command.Transmission.Format,
    fill: u8,
) !void {
    const bpp = command.Transmission.formatBpp(format);
    const data = try alloc.alloc(u8, width * height * bpp);
    @memset(data, fill);
    errdefer alloc.free(data);
    try s.addImage(alloc, screen, .{
        .id = id,
        .width = width,
        .height = height,
        .format = format,
        .data = data,
    });
}

test "storage: animation frame append" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddImage(&s, alloc, t.screens.active, 1, 2, 2, .rgba, 0);

    // A full-canvas frame with no explicit gap.
    const src: [2 * 2 * 4]u8 = @splat(7);
    const result = try s.addAnimationFrame(alloc, t.screens.active, 1, .{}, &src, .rgba, 2, 2, 100);

    // The frame is appended as protocol frame 2 and is never the current
    // one, so nothing on screen changed.
    try testing.expectEqual(@as(u32, 2), result.frame);
    try testing.expect(!result.visible);

    const img = s.images.getPtr(1).?;
    const anim = img.anim.?;
    try testing.expectEqual(@as(usize, 1), anim.frames.items.len);
    try testing.expectEqual(animation.default_gap_ms, anim.frames.items[0].gap_ms);
    try testing.expectEqual(@as(u32, 1), anim.nonzero_gap_count);
    try testing.expectEqualSlices(u8, &src, anim.frames.items[0].data);
    try testing.expectEqual(@as(usize, 1), s.animation_count);

    // The root is still what's rendered.
    try testing.expectEqualSlices(u8, img.data, img.renderData());
}

test "storage: the root is widened only when it is involved" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);

    // An RGB root is 12 bytes for 2x2.
    try testAddImage(&s, alloc, t.screens.active, 1, 2, 2, .rgb, 3);
    try testing.expectEqual(@as(usize, 12), s.total_bytes);

    // A frame with no base frame is composed onto the background, so it
    // never reads the root. Widening the root here would grow an
    // untouched image from 12 to 16 bytes for nothing.
    const src: [2 * 2 * 3]u8 = @splat(9);
    _ = try s.addAnimationFrame(alloc, t.screens.active, 1, .{}, &src, .rgb, 2, 2, 0);
    {
        const img = s.images.getPtr(1).?;
        try testing.expectEqual(command.Transmission.Format.rgb, img.format);
        try testing.expectEqual(@as(usize, 12), img.data.len);
        try testing.expectEqual(@as(usize, 12 + 16), s.total_bytes);

        // The transmitted RGB rect is still widened into the frame.
        try testing.expectEqualSlices(
            u8,
            &.{ 9, 9, 9, 255, 9, 9, 9, 255, 9, 9, 9, 255, 9, 9, 9, 255 },
            img.anim.?.frames.items[0].data,
        );

        // The root is what's on screen, and it is still RGB.
        try testing.expectEqual(command.Transmission.Format.rgb, img.renderFormat());
    }

    // Basing a new frame on the root does read it, so now it widens.
    _ = try s.addAnimationFrame(
        alloc,
        t.screens.active,
        1,
        .{ .create_frame = 1 },
        &src,
        .rgb,
        2,
        2,
        0,
    );
    {
        const img = s.images.getPtr(1).?;
        try testing.expectEqual(command.Transmission.Format.rgba, img.format);
        try testing.expectEqualSlices(
            u8,
            &.{ 3, 3, 3, 255, 3, 3, 3, 255, 3, 3, 3, 255, 3, 3, 3, 255 },
            img.data,
        );
        try testing.expectEqual(command.Transmission.Format.rgba, img.renderFormat());
    }
    try expectAccountingExact(&s);
}

test "storage: animation frame edit" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddImage(&s, alloc, t.screens.active, 1, 2, 2, .rgba, 0);

    const src: [2 * 2 * 4]u8 = @splat(7);
    _ = try s.addAnimationFrame(alloc, t.screens.active, 1, .{}, &src, .rgba, 2, 2, 0);

    // Editing a frame that isn't current changes no visible pixels, so
    // the image keeps its generation stamp.
    const img = s.images.getPtr(1).?;
    const gen_before = img.generation;
    const one: [1 * 1 * 4]u8 = @splat(5);
    const edit = try s.addAnimationFrame(
        alloc,
        t.screens.active,
        1,
        .{ .edit_frame = 2, .x = 1, .y = 1, .composition_mode = .overwrite },
        &one,
        .rgba,
        1,
        1,
        500,
    );
    try testing.expectEqual(@as(u32, 2), edit.frame);
    try testing.expect(!edit.visible);
    try testing.expectEqual(gen_before, img.generation);
    try testing.expectEqual(@as(usize, 1), img.anim.?.frames.items.len);
    try testing.expectEqualSlices(u8, &.{
        7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 5, 5, 5, 5,
    }, img.anim.?.frames.items[0].data);

    // Editing the current frame (the root) does change what's on screen.
    const edit_root = try s.addAnimationFrame(
        alloc,
        t.screens.active,
        1,
        .{ .edit_frame = 1, .composition_mode = .overwrite },
        &src,
        .rgba,
        2,
        2,
        500,
    );
    try testing.expectEqual(@as(u32, 1), edit_root.frame);
    try testing.expect(edit_root.visible);
    try testing.expectEqual(s.generation, img.generation);
    try testing.expectEqual(@as(u64, 500), img.anim.?.last_frame_ms);
}

test "storage: animation frame clamps out of range edits" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddImage(&s, alloc, t.screens.active, 1, 1, 1, .rgba, 0);

    const src: [4]u8 = @splat(1);

    // "r" past the end appends rather than failing, and reports the frame
    // it actually resolved to.
    const far = try s.addAnimationFrame(alloc, t.screens.active, 1, .{ .edit_frame = 99 }, &src, .rgba, 1, 1, 0);
    try testing.expectEqual(@as(u32, 2), far.frame);

    // The next frame number is an append too.
    const next = try s.addAnimationFrame(alloc, t.screens.active, 1, .{ .edit_frame = 3 }, &src, .rgba, 1, 1, 0);
    try testing.expectEqual(@as(u32, 3), next.frame);
    try testing.expectEqual(@as(usize, 2), s.images.getPtr(1).?.anim.?.frames.items.len);
}

test "storage: animation frame errors leave storage untouched" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddImage(&s, alloc, t.screens.active, 1, 2, 2, .rgba, 4);

    const bytes_before = s.total_bytes;
    const gen_before = s.generation;

    // A rect bigger than the image is rejected...
    const big: [3 * 3 * 4]u8 = @splat(1);
    try testing.expectError(error.InvalidRect, s.addAnimationFrame(
        alloc,
        t.screens.active,
        1,
        .{},
        &big,
        .rgba,
        3,
        3,
        0,
    ));

    // ...as is a base frame that doesn't exist.
    const src: [2 * 2 * 4]u8 = @splat(1);
    try testing.expectError(error.BaseFrameNotFound, s.addAnimationFrame(
        alloc,
        t.screens.active,
        1,
        .{ .create_frame = 5 },
        &src,
        .rgba,
        2,
        2,
        0,
    ));

    // Neither touched anything at all.
    const img = s.images.getPtr(1).?;
    try testing.expect(img.anim == null);
    try testing.expectEqual(@as(usize, 0), s.animation_count);
    try testing.expectEqual(bytes_before, s.total_bytes);
    try testing.expectEqual(gen_before, s.generation);
    try testing.expectEqual(command.Transmission.Format.rgba, img.format);
}

test "storage: animation frame base canvas and background" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddImage(&s, alloc, t.screens.active, 1, 2, 1, .rgba, 8);

    // A new frame based on the root starts as a copy of it, with the
    // transmitted rect composed over the left pixel only.
    const one: [4]u8 = .{ 1, 2, 3, 255 };
    _ = try s.addAnimationFrame(
        alloc,
        t.screens.active,
        1,
        .{ .create_frame = 1, .composition_mode = .overwrite },
        &one,
        .rgba,
        1,
        1,
        0,
    );
    try testing.expectEqualSlices(
        u8,
        &.{ 1, 2, 3, 255, 8, 8, 8, 8 },
        s.images.getPtr(1).?.anim.?.frames.items[0].data,
    );

    // With no base frame the canvas is the background color instead.
    _ = try s.addAnimationFrame(
        alloc,
        t.screens.active,
        1,
        .{ .background = .{ .r = 9, .a = 255 }, .composition_mode = .overwrite },
        &one,
        .rgba,
        1,
        1,
        0,
    );
    try testing.expectEqualSlices(
        u8,
        &.{ 1, 2, 3, 255, 9, 0, 0, 255 },
        s.images.getPtr(1).?.anim.?.frames.items[1].data,
    );
}

test "storage: retransmit clears animation" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddImage(&s, alloc, t.screens.active, 1, 2, 2, .rgba, 0);

    const src: [2 * 2 * 4]u8 = @splat(7);
    _ = try s.addAnimationFrame(alloc, t.screens.active, 1, .{}, &src, .rgba, 2, 2, 0);
    try testing.expectEqual(@as(usize, 1), s.animation_count);
    try testing.expectEqual(@as(usize, 32), s.total_bytes);

    // Retransmitting the same ID drops the animation entirely, and all of
    // its frame bytes with it.
    try testAddImage(&s, alloc, t.screens.active, 1, 2, 2, .rgba, 1);
    try testing.expectEqual(@as(usize, 0), s.animation_count);
    try testing.expectEqual(@as(usize, 16), s.total_bytes);
    try testing.expect(s.images.getPtr(1).?.anim == null);
}

test "storage: deleting an image frees its frame bytes" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddImage(&s, alloc, t.screens.active, 1, 2, 2, .rgba, 0);

    const src: [2 * 2 * 4]u8 = @splat(7);
    _ = try s.addAnimationFrame(alloc, t.screens.active, 1, .{}, &src, .rgba, 2, 2, 0);
    try testing.expectEqual(@as(usize, 32), s.total_bytes);

    s.delete(alloc, &t, .{ .id = .{ .image_id = 1, .delete = true } });
    try testing.expectEqual(@as(usize, 0), s.total_bytes);
    try testing.expectEqual(@as(usize, 0), s.animation_count);
    try testing.expectEqual(@as(usize, 0), s.images.count());
}

test "storage: animation reservation evicts but excludes the target" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);

    // Two 2x2 RGBA images is 32 bytes; the limit leaves room for exactly
    // one more frame.
    try testAddImage(&s, alloc, t.screens.active, 1, 2, 2, .rgba, 1);
    try testAddImage(&s, alloc, t.screens.active, 2, 2, 2, .rgba, 2);
    s.total_limit = 32;

    // Adding a frame to image 2 needs 16 bytes, which can only come from
    // evicting image 1: the target must never be evicted to make room for
    // its own frame.
    const src: [2 * 2 * 4]u8 = @splat(7);
    _ = try s.addAnimationFrame(alloc, t.screens.active, 2, .{}, &src, .rgba, 2, 2, 0);

    try testing.expect(s.images.getPtr(1) == null);
    try testing.expectEqual(@as(usize, 32), s.total_bytes);
    try testing.expectEqual(@as(usize, 1), s.images.getPtr(2).?.anim.?.frames.items.len);
}

test "storage: animation frame over the limit fails cleanly" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddImage(&s, alloc, t.screens.active, 1, 2, 2, .rgba, 1);

    // Only the target exists, and it can't be evicted for itself, so
    // there's no way to find room for the frame.
    s.total_limit = 16;
    const src: [2 * 2 * 4]u8 = @splat(7);
    try testing.expectError(error.OutOfSpace, s.addAnimationFrame(
        alloc,
        t.screens.active,
        1,
        .{},
        &src,
        .rgba,
        2,
        2,
        0,
    ));

    // Nothing changed.
    try testing.expectEqual(@as(usize, 16), s.total_bytes);
    try testing.expect(s.images.getPtr(1).?.anim == null);
    try testing.expectEqual(@as(usize, 0), s.animation_count);
}

test "storage: compose animation frames" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddImage(&s, alloc, t.screens.active, 1, 2, 1, .rgba, 0);

    // Frame 2 is all 7s.
    const src: [2 * 1 * 4]u8 = @splat(7);
    _ = try s.addAnimationFrame(alloc, t.screens.active, 1, .{ .composition_mode = .overwrite }, &src, .rgba, 2, 1, 0);

    // Compose the left pixel of frame 2 (source, "r") onto the right
    // pixel of frame 1 (destination, "c"). The root is current, so this
    // changes what's on screen.
    const visible = try s.composeAnimationFrames(alloc, t.screens.active, 1, .{
        .edit_frame = 2,
        .frame = 1,
        .width = 1,
        .height = 1,
        .left_edge = 0,
        .top_edge = 0,
        .x = 1,
        .y = 0,
        .composition_mode = .overwrite,
    }, 700);
    try testing.expect(visible);

    const img = s.images.getPtr(1).?;
    try testing.expectEqualSlices(u8, &.{ 0, 0, 0, 0, 7, 7, 7, 7 }, img.data);
    try testing.expectEqual(s.generation, img.generation);
    try testing.expectEqual(@as(u64, 700), img.anim.?.last_frame_ms);
}

test "storage: compose animation frames validates" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddImage(&s, alloc, t.screens.active, 1, 2, 2, .rgba, 0);

    // Missing source and destination frames.
    try testing.expectError(error.FrameNotFound, s.composeAnimationFrames(alloc, t.screens.active, 1, .{
        .edit_frame = 0,
        .frame = 1,
    }, 0));
    try testing.expectError(error.FrameNotFound, s.composeAnimationFrames(alloc, t.screens.active, 1, .{
        .edit_frame = 1,
        .frame = 9,
    }, 0));

    // A rect running off the image, in the source and in the dest.
    try testing.expectError(error.InvalidRect, s.composeAnimationFrames(alloc, t.screens.active, 1, .{
        .edit_frame = 1,
        .frame = 1,
        .width = 2,
        .height = 2,
        .left_edge = 1,
    }, 0));
    try testing.expectError(error.InvalidRect, s.composeAnimationFrames(alloc, t.screens.active, 1, .{
        .edit_frame = 1,
        .frame = 1,
        .width = 2,
        .height = 2,
        .x = 1,
    }, 0));

    // Composing a frame onto itself with overlapping rects.
    try testing.expectError(error.InvalidRect, s.composeAnimationFrames(alloc, t.screens.active, 1, .{
        .edit_frame = 1,
        .frame = 1,
        .width = 2,
        .height = 2,
    }, 0));

    // But the same frame with disjoint rects is fine.
    _ = try s.composeAnimationFrames(alloc, t.screens.active, 1, .{
        .edit_frame = 1,
        .frame = 1,
        .width = 1,
        .height = 1,
        .left_edge = 1,
        .x = 0,
    }, 0);
}

/// Build an image with `extra` animation frames beyond the root, each
/// filled with a distinct byte (root is 1, frame 2 is 2, ...) so that a
/// test can tell which frame is being rendered from a single pixel.
fn testAddAnimatedImage(
    s: *ImageStorage,
    alloc: Allocator,
    screen: *terminal.Screen,
    id: u32,
    extra: u8,
) !void {
    try testAddImage(s, alloc, screen, id, 1, 1, .rgba, 1);
    for (0..extra) |i| {
        const src: [4]u8 = @splat(@intCast(i + 2));
        _ = try s.addAnimationFrame(
            alloc,
            screen,
            id,
            .{ .composition_mode = .overwrite },
            &src,
            .rgba,
            1,
            1,
            0,
        );
    }
}

test "storage: delete animation frame identifiers" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddAnimatedImage(&s, alloc, t.screens.active, 1, 1);

    // An ID and a number together is malformed.
    try testing.expectEqual(ImageStorage.FrameDeleteResult.invalid_identifiers, s.deleteAnimationFrame(
        alloc,
        &t,
        .{ .image_id = 1, .image_number = 1 },
        0,
    ));

    // Neither identifier, and an identifier that resolves to nothing, are
    // both quietly ignored.
    try testing.expectEqual(ImageStorage.FrameDeleteResult.no_op_or_missing, s.deleteAnimationFrame(
        alloc,
        &t,
        .{},
        0,
    ));
    try testing.expectEqual(ImageStorage.FrameDeleteResult.no_op_or_missing, s.deleteAnimationFrame(
        alloc,
        &t,
        .{ .image_id = 42 },
        0,
    ));
}

test "storage: delete animation frame with no frames" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddImage(&s, alloc, t.screens.active, 1, 1, 1, .rgba, 1);
    try s.addPlacement(alloc, 1, 0, .{
        .location = .{ .pin = try trackPin(&t, .{ .x = 0, .y = 0 }) },
    });

    // The lowercase form has nothing to delete.
    try testing.expectEqual(ImageStorage.FrameDeleteResult.no_op_or_missing, s.deleteAnimationFrame(
        alloc,
        &t,
        .{ .image_id = 1 },
        0,
    ));
    try testing.expectEqual(@as(usize, 1), s.images.count());

    // The uppercase form deletes the whole image and its placements.
    try testing.expectEqual(ImageStorage.FrameDeleteResult.changed, s.deleteAnimationFrame(
        alloc,
        &t,
        .{ .image_id = 1, .delete = true },
        0,
    ));
    try testing.expectEqual(@as(usize, 0), s.images.count());
    try testing.expectEqual(@as(usize, 0), s.placements.count());
    try testing.expectEqual(@as(usize, 0), s.total_bytes);
}

test "storage: delete animation root frame promotes" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);

    // Root=1, frame 2=2, frame 3=3, each 4 bytes.
    try testAddAnimatedImage(&s, alloc, t.screens.active, 1, 2);
    try testing.expectEqual(@as(usize, 12), s.total_bytes);

    const img = s.images.getPtr(1).?;
    img.anim.?.setGap(0, 111);
    img.anim.?.setGap(1, 222);
    const gen_before = img.generation;

    // Deleting the root promotes frame 2 into it, taking over both its
    // buffer and its gap. Only the old root's bytes go away.
    try testing.expectEqual(ImageStorage.FrameDeleteResult.changed, s.deleteAnimationFrame(
        alloc,
        &t,
        .{ .image_id = 1, .frame = 1 },
        900,
    ));

    try testing.expectEqual(@as(usize, 8), s.total_bytes);
    try testing.expectEqual(@as(usize, 2), img.anim.?.frameCount());
    try testing.expectEqual(@as(u32, 222), img.anim.?.root_gap_ms);

    // The promoted root (222) and the remaining frame (the default gap)
    // are both nonzero.
    try testing.expectEqual(@as(u32, 2), img.anim.?.nonzero_gap_count);

    // The root frame was on screen, so the pixels changed: frame 2 is now
    // showing, and the image is stamped and its interval restarted.
    try testing.expectEqual(@as(u8, 2), img.renderData()[0]);
    try testing.expect(img.generation != gen_before);
    try testing.expectEqual(s.generation, img.generation);
    try testing.expectEqual(@as(u64, 900), img.anim.?.last_frame_ms);
}

test "storage: delete animation root frame keeping current pixels" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddAnimatedImage(&s, alloc, t.screens.active, 1, 2);

    const img = s.images.getPtr(1).?;

    // Frame 2 is current, and deleting the root promotes frame 2 into the
    // root slot: the index changes but the very same pixels stay up, so
    // the image must NOT be restamped.
    img.anim.?.current_frame = 1;
    const gen_before = img.generation;
    img.anim.?.last_frame_ms = 5;

    try testing.expectEqual(ImageStorage.FrameDeleteResult.changed, s.deleteAnimationFrame(
        alloc,
        &t,
        .{ .image_id = 1, .frame = 1 },
        900,
    ));

    try testing.expectEqual(@as(u32, 0), img.anim.?.current_frame);
    try testing.expectEqual(@as(u8, 2), img.renderData()[0]);
    try testing.expectEqual(gen_before, img.generation);
    try testing.expectEqual(@as(u64, 5), img.anim.?.last_frame_ms);
}

test "storage: delete animation frame current index rules" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    // Deleting a frame before the current one shifts the current index
    // down so the same frame stays on screen.
    {
        var s: ImageStorage = .{};
        defer s.deinit(alloc, t.screens.active);
        try testAddAnimatedImage(&s, alloc, t.screens.active, 1, 2);
        const img = s.images.getPtr(1).?;
        img.anim.?.current_frame = 2; // frame 3

        _ = s.deleteAnimationFrame(alloc, &t, .{ .image_id = 1, .frame = 2 }, 0);
        try testing.expectEqual(@as(u32, 1), img.anim.?.current_frame);
        try testing.expectEqual(@as(u8, 3), img.renderData()[0]);
        try testing.expectEqual(@as(usize, 8), s.total_bytes);
    }

    // Deleting a frame after the current one leaves it alone.
    {
        var s: ImageStorage = .{};
        defer s.deinit(alloc, t.screens.active);
        try testAddAnimatedImage(&s, alloc, t.screens.active, 1, 2);
        const img = s.images.getPtr(1).?;
        img.anim.?.current_frame = 1; // frame 2

        _ = s.deleteAnimationFrame(alloc, &t, .{ .image_id = 1, .frame = 3 }, 0);
        try testing.expectEqual(@as(u32, 1), img.anim.?.current_frame);
        try testing.expectEqual(@as(u8, 2), img.renderData()[0]);
    }

    // Deleting the current frame when it's last clamps back onto the new
    // last frame, which is different pixels.
    {
        var s: ImageStorage = .{};
        defer s.deinit(alloc, t.screens.active);
        try testAddAnimatedImage(&s, alloc, t.screens.active, 1, 2);
        const img = s.images.getPtr(1).?;
        img.anim.?.current_frame = 2; // frame 3
        const gen_before = img.generation;

        _ = s.deleteAnimationFrame(alloc, &t, .{ .image_id = 1, .frame = 3 }, 900);
        try testing.expectEqual(@as(u32, 1), img.anim.?.current_frame);
        try testing.expectEqual(@as(u8, 2), img.renderData()[0]);
        try testing.expect(img.generation != gen_before);
        try testing.expectEqual(@as(u64, 900), img.anim.?.last_frame_ms);
    }

    // Deleting the only extra frame while it is current falls back to the
    // root.
    {
        var s: ImageStorage = .{};
        defer s.deinit(alloc, t.screens.active);
        try testAddAnimatedImage(&s, alloc, t.screens.active, 1, 1);
        const img = s.images.getPtr(1).?;
        img.anim.?.current_frame = 1;

        _ = s.deleteAnimationFrame(alloc, &t, .{ .image_id = 1, .frame = 2 }, 0);
        try testing.expectEqual(@as(u32, 0), img.anim.?.current_frame);
        try testing.expectEqual(@as(u8, 1), img.renderData()[0]);
        try testing.expectEqual(@as(usize, 0), img.anim.?.frames.items.len);
    }
}

test "storage: delete animation frame clamps" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddAnimatedImage(&s, alloc, t.screens.active, 1, 2);
    const img = s.images.getPtr(1).?;

    // Out of range clamps onto the last frame rather than failing.
    _ = s.deleteAnimationFrame(alloc, &t, .{ .image_id = 1, .frame = 99 }, 0);
    try testing.expectEqual(@as(usize, 1), img.anim.?.frames.items.len);
    try testing.expectEqual(@as(u8, 2), img.anim.?.frames.items[0].data[0]);

    // Frame 0 means the first frame, i.e. the root.
    _ = s.deleteAnimationFrame(alloc, &t, .{ .image_id = 1, .frame = 0 }, 0);
    try testing.expectEqual(@as(u8, 2), img.renderData()[0]);
    try testing.expectEqual(@as(usize, 0), img.anim.?.frames.items.len);
}

/// An animated image with a placement, ready to tick: a root plus `extra`
/// frames, each with a 100ms gap, running.
fn testAddPlayableImage(
    s: *ImageStorage,
    alloc: Allocator,
    t: *terminal.Terminal,
    id: u32,
    extra: u8,
) !void {
    try testAddAnimatedImage(s, alloc, t.screens.active, id, extra);
    try s.addPlacement(alloc, id, 0, .{
        .location = .{ .pin = try trackPin(t, .{ .x = 0, .y = 0 }) },
    });

    // Stand in for the renderer: only what is actually drawn is
    // scheduled, and there is no renderer in these tests.
    s.images.getPtr(id).?.drawn = true;

    const anim = s.images.getPtr(id).?.anim.?;
    anim.state = .running;
    anim.setGap(0, 100);
    for (0..anim.frames.items.len) |i| anim.setGap(@intCast(i + 1), 100);
}

test "storage: animation tick advances on schedule" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddPlayableImage(&s, alloc, &t, 1, 1);

    const img = s.images.getPtr(1).?;

    // Nothing is due yet, but the deadline is reported so the caller can
    // arm a timer for it.
    {
        const r = s.animationTick(50);
        try testing.expect(!r.dirtied);
        try testing.expectEqual(@as(?u64, 100), r.next_due_ms);
        try testing.expectEqual(@as(?u64, 100), s.nextAnimationDeadline());
        try testing.expectEqual(@as(u32, 0), img.anim.?.current_frame);
    }

    // At the deadline the frame advances, and the next one is due a gap
    // later.
    {
        const gen_before = img.generation;
        const r = s.animationTick(100);
        try testing.expect(r.dirtied);
        try testing.expectEqual(@as(?u64, 200), r.next_due_ms);
        try testing.expectEqual(@as(u32, 1), img.anim.?.current_frame);
        try testing.expectEqual(@as(u8, 2), img.renderData()[0]);
        try testing.expect(img.generation != gen_before);
    }

    // Only one frame per tick: Kitty doesn't catch up on missed frames
    // either, so a very late tick still advances just once.
    {
        const r = s.animationTick(10_000);
        try testing.expect(r.dirtied);
        try testing.expectEqual(@as(u32, 0), img.anim.?.current_frame);
        try testing.expectEqual(@as(?u64, 10_100), r.next_due_ms);
    }
}

test "storage: animation tick is dormant when not drawn" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);

    // A playable animation that the renderer does not draw: it may be
    // scrolled into the scrollback, on an inactive screen, or a virtual
    // placement with no placeholder in the viewport. Being in storage is
    // not being visible, so none of it may be scheduled.
    try testAddAnimatedImage(&s, alloc, t.screens.active, 1, 1);
    try s.addPlacement(alloc, 1, 0, .{
        .location = .{ .pin = try trackPin(&t, .{ .x = 0, .y = 0 }) },
    });
    const anim = s.images.getPtr(1).?.anim.?;
    anim.state = .running;
    anim.setGap(0, 100);
    try testing.expect(!s.images.getPtr(1).?.drawn);

    const r = s.animationTick(10_000);
    try testing.expect(!r.dirtied);
    try testing.expectEqual(@as(?u64, null), r.next_due_ms);
    try testing.expectEqual(@as(?u64, null), s.nextAnimationDeadline());
}

test "storage: animation tick is dormant when stopped or gapless" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddPlayableImage(&s, alloc, &t, 1, 1);
    const anim = s.images.getPtr(1).?.anim.?;

    // Stopped freezes on the current frame.
    anim.state = .stopped;
    try testing.expectEqual(@as(?u64, null), s.nextAnimationDeadline());
    try testing.expect(!s.animationTick(10_000).dirtied);
    try testing.expectEqual(@as(u32, 0), anim.current_frame);

    // An animation where every frame is gapless has no frame to stop on,
    // and must not be ticked at all: advance() would spin forever.
    anim.state = .running;
    anim.setGap(0, 0);
    anim.setGap(1, 0);
    try testing.expectEqual(@as(u32, 0), anim.nonzero_gap_count);
    try testing.expectEqual(@as(?u64, null), s.nextAnimationDeadline());
    try testing.expect(!s.animationTick(10_000).dirtied);
}

test "storage: animation tick with no animations early-outs" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddImage(&s, alloc, t.screens.active, 1, 1, 1, .rgba, 1);

    try testing.expectEqual(@as(usize, 0), s.animation_count);
    try testing.expectEqual(@as(?u64, null), s.nextAnimationDeadline());
    try testing.expect(!s.animationTick(0).dirtied);
}

test "storage: animation tick skips gapless frames" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddPlayableImage(&s, alloc, &t, 1, 2);

    // Make frame 2 gapless: it must be skipped straight over rather than
    // ever being displayed.
    const img = s.images.getPtr(1).?;
    img.anim.?.setGap(1, 0);

    const r = s.animationTick(100);
    try testing.expect(r.dirtied);
    try testing.expectEqual(@as(u32, 2), img.anim.?.current_frame);
    try testing.expectEqual(@as(u8, 3), img.renderData()[0]);
    try testing.expectEqual(@as(?u64, 200), r.next_due_ms);
}

test "storage: animation tick loop counting" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    // v=2 means max_loops=1: one pass that ends on the last frame.
    {
        var s: ImageStorage = .{};
        defer s.deinit(alloc, t.screens.active);
        try testAddPlayableImage(&s, alloc, &t, 1, 1);
        const anim = s.images.getPtr(1).?.anim.?;
        anim.max_loops = 1;

        // Root -> frame 2.
        try testing.expect(s.animationTick(100).dirtied);
        try testing.expectEqual(@as(u32, 1), anim.current_frame);

        // The wrap is counted but not committed, so the last frame stays
        // up and the animation goes quiet.
        const r = s.animationTick(200);
        try testing.expect(!r.dirtied);
        try testing.expectEqual(@as(?u64, null), r.next_due_ms);
        try testing.expectEqual(@as(u32, 1), anim.current_frame);
        try testing.expectEqual(@as(u32, 1), anim.current_loop);
        try testing.expectEqual(@as(?u64, null), s.nextAnimationDeadline());
    }

    // v=3 means max_loops=2: two full passes.
    {
        var s: ImageStorage = .{};
        defer s.deinit(alloc, t.screens.active);
        try testAddPlayableImage(&s, alloc, &t, 1, 1);
        const anim = s.images.getPtr(1).?.anim.?;
        anim.max_loops = 2;

        try testing.expect(s.animationTick(100).dirtied); // -> frame 2
        try testing.expect(s.animationTick(200).dirtied); // -> root, loop 1
        try testing.expectEqual(@as(u32, 0), anim.current_frame);
        try testing.expectEqual(@as(u32, 1), anim.current_loop);

        try testing.expect(s.animationTick(300).dirtied); // -> frame 2
        try testing.expect(!s.animationTick(400).dirtied); // done
        try testing.expectEqual(@as(u32, 1), anim.current_frame);
        try testing.expectEqual(@as(u32, 2), anim.current_loop);
    }

    // v=1 means max_loops=0, which is forever.
    {
        var s: ImageStorage = .{};
        defer s.deinit(alloc, t.screens.active);
        try testAddPlayableImage(&s, alloc, &t, 1, 1);
        const anim = s.images.getPtr(1).?.anim.?;

        var now: u64 = 100;
        for (0..10) |_| {
            try testing.expect(s.animationTick(now).dirtied);
            now += 100;
        }
        try testing.expect(anim.current_loop > 1);
        try testing.expect(s.nextAnimationDeadline() != null);
    }
}

test "storage: animation tick waits at the tail while loading" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddPlayableImage(&s, alloc, &t, 1, 1);

    const img = s.images.getPtr(1).?;
    img.anim.?.state = .loading;

    // Root -> frame 2, then wait at the tail rather than looping. Nothing
    // is scheduled, so no timer spins while we wait.
    try testing.expect(s.animationTick(100).dirtied);
    const r = s.animationTick(200);
    try testing.expect(!r.dirtied);
    try testing.expectEqual(@as(?u64, null), r.next_due_ms);
    try testing.expectEqual(@as(u32, 1), img.anim.?.current_frame);
    try testing.expectEqual(@as(u32, 0), img.anim.?.current_loop);

    // A new frame arriving resumes playback.
    const src: [4]u8 = @splat(9);
    _ = try s.addAnimationFrame(
        alloc,
        t.screens.active,
        1,
        .{ .composition_mode = .overwrite },
        &src,
        .rgba,
        1,
        1,
        200,
    );
    try testing.expect(s.animationTick(300).dirtied);
    try testing.expectEqual(@as(u32, 2), img.anim.?.current_frame);
    try testing.expectEqual(@as(u8, 9), img.renderData()[0]);
}

test "storage: animation deadline saturates" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddPlayableImage(&s, alloc, &t, 1, 1);

    // A timestamp near the end of the clock must not wrap around into a
    // deadline in the distant past.
    const anim = s.images.getPtr(1).?.anim.?;
    anim.last_frame_ms = std.math.maxInt(u64) - 1;
    try testing.expectEqual(
        @as(?u64, std.math.maxInt(u64)),
        s.nextAnimationDeadline(),
    );
}

test "storage: animation deadline picks the earliest" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 3, .cols = 3 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddPlayableImage(&s, alloc, &t, 1, 1);
    try testAddPlayableImage(&s, alloc, &t, 2, 1);

    // Two animations with different gaps: the timer must be armed for
    // whichever comes first.
    s.images.getPtr(1).?.anim.?.setGap(0, 500);
    s.images.getPtr(2).?.anim.?.setGap(0, 250);
    try testing.expectEqual(@as(?u64, 250), s.nextAnimationDeadline());

    // Ticking at that deadline advances only the one that's due.
    const r = s.animationTick(250);
    try testing.expect(r.dirtied);
    try testing.expectEqual(@as(u32, 0), s.images.getPtr(1).?.anim.?.current_frame);
    try testing.expectEqual(@as(u32, 1), s.images.getPtr(2).?.anim.?.current_frame);
    try testing.expectEqual(@as(?u64, 350), r.next_due_ms);
}

test "storage: eviction releases placement pins" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 10, .cols = 10 });
    defer t.deinit(alloc);
    const baseline = t.screens.active.pages.countTrackedPins();

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);

    // Evicting an image must release the pins its placements tracked, or
    // PageList accumulates them forever and every page update gets slower.
    for (0..8) |_| {
        try testAddImage(&s, alloc, t.screens.active, 1, 2, 2, .rgba, 1);
        try s.addPlacement(alloc, 1, 0, .{
            .location = .{ .pin = try trackPin(&t, .{ .x = 0, .y = 0 }) },
        });
        try s.addPlacement(alloc, 1, 0, .{
            .location = .{ .pin = try trackPin(&t, .{ .x = 1, .y = 1 }) },
        });

        // Lowering the limit below what's resident evicts, which must
        // release the evicted image's placement pins.
        try s.setLimit(alloc, t.screens.active, 8);
        try testing.expect(s.images.getPtr(1) == null);
        try testing.expectEqual(baseline, t.screens.active.pages.countTrackedPins());

        try s.setLimit(alloc, t.screens.active, 320 * 1000 * 1000);
    }
}

test "storage: animation reservation eviction releases placement pins" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 10, .cols = 10 });
    defer t.deinit(alloc);
    const baseline = t.screens.active.pages.countTrackedPins();

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);

    for (0..8) |_| {
        // Image 1 is the eviction victim, image 2 the frame target.
        try testAddImage(&s, alloc, t.screens.active, 1, 2, 2, .rgba, 1);
        try s.addPlacement(alloc, 1, 0, .{
            .location = .{ .pin = try trackPin(&t, .{ .x = 0, .y = 0 }) },
        });
        try testAddImage(&s, alloc, t.screens.active, 2, 2, 2, .rgba, 2);
        s.total_limit = 32;

        const src: [2 * 2 * 4]u8 = @splat(7);
        _ = try s.addAnimationFrame(alloc, t.screens.active, 2, .{}, &src, .rgba, 2, 2, 0);
        try testing.expect(s.images.getPtr(1) == null);

        // The evicted image's placement pin must be released too.
        try testing.expectEqual(baseline, t.screens.active.pages.countTrackedPins());

        s.total_limit = 320 * 1000 * 1000;
        s.delete(alloc, &t, .{ .all = true });
    }
}

/// The byte total must always equal what the images actually own.
fn expectAccountingExact(s: *const ImageStorage) !void {
    var sum: usize = 0;
    var it = s.images.iterator();
    while (it.next()) |kv| sum += kv.value_ptr.byteSize();
    try std.testing.expectEqual(sum, s.total_bytes);
}

test "storage: replacing an image does not evict for bytes it frees" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 10, .cols = 10 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);

    // Two 2x2 RGBA images exactly fill the limit.
    try testAddImage(&s, alloc, t.screens.active, 1, 2, 2, .rgba, 1);
    try testAddImage(&s, alloc, t.screens.active, 2, 2, 2, .rgba, 2);
    try s.addPlacement(alloc, 1, 0, .{
        .location = .{ .pin = try trackPin(&t, .{ .x = 0, .y = 0 }) },
    });
    s.total_limit = 32;
    try expectAccountingExact(&s);

    // Replacing image 2 with an identical-size image frees exactly what it
    // adds, so it must not evict anything: the net growth is zero.
    try testAddImage(&s, alloc, t.screens.active, 2, 2, 2, .rgba, 3);
    try testing.expect(s.images.getPtr(1) != null);
    try testing.expectEqual(@as(usize, 32), s.total_bytes);
    try expectAccountingExact(&s);

    // Replacing it with a smaller image must not evict either.
    try testAddImage(&s, alloc, t.screens.active, 2, 1, 1, .rgba, 4);
    try testing.expect(s.images.getPtr(1) != null);
    try testing.expectEqual(@as(usize, 20), s.total_bytes);
    try expectAccountingExact(&s);
}

test "storage: replacement evicts only the net growth" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 10, .cols = 10 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);

    // 16 + 16 = 32 of a 36 byte budget.
    try testAddImage(&s, alloc, t.screens.active, 1, 2, 2, .rgba, 1);
    try testAddImage(&s, alloc, t.screens.active, 2, 2, 2, .rgba, 2);
    s.total_limit = 36;

    // Replace image 2 (16 bytes) with a 36 byte image: net growth is 20,
    // which needs image 1 gone but nothing more.
    try testAddImage(&s, alloc, t.screens.active, 2, 3, 3, .rgba, 3);
    try testing.expect(s.images.getPtr(1) == null);
    try testing.expect(s.images.getPtr(2) != null);
    try testing.expectEqual(@as(usize, 36), s.total_bytes);
    try expectAccountingExact(&s);
}

test "storage: replacement never evicts its own target" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 10, .cols = 10 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);

    // The target is the only image, and it is the oldest candidate, so a
    // naive eviction plan would delete it (and its placements) to make
    // room for its own replacement.
    try testAddImage(&s, alloc, t.screens.active, 1, 2, 2, .rgba, 1);
    try s.addPlacement(alloc, 1, 0, .{
        .location = .{ .pin = try trackPin(&t, .{ .x = 0, .y = 0 }) },
    });
    s.total_limit = 16;

    try testAddImage(&s, alloc, t.screens.active, 1, 2, 2, .rgba, 9);
    try testing.expect(s.images.getPtr(1) != null);
    try testing.expectEqual(@as(u8, 9), s.images.getPtr(1).?.data[0]);
    try testing.expectEqual(@as(usize, 1), s.placements.count());
    try expectAccountingExact(&s);
}

test "storage: exact-limit eviction succeeds" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 10, .cols = 10 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);

    // Freeing exactly the required bytes is success, not failure.
    try testAddImage(&s, alloc, t.screens.active, 1, 2, 2, .rgba, 1);
    s.total_limit = 16;
    try testAddImage(&s, alloc, t.screens.active, 2, 2, 2, .rgba, 2);
    try testing.expect(s.images.getPtr(1) == null);
    try testing.expect(s.images.getPtr(2) != null);
    try testing.expectEqual(@as(usize, 16), s.total_bytes);
    try expectAccountingExact(&s);
}

test "storage: allocation failure leaves storage unchanged" {
    const testing = std.testing;
    var t = try terminal.Terminal.init(testing.allocator, .{ .rows = 10, .cols = 10 });
    defer t.deinit(testing.allocator);

    // Fail at every allocation in turn and prove that whatever fails, the
    // storage is left exactly as it was: same images, same bytes, same
    // frame count, same placements.
    var fail_index: usize = 0;
    while (fail_index < 64) : (fail_index += 1) {
        var failing: std.testing.FailingAllocator = .init(testing.allocator, .{
            .fail_index = fail_index,
        });
        const alloc = failing.allocator();

        var s: ImageStorage = .{};
        defer s.deinit(alloc, t.screens.active);

        // Build a baseline that must survive: if any of this fails we have
        // nothing to assert about yet, so just move on.
        testAddImage(&s, alloc, t.screens.active, 1, 2, 2, .rgba, 1) catch continue;
        const src: [2 * 2 * 4]u8 = @splat(7);
        _ = s.addAnimationFrame(alloc, t.screens.active, 1, .{}, &src, .rgba, 2, 2, 0) catch continue;

        const bytes = s.total_bytes;
        const images = s.images.count();
        const frames = s.images.getPtr(1).?.anim.?.frames.items.len;
        const anims = s.animation_count;
        const root0 = s.images.getPtr(1).?.data[0];

        // Now provoke more work with the same failing allocator. Each of
        // these may or may not fail depending on fail_index; either way
        // the invariants below must hold.
        _ = s.addAnimationFrame(alloc, t.screens.active, 1, .{}, &src, .rgba, 2, 2, 0) catch {};
        _ = s.composeAnimationFrames(alloc, t.screens.active, 1, .{
            .edit_frame = 2,
            .frame = 1,
            .width = 1,
            .height = 1,
        }, 0) catch {};
        testAddImage(&s, alloc, t.screens.active, 2, 2, 2, .rgba, 2) catch {};

        try expectAccountingExact(&s);
        if (s.images.getPtr(1)) |img| {
            // The original image must never be left partially mutated.
            try testing.expect(img.data.len == 16);
            try testing.expect(img.anim.?.frames.items.len >= frames);
            _ = root0;
        }
        try testing.expect(s.total_bytes >= bytes);
        try testing.expect(s.images.count() >= images);
        try testing.expect(s.animation_count >= anims);
    }
}

test "storage: accounting stays exact across mutations" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 10, .cols = 10 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);

    const src: [2 * 2 * 4]u8 = @splat(7);
    const one: [4]u8 = @splat(3);

    try testAddImage(&s, alloc, t.screens.active, 1, 2, 2, .rgb, 1);
    try expectAccountingExact(&s);

    // Root widening changes the byte total; it must stay exact.
    _ = try s.addAnimationFrame(alloc, t.screens.active, 1, .{}, &src, .rgba, 2, 2, 0);
    try expectAccountingExact(&s);

    _ = try s.addAnimationFrame(alloc, t.screens.active, 1, .{ .edit_frame = 2 }, &one, .rgba, 1, 1, 0);
    try expectAccountingExact(&s);

    _ = try s.composeAnimationFrames(alloc, t.screens.active, 1, .{
        .edit_frame = 2,
        .frame = 1,
        .width = 1,
        .height = 1,
    }, 0);
    try expectAccountingExact(&s);

    _ = s.deleteAnimationFrame(alloc, &t, .{ .image_id = 1, .frame = 1 }, 0);
    try expectAccountingExact(&s);

    // Replacement, then delete.
    try testAddImage(&s, alloc, t.screens.active, 1, 3, 3, .rgba, 4);
    try expectAccountingExact(&s);

    s.delete(alloc, &t, .{ .all = true });
    try expectAccountingExact(&s);
    try testing.expectEqual(@as(usize, 0), s.total_bytes);
    try testing.expectEqual(@as(usize, 0), s.animation_count);
}

/// An allocator that records the peak live bytes it handed out, so tests
/// can assert an operation's transient cost rather than trusting the code.
const PeakAllocator = struct {
    parent: Allocator,
    live: usize = 0,
    peak: usize = 0,

    fn allocator(self: *PeakAllocator) Allocator {
        return .{ .ptr = self, .vtable = &.{
            .alloc = alloc,
            .resize = resize,
            .remap = remap,
            .free = free,
        } };
    }

    fn alloc(ctx: *anyopaque, len: usize, a: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *PeakAllocator = @ptrCast(@alignCast(ctx));
        const p = self.parent.rawAlloc(len, a, ra) orelse return null;
        self.live += len;
        self.peak = @max(self.peak, self.live);
        return p;
    }

    fn resize(ctx: *anyopaque, buf: []u8, a: std.mem.Alignment, new_len: usize, ra: usize) bool {
        const self: *PeakAllocator = @ptrCast(@alignCast(ctx));
        if (!self.parent.rawResize(buf, a, new_len, ra)) return false;
        self.live = self.live - buf.len + new_len;
        self.peak = @max(self.peak, self.live);
        return true;
    }

    fn remap(ctx: *anyopaque, buf: []u8, a: std.mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        const self: *PeakAllocator = @ptrCast(@alignCast(ctx));
        const p = self.parent.rawRemap(buf, a, new_len, ra) orelse return null;
        self.live = self.live - buf.len + new_len;
        self.peak = @max(self.peak, self.live);
        return p;
    }

    fn free(ctx: *anyopaque, buf: []u8, a: std.mem.Alignment, ra: usize) void {
        const self: *PeakAllocator = @ptrCast(@alignCast(ctx));
        self.parent.rawFree(buf, a, ra);
        self.live -= buf.len;
    }
};

test "storage: frame ingestion transient peak is bounded" {
    const testing = std.testing;
    var t = try terminal.Terminal.init(testing.allocator, .{ .rows = 10, .cols = 10 });
    defer t.deinit(testing.allocator);

    var peak: PeakAllocator = .{ .parent = testing.allocator };
    const alloc = peak.allocator();

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);

    const dim = 64;
    const canvas = dim * dim * 4;
    try testAddImage(&s, alloc, t.screens.active, 1, dim, dim, .rgba, 0);

    const src = try testing.allocator.alloc(u8, canvas);
    defer testing.allocator.free(src);
    @memset(src, 0x40);

    // Appending a frame: the resident image and the new frame are both
    // persistent, so the transient part on top is what matters. An RGBA
    // source is borrowed rather than copied, so a single canvas covers it.
    const before = peak.live;
    peak.peak = peak.live;
    _ = try s.addAnimationFrame(alloc, t.screens.active, 1, .{}, src, .rgba, dim, dim, 0);
    const append_transient = peak.peak - before;
    try testing.expect(append_transient <= 2 * canvas);

    // Editing the current (root) frame: one widened/copied canvas, and
    // again no copy of the RGBA source.
    peak.peak = peak.live;
    const live_before_edit = peak.live;
    _ = try s.addAnimationFrame(
        alloc,
        t.screens.active,
        1,
        .{ .edit_frame = 1 },
        src,
        .rgba,
        dim,
        dim,
        0,
    );
    try testing.expect(peak.peak - live_before_edit <= 2 * canvas);

    // An RGB source must widen, which is one rect-sized copy on top.
    const rgb = try testing.allocator.alloc(u8, dim * dim * 3);
    defer testing.allocator.free(rgb);
    @memset(rgb, 0x20);
    peak.peak = peak.live;
    const live_before_rgb = peak.live;
    _ = try s.addAnimationFrame(alloc, t.screens.active, 1, .{}, rgb, .rgb, dim, dim, 0);
    try testing.expect(peak.peak - live_before_rgb <= 3 * canvas);
}

test "storage: transient budget rejects oversized operations" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 10, .cols = 10 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);
    try testAddImage(&s, alloc, t.screens.active, 1, 8, 8, .rgba, 1);

    // An edit persists nothing, so only the transient budget can stop it
    // from allocating canvases far larger than the quota.
    const bytes_before = s.total_bytes;
    s.total_limit = 64;

    const src: [4]u8 = @splat(9);
    try testing.expectError(error.OutOfSpace, s.addAnimationFrame(
        alloc,
        t.screens.active,
        1,
        .{ .edit_frame = 1 },
        &src,
        .rgba,
        1,
        1,
        0,
    ));
    try testing.expectError(error.OutOfSpace, s.composeAnimationFrames(alloc, t.screens.active, 1, .{
        .edit_frame = 1,
        .frame = 1,
        .width = 1,
        .height = 1,
        .left_edge = 1,
    }, 0));

    // Rejected before allocating: nothing moved.
    try testing.expectEqual(bytes_before, s.total_bytes);
    try testing.expect(s.images.getPtr(1).?.anim == null);
    try expectAccountingExact(&s);
}

test "storage: malformed dimensions do not wrap" {
    const testing = std.testing;
    const alloc = testing.allocator;
    var t = try terminal.Terminal.init(alloc, .{ .rows = 10, .cols = 10 });
    defer t.deinit(alloc);

    var s: ImageStorage = .{};
    defer s.deinit(alloc, t.screens.active);

    // An image whose dimensions multiply past usize must be refused by the
    // checked size math rather than wrapping into a small allocation that
    // the pixel loops then run off the end of. Such an image can only
    // exist if it was never validated on the way in, so build it directly.
    try s.images.put(alloc, 1, .{
        .id = 1,
        .width = std.math.maxInt(u32),
        .height = std.math.maxInt(u32),
        .format = .rgba,
        .data = "",
    });
    defer _ = s.images.remove(1);

    const src: [4]u8 = @splat(1);
    try testing.expectError(error.OutOfSpace, s.addAnimationFrame(
        alloc,
        t.screens.active,
        1,
        .{},
        &src,
        .rgba,
        1,
        1,
        0,
    ));
    // Disjoint rectangles, so this gets past the overlap check and
    // actually reaches the size arithmetic.
    try testing.expectError(error.OutOfSpace, s.composeAnimationFrames(alloc, t.screens.active, 1, .{
        .edit_frame = 1,
        .frame = 1,
        .width = 1,
        .height = 1,
        .left_edge = 1,
    }, 0));
}
