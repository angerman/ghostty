//! Animation state for the Kitty graphics protocol.
//!
//! This file is deliberately a dependency leaf: an Image owns an Animation,
//! so this must never import graphics_image.zig or graphics_storage.zig or
//! we'd have an import cycle. Everything here is either plain animation
//! state or a pure pixel operation, and everything that needs to reach the
//! wider storage (quota accounting, generation stamping, scanning for due
//! animations) lives in graphics_storage.zig instead.
//!
//! Frames are stored as fully-composed, full-canvas RGBA buffers. Kitty
//! instead stores partial frames that reference a base frame and coalesces
//! them on demand, backed by a disk cache. We trade the memory for a much
//! simpler implementation; see the notes in graphics.zig.

const std = @import("std");
const Allocator = std.mem.Allocator;

const fastmem = @import("../../fastmem.zig");
const command = @import("graphics_command.zig");

const Background = command.AnimationFrameLoading.Background;
const CompositionMode = command.CompositionMode;
const Format = command.Transmission.Format;

/// The default gap for a new frame, in milliseconds. Taken from Kitty.
pub const default_gap_ms: u32 = 40;

/// The animation state of a single image.
///
/// This is heap-allocated and hung off Image.anim only for images that
/// actually take part in animation, since the vast majority never do.
///
/// Note that an Image is copied by value out of the storage map, which is
/// safe to read through because this is behind a pointer. All *mutation*
/// must go through ImageStorage.images.getPtr, or it would be written to a
/// temporary copy and lost.
pub const Animation = struct {
    /// The frames beyond the root frame. The root frame is the image's own
    /// pixel data, so frames.items[i] is protocol frame number i+2 and the
    /// protocol's frame count is frames.items.len + 1.
    frames: std.ArrayListUnmanaged(Frame) = .{},

    /// The playback state. Note that "loading" means running, but waiting
    /// at the last frame for more frames to arrive instead of looping.
    state: State = .stopped,

    /// The currently displayed frame, 0-based, where 0 is the root frame.
    current_frame: u32 = 0,

    /// The gap of the root frame. The protocol defaults this to 0 (unlike
    /// new frames, which default to default_gap_ms), so an animation whose
    /// root should be shown for a while needs an explicit "a=a,r=1,z=..".
    root_gap_ms: u32 = 0,

    /// The number of frames (root included) whose gap is nonzero, i.e. the
    /// number of frames that are actually shown rather than skipped.
    ///
    /// This is maintained incrementally by setGap/appendFrame/removeFrame
    /// so that playback can check "does this animation have any visible
    /// frame at all" in O(1). That check is not an optimization but a
    /// correctness guard: advance() below would spin forever on an
    /// animation whose every frame is gapless. It is equivalent to Kitty's
    /// cached animation_duration being nonzero, since gaps are unsigned.
    nonzero_gap_count: u32 = 0,

    /// The number of loops to play, where 0 means loop forever. This is
    /// already adjusted from the protocol's "v" key, which is off by one.
    max_loops: u32 = 0,
    current_loop: u32 = 0,

    /// When the current frame was first shown, in milliseconds on the
    /// clock owned by the image's ImageStorage. Only meaningful while
    /// running; see ImageStorage.animationNowMs for why the clock must be
    /// shared rather than sampled independently.
    last_frame_ms: u64 = 0,

    pub const State = enum { stopped, loading, running };

    /// A single frame beyond the root. The data is always a fully-composed
    /// full-canvas RGBA buffer, so its length is always width * height * 4
    /// for the owning image.
    pub const Frame = struct {
        data: []u8,
        gap_ms: u32,
    };

    pub fn deinit(self: *Animation, alloc: Allocator) void {
        for (self.frames.items) |frame| alloc.free(frame.data);
        self.frames.deinit(alloc);
    }

    /// The total number of frames including the root frame. This is the
    /// protocol's frame count, so a valid 1-based frame number "r" is in
    /// the range [1, frameCount()].
    pub fn frameCount(self: *const Animation) usize {
        return self.frames.items.len + 1;
    }

    /// The gap of the given 0-based frame index, where 0 is the root.
    pub fn gapOf(self: *const Animation, index: u32) u32 {
        if (index == 0) return self.root_gap_ms;
        return self.frames.items[index - 1].gap_ms;
    }

    fn gapPtr(self: *Animation, index: u32) *u32 {
        if (index == 0) return &self.root_gap_ms;
        return &self.frames.items[index - 1].gap_ms;
    }

    /// Set the gap of the given 0-based frame index, keeping
    /// nonzero_gap_count exact. This is the only way gaps may be changed.
    pub fn setGap(self: *Animation, index: u32, gap_ms: u32) void {
        const ptr = self.gapPtr(index);
        if ((ptr.* != 0) != (gap_ms != 0)) {
            if (gap_ms != 0) {
                self.nonzero_gap_count += 1;
            } else {
                self.nonzero_gap_count -= 1;
            }
        }
        ptr.* = gap_ms;
    }

    /// Append a frame, taking ownership of its data. The caller must have
    /// reserved capacity in advance so that this cannot fail: appending is
    /// the last step of a transactional frame creation.
    pub fn appendFrameAssumeCapacity(self: *Animation, frame: Frame) void {
        self.frames.appendAssumeCapacity(frame);
        if (frame.gap_ms != 0) self.nonzero_gap_count += 1;
    }

    /// Remove the frame at the given index into `frames` (i.e. protocol
    /// frame index+2), returning it. The caller owns the returned data.
    pub fn removeFrame(self: *Animation, index: usize) Frame {
        const frame = self.frames.orderedRemove(index);
        if (frame.gap_ms != 0) self.nonzero_gap_count -= 1;
        return frame;
    }

    /// Advance to the next frame that should actually be shown, following
    /// Kitty's rules: gapless frames are skipped instantly, and wrapping
    /// past the last frame either waits for more frames (loading state) or
    /// counts a loop and possibly stops.
    ///
    /// Returns true if a new frame became current, in which case the caller
    /// must stamp last_frame_ms and mark the image's pixels changed. False
    /// means the animation has nothing new to show: it is either waiting
    /// for more frames, or it has just finished its final loop.
    ///
    /// The caller must only call this when the animation is actually due,
    /// and must have checked nonzero_gap_count > 0 first, or a fully
    /// gapless animation would loop here forever looking for a frame to
    /// show.
    pub fn advance(self: *Animation) bool {
        std.debug.assert(self.nonzero_gap_count > 0);

        const total = self.frameCount();
        while (true) {
            const next: u32 = @intCast((self.current_frame + 1) % total);
            if (next == 0) {
                // We wrapped past the last frame.
                if (self.state == .loading) return false;

                // Note that the loop counter is incremented on the
                // prospective wrap, before the bound check, so reaching the
                // bound leaves the last frame current rather than wrapping
                // back to the root.
                self.current_loop += 1;
                if (self.max_loops != 0 and self.current_loop >= self.max_loops) {
                    return false;
                }
            }

            self.current_frame = next;
            if (self.gapOf(next) != 0) return true;
        }
    }
};

/// The gap to store for a "z" value on an existing frame. A negative gap
/// means the frame is gapless, which we store as a zero gap.
pub fn resolveGap(z: i32) u32 {
    return if (z < 0) 0 else @intCast(z);
}

/// The gap to store for a "z" value on a newly created frame. Unlike an
/// edit, where zero means "leave the gap alone", zero on a new frame means
/// the protocol's default gap.
pub fn resolveNewGap(z: i32) u32 {
    if (z == 0) return default_gap_ms;
    return resolveGap(z);
}

/// True if two equally-sized rectangles overlap.
pub fn rectsOverlap(ax: u32, ay: u32, bx: u32, by: u32, w: u32, h: u32) bool {
    return ax < @as(u64, bx) + w and bx < @as(u64, ax) + w and
        ay < @as(u64, by) + h and by < @as(u64, ay) + h;
}

/// Fill a full-canvas RGBA buffer with a background color.
pub fn fill(canvas: []u8, bg: Background) void {
    const px: [4]u8 = .{ bg.r, bg.g, bg.b, bg.a };

    // Transparent black is overwhelmingly the common case (it's the
    // protocol default), and memset is much faster than the pixel loop.
    if (@as(u32, @bitCast(px)) == 0) {
        @memset(canvas, 0);
        return;
    }

    std.debug.assert(canvas.len % 4 == 0);
    var i: usize = 0;
    while (i < canvas.len) : (i += 4) canvas[i..][0..4].* = px;
}

/// Alpha-blend a source pixel over a destination pixel in place.
///
/// This reproduces Kitty's blend exactly, including its use of f32 and its
/// truncation back to bytes, so that a frame composed by Ghostty is
/// pixel-identical to the same frame composed by Kitty.
inline fn alphaBlend(dst: *[4]u8, src: *const [4]u8) void {
    // A fully transparent source contributes nothing at all, not even to
    // the destination alpha.
    if (src[3] == 0) return;

    const src_a: f32 = @as(f32, @floatFromInt(src[3])) / 255.0;
    const dst_a: f32 = @as(f32, @floatFromInt(dst[3])) / 255.0;

    // The source alpha is nonzero, so the output alpha is at least as
    // large and the division below can never be by zero.
    const out_a: f32 = src_a + dst_a * (1 - src_a);

    inline for (0..3) |i| {
        const s: f32 = @floatFromInt(src[i]);
        const d: f32 = @floatFromInt(dst[i]);
        dst[i] = @intFromFloat((s * src_a + d * dst_a * (1 - src_a)) / out_a);
    }

    // Written last because the blend above reads the old destination alpha.
    dst[3] = @intFromFloat(255.0 * out_a);
}

/// Compose one row of RGBA pixels onto another.
fn composeRow(dst: []u8, src: []const u8, mode: CompositionMode) void {
    std.debug.assert(dst.len == src.len);
    switch (mode) {
        .overwrite => fastmem.copy(u8, dst, src),
        .alpha_blend => {
            var i: usize = 0;
            while (i < dst.len) : (i += 4) {
                alphaBlend(dst[i..][0..4], src[i..][0..4]);
            }
        },
    }
}

/// Compose a transmitted rectangle onto a frame canvas, as used by "a=f".
///
/// The source is a tightly packed rect_width x rect_height RGBA rectangle,
/// and it is placed at (x, y) within a canvas of canvas_width x
/// canvas_height. Unlike "a=c", the protocol does not reject a rectangle
/// that runs off the canvas here: it is clipped at the right and bottom
/// edges, and a rectangle placed entirely off the canvas composes nothing
/// at all.
pub fn composeTransmitted(
    canvas: []u8,
    canvas_width: u32,
    canvas_height: u32,
    src: []const u8,
    rect_width: u32,
    rect_height: u32,
    x: u32,
    y: u32,
    mode: CompositionMode,
) void {
    // Clip once up front rather than per pixel.
    if (x >= canvas_width or y >= canvas_height) return;
    const w = @min(rect_width, canvas_width - x);
    const h = @min(rect_height, canvas_height - y);
    if (w == 0 or h == 0) return;

    const row_len = w * 4;
    for (0..h) |row| {
        const dst_start = ((y + row) * canvas_width + x) * 4;
        const src_start = row * rect_width * 4;
        composeRow(
            canvas[dst_start..][0..row_len],
            src[src_start..][0..row_len],
            mode,
        );
    }
}

/// Compose a rectangle from one frame onto another, as used by "a=c".
///
/// Both buffers are full canvases of the same image, so both use the image
/// width as their stride. The rectangles are validated by the caller and
/// are never clipped here.
///
/// `dst` and `src` must not overlap. When the protocol's source and
/// destination frames are the same frame, the caller composes into a copy,
/// which also gives the transactional behavior we want.
pub fn composeFrames(
    dst: []u8,
    src: []const u8,
    image_width: u32,
    w: u32,
    h: u32,
    src_x: u32,
    src_y: u32,
    dst_x: u32,
    dst_y: u32,
    mode: CompositionMode,
) void {
    if (w == 0 or h == 0) return;

    const row_len = w * 4;
    for (0..h) |row| {
        const dst_start = ((dst_y + row) * image_width + dst_x) * 4;
        const src_start = ((src_y + row) * image_width + src_x) * 4;
        composeRow(
            dst[dst_start..][0..row_len],
            src[src_start..][0..row_len],
            mode,
        );
    }
}

/// Allocate an RGBA copy of a pixel buffer in any of the stored formats.
///
/// Composition is always RGBA-on-RGBA, so both the root frame and every
/// transmitted rectangle are widened to RGBA before use. Gray formats
/// replicate the luminance across the color channels and formats without
/// an alpha channel become fully opaque, matching Kitty.
///
/// The caller owns the result. This never mutates storage accounting; the
/// caller reserves and commits the byte delta transactionally.
pub fn allocRGBA(
    alloc: Allocator,
    data: []const u8,
    format: Format,
) Allocator.Error![]u8 {
    // A stored image is never still PNG-encoded: loading decodes it to
    // RGBA before the image is completed.
    std.debug.assert(format != .png);

    const bpp = command.Transmission.formatBpp(format);
    const pixels = data.len / bpp;
    const out = try alloc.alloc(u8, pixels * 4);
    errdefer alloc.free(out);

    switch (format) {
        .png => unreachable,
        .rgba => fastmem.copy(u8, out, data[0 .. pixels * 4]),
        .rgb => for (0..pixels) |i| {
            out[i * 4 ..][0..4].* = .{ data[i * 3], data[i * 3 + 1], data[i * 3 + 2], 255 };
        },
        .gray_alpha => for (0..pixels) |i| {
            const g = data[i * 2];
            out[i * 4 ..][0..4].* = .{ g, g, g, data[i * 2 + 1] };
        },
        .gray => for (0..pixels) |i| {
            const g = data[i];
            out[i * 4 ..][0..4].* = .{ g, g, g, 255 };
        },
    }

    return out;
}

test "animation: gap bookkeeping" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var anim: Animation = .{};
    defer anim.deinit(alloc);

    // The root frame starts gapless, so nothing is visible yet.
    try testing.expectEqual(@as(u32, 0), anim.nonzero_gap_count);
    try testing.expectEqual(@as(usize, 1), anim.frameCount());

    anim.setGap(0, 100);
    try testing.expectEqual(@as(u32, 1), anim.nonzero_gap_count);
    try testing.expectEqual(@as(u32, 100), anim.gapOf(0));

    // Setting an already-nonzero gap again must not double count.
    anim.setGap(0, 50);
    try testing.expectEqual(@as(u32, 1), anim.nonzero_gap_count);

    anim.setGap(0, 0);
    try testing.expectEqual(@as(u32, 0), anim.nonzero_gap_count);

    // Appending and removing frames keeps the count exact.
    try anim.frames.ensureUnusedCapacity(alloc, 2);
    anim.appendFrameAssumeCapacity(.{ .data = try alloc.alloc(u8, 4), .gap_ms = 40 });
    anim.appendFrameAssumeCapacity(.{ .data = try alloc.alloc(u8, 4), .gap_ms = 0 });
    try testing.expectEqual(@as(u32, 1), anim.nonzero_gap_count);
    try testing.expectEqual(@as(usize, 3), anim.frameCount());

    anim.setGap(2, 20);
    try testing.expectEqual(@as(u32, 2), anim.nonzero_gap_count);

    const removed = anim.removeFrame(0);
    alloc.free(removed.data);
    try testing.expectEqual(@as(u32, 1), anim.nonzero_gap_count);
    try testing.expectEqual(@as(u32, 20), anim.gapOf(1));
}

test "animation: advance skips gapless frames" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var anim: Animation = .{ .state = .running };
    defer anim.deinit(alloc);
    anim.setGap(0, 100);

    // Frame 2 is gapless and must be skipped straight over.
    try anim.frames.ensureUnusedCapacity(alloc, 2);
    anim.appendFrameAssumeCapacity(.{ .data = try alloc.alloc(u8, 4), .gap_ms = 0 });
    anim.appendFrameAssumeCapacity(.{ .data = try alloc.alloc(u8, 4), .gap_ms = 100 });

    try testing.expect(anim.advance());
    try testing.expectEqual(@as(u32, 2), anim.current_frame);

    // Wrapping back to the root counts a loop.
    try testing.expect(anim.advance());
    try testing.expectEqual(@as(u32, 0), anim.current_frame);
    try testing.expectEqual(@as(u32, 1), anim.current_loop);
}

test "animation: advance waits at the tail while loading" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var anim: Animation = .{ .state = .loading };
    defer anim.deinit(alloc);
    anim.setGap(0, 100);

    try anim.frames.ensureUnusedCapacity(alloc, 2);
    anim.appendFrameAssumeCapacity(.{ .data = try alloc.alloc(u8, 4), .gap_ms = 100 });

    try testing.expect(anim.advance());
    try testing.expectEqual(@as(u32, 1), anim.current_frame);

    // At the last frame we wait rather than loop, and don't count a loop.
    try testing.expect(!anim.advance());
    try testing.expectEqual(@as(u32, 1), anim.current_frame);
    try testing.expectEqual(@as(u32, 0), anim.current_loop);

    // Once another frame arrives, playback continues.
    anim.appendFrameAssumeCapacity(.{ .data = try alloc.alloc(u8, 4), .gap_ms = 100 });
    try testing.expect(anim.advance());
    try testing.expectEqual(@as(u32, 2), anim.current_frame);
}

test "animation: advance exhausts loops" {
    const testing = std.testing;
    const alloc = testing.allocator;

    // v=2 means max_loops=1: a single pass that ends on the last frame.
    var anim: Animation = .{ .state = .running, .max_loops = 1 };
    defer anim.deinit(alloc);
    anim.setGap(0, 100);

    try anim.frames.ensureUnusedCapacity(alloc, 1);
    anim.appendFrameAssumeCapacity(.{ .data = try alloc.alloc(u8, 4), .gap_ms = 100 });

    try testing.expect(anim.advance());
    try testing.expectEqual(@as(u32, 1), anim.current_frame);

    // The wrap is counted but not committed, so the last frame stays up.
    try testing.expect(!anim.advance());
    try testing.expectEqual(@as(u32, 1), anim.current_frame);
    try testing.expectEqual(@as(u32, 1), anim.current_loop);
}

test "animation: gap resolution" {
    const testing = std.testing;

    // A new frame defaults to the protocol default gap, an edit keeps its
    // gap, and a negative gap means gapless either way.
    try testing.expectEqual(default_gap_ms, resolveNewGap(0));
    try testing.expectEqual(@as(u32, 100), resolveNewGap(100));
    try testing.expectEqual(@as(u32, 0), resolveNewGap(-1));
    try testing.expectEqual(@as(u32, 0), resolveGap(0));
    try testing.expectEqual(@as(u32, 100), resolveGap(100));
    try testing.expectEqual(@as(u32, 0), resolveGap(-5));
}

test "animation: rectsOverlap" {
    const testing = std.testing;

    try testing.expect(rectsOverlap(0, 0, 0, 0, 2, 2));
    try testing.expect(rectsOverlap(0, 0, 1, 1, 2, 2));

    // Adjacent but not overlapping.
    try testing.expect(!rectsOverlap(0, 0, 2, 0, 2, 2));
    try testing.expect(!rectsOverlap(0, 0, 0, 2, 2, 2));
}

test "animation: fill" {
    const testing = std.testing;

    var canvas: [8]u8 = undefined;
    fill(&canvas, .{ .r = 1, .g = 2, .b = 3, .a = 4 });
    try testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 1, 2, 3, 4 }, &canvas);

    fill(&canvas, .{});
    try testing.expectEqualSlices(u8, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, &canvas);
}

test "animation: alpha blend" {
    const testing = std.testing;

    // A transparent source leaves the destination completely untouched.
    {
        var dst: [4]u8 = .{ 10, 20, 30, 40 };
        alphaBlend(&dst, &.{ 255, 255, 255, 0 });
        try testing.expectEqualSlices(u8, &.{ 10, 20, 30, 40 }, &dst);
    }

    // An opaque source replaces the destination.
    {
        var dst: [4]u8 = .{ 10, 20, 30, 40 };
        alphaBlend(&dst, &.{ 1, 2, 3, 255 });
        try testing.expectEqualSlices(u8, &.{ 1, 2, 3, 255 }, &dst);
    }

    // Blending onto a transparent destination keeps the source color.
    {
        var dst: [4]u8 = .{ 99, 99, 99, 0 };
        alphaBlend(&dst, &.{ 10, 20, 30, 128 });
        try testing.expectEqualSlices(u8, &.{ 10, 20, 30, 128 }, &dst);
    }

    // A mid-alpha blend, pinned to the exact bytes. Half-red over green
    // with src_a = 128/255 gives out_a = 1, so each channel works out to
    // src * 128/255 + dst * (127/255).
    //
    // Green lands on 126 rather than the 127 the real arithmetic gives:
    // 128/255 is not representable in f32 and rounds up, so the result is
    // 126.99999 and the conversion to a byte truncates. That's not a bug
    // to fix -- Kitty computes this the same way, and matching it byte for
    // byte is the point. Using f64, or rounding instead of truncating,
    // would drift away from Kitty here.
    {
        var dst: [4]u8 = .{ 0, 255, 0, 255 };
        alphaBlend(&dst, &.{ 255, 0, 0, 128 });
        try testing.expectEqualSlices(u8, &.{ 128, 126, 0, 255 }, &dst);
    }
}

test "animation: composeTransmitted clips" {
    const testing = std.testing;

    // A 2x2 canvas with a 2x2 source placed at (1,1) keeps only one pixel.
    var canvas: [2 * 2 * 4]u8 = @splat(0);
    const src: [2 * 2 * 4]u8 = @splat(9);
    composeTransmitted(&canvas, 2, 2, &src, 2, 2, 1, 1, .overwrite);
    try testing.expectEqualSlices(u8, &.{
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 9, 9, 9, 9,
    }, &canvas);

    // Wholly off-canvas composes nothing and is not an error.
    var canvas2: [2 * 2 * 4]u8 = @splat(0);
    const zeroes: [2 * 2 * 4]u8 = @splat(0);
    composeTransmitted(&canvas2, 2, 2, &src, 2, 2, 2, 0, .overwrite);
    try testing.expectEqualSlices(u8, &zeroes, &canvas2);
}

test "animation: composeFrames" {
    const testing = std.testing;

    // Copy the bottom-right pixel of the source to the top-left of dest.
    var dst: [2 * 2 * 4]u8 = @splat(0);
    const src: [2 * 2 * 4]u8 = .{
        1, 1, 1, 1, 2, 2, 2, 2,
        3, 3, 3, 3, 4, 4, 4, 4,
    };
    composeFrames(&dst, &src, 2, 1, 1, 1, 1, 0, 0, .overwrite);
    try testing.expectEqualSlices(u8, &.{
        4, 4, 4, 4, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    }, &dst);
}

test "animation: allocRGBA widening" {
    const testing = std.testing;
    const alloc = testing.allocator;

    {
        const out = try allocRGBA(alloc, &.{ 1, 2, 3 }, .rgb);
        defer alloc.free(out);
        try testing.expectEqualSlices(u8, &.{ 1, 2, 3, 255 }, out);
    }

    {
        const out = try allocRGBA(alloc, &.{ 7, 8 }, .gray_alpha);
        defer alloc.free(out);
        try testing.expectEqualSlices(u8, &.{ 7, 7, 7, 8 }, out);
    }

    {
        const out = try allocRGBA(alloc, &.{5}, .gray);
        defer alloc.free(out);
        try testing.expectEqualSlices(u8, &.{ 5, 5, 5, 255 }, out);
    }

    {
        const out = try allocRGBA(alloc, &.{ 1, 2, 3, 4 }, .rgba);
        defer alloc.free(out);
        try testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, out);
    }
}
