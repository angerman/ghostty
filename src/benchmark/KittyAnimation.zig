//! This benchmark measures the Kitty graphics animation paths that run on
//! every displayed frame.
//!
//! The costs worth knowing here are asymmetric. Scheduling runs on *every*
//! renderer frame whether or not anything animates, so its idle cost is
//! paid ~120 times a second forever and has to be ~nothing. Ingestion runs
//! once per transmitted frame, which for a preloaded GIF is a one-off but
//! for a live VNC or video client is per displayed frame, so it lands
//! straight in the frame budget.
//!
//! The modes deliberately include adversarial shapes -- thousands of
//! placements, matches at the end of iteration order -- because the
//! interesting question is not the happy path but whether the scan degrades
//! with things that have nothing to do with the animation.
const KittyAnimation = @This();

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Benchmark = @import("Benchmark.zig");
const terminal = @import("../terminal/main.zig");
const kitty = terminal.kitty.graphics;

const log = std.log.scoped(.@"kitty-animation-bench");

opts: Options,

/// Built in setup, torn down after.
term: ?*terminal.Terminal = null,
storage: ?*kitty.ImageStorage = null,
alloc: Allocator = undefined,

/// Source pixels for ingestion modes.
src: []u8 = &.{},

/// A monotonically advancing fake clock, so the tick modes are
/// deterministic and independent of how fast the machine runs them.
now_ms: u64 = 0,

pub const Options = struct {
    /// What to measure.
    mode: Mode = .@"schedule-idle",

    /// The number of animated images to create.
    images: usize = 1,

    /// The number of placements to create. Spread across the images, so
    /// `placements` >> `images` is the adversarial case: it is what the
    /// scan used to walk for every animated image.
    placements: usize = 1,

    /// Image dimensions for the ingestion modes.
    width: u32 = 512,
    height: u32 = 512,

    /// The fraction of the image an ingestion mode transmits, as a
    /// percent. 100 is a full frame; 10 approximates a VNC-sized update.
    damage_percent: u32 = 100,

    /// How many operations one benchmark step performs.
    ///
    /// The harness runs a step once and hyperfine times the process, so a
    /// step has to be big enough to measure. Divide the reported time by
    /// this (after subtracting the same run at iterations=0, which is the
    /// setup and process cost) to get the per-operation figure.
    iterations: usize = 100_000,
};

pub const Mode = enum {
    /// The complete idle renderer scheduling sequence: the deadline
    /// lookup *and* the clock read, in the order updateFrame does them,
    /// with nothing animating. This is the always-on cost.
    @"schedule-idle",

    /// The same sequence with animations present and due later. This is
    /// what an active animation costs every frame it is *not* due.
    @"schedule-active",

    /// A tick that actually advances a frame.
    @"tick-advance",

    /// A tick where nothing is due yet.
    @"tick-not-due",

    /// Append a new frame: allocate a canvas, fill, compose, account.
    @"ingest-append",

    /// Edit the current frame: the live VNC/video shape, where the cost
    /// lands in the frame budget rather than in load time.
    @"ingest-edit",
};

pub fn create(alloc: Allocator, opts: Options) !*KittyAnimation {
    const ptr = try alloc.create(KittyAnimation);
    errdefer alloc.destroy(ptr);
    ptr.* = .{ .opts = opts, .alloc = alloc };
    return ptr;
}

pub fn destroy(self: *KittyAnimation, alloc: Allocator) void {
    alloc.destroy(self);
}

pub fn benchmark(self: *KittyAnimation) Benchmark {
    return .init(self, .{
        .stepFn = switch (self.opts.mode) {
            .@"schedule-idle", .@"schedule-active" => stepSchedule,
            .@"tick-advance", .@"tick-not-due" => stepTick,
            .@"ingest-append" => stepIngestAppend,
            .@"ingest-edit" => stepIngestEdit,
        },
        .setupFn = setup,
        .teardownFn = teardown,
    });
}

fn setup(ptr: *anyopaque) Benchmark.Error!void {
    const self: *KittyAnimation = @ptrCast(@alignCast(ptr));
    const alloc = self.alloc;

    const term = alloc.create(terminal.Terminal) catch return error.BenchmarkFailed;
    errdefer alloc.destroy(term);
    term.* = terminal.Terminal.init(alloc, .{
        .rows = 100,
        .cols = 200,
    }) catch return error.BenchmarkFailed;
    term.width_px = 2000;
    term.height_px = 1000;
    self.term = term;

    const storage = &term.screens.active.kitty_images;
    self.storage = storage;

    const animated = self.opts.mode != .@"schedule-idle";
    const w = self.opts.width;
    const h = self.opts.height;

    // Ingestion works on one image; scheduling wants many.
    const image_count: usize = switch (self.opts.mode) {
        .@"ingest-append", .@"ingest-edit" => 1,
        else => @max(self.opts.images, 1),
    };

    for (0..image_count) |i| {
        const id: u32 = @intCast(i + 1);
        const data = alloc.alloc(u8, @as(usize, w) * h * 4) catch
            return error.BenchmarkFailed;
        @memset(data, 0x40);
        storage.addImage(alloc, term.screens.active, .{
            .id = id,
            .width = w,
            .height = h,
            .format = .rgba,
            .data = data,
        }) catch {
            alloc.free(data);
            return error.BenchmarkFailed;
        };
    }

    // Placements. These are what the pre-fix scan walked for every
    // animated image, so the adversarial counts matter: put the matching
    // ones last so a linear scan pays its worst case.
    const placement_count = @max(self.opts.placements, 1);
    for (0..placement_count) |i| {
        const id: u32 = @intCast((i % image_count) + 1);
        const pin = term.screens.active.pages.trackPin(
            term.screens.active.pages.pin(.{ .active = .{
                .x = @intCast(i % 100),
                .y = @intCast((i / 100) % 100),
            } }).?,
        ) catch return error.BenchmarkFailed;
        storage.addPlacement(alloc, id, 0, .{
            .location = .{ .pin = pin },
        }) catch return error.BenchmarkFailed;
    }

    if (animated) {
        const frame = alloc.alloc(u8, @as(usize, w) * h * 4) catch
            return error.BenchmarkFailed;
        defer alloc.free(frame);
        @memset(frame, 0x80);

        for (0..image_count) |i| {
            const id: u32 = @intCast(i + 1);
            _ = storage.addAnimationFrame(
                alloc,
                term.screens.active,
                id,
                .{ .composition_mode = .overwrite },
                frame,
                .rgba,
                w,
                h,
                0,
            ) catch return error.BenchmarkFailed;

            const img = storage.images.getPtr(id).?;

            // Stand in for the renderer: only what is drawn is scheduled.
            img.drawn = true;

            const anim = img.anim.?;
            anim.state = .running;
            anim.setGap(0, 100);
            anim.setGap(1, 100);
        }
    }

    // The rectangle the ingestion modes transmit.
    const pct = @max(@min(self.opts.damage_percent, 100), 1);
    const dw = @max(w * pct / 100, 1);
    const dh = @max(h * pct / 100, 1);
    self.src = alloc.alloc(u8, @as(usize, dw) * dh * 4) catch
        return error.BenchmarkFailed;
    @memset(self.src, 0xC0);
}

fn teardown(ptr: *anyopaque) void {
    const self: *KittyAnimation = @ptrCast(@alignCast(ptr));
    const alloc = self.alloc;
    if (self.src.len > 0) alloc.free(self.src);
    self.src = &.{};
    if (self.term) |term| {
        term.deinit(alloc);
        alloc.destroy(term);
        self.term = null;
        self.storage = null;
    }
}

/// The exact sequence renderer/generic.zig runs inside every updateFrame.
fn stepSchedule(ptr: *anyopaque) Benchmark.Error!void {
    const self: *KittyAnimation = @ptrCast(@alignCast(ptr));
    const storage = self.storage.?;

    for (0..self.opts.iterations) |_| {
        // Deadline first, then the clock, exactly as updateFrame does:
        // the whole point of that order is that idle never reads a clock,
        // so a benchmark that skipped it would measure the wrong thing.
        const due = storage.nextAnimationDeadline();
        std.mem.doNotOptimizeAway(due);
        if (due) |d| {
            const now = storage.animationNowMs();
            std.mem.doNotOptimizeAway(now);
            std.mem.doNotOptimizeAway(d -| (now orelse 0));
        }
    }
}

fn stepTick(ptr: *anyopaque) Benchmark.Error!void {
    const self: *KittyAnimation = @ptrCast(@alignCast(ptr));
    const storage = self.storage.?;

    for (0..self.opts.iterations) |_| {
        // Advance the fake clock past every gap, or not at all, so the
        // mode measures the branch it names rather than a mixture.
        if (self.opts.mode == .@"tick-advance") self.now_ms += 1000;
        const r = storage.animationTick(self.now_ms);
        std.mem.doNotOptimizeAway(r.dirtied);
        std.mem.doNotOptimizeAway(r.next_due_ms);
    }
}

fn stepIngestAppend(ptr: *anyopaque) Benchmark.Error!void {
    const self: *KittyAnimation = @ptrCast(@alignCast(ptr));
    const storage = self.storage.?;
    const term = self.term.?;

    const pct = @max(@min(self.opts.damage_percent, 100), 1);
    const dw = @max(self.opts.width * pct / 100, 1);
    const dh = @max(self.opts.height * pct / 100, 1);

    for (0..self.opts.iterations) |_| {
        const r = storage.addAnimationFrame(
            self.alloc,
            term.screens.active,
            1,
            .{},
            self.src,
            .rgba,
            dw,
            dh,
            0,
        ) catch |err| {
            log.warn("append failed err={}", .{err});
            return error.BenchmarkFailed;
        };
        std.mem.doNotOptimizeAway(r.frame);

        // Drop it again, so this measures one append rather than the cost
        // of an ever-growing animation (and the quota it would exhaust).
        _ = storage.deleteAnimationFrame(
            self.alloc,
            term,
            .{
                .image_id = 1,
                .frame = @intCast(storage.images.getPtr(1).?.anim.?.frameCount()),
            },
            0,
        );
    }
}

fn stepIngestEdit(ptr: *anyopaque) Benchmark.Error!void {
    const self: *KittyAnimation = @ptrCast(@alignCast(ptr));
    const storage = self.storage.?;
    const term = self.term.?;

    const pct = @max(@min(self.opts.damage_percent, 100), 1);
    const dw = @max(self.opts.width * pct / 100, 1);
    const dh = @max(self.opts.height * pct / 100, 1);

    for (0..self.opts.iterations) |_| {
        // Edit the root, which is the frame on screen: this is the live
        // client shape, where the work lands in the frame budget.
        const r = storage.addAnimationFrame(
            self.alloc,
            term.screens.active,
            1,
            .{ .edit_frame = 1 },
            self.src,
            .rgba,
            dw,
            dh,
            0,
        ) catch |err| {
            log.warn("edit failed err={}", .{err});
            return error.BenchmarkFailed;
        };
        std.mem.doNotOptimizeAway(r.visible);
    }
}

test KittyAnimation {
    const testing = std.testing;
    const alloc = testing.allocator;

    inline for (@typeInfo(Mode).@"enum".fields) |field| {
        const impl: *KittyAnimation = try .create(alloc, .{
            .mode = @field(Mode, field.name),
            .width = 8,
            .height = 8,
            .images = 2,
            .placements = 4,
            .iterations = 4,
        });
        defer impl.destroy(alloc);

        const bench = impl.benchmark();
        _ = try bench.run(.once);
    }
}
