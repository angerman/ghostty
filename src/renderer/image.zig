const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = @import("../quirks.zig").inlineAssert;
const wuffs = @import("wuffs");
const terminal = @import("../terminal/main.zig");

const Renderer = @import("../renderer.zig").Renderer;
const GraphicsAPI = Renderer.API;
const Texture = GraphicsAPI.Texture;
const CellSize = @import("size.zig").CellSize;
const Overlay = @import("Overlay.zig");

const log = std.log.scoped(.renderer_image);

/// Generic image rendering state for the renderer. This stores all
/// images and their placements and exposes only a limited public API
/// for adding images and placements and drawing them.
pub const State = struct {
    /// The full image state for the renderer that specifies what images
    /// need to be uploaded, pruned, etc.
    images: ImageMap,

    /// The placements for the Kitty image protocol.
    kitty_placements: std.ArrayListUnmanaged(Placement),

    /// The end index (exclusive) for placements that should be
    /// drawn below the background, below the text, etc.
    kitty_bg_end: u32,
    kitty_text_end: u32,

    /// True if there are any virtual placements. This needs to be known
    /// because virtual placements need to be recalculated more often
    /// on frame builds and are generally more expensive to handle.
    kitty_virtual: bool,

    /// The Kitty images that the current layout actually draws.
    ///
    /// Retained from the last layout synchronization so that a pixel-only
    /// update -- an animation frame advancing, or a client editing pixels
    /// in place -- visits each changed image exactly once without
    /// rebuilding or re-sorting a single placement, however many
    /// placements reference it. It is an ordered set so that iteration is
    /// deterministic and deduplication is O(1) per placement.
    ///
    /// This is also what "visible" means for animation scheduling: an
    /// image present in terminal storage but not drawn is not here.
    kitty_visible: std.AutoArrayHashMapUnmanaged(u32, void),

    /// Overlays
    overlay_placements: std.ArrayListUnmanaged(Placement),

    pub const empty: State = .{
        .images = .empty,
        .kitty_placements = .empty,
        .kitty_bg_end = 0,
        .kitty_text_end = 0,
        .kitty_virtual = false,
        .kitty_visible = .empty,
        .overlay_placements = .empty,
    };

    pub fn deinit(self: *State, alloc: Allocator) void {
        {
            var it = self.images.iterator();
            while (it.next()) |kv| kv.value_ptr.deinit(alloc);
            self.images.deinit(alloc);
        }
        self.kitty_placements.deinit(alloc);
        self.kitty_visible.deinit(alloc);
        self.overlay_placements.deinit(alloc);
    }

    /// Upload any images to the GPU that need to be uploaded,
    /// and remove any images that are no longer needed on the GPU.
    ///
    /// If any uploads fail, they are ignored. The return value
    /// can be used to detect if upload was a total success (true)
    /// or not (false).
    pub fn upload(
        self: *State,
        alloc: Allocator,
        api: *GraphicsAPI,
    ) bool {
        var success: bool = true;
        var image_it = self.images.iterator();
        while (image_it.next()) |kv| {
            const img = &kv.value_ptr.image;
            if (img.isUnloading()) {
                kv.value_ptr.deinit(alloc);
                self.images.removeByPtr(kv.key_ptr);
                continue;
            }

            if (img.isPending()) {
                img.upload(
                    alloc,
                    api,
                ) catch |err| {
                    log.warn("error uploading image to GPU err={}", .{err});
                    success = false;
                };
            }
        }

        return success;
    }

    pub const DrawPlacements = enum {
        kitty_below_bg,
        kitty_below_text,
        kitty_above_text,
        overlay,
    };

    /// Draw the given named set of placements.
    ///
    /// Any placements that have non-uploaded images are ignored. Any
    /// graphics API errors during drawing are also ignored.
    pub fn draw(
        self: *State,
        api: *GraphicsAPI,
        pipeline: GraphicsAPI.Pipeline,
        pass: *GraphicsAPI.RenderPass,
        placement_type: DrawPlacements,
    ) void {
        const placements: []const Placement = switch (placement_type) {
            .kitty_below_bg => self.kitty_placements.items[0..self.kitty_bg_end],
            .kitty_below_text => self.kitty_placements.items[self.kitty_bg_end..self.kitty_text_end],
            .kitty_above_text => self.kitty_placements.items[self.kitty_text_end..],
            .overlay => self.overlay_placements.items,
        };

        for (placements) |p| {
            // Look up the image
            const image = self.images.get(p.image_id) orelse {
                log.warn("image not found for placement image_id={}", .{p.image_id});
                continue;
            };

            // Get the texture
            const texture = switch (image.image) {
                .ready,
                .unload_ready,
                => |t| t,
                else => {
                    log.warn("image not ready for placement image_id={}", .{p.image_id});
                    continue;
                },
            };

            // Create our vertex buffer, which is always exactly one item.
            // future(mitchellh): we can group rendering multiple instances of a single image
            var buf = GraphicsAPI.Buffer(GraphicsAPI.shaders.Image).initFill(
                api.imageBufferOptions(),
                &.{.{
                    .grid_pos = .{
                        @as(f32, @floatFromInt(p.x)),
                        @as(f32, @floatFromInt(p.y)),
                    },

                    .cell_offset = .{
                        @as(f32, @floatFromInt(p.cell_offset_x)),
                        @as(f32, @floatFromInt(p.cell_offset_y)),
                    },

                    .source_rect = .{
                        @as(f32, @floatFromInt(p.source_x)),
                        @as(f32, @floatFromInt(p.source_y)),
                        @as(f32, @floatFromInt(p.source_width)),
                        @as(f32, @floatFromInt(p.source_height)),
                    },

                    .dest_size = .{
                        @as(f32, @floatFromInt(p.width)),
                        @as(f32, @floatFromInt(p.height)),
                    },
                }},
            ) catch |err| {
                log.warn("error creating image vertex buffer err={}", .{err});
                continue;
            };
            defer buf.deinit();

            pass.step(.{
                .pipeline = pipeline,
                .buffers = &.{buf.buffer},
                .textures = &.{texture},
                .draw = .{
                    .type = .triangle_strip,
                    .vertex_count = 4,
                },
            });
        }
    }

    /// Update our overlay state. Null value deletes any existing overlay.
    pub fn overlayUpdate(
        self: *State,
        alloc: Allocator,
        overlay_: ?Overlay,
    ) !void {
        const overlay = overlay_ orelse {
            // If we don't have an overlay, remove any existing one.
            if (self.images.getPtr(.overlay)) |data| {
                data.image.markForUnload();
            }
            return;
        };

        // Overlays are always considered new content, so we take a
        // fresh generation stamp to force replacing any existing one.
        const generation = terminal.kitty.graphics.nextGeneration();

        // Ensure we have space for our overlay placement. Do this before
        // we upload our image so we don't have to deal with cleaning
        // that up.
        self.overlay_placements.clearRetainingCapacity();
        try self.overlay_placements.ensureUnusedCapacity(alloc, 1);

        // Setup our image.
        const pending = overlay.pendingImage();
        try self.prepImage(
            alloc,
            .overlay,
            generation,
            pending,
            // Overlays are rebuilt whole; they have no damage tracking.
            null,
        );
        errdefer comptime unreachable;

        // Setup our placement
        self.overlay_placements.appendAssumeCapacity(.{
            .image_id = .overlay,
            .x = 0,
            .y = 0,
            .z = 0,
            .width = pending.width,
            .height = pending.height,
            .cell_offset_x = 0,
            .cell_offset_y = 0,
            .source_x = 0,
            .source_y = 0,
            .source_width = pending.width,
            .source_height = pending.height,
        });
    }

    /// Returns true if the Kitty graphics state requires an update based
    /// on the terminal state and our internal state.
    ///
    /// This does not read/write state used by drawing.
    /// Whether the placement list has to be rebuilt: the set of images or
    /// placements changed, or the geometry they sit at moved.
    pub fn kittyRequiresLayoutUpdate(
        self: *const State,
        t: *const terminal.Terminal,
    ) bool {
        // If the terminal kitty image state is dirty, we must update.
        if (t.screens.active.kitty_images.layout_dirty) return true;

        // If we have any virtual references, we must also rebuild our
        // kitty state on every frame because any cell change can move
        // an image. If the virtual placements were removed, this will
        // be set to false on the next update.
        if (self.kitty_virtual) return true;

        return false;
    }

    /// Whether some drawn image's pixels changed while its dimensions
    /// stayed the same. This is the animation steady state, and it must
    /// stay strictly cheaper than a layout update.
    pub fn kittyRequiresPixelUpdate(
        _: *const State,
        t: *const terminal.Terminal,
    ) bool {
        return t.screens.active.kitty_images.pixel_dirty;
    }

    /// Rebuild the placement list and synchronize every drawn image.
    ///
    /// This is the expensive path: it clears and rebuilds every placement,
    /// sorts them by z, and probes viewport geometry. Only layout changes
    /// may come here. An animation tick must use kittyUpdatePixels, or
    /// every frame would pay for all of this to produce the same list.
    ///
    /// This reads/writes state used by drawing.
    pub fn kittyUpdatePlacements(
        self: *State,
        alloc: Allocator,
        t: *const terminal.Terminal,
        cell_size: CellSize,
    ) void {
        const storage = &t.screens.active.kitty_images;
        defer storage.layout_dirty = false;

        // A rebuild synchronizes every drawn image's pixels on the way
        // through, so it satisfies any outstanding pixel invalidation too.
        defer storage.pixel_dirty = false;

        // We always clear our previous placements no matter what because
        // we rebuild them from scratch.
        self.kitty_placements.clearRetainingCapacity();
        self.kitty_visible.clearRetainingCapacity();
        self.kitty_virtual = false;

        // Go through our known images and if there are any that are no longer
        // in use then mark them to be freed.
        //
        // This never conflicts with the below because a placement can't
        // reference an image that doesn't exist.
        {
            var it = self.images.iterator();
            while (it.next()) |kv| {
                switch (kv.key_ptr.*) {
                    // We're only looking at Kitty images
                    .kitty => |id| if (storage.imageById(id) == null) {
                        kv.value_ptr.image.markForUnload();
                    },

                    .overlay => {},
                }
            }
        }

        // The top-left and bottom-right corners of our viewport in screen
        // points. This lets us determine offsets and containment of placements.
        const top = t.screens.active.pages.getTopLeft(.viewport);
        const bot = t.screens.active.pages.getBottomRight(.viewport).?;
        const top_y = t.screens.active.pages.pointFromPin(.screen, top).?.screen.y;
        const bot_y = t.screens.active.pages.pointFromPin(.screen, bot).?.screen.y;

        // Go through the placements and ensure the image is
        // on the GPU or else is ready to be sent to the GPU.
        var it = storage.placements.iterator();
        while (it.next()) |kv| {
            const p = kv.value_ptr;

            // Special logic based on location
            switch (p.location) {
                .pin => {},
                .virtual => {
                    // We need to mark virtual placements on our renderer so that
                    // we know to rebuild in more scenarios since cell changes can
                    // now trigger placement changes.
                    self.kitty_virtual = true;

                    // We also continue out because virtual placements are
                    // only triggered by the unicode placeholder, not by the
                    // placement itself.
                    continue;
                },
            }

            // Get the image for the placement
            const image = storage.imageById(kv.key_ptr.image_id) orelse {
                log.warn(
                    "missing image for placement, ignoring image_id={}",
                    .{kv.key_ptr.image_id},
                );
                continue;
            };

            self.prepKittyPlacement(
                alloc,
                t,
                top_y,
                bot_y,
                &image,
                p,
            ) catch |err| {
                // For errors we log and continue. We try to place
                // other placements even if one fails.
                log.warn("error preparing kitty placement err={}", .{err});
            };
        }

        // If we have virtual placements then we need to scan for placeholders.
        if (self.kitty_virtual) {
            var v_it = terminal.kitty.graphics.unicode.placementIterator(top, bot);
            while (v_it.next()) |virtual_p| {
                self.prepKittyVirtualPlacement(
                    alloc,
                    t,
                    &virtual_p,
                    cell_size,
                ) catch |err| {
                    // For errors we log and continue. We try to place
                    // other placements even if one fails.
                    log.warn("error preparing kitty placement err={}", .{err});
                };
            }
        }

        // Tell terminal storage which images we actually draw, so that
        // animation scheduling follows what is on screen rather than what
        // merely exists. This is the only place that knows.
        storage.setDrawn(&self.kitty_visible);

        // Sort the placements by their Z value.
        std.mem.sortUnstable(
            Placement,
            self.kitty_placements.items,
            {},
            struct {
                fn lessThan(
                    ctx: void,
                    lhs: Placement,
                    rhs: Placement,
                ) bool {
                    _ = ctx;
                    return lhs.z < rhs.z or
                        (lhs.z == rhs.z and lhs.image_id.zLessThan(rhs.image_id));
                }
            }.lessThan,
        );

        // Find our indices. The values are sorted by z so we can
        // find the first placement out of bounds to find the limits.
        const bg_limit = std.math.minInt(i32) / 2;
        var bg_end: ?u32 = null;
        var text_end: ?u32 = null;
        for (self.kitty_placements.items, 0..) |p, i| {
            if (bg_end == null and p.z >= bg_limit) bg_end = @intCast(i);
            if (text_end == null and p.z >= 0) text_end = @intCast(i);
        }

        // If we didn't see any images with a z > the bg limit,
        // then our bg end is the end of our placement list.
        self.kitty_bg_end =
            bg_end orelse @intCast(self.kitty_placements.items.len);
        // Same idea for the image_text_end.
        self.kitty_text_end =
            text_end orelse @intCast(self.kitty_placements.items.len);
    }

    /// Re-upload the pixels of drawn images whose content changed, without
    /// touching placements.
    ///
    /// This is the animation steady state. It visits each drawn image once
    /// -- not once per placement -- and prepImage's generation check makes
    /// the images that did not change free, so an advancing frame costs one
    /// image's upload and nothing else.
    pub fn kittyUpdatePixels(
        self: *State,
        alloc: Allocator,
        t: *const terminal.Terminal,
    ) void {
        const storage = &t.screens.active.kitty_images;
        defer storage.pixel_dirty = false;

        for (self.kitty_visible.keys()) |id| {
            const img = storage.images.getPtr(id) orelse continue;
            self.prepKittyImage(alloc, img) catch |err| {
                // Leave pixel_dirty set, and the damage un-acknowledged,
                // so the next frame retries the whole accumulated region
                // rather than losing the part of it that failed.
                storage.pixel_dirty = true;
                log.warn("error preparing kitty image id={} err={}", .{ id, err });
                continue;
            };

            // Staged successfully: everything up to here is accounted for,
            // so start accumulating again from nothing.
            img.damage = .none;
        }
    }

    const PrepImageError = error{
        OutOfMemory,
        ImageConversionError,
    };

    /// Get the viewport-relative position for this
    /// placement and add it to the placements list.
    fn prepKittyPlacement(
        self: *State,
        alloc: Allocator,
        t: *const terminal.Terminal,
        top_y: u32,
        bot_y: u32,
        image: *const terminal.kitty.graphics.Image,
        p: *const terminal.kitty.graphics.ImageStorage.Placement,
    ) PrepImageError!void {
        // Get the rect for the placement. If this placement doesn't have
        // a rect then its virtual or something so skip it.
        const rect = p.rect(image.*, t) orelse return;

        // This is expensive but necessary.
        const img_top_y = t.screens.active.pages.pointFromPin(.screen, rect.top_left).?.screen.y;
        const img_bot_y = t.screens.active.pages.pointFromPin(.screen, rect.bottom_right).?.screen.y;

        // If the selection isn't within our viewport then skip it.
        if (img_top_y > bot_y) return;
        if (img_bot_y < top_y) return;

        // We need to prep this image for upload if it isn't in the
        // cache OR it is in the cache but the transmit time doesn't
        // match meaning this image is different.
        try self.prepKittyImage(alloc, image);

        // Calculate the dimensions of our image, taking in to
        // account the rows / columns specified by the placement.
        const dest_size = p.pixelSize(image.*, t);

        // Calculate the source rectangle
        const source_x = @min(image.width, p.source_x);
        const source_y = @min(image.height, p.source_y);
        const source_width = if (p.source_width > 0)
            @min(image.width - source_x, p.source_width)
        else
            image.width;
        const source_height = if (p.source_height > 0)
            @min(image.height - source_y, p.source_height)
        else
            image.height;

        // Get the viewport-relative Y position of the placement.
        const y_pos: i32 = @as(i32, @intCast(img_top_y)) - @as(i32, @intCast(top_y));

        // Accumulate the placement
        if (dest_size.width > 0 and dest_size.height > 0) {
            try self.kitty_placements.append(alloc, .{
                .image_id = .{ .kitty = image.id },
                .x = @intCast(rect.top_left.x),
                .y = y_pos,
                .z = p.z,
                .width = dest_size.width,
                .height = dest_size.height,
                .cell_offset_x = p.x_offset,
                .cell_offset_y = p.y_offset,
                .source_x = source_x,
                .source_y = source_y,
                .source_width = source_width,
                .source_height = source_height,
            });
        }
    }

    fn prepKittyVirtualPlacement(
        self: *State,
        alloc: Allocator,
        t: *const terminal.Terminal,
        p: *const terminal.kitty.graphics.unicode.Placement,
        cell_size: CellSize,
    ) PrepImageError!void {
        const storage = &t.screens.active.kitty_images;
        const image = storage.imageById(p.image_id) orelse {
            log.warn(
                "missing image for virtual placement, ignoring image_id={}",
                .{p.image_id},
            );
            return;
        };

        const rp = p.renderPlacement(
            storage,
            &image,
            cell_size.width,
            cell_size.height,
        ) catch |err| {
            log.warn("error rendering virtual placement err={}", .{err});
            return;
        };

        // If our placement is zero sized then we don't do anything.
        if (rp.dest_width == 0 or rp.dest_height == 0) return;

        const viewport: terminal.point.Point = t.screens.active.pages.pointFromPin(
            .viewport,
            rp.top_left,
        ) orelse {
            // This is unreachable with virtual placements because we should
            // only ever be looking at virtual placements that are in our
            // viewport in the renderer and virtual placements only ever take
            // up one row.
            unreachable;
        };

        // Prepare the image for the GPU and store the placement.
        try self.prepKittyImage(alloc, &image);
        try self.kitty_placements.append(alloc, .{
            .image_id = .{ .kitty = image.id },
            .x = @intCast(rp.top_left.x),
            .y = @intCast(viewport.viewport.y),
            .z = -1,
            .width = rp.dest_width,
            .height = rp.dest_height,
            .cell_offset_x = rp.offset_x,
            .cell_offset_y = rp.offset_y,
            .source_x = rp.source_x,
            .source_y = rp.source_y,
            .source_width = rp.source_width,
            .source_height = rp.source_height,
        });
    }

    /// Prepare an image for upload to the GPU.
    fn prepImage(
        self: *State,
        alloc: Allocator,
        id: Id,
        generation: u64,
        pending: Image.Pending,
        /// The region of `pending` that actually changed, or null for all
        /// of it. Only honored when a texture of the right shape already
        /// exists: the first upload of an image has to be whole, because
        /// a region cannot create a texture.
        damage: ?Image.Damage.Rect,
    ) PrepImageError!void {
        // If this image exists and its generation is the same it is the
        // identical image so we don't need to send it to the GPU.
        const gop = try self.images.getOrPut(alloc, id);
        if (gop.found_existing and
            gop.value_ptr.generation == generation)
        {
            return;
        }

        // Copy the pixels out of terminal state and into our own staging,
        // because the upload happens later without the terminal lock. The
        // staging buffer is reused across frames and only grows, so a
        // dimension-stable animation makes no allocator call here at all
        // after its first frame.
        if (!gop.found_existing) gop.value_ptr.* = .{
            .image = .{ .pending = pending },
            .generation = 0,
            .staging = &.{},
        };

        // Decide whether this can be a partial upload before staging,
        // because a partial stages only the damaged rows.
        const region = Image.stagedRegion(
            if (gop.value_ptr.image.getTexture()) |tex| .{
                .width = tex.width,
                .height = tex.height,
            } else null,
            pending,
            damage,
        );

        const bpp = pending.pixel_format.bpp();
        const stride = pending.width * bpp;
        const staged_len: usize = if (region) |r|
            @as(usize, r.width) * r.height * bpp
        else
            pending.dataSlice().len;

        if (gop.value_ptr.staging.len < staged_len) {
            const grown = alloc.realloc(gop.value_ptr.staging, staged_len) catch {
                if (!gop.found_existing) {
                    // If this is a new entry we can just remove it since it
                    // was never sent to the GPU.
                    _ = self.images.remove(id);
                } else {
                    // If this was an existing entry, it is invalid and
                    // we must unload it.
                    gop.value_ptr.image.markForUnload();
                }

                return error.OutOfMemory;
            };
            gop.value_ptr.staging = grown;
        }
        const data = gop.value_ptr.staging[0..staged_len];
        if (region) |r| {
            // Pack the damaged rows tightly: replaceRegion takes the
            // region's own rows, not a window into the full image.
            const src_all = pending.data;
            for (0..r.height) |row| {
                const src_off = (@as(usize, r.y) + row) * stride + @as(usize, r.x) * bpp;
                const dst_off = row * r.width * bpp;
                @memcpy(
                    data[dst_off..][0 .. r.width * bpp],
                    src_all[src_off..][0 .. r.width * bpp],
                );
            }
        } else {
            @memcpy(data, pending.dataSlice());
        }

        // Store it in the map. The pending pixels are borrowed from the
        // staging above, so nothing frees them but the entry itself.
        const new_image: Image = .{
            .pending = .{
                .width = if (region) |r| r.width else pending.width,
                .height = if (region) |r| r.height else pending.height,
                .pixel_format = pending.pixel_format,
                .data = data.ptr,
                .owned = false,
                .dest = if (region) |r| .{ .x = r.x, .y = r.y } else null,
            },
        };
        if (!gop.found_existing) {
            gop.value_ptr.image = new_image;
        } else {
            gop.value_ptr.image.markForReplace(
                alloc,
                new_image,
            );
        }

        // If any error happens, we unload the image and it is invalid.
        errdefer gop.value_ptr.image.markForUnload();

        gop.value_ptr.image.prepForUpload(alloc) catch |err| {
            log.warn("error preparing image for upload err={}", .{err});
            return error.ImageConversionError;
        };
        gop.value_ptr.generation = generation;
    }

    /// Prepare the provided Kitty image for upload to the GPU by copying its
    /// data with our allocator and setting it to the pending state.
    fn prepKittyImage(
        self: *State,
        alloc: Allocator,
        image: *const terminal.kitty.graphics.Image,
    ) PrepImageError!void {
        // Record that this image is actually drawn. Pixel-only updates and
        // animation scheduling both work from this set rather than from
        // everything terminal storage happens to hold.
        try self.kitty_visible.put(alloc, image.id, {});

        // An animated image renders its current frame rather than its own
        // data. Every frame has the image's dimensions, so nothing else
        // about the placement changes. The image's generation is bumped
        // whenever the frame changes, which is what gets us back here to
        // re-upload the texture.
        const data = image.renderData();

        try self.prepImage(
            alloc,
            .{ .kitty = image.id },
            image.generation,
            .{
                .width = image.width,
                .height = image.height,
                .pixel_format = switch (image.renderFormat()) {
                    .gray => .gray,
                    .gray_alpha => .gray_alpha,
                    .rgb => .rgb,
                    .rgba => .rgba,
                    .png => unreachable, // should be decoded by now
                },

                // constCasts are always gross but this one is safe is because
                // the data is only read from here and copied into its own
                // buffer.
                .data = @constCast(data.ptr),
            },
            image.damage.region(image.width, image.height),
        );
    }
};

/// Represents a single image placement on the grid.
/// A placement is a request to render an instance of an image.
pub const Placement = struct {
    /// The image being rendered. This MUST be in the image map.
    image_id: Id,

    /// The grid x/y where this placement is located.
    x: i32,
    y: i32,
    z: i32,

    /// The width/height of the placed image.
    width: u32,
    height: u32,

    /// The offset in pixels from the top left of the cell.
    /// This is clamped to the size of a cell.
    cell_offset_x: u32,
    cell_offset_y: u32,

    /// The source rectangle of the placement.
    source_x: u32,
    source_y: u32,
    source_width: u32,
    source_height: u32,
};

/// Image identifier used to store and lookup images.
///
/// This is tagged by different image types to make it easier to
/// store different kinds of images in the same map without having
/// to worry about ID collisions.
pub const Id = union(enum) {
    /// Image sent to the terminal state via the kitty graphics protocol.
    /// The value is the ID assigned by the terminal.
    kitty: u32,

    /// Debug overlay. This is always composited down to a single
    /// image for now. In the future we can support layers here if we want.
    overlay,

    /// Z-ordering tie-breaker for images with the same z value.
    pub fn zLessThan(lhs: Id, rhs: Id) bool {
        // If our tags aren't the same, we sort by tag.
        if (std.meta.activeTag(lhs) != std.meta.activeTag(rhs)) {
            return switch (lhs) {
                // Kitty images always sort before (lower z) non-kitty images.
                .kitty => true,

                .overlay => false,
            };
        }

        switch (lhs) {
            .kitty => |lhs_id| {
                const rhs_id = rhs.kitty;
                return lhs_id < rhs_id;
            },

            // No sensical ordering
            .overlay => return false,
        }
    }
};

/// The map used for storing images.
pub const ImageMap = std.AutoHashMapUnmanaged(Id, Entry);

pub const Entry = struct {
    image: Image,

    /// The generation of the terminal image this was created from
    /// (see terminal.kitty.graphics.Image.generation). Used to detect
    /// staleness: a differing generation for the same ID means the
    /// contents changed and the texture must be replaced. Zero is
    /// never a valid stored generation so it marks "not yet uploaded".
    generation: u64,

    /// Reusable CPU staging for this image's uploads.
    ///
    /// Pixels have to be copied out of terminal state before the upload,
    /// because the upload happens without the terminal lock held. Doing
    /// that with a fresh allocation every time means an allocate and a
    /// free per displayed frame, forever, for buffers that are the same
    /// size every time. So the buffer lives here instead and only grows.
    ///
    /// A Pending pointing into this borrows it (Pending.owned == false).
    staging: []u8 = &.{},

    pub fn deinit(self: *Entry, alloc: Allocator) void {
        self.image.deinit(alloc);
        if (self.staging.len > 0) alloc.free(self.staging);
    }
};

/// The state for a single image that is to be rendered.
pub const Image = union(enum) {
    /// The image data is pending upload to the GPU.
    ///
    /// This data is owned by this union so it must be freed once uploaded.
    pending: Pending,

    /// This is the same as the pending states but there is
    /// a texture already allocated that we want to replace.
    replace: Replace,

    /// The image is uploaded and ready to be used.
    ready: Texture,

    /// The image isn't uploaded yet but is scheduled to be unloaded.
    unload_pending: Pending,
    /// The image is uploaded and is scheduled to be unloaded.
    unload_ready: Texture,
    /// The image is uploaded and scheduled to be replaced
    /// with new data, but it's also scheduled to be unloaded.
    unload_replace: Replace,

    pub const Replace = struct {
        texture: Texture,
        pending: Pending,
    };

    /// Pending image data that needs to be uploaded to the GPU.
    pub const Origin = struct { x: u32, y: u32 };

    /// The damage description the terminal side produces.
    pub const Damage = terminal.kitty.graphics.Damage;

    pub const Pending = struct {
        height: u32,
        width: u32,
        pixel_format: PixelFormat,

        /// Data is always expected to be (width * height * bpp).
        data: [*]u8,

        /// Where these pixels belong in the texture.
        ///
        /// Null means they are the whole image, which is the only thing
        /// that can create a texture. A non-null origin means this is a
        /// partial update of a texture that already exists: `width` and
        /// `height` are the damaged region's, not the image's, and `data`
        /// is that region's rows tightly packed.
        dest: ?Origin = null,

        /// Whether `data` is owned by this Image and must be freed with
        /// it.
        ///
        /// Borrowed data points into the reusable staging buffer owned by
        /// the image's map entry, which is how a dimension-stable
        /// animation avoids allocating and freeing a full frame every
        /// time it advances. A format conversion allocates and takes
        /// ownership, so this cannot be assumed either way.
        owned: bool = true,

        pub fn dataSlice(self: Pending) []u8 {
            return self.data[0..self.len()];
        }

        pub fn len(self: Pending) usize {
            return self.width * self.height * self.pixel_format.bpp();
        }

        pub const PixelFormat = enum {
            /// 1 byte per pixel grayscale.
            gray,
            /// 2 bytes per pixel grayscale + alpha.
            gray_alpha,
            /// 3 bytes per pixel RGB.
            rgb,
            /// 3 bytes per pixel BGR.
            bgr,
            /// 4 byte per pixel RGBA.
            rgba,
            /// 4 byte per pixel BGRA.
            bgra,

            /// Get bytes per pixel for this format.
            pub inline fn bpp(self: PixelFormat) usize {
                return switch (self) {
                    .gray => 1,
                    .gray_alpha => 2,
                    .rgb => 3,
                    .bgr => 3,
                    .rgba => 4,
                    .bgra => 4,
                };
            }
        };
    };

    pub fn deinit(self: Image, alloc: Allocator) void {
        switch (self) {
            .pending,
            .unload_pending,
            => |p| if (p.owned) alloc.free(p.dataSlice()),

            .replace, .unload_replace => |r| {
                if (r.pending.owned) alloc.free(r.pending.dataSlice());
                r.texture.deinit();
            },

            .ready,
            .unload_ready,
            => |t| t.deinit(),
        }
    }

    /// Mark this image for unload whatever state it is in.
    pub fn markForUnload(self: *Image) void {
        self.* = switch (self.*) {
            .unload_pending,
            .unload_replace,
            .unload_ready,
            => return,

            .ready => |t| .{ .unload_ready = t },
            .pending => |p| .{ .unload_pending = p },
            .replace => |r| .{ .unload_replace = r },
        };
    }

    /// Mark the current image to be replaced with a pending one. This will
    /// attempt to update the existing texture if we have one, otherwise it
    /// will act like a new upload.
    pub fn markForReplace(self: *Image, alloc: Allocator, img: Image) void {
        assert(img.isPending());

        // If we have pending data right now, free it.
        if (self.getPending()) |p| {
            if (p.owned) alloc.free(p.dataSlice());
        }
        // If we have an existing texture, use it in the replace.
        if (self.getTexture()) |t| {
            self.* = .{ .replace = .{
                .texture = t,
                .pending = img.getPending().?,
            } };
            return;
        }
        // Otherwise we just become a pending image.
        self.* = .{ .pending = img.getPending().? };
    }

    /// Returns true if this image is pending upload.
    pub fn isPending(self: Image) bool {
        return self.getPending() != null;
    }

    /// Returns true if this image has an associated texture.
    pub fn hasTexture(self: Image) bool {
        return self.getTexture() != null;
    }

    /// Returns true if this image is marked for unload.
    pub fn isUnloading(self: Image) bool {
        return switch (self) {
            .unload_pending,
            .unload_replace,
            .unload_ready,
            => true,

            .pending,
            .replace,
            .ready,
            => false,
        };
    }

    /// Converts the image data to a format that can be uploaded to the GPU.
    /// If the data is already in a format that can be uploaded, this is a
    /// no-op.
    fn convert(self: *Image, alloc: Allocator) wuffs.Error!void {
        const p = self.getPendingPointer().?;
        // As things stand, we currently convert all images to RGBA before
        // uploading to the GPU. This just makes things easier. In the future
        // we may want to support other formats.
        if (p.pixel_format == .rgba) return;
        // If the pending data isn't RGBA we'll need to swizzle it.
        const data = p.dataSlice();
        const rgba = try switch (p.pixel_format) {
            .gray => wuffs.swizzle.gToRgba(alloc, data),
            .gray_alpha => wuffs.swizzle.gaToRgba(alloc, data),
            .rgb => wuffs.swizzle.rgbToRgba(alloc, data),
            .bgr => wuffs.swizzle.bgrToRgba(alloc, data),
            .rgba => unreachable,
            .bgra => wuffs.swizzle.bgraToRgba(alloc, data),
        };
        if (p.owned) alloc.free(data);
        p.data = rgba.ptr;
        p.owned = true;
        p.pixel_format = .rgba;
    }

    /// Prepare the pending image data for upload to the GPU.
    /// This doesn't need GPU access so is safe to call any time.
    fn prepForUpload(self: *Image, alloc: Allocator) wuffs.Error!void {
        assert(self.isPending());
        try self.convert(alloc);
    }

    /// Upload the pending image to the GPU and change the state of this
    /// image to ready.
    pub fn upload(
        self: *Image,
        alloc: Allocator,
        api: *const GraphicsAPI,
    ) (wuffs.Error || error{
        /// Texture creation failed, usually a GPU memory issue.
        UploadFailed,
    })!void {
        assert(self.isPending());

        // No error recover is required after this call because it just
        // converts in place and is idempotent.
        try self.prepForUpload(alloc);

        // Get our pending info
        const p = self.getPending().?;

        // If we're replacing an image whose shape is unchanged, upload
        // into the texture we already have rather than building a new one
        // and throwing the old one away. That is the animation steady
        // state: a playing animation would otherwise create and destroy a
        // texture on every single frame.
        //
        // Everything is converted to RGBA above and every texture is made
        // with the same options, so matching dimensions is enough to know
        // the existing texture can hold these pixels.
        if (self.getTexture()) |existing| {
            // A partial update always replaces a region: prepImage only
            // produces one when a texture of the right shape exists.
            const origin: ?Origin = p.dest orelse
                if (textureFits(existing.width, existing.height, p))
                    .{ .x = 0, .y = 0 }
                else
                    null;

            if (origin) |o| {
                existing.replaceRegion(
                    o.x,
                    o.y,
                    p.width,
                    p.height,
                    p.dataSlice(),
                ) catch |err| {
                    // Keep the old texture and the pending bytes so the
                    // next frame retries. Never leave a valid frame
                    // replaced by blank or half-written texture state.
                    log.warn("error replacing texture region err={}", .{err});
                    return error.UploadFailed;
                };
                errdefer comptime unreachable;

                // The pixels are in the texture we already had, so only
                // the pending copy is released, and only if we own it:
                // borrowed staging stays alive for the next frame. The
                // texture identity, and anything the GPU holds
                // referencing it, is preserved.
                if (p.owned) alloc.free(p.dataSlice());
                self.* = .{ .ready = existing };
                return;
            }
        }

        // Create our texture
        const texture = Texture.init(
            api.imageTextureOptions(.rgba, true),
            @intCast(p.width),
            @intCast(p.height),
            p.dataSlice(),
        ) catch return error.UploadFailed;
        errdefer comptime unreachable;

        // Uploaded. We can now clear our data and change our state.
        //
        // NOTE: For the `replace` state, this frees the old texture. We
        //       only get here when the shape genuinely changed, so the old
        //       texture cannot hold the new pixels.
        self.deinit(alloc);
        self.* = .{ .ready = texture };
    }

    /// The region to stage and upload, or null to upload the whole image.
    ///
    /// A region can only ever *update* a texture, so the first upload of an
    /// image is always whole however little of it changed, and so is one
    /// whose shape just changed. A region covering everything is a full
    /// upload too: the full path stages one contiguous copy instead of
    /// row by row.
    ///
    /// Split out from prepImage so the rule can be tested without a GPU:
    /// a partial upload by definition needs a texture to already exist.
    fn stagedRegion(
        texture: ?struct { width: usize, height: usize },
        pending: Image.Pending,
        damage: ?Image.Damage.Rect,
    ) ?Image.Damage.Rect {
        const r = damage orelse return null;
        if (r.width == 0 or r.height == 0) return null;

        const tex = texture orelse return null;
        if (tex.width != pending.width or tex.height != pending.height) return null;
        if (r.width == pending.width and r.height == pending.height) return null;

        return r;
    }

    /// Whether pixels of the pending shape can be uploaded into an
    /// existing texture of the given shape rather than requiring a new
    /// texture to be created.
    ///
    /// Everything is converted to RGBA before upload and every image
    /// texture is created with the same options, so matching dimensions is
    /// the whole condition. Split out from upload so the rule can be
    /// tested without a GPU.
    fn textureFits(tex_width: usize, tex_height: usize, p: Pending) bool {
        return tex_width == p.width and tex_height == p.height;
    }

    /// Returns any pending image data for this image that requires upload.
    ///
    /// If there is no pending data to upload, returns null.
    fn getPending(self: Image) ?Pending {
        return switch (self) {
            .pending,
            .unload_pending,
            => |p| p,

            .replace,
            .unload_replace,
            => |r| r.pending,

            else => null,
        };
    }

    /// Returns the texture for this image.
    ///
    /// If there is no texture for it yet, returns null.
    fn getTexture(self: Image) ?Texture {
        return switch (self) {
            .ready,
            .unload_ready,
            => |t| t,

            .replace,
            .unload_replace,
            => |r| r.texture,

            else => null,
        };
    }

    // Same as getPending but returns a pointer instead of a copy.
    fn getPendingPointer(self: *Image) ?*Pending {
        return switch (self.*) {
            .pending => return &self.pending,
            .unload_pending => return &self.unload_pending,

            .replace => return &self.replace.pending,
            .unload_replace => return &self.unload_replace.pending,

            else => null,
        };
    }
};

test "kitty: a frame advance does not rebuild placements" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try terminal.Terminal.init(alloc, .{ .rows = 10, .cols = 10 });
    defer t.deinit(alloc);
    t.width_px = 100;
    t.height_px = 100;

    const storage = &t.screens.active.kitty_images;

    // A 2x2 RGBA image with a frame, placed and playing.
    const data = try alloc.alloc(u8, 2 * 2 * 4);
    @memset(data, 1);
    {
        // Scoped so the errdefer covers only the handoff: addImage takes
        // ownership on success and storage frees it from then on.
        errdefer alloc.free(data);
        try storage.addImage(alloc, t.screens.active, .{
            .id = 1,
            .width = 2,
            .height = 2,
            .format = .rgba,
            .data = data,
        });
    }
    const pin = try t.screens.active.pages.trackPin(
        t.screens.active.pages.pin(.{ .active = .{ .x = 0, .y = 0 } }).?,
    );
    try storage.addPlacement(alloc, 1, 0, .{ .location = .{ .pin = pin } });

    const src: [2 * 2 * 4]u8 = @splat(9);
    _ = try storage.addAnimationFrame(alloc, t.screens.active, 1, .{}, &src, .rgba, 2, 2, 0);
    const anim = storage.images.getPtr(1).?.anim.?;
    anim.state = .running;
    anim.setGap(0, 100);
    anim.setGap(1, 100);

    var state: State = .empty;
    defer state.deinit(alloc);
    const cell_size: CellSize = .{ .width = 10, .height = 10 };

    // One layout synchronization to warm up.
    try testing.expect(state.kittyRequiresLayoutUpdate(&t));
    state.kittyUpdatePlacements(alloc, &t, cell_size);
    try testing.expectEqual(@as(usize, 1), state.kitty_placements.items.len);
    try testing.expectEqual(@as(usize, 1), state.kitty_visible.count());
    try testing.expect(!state.kittyRequiresLayoutUpdate(&t));
    try testing.expect(!state.kittyRequiresPixelUpdate(&t));

    // Now advance a frame. This is the animation steady state: it must
    // ask for a pixel update and must NOT ask for a layout rebuild.
    const r = storage.animationTick(100);
    try testing.expect(r.dirtied);
    try testing.expect(!state.kittyRequiresLayoutUpdate(&t));
    try testing.expect(state.kittyRequiresPixelUpdate(&t));

    // The pixel update leaves the placement list untouched, and the
    // renderer's copy of the image follows the new frame.
    const placements_ptr = state.kitty_placements.items.ptr;
    state.kittyUpdatePixels(alloc, &t);
    try testing.expectEqual(@as(usize, 1), state.kitty_placements.items.len);
    try testing.expectEqual(placements_ptr, state.kitty_placements.items.ptr);
    try testing.expect(!state.kittyRequiresPixelUpdate(&t));

    const prepped = state.images.get(.{ .kitty = 1 }).?;
    try testing.expectEqual(storage.images.getPtr(1).?.generation, prepped.generation);

    // A schedule-only change is neither: it must not touch pixels or
    // placements at all.
    storage.markScheduleMutated();
    try testing.expect(!state.kittyRequiresLayoutUpdate(&t));
    try testing.expect(!state.kittyRequiresPixelUpdate(&t));
}

test "kitty: an unchanged shape reuses the texture" {
    const testing = std.testing;

    // A dimension-stable animation must reuse its texture: recreating one
    // per frame is exactly what this path exists to avoid.
    var bytes: [4]u8 = @splat(0);
    const p: Image.Pending = .{
        .width = 64,
        .height = 32,
        .pixel_format = .rgba,
        .data = &bytes,
    };
    try testing.expect(Image.textureFits(64, 32, p));

    // A genuine shape change cannot go into the existing texture.
    try testing.expect(!Image.textureFits(64, 33, p));
    try testing.expect(!Image.textureFits(65, 32, p));
    try testing.expect(!Image.textureFits(0, 0, p));
}

/// Counts allocator calls so tests can assert a steady-state frame makes
/// none, rather than asserting that the code looks like it wouldn't.
const CountingAllocator = struct {
    parent: Allocator,
    allocs: usize = 0,
    frees: usize = 0,
    resizes: usize = 0,

    fn allocator(self: *CountingAllocator) Allocator {
        return .{ .ptr = self, .vtable = &.{
            .alloc = alloc,
            .resize = resize,
            .remap = remap,
            .free = free,
        } };
    }

    fn alloc(ctx: *anyopaque, len: usize, a: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.allocs += 1;
        return self.parent.rawAlloc(len, a, ra);
    }

    fn resize(ctx: *anyopaque, buf: []u8, a: std.mem.Alignment, new_len: usize, ra: usize) bool {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.resizes += 1;
        return self.parent.rawResize(buf, a, new_len, ra);
    }

    fn remap(ctx: *anyopaque, buf: []u8, a: std.mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.resizes += 1;
        return self.parent.rawRemap(buf, a, new_len, ra);
    }

    fn free(ctx: *anyopaque, buf: []u8, a: std.mem.Alignment, ra: usize) void {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.frees += 1;
        self.parent.rawFree(buf, a, ra);
    }
};

test "kitty: steady-state frames do not touch the allocator" {
    const testing = std.testing;
    var counting: CountingAllocator = .{ .parent = testing.allocator };
    const alloc = counting.allocator();

    var t = try terminal.Terminal.init(alloc, .{ .rows = 10, .cols = 10 });
    defer t.deinit(alloc);
    t.width_px = 100;
    t.height_px = 100;

    const storage = &t.screens.active.kitty_images;

    const data = try alloc.alloc(u8, 4 * 4 * 4);
    @memset(data, 1);
    {
        errdefer alloc.free(data);
        try storage.addImage(alloc, t.screens.active, .{
            .id = 1,
            .width = 4,
            .height = 4,
            .format = .rgba,
            .data = data,
        });
    }
    const pin = try t.screens.active.pages.trackPin(
        t.screens.active.pages.pin(.{ .active = .{ .x = 0, .y = 0 } }).?,
    );
    try storage.addPlacement(alloc, 1, 0, .{ .location = .{ .pin = pin } });

    // Two frames so the animation has something to alternate between.
    const src: [4 * 4 * 4]u8 = @splat(9);
    _ = try storage.addAnimationFrame(alloc, t.screens.active, 1, .{}, &src, .rgba, 4, 4, 0);
    const anim = storage.images.getPtr(1).?.anim.?;
    anim.state = .running;
    anim.setGap(0, 100);
    anim.setGap(1, 100);

    var state: State = .empty;
    defer state.deinit(alloc);

    // Warm up: the first synchronization is allowed to allocate.
    state.kittyUpdatePlacements(alloc, &t, .{ .width = 10, .height = 10 });

    // Now the steady state. Advancing a frame and synchronizing its
    // pixels must not allocate, free, or resize anything: the staging
    // buffer is reused and the placements are untouched.
    var now: u64 = 100;
    for (0..16) |_| {
        _ = storage.animationTick(now);
        state.kittyUpdatePixels(alloc, &t);
        now += 100;
    }

    const allocs = counting.allocs;
    const frees = counting.frees;
    const resizes = counting.resizes;
    for (0..16) |_| {
        _ = storage.animationTick(now);
        state.kittyUpdatePixels(alloc, &t);
        now += 100;
    }
    try testing.expectEqual(allocs, counting.allocs);
    try testing.expectEqual(frees, counting.frees);
    try testing.expectEqual(resizes, counting.resizes);
}

test "kitty: scrolling out of view stops scheduling" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try terminal.Terminal.init(alloc, .{ .rows = 4, .cols = 10 });
    defer t.deinit(alloc);
    t.width_px = 100;
    t.height_px = 40;

    const storage = &t.screens.active.kitty_images;

    const data = try alloc.alloc(u8, 2 * 2 * 4);
    @memset(data, 1);
    {
        errdefer alloc.free(data);
        try storage.addImage(alloc, t.screens.active, .{
            .id = 1,
            .width = 2,
            .height = 2,
            .format = .rgba,
            .data = data,
        });
    }
    const pin = try t.screens.active.pages.trackPin(
        t.screens.active.pages.pin(.{ .active = .{ .x = 0, .y = 0 } }).?,
    );
    try storage.addPlacement(alloc, 1, 0, .{ .location = .{ .pin = pin } });

    const src: [2 * 2 * 4]u8 = @splat(9);
    _ = try storage.addAnimationFrame(alloc, t.screens.active, 1, .{}, &src, .rgba, 2, 2, 0);
    const anim = storage.images.getPtr(1).?.anim.?;
    anim.state = .running;
    anim.setGap(0, 100);
    anim.setGap(1, 100);

    var state: State = .empty;
    defer state.deinit(alloc);
    const cell_size: CellSize = .{ .width = 10, .height = 10 };

    // Drawn: it schedules.
    state.kittyUpdatePlacements(alloc, &t, cell_size);
    try testing.expect(storage.images.getPtr(1).?.drawn);
    try testing.expect(storage.nextAnimationDeadline() != null);

    // Scroll the placement far into the scrollback. The image is still in
    // storage and still has a placement, but nothing draws it, so it must
    // not keep a timer armed or wake the renderer.
    for (0..64) |_| try t.index();
    state.kittyUpdatePlacements(alloc, &t, cell_size);
    try testing.expect(!storage.images.getPtr(1).?.drawn);
    try testing.expectEqual(@as(?u64, null), storage.nextAnimationDeadline());

    const r = storage.animationTick(10_000);
    try testing.expect(!r.dirtied);
    try testing.expectEqual(@as(?u64, null), r.next_due_ms);
    try testing.expect(!state.kittyRequiresPixelUpdate(&t));
}

test "kitty: a small edit uploads only its damage" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try terminal.Terminal.init(alloc, .{ .rows = 10, .cols = 10 });
    defer t.deinit(alloc);
    t.width_px = 100;
    t.height_px = 100;

    const storage = &t.screens.active.kitty_images;

    // An 8x8 RGBA image, placed.
    const data = try alloc.alloc(u8, 8 * 8 * 4);
    @memset(data, 1);
    {
        errdefer alloc.free(data);
        try storage.addImage(alloc, t.screens.active, .{
            .id = 1,
            .width = 8,
            .height = 8,
            .format = .rgba,
            .data = data,
        });
    }
    const pin = try t.screens.active.pages.trackPin(
        t.screens.active.pages.pin(.{ .active = .{ .x = 0, .y = 0 } }).?,
    );
    try storage.addPlacement(alloc, 1, 0, .{ .location = .{ .pin = pin } });

    var state: State = .empty;
    defer state.deinit(alloc);

    // First synchronization uploads the whole image: there is no texture
    // to update a region of yet.
    state.kittyUpdatePlacements(alloc, &t, .{ .width = 10, .height = 10 });
    try testing.expect(storage.images.getPtr(1).?.damage == .none);

    // Edit a 2x2 corner of the current (root) frame, the way a VNC client
    // sends a cursor-sized update.
    const src: [2 * 2 * 4]u8 = @splat(9);
    _ = try storage.addAnimationFrame(
        alloc,
        t.screens.active,
        1,
        .{ .edit_frame = 1, .x = 5, .y = 3, .composition_mode = .overwrite },
        &src,
        .rgba,
        2,
        2,
        0,
    );

    // The damage is exactly what was composed, not the whole image.
    try testing.expectEqual(
        terminal.kitty.graphics.Damage.Rect{ .x = 5, .y = 3, .width = 2, .height = 2 },
        storage.images.getPtr(1).?.damage.region(8, 8).?,
    );

    // A second edit elsewhere accumulates rather than replacing: the
    // first one has not been uploaded yet.
    _ = try storage.addAnimationFrame(
        alloc,
        t.screens.active,
        1,
        .{ .edit_frame = 1, .x = 0, .y = 0, .composition_mode = .overwrite },
        &src,
        .rgba,
        2,
        2,
        0,
    );
    try testing.expectEqual(
        terminal.kitty.graphics.Damage.Rect{ .x = 0, .y = 0, .width = 7, .height = 5 },
        storage.images.getPtr(1).?.damage.region(8, 8).?,
    );

    // Synchronizing pixels acknowledges the damage, so the next edit
    // starts accumulating from nothing again.
    state.kittyUpdatePixels(alloc, &t);
    try testing.expect(storage.images.getPtr(1).?.damage == .none);

    // Note this upload is whole despite the small damage, and correctly
    // so: there is no texture yet to update a region of. These tests have
    // no GPU, so the region decision itself is covered by the unit test
    // for stagedRegion below.
    const entry = state.images.get(.{ .kitty = 1 }).?;
    const pending = entry.image.getPending().?;
    try testing.expectEqual(@as(u32, 8), pending.width);
    try testing.expectEqual(@as(?Image.Origin, null), pending.dest);
}

test "kitty: only a real region of an existing texture is staged" {
    const testing = std.testing;
    var bytes: [8 * 8 * 4]u8 = @splat(0);
    const p: Image.Pending = .{
        .width = 8,
        .height = 8,
        .pixel_format = .rgba,
        .data = &bytes,
    };
    const small: Image.Damage.Rect = .{ .x = 1, .y = 1, .width = 2, .height = 2 };

    // The common case: a texture of the right shape and a small change.
    try testing.expectEqual(small, Image.stagedRegion(
        .{ .width = 8, .height = 8 },
        p,
        small,
    ).?);

    // No texture yet: a region cannot create one, so upload it all.
    try testing.expectEqual(@as(?Image.Damage.Rect, null), Image.stagedRegion(
        null,
        p,
        small,
    ));

    // The shape changed: the old texture cannot take these pixels.
    try testing.expectEqual(@as(?Image.Damage.Rect, null), Image.stagedRegion(
        .{ .width = 4, .height = 8 },
        p,
        small,
    ));

    // No damage recorded means upload it all.
    try testing.expectEqual(@as(?Image.Damage.Rect, null), Image.stagedRegion(
        .{ .width = 8, .height = 8 },
        p,
        null,
    ));

    // Damage covering the whole image is a full upload: staging it row by
    // row would be the same bytes and more work.
    try testing.expectEqual(@as(?Image.Damage.Rect, null), Image.stagedRegion(
        .{ .width = 8, .height = 8 },
        p,
        .{ .x = 0, .y = 0, .width = 8, .height = 8 },
    ));

    // An empty rectangle is not damage.
    try testing.expectEqual(@as(?Image.Damage.Rect, null), Image.stagedRegion(
        .{ .width = 8, .height = 8 },
        p,
        .{ .x = 0, .y = 0, .width = 0, .height = 4 },
    ));
}
