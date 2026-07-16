const std = @import("std");
const assert = @import("../../quirks.zig").inlineAssert;
const Allocator = std.mem.Allocator;

const Terminal = @import("../Terminal.zig");
const kitty_animation = @import("graphics_animation.zig");
const command = @import("graphics_command.zig");
const image = @import("graphics_image.zig");
const Command = command.Command;
const Response = command.Response;
const LoadingImage = image.LoadingImage;
const Image = image.Image;
const ImageStorage = @import("graphics_storage.zig").ImageStorage;
const sys = @import("../sys.zig");

const log = std.log.scoped(.kitty_gfx);

/// Execute a Kitty graphics command against the given terminal. This
/// will never fail, but the response may indicate an error and the
/// terminal state may not be updated to reflect the command. This will
/// never put the terminal in an unrecoverable state, however.
///
/// The allocator must be the same allocator that was used to build
/// the command.
pub fn execute(
    alloc: Allocator,
    terminal: *Terminal,
    cmd: *const Command,
) ?Response {
    // If storage is disabled then we disable the full protocol. This means
    // we don't even respond to queries so the terminal completely acts as
    // if this feature is not supported.
    if (!terminal.screens.active.kitty_images.enabled()) {
        log.debug("kitty graphics requested but disabled", .{});
        return null;
    }

    log.debug("executing kitty graphics command: quiet={} control={}", .{
        cmd.quiet,
        cmd.control,
    });

    // The quiet settings used to control the response. We have to make this
    // a var because in certain special cases (namely chunked transmissions)
    // this can change.
    var quiet = cmd.quiet;

    const resp_: ?Response = switch (cmd.control) {
        .query => query(alloc, terminal, cmd),
        .display => display(alloc, terminal, cmd),
        .delete => delete(alloc, terminal, cmd),
        .control_animation => controlAnimation(alloc, terminal, cmd),
        .compose_animation => composeAnimation(alloc, terminal, cmd),

        .transmit,
        .transmit_and_display,
        .transmit_animation_frame,
        => resp: {
            // If we're transmitting, then our `q` setting value is complicated.
            // The `q` setting inherits the value from the starting command
            // unless `q` is set >= 1 on this command. If it is, then we save
            // that as the new `q` setting.
            const storage = &terminal.screens.active.kitty_images;
            if (storage.loading) |loading| switch (cmd.quiet) {
                // q=0 we use whatever the start command value is
                .no => quiet = loading.quiet,

                // q>=1 we use the new value, but we should already be set to it
                inline .ok, .failures => |tag| {
                    assert(quiet == tag);
                    loading.quiet = tag;
                },
            };

            break :resp transmitFamily(alloc, terminal, cmd);
        },
    };

    // Handle the quiet settings
    if (resp_) |resp| {
        if (!resp.ok()) {
            log.warn("erroneous kitty graphics response: {s}", .{resp.message});
        }

        return switch (quiet) {
            .no => if (resp.empty()) null else resp,
            .ok => if (resp.ok()) null else resp,
            .failures => null,
        };
    }

    return null;
}
/// Execute a "query" command.
///
/// This command is used to attempt to load an image and respond with
/// success/error but does not persist any of the command to the terminal
/// state.
fn query(
    alloc: Allocator,
    terminal: *const Terminal,
    cmd: *const Command,
) Response {
    const t = cmd.control.query;

    // Query requires image ID. We can't actually send a response without
    // an image ID either but we return an error and this will be logged
    // downstream.
    if (t.image_id == 0) {
        return .{ .message = "EINVAL: image ID required" };
    }

    // Build a partial response to start
    var result: Response = .{
        .id = t.image_id,
        .image_number = t.image_number,
        .placement_id = t.placement_id,
    };

    // Attempt to load the image. If we cannot, then set an appropriate error.
    const storage = &terminal.screens.active.kitty_images;
    var loading = LoadingImage.init(alloc, cmd, storage.image_limits) catch |err| {
        encodeError(&result, err);
        return result;
    };
    loading.deinit(alloc);

    return result;
}

/// Take the in-progress load out of storage, transferring ownership to the
/// caller.
///
/// This is the only way a load leaves storage, so completion and error
/// paths cannot disagree about who frees it.
fn takeLoading(storage: *ImageStorage) ?*LoadingImage {
    const loading = storage.loading orelse return null;
    storage.loading = null;
    return loading;
}

/// Abandon any load in progress, freeing it.
///
/// Call this at the point a new load is definitely starting: Kitty does
/// exactly this (initialize_load_data begins with free_load_data), and it
/// is what stops a malformed interleaved stream from stranding buffered
/// payloads for unbounded memory growth. Validation failures must happen
/// before this so a rejected command leaves the active load usable.
fn abandonLoading(alloc: Allocator, storage: *ImageStorage) void {
    const old = takeLoading(storage) orelse return;
    log.debug("abandoning in-progress load for a new one", .{});
    old.destroy(alloc);
}

/// Install a newly started load.
///
/// The caller must already have abandoned any load in progress, so this
/// can never strand one; the assert keeps that structural.
fn installLoading(storage: *ImageStorage, loading: *LoadingImage) void {
    assert(storage.loading == null);
    storage.loading = loading;
}

/// Execute any command that transmits data: "a=t", "a=T", "a=f", and the
/// actionless continuation chunk that Ghostty also accepts.
///
/// They share an entry point because only one load may be in progress and
/// it must have exactly one owner. Kitty routes *any* direct-medium
/// transmission command into the load already in progress, using the
/// parameters that load began with: the identifiers, format and dimensions
/// on a continuation are ignored, and an "a=f" resolves its target from the
/// active load rather than from its own "i"/"I". See graphics.c at
/// f47590533d7177daf0b74963f9d1b7581467af20: the `init_img` test in
/// handle_add_command, the `case 'f'` dispatch that prefers
/// `currently_loading.loading_for.image_id`, and INIT_CHUNKED_LOAD.
fn transmitFamily(
    alloc: Allocator,
    terminal: *Terminal,
    cmd: *const Command,
) Response {
    const storage = &terminal.screens.active.kitty_images;

    // Only the direct medium streams in chunks; a file or shared-memory
    // command carries its path in one go and always starts a new load.
    if (storage.loading != null and cmd.transmission().?.medium == .direct) {
        return continueLoad(alloc, terminal, cmd);
    }

    return switch (cmd.control) {
        .transmit_animation_frame => startFrameLoad(alloc, terminal, cmd),
        else => transmit(alloc, terminal, cmd),
    };
}

/// Append a chunk to the in-progress load, finishing it as whatever kind of
/// load it started as once the last chunk arrives.
fn continueLoad(
    alloc: Allocator,
    terminal: *Terminal,
    cmd: *const Command,
) Response {
    const storage = &terminal.screens.active.kitty_images;
    const loading = storage.loading.?;

    // The response identifies the load, not this chunk: a continuation
    // carries no identifiers of its own.
    var result: Response = if (loading.frame) |f| .{
        .id = f.target_image_id,
    } else .{
        .id = loading.image.id,
        .image_number = loading.image.number,
    };

    loading.addData(alloc, cmd.data) catch |err| {
        takeLoading(storage).?.destroy(alloc);
        encodeError(&result, err);
        return result;
    };

    // Intermediate chunks are never responded to.
    if (cmd.transmission().?.more_chunks) return .{};

    // That was the last chunk. Take the load out of storage so that it has
    // exactly one owner from here on, whatever happens.
    const owned = takeLoading(storage).?;
    defer alloc.destroy(owned);
    if (owned.frame != null) return finishAnimationFrame(alloc, terminal, owned, result);
    return finishTransmit(alloc, terminal, owned, cmd, result);
}

/// Transmit image data.
///
/// This loads the image, validates it, and puts it into the terminal
/// screen storage. It does not display the image.
fn transmit(
    alloc: Allocator,
    terminal: *Terminal,
    cmd: *const Command,
) Response {
    const t = cmd.transmission().?;
    var result: Response = .{
        .id = t.image_id,
        .image_number = t.image_number,
        .placement_id = t.placement_id,
    };
    if (t.image_id > 0 and t.image_number > 0) {
        return .{ .message = "EINVAL: image ID and number are mutually exclusive" };
    }

    const storage = &terminal.screens.active.kitty_images;
    var loading = LoadingImage.init(alloc, cmd, storage.image_limits) catch |err| {
        // Rejected before starting, so an active load stays usable.
        encodeError(&result, err);
        return result;
    };

    // We are definitely starting a load now, so any load in progress is
    // abandoned here, which is where Kitty abandons it too.
    abandonLoading(alloc, storage);

    // If the image has no ID, we assign one
    if (loading.image.id == 0) {
        loading.image.id = storage.next_image_id;
        storage.next_image_id +%= 1;

        // If the image also has no number then its auto-ID is "implicit".
        // See the doc comment on the Image.implicit_id field for more detail.
        if (loading.image.number == 0) loading.image.implicit_id = true;
    }

    // If more chunks are coming, park the load until they arrive. We
    // allocate the pointer on the heap because its rare and we don't want
    // to always pay the memory cost to keep it around.
    if (t.more_chunks) {
        const ptr = alloc.create(LoadingImage) catch {
            loading.deinit(alloc);
            result.message = "ENOMEM: out of memory";
            return result;
        };
        ptr.* = loading;
        installLoading(storage, ptr);
        return .{};
    }

    return finishTransmit(alloc, terminal, &loading, cmd, result);
}

/// Complete a fully-received image load, store it, and display it if the
/// command that started the load asked for that.
///
/// Deinitializes the load's contents on every path; the caller owns any
/// heap allocation holding the load itself.
fn finishTransmit(
    alloc: Allocator,
    terminal: *Terminal,
    loading: *LoadingImage,
    cmd: *const Command,
    base: Response,
) Response {
    defer loading.deinit(alloc);

    const storage = &terminal.screens.active.kitty_images;
    var result = base;

    // Dump the image data before it is decompressed
    // loading.debugDump() catch unreachable;

    // Validate and store our image. complete() takes the pixels out of the
    // load, so the deferred deinit above won't double free them.
    var img = loading.complete(alloc) catch |err| {
        encodeError(&result, err);
        return result;
    };
    storage.addImage(alloc, terminal.screens.active, img) catch |err| {
        img.deinit(alloc);
        encodeError(&result, err);
        return result;
    };

    // If we're also displaying, then do that now. The display is carried
    // by the load because it may have been deferred across chunks.
    if (loading.display) |d| {
        var d_copy = d;
        d_copy.image_id = img.id;
        result = display(alloc, terminal, &.{
            .control = .{ .display = d_copy },
            .quiet = cmd.quiet,
        });
    }

    // If the loaded image was assigned its ID automatically, not based
    // on a number or explicitly specified ID, then we don't respond.
    if (img.implicit_id) return .{};

    // After the image is added, set the ID in case it changed.
    // The resulting image number and placement ID never change.
    result.id = img.id;

    return result;
}

/// Display a previously transmitted image.
fn display(
    alloc: Allocator,
    terminal: *Terminal,
    cmd: *const Command,
) Response {
    const d = cmd.display().?;

    // Display requires image ID or number.
    if (d.image_id == 0 and d.image_number == 0) {
        return .{ .message = "EINVAL: image ID or number required" };
    }

    // Build up our response
    var result: Response = .{
        .id = d.image_id,
        .image_number = d.image_number,
        .placement_id = d.placement_id,
    };

    // Verify the requested image exists if we have an ID
    const storage = &terminal.screens.active.kitty_images;
    const img_: ?Image = if (d.image_id != 0)
        storage.imageById(d.image_id)
    else
        storage.imageByNumber(d.image_number);
    const img = img_ orelse {
        result.message = "ENOENT: image not found";
        return result;
    };

    // Make sure our response has the image id in case we looked up by number
    result.id = img.id;

    // Location where the placement will go.
    const location: ImageStorage.Placement.Location = location: {
        // Virtual placements are not tracked
        if (d.virtual_placement) {
            if (d.parent_id > 0) {
                result.message = "EINVAL: virtual placement cannot refer to a parent";
                return result;
            }

            break :location .{ .virtual = {} };
        }

        // Track a new pin for our cursor. The cursor is always tracked but we
        // don't want this one to move with the cursor.
        const pin = terminal.screens.active.pages.trackPin(
            terminal.screens.active.cursor.page_pin.*,
        ) catch |err| {
            log.warn("failed to create pin for Kitty graphics err={}", .{err});
            result.message = "EINVAL: failed to prepare terminal state";
            return result;
        };
        break :location .{ .pin = pin };
    };

    // Add the placement
    const p: ImageStorage.Placement = .{
        .location = location,
        .x_offset = d.x_offset,
        .y_offset = d.y_offset,
        .source_x = d.x,
        .source_y = d.y,
        .source_width = d.width,
        .source_height = d.height,
        .columns = d.columns,
        .rows = d.rows,
        .z = d.z,
    };
    storage.addPlacement(
        alloc,
        img.id,
        result.placement_id,
        p,
    ) catch |err| {
        p.deinit(terminal.screens.active);
        encodeError(&result, err);
        return result;
    };

    // Apply cursor movement setting. This only applies to pin placements.
    switch (p.location) {
        .virtual => {},
        .pin => |pin| switch (d.cursor_movement) {
            .none => {},
            .after => {
                // We use terminal.index to properly handle scroll regions.
                const size = p.gridSize(img, terminal);
                for (0..size.rows) |_| terminal.index() catch |err| {
                    log.warn("failed to move cursor: {}", .{err});
                    break;
                };

                terminal.setCursorPos(
                    terminal.screens.active.cursor.y,
                    pin.x + size.cols + 1,
                );
            },
        },
    }

    return result;
}

/// Display a previously transmitted image.
fn delete(
    alloc: Allocator,
    terminal: *Terminal,
    cmd: *const Command,
) Response {
    const storage = &terminal.screens.active.kitty_images;

    switch (cmd.control.delete) {
        // Frame deletion is the only delete that can produce a response,
        // and only to reject a malformed command: a missing image is
        // logged rather than reported, like every other delete.
        .animation_frames => |v| switch (storage.deleteAnimationFrame(
            alloc,
            terminal,
            v,
            storage.animationNowMs() orelse 0,
        )) {
            .invalid_identifiers => return .{
                .id = v.image_id,
                .image_number = v.image_number,
                .message = "EINVAL: image ID and number are mutually exclusive",
            },
            .changed, .no_op_or_missing => {},
        },

        else => storage.delete(alloc, terminal, cmd.control.delete),
    }

    // Delete never responds on success
    return .{};
}

/// Start an animation frame load ("a=f").
///
/// The frame's pixels are composed into an existing image rather than
/// becoming an image of their own, so unlike a transmit this needs the
/// target to already exist. Continuation chunks do not come here; see
/// transmitFamily.
fn startFrameLoad(
    alloc: Allocator,
    terminal: *Terminal,
    cmd: *const Command,
) Response {
    const storage = &terminal.screens.active.kitty_images;
    const t = cmd.transmission().?;

    const f = cmd.control.transmit_animation_frame.frame;
    var result: Response = .{ .id = t.image_id, .image_number = t.image_number };

    const target_id = switch (storage.resolveAnimationImagePtr(
        t.image_id,
        t.image_number,
    )) {
        .found => |img| img.id,
        .conflict => {
            result.message = "EINVAL: image ID and number are mutually exclusive";
            return result;
        },
        .no_identifier => {
            log.warn("animation frame requires an image id or number", .{});
            return .{};
        },
        .not_found => {
            result.message = "ENOENT: image not found";
            return result;
        },
    };
    result.id = target_id;

    var loading = LoadingImage.init(alloc, cmd, storage.image_limits) catch |err| {
        encodeError(&result, err);
        return result;
    };
    loading.frame = .{ .target_image_id = target_id, .params = f };

    // As in transmit: the load has definitely started, so abandon any that
    // was in progress. Everything above this can still reject and leave it.
    abandonLoading(alloc, storage);

    // If more chunks are coming, park the load until they arrive.
    if (t.more_chunks) {
        const ptr = alloc.create(LoadingImage) catch {
            loading.deinit(alloc);
            result.message = "ENOMEM: out of memory";
            return result;
        };
        ptr.* = loading;
        installLoading(storage, ptr);
        return .{};
    }

    return finishAnimationFrame(alloc, terminal, &loading, result);
}

/// Decode a fully-received animation frame and compose it into its image.
/// Takes ownership of the load, which is cleaned up on every path.
fn finishAnimationFrame(
    alloc: Allocator,
    terminal: *Terminal,
    loading: *LoadingImage,
    base: Response,
) Response {
    defer loading.deinit(alloc);

    const storage = &terminal.screens.active.kitty_images;
    const frame = loading.frame.?;
    var result = base;

    loading.completeFrame(alloc) catch |err| {
        encodeError(&result, err);
        return result;
    };

    // The image can go away between chunks, e.g. evicted to make room for
    // the very data we were loading.
    if (storage.images.getPtr(frame.target_image_id) == null) {
        result.message = "ENOENT: image not found";
        return result;
    }

    const stored = storage.addAnimationFrame(
        alloc,
        terminal.screens.active,
        frame.target_image_id,
        frame.params,
        loading.data.items,
        loading.image.format,
        loading.image.width,
        loading.image.height,
        storage.animationNowMs() orelse 0,
    ) catch |err| {
        encodeAnimationError(&result, err);
        return result;
    };

    // Report the frame we actually resolved to: "r" may have been clamped
    // or the frame appended, so the client can't work it out itself.
    result.frame_number = stored.frame;
    return result;
}

/// Execute an "a=a" (animation control) command.
fn controlAnimation(
    alloc: Allocator,
    terminal: *Terminal,
    cmd: *const Command,
) Response {
    const storage = &terminal.screens.active.kitty_images;
    const c = cmd.control.control_animation;

    var result: Response = .{ .id = c.image_id, .image_number = c.image_number };

    const img = switch (storage.resolveAnimationImagePtr(c.image_id, c.image_number)) {
        .found => |img| img,
        .conflict => {
            result.message = "EINVAL: image ID and number are mutually exclusive";
            return result;
        },
        .no_identifier => {
            log.warn("animation control requires an image id or number", .{});
            return .{};
        },
        .not_found => {
            result.message = "ENOENT: image not found";
            return result;
        },
    };

    // Controlling an image that has no animation state yet is legal, and
    // has to allocate it. A command that can't change anything doesn't:
    // a bare "a=a", or one naming only the frame that's already current,
    // has nothing to store. Note that this path never touches pixels, so
    // it neither widens the root nor consumes any of the byte quota.
    const anim: *kitty_animation.Animation = img.anim orelse anim: {
        const changes_state = c.action != .invalid or
            c.loops > 0 or
            (c.frame > 0 and c.gap != 0);
        if (!changes_state) return .{};

        const new = alloc.create(kitty_animation.Animation) catch {
            result.message = "ENOMEM: out of memory";
            return result;
        };
        new.* = .{};
        img.anim = new;
        storage.animation_count += 1;
        break :anim new;
    };

    const now_ms = storage.animationNowMs() orelse 0;

    // "r" plus "z" sets a frame's gap. This is the only way to give the
    // root frame one, since it defaults to no gap. An out of range frame
    // is ignored rather than being an error.
    if (c.frame > 0 and c.gap != 0 and @as(usize, c.frame) <= anim.frameCount()) {
        anim.setGap(c.frame - 1, kitty_animation.resolveGap(c.gap));
    }

    // "c" switches the frame being displayed.
    var visible = false;
    if (c.current_frame > 0 and
        @as(usize, c.current_frame) <= anim.frameCount() and
        c.current_frame - 1 != anim.current_frame)
    {
        anim.current_frame = c.current_frame - 1;
        anim.last_frame_ms = now_ms;
        visible = true;
    }

    // "s" sets the playback state. Any state command restarts the loop
    // count, and starting a stopped animation restarts its frame timer so
    // the current frame gets its full gap.
    if (c.action != .invalid) {
        const was_stopped = anim.state == .stopped;
        anim.state = switch (c.action) {
            .invalid => unreachable,
            .stop => .stopped,
            .run_wait => .loading,
            .run => .running,
        };
        anim.current_loop = 0;
        if (was_stopped and anim.state != .stopped) anim.last_frame_ms = now_ms;
    }

    // "v" is the loop count, off by one: 1 means loop forever.
    if (c.loops > 0) anim.max_loops = c.loops - 1;

    if (visible) {
        // The pixels on screen changed, so the image needs a new stamp to
        // make the renderer re-upload its texture.
        storage.markMutated();
        img.generation = storage.generation;
    } else {
        // Gap, state and loop changes only affect *when* frames are
        // shown. Marking the storage dirty gets the renderer to recompute
        // its schedule; bumping the generation would pointlessly re-upload
        // an unchanged texture on every such command.
        storage.dirty = true;
    }

    // Kitty sends no response at all for a successful animation control.
    return .{};
}

/// Execute an "a=c" (compose animation frames) command.
fn composeAnimation(
    alloc: Allocator,
    terminal: *Terminal,
    cmd: *const Command,
) Response {
    const storage = &terminal.screens.active.kitty_images;
    const c = cmd.control.compose_animation;

    var result: Response = .{ .id = c.image_id, .image_number = c.image_number };

    const target_id = switch (storage.resolveAnimationImagePtr(
        c.image_id,
        c.image_number,
    )) {
        .found => |img| img.id,
        .conflict => {
            result.message = "EINVAL: image ID and number are mutually exclusive";
            return result;
        },
        .no_identifier => {
            log.warn("animation compose requires an image id or number", .{});
            return .{};
        },
        .not_found => {
            result.message = "ENOENT: image not found";
            return result;
        },
    };
    result.id = target_id;

    _ = storage.composeAnimationFrames(
        alloc,
        terminal.screens.active,
        target_id,
        c,
        storage.animationNowMs() orelse 0,
    ) catch |err| {
        encodeAnimationError(&result, err);
        return result;
    };

    return result;
}

const EncodeableError = Image.Error || Allocator.Error;

/// Encode an animation storage error into a message for a response.
fn encodeAnimationError(r: *Response, err: ImageStorage.AnimationError) void {
    switch (err) {
        error.OutOfMemory => r.message = "ENOMEM: out of memory",
        error.OutOfSpace => r.message = "ENOSPC: no space for animation frame",
        error.BaseFrameNotFound => r.message = "EINVAL: base frame does not exist",
        error.FrameNotFound => r.message = "ENOENT: frame does not exist",
        error.InvalidRect => r.message = "EINVAL: invalid rectangle",
    }
}

/// Encode an error code into a message for a response.
fn encodeError(r: *Response, err: EncodeableError) void {
    switch (err) {
        error.OutOfMemory => r.message = "ENOMEM: out of memory",
        error.InvalidData => r.message = "EINVAL: invalid data",
        error.InsufficientData => r.message = "ENODATA: insufficient data for frame",
        error.DecompressionFailed => r.message = "EINVAL: decompression failed",
        error.FilePathTooLong => r.message = "EINVAL: file path too long",
        error.TemporaryFileNotInTempDir => r.message = "EINVAL: temporary file not in temp dir",
        error.TemporaryFileNotNamedCorrectly => r.message = "EINVAL: temporary file not named correctly",
        error.UnsupportedFormat => r.message = "EINVAL: unsupported format",
        error.UnsupportedMedium => r.message = "EINVAL: unsupported medium",
        error.UnsupportedDepth => r.message = "EINVAL: unsupported pixel depth",
        error.DimensionsRequired => r.message = "EINVAL: dimensions required",
        error.DimensionsTooLarge => r.message = "EINVAL: dimensions too large",
    }
}

test "kittygfx more chunks with q=1" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);

    // Initial chunk has q=1
    {
        const cmd = try command.Parser.parseString(
            alloc,
            "a=T,f=24,t=d,i=1,s=1,v=2,c=10,r=1,m=1,q=1;////",
        );
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd);
        try testing.expect(resp == null);
    }

    // Subsequent chunk has no q but should respect initial
    {
        const cmd = try command.Parser.parseString(
            alloc,
            "m=0;////",
        );
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd);
        try testing.expect(resp == null);
    }
}

test "kittygfx more chunks with q=0" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);

    // Initial chunk has q=0
    {
        const cmd = try command.Parser.parseString(
            alloc,
            "a=t,f=24,t=d,s=1,v=2,c=10,r=1,m=1,i=1,q=0;////",
        );
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd);
        try testing.expect(resp == null);
    }

    // Subsequent chunk has no q so should respond OK
    {
        const cmd = try command.Parser.parseString(
            alloc,
            "m=0;////",
        );
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd).?;
        try testing.expect(resp.ok());
    }
}

test "kittygfx more chunks with chunk increasing q" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);

    // Initial chunk has q=0
    {
        const cmd = try command.Parser.parseString(
            alloc,
            "a=t,f=24,t=d,s=1,v=2,c=10,r=1,m=1,i=1,q=0;////",
        );
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd);
        try testing.expect(resp == null);
    }

    // Subsequent chunk sets q=1 so should not respond
    {
        const cmd = try command.Parser.parseString(
            alloc,
            "m=0,q=1;////",
        );
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd);
        try testing.expect(resp == null);
    }
}

test "kittygfx default format is rgba" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);

    const cmd = try command.Parser.parseString(
        alloc,
        "a=t,t=d,i=1,s=1,v=2,c=10,r=1;///////////",
    );
    defer cmd.deinit(alloc);
    const resp = execute(alloc, &t, &cmd).?;
    try testing.expect(resp.ok());

    const storage = &t.screens.active.kitty_images;
    const img = storage.imageById(1).?;
    try testing.expectEqual(command.Transmission.Format.rgba, img.format);
}

test "kittygfx test valid u32 (expect invalid image ID)" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);

    const cmd = try command.Parser.parseString(
        alloc,
        "a=p,i=4294967295",
    );
    defer cmd.deinit(alloc);
    const resp = execute(alloc, &t, &cmd).?;
    try testing.expect(!resp.ok());
    try testing.expectEqual(resp.message, "ENOENT: image not found");
}

test "kittygfx test valid i32 (expect invalid image ID)" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);

    const cmd = try command.Parser.parseString(
        alloc,
        "a=p,i=1,z=-2147483648",
    );
    defer cmd.deinit(alloc);
    const resp = execute(alloc, &t, &cmd).?;
    try testing.expect(!resp.ok());
    try testing.expectEqual(resp.message, "ENOENT: image not found");
}

test "kittygfx no response with no image ID or number" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);

    {
        const cmd = try command.Parser.parseString(
            alloc,
            "a=t,f=24,t=d,s=1,v=2,c=10,r=1,i=0,I=0;////////",
        );
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd);
        try testing.expect(resp == null);
    }
}

test "kittygfx no response with no image ID or number load and display" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);

    {
        const cmd = try command.Parser.parseString(
            alloc,
            "a=T,f=24,t=d,s=1,v=2,c=10,r=1,i=0,I=0;////////",
        );
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd);
        try testing.expect(resp == null);
    }
}

test "kittygfx retransmit same id gets fresh image generation" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);
    const storage = &t.screens.active.kitty_images;

    // Transmit a 1x2 RGB image with id=1.
    {
        const cmd = try command.Parser.parseString(
            alloc,
            "a=t,t=d,f=24,i=1,s=1,v=2;////////",
        );
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd).?;
        try testing.expect(resp.ok());
    }
    const gen1 = storage.imageById(1).?.generation;
    try testing.expect(gen1 > 0);
    try testing.expectEqual(gen1, storage.generation);

    // Retransmit the same id with identical dimensions/length. The
    // (width, height, format, len) tuple is identical, so only the
    // generation can reveal that the contents were replaced.
    {
        const cmd = try command.Parser.parseString(
            alloc,
            "a=t,t=d,f=24,i=1,s=1,v=2;AAAAAAAA",
        );
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd).?;
        try testing.expect(resp.ok());
    }
    const gen2 = storage.imageById(1).?.generation;
    try testing.expect(gen2 > gen1);
    try testing.expectEqual(gen2, storage.generation);
}

test "kittygfx delete then retransmit same id gets fresh generation" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);
    const storage = &t.screens.active.kitty_images;

    // Transmit and display, then delete everything (including image
    // data), then retransmit the same ID.
    {
        const cmd = try command.Parser.parseString(
            alloc,
            "a=T,t=d,f=24,i=1,s=1,v=2,c=1,r=1;////////",
        );
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd).?;
        try testing.expect(resp.ok());
    }
    const gen1 = storage.imageById(1).?.generation;

    {
        const cmd = try command.Parser.parseString(alloc, "a=d,d=A");
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd);
        try testing.expect(resp == null);
    }
    try testing.expect(storage.imageById(1) == null);
    const gen_delete = storage.generation;
    try testing.expect(gen_delete > gen1);

    {
        const cmd = try command.Parser.parseString(
            alloc,
            "a=t,t=d,f=24,i=1,s=1,v=2;////////",
        );
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd).?;
        try testing.expect(resp.ok());
    }
    const gen2 = storage.imageById(1).?.generation;
    try testing.expect(gen2 > gen1);
    try testing.expect(gen2 > gen_delete);
}

/// Transmit a 2x2 RGBA image with the given ID for animation tests.
fn testTransmitImage(alloc: Allocator, t: *Terminal, id: u32) !void {
    // 2x2 RGBA of all zero bytes, base64 encoded.
    var buf: [64]u8 = undefined;
    const str = try std.fmt.bufPrint(
        &buf,
        "a=t,f=32,t=d,i={},s=2,v=2;AAAAAAAAAAAAAAAAAAAAAA==",
        .{id},
    );
    const cmd = try command.Parser.parseString(alloc, str);
    defer cmd.deinit(alloc);
    const resp = execute(alloc, t, &cmd).?;
    try std.testing.expect(resp.ok());
}

test "kittygfx animation frame: append reports resolved frame" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);
    try testTransmitImage(alloc, &t, 1);

    // A full 2x2 RGBA frame.
    const cmd = try command.Parser.parseString(
        alloc,
        "a=f,i=1,f=32,s=2,v=2;/////////////////////w==",
    );
    defer cmd.deinit(alloc);
    const resp = execute(alloc, &t, &cmd).?;

    try testing.expect(resp.ok());
    try testing.expectEqual(@as(u32, 1), resp.id);
    try testing.expectEqual(@as(u32, 2), resp.frame_number);

    // Encoded, the response reports the frame it resolved to.
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    try resp.encode(&writer);
    try testing.expectEqualStrings("\x1b_Gi=1,r=2;OK\x1b\\", writer.buffered());

    const img = t.screens.active.kitty_images.images.getPtr(1).?;
    try testing.expectEqual(@as(usize, 1), img.anim.?.frames.items.len);
    try testing.expectEqual(@as(u32, 40), img.anim.?.frames.items[0].gap_ms);
}

test "kittygfx animation frame: identifiers" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);
    try testTransmitImage(alloc, &t, 1);

    // An ID and a number together is invalid.
    {
        const cmd = try command.Parser.parseString(alloc, "a=f,i=1,I=2,f=32,s=1,v=1;AAAAAA==");
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd).?;
        try testing.expect(std.mem.startsWith(u8, resp.message, "EINVAL"));
    }

    // No identifier at all is not responded to.
    {
        const cmd = try command.Parser.parseString(alloc, "a=f,f=32,s=1,v=1;AAAAAA==");
        defer cmd.deinit(alloc);
        try testing.expect(execute(alloc, &t, &cmd) == null);
    }

    // An unknown image, by ID and by number.
    {
        const cmd = try command.Parser.parseString(alloc, "a=f,i=42,f=32,s=1,v=1;AAAAAA==");
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd).?;
        try testing.expect(std.mem.startsWith(u8, resp.message, "ENOENT"));
    }
    {
        const cmd = try command.Parser.parseString(alloc, "a=f,I=42,f=32,s=1,v=1;AAAAAA==");
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd).?;
        try testing.expect(std.mem.startsWith(u8, resp.message, "ENOENT"));
    }
}

test "kittygfx animation frame: data sizing" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);
    try testTransmitImage(alloc, &t, 1);

    // Short data for the declared rectangle is ENODATA.
    {
        const cmd = try command.Parser.parseString(alloc, "a=f,i=1,f=32,s=2,v=2;AAAAAA==");
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd).?;
        try testing.expect(std.mem.startsWith(u8, resp.message, "ENODATA"));
    }

    // Surplus data succeeds, using only the prefix it needs. Here a 1x1
    // rect is sent two pixels of data.
    {
        const cmd = try command.Parser.parseString(alloc, "a=f,i=1,f=32,s=1,v=1;/wAA/xERERE=");
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd).?;
        try testing.expect(resp.ok());
    }

    // A rectangle larger than the image is EINVAL.
    {
        const cmd = try command.Parser.parseString(
            alloc,
            "a=f,i=1,f=32,s=3,v=3;" ++ "/" ** 48,
        );
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd).?;
        try testing.expect(std.mem.startsWith(u8, resp.message, "EINVAL"));
    }
}

test "kittygfx animation frame: chunked" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);
    try testTransmitImage(alloc, &t, 1);

    // The initial chunk carries the metadata and is not responded to.
    {
        const cmd = try command.Parser.parseString(
            alloc,
            "a=f,i=1,f=32,s=2,v=2,z=100,m=1;//////////8=",
        );
        defer cmd.deinit(alloc);
        try testing.expect(execute(alloc, &t, &cmd) == null);
        try testing.expect(t.screens.active.kitty_images.loading != null);
    }

    // The protocol requires "a=f" on every chunk.
    {
        const cmd = try command.Parser.parseString(alloc, "a=f,m=0;//////////8=");
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd).?;
        try testing.expect(resp.ok());
        try testing.expectEqual(@as(u32, 2), resp.frame_number);
    }

    const storage = &t.screens.active.kitty_images;
    try testing.expect(storage.loading == null);
    const img = storage.images.getPtr(1).?;
    try testing.expectEqual(@as(usize, 1), img.anim.?.frames.items.len);
    try testing.expectEqual(@as(u32, 100), img.anim.?.frames.items[0].gap_ms);

    // The frame parameters came from the initial chunk, not the final one.
    const expected: [16]u8 = @splat(255);
    try testing.expectEqualSlices(u8, &expected, img.anim.?.frames.items[0].data);
}

test "kittygfx animation frame: chunked with actionless final chunk" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);
    try testTransmitImage(alloc, &t, 1);

    {
        const cmd = try command.Parser.parseString(
            alloc,
            "a=f,i=1,f=32,s=2,v=2,m=1;//////////8=",
        );
        defer cmd.deinit(alloc);
        try testing.expect(execute(alloc, &t, &cmd) == null);
    }

    // Ghostty also accepts a final chunk with no action, which parses as
    // a plain transmit but belongs to the frame load in progress.
    {
        const cmd = try command.Parser.parseString(alloc, "m=0;//////////8=");
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd).?;
        try testing.expect(resp.ok());
        try testing.expectEqual(@as(u32, 2), resp.frame_number);
    }

    const storage = &t.screens.active.kitty_images;
    try testing.expect(storage.loading == null);
    try testing.expectEqual(
        @as(usize, 1),
        storage.images.getPtr(1).?.anim.?.frames.items.len,
    );
}

test "kittygfx animation frame: chunked failure clears loading" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);
    try testTransmitImage(alloc, &t, 1);

    // Declare a 2x2 frame but only ever send one pixel.
    {
        const cmd = try command.Parser.parseString(alloc, "a=f,i=1,f=32,s=2,v=2,m=1;AAAAAA==");
        defer cmd.deinit(alloc);
        try testing.expect(execute(alloc, &t, &cmd) == null);
    }
    {
        const cmd = try command.Parser.parseString(alloc, "a=f,m=0;");
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd).?;
        try testing.expect(std.mem.startsWith(u8, resp.message, "ENODATA"));
    }

    // The failed load is cleaned up and the image is untouched.
    const storage = &t.screens.active.kitty_images;
    try testing.expect(storage.loading == null);
    try testing.expect(storage.images.getPtr(1).?.anim == null);
}

test "kittygfx animation frame: quiet inherits across chunks" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);
    try testTransmitImage(alloc, &t, 1);

    // q=1 on the initial chunk suppresses the OK from the final one.
    {
        const cmd = try command.Parser.parseString(
            alloc,
            "a=f,i=1,f=32,s=2,v=2,m=1,q=1;////////////////",
        );
        defer cmd.deinit(alloc);
        try testing.expect(execute(alloc, &t, &cmd) == null);
    }
    {
        const cmd = try command.Parser.parseString(alloc, "a=f,m=0;//////////8=");
        defer cmd.deinit(alloc);
        try testing.expect(execute(alloc, &t, &cmd) == null);
    }

    try testing.expectEqual(
        @as(usize, 1),
        t.screens.active.kitty_images.images.getPtr(1).?.anim.?.frames.items.len,
    );
}

test "kittygfx animation control: no response on success" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);
    try testTransmitImage(alloc, &t, 1);

    // A successful animation control is silent, even for q=0.
    const cmd = try command.Parser.parseString(alloc, "a=a,i=1,s=3,v=1");
    defer cmd.deinit(alloc);
    try testing.expect(execute(alloc, &t, &cmd) == null);

    const anim = t.screens.active.kitty_images.images.getPtr(1).?.anim.?;
    try testing.expectEqual(kitty_animation.Animation.State.running, anim.state);
    try testing.expectEqual(@as(u32, 0), anim.max_loops); // v=1 is forever
}

test "kittygfx animation control: missing image" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);

    const cmd = try command.Parser.parseString(alloc, "a=a,i=42,s=3");
    defer cmd.deinit(alloc);
    const resp = execute(alloc, &t, &cmd).?;
    try testing.expect(std.mem.startsWith(u8, resp.message, "ENOENT"));
}

test "kittygfx animation control: gaps loops and frames" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);
    try testTransmitImage(alloc, &t, 1);

    // Give the root a gap. This is the only way to do it.
    {
        const cmd = try command.Parser.parseString(alloc, "a=a,i=1,r=1,z=500");
        defer cmd.deinit(alloc);
        try testing.expect(execute(alloc, &t, &cmd) == null);
    }

    const storage = &t.screens.active.kitty_images;
    const img = storage.images.getPtr(1).?;
    try testing.expectEqual(@as(u32, 500), img.anim.?.root_gap_ms);
    try testing.expectEqual(@as(usize, 1), storage.animation_count);

    // A negative gap means gapless.
    {
        const cmd = try command.Parser.parseString(alloc, "a=a,i=1,r=1,z=-1");
        defer cmd.deinit(alloc);
        try testing.expect(execute(alloc, &t, &cmd) == null);
    }
    try testing.expectEqual(@as(u32, 0), img.anim.?.root_gap_ms);
    try testing.expectEqual(@as(u32, 0), img.anim.?.nonzero_gap_count);

    // An out of range frame is ignored rather than being an error.
    {
        const cmd = try command.Parser.parseString(alloc, "a=a,i=1,r=99,z=100");
        defer cmd.deinit(alloc);
        try testing.expect(execute(alloc, &t, &cmd) == null);
    }
    try testing.expectEqual(@as(u32, 0), img.anim.?.root_gap_ms);

    // v=5 means four loops.
    {
        const cmd = try command.Parser.parseString(alloc, "a=a,i=1,v=5");
        defer cmd.deinit(alloc);
        try testing.expect(execute(alloc, &t, &cmd) == null);
    }
    try testing.expectEqual(@as(u32, 4), img.anim.?.max_loops);
}

test "kittygfx animation control: bare command allocates nothing" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);
    try testTransmitImage(alloc, &t, 1);

    const storage = &t.screens.active.kitty_images;
    const bytes_before = storage.total_bytes;

    // Neither a bare "a=a" nor one naming the frame that's already
    // current changes anything, so neither needs animation state. Note
    // this path must not widen the root either.
    for ([_][]const u8{ "a=a,i=1", "a=a,i=1,c=1" }) |input| {
        const cmd = try command.Parser.parseString(alloc, input);
        defer cmd.deinit(alloc);
        try testing.expect(execute(alloc, &t, &cmd) == null);
    }

    const img = storage.images.getPtr(1).?;
    try testing.expect(img.anim == null);
    try testing.expectEqual(@as(usize, 0), storage.animation_count);
    try testing.expectEqual(bytes_before, storage.total_bytes);
    try testing.expectEqual(command.Transmission.Format.rgba, img.format);
}

test "kittygfx animation control: switching frames" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);
    try testTransmitImage(alloc, &t, 1);

    {
        const cmd = try command.Parser.parseString(
            alloc,
            "a=f,i=1,f=32,s=2,v=2;/////////////////////w==",
        );
        defer cmd.deinit(alloc);
        _ = execute(alloc, &t, &cmd);
    }

    const storage = &t.screens.active.kitty_images;
    const img = storage.images.getPtr(1).?;
    const gen_before = img.generation;

    // Switching to frame 2 changes the pixels on screen, so the image is
    // restamped to make the renderer re-upload.
    {
        const cmd = try command.Parser.parseString(alloc, "a=a,i=1,c=2");
        defer cmd.deinit(alloc);
        try testing.expect(execute(alloc, &t, &cmd) == null);
    }
    try testing.expectEqual(@as(u32, 1), img.anim.?.current_frame);
    try testing.expect(img.generation != gen_before);
    try testing.expectEqual(@as(u8, 255), img.renderData()[0]);

    // A state-only change reschedules but must not restamp the image.
    const gen_running = img.generation;
    storage.dirty = false;
    {
        const cmd = try command.Parser.parseString(alloc, "a=a,i=1,s=3");
        defer cmd.deinit(alloc);
        try testing.expect(execute(alloc, &t, &cmd) == null);
    }
    try testing.expectEqual(gen_running, img.generation);
    try testing.expect(storage.dirty);
}

test "kittygfx animation compose" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);
    try testTransmitImage(alloc, &t, 1);

    // Frame 2, all white.
    {
        const cmd = try command.Parser.parseString(
            alloc,
            "a=f,i=1,f=32,s=2,v=2;/////////////////////w==",
        );
        defer cmd.deinit(alloc);
        _ = execute(alloc, &t, &cmd);
    }

    // Compose a 1x1 rect from frame 2 (source, "r") onto frame 1
    // (destination, "c").
    {
        const cmd = try command.Parser.parseString(alloc, "a=c,i=1,r=2,c=1,w=1,h=1,C=1");
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd).?;
        try testing.expect(resp.ok());
    }

    const img = t.screens.active.kitty_images.images.getPtr(1).?;
    try testing.expectEqualSlices(u8, &.{
        255, 255, 255, 255, 0, 0, 0, 0,
        0,   0,   0,   0,   0, 0, 0, 0,
    }, img.data);
}

test "kittygfx animation compose: errors" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);
    try testTransmitImage(alloc, &t, 1);

    // A missing source or destination frame.
    {
        const cmd = try command.Parser.parseString(alloc, "a=c,i=1,r=9,c=1");
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd).?;
        try testing.expect(std.mem.startsWith(u8, resp.message, "ENOENT"));
    }

    // Composing a frame onto itself with overlapping rects.
    {
        const cmd = try command.Parser.parseString(alloc, "a=c,i=1,r=1,c=1");
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd).?;
        try testing.expect(std.mem.startsWith(u8, resp.message, "EINVAL"));
    }

    // A rect running off the image.
    {
        const cmd = try command.Parser.parseString(alloc, "a=c,i=1,r=1,c=1,w=2,h=2,x=1");
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd).?;
        try testing.expect(std.mem.startsWith(u8, resp.message, "EINVAL"));
    }
}

test "kittygfx delete animation frames" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);
    try testTransmitImage(alloc, &t, 1);

    {
        const cmd = try command.Parser.parseString(
            alloc,
            "a=f,i=1,f=32,s=2,v=2;/////////////////////w==",
        );
        defer cmd.deinit(alloc);
        _ = execute(alloc, &t, &cmd);
    }

    // Deleting a frame never responds.
    {
        const cmd = try command.Parser.parseString(alloc, "a=d,d=f,i=1,r=2");
        defer cmd.deinit(alloc);
        try testing.expect(execute(alloc, &t, &cmd) == null);
    }
    const storage = &t.screens.active.kitty_images;
    try testing.expectEqual(
        @as(usize, 0),
        storage.images.getPtr(1).?.anim.?.frames.items.len,
    );

    // An ID and a number together is still rejected.
    {
        const cmd = try command.Parser.parseString(alloc, "a=d,d=f,i=1,I=2");
        defer cmd.deinit(alloc);
        const resp = execute(alloc, &t, &cmd).?;
        try testing.expect(std.mem.startsWith(u8, resp.message, "EINVAL"));
    }

    // With no frames left, the uppercase form deletes the image.
    {
        const cmd = try command.Parser.parseString(alloc, "a=d,d=F,i=1");
        defer cmd.deinit(alloc);
        try testing.expect(execute(alloc, &t, &cmd) == null);
    }
    try testing.expectEqual(@as(usize, 0), storage.images.count());
}

test "kittygfx animation frame: formats agree" {
    const testing = std.testing;
    const alloc = testing.allocator;

    // Whatever a frame is transmitted as, it is stored as RGBA, so an
    // opaque white 2x2 frame is the same bytes every way it can be sent.
    const white: [2 * 2 * 4]u8 = @splat(255);

    // f=32: RGBA directly.
    {
        var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
        defer t.deinit(alloc);
        try testTransmitImage(alloc, &t, 1);

        const cmd = try command.Parser.parseString(
            alloc,
            "a=f,i=1,f=32,s=2,v=2;/////////////////////w==",
        );
        defer cmd.deinit(alloc);
        try testing.expect(execute(alloc, &t, &cmd).?.ok());
        try testing.expectEqualSlices(
            u8,
            &white,
            t.screens.active.kitty_images.images.getPtr(1).?.anim.?.frames.items[0].data,
        );
    }

    // f=24: RGB, widened to opaque RGBA.
    {
        var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
        defer t.deinit(alloc);
        try testTransmitImage(alloc, &t, 1);

        const cmd = try command.Parser.parseString(
            alloc,
            "a=f,i=1,f=24,s=2,v=2;////////////////",
        );
        defer cmd.deinit(alloc);
        try testing.expect(execute(alloc, &t, &cmd).?.ok());
        try testing.expectEqualSlices(
            u8,
            &white,
            t.screens.active.kitty_images.images.getPtr(1).?.anim.?.frames.items[0].data,
        );
    }

    // f=100: PNG, which also supplies the rectangle's dimensions itself.
    {
        if (sys.decode_png == null) return error.SkipZigTest;

        var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
        defer t.deinit(alloc);
        try testTransmitImage(alloc, &t, 1);

        const cmd = try command.Parser.parseString(
            alloc,
            "a=f,i=1,f=100;iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAADklEQVR4nGP4" ++
                "DwUMMAYAj4IP8TylVlEAAAAASUVORK5CYII=",
        );
        defer cmd.deinit(alloc);
        try testing.expect(execute(alloc, &t, &cmd).?.ok());
        try testing.expectEqualSlices(
            u8,
            &white,
            t.screens.active.kitty_images.images.getPtr(1).?.anim.?.frames.items[0].data,
        );
    }
}

/// Drive a command string and drop the response, for tests that only care
/// about the resulting state.
fn testExec(alloc: Allocator, t: *Terminal, str: []const u8) !void {
    const cmd = try command.Parser.parseString(alloc, str);
    defer cmd.deinit(alloc);
    _ = execute(alloc, t, &cmd);
}

test "kittygfx active load swallows interleaved transmissions" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);
    try testTransmitImage(alloc, &t, 1);

    const storage = &t.screens.active.kitty_images;

    // While a direct-medium load is in progress, Kitty routes every
    // further direct transmission into it and ignores the identifiers,
    // dimensions and action those commands carry: an "a=f" here resolves
    // its target from the active load, not from its own "i". So this is
    // one four-chunk load of image 2, not a frame of image 1.
    try testExec(alloc, &t, "a=t,f=32,t=d,i=2,s=2,v=2,m=1;AAAA");
    try testing.expect(storage.loading != null);
    try testing.expect(storage.loading.?.frame == null);

    try testExec(alloc, &t, "a=f,i=1,f=32,s=99,v=99,m=1;AAAA");
    try testing.expect(storage.loading.?.frame == null);
    try testing.expectEqual(@as(u32, 2), storage.loading.?.image.id);

    try testExec(alloc, &t, "a=T,i=7,m=1;AAAA");
    try testing.expectEqual(@as(u32, 2), storage.loading.?.image.id);

    try testExec(alloc, &t, "m=0;AAAAAAAAAA==");
    try testing.expect(storage.loading == null);

    // The completed image is the one the load started as: 2x2 RGBA.
    const img = storage.images.getPtr(2).?;
    try testing.expectEqual(@as(u32, 2), img.width);
    try testing.expectEqual(@as(u32, 2), img.height);
    try testing.expect(img.anim == null);

    // Image 1 was never touched by the interleaved "a=f".
    try testing.expect(storage.images.getPtr(1).?.anim == null);
}

test "kittygfx starting a new load abandons the active one" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);

    const storage = &t.screens.active.kitty_images;

    // A non-direct medium never continues a load, so it starts a new one
    // and the abandoned load's buffered payload must be freed rather than
    // stranded. Repeated, stranding it is unbounded memory growth from a
    // malformed stream. The file load itself fails here (no such file),
    // which is fine: what matters is that the owner pointer is not leaked.
    for (0..16) |_| {
        try testExec(alloc, &t, "a=t,f=32,t=d,i=3,s=64,v=64,m=1;AAAAAAAAAAAAAAAA");
        try testing.expect(storage.loading != null);
        try testExec(alloc, &t, "a=t,f=32,t=f,i=4,s=1,v=1;L3RtcC9ub3BlLWdob3N0dHk=");
    }

    // Terminate the last load so the test's own teardown isn't what frees
    // it, then prove nothing is retained.
    try testExec(alloc, &t, "m=0;AAAA");
    try testing.expect(storage.loading == null);
}

test "kittygfx load rejected before starting keeps the active load" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);

    const storage = &t.screens.active.kitty_images;
    try testExec(alloc, &t, "a=t,f=32,t=d,i=5,s=2,v=2,m=1;AAAA");
    try testing.expect(storage.loading != null);

    // A command that fails validation before its load starts must leave
    // the in-progress load usable, not destroy it. An unsupported medium
    // is rejected inside LoadingImage.init.
    try testExec(alloc, &t, "a=t,f=32,t=s,i=6,s=1,v=1;L25vcGU=");
    try testing.expect(storage.loading != null);
    try testing.expectEqual(@as(u32, 5), storage.loading.?.image.id);

    try testExec(alloc, &t, "m=0;AAAAAAAAAAAAAAAAAA==");
    try testing.expect(storage.loading == null);
    try testing.expect(storage.images.getPtr(5) != null);
}

test "kittygfx interrupted loads retain nothing" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var t = try Terminal.init(alloc, .{ .rows = 5, .cols = 5 });
    defer t.deinit(alloc);
    try testTransmitImage(alloc, &t, 1);

    const storage = &t.screens.active.kitty_images;
    const bytes_before = storage.total_bytes;

    // Every way a load can end badly, repeated: the testing allocator
    // fails the test if any of them retains an allocation.
    for (0..8) |_| {
        // Decode failure: too little data for the declared frame.
        try testExec(alloc, &t, "a=f,i=1,f=32,s=2,v=2,m=1;AAAA");
        try testExec(alloc, &t, "a=f,m=0;");
        try testing.expect(storage.loading == null);

        // Frame load left dangling, then abandoned by a fresh load.
        try testExec(alloc, &t, "a=f,i=1,f=32,s=2,v=2,m=1;AAAA");
        try testExec(alloc, &t, "a=t,f=32,t=f,i=9,s=1,v=1;L3RtcC9ub3Bl");
        try testExec(alloc, &t, "m=0;AAAA");
        try testing.expect(storage.loading == null);

        // Oversized payload for the declared image: completes with the
        // required prefix for a frame, so terminate it cleanly.
        try testExec(alloc, &t, "a=f,i=1,f=32,s=1,v=1,m=1;/wAA/xERERE=");
        try testExec(alloc, &t, "a=f,m=0;");
        try testing.expect(storage.loading == null);
    }

    // The interrupted loads stored one frame per successful oversized
    // case; nothing else grew, and no load is retained.
    try testing.expect(storage.total_bytes >= bytes_before);
}
