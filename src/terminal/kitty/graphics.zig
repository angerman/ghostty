//! Kitty graphics protocol support.
//!
//! Documentation:
//! https://sw.kovidgoyal.net/kitty/graphics-protocol
//!
//! Unimplemented features that are still todo:
//! - shared memory transmit
//! - virtual placement w/ unicode
//!
//! Animation:
//! Animation is implemented, but stores frames differently from Kitty.
//! Every frame is a fully-composed, full-canvas RGBA buffer, where Kitty
//! keeps partial frames that reference a base frame and coalesces them on
//! demand. Kitty can afford that because it also has a disk cache to spill
//! frames to; we don't, so we trade the memory for a much simpler
//! implementation. For the same reason frames are charged against the same
//! per-screen RAM quota as images rather than Kitty's separate 5x disk
//! quota, and a large animation is more likely to evict images here than
//! it would there.
//!
//! Playback advances from the renderer thread, which is the only thread
//! that progresses without client input. An animation is considered
//! playable if it has any placement at all, which only approximates
//! Kitty's renderer-tracked "is drawn": an image whose sole placement has
//! scrolled into the scrollback keeps ticking.
//!
//! Performance:
//! The performance of this particular subsystem of Ghostty is not great.
//! We can avoid a lot more allocations, we can replace some C code (which
//! implicitly allocates) with native Zig, we can improve the data structures
//! to avoid repeated lookups, etc. I tried to avoid pessimization but my
//! aim to ship a v1 of this implementation came at some cost. I learned a lot
//! though and I think we can go back through and fix this up.

const animation = @import("graphics_animation.zig");
const render = @import("graphics_render.zig");
const command = @import("graphics_command.zig");
const exec = @import("graphics_exec.zig");
const image = @import("graphics_image.zig");
const storage = @import("graphics_storage.zig");
pub const unicode = @import("graphics_unicode.zig");
pub const Animation = animation.Animation;
pub const Damage = animation.Damage;
pub const Command = command.Command;
pub const CommandParser = command.Parser;
pub const Image = image.Image;
pub const LoadingImage = image.LoadingImage;
pub const ImageStorage = storage.ImageStorage;
pub const RenderPlacement = render.Placement;
pub const Response = command.Response;
pub const nextGeneration = storage.nextGeneration;

pub const execute = exec.execute;

test {
    @import("std").testing.refAllDecls(@This());
}
