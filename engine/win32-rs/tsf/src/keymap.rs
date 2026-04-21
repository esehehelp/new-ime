//! Virtual-key → engine key classification.

use crate::engine_bridge::KeyKind;

/// Windows virtual-key codes we care about. The rest fall through to the host.
pub fn classify_vk(vk: u32, shift: bool) -> Option<KeyKind> {
    match vk {
        0x08 => Some(KeyKind::Backspace),                 // VK_BACK
        0x0D => Some(KeyKind::Enter),                     // VK_RETURN
        0x1B => Some(KeyKind::Escape),                    // VK_ESCAPE
        0x20 => Some(KeyKind::Space),                     // VK_SPACE
        0x21 => Some(KeyKind::PageUp),                    // VK_PRIOR
        0x22 => Some(KeyKind::PageDown),                  // VK_NEXT
        0x25 => Some(KeyKind::Left),                      // VK_LEFT
        0x26 => Some(KeyKind::Up),                        // VK_UP
        0x27 => Some(KeyKind::Right),                     // VK_RIGHT
        0x28 => Some(KeyKind::Down),                      // VK_DOWN
        0xBC => Some(KeyKind::Comma),                     // VK_OEM_COMMA
        0xBE => Some(KeyKind::Period),                    // VK_OEM_PERIOD
        // Shift+1 → `!` (full-width `！` handled in engine_bridge).
        0x31 if shift => Some(KeyKind::Exclaim),
        // Shift+`/` → `?` (full-width `？`).
        0xBF if shift => Some(KeyKind::Question),
        0x30..=0x39 => Some(KeyKind::Char((vk as u8) as char)), // '0'..='9'
        0x41..=0x5A => {
            // 'A'..='Z'
            let base = b'a' + (vk as u8 - 0x41);
            let ch = if shift {
                (base as char).to_ascii_uppercase()
            } else {
                base as char
            };
            Some(KeyKind::Char(ch))
        }
        0xBD => Some(KeyKind::Char('-')), // VK_OEM_MINUS
        _ => None,
    }
}
