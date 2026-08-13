//! Advisory lines a render emits about itself, and where they go.
//!
//! A render says things on stderr that are neither a failure nor part of
//! the product: a variable that fell back to the generic ramp, a
//! subtitle that did not fit.  Written straight to stderr from inside a
//! worker pool they arrive in completion order, so the same gallery
//! prints the same lines in a different sequence on every run while
//! stdout -- reported in catalog order on the caller's thread -- stays
//! byte-stable.  One stream reproducible and the other not is worse than
//! either: a reader cannot tell a reordered console from a changed one.
//!
//! So a caller that reports in a fixed order installs a [`hold`] around
//! each unit of work, collects what that unit had to say, and prints it
//! where that unit belongs in the order.  With no hold installed -- every
//! single-plot caller -- a line goes to stderr immediately, exactly as
//! before.
//!
//! [`advise_once`] carries the "say this once, however many plots hit
//! it" rule that a spammy advisory needs.  Under a hold the dedup CANNOT
//! live at the emitting site: which worker arrives first is precisely
//! what is not fixed, so the surviving copy would attach to an arbitrary
//! product.  The key travels with the line instead and the drain, which
//! runs in catalog order, decides -- so the first plot in CATALOG order
//! keeps it, which is what the serial loop did.

use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

/// One advisory line, plus the key that makes it say-once, if it has one.
#[derive(Debug, Clone)]
pub struct Advisory {
    /// `Some(key)`: the drain prints the first line per key and drops the
    /// rest.  `None`: every occurrence prints.
    pub once_key: Option<String>,
    /// The line, without its trailing newline.
    pub line: String,
}

thread_local! {
    static HELD: RefCell<Option<Vec<Advisory>>> = const { RefCell::new(None) };
}

/// Emit an advisory line every time it happens.
pub fn advise(line: String) {
    route(Advisory { once_key: None, line });
}

/// Emit an advisory line at most once per `key`.
pub fn advise_once(key: impl Into<String>, line: String) {
    route(Advisory { once_key: Some(key.into()), line });
}

fn route(advisory: Advisory) {
    // Move it into the hold if there is one; get it back if there is not.
    let escaped = HELD.with(|slot| match slot.borrow_mut().as_mut() {
        Some(held) => {
            held.push(advisory);
            None
        }
        None => Some(advisory),
    });
    let Some(advisory) = escaped else { return };
    if let Some(key) = advisory.once_key {
        static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        if let Ok(mut seen) = seen.lock() {
            if !seen.insert(key) {
                return;
            }
        }
    }
    eprintln!("{}", advisory.line);
}

/// A held drain: what `body` produced, and everything it advised, in the
/// order it advised it.
///
/// Holds nest: an inner hold takes the lines an outer one would have
/// received, and the outer hold is restored intact afterwards.  If `body`
/// unwinds, its advice is printed rather than dropped -- a caller that
/// catches the panic never reaches the drain, and a lost advisory is
/// exactly the silence this module exists to prevent.
pub fn hold<R>(body: impl FnOnce() -> R) -> (R, Vec<Advisory>) {
    struct Restore(Option<Vec<Advisory>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let abandoned = HELD.with(|slot| {
                std::mem::replace(&mut *slot.borrow_mut(), self.0.take())
            });
            for advisory in abandoned.into_iter().flatten() {
                eprintln!("{}", advisory.line);
            }
        }
    }
    let restore = Restore(HELD.with(|slot| slot.borrow_mut().replace(Vec::new())));
    let produced = body();
    // Take the advice before the guard runs, so the guard's own drain
    // sees an empty hold on the path where nothing went wrong.
    let held = HELD
        .with(|slot| slot.borrow_mut().replace(Vec::new()))
        .unwrap_or_default();
    drop(restore);
    (produced, held)
}

/// Print one held advisory unless `seen` already carries its key.
///
/// `seen` belongs to the caller and spans whatever the caller considers
/// one run, so a say-once line is said once across the whole of it.
pub fn drain_one(advisory: Advisory, seen: &mut HashSet<String>) {
    if let Some(key) = advisory.once_key {
        if !seen.insert(key) {
            return;
        }
    }
    eprintln!("{}", advisory.line);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hold_collects_in_emission_order_and_restores_the_outer_one() {
        let (inner, held) = hold(|| {
            advise("first".into());
            advise_once("k", "second".into());
            7
        });
        assert_eq!(inner, 7);
        assert_eq!(
            held.iter().map(|a| a.line.as_str()).collect::<Vec<_>>(),
            ["first", "second"]
        );
        assert_eq!(held[0].once_key, None);
        assert_eq!(held[1].once_key.as_deref(), Some("k"));
        // Nothing is held once the hold is over.
        HELD.with(|slot| assert!(slot.borrow().is_none()));
    }

    #[test]
    fn a_panicking_body_does_not_leave_the_hold_installed() {
        let caught = std::panic::catch_unwind(|| {
            hold(|| {
                advise("said before the panic".into());
                panic!("render blew up");
            })
        });
        assert!(caught.is_err());
        HELD.with(|slot| assert!(slot.borrow().is_none(), "hold leaked past a panic"));
    }

    #[test]
    fn the_drain_keeps_the_first_line_per_key_and_every_keyless_line() {
        let mut seen = HashSet::new();
        let mut kept = Vec::new();
        for advisory in [
            Advisory { once_key: Some("a".into()), line: "one".into() },
            Advisory { once_key: Some("a".into()), line: "one again".into() },
            Advisory { once_key: None, line: "always".into() },
            Advisory { once_key: None, line: "always".into() },
        ] {
            // Same predicate `drain_one` applies, without capturing stderr.
            let suppressed = advisory
                .once_key
                .as_ref()
                .is_some_and(|key| !seen.insert(key.clone()));
            if !suppressed {
                kept.push(advisory.line);
            }
        }
        assert_eq!(kept, ["one", "always", "always"]);
    }
}
