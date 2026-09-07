//! Explicit user-triggered clipboard writes. No terminal escape sequences or
//! shell interpolation: a successful action means the OS/helper accepted text.
use std::io::Read;
use std::path::Path;

pub const MAX_BYTES: usize = 16 * 1024 * 1024;

pub fn read_log(path: &Path) -> Result<String, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("Cannot read saved log: {e}. Press Home for its path."))?;
    let mut bytes = Vec::new();
    file.take(MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("Cannot read saved log: {e}. Press Home for its path."))?;
    if bytes.len() > MAX_BYTES {
        return Err(
            "Saved log exceeds 16 MiB. Copy details, or open the saved file (Home shows its path)."
                .into(),
        );
    }
    String::from_utf8(bytes).map_err(|_| {
        "Saved log is not UTF-8 text. Open the saved file; Home shows its path.".into()
    })
}

pub fn copy(text: &str) -> Result<(), String> {
    if text.len() > MAX_BYTES {
        return Err("Text exceeds the 16 MiB clipboard limit. Open the saved log instead.".into());
    }
    if text.contains('\0') {
        return Err("Text contains a NUL character. Open the saved log instead.".into());
    }
    platform_copy(text)
}

#[cfg(windows)]
fn platform_copy(text: &str) -> Result<(), String> {
    use std::{io, ptr, thread, time::Duration};
    use windows_sys::Win32::{
        Foundation::{GlobalFree, HGLOBAL, HWND},
        System::{
            DataExchange::{CloseClipboard, EmptyClipboard, OpenClipboard, SetClipboardData},
            Memory::{GlobalAlloc, GlobalLock, GlobalUnlock, GMEM_MOVEABLE},
        },
        UI::WindowsAndMessaging::{CreateWindowExW, DestroyWindow, HWND_MESSAGE},
    };
    struct Owner(HWND);
    impl Drop for Owner {
        fn drop(&mut self) {
            unsafe {
                DestroyWindow(self.0);
            }
        }
    }
    struct Memory(HGLOBAL);
    impl Drop for Memory {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe {
                    GlobalFree(self.0);
                }
            }
        }
    }
    struct Clipboard;
    impl Drop for Clipboard {
        fn drop(&mut self) {
            unsafe {
                CloseClipboard();
            }
        }
    }
    let error = |step| {
        format!(
            "{step}: {}. Retry Copy or open the saved log (Home).",
            io::Error::last_os_error()
        )
    };
    let wide: Vec<u16> = text.encode_utf16().chain(Some(0)).collect();
    // A message-only STATIC window supplies a valid clipboard owner even in
    // ConPTY or without a console HWND. It never appears on the desktop.
    let owner = Owner(unsafe {
        CreateWindowExW(
            0,
            windows_sys::core::w!("STATIC"),
            ptr::null(),
            0,
            0,
            0,
            0,
            0,
            HWND_MESSAGE,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null(),
        )
    });
    if owner.0.is_null() {
        return Err(error("Cannot create clipboard owner"));
    }
    let mut memory = Memory(unsafe { GlobalAlloc(GMEM_MOVEABLE, wide.len() * 2) });
    if memory.0.is_null() {
        return Err(error("Cannot allocate clipboard text"));
    }
    let data = unsafe { GlobalLock(memory.0) };
    if data.is_null() {
        return Err(error("Cannot prepare clipboard text"));
    }
    unsafe {
        ptr::copy_nonoverlapping(wide.as_ptr(), data.cast::<u16>(), wide.len());
        GlobalUnlock(memory.0);
    }
    // Another app can briefly own the clipboard. Bound retries to 500 ms.
    let mut opened = false;
    for attempt in 0..11 {
        if unsafe { OpenClipboard(owner.0) } != 0 {
            opened = true;
            break;
        }
        if attempt < 10 {
            thread::sleep(Duration::from_millis(50));
        }
    }
    if !opened {
        return Err(error("Cannot open Windows clipboard"));
    }
    let _clipboard = Clipboard;
    if unsafe { EmptyClipboard() } == 0 {
        return Err(error("Cannot clear Windows clipboard"));
    }
    // CF_UNICODETEXT. On success Windows owns the movable allocation.
    if unsafe { SetClipboardData(13, memory.0) }.is_null() {
        return Err(error("Cannot set Windows clipboard"));
    }
    memory.0 = ptr::null_mut();
    Ok(())
}

#[cfg(unix)]
fn run_helper(
    command: &mut std::process::Command,
    text: &str,
    timeout: std::time::Duration,
) -> std::io::Result<()> {
    use std::{
        io::{self, Write},
        os::fd::AsRawFd,
        process::Stdio,
        thread,
        time::{Duration, Instant},
    };
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    let result = (|| {
        let mut input = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("clipboard input unavailable"))?;
        // A helper which never reads stdin must not freeze the UI with a full pipe.
        let flags = unsafe { libc::fcntl(input.as_raw_fd(), libc::F_GETFL) };
        if flags == -1
            || unsafe { libc::fcntl(input.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) }
                == -1
        {
            return Err(io::Error::last_os_error());
        }
        let deadline = Instant::now() + timeout;
        let mut remaining = text.as_bytes();
        while !remaining.is_empty() {
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "clipboard helper timed out",
                ));
            }
            match input.write(remaining) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "clipboard input closed",
                    ))
                }
                Ok(count) => remaining = &remaining[count..],
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(10))
                }
                Err(error) => return Err(error),
            }
        }
        drop(input);
        loop {
            if let Some(status) = child.try_wait()? {
                return if status.success() {
                    Ok(())
                } else {
                    Err(io::Error::other(format!(
                        "clipboard helper exited {status}"
                    )))
                };
            }
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "clipboard helper timed out",
                ));
            }
            thread::sleep(Duration::from_millis(10));
        }
    })();
    if result.is_err() {
        let _ = child.kill();
        let _ = child.wait();
    }
    result
}

#[cfg(unix)]
fn platform_copy(text: &str) -> Result<(), String> {
    use std::{env, process::Command, time::Duration};
    let mut candidates: Vec<(&str, &[&str])> = Vec::new();
    if cfg!(target_os = "macos") {
        candidates.push(("pbcopy", &[]));
    }
    if env::var_os("WAYLAND_DISPLAY").is_some_and(|v| !v.is_empty()) {
        candidates.push(("wl-copy", &["--type", "text/plain;charset=utf-8"]));
    }
    if env::var_os("DISPLAY").is_some_and(|v| !v.is_empty()) {
        candidates.push(("xclip", &["-selection", "clipboard", "-in"]));
        candidates.push(("xsel", &["--clipboard", "--input"]));
    }
    for (program, args) in candidates {
        if run_helper(
            Command::new(program).args(args),
            text,
            Duration::from_secs(2),
        )
        .is_ok()
        {
            return Ok(());
        }
    }
    Err("Clipboard unavailable. Use wl-copy (Wayland) or xclip/xsel (X11) in a desktop session, or open the saved log (Home shows its path).".into())
}

#[cfg(not(any(unix, windows)))]
fn platform_copy(_: &str) -> Result<(), String> {
    Err("Clipboard unavailable on this platform. Open the saved log; Home shows its path.".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_rejects_text_that_could_be_silently_truncated() {
        assert!(copy("before\0after").unwrap_err().contains("NUL"));
        assert!(copy(&"x".repeat(MAX_BYTES + 1))
            .unwrap_err()
            .contains("16 MiB"));
    }

    #[test]
    fn full_log_read_preserves_unicode_and_refuses_missing_or_oversized_files() {
        let path = std::env::temp_dir().join(format!(
            "arwen-clipboard-read-{}-{:?}.log",
            std::process::id(),
            std::thread::current().id()
        ));
        assert!(read_log(&path).is_err());
        std::fs::write(&path, "First line\ncafé 界 🌦\nlast line\n").unwrap();
        assert_eq!(
            read_log(&path).unwrap(),
            "First line\ncafé 界 🌦\nlast line\n"
        );
        std::fs::File::create(&path)
            .unwrap()
            .set_len(MAX_BYTES as u64 + 1)
            .unwrap();
        assert!(read_log(&path).unwrap_err().contains("16 MiB"));
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn helper_handles_unicode_nonzero_exit_and_a_blocked_input_pipe() {
        use std::{
            process::Command,
            time::{Duration, Instant},
        };
        let path =
            std::env::temp_dir().join(format!("arwen-clipboard-helper-{}.txt", std::process::id()));
        let fixture = "café 界 🌦\n$(literal) 'quoted' `unchanged`\n";
        run_helper(
            Command::new("sh")
                .args(["-c", "cat > \"$1\"", "clipboard-test"])
                .arg(&path),
            fixture,
            Duration::from_secs(2),
        )
        .unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), fixture);
        std::fs::remove_file(path).unwrap();
        assert!(run_helper(
            Command::new("sh").args(["-c", "exit 3"]),
            "text",
            Duration::from_secs(2)
        )
        .is_err());
        let start = Instant::now();
        let error = run_helper(
            Command::new("sleep").arg("10"),
            &"x".repeat(1024 * 1024),
            Duration::from_millis(100),
        )
        .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
        assert!(start.elapsed() < Duration::from_secs(2));
    }
}
