//! The terminal the user is left holding, proven on a real pty rather than a
//! mock: a crash and a signal both have to hand the shell back a terminal that
//! is out of the alternate screen, out of raw mode and no longer reporting
//! mouse motion. Without that last one a user's next mouse move becomes a
//! stream of `ESC[<35;..M` "command not found" lines in their shell, which is
//! exactly how this defect was reported.
//!
//! Debug builds only: the forced-crash probe the first case needs is compiled
//! out of a release build on purpose, so there is nothing here to test then.
#![cfg(all(unix, debug_assertions))]

use std::fs;
use std::io::Read;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const ENTER_ALTERNATE: &str = "\x1b[?1049h";
const LEAVE_ALTERNATE: &str = "\x1b[?1049l";
const ENABLE_ANY_MOTION: &str = "\x1b[?1003h";
const DISABLE_SGR_MOUSE: &str = "\x1b[?1006l";
const DISABLE_ANY_MOTION: &str = "\x1b[?1003l";
const DISABLE_NORMAL_MOUSE: &str = "\x1b[?1000l";
const DISABLE_BRACKETED_PASTE: &str = "\x1b[?2004l";
const SHOW_CURSOR: &str = "\x1b[?25h";

struct Pty {
    master: OwnedFd,
    slave: OwnedFd,
}

fn pty(columns: u16, rows: u16) -> Pty {
    let (mut master, mut slave) = (-1, -1);
    let size = libc::winsize {
        ws_row: rows,
        ws_col: columns,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    let opened = unsafe {
        libc::openpty(
            &mut master,
            &mut slave,
            std::ptr::null_mut(),
            std::ptr::null(),
            &size,
        )
    };
    assert_eq!(opened, 0, "openpty: {}", std::io::Error::last_os_error());
    // Non-blocking: the pty has to be drained while the child runs, and after
    // the child exits a blocking read on the master would never return, since
    // this process keeps a slave open for the terminal-mode assertions.
    unsafe { libc::fcntl(master, libc::F_SETFL, libc::O_NONBLOCK) };
    unsafe {
        Pty {
            master: OwnedFd::from_raw_fd(master),
            slave: OwnedFd::from_raw_fd(slave),
        }
    }
}

fn scratch(label: &str) -> PathBuf {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!(
        "arwen-tui-terminal-{label}-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir(&path).unwrap();
    path
}

/// Start the real executable with the pty as its terminal. `setsid` plus
/// `TIOCSCTTY` make it the session leader of that pty, so `enable_raw_mode`'s
/// `tcsetattr` is a foreground call and never raises SIGTTOU.
fn start(terminal: &Pty, output: &PathBuf, stderr: &PathBuf, probe: bool) -> Child {
    let slave = terminal.slave.try_clone().unwrap();
    let (input, screen) = (slave.try_clone().unwrap(), slave);
    let mut command = Command::new(env!("CARGO_BIN_EXE_arwen-tui"));
    command
        .arg("--output")
        .arg(output)
        .current_dir(output)
        .env_remove("GPUWM_TUI_PANIC_PROBE")
        .stdin(Stdio::from(input))
        .stdout(Stdio::from(screen))
        .stderr(Stdio::from(fs::File::create(stderr).unwrap()));
    if probe {
        command.env("GPUWM_TUI_PANIC_PROBE", "1");
    }
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::ioctl(0, libc::TIOCSCTTY, 0) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    command.spawn().unwrap()
}

fn drain(terminal: &Pty, captured: &mut Vec<u8>) {
    let mut buffer = [0_u8; 8192];
    let mut master = unsafe { std::fs::File::from_raw_fd(terminal.master.as_raw_fd()) };
    loop {
        match master.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => captured.extend_from_slice(&buffer[..count]),
            Err(_) => break,
        }
    }
    std::mem::forget(master);
}

/// Run until `ready` is satisfied, then `act`, then until the child exits.
fn run(
    terminal: &Pty,
    child: &mut Child,
    ready: impl Fn(&str) -> bool,
    act: impl FnOnce(&mut Child),
) -> (String, std::process::ExitStatus) {
    let mut captured = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut act = Some(act);
    loop {
        drain(terminal, &mut captured);
        if act.is_some() && ready(&String::from_utf8_lossy(&captured)) {
            act.take().unwrap()(child);
        }
        if let Some(status) = child.try_wait().unwrap() {
            std::thread::sleep(Duration::from_millis(100));
            drain(terminal, &mut captured);
            assert!(act.is_none(), "the session never reached its ready state");
            return (String::from_utf8_lossy(&captured).into_owned(), status);
        }
        assert!(
            Instant::now() < deadline,
            "the terminal session never ended; captured {} bytes",
            captured.len()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn assert_restored(captured: &str, what: &str) {
    let entered = captured.rfind(ENTER_ALTERNATE).unwrap_or_else(|| {
        panic!("{what}: the session never entered the alternate screen")
    });
    let captured_mouse = captured
        .rfind(ENABLE_ANY_MOTION)
        .unwrap_or_else(|| panic!("{what}: the session never enabled mouse capture"));
    for (label, sequence) in [
        ("SGR mouse reporting", DISABLE_SGR_MOUSE),
        ("any-motion mouse reporting", DISABLE_ANY_MOTION),
        ("normal mouse reporting", DISABLE_NORMAL_MOUSE),
        ("bracketed paste", DISABLE_BRACKETED_PASTE),
        ("the alternate screen", LEAVE_ALTERNATE),
        ("the hidden cursor", SHOW_CURSOR),
    ] {
        let at = captured.rfind(sequence).unwrap_or_else(|| {
            panic!("{what}: {label} was never turned off; the shell inherits it")
        });
        assert!(
            at > entered && at > captured_mouse,
            "{what}: {label} was turned off before the session turned it on"
        );
    }
}

fn assert_terminal_is_usable(terminal: &Pty, what: &str) {
    let mut mode: libc::termios = unsafe { std::mem::zeroed() };
    assert_eq!(
        unsafe { libc::tcgetattr(terminal.slave.as_raw_fd(), &mut mode) },
        0,
        "{what}: tcgetattr"
    );
    for (label, flag) in [
        ("canonical input", libc::ICANON),
        ("echo", libc::ECHO),
        ("signal keys", libc::ISIG),
    ] {
        assert!(
            mode.c_lflag & flag != 0,
            "{what}: raw mode left {label} off, so the shell reads keys one byte at a time"
        );
    }
}

#[test]
fn a_crash_hands_back_a_terminal_that_has_stopped_reporting_the_mouse() {
    let terminal = pty(120, 40);
    let output = scratch("crash");
    let stderr = output.join("stderr.txt");
    let mut child = start(&terminal, &output, &stderr, true);
    let (captured, status) = run(
        &terminal,
        &mut child,
        |seen| seen.contains(ENTER_ALTERNATE),
        |_| {},
    );
    assert_eq!(status.code(), Some(101), "a panic exits 101: {status:?}");
    assert_restored(&captured, "crash");
    assert_terminal_is_usable(&terminal, "crash");
    let reported = fs::read_to_string(&stderr).unwrap();
    assert!(
        reported.contains("GPUWM_TUI_PANIC_PROBE: forced crash"),
        "the panic message never reached stderr: {reported}"
    );
    let log = fs::read_to_string(output.join(".arwen-tui").join("controller.log")).unwrap();
    assert!(
        log.contains("The ArWen terminal stopped unexpectedly:")
            && log.contains("GPUWM_TUI_PANIC_PROBE: forced crash"),
        "controller.log kept no readable record: {log}"
    );
    fs::remove_dir_all(&output).unwrap();
}

#[test]
fn a_termination_signal_hands_back_a_terminal_that_has_stopped_reporting_the_mouse() {
    let terminal = pty(120, 40);
    let output = scratch("signal");
    let stderr = output.join("stderr.txt");
    let mut child = start(&terminal, &output, &stderr, false);
    let (captured, status) = run(
        &terminal,
        &mut child,
        |seen| seen.contains(ENTER_ALTERNATE) && seen.contains(ENABLE_ANY_MOTION),
        |child| {
            assert_eq!(
                unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) },
                0
            );
        },
    );
    assert_eq!(
        status.signal(),
        Some(libc::SIGTERM),
        "the handler must re-raise, so the exit status stays the signal: {status:?}"
    );
    assert_restored(&captured, "SIGTERM");
    assert_terminal_is_usable(&terminal, "SIGTERM");
    fs::remove_dir_all(&output).unwrap();
}
