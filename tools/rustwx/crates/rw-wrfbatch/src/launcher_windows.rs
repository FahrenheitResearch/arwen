//! Assign a suspended child to a non-inheritable kill-on-close Job before
//! running any renderer code. Closing or killing this launcher ends its tree.
use std::ffi::c_void;
use std::io;
use std::os::windows::ffi::OsStrExt;
use std::path::Path;
use std::ptr::{null, null_mut};

type Handle = *mut c_void;
#[repr(C)]
#[derive(Default)]
struct BasicLimits {
    process_time: i64, job_time: i64, flags: u32,
    min_working_set: usize, max_working_set: usize, active_processes: u32,
    affinity: usize, priority: u32, scheduling: u32,
}
#[repr(C)]
#[derive(Default)]
struct IoCounters { read_ops: u64, write_ops: u64, other_ops: u64,
    read_bytes: u64, write_bytes: u64, other_bytes: u64 }
#[repr(C)]
#[derive(Default)]
struct ExtendedLimits { basic: BasicLimits, io: IoCounters,
    process_memory: usize, job_memory: usize, peak_process_memory: usize, peak_job_memory: usize }
#[repr(C)]
struct StartupInfo {
    size: u32, reserved: *mut u16, desktop: *mut u16, title: *mut u16,
    x: u32, y: u32, x_size: u32, y_size: u32, x_chars: u32, y_chars: u32,
    fill: u32, flags: u32, show: u16, reserved_size: u16, reserved_bytes: *mut u8,
    stdin: Handle, stdout: Handle, stderr: Handle,
}
#[repr(C)]
struct ProcessInfo { process: Handle, thread: Handle, process_id: u32, thread_id: u32 }
#[link(name = "kernel32")]
unsafe extern "system" {
    fn CreateJobObjectW(attributes: *const c_void, name: *const u16) -> Handle;
    fn SetInformationJobObject(job: Handle, class: u32, info: *const c_void, size: u32) -> i32;
    fn AssignProcessToJobObject(job: Handle, process: Handle) -> i32;
    fn CreateProcessW(application: *const u16, command: *mut u16,
        process_attributes: *const c_void, thread_attributes: *const c_void,
        inherit: i32, flags: u32, environment: *const c_void, directory: *const u16,
        startup: *const StartupInfo, process: *mut ProcessInfo) -> i32;
    fn GetCommandLineW() -> *const u16;
    fn GetStartupInfoW(startup: *mut StartupInfo);
    fn GetStdHandle(which: u32) -> Handle;
    fn ResumeThread(thread: Handle) -> u32;
    fn WaitForSingleObject(handle: Handle, milliseconds: u32) -> u32;
    fn GetExitCodeProcess(process: Handle, code: *mut u32) -> i32;
    fn TerminateProcess(process: Handle, code: u32) -> i32;
    fn CloseHandle(handle: Handle) -> i32;
    fn ExitProcess(code: u32) -> !;
}
struct OwnedHandle(Handle);
impl Drop for OwnedHandle {
    fn drop(&mut self) { unsafe { CloseHandle(self.0); } }
}

pub fn launch(executable: &Path) -> io::Result<()> {
    // SAFETY: repr(C) structures follow the Windows SDK definitions. Buffers
    // remain alive through CreateProcessW, and handles are owned exactly once.
    unsafe {
        let job = OwnedHandle(CreateJobObjectW(null(), null()));
        if job.0.is_null() { return Err(io::Error::last_os_error()); }
        let mut limits = ExtendedLimits::default();
        limits.basic.flags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if SetInformationJobObject(job.0, 9, &limits as *const _ as *const c_void,
            std::mem::size_of::<ExtendedLimits>() as u32) == 0 {
            return Err(io::Error::last_os_error());
        }
        let application: Vec<u16> = executable.as_os_str().encode_wide().chain(Some(0)).collect();
        let raw = GetCommandLineW();
        let mut length = 0;
        while *raw.add(length) != 0 { length += 1; }
        // Preserve Windows quoting, empty arguments and argv[0] byte for byte.
        let mut command = std::slice::from_raw_parts(raw, length + 1).to_vec();
        let mut startup: StartupInfo = std::mem::zeroed();
        startup.size = std::mem::size_of::<StartupInfo>() as u32;
        GetStartupInfoW(&mut startup);
        startup.flags |= 0x100; // STARTF_USESTDHANDLES
        startup.stdin = GetStdHandle(-10i32 as u32);
        startup.stdout = GetStdHandle(-11i32 as u32);
        startup.stderr = GetStdHandle(-12i32 as u32);
        let mut child: ProcessInfo = std::mem::zeroed();
        if CreateProcessW(application.as_ptr(), command.as_mut_ptr(), null(), null(),
            1, 4, null(), null(), &startup, &mut child) == 0 { // CREATE_SUSPENDED
            return Err(io::Error::last_os_error());
        }
        let process = OwnedHandle(child.process);
        let thread = OwnedHandle(child.thread);
        if AssignProcessToJobObject(job.0, process.0) == 0 {
            let error = io::Error::last_os_error();
            TerminateProcess(process.0, 1);
            WaitForSingleObject(process.0, u32::MAX);
            return Err(error);
        }
        if ResumeThread(thread.0) == u32::MAX { return Err(io::Error::last_os_error()); }
        if WaitForSingleObject(process.0, u32::MAX) != 0 { return Err(io::Error::last_os_error()); }
        let mut code = 0;
        if GetExitCodeProcess(process.0, &mut code) == 0 { return Err(io::Error::last_os_error()); }
        drop(thread);
        drop(process);
        drop(job);
        ExitProcess(code)
    }
}
