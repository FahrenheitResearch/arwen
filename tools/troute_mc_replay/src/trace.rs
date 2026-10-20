//! Observe an unmodified native x86_64 process without adding a target frame.
//! SPDX-License-Identifier: Apache-2.0
use std::ffi::{c_int, c_void};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::os::unix::process::CommandExt;
use std::process::Command;
use std::time::Instant;

extern "C" {
    fn ptrace(request: u32, pid: c_int, addr: *mut c_void, data: *mut c_void) -> isize;
    fn waitpid(pid: c_int, status: *mut c_int, options: c_int) -> c_int;
    fn __errno_location() -> *mut c_int;
}

#[repr(C)]
#[derive(Default)]
struct Regs {
    r15: u64,
    r14: u64,
    r13: u64,
    r12: u64,
    rbp: u64,
    rbx: u64,
    r11: u64,
    r10: u64,
    r9: u64,
    r8: u64,
    rax: u64,
    rcx: u64,
    rdx: u64,
    rsi: u64,
    rdi: u64,
    orig_rax: u64,
    rip: u64,
    cs: u64,
    eflags: u64,
    rsp: u64,
    ss: u64,
    fs_base: u64,
    gs_base: u64,
    ds: u64,
    es: u64,
    fs: u64,
    gs: u64,
}

fn request(pid: i32, code: u32, addr: usize, data: usize) -> Result<usize, String> {
    unsafe {
        *__errno_location() = 0;
        let value = ptrace(code, pid, addr as *mut c_void, data as *mut c_void);
        if value == -1 && *__errno_location() != 0 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        Ok(value as usize)
    }
}

fn wait(pid: i32) -> Result<i32, String> {
    let mut status = 0;
    if unsafe { waitpid(pid, &mut status, 0) } != pid {
        return Err(std::io::Error::last_os_error().to_string());
    }
    Ok(status)
}

fn stopped(status: i32) -> bool {
    status & 0xff == 0x7f
}
fn signal(status: i32) -> i32 {
    (status >> 8) & 0xff
}

fn regs(pid: i32) -> Result<Regs, String> {
    let mut regs = Regs::default();
    request(pid, 12, 0, &mut regs as *mut Regs as usize)?; // GETREGS
    Ok(regs)
}

fn set_regs(pid: i32, regs: &Regs) -> Result<(), String> {
    request(pid, 13, 0, regs as *const Regs as usize)?;
    Ok(())
}

fn xmm_words(pid: i32) -> Result<[u32; 2], String> {
    // Linux x86_64 user_fpregs_struct is the 512-byte FXSAVE layout.
    let mut fp = [0_u64; 64];
    request(pid, 14, 0, fp.as_mut_ptr() as usize)?; // GETFPREGS
    Ok([fp[20] as u32, fp[22] as u32]) // low float32 of XMM0/XMM1
}

fn breakpoint(pid: i32, address: usize) -> Result<usize, String> {
    let word = request(pid, 1, address, 0)?; // PEEKTEXT
    request(pid, 4, address, (word & !255) | 0xcc)?; // POKETEXT
    Ok(word)
}

fn step_over(pid: i32, mut regs: Regs, address: usize, word: usize) -> Result<(), String> {
    request(pid, 4, address, word)?;
    regs.rip = address as u64;
    set_regs(pid, &regs)?;
    request(pid, 9, 0, 0)?; // SINGLESTEP, exact original instruction
    let status = wait(pid)?;
    if !stopped(status) || signal(status) != 5 {
        return Err(format!(
            "native process did not single-step normally: status {status}"
        ));
    }
    Ok(())
}

fn read<const N: usize>(memory: &File, address: u64) -> Result<[u8; N], String> {
    let mut bytes = [0_u8; N];
    memory
        .read_exact_at(&mut bytes, address)
        .map_err(|e| e.to_string())?;
    Ok(bytes)
}

fn arguments(memory: &File, regs: &Regs) -> Result<[u64; 21], String> {
    let mut pointers = [0_u64; 21];
    pointers[..6].copy_from_slice(&[regs.rdi, regs.rsi, regs.rdx, regs.rcx, regs.r8, regs.r9]);
    for (i, pointer) in pointers[6..].iter_mut().enumerate() {
        *pointer = u64::from_le_bytes(read::<8>(memory, regs.rsp + 8 + 8 * i as u64)?);
    }
    Ok(pointers)
}

struct Witness {
    offset: usize,
    first_call: usize,
    calls: usize,
    limit: usize,
    records: usize,
    file: File,
}

struct PowWitness {
    offset: usize,
    call: usize,
    limit: usize,
    records: usize,
    file: File,
    pending: Option<(usize, usize, [u32; 2], usize)>,
}

impl PowWitness {
    fn from_env() -> Result<Option<Self>, String> {
        let Ok(offset) = std::env::var("MC_TRACE_POWF_OFFSET") else {
            return Ok(None);
        };
        let number = |name: &str| -> Result<usize, String> {
            std::env::var(name)
                .map_err(|_| format!("missing {name}"))?
                .parse()
                .map_err(|_| format!("invalid {name}"))
        };
        let limit = number("MC_TRACE_POWF_LIMIT")?;
        if limit == 0 {
            return Err("powf witness limit must be positive".into());
        }
        Ok(Some(Self {
            offset: usize::from_str_radix(offset.trim_start_matches("0x"), 16)
                .map_err(|e| e.to_string())?,
            call: number("MC_TRACE_POWF_CALL")?,
            limit,
            records: 0,
            file: OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(std::env::var_os("MC_TRACE_POWF_PATH").ok_or("missing powf witness path")?)
                .map_err(|e| e.to_string())?,
            pending: None,
        }))
    }

    fn active(&self, call: usize) -> bool {
        call == self.call && self.records < self.limit
    }
}

impl Witness {
    fn from_env() -> Result<Option<Self>, String> {
        let Some(offset) = std::env::var_os("MC_TRACE_SECANT_OFFSET") else {
            return Ok(None);
        };
        let number = |name: &str| -> Result<usize, String> {
            std::env::var(name)
                .map_err(|_| format!("missing {name}"))?
                .parse()
                .map_err(|_| format!("invalid {name}"))
        };
        let offset = usize::from_str_radix(
            offset
                .to_str()
                .ok_or("invalid secant offset")?
                .trim_start_matches("0x"),
            16,
        )
        .map_err(|e| e.to_string())?;
        let first_call = number("MC_TRACE_WITNESS_FIRST")?;
        let calls = number("MC_TRACE_WITNESS_CALLS")?;
        let limit = number("MC_TRACE_WITNESS_LIMIT")?;
        if calls == 0 || limit == 0 {
            return Err("witness calls/limit must be positive".into());
        }
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(std::env::var_os("MC_TRACE_WITNESS_PATH").ok_or("missing witness path")?)
            .map_err(|e| e.to_string())?;
        Ok(Some(Self {
            offset,
            first_call,
            calls,
            limit,
            records: 0,
            file,
        }))
    }

    fn active(&self, call: usize) -> bool {
        call >= self.first_call && call - self.first_call < self.calls && self.records < self.limit
    }

    fn observe(&mut self, call: usize, memory: &File, regs: &Regs) -> Result<(), String> {
        let pointers = arguments(memory, regs)?;
        let words: Vec<u32> = pointers
            .into_iter()
            .map(|p| read::<4>(memory, p).map(u32::from_le_bytes))
            .collect::<Result<_, _>>()?;
        let bits = words
            .iter()
            .map(|word| format!("\"0x{word:08x}\""))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(
            self.file,
            "{{\"call\":{call},\"interval\":{},\"argument_bits\":[{bits}]}}",
            words[14] as i32
        )
        .map_err(|e| e.to_string())?;
        self.records += 1;
        Ok(())
    }
}

fn entry_address(pid: i32, library: &str, symbol_offset: usize) -> Result<Option<usize>, String> {
    let maps = fs::read_to_string(format!("/proc/{pid}/maps")).map_err(|e| e.to_string())?;
    for line in maps.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() >= 6 && fields[1].starts_with("r-x") && fields[5] == library {
            let start = fields[0].split('-').next().unwrap();
            let start = usize::from_str_radix(start, 16).map_err(|e| e.to_string())?;
            let offset = usize::from_str_radix(fields[2], 16).map_err(|e| e.to_string())?;
            return Ok(Some(start - offset + symbol_offset));
        }
    }
    Ok(None)
}

pub fn run(args: &[String]) -> Result<bool, String> {
    if args.len() < 8 || args[6] != "--" {
        return Err("usage: troute-mc-replay trace NATIVE_SO HEX_SYMBOL_OFFSET NEW_CAPTURE LIMIT -- COMMAND [ARGS...]".into());
    }
    let library = fs::canonicalize(&args[2]).map_err(|e| e.to_string())?;
    let library = library.to_str().ok_or("native path is not UTF-8")?;
    let offset =
        usize::from_str_radix(args[3].trim_start_matches("0x"), 16).map_err(|e| e.to_string())?;
    let limit: usize = args[5]
        .parse()
        .map_err(|_| "capture limit must be an integer")?;
    if limit == 0 {
        return Err("capture limit must be positive".into());
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[4])
        .map_err(|e| e.to_string())?;
    let mut witness = Witness::from_env()?;
    let mut pow = PowWitness::from_env()?;
    let start = Instant::now();
    let mut command = Command::new(&args[7]);
    command.args(&args[8..]);
    // A traced child is the only target. Never attach to other fleet processes.
    unsafe {
        command.pre_exec(|| {
            if ptrace(0, 0, std::ptr::null_mut(), std::ptr::null_mut()) == -1 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        });
    }
    let mut child = command.spawn().map_err(|e| e.to_string())?;
    let pid = child.id() as i32;
    let result = (|| {
        let status = wait(pid)?; // exec trap
        if !stopped(status) {
            return Err("native child did not stop at exec".into());
        }
        // TRACESYSGOOD distinguishes startup syscalls; EXITKILL prevents a live
        // stopped child if the observer is killed.
        request(pid, 0x4200, 0, 1 | 0x0010_0000)?;
        let mut entry = None;
        let startup_stride: usize = std::env::var("MC_TRACE_STARTUP_STRIDE")
            .unwrap_or_else(|_| "16".into()).parse().map_err(|_| "invalid startup stride")?;
        if startup_stride == 0 { return Err("startup stride must be positive".into()); }
        // Detect dlopen's executable mapping at syscall boundaries, then stop
        // observing syscalls. Read maps only every 16 stops during startup.
        for index in 0..200_000 {
            request(pid, 24, 0, 0)?; // SYSCALL
            let status = wait(pid)?;
            if !stopped(status) {
                return Err(format!("native exited before loading MC: status {status}"));
            }
            if signal(status) != 133 && signal(status) != 5 {
                return Err(format!("unexpected startup signal {}", signal(status)));
            }
            if index % startup_stride == 0 {
                entry = entry_address(pid, library, offset)?;
                if entry.is_some() {
                    break;
                }
            }
        }
        let entry = entry.ok_or("native MC executable mapping never appeared")?;
        let entry_word = breakpoint(pid, entry)?;
        let secant_address = witness.as_ref().map(|w| entry - offset + w.offset);
        let mut secant_breakpoint: Option<usize> = None;
        let pow_address = pow.as_ref().map(|w| entry - offset + w.offset);
        let mut pow_breakpoint: Option<usize> = None;
        let memory = File::open(format!("/proc/{pid}/mem")).map_err(|e| e.to_string())?;
        let mut pending: Option<(usize, usize, [u64; 21], Vec<u8>)> = None;
        let mut count = 0;
        loop {
            request(pid, 7, 0, 0)?; // CONT
            let status = wait(pid)?;
            if !stopped(status) {
                if status != 0 {
                    return Err(format!("native child failed: status {status}"));
                }
                if pending.is_some() || count == 0 {
                    return Err("native ended without complete observed calls".into());
                }
                eprintln!("{{\"schema\":\"troute-ptrace-v1\",\"calls\":{count},\"limit\":{limit},\"seconds\":{},\"pass\":true}}", start.elapsed().as_secs_f64());
                return Ok(true);
            }
            let sig = signal(status);
            if sig != 5 {
                let r = regs(pid)?;
                return Err(format!("unexpected native signal {sig} after {count} calls; rip=0x{:x} native_offset=0x{:x} rsp=0x{:x}", r.rip, r.rip.wrapping_sub((entry-offset) as u64), r.rsp));
            }
            let registers = regs(pid)?;
            let address = registers.rip.checked_sub(1).ok_or("invalid native RIP")? as usize;
            if pow
                .as_ref()
                .and_then(|w| w.pending.as_ref())
                .is_some_and(|p| p.0 == address)
            {
                let result = xmm_words(pid)?[0];
                let w = pow.as_mut().unwrap();
                let (ret, word, inputs, implementation) = w.pending.take().unwrap();
                writeln!(w.file, "{{\"call\":{count},\"return_offset\":\"0x{:x}\",\"implementation_address\":\"0x{implementation:x}\",\"x_bits\":\"0x{:08x}\",\"y_bits\":\"0x{:08x}\",\"result_bits\":\"0x{result:08x}\"}}", ret - (entry-offset), inputs[0], inputs[1]).map_err(|e| e.to_string())?;
                w.records += 1;
                if w.records == w.limit {
                    if let Some(word) = pow_breakpoint.take() {
                        request(pid, 4, pow_address.unwrap(), word)?;
                    }
                }
                step_over(pid, registers, ret, word)?;
            } else if Some(address) == pow_address && pow_breakpoint.is_some() {
                let inputs = xmm_words(pid)?;
                let ret = u64::from_le_bytes(read::<8>(&memory, registers.rsp)?) as usize;
                let word = pow_breakpoint.take().unwrap();
                step_over(pid, registers, address, word)?;
                let implementation = regs(pid)?.rip as usize;
                if pow.as_ref().unwrap().records == 0 {
                    let maps = fs::read_to_string(format!("/proc/{pid}/maps"))
                        .map_err(|e| e.to_string())?;
                    let file = std::env::var_os("MC_TRACE_POWF_PATH").unwrap();
                    let mut path = std::path::PathBuf::from(file);
                    path.set_extension("maps");
                    OpenOptions::new()
                        .create_new(true)
                        .write(true)
                        .open(path)
                        .and_then(|mut f| f.write_all(maps.as_bytes()))
                        .map_err(|e| e.to_string())?;
                }
                pow.as_mut().unwrap().pending =
                    Some((ret, breakpoint(pid, ret)?, inputs, implementation));
                if pow.as_ref().unwrap().active(count) {
                    pow_breakpoint = Some(breakpoint(pid, address)?);
                }
            } else if Some(address) == secant_address && secant_breakpoint.is_some() {
                let word = secant_breakpoint.take().unwrap();
                witness
                    .as_mut()
                    .unwrap()
                    .observe(count, &memory, &registers)?;
                step_over(pid, registers, address, word)?;
                if witness.as_ref().unwrap().active(count) {
                    secant_breakpoint = Some(breakpoint(pid, address)?);
                }
            } else if address == entry && pending.is_none() {
                let pointers = arguments(&memory, &registers)?;
                let mut before = Vec::with_capacity(108);
                for pointer in pointers {
                    before.extend_from_slice(&read::<4>(&memory, pointer)?);
                }
                let ret = u64::from_le_bytes(read::<8>(&memory, registers.rsp)?) as usize;
                step_over(pid, registers, entry, entry_word)?;
                breakpoint(pid, entry)?;
                let ret_word = breakpoint(pid, ret)?;
                pending = Some((ret, ret_word, pointers, before));
                if witness.as_ref().is_some_and(|w| w.active(count)) {
                    secant_breakpoint = Some(breakpoint(pid, secant_address.unwrap())?);
                }
                if pow.as_ref().is_some_and(|w| w.active(count)) {
                    pow_breakpoint = Some(breakpoint(pid, pow_address.unwrap())?);
                }
            } else if pending.as_ref().is_some_and(|p| p.0 == address) {
                let (ret, ret_word, pointers, mut record) = pending.take().unwrap();
                for pointer in &pointers[15..] {
                    record.extend_from_slice(&read::<4>(&memory, *pointer)?);
                }
                file.write_all(&record).map_err(|e| e.to_string())?;
                count += 1;
                if let Some(word) = secant_breakpoint.take() {
                    request(pid, 4, secant_address.unwrap(), word)?;
                }
                if let Some(word) = pow_breakpoint.take() {
                    request(pid, 4, pow_address.unwrap(), word)?;
                }
                if pow.as_ref().is_some_and(|w| w.pending.is_some()) {
                    return Err("native MC returned with incomplete powf witness".into());
                }
                step_over(pid, registers, ret, ret_word)?;
                if count == limit {
                    request(pid, 4, entry, entry_word)?;
                }
            } else {
                return Err(format!("unexpected breakpoint at {address:x}"));
            }
        }
    })();
    if result.is_err() {
        let _ = child.kill();
        let _ = child.wait();
    }
    result
}
