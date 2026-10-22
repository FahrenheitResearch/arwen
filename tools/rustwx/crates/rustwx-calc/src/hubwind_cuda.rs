//! Original optional CUDA column kernel. Driver/NVRTC are discovered at runtime;
//! CPU installations retain the same Rust calculation contract. No CUDA SDK or
//! redistributable CUDA code is bundled. A caller must own its GPU allocation.
use crate::hubwind::{validate_columns, Columns, Plane};
use std::{
    ffi::{c_char, c_void, CString},
    ptr,
};
type Handle = *mut c_void;
type Init = unsafe extern "C" fn(u32) -> i32;
type Device = unsafe extern "C" fn(*mut i32, i32) -> i32;
type Count = unsafe extern "C" fn(*mut i32) -> i32;
type Create = unsafe extern "C" fn(*mut Handle, u32, i32) -> i32;
type Destroy = unsafe extern "C" fn(Handle) -> i32;
type Load = unsafe extern "C" fn(*mut Handle, *const c_void) -> i32;
type Function = unsafe extern "C" fn(*mut Handle, Handle, *const c_char) -> i32;
type Alloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
type Free = unsafe extern "C" fn(u64) -> i32;
type HtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
type DtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
type Launch = unsafe extern "C" fn(
    Handle,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    Handle,
    *mut *mut c_void,
    *mut *mut c_void,
) -> i32;
type Sync = unsafe extern "C" fn() -> i32;
// Only the installed OS loader is linked; no loader implementation is vendored.
struct Library(Handle);
#[cfg(unix)]
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(name: *const c_char, flags: i32) -> Handle;
    fn dlsym(library: Handle, name: *const c_char) -> Handle;
    fn dlclose(library: Handle) -> i32;
}
#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryA(name: *const c_char) -> Handle;
    fn GetProcAddress(library: Handle, name: *const c_char) -> Handle;
    fn FreeLibrary(library: Handle) -> i32;
}
impl Library {
    unsafe fn new(name: &str) -> Result<Self, String> {
        let name = CString::new(name).map_err(|e| e.to_string())?;
        #[cfg(unix)]
        let handle = unsafe { dlopen(name.as_ptr(), 2) };
        #[cfg(windows)]
        let handle = unsafe { LoadLibraryA(name.as_ptr()) };
        if handle.is_null() {
            Err("dynamic runtime is unavailable".into())
        } else {
            Ok(Self(handle))
        }
    }
}
impl Drop for Library {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            dlclose(self.0);
        }
        #[cfg(windows)]
        unsafe {
            FreeLibrary(self.0);
        }
    }
}
unsafe fn symbol<T: Copy>(lib: &Library, name: &[u8]) -> Result<T, String> {
    #[cfg(unix)]
    let address = unsafe { dlsym(lib.0, name.as_ptr().cast()) };
    #[cfg(windows)]
    let address = unsafe { GetProcAddress(lib.0, name.as_ptr().cast()) };
    if address.is_null() {
        return Err(format!(
            "missing runtime symbol {}",
            String::from_utf8_lossy(name)
        ));
    }
    assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<Handle>());
    Ok(unsafe { std::mem::transmute_copy(&address) })
}
fn check(code: i32, stage: &str) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("CUDA hubwind {stage} returned {code}"))
    }
}
fn library(names: &[&str]) -> Option<Library> {
    names.iter().find_map(|n| unsafe { Library::new(n).ok() })
}
struct Session {
    context: Handle,
    module: Handle,
    buffers: Vec<u64>,
    bytes: usize,
    destroy: Destroy,
    unload: Destroy,
    alloc: Alloc,
    free: Free,
    copy: HtoD,
    back: DtoH,
    launch: Launch,
    sync: Sync,
    _driver: Library,
}
impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            for &p in &self.buffers {
                (self.free)(p);
            }
            if !self.module.is_null() {
                (self.unload)(self.module);
            }
            if !self.context.is_null() {
                (self.destroy)(self.context);
            }
        }
    }
}
impl Session {
    fn buffer<T>(&mut self, v: &[T]) -> Result<u64, String> {
        let size = std::mem::size_of_val(v);
        let mut p = 0;
        unsafe {
            check((self.alloc)(&mut p, size), "allocate")?;
        }
        self.buffers.push(p);
        self.bytes += size;
        unsafe {
            check((self.copy)(p, v.as_ptr().cast(), size), "upload")?;
        }
        Ok(p)
    }
}
fn ptx(nvrtc: &Library) -> Result<Vec<u8>, String> {
    type New = unsafe extern "C" fn(
        *mut Handle,
        *const c_char,
        *const c_char,
        i32,
        *const *const c_char,
        *const *const c_char,
    ) -> i32;
    type Compile = unsafe extern "C" fn(Handle, i32, *const *const c_char) -> i32;
    type Size = unsafe extern "C" fn(Handle, *mut usize) -> i32;
    type Get = unsafe extern "C" fn(Handle, *mut c_char) -> i32;
    type Delete = unsafe extern "C" fn(*mut Handle) -> i32;
    unsafe {
        let new: New = symbol(nvrtc, b"nvrtcCreateProgram\0")?;
        let compile: Compile = symbol(nvrtc, b"nvrtcCompileProgram\0")?;
        let size: Size = symbol(nvrtc, b"nvrtcGetPTXSize\0")?;
        let get: Get = symbol(nvrtc, b"nvrtcGetPTX\0")?;
        let log_size: Size = symbol(nvrtc, b"nvrtcGetProgramLogSize\0")?;
        let log_get: Get = symbol(nvrtc, b"nvrtcGetProgramLog\0")?;
        let delete: Delete = symbol(nvrtc, b"nvrtcDestroyProgram\0")?;
        let source = CString::new(KERNEL).unwrap();
        let name = CString::new("hubwind.cu").unwrap();
        let mut program = ptr::null_mut();
        check(
            new(
                &mut program,
                source.as_ptr(),
                name.as_ptr(),
                0,
                ptr::null(),
                ptr::null(),
            ),
            "create NVRTC program",
        )?;
        // Architecture is virtual PTX for JIT on current/future supported devices.
        let opts = [
            CString::new("--gpu-architecture=compute_80").unwrap(),
            CString::new("--fmad=false").unwrap(),
        ];
        let ptrs: Vec<_> = opts.iter().map(|o| o.as_ptr()).collect();
        let code = compile(program, ptrs.len() as i32, ptrs.as_ptr());
        if code != 0 {
            let mut n = 0;
            let _ = log_size(program, &mut n);
            let mut log = vec![0u8; n];
            let _ = log_get(program, log.as_mut_ptr().cast());
            let _ = delete(&mut program);
            return Err(format!(
                "NVRTC hubwind compile {code}: {}",
                String::from_utf8_lossy(&log)
            ));
        }
        let result = (|| {
            let mut n = 0;
            check(size(program, &mut n), "PTX size")?;
            let mut out = vec![0; n];
            check(get(program, out.as_mut_ptr().cast()), "PTX")?;
            Ok(out)
        })();
        let _ = delete(&mut program);
        result
    }
}
/// None means no usable CUDA runtime/device. Any error after device selection
/// is surfaced, so a failed accelerator cannot quietly replace a requested run.
pub fn try_interpolate(
    c: &Columns<'_>,
    heights: &[f64],
) -> Result<Option<(Vec<Plane>, usize)>, String> {
    validate_columns(c, heights)?;
    let Some(driver) = library(&["libcuda.so.1", "nvcuda.dll"]) else {
        return Ok(None);
    };
    let Some(nvrtc) = library(&[
        "libnvrtc.so",
        "libnvrtc.so.13",
        "libnvrtc.so.12",
        "/usr/local/cuda/lib64/libnvrtc.so",
        "nvrtc64_130_0.dll",
        "nvrtc64_120_0.dll",
    ]) else {
        return Ok(None);
    };
    unsafe {
        let init: Init = symbol(&driver, b"cuInit\0")?;
        let count: Count = symbol(&driver, b"cuDeviceGetCount\0")?;
        let code = init(0);
        if code == 100 {
            return Ok(None);
        }
        check(code, "initialize")?;
        let mut devices = 0;
        check(count(&mut devices), "device count")?;
        if devices == 0 {
            return Ok(None);
        }
        let device: Device = symbol(&driver, b"cuDeviceGet\0")?;
        let create: Create = symbol(&driver, b"cuCtxCreate_v2\0")?;
        let load: Load = symbol(&driver, b"cuModuleLoadData\0")?;
        let function: Function = symbol(&driver, b"cuModuleGetFunction\0")?;
        let mut s = Session {
            context: ptr::null_mut(),
            module: ptr::null_mut(),
            buffers: vec![],
            bytes: 0,
            destroy: symbol(&driver, b"cuCtxDestroy_v2\0")?,
            unload: symbol(&driver, b"cuModuleUnload\0")?,
            alloc: symbol(&driver, b"cuMemAlloc_v2\0")?,
            free: symbol(&driver, b"cuMemFree_v2\0")?,
            copy: symbol(&driver, b"cuMemcpyHtoD_v2\0")?,
            back: symbol(&driver, b"cuMemcpyDtoH_v2\0")?,
            launch: symbol(&driver, b"cuLaunchKernel\0")?,
            sync: symbol(&driver, b"cuCtxSynchronize\0")?,
            _driver: driver,
        };
        let mut ordinal = 0;
        check(device(&mut ordinal, 0), "select device")?;
        check(create(&mut s.context, 0, ordinal), "create context")?;
        let code = ptx(&nvrtc)?;
        check(load(&mut s.module, code.as_ptr().cast()), "load PTX")?;
        let mut kernel = ptr::null_mut();
        check(
            function(&mut kernel, s.module, c"hubwind".as_ptr()),
            "find kernel",
        )?;
        let mut z = s.buffer(c.z_agl_m)?;
        let mut u = s.buffer(c.u_ms)?;
        let mut v = s.buffer(c.v_ms)?;
        let mut sin = s.buffer(c.sin)?;
        let mut cos = s.buffer(c.cos)?;
        let mut tke = match c.tke {
            Some(v) => s.buffer(v)?,
            None => 0,
        };
        let mut su = 0;
        let mut sv = 0;
        if let Some((u, v)) = c.surface {
            su = s.buffer(u)?;
            sv = s.buffer(v)?;
        }
        let mut h = s.buffer(heights)?;
        let nout = c
            .cells
            .checked_mul(heights.len())
            .and_then(|n| n.checked_mul(3))
            .ok_or("CUDA output size overflow")?;
        let mut out = vec![f32::NAN; nout];
        let mut output = s.buffer(&out)?;
        let mut cells = c.cells as u64;
        let mut levels = u32::try_from(c.levels).map_err(|_| "CUDA levels overflow")?;
        let mut nh = u32::try_from(heights.len()).map_err(|_| "CUDA heights overflow")?;
        let mut params: Vec<*mut c_void> = vec![
            (&mut z as *mut u64).cast(),
            (&mut u as *mut u64).cast(),
            (&mut v as *mut u64).cast(),
            (&mut sin as *mut u64).cast(),
            (&mut cos as *mut u64).cast(),
            (&mut tke as *mut u64).cast(),
            (&mut su as *mut u64).cast(),
            (&mut sv as *mut u64).cast(),
            (&mut h as *mut u64).cast(),
            (&mut output as *mut u64).cast(),
            (&mut cells as *mut u64).cast(),
            (&mut levels as *mut u32).cast(),
            (&mut nh as *mut u32).cast(),
        ];
        let blocks = u32::try_from((c.cells * heights.len()).div_ceil(256))
            .map_err(|_| "CUDA launch size overflow")?;
        check(
            (s.launch)(
                kernel,
                blocks,
                1,
                1,
                256,
                1,
                1,
                0,
                ptr::null_mut(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            ),
            "launch",
        )?;
        check((s.sync)(), "synchronize")?;
        check(
            (s.back)(
                out.as_mut_ptr().cast(),
                output,
                std::mem::size_of_val(out.as_slice()),
            ),
            "download",
        )?;
        let n = c.cells;
        let total = n * heights.len();
        let planes = heights
            .iter()
            .enumerate()
            .map(|(j, &height_m)| Plane {
                height_m,
                speed: out[j * n..(j + 1) * n].to_vec(),
                direction: out[total + j * n..total + (j + 1) * n].to_vec(),
                ti: out[2 * total + j * n..2 * total + (j + 1) * n].to_vec(),
            })
            .collect();
        Ok(Some((planes, s.bytes)))
    }
}
const KERNEL: &str = r#"
extern "C" __global__ void hubwind(const double*z,const double*u,const double*v,const double*sn,const double*cs,const double*tke,const double*su,const double*sv,const double*heights,float*out,unsigned long long n,unsigned int nz,unsigned int nh){
 unsigned long long idx=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,total=n*nh;if(idx>=total)return;
 unsigned long long i=idx%n;double h=heights[idx/n];float missing=__int_as_float(0x7fffffff);out[idx]=missing;out[total+idx]=missing;out[2*total+idx]=missing;
 if(!isfinite(sn[i])||!isfinite(cs[i])||fabs(sqrt(sn[i]*sn[i]+cs[i]*cs[i])-1.)>0.01)return;
 for(unsigned int k=0;k<nz;++k){if(!isfinite(z[k*n+i])||z[k*n+i]<0.||(k&&z[k*n+i]<=z[(k-1)*n+i]))return;}
 int a=-1,b=-1;double f=0.;for(unsigned int k=0;k<nz;++k){double zk=z[k*n+i];if(h==zk){a=k;b=k;break;}if(k+1<nz&&h>zk&&h<z[(k+1)*n+i]){a=k;b=k+1;f=(h-zk)/(z[(k+1)*n+i]-zk);break;}}
 double ue,ve,t=-1.;if(a>=0){unsigned long long ia=(unsigned long long)a*n+i,ib=(unsigned long long)b*n+i;double ug=u[ia]+(u[ib]-u[ia])*f,vg=v[ia]+(v[ib]-v[ia])*f;ue=ug*cs[i]-vg*sn[i];ve=ug*sn[i]+vg*cs[i];if(tke&&tke[ia]>=0.&&tke[ib]>=0.)t=tke[ia]+(tke[ib]-tke[ia])*f;}
 else if(su&&sv&&h>=10.&&h<z[i]&&z[i]>10.){f=(h-10.)/(z[i]-10.);ue=su[i]+(u[i]*cs[i]-v[i]*sn[i]-su[i])*f;ve=sv[i]+(u[i]*sn[i]+v[i]*cs[i]-sv[i])*f;}else return;
 double speed=sqrt(ue*ue+ve*ve);if(!isfinite(speed))return;out[idx]=(float)speed;if(speed>1e-6){double d=fmod(270.-atan2(ve,ue)*57.2957795130823208768,360.);if(d<0.)d+=360.;out[total+idx]=(float)d;if(isfinite(t)&&t>=0.)out[2*total+idx]=(float)(sqrt(2.*t/3.)/speed);}
}
"#;
