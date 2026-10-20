//! Offline source-fixture extraction and native record layout conversion.
//! Never linked into the product. No numerical reservoir implementation here.
use std::{env, fs, path::Path};

fn scalar(section: &str, key: &str) -> String {
    section.lines().filter_map(|line| {
        let (name, value) = line.trim().split_once('=')?;
        if name.trim() == key { Some(value.trim().to_owned()) } else { None }
    }).last().unwrap_or_else(|| panic!("missing source parameter {key}"))
}

fn main() {
    let args: Vec<_> = env::args().collect();
    if args[1] == "pack" {
        let data = fs::read(&args[2]).unwrap();
        assert_eq!(data.len() % 64, 0, "native records must be 16 float32 words");
        let n = data.len() / 64;
        let mut input = vec![0u8; n * 56];
        let mut output = vec![0u8; n * 8];
        for r in 0..n {
            for c in 0..16 {
                let source = &data[(16*r+c)*4..(16*r+c+1)*4];
                let (target, column) = if c < 14 { (&mut input, c) } else { (&mut output, c-14) };
                target[(column*n+r)*4..(column*n+r+1)*4].copy_from_slice(source);
            }
        }
        fs::write(Path::new(&args[3]).join("input.f32"), input).unwrap();
        fs::write(Path::new(&args[3]).join("expected.f32"), output).unwrap();
        println!("{{\"calls\":{n},\"input_columns\":14,\"output_columns\":2}}");
        return;
    }
    let root = Path::new(&args[1]);
    let output = Path::new(&args[2]);
    let source = fs::read_to_string(root.join("module_levelpool.F")).unwrap();
    let begin = source.find("    subroutine LEVELPOOL_PHYSICS(").unwrap();
    let tail = "    end subroutine LEVELPOOL_PHYSICS";
    let end = source[begin..].find(tail).unwrap() + begin + tail.len();
    fs::write(output.join("native_levelpool.f90"), &source[begin..end]).unwrap();
    let tests = fs::read_to_string(root.join("test_compute_kernel.py")).unwrap();
    let fixture = tests.split("def lp_reservoir3():").nth(1).unwrap().split("@pytest.fixture()").next().unwrap();
    let run = tests.split("def test_lp3_run(").nth(1).unwrap();
    let list = run.split("inflow_list = [").nth(1).unwrap().split(']').next().unwrap();
    let inflows: Vec<f32> = list.split(',').filter_map(|v| {
        let s=v.trim(); if s.is_empty() {None} else {Some(s.parse().unwrap())}
    }).collect();
    fs::write(output.join("inflows.f32"), inflows.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>()).unwrap();
    let names = ["lake_area", "weir_elevation", "max_depth", "weir_coefficient", "weir_length", "dam_length", "orifice_elevation", "orifice_coefficient", "orifice_area"];
    let values: Vec<_> = names.iter().map(|key|scalar(fixture,key)).collect();
    let parameters = values.iter().map(|s|s.parse::<f32>().unwrap()).flat_map(|f|f.to_le_bytes()).collect::<Vec<_>>();
    fs::write(output.join("parameters.f32"), parameters).unwrap();
    fs::write(output.join("height.f32"), scalar(fixture,"water_elevation").parse::<f32>().unwrap().to_le_bytes()).unwrap();
    let driver = format!(r#"program capture
implicit none
integer :: i
real :: inflows({n}), p(9), h, initial, q
open(10,file='inflows.f32',access='stream',form='unformatted',status='old')
read(10) inflows
close(10)
open(10,file='parameters.f32',access='stream',form='unformatted',status='old')
read(10) p
close(10)
open(10,file='height.f32',access='stream',form='unformatted',status='old')
read(10) h
close(10)
open(11,file='native-records.f32',access='stream',form='unformatted',status='replace')
do i=1,{n}
  initial=h
  ! bind_lp.f90 passes current inflow for BOTH arguments, not previous step.
  call LEVELPOOL_PHYSICS(4185105,inflows(i),inflows(i),q,0.0,300.0,h, &
       p(1),p(2),p(3),p(4),p(5),p(6),p(7),p(8),p(9))
  write(11) inflows(i),inflows(i),0.0,300.0,initial,p,h,q
end do
close(11)
end program capture
"#, n=inflows.len());
    fs::write(output.join("capture.f90"), driver).unwrap();
    println!("{{\"case\":\"t-route shipped Pocono network inflows\",\"lake_id\":4185105,\"calls\":{},\"dt_seconds\":300}}", inflows.len());
}
