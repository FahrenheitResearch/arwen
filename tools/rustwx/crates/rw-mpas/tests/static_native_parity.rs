//! Golden words emitted by pristine MPAS-A v8.4.2 interpolation, GNU strict.
//! Regenerate with tests/native_static/interpolation.f90 and the pinned module.
use rw_mpas::static_native::Plane;

#[test]
fn static_support_neighbor_walk_and_owned_diameter_match_pristine_words() {
    use rw_mpas::static_native::{nearest_cell,max_cell_diameter};
    let mut lat=Vec::new();let mut lon=Vec::new();let mut counts=Vec::new();
    let mut neighbors=Vec::new();let mut vertices=Vec::new();
    let mut vertex_lat=Vec::new();let mut vertex_lon=Vec::new();
    for cell in 1usize..=16 {
        lat.push((cell as i32-8) as f32*0.03);
        lon.push(cell as f32*0.31-3.0);
        counts.push(2+cell%3);
        neighbors.extend([1+cell%16,1+(cell as i32-2).rem_euclid(16) as usize,1+(cell+1)%16,17]);
        for edge in 1..=4 {
            vertices.push((cell-1)*4+edge);
            vertex_lat.push(lat[cell-1]+(edge as i32-2) as f32*0.007);
            vertex_lon.push(lon[cell-1]+(edge as i32-3) as f32*0.013);
        }
    }
    let mut n=0;
    for line in include_str!("native_static/static_support.words").lines() {
        let fields:Vec<_>=line.split_whitespace().collect();
        let q:usize=fields[0].parse().unwrap();
        let target_lat=-0.25+(q*31%701) as f32*0.0007;
        let target_lon=-3.0+(q*17%1257) as f32*0.005;
        let cell=nearest_cell(target_lat,target_lon,1+q%16,4,&counts,&neighbors,&lat,&lon);
        assert_eq!(cell,fields[1].parse::<usize>().unwrap(),"neighbor walk q={q}");
        let radius=6371229.0*(1.0+(q%11) as f32*0.001);
        let diameter=max_cell_diameter(q%17,4,&counts,&vertices,&lat,&lon,&vertex_lat,&vertex_lon,radius);
        assert_eq!(diameter.to_bits(),u32::from_str_radix(fields[2],16).unwrap(),"diameter q={q}");
        n+=1;
    }
    assert_eq!(n,2048);
}

#[test]
fn regional_containment_matches_pristine_math_and_decisions() {
    use rw_mpas::static_native::{xyz,arc_length,sphere_angle,mirror_point,in_cell};
    let r=6371229.0;
    let mut differences=Vec::new();
    let mut count=0;
    for line in include_str!("native_static/containment.words").lines() {
        let fields:Vec<_>=line.split_whitespace().collect();
        let q:usize=fields[0].parse().unwrap();
        let lat=-1.4+(q*31%2801) as f32*0.001;
        let lon=(q*71%6281) as f32*0.001;
        let cell=xyz(r,lat,lon);
        let a=xyz(r,lat+0.01,lon-0.02);
        let b=xyz(r,lat-0.02,lon+0.01);
        let p=xyz(r,lat+((q%41) as i32-20) as f32*0.001,lon+0.007);
        let vertices=[a,b,xyz(r,lat-0.015,lon-0.03)];
        let unit=|v:[f32;3]| v.map(|x|x/r);
        let mirror=mirror_point(unit(cell),unit(a),unit(b));
        let actual=[arc_length(p,cell),sphere_angle(unit(a),unit(cell),unit(b)),mirror[0],mirror[1],mirror[2]];
        for k in 0..5 {
            let expected=u32::from_str_radix(fields[k+1],16).unwrap();
            if actual[k].to_bits()!=expected && differences.len()<20 { differences.push((q,k,expected,actual[k].to_bits())); }
        }
        let inside=in_cell(p,cell,&vertices);
        if inside!=(fields[6]=="1") && differences.len()<20 { differences.push((q,5,u32::from(fields[6]=="1"),u32::from(inside))); }
        count+=1;
    }
    assert_eq!(count,4096);
    assert!(differences.is_empty(),"pristine containment differs: {differences:?}");
}

#[test]
fn interpolation_matches_pristine_mpas_on_all_fallbacks() {
    let words = include_str!("native_static/interpolation.words");
    let mut planes=Vec::new();
    for mode in 0..6 {
        let mut values=Vec::new();
        for j in -2i32..=13 { for i in -2i32..=13 {
            let mut v=(11+i*17+j*23) as f32 * 0.0137;
            if (mode==1 && (i+2*j).rem_euclid(3)==0)
                || (mode==2 && (2..=9).contains(&i) && (2..=9).contains(&j))
                || mode==3 || (mode==4 && (i+j).rem_euclid(2)==0)
                || (mode==5 && (i!=0 || j!=0)) { v=0.0; }
            values.push(v);
        }}
        planes.push(Plane{x0:-2,y0:-2,nx:16,ny:16,values});
    }
    let mut mismatches=Vec::new();
    let mut n=0;
    for line in words.lines() {
        let fields:Vec<_>=line.split_whitespace().collect();
        let mode:usize=fields[0].parse().unwrap();
        let x=f32::from_bits(fields[2].parse::<i32>().unwrap() as u32);
        let y=f32::from_bits(fields[3].parse::<i32>().unwrap() as u32);
        let expected=u32::from_str_radix(fields[4],16).unwrap();
        let actual=planes[mode].interpolate(x,y,0.0).to_bits();
        if actual!=expected && mismatches.len()<20 { mismatches.push((mode,fields[1],x,y,expected,actual)); }
        n+=1;
    }
    assert_eq!(n,6144);
    assert!(mismatches.is_empty(), "native interpolation words differ: {mismatches:?}");
}

#[test]
fn tile_coordinates_cartesian_points_and_sphere_distance_match_native_words() {
    use rw_mpas::static_native::{tile_latlon,xyz,sphere_distance};
    let mut mismatches=Vec::new();
    let mut n=0;
    for line in include_str!("native_static/geometry.words").lines() {
        let f:Vec<_>=line.split_whitespace().collect();
        let q:usize=f[0].parse().unwrap();
        let s=1+q%7;
        let (nx,ny,lat0,lon0,dx,dy)=if q%2==0 {(6,3,89.975,-179.975,0.05,-0.05)}
            else {(36,18,-89.995833,-179.99583,0.00833333,0.00833333)};
        let xs=(1+1200*(q*17%nx)) as i64;
        let ys=(1+1200*(q*11%ny)) as i64;
        let i=(q*43%1200)*s+q*71%s;
        let j=(q*59%1200)*s+q*37%s;
        let (lat,lon)=tile_latlon(lat0,lon0,dy,dx,xs,ys,i,j,s);
        let [x,y,z]=xyz(6371229.0,lat,lon);
        let d=sphere_distance(lat,lon,lat+0.01,lon+0.02,6371229.0);
        let actual=[lat,lon,x,y,z,d].map(f32::to_bits);
        for k in 0..6 {
            let expected=u32::from_str_radix(f[k+1],16).unwrap();
            if actual[k]!=expected && mismatches.len()<20 { mismatches.push((q,k,expected,actual[k])); }
        }
        n+=1;
    }
    assert_eq!(n,4096);
    assert!(mismatches.is_empty(),"native coordinate words differ: {mismatches:?}");
}

#[test]
fn geogrid_decode_matches_native_for_all_sizes_endians_and_sign_boundaries() {
    use rw_mpas::static_geog::GeogDataset;
    let lines:Vec<_>=include_str!("native_static/decode.words").lines().collect();
    assert_eq!(lines.len(),256);
    for group in lines.chunks_exact(16) {
        let header:Vec<_>=group[0].split_whitespace().collect();
        let ws:usize=header[0].parse().unwrap();
        let signed=header[1]=="1";
        let little=header[2]=="1";
        let dir=std::env::temp_dir().join(format!("rw-mpas-native-decode-{}-{ws}-{signed}-{little}",std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let index=format!("type=continuous\nprojection=regular_ll\ndx=1\ndy=1\nknown_lat=0\nknown_lon=0\ntile_x=16\ntile_y=1\ntile_z=1\nwordsize={ws}\nendian={}\nsigned={}\n",if little {"little"} else {"big"},if signed {"yes"} else {"no"});
        std::fs::write(dir.join("index"),index).unwrap();
        let mut bytes=Vec::new();
        let mut expected=Vec::new();
        for line in group {
            let f:Vec<_>=line.split_whitespace().collect();
            let raw=u32::from_str_radix(f[4],16).unwrap();
            for b in 0..ws {
                let shift=8*if little {b} else {ws-1-b};
                bytes.push((raw>>shift) as u8);
            }
            expected.push(f[5].parse::<i64>().unwrap());
        }
        std::fs::write(dir.join("00001-00016.00001-00001"),bytes).unwrap();
        let ds=GeogDataset::open(&dir).unwrap();
        let actual=ds.read_plane(&ds.tiles[0],0).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(actual,expected,"native decode ws={ws} signed={signed} little={little}");
    }
}

/// The replay supplies an actual pinned strict native static and a freshly
/// generated shipping static, never a copy of the writer's expected metadata.
#[test]
#[ignore = "requires published native static and freshly generated shipping candidate"]
fn every_static_variable_attribute_matches_native() {
    let native=std::env::var_os("RW_MPAS_NATIVE_STATIC").expect("RW_MPAS_NATIVE_STATIC");
    let candidate=std::env::var_os("RW_MPAS_CANDIDATE_STATIC").expect("RW_MPAS_CANDIDATE_STATIC");
    let result=rw_mpas::staticfile::compare::compare(std::path::Path::new(&native),std::path::Path::new(&candidate)).unwrap();
    assert!(result.attributes_compared > 0);
    assert!(result.attributes_identical, "{:?}; missing {:?}; extra {:?}",
        result.attribute_mismatch,result.only_in_reference,result.only_in_candidate);
    println!("All {} variable attributes match native across {} variables",result.attributes_compared,result.fields.len());
}
