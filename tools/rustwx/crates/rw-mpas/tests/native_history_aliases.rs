//! Native v8.4.2 stored fields feed the real render conversion without renaming data in Python.
use rw_mpas::history::{read_history, Timestamp};
use rw_store::netcdf_classic::{NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef};

#[test]
fn canonical_native_carriers_and_legacy_fallback_preserve_stored_words() {
    let pairs = [("U", "uReconstructZonal", "u_zonal", true),
        ("V", "uReconstructMeridional", "v_meridional", true),
        ("T2", "t2m", "t2", false), ("TSK", "skintemp", "tsk", false),
        ("PBLH", "hpbl", "pblh", false), ("SWDOWN", "swdnb", "swdown", false)];
    for (native, legacy) in [(true, false), (false, true), (true, true)] {
        let directory = std::env::temp_dir().join(format!("rw-mpas-native-history-{}-{native}-{legacy}", std::process::id()));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("history.nc");
        let dimensions = vec![NcDim::record("Time"), NcDim::fixed("nCells", 2), NcDim::fixed("nVertLevels", 2)];
        let mut variables = vec![NcVarDef::new("latCell", NcType::Double, vec![1]), NcVarDef::new("lonCell", NcType::Double, vec![1])];
        for (_, canonical, old, mass) in pairs {
            for name in [native.then_some(canonical), legacy.then_some(old)].into_iter().flatten() {
                variables.push(NcVarDef::new(name, NcType::Float, if mass {vec![0,1,2]} else {vec![0,1]}));
            }
        }
        let mut writer = NcClassicWriter::create(&path, NcFormat::Data64, dimensions, vec![], variables, 1).unwrap();
        writer.put("latCell", NcData::Doubles(&[0., 0.1])).unwrap();
        writer.put("lonCell", NcData::Doubles(&[0., 0.1])).unwrap();
        for (_, canonical, old, mass) in pairs {
            let values = if mass {vec![-0., f32::from_bits(1), 1., -2.]} else {vec![280., 281.]};
            if native {writer.put_record(canonical, 0, NcData::Floats(&values)).unwrap();}
            if legacy {
                let fallback: Vec<f32> = values.iter().map(|x| if native {*x+32.} else {*x}).collect();
                writer.put_record(old, 0, NcData::Floats(&fallback)).unwrap();
            }
        }
        writer.finish().unwrap();
        let frame = read_history(&path, None, Some(Timestamp::parse("2000-01-01_00:00:00").unwrap()), None, None).unwrap();
        for (wrf, _, _, mass) in pairs {
            let want: Vec<f32> = if mass {vec![-0., f32::from_bits(1), 1., -2.]} else {vec![280., 281.]};
            assert_eq!(frame.fields[wrf].0.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "{wrf} native={native} legacy={legacy}");
            assert!(!frame.absent.iter().any(|x| x==wrf));
        }
        assert!(frame.absent.iter().any(|x| x=="Q2"), "aliases must not fabricate unrelated fields");
        std::fs::remove_dir_all(directory).unwrap();
    }
}
