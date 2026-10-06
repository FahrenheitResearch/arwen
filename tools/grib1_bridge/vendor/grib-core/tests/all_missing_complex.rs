//! Captured native exporter messages and their product metadata.
//! Source set SHA256 9349f4400d66d0243932c5052016699a18d52b1ec95e68918a5d7f35dcf938af.
//! Missing message offset569,length554,SHA256348369d9e8d430f6d3febf6f6577098ffc8b1ef9b7fbfa74c01b02a03cdbd7dd.
//! Normal message offset0,length569,SHA256f5b9c057aa98eeafb071b0fc47175be20243fbe343ce180f8b3da409deecc111.
use grib_core::grib2::{Grib2File,Grib2Message,unpack_message,unpack_message_normalized,
    unpack_message_scan_normalized_row_window};

fn decode_hex(hex:&str)->Vec<u8>{
    let chars:Vec<_>=hex.bytes().filter(|b|!b.is_ascii_whitespace()).collect();
    assert_eq!(chars.len()%2,0);
    chars.chunks_exact(2).map(|c|u8::from_str_radix(std::str::from_utf8(c).unwrap(),16).unwrap()).collect()
}
fn missing()->Grib2Message {Grib2File::from_bytes(&decode_hex(MISSING)).unwrap().messages.remove(0)}
fn assert_missing(msg:&Grib2Message){
    let n=msg.grid.nx as usize*msg.grid.ny as usize;
    let full=unpack_message(msg).unwrap();
    assert_eq!(full.len(),n);assert!(full.iter().all(|v|v.is_nan()));
    for start in 0..=msg.grid.ny as usize {for end in start..=msg.grid.ny as usize{
        let rows=unpack_message_scan_normalized_row_window(msg,start,end).unwrap();
        assert_eq!(rows.len(),msg.grid.nx as usize*(end-start));
        assert!(rows.iter().all(|v|v.is_nan()));
    }}
}
fn assert_refused(msg:&Grib2Message){
    assert!(unpack_message(msg).is_err(),"full decode accepted {msg:?}");
    assert!(unpack_message_scan_normalized_row_window(msg,0,msg.grid.ny as usize).is_err(),
        "window decode accepted {msg:?}");
}

#[test]
fn actual_export_keeps_zero_spatial_descriptors_and_missing_cells(){
    let mut msg=missing();
    assert_eq!((msg.grid.nx,msg.grid.ny),(5,5));
    assert_eq!(msg.data_rep.template,3);
    assert_eq!(msg.data_rep.section5_num_data_points,0);
    assert_eq!(msg.data_rep.num_groups,0);
    assert_eq!(msg.raw_data,[0,0]);
    assert_missing(&msg);
    msg.grid.scan_mode=0;assert_missing(&msg);
}

#[test]
fn all_missing_simple_and_complex_canonical_forms_agree(){
    for template in [0,2,3]{
        let mut msg=missing();msg.data_rep.template=template;
        if template!=3{msg.data_rep.spatial_diff_order=0;msg.data_rep.spatial_diff_bytes=0;msg.raw_data.clear();}
        assert_missing(&msg);
        msg.bitmap=Some(vec![false;25]);assert_missing(&msg);
        msg.bitmap=Some(vec![false;32]);assert_missing(&msg);
        if template==3{
            for order in [1,2]{for width in 0..=8{
                msg.data_rep.spatial_diff_order=order;msg.data_rep.spatial_diff_bytes=width;
                msg.raw_data=vec![0;(order as usize+1)*width as usize];
                assert_missing(&msg);
            }}
        }
    }
}

#[test]
fn missing_or_malformed_bitmap_never_becomes_an_empty_field(){
    for template in [0,2,3]{
        let mut base=missing();base.data_rep.template=template;
        if template!=3{base.data_rep.spatial_diff_order=0;base.data_rep.spatial_diff_bytes=0;base.raw_data.clear();}
        for bitmap in [None,Some(vec![false;24]),Some(vec![false;26]),Some(vec![false;40])]{
            let mut msg=base.clone();msg.bitmap=bitmap;assert_refused(&msg);
        }
        for index in [0,24,31]{let mut msg=base.clone();let mut bitmap=vec![false;32];bitmap[index]=true;msg.bitmap=Some(bitmap);assert_refused(&msg);}
    }
}

#[test]
fn zero_count_refuses_groups_bad_descriptors_data_tails_and_unsupported_templates(){
    let base=missing();
    for change in 0..15{
        let mut msg=base.clone();
        match change{
            0=>msg.data_rep.num_groups=1,
            1=>msg.data_rep.group_width_ref=1,
            2=>msg.data_rep.group_width_bits=1,
            3=>msg.data_rep.last_group_length=1,
            4=>msg.data_rep.group_length_bits=1,
            5=>msg.data_rep.bits_per_value=1,
            6=>msg.data_rep.group_splitting_method=0,
            7=>msg.data_rep.missing_value_management=1,
            8=>msg.data_rep.original_field_type=2,
            9=>msg.data_rep.reference_value=f32::NAN,
            10=>msg.data_rep.spatial_diff_order=0,
            11=>msg.data_rep.spatial_diff_order=3,
            12=>msg.data_rep.spatial_diff_bytes=9,
            13=>msg.raw_data[0]=1,
            14=>msg.raw_data.push(0),
            _=>unreachable!(),
        }
        assert_refused(&msg);
    }
    for payload in [vec![],vec![0],vec![0,0,1]]{let mut msg=base.clone();msg.raw_data=payload;assert_refused(&msg);}
    for template in [0,2]{let mut msg=base.clone();msg.data_rep.template=template;msg.data_rep.spatial_diff_order=0;msg.data_rep.spatial_diff_bytes=0;msg.raw_data=vec![0];assert_refused(&msg);}
    for template in [4,40,41,42,51,999]{let mut msg=base.clone();msg.data_rep.template=template;msg.raw_data.clear();assert_refused(&msg);}
}

#[test]
fn ordinary_captured_message_bytes_and_values_are_unchanged(){
    let bytes=decode_hex(NORMAL);let preserved=bytes.clone();
    let msg=Grib2File::from_bytes(&bytes).unwrap().messages.remove(0);
    // Captured before the reader repair: five identical x-gradient rows.
    let want=[320.,322.,324.,326.,328.,320.,322.,324.,326.,328.,320.,322.,324.,326.,328.,
        320.,322.,324.,326.,328.,320.,322.,324.,326.,328.];
    let full=unpack_message(&msg).unwrap();
    for (&a,&b) in full.iter().zip(&want){assert_eq!(a.to_bits(),(b as f64).to_bits());}
    assert_eq!(full.len(),25);
    let normalized=unpack_message_normalized(&msg).unwrap();
    for start in 0..=5 {for end in start..=5 {
        let rows=unpack_message_scan_normalized_row_window(&msg,start,end).unwrap();
        assert_eq!(rows,normalized[start*5..end*5]);
    }}
    assert_eq!(bytes,preserved);
}

const MISSING:&str=r#"
4752494200000002000000000000022a00000015010007000002010107ea0a0211000000010000014c027b22646566696e6974696f6e223a22555050204e4341
5220544832206174207265636f6e7374727563746564207368656c7465722070726573737572653b204341504120302e3238353839363431222c22646566696e
6974696f6e73223a22757070222c226669656c64223a227432222c22677269645f636f7272656374696f6e223a226e617469766520575246206c6f636174696f
6e73207072657365727665643a2070726f6a65637465642073706163696e67207363616c656420627920363337313232392f363337303030303b20616e67756c
61722073706163696e6720756e6368616e676564222c22736368656d61223a2267726962322d6578706f72742e6669656c642f7631222c22756e697473223a22
4b222c227570705f70726f66696c65223a227570702d7372772d76322e322e30222c2277696e6473223a2267726964227d000000510300000000190000001e06
000000000000000000000000000000000000050000000502625a000f7f49000801c9c3800f7f4900002dc903002dc903004001c9c38001c9c380000000000000
000000000022040000000000000200740000000100000001670000000002ff00000000000000003105000000000003000000000000000000000100ffffffffff
ffffff0000000000000000001001000000000001010000000a0600000000000000000707000037373737
"#;

const NORMAL:&str=r#"
4752494200000002000000000000023900000015010007000002010107ea0a0211000000010000014b027b22646566696e6974696f6e223a22555050204e4341
52207375726661636520696e746572666163652067656f706f74656e7469616c206469766964656420627920392e3831222c22646566696e6974696f6e73223a
22757070222c226669656c64223a227465727261696e222c22677269645f636f7272656374696f6e223a226e617469766520575246206c6f636174696f6e7320
7072657365727665643a2070726f6a65637465642073706163696e67207363616c656420627920363337313232392f363337303030303b20616e67756c617220
73706163696e6720756e6368616e676564222c22736368656d61223a2267726962322d6578706f72742e6669656c642f7631222c22756e697473223a2267706d
222c227570705f70726f66696c65223a227570702d7372772d76322e322e30222c2277696e6473223a2267726964227d000000510300000000190000001e0600
0000000000000000000000000000000000050000000502625a000f7f49000801c9c3800f7f4900002dc903002dc903004001c9c38001c9c38000000000000000
0000000022040000000003050200740000000100000001010000000000ff0000000000000000310500000019000343a000008001000000000100ffffffffffff
ffff0000000206000000001001000000090002010000000606ff0000001b07000494000514500a14514028514500a145140285145037373737
"#;
