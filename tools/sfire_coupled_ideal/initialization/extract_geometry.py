"""Byte-extract ideal-fire grid and soil routines into a minimal native control."""
from pathlib import Path
import argparse
import hashlib
import json

FIELDS = ("znw", "znu", "dnw", "rdnw", "dn", "rdn", "fnp", "fnm",
          "c1f", "c2f", "c3f", "c4f", "c1h", "c2h", "c3h", "c4h")


def section(text, first, last):
    start = text.index(first)
    end = text.index(last, start)
    return text[start:end]


def extract(source_root, destination, original=False):
    root = Path(source_root)
    source = (root / "dyn_em/module_initialize_fire.F").read_text()
    soil = (root / "share/module_soil_pre.F").read_text()
    vertical = section(source, "   IF (model_config_rec%eta_levels(1)", "!  get the sounding from the ascii")
    hybrid = section(source, "  !  For hybrid coord", "! get fire mesh dimensions")
    mountain = section(source, "  IF(mtn_type .ne. 0)THEN", "  ELSE ! mtn_type") + "  ENDIF\n"
    gradient = section(source, "  if(have_fire_grad)then", "   if(.not.have_fire_grad)call crash")
    soil_names = ("process_soil_ideal", "init_soil_depth_1", "init_soil_depth_2", "init_soil_depth_3",
                  "init_soil_1_ideal", "init_soil_2_ideal")
    soil_routines = []
    for name in soil_names:
        start = soil.index("   SUBROUTINE " + name)
        end_marker = "   END SUBROUTINE " + name
        end = soil.index(end_marker, start) + len(end_marker)
        soil_routines.append(soil[start:end] + "\n")
    original_soil = list(soil_routines)
    if not original:
        marker = "      dzs(1) = zs2(2) - zs2(1)\n"
        if soil_routines[3].count(marker) != 1:
            raise ValueError("native RUC first depth-edge calculation changed")
        soil_routines[3] = soil_routines[3].replace(marker, marker + "      zs2(1) = zs2(2) ! preserve the first midpoint edge before the loop\n")
    prefix = """module ideal_geometry_control
implicit none
integer,parameter::max_eta=1000,SLABSCHEME=1,LSMSCHEME=2,NOAHMPSCHEME=4,RUCLSMSCHEME=3
type grid_record
 integer::id=1
 real::cf1=0.,cf2=0.,cf3=0.,cfn=0.,cfn1=0.,rdx=0.,rdy=0.,p_top=0.,etac=0.,dx=0.,dy=0.
 real,allocatable::ht(:,:),zsf(:,:),dzdxf(:,:),dzdyf(:,:)
""" + "\n".join(" real,allocatable::" + name + "(:)" for name in FIELDS) + """
end type
type configuration_record
 real::dx=0.,dy=0.
 integer::hybrid_opt=0
end type
type model_record
 real::eta_levels(max_eta)=-1.
 integer::e_vert(1)=0
end type
contains
subroutine initialize_vertical(grid,nz,stretch_grd,stretch_hyp,z_scale,config_flags,model_config_rec)
 type(grid_record),intent(inout)::grid
 type(configuration_record),intent(in)::config_flags
 type(model_record),intent(in)::model_config_rec
 integer,intent(in)::nz
 logical,intent(in)::stretch_grd,stretch_hyp
 real,intent(in)::z_scale
 integer::k,kds,kde,kts,kte,ks,ke,id
 real::cof1,cof2,B1,B2,B3,B4,B5
 real,parameter::p1000mb=100000.
 kds=1;kde=nz+1;kts=1;kte=nz+1
"""
    mountain_prefix = """subroutine initialize_mountain(grid,nx,ny,fx,fy,fdx,fdy,mtn_type,mtn_xs,mtn_ys,mtn_xe,mtn_ye,mtn_ht)
 type(grid_record),intent(inout)::grid
 integer,intent(in)::nx,ny,fx,fy,mtn_type
 real,intent(in)::fdx,fdy,mtn_xs,mtn_ys,mtn_xe,mtn_ye,mtn_ht
 integer::i,j,ids,jds,ifds,jfds,its,ite,jts,jte,ifts,ifte,jfts,jfte
 real::pi,mtn_axs,mtn_axe,mtn_ays,mtn_aye,mtn_fxs,mtn_fxe,mtn_fys,mtn_fye,mtn_x,mtn_y,mtn_z,mtn_max
 logical::have_fire_ht
 ids=1;jds=1;ifds=1;jfds=1;its=1;ite=nx;jts=1;jte=ny;ifts=1;ifte=fx;jfts=1;jfte=fy
 pi=2.*asin(1.0)
"""
    gradient_prefix = """subroutine initialize_gradient(grid,fx,fy,fdx,fdy)
 use module_fr_fire_util,only:continue_at_boundary
 type(grid_record),intent(inout)::grid
 integer,intent(in)::fx,fy
 real,intent(in)::fdx,fdy
 integer::i,j,ifms,ifme,jfms,jfme,ifds,ifde,jfds,jfde,ifts,ifte,jfts,jfte,iots,iote,jots,jote
 logical::have_fire_grad,have_fire_ht
 ifms=0;ifme=fx+1;jfms=0;jfme=fy+1;ifds=1;ifde=fx;jfds=1;jfde=fy;ifts=1;ifte=fx;jfts=1;jfte=fy
 have_fire_grad=.false.;have_fire_ht=.true.
"""
    output = prefix + vertical + hybrid + "end subroutine\n" + mountain_prefix + mountain + "end subroutine\n"
    output += gradient_prefix + gradient + "end subroutine\n"
    output += "\n".join(soil_routines) + "end module\n"
    if original:
        output = output.replace("module ideal_geometry_control", "module ideal_geometry_control_original")
    Path(destination).write_text(output)
    return {"source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "soil_source_sha256": hashlib.sha256(soil.encode()).hexdigest(),
            "extracted_block_sha256": {name: hashlib.sha256(block.encode()).hexdigest() for name, block in
                (("vertical", vertical), ("hybrid", hybrid), ("mountain", mountain), ("gradient", gradient),
                 *zip(soil_names, original_soil))},
            "correction": "none" if original else "RUC DZS(2) first midpoint edge preserved; source DZS(2) overlaps layer one",
            "control_sha256": hashlib.sha256(output.encode()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root")
    parser.add_argument("destination")
    parser.add_argument("--original", action="store_true")
    args = parser.parse_args()
    print(json.dumps(extract(args.source_root, args.destination, args.original), indent=2))
