"""Generate the mosaic Fortran harness from the existing full-state oracle.

The WRF module is unmodified. The harness changes only initial conditions
and switches. Warm columns avoid WRF's undefined thin-snow stack reads.
"""

from pathlib import Path
import sys


def generate(output):
    original = Path(__file__).parents[1] / "ruc_wrf461_oracle/run_lsmruc.F90"
    text = original.read_text()
    text = text[text.index("program run_ruc_lsmruc_oracle"):]
    # Keep the declarations and driver, replacing the stale original preface.
    use = text.index("  use module_sf_ruclsm")
    text = "program run_ruc_mosaic_oracle\n" + text[use:]
    text = text.replace("end program run_ruc_lsmruc_oracle", "end program run_ruc_mosaic_oracle")
    text = text.replace("integer :: ktau, iswater, isice, mosaic_lu, mosaic_soil",
                        "integer :: ktau, iswater, isice, mosaic_lu, mosaic_soil, nactive, cropcat, naturalcat")
    text = text.replace("mosaic_lu = 0", "mosaic_lu = 1").replace("mosaic_soil = 0", "mosaic_soil = 1")
    begin = text.index("    landusef = 0.0")
    end = text.index("    call ruclsminit", begin)
    text = text[:begin] + """    ! Common warm-column forcing isolates mosaic arithmetic and irrigation.
    names = [character(len=24) :: 'dry_root_irrigation', 'threshold_greenness', &
        'below_threshold', 'no_irrigation_cover', 'partial_land_area', &
        'excess_land_area', 'water_soil_fallback', 'excess_soil_area', &
        'wet_soil', 'mixed_cover_forest', 'mixed_cover_savanna', 'mixed_cover_shrub']
    nactive = merge(21,28,run==1)
    cropcat = merge(12,3,run==1)
    naturalcat = merge(10,5,run==1)
    isltyp = 4
    xland = 1.0
    xice = 0.0
    snow = 0.0
    snowh = 0.0
    snowc = 0.0
    frzfrac = 0.0
    tso = 295.0
    soilt = 297.0
    soilt1 = 296.0
    tbot = 290.0
    t3d = 298.0
    qv3d = 0.008
    rainbl = 0.0
    soilmois = 0.12
    shdmin = 10.0
    shdmax = 90.0
    vegfra = 85.0
    vegfra(2,1)=70.0
    vegfra(3,1)=69.0
    landusef = 0.0
    soilctop = 0.0
    do i=1,ncol
      landusef(i,cropcat,1)=0.55
      landusef(i,naturalcat,1)=0.30
      landusef(i,1,1)=0.15
      soilctop(i,6,1)=0.35
      soilctop(i,8,1)=0.50
      soilctop(i,14,1)=0.15
    enddo
    ! No irrigation cover, incomplete area, area cap, and soil fallback.
    landusef(4,:,1)=0.0
    landusef(4,1,1)=1.0
    landusef(5,:,1)=landusef(5,:,1)*0.5
    landusef(6,:,1)=landusef(6,:,1)*1.1
    soilctop(7,:,1)=0.0
    soilctop(7,14,1)=1.0
    soilctop(8,:,1)=soilctop(8,:,1)*1.1
    soilmois(9,:,1)=0.35
    do i=1,ncol
      if(i==5) ivgtyp(i,1)=1
    enddo

""" + text[end:]
    text = text.replace("mminlu, landusef, nlcat, mosaic_lu, mosaic_soil, soilctop,",
                        "mminlu, landusef(:,1:nactive,:), nactive, mosaic_lu, mosaic_soil, soilctop,")
    text = text.replace("xice_threshold, nlcat, nscat,", "xice_threshold, nactive, nscat,")
    Path(output).write_text(text)


if __name__ == "__main__":
    generate(sys.argv[1])
