"""Produce an explicitly corrected WRF atmospheric SFIRE source and diff.

The pinned original remains unchanged. Corrections use integrated truncated
Gaussian layer fractions and source tendencies, rather than differentiating
sampled PDF values as if they were remaining upward interface fluxes.
"""
from pathlib import Path
import argparse
import difflib


def correct(source):
    def once(old,new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f"pinned WRF source anchor differs: {old[:60]!r}")
        source=source.replace(old,new,1)

    once("real, dimension(its:ite,kts:kte,jts:jte) :: prop_smk",
         "real, dimension(ims:ime,kms:kme,jms:jme) :: prop_smk")
    once("REAL, DIMENSION( its:ite,kts:kte,jts:jte ) :: prop_heat",
         "REAL, DIMENSION( ims:ime,kms:kme,jms:jme ) :: prop_crown\n"
         "   REAL, DIMENSION( ims:ime,kms:kme,jms:jme ) :: prop_heat")
    once("i_st,i_en, j_st,j_en, k_st,k_en, dz8w, &\n                   fire_heat_peak",
         "i_st,i_en, j_st,j_en, k_st,k_en-1, dz8w, &\n                   fire_heat_peak")
    once("                   prop_heat)\n   end if",
         "                   prop_heat)\n"
         "      prop_crown=0.\n"
         "      if(any(canhfx.ne.0.).or.any(canqfx.ne.0.)) then\n"
         "         call tg_dist(ims,ime,kms,kme,jms,jme, &\n"
         "              i_st,i_en,j_st,j_en,k_st,k_en-1,dz8w, &\n"
         "              fire_heat_peak,fire_tg_ub,alfc,z_at_w,zs+z1can,prop_crown)\n"
         "      endif\n   end if")
    begin=source.index("            else if (fire_sfc_flx .eq. 1) then !Truncated Gaussian scheme")
    end=source.index("            else\n",begin)
    source=source[:begin]+"""            else if (fire_sfc_flx .eq. 1) then
               ! Corrected TG: each nonnegative fraction is a source in a mass layer.
               if(k.lt.k_en)then
                  rho_i=1./rho(i,k,j)
                  hfx(i,k,j)=(prop_heat(i,k,j)*grnhfx(i,j)+prop_crown(i,k,j)*canhfx(i,j))*cp_i
                  qfx(i,k,j)=(prop_heat(i,k,j)*grnqfx(i,j)+prop_crown(i,k,j)*canqfx(i,j))*xlv_i
                  rthfrten(i,k,j)=(c1h(k)*mu(i,j)+c2h(k))*rho_i*hfx(i,k,j)/dz8w(i,k,j)
                  rqvfrten(i,k,j)=(c1h(k)*mu(i,j)+c2h(k))*rho_i*qfx(i,k,j)/dz8w(i,k,j)
               endif

"""+source[end:]
    once("   DO j = j_st,j_en\n      DO k = k_st,k_en-1",
         "   IF (fire_sfc_flx.eq.0) THEN\n   DO j = j_st,j_en\n      DO k = k_st,k_en-1")
    end=source.index("END SUBROUTINE fire_tendency")
    source=source[:end]+"   END IF\n\n"+source[end:]
    once("REAL, INTENT(out), DIMENSION( i_st:i_en,k_st:k_en,j_st:j_en ) :: prop",
         "REAL, INTENT(out), DIMENSION( ims:ime,kms:kme,jms:jme ) :: prop")
    begin=source.index("   xia = (fire_tg_lb-fire_peak_hgt)")
    end=source.index("END SUBROUTINE tg_dist",begin)
    source=source[:begin]+"""   ! Corrected TG: integrate the original approximate CDF over complete
   ! mass layers. Normalize over the represented column and clipped support.
   prop=0.
   DO j=j_st,j_en
      DO i=i_st,i_en
         dz=0.
         DO k=k_st,k_en
            xia=(min(max(z_at_w(i,k,j)-zs(i,j),0.),fire_tg_ub)-fire_peak_hgt)/(0.5*fire_ext_depth)
            xib=(min(max(z_at_w(i,k+1,j)-zs(i,j),0.),fire_tg_ub)-fire_peak_hgt)/(0.5*fire_ext_depth)
            phi_a=0.5*(1.+tanh(acoef*xia+bcoef*(xia**3)))
            phi_b=0.5*(1.+tanh(acoef*xib+bcoef*(xib**3)))
            prop(i,k,j)=max(phi_b-phi_a,0.)
            dz=dz+prop(i,k,j)
         END DO
         if(.not.(dz.gt.0.)) call wrf_error_fatal('Corrected TG has no representable mass in the active column')
         DO k=k_st,k_en
            prop(i,k,j)=prop(i,k,j)/dz
         END DO
      END DO
   END DO

"""+source[end:]
    # Also repair the original smoke loop while the TG distribution is corrected.
    once("    do i=i_st,i_st", "    do i=i_st,i_en")
    return source


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("original",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--diff",type=Path)
    args=parser.parse_args()
    original=args.original.read_text(encoding="utf-8")
    corrected=correct(original)
    args.output.write_text(corrected,encoding="utf-8")
    if args.diff:
        args.diff.write_text("".join(difflib.unified_diff(original.splitlines(True),
            corrected.splitlines(True),fromfile="WRF-v4.7.1/module_fr_fire_atm.F",
            tofile="corrected-TG/module_fr_fire_atm.F",n=0)),encoding="utf-8")


if __name__=="__main__":
    main()
