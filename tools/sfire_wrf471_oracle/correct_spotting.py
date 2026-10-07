"""Label finite-state fixes while retaining an immutable original control."""
from pathlib import Path
import argparse
import hashlib
import json

from tools.sfire_wrf471_oracle.extract_spotting import SOURCE_SHA256


def correct(source, destination):
    raw = Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Spotting corrections require pinned original WRF source")
    replacements = [
        (b"(prop%p_effd/1000.0_dp)**2",b"(prop%p_effd/1000.0_dp)**3"),
        (b"yp_m0 = yp(pp) / mf(FLOOR(xp(pp)),FLOOR(yp(pp)))",
         b"yp_m0 = yp(pp) / (grid%rdy*grid%msfty(FLOOR(xp(pp)),FLOOR(yp(pp))))"),
        (b"yp1 = yp_m1 * mf(FLOOR(xp(pp)),FLOOR(yp(pp)))",
         b"yp1 = yp_m1 * (grid%rdy*grid%msfty(FLOOR(xp(pp)),FLOOR(yp(pp))))"),
        (b"yp2 = yp_m2 * mf(FLOOR(xp1),FLOOR(yp1))",
         b"yp2 = yp_m2 * (grid%rdy*grid%msfty(FLOOR(xp1),FLOOR(yp1)))"),
        (b"mask=(RelPot(ifps:ifpe,jfps:jfpe) >= PotThr), FIELD=0)",
         b"mask=(RelPot(ifps:ifpe,jfps:jfpe)>LimPot .AND. RelPot(ifps:ifpe,jfps:jfpe)>=PotThr), FIELD=0)"),
        (b"mask=(fs_fire_ROSdt(ifps:ifpe,jfps:jfpe) > ZERO_dp), FIELD=0)",
         b"mask=(RelPot(ifps:ifpe,jfps:jfpe)>LimPot), FIELD=0)"),
        (b"            seeds = [(( release_i(ii) * release_j(ii) * levrand_seed * kk, &\n                        kk=1, fs_gen_levels), &\n                        ii=1, SIZE(release_n))]",
         b"            seeds = [(INT(release_i(MOD(ii-1,SIZE(release_n)*fs_gen_levels)/fs_gen_levels+1) * &\n              release_j(MOD(ii-1,SIZE(release_n)*fs_gen_levels)/fs_gen_levels+1) * levrand_seed * &\n              (MOD(MOD(ii-1,SIZE(release_n)*fs_gen_levels),fs_gen_levels)+1)),ii=1,nseeds)]"),
        (b"th_phy(ims:, ks:, jms:) = grid%t_2", b"th_phy(ims:, ks:k_end, jms:) = grid%t_2"),
        (b"th_phy = th_phy/(dp1 + Rv/Rd * grid%moist", b"th_phy(ims:,ks:k_end,jms:) = th_phy(ims:,ks:k_end,jms:)/(dp1 + Rv/Rd * grid%moist"),
        (b"            DO kk = 1, ke\n                z_at_w(:,kk,:)= z_at_w(:,kk,:) - z_at_w(:,1,:)",
         b"            w1 = z_at_w(:,1,:)\n            DO kk = 1, ke\n                z_at_w(:,kk,:)= z_at_w(:,kk,:) - w1"),
        (b"bounds_mask = ((FLOOR(fs_p_x) >= is) .OR. &\n                   (FLOOR(fs_p_x) <= ie) .OR. &\n                   (FLOOR(fs_p_y) >= js) .OR. &",
         b"bounds_mask = ((FLOOR(fs_p_x) >= is) .AND. &\n                   (FLOOR(fs_p_x) <= ie) .AND. &\n                   (FLOOR(fs_p_y) >= js) .AND. &"),
        (b"            IF (IEEE_IS_NAN(fs_p_prop(pp)%p_tvel)) THEN",
         b"            IF (IEEE_IS_NAN(fs_p_prop(pp)%p_tvel) .OR. IEEE_IS_NAN(fs_p_prop(pp)%p_mass) .OR. &\n                IEEE_IS_NAN(fs_p_prop(pp)%p_diam) .OR. IEEE_IS_NAN(fs_p_prop(pp)%p_effd) .OR. &\n                IEEE_IS_NAN(fs_p_prop(pp)%p_temp) .OR. fs_p_prop(pp)%p_mass<=0. .OR. &\n                fs_p_prop(pp)%p_diam<=0. .OR. fs_p_prop(pp)%p_effd<=0.) THEN"),
        (b"                zout(pp) = ZERO_dp\n\n                ! WRITE (msg,'(3(i8,1x),4(f12.6,1x))')",
         b"                zout(pp) = ZERO_dp\n                xout(pp) = ZERO_dp\n                yout(pp) = ZERO_dp\n\n                ! WRITE (msg,'(3(i8,1x),4(f12.6,1x))')"),
        (b"                    np_lkhd(1:ndep_total) = np_lkhd(1:ndep_total)/MAXVAL(np_lkhd)",
         b"                    IF(MAXVAL(np_lkhd)>ZERO_dp)THEN\n                      np_lkhd(1:ndep_total) = np_lkhd(1:ndep_total)/MAXVAL(np_lkhd)\n                    ELSE\n                      np_lkhd=ZERO_dp\n                    ENDIF"),
    ]
    original = raw
    for before, after in replacements:
        if raw.count(before) != 1:
            raise ValueError("Spotting correction no longer matches one original source statement")
        raw = raw.replace(before, after)
    Path(destination).write_bytes(raw)
    receipt = dict(original_sha256=hashlib.sha256(original).hexdigest(), corrected_sha256=hashlib.sha256(raw).hexdigest(),
        changes=[dict(before=a.decode(),after=b.decode()) for a,b in replacements])
    Path(destination).with_suffix(".json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("destination")
    a=p.parse_args()
    correct(a.source,a.destination)
