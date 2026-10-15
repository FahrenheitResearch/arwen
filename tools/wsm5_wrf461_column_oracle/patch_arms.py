"""Write the defined arms of WRF v4.6.1 WSM5 from the build tree's own source.

Input: the build tree's preprocessed ``phys/module_mp_wsm5.f90`` (what its
gfortran compiled).  Output, one file per arm, each the input with only the
edits named below (every edit must match its anchor exactly once, or this
script refuses):

``stock``   the input plus the probe counters only (a recompile control:
            its words must equal the build tree's own object's).
``remap``   the corrected sedimentation search (lane 073's fix): the top
            search of ``nislfv_rain_plm`` tests the final arrival interface
            ``za(km+1)``, takes the cell above it when no interface is high
            enough, and never leaves ``kt`` below ``kb``.
``psevp``   the evaporation of melting snow ([HL A35] [RH83 A27]) takes
            WSM6's own line.  WRF's WSM5 forms it as ``psdep*work1(2)/
            work1(1)``, but ``psdep`` is set only where ``supcol > 0`` and is
            zero wherever this branch (``supcol < 0``) runs, so stock WSM5
            never evaporates melting snow.
``defined`` both edits: what WOOF's kernel computes.

Every arm carries two public per-column counters (the probe):
``wsm5_probe_remap(i)`` counts target cells whose remapped word, formed with
the STOCK search's ``kt`` (its own chain, carried beside the arm's), differs
from the arm's own word -- the value-changing fall-throughs; it is zero
by construction in the stock arm.  The stock search falls through on every
target cell whose top lies above every arrival interface up to ``za(km)``
(always the top cell of a falling column), but it changes a word only where
the cells it skips carry condensate.  ``wsm5_probe_psevp(i)`` counts levels
where the arm's ``psevp`` is non-zero after its clamp.  The probes are an
integer side effect plus float work into probe-only locals; no word the arm
outputs depends on them.

usage: python patch_arms.py TREE_MODULE_F90 OUT_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

COUNTERS_ANCHOR = """             rsloper3max,rslopes3max,rslopeg3max
"""
COUNTERS = """             rsloper3max,rslopes3max,rslopeg3max
   INTEGER, PUBLIC, SAVE :: wsm5_probe_remap(8192) = 0
   INTEGER, PUBLIC, SAVE :: wsm5_probe_psevp(8192) = 0
"""

DECL_ANCHOR = "      integer  i,k,n,m,kk,kb,kt,iter\n"
DECL = ("      integer  i,k,n,m,kk,kb,kt,iter,ktfound,ktst\n"
        "      real  qnst,tls,ths,qqds,dqls,dqhs,zsums,qsums\n")

# The stock search chain is carried beside the arm's own (ktst), so the
# probe can form the value the STOCK search gives for the same target cell.
CHAIN_INIT = """      kb=1
      kt=1
      intp : do k=1,km
             kb=max(kb-1,1)
             kt=max(kt-1,1)
"""
CHAIN_INIT_PROBED = """      kb=1
      kt=1
      ktst=1
      intp : do k=1,km
             kb=max(kb-1,1)
             kt=max(kt-1,1)
             ktst=max(ktst-1,1)
"""

SEARCH_STOCK = """               find_kt : do kk=kt,km
                         if( zi(k+1).le.za(kk) ) then
                           kt = kk
                           exit find_kt
                         else
                           cycle find_kt
                         endif
               enddo find_kt
               kt = kt - 1
"""

# Stock arm: the stock search, unchanged; the chain copy follows it exactly.
SEARCH_STOCK_PROBED = SEARCH_STOCK + """               ktst = kt
"""

# The corrected search: the last arrival interface za(km+1) is tested, the
# cell above it (qa(km+1) = 0, dza(km+1) = zi(km+1)-za(km+1)) is taken when
# none is high enough, and kt never falls below kb.  The stock chain runs
# beside it for the probe.
SEARCH_FIXED = """               ktfound = 0
               find_kt : do kk=kt,km+1
                         if( zi(k+1).le.za(kk) ) then
                           ktfound = kk
                           exit find_kt
                         endif
               enddo find_kt
               if( ktfound.eq.0 ) ktfound = km + 2
               kt = max(ktfound - 1, kb)
               find_kst : do kk=ktst,km
                         if( zi(k+1).le.za(kk) ) then
                           ktst = kk
                           exit find_kst
                         endif
               enddo find_kst
               ktst = ktst - 1
"""

# After the arm's own value for target cell k: the value the stock search
# would have given (WRF's own two formulas with the stock kt); a cell
# where the two words differ is a value-changing fall-through.
BRANCH_END = """                 qn(k) = qsum/zsum
               endif
               cycle intp
"""
BRANCH_END_PROBED = """                 qn(k) = qsum/zsum
               endif
               qnst = 0.0
               if( ktst.eq.kb ) then
                 tls=(zi(k)-za(kb))/dza(kb)
                 ths=(zi(k+1)-za(kb))/dza(kb)
                 qqds=0.5*(qpi(kb)-qmi(kb))
                 qnst = ((qqds*(ths*ths)+qmi(kb)*ths)-(qqds*(tls*tls)+qmi(kb)*tls))/(ths-tls)
               else if( ktst.gt.kb ) then
                 tls=(zi(k)-za(kb))/dza(kb)
                 qqds=0.5*(qpi(kb)-qmi(kb))
                 dqls = qa(kb)-(qqds*(tls*tls)+qmi(kb)*tls)
                 zsums  = (1.-tls)*dza(kb)
                 qsums  = dqls*dza(kb)
                 if( ktst-kb.gt.1 ) then
                 do m=kb+1,ktst-1
                   zsums = zsums + dza(m)
                   qsums = qsums + qa(m) * dza(m)
                 enddo
                 endif
                 ths=(zi(k+1)-za(ktst))/dza(ktst)
                 qqds=0.5*(qpi(ktst)-qmi(ktst))
                 dqhs=qqds*(ths*ths)+qmi(ktst)*ths
                 zsums  = zsums + ths*dza(ktst)
                 qsums  = qsums + dqhs*dza(ktst)
                 qnst = qsums/zsums
               endif
               if( qnst.ne.qn(k) ) wsm5_probe_remap(i) = wsm5_probe_remap(i) + 1
               cycle intp
"""

PSEVP_STOCK = """            if(qrs(i,k,2).gt.0..and.rh(i,k,1).lt.1.)                           &
              psevp(i,k) = psdep(i,k)*work1(i,k,2)/work1(i,k,1)

              psevp(i,k) = min(max(psevp(i,k),-qrs(i,k,2)*rdtcld),0.)
"""
PSEVP_STOCK_PROBED = """            if(qrs(i,k,2).gt.0..and.rh(i,k,1).lt.1.)                           &
              psevp(i,k) = psdep(i,k)*work1(i,k,2)/work1(i,k,1)

              psevp(i,k) = min(max(psevp(i,k),-qrs(i,k,2)*rdtcld),0.)
              if( psevp(i,k).ne.0. ) wsm5_probe_psevp(i) = wsm5_probe_psevp(i) + 1
"""
PSEVP_FIXED = """            if(qrs(i,k,2).gt.0..and.rh(i,k,1).lt.1.) then
              coeres = rslope2(i,k,2)*sqrt(rslope(i,k,2)*rslopeb(i,k,2))
              psevp(i,k) = (rh(i,k,1)-1.)*n0sfac(i,k)*(precs1*rslope2(i,k,2)   &
                           +precs2*work2(i,k)*coeres)/work1(i,k,1)
            endif

              psevp(i,k) = min(max(psevp(i,k),-qrs(i,k,2)*rdtcld),0.)
              if( psevp(i,k).ne.0. ) wsm5_probe_psevp(i) = wsm5_probe_psevp(i) + 1
"""

ARMS = {
    "stock": {"remap": False, "psevp": False},
    "remap": {"remap": True, "psevp": False},
    "psevp": {"remap": False, "psevp": True},
    "defined": {"remap": True, "psevp": True},
}


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"anchor for {what} matched {n} times, not once")
    return text.replace(old, new)


def make_arm(source: str, remap: bool, psevp: bool) -> str:
    text = _replace_once(source, COUNTERS_ANCHOR, COUNTERS, "probe counters")
    text = _replace_once(text, DECL_ANCHOR, DECL, "nislfv locals")
    text = _replace_once(text, CHAIN_INIT, CHAIN_INIT_PROBED, "search chain")
    text = _replace_once(text, SEARCH_STOCK,
                         SEARCH_FIXED if remap else SEARCH_STOCK_PROBED, "top search")
    text = _replace_once(text, BRANCH_END, BRANCH_END_PROBED, "remap probe")
    text = _replace_once(text, PSEVP_STOCK,
                         PSEVP_FIXED if psevp else PSEVP_STOCK_PROBED, "psevp")
    return text


def main(argv: list[str]) -> int:
    source = Path(argv[1]).read_text()
    out = Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for name, edits in ARMS.items():
        (out / f"module_mp_wsm5_{name}.f90").write_text(make_arm(source, **edits))
        print(f"wrote arm {name}: {edits}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
