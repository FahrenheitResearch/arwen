"""CPU reference of the WRF v4.7.1 UW moist-turbulence PBL (bl_pbl_physics=9).

A literal Python transcription of module_bl_camuwpbl_driver.F and the CAM
modules it calls, in binary64 Python floats, graded bit for bit against the
gfortran oracle in tools/uwpbl_wrf471_oracle.  Its transcendentals are the
host C library's (math.exp/log/pow/cos/acos): on a glibc 2.43 x86-64 host
those are the oracle's own functions, so the reference is exact there and
only there; elsewhere it is a readable reference whose last bits follow the
host.  The GPU port is gpuwm/core/uwpbl.py.
"""
