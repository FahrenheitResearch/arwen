# Rounded MYNN mixing length

The ordinary `mynn_mixlength_default_columns` entry point used a second
implementation of the law already owned by `mynn_mym_length_column` during
initialization. Plain multiply/add expressions and device transcendental
functions in that copy differed from the rounded FP32 reference. With the
unchanged four-column driver inputs, the cold-start zonal tendency reached
3277 ULP against its 819 ULP regression limit. Other cold-start fields also
exceeded their historical maxima.

Both entry points now call the same existing helper. The committed WRF
mixing-length oracle is exact for `el` and `qkw`, replacing the old 384/1 ULP
allowance. Actual coupled driver inputs are exact against the independent
CPU leaf with production contraction and with contraction disabled. The
full driver retains its original limits and unresolved differences in other
leaves. This is an arithmetic-owner repair, not full WRF trajectory, column
conservation or forecast-skill certification.

PBL continuation identity advances to
`mynn-edmf-pbl-wrf-v4.6.1-v2-rounded-mixing-length`. An older checkpoint
cannot silently continue with the changed tendencies. The DMP sibling keeps
its historical source digest. Its source test compares every byte outside
the relocated ordinary mixing-length entry, and its numerical test compares
DMP on the arrays that DMP actually consumed. Whole-driver replay is kept
as a separate residual check because upstream cloud and PBL-height inputs
can differ before either DMP call.

On the measured device, mixing-length registers change from 48 to 58.
Local and shared storage remain zero and the maximum block size remains
1024 threads, above the existing 128-thread launch. The other five checked
MYNN entry points retain their resource attributes; scratch ownership does
not change.

OLR production ownership already carries `diag/olr` through checkpoints.
The last computed TOA longwave flux must survive until the next radiation
call, including intervening output frames. The obsolete test demanding a
rebuilt diagnostic is corrected. Missing TOA producers still omit OLR;
declared producers that return no flux still fail. Older checkpoints without
the optional diagnostic namespace retain their existing restore behavior.
