# Vertical velocity at 1 km and 3 km above ground

Render the geometric vertical velocity from a WRF-shaped history file:

```sh
gpuwm render /path/to/wrfout --products 1km_vertical_velocity,3km_vertical_velocity --out products
```

The same product names work through the hex render door after its native
history conversion, and directly through `rw_wrfbatch --products`.

Both products show upward-positive velocity in m s-1. Blue is descent,
orange/red is ascent, and values between -0.1 and +0.1 m s-1 are neutral.
The fixed scale is shared by both heights and every frame, with extensions
beyond -20 and +20 m s-1. Output follows the ordinary domain, product and
valid-day folders.

The importer destaggers native W onto mass levels and interpolates linearly
between adjacent levels using their geometric height above local terrain.
A target outside the column, or without a valid bracket, stays missing.
Missing W or geometry is reported as unavailable. Pressure velocity is a
different quantity and is never substituted. A request for these products
alone reads W and geometry without computing thermodynamic diagnostics.
