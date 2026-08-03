"""S3-6h jet-decoupling fixture profile (CODE-GENERATED --
regenerate with tools/sase_smoke/extract_jet_profile.py; do
not hand-edit).

PROVENANCE: d03-box mean single-column state from Drew's
independent spun-up CPU WRF frame
wrfout_d02_1974-04-03_13_08_00 (3 km, 13:08 UTC 1974-04-03;
box |lat-39.7|<2.3, |lon+84|<2.9 = 28080 columns; the
truth-side reference of the S3-6h obs arbitration, controller
ledger 2026-07-21 ~01:0x; extraction recipe in the extractor
docstring).  Lowest 26 levels (top interface ~6.4 km):
strongly stable surface-based morning layer (theta 286.9 ->
295.8 K across the lowest 470 m) under the low-level jet
(17.64 m/s at 472 m) with 4.33 m/s at the 8.8 m first level;
E0 = QKE/2 is the frame's own MYNN turbulence energy (0.95
m2/s2 at the surface decaying through the shear layer), so
the fixture presents the closure with the REAL spun-up 13:08Z
state.  UST is the box-mean u* (the profile's surface stress
estimate); SPD10_WRF is the frame's own 10 m diagnostic (the
fixture's log-interpolated initial 10 m wind must land near
it).  Obs anchor: NCEI ISD 8-station 13Z mean 5.61 m/s
(out/obs-19740403/obs-arbitration.md)."""
import numpy as np

THETA = np.array([
    np.float64(286.91926356041193),
    np.float64(287.2767747971407),
    np.float64(287.5322869010121),
    np.float64(287.8996014391255),
    np.float64(288.47063039469924),
    np.float64(289.3404001523966),
    np.float64(290.5209080274968),
    np.float64(291.9318146914838),
    np.float64(293.4145036333307),
    np.float64(294.76263380892937),
    np.float64(295.83009898411245),
    np.float64(296.6336218222594),
    np.float64(297.2584648143192),
    np.float64(297.7909551843279),
    np.float64(298.3192192653645),
    np.float64(298.92665502542786),
    np.float64(299.76082680748397),
    np.float64(301.02952101726476),
    np.float64(302.67370455923924),
    np.float64(304.3959569643026),
    np.float64(305.83602517934946),
    np.float64(306.9351739888857),
    np.float64(308.0754583146837),
    np.float64(309.90880352248496),
    np.float64(312.0698203866638),
    np.float64(314.320473999719),
])

U = np.array([
    np.float64(-2.29681621553765),
    np.float64(-3.1415418257655583),
    np.float64(-3.4649227755331298),
    np.float64(-3.793881899750864),
    np.float64(-3.971131412696592),
    np.float64(-3.8722202228395495),
    np.float64(-3.2391091912807073),
    np.float64(-1.8892989632958133),
    np.float64(0.04265605117890193),
    np.float64(2.2120817637754997),
    np.float64(4.1448230166050894),
    np.float64(5.472270147481195),
    np.float64(6.211840552218959),
    np.float64(6.581582341532083),
    np.float64(6.752352450282825),
    np.float64(6.7951307150458335),
    np.float64(6.583096967828579),
    np.float64(6.016152189928837),
    np.float64(5.605384954040607),
    np.float64(6.243468899189884),
    np.float64(8.425378432417517),
    np.float64(11.823381373978238),
    np.float64(15.491192932193435),
    np.float64(18.972992495491972),
    np.float64(23.13457604996499),
    np.float64(26.23088463266351),
])

V = np.array([
    np.float64(3.666236254360402),
    np.float64(5.551814650966019),
    np.float64(6.448731812021756),
    np.float64(7.724968221541653),
    np.float64(9.180848346773715),
    np.float64(10.982763011766295),
    np.float64(13.084601077955332),
    np.float64(15.15758907703253),
    np.float64(16.722245483792406),
    np.float64(17.41176114778573),
    np.float64(17.141814257310667),
    np.float64(16.229014187117247),
    np.float64(15.08183532114382),
    np.float64(13.88629918028826),
    np.float64(12.68631314540181),
    np.float64(11.58262524083469),
    np.float64(10.78315340097134),
    np.float64(10.483525952620385),
    np.float64(10.700386728503426),
    np.float64(11.20954446503919),
    np.float64(11.7174890864269),
    np.float64(12.766375770697906),
    np.float64(14.680682937728713),
    np.float64(16.08848424085185),
    np.float64(15.627475917050642),
    np.float64(14.174592435661374),
])

THICK = np.array([
    np.float64(17.572513358540245),
    np.float64(20.897999366235943),
    np.float64(24.6395786662216),
    np.float64(29.223860568974153),
    np.float64(34.5169295521719),
    np.float64(40.97173072783057),
    np.float64(48.561155894297215),
    np.float64(57.67388659985295),
    np.float64(68.36402527850059),
    np.float64(81.15804182163768),
    np.float64(96.07988191260054),
    np.float64(113.45846146938791),
    np.float64(133.79175352873182),
    np.float64(157.577615339402),
    np.float64(185.40267263092122),
    np.float64(217.66482158490624),
    np.float64(255.36828847367136),
    np.float64(299.0073988482943),
    np.float64(349.98882296083053),
    np.float64(409.623237677509),
    np.float64(478.460681479596),
    np.float64(557.5051189120284),
    np.float64(648.2686589232545),
    np.float64(705.0805921510889),
    np.float64(691.8770302191286),
    np.float64(679.5851314731274),
])

E0 = np.array([
    np.float64(0.9538586862101941),
    np.float64(0.9562886666016988),
    np.float64(0.9080208385408991),
    np.float64(0.8701669434779006),
    np.float64(0.7881552970407005),
    np.float64(0.6674051235929019),
    np.float64(0.5171673549536931),
    np.float64(0.3604462433968401),
    np.float64(0.2302072310213892),
    np.float64(0.1483137465061587),
    np.float64(0.10659943376859128),
    np.float64(0.0818782488871165),
    np.float64(0.05760828714943983),
    np.float64(0.03277661501488597),
    np.float64(0.01606661218370689),
    np.float64(0.007625317075597476),
    np.float64(0.003118060523801762),
    np.float64(0.002707918062014448),
    np.float64(0.009914258885867009),
    np.float64(0.025283716543970543),
    np.float64(0.04003647810530756),
    np.float64(0.03947176062857227),
    np.float64(0.017012664356058107),
    np.float64(0.0034746218126453375),
    np.float64(0.0005770417951919787),
    np.float64(0.00014871097499891166),
])

UST = 0.45420628786087036
SPD10_WRF = 4.666643529089358
N_BOX_COLUMNS = 28080
