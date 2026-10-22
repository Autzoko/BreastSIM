# torso.json - attribution and provenance

The torso mesh in `torso.json` is derived from **MakeHuman** assets, which the MakeHuman
team released under **CC0 1.0 Universal** (public domain dedication) in September 2020
(see `LICENSE.md` section C and `LICENSE.ASSETS.md` in the MakeHuman repository).
Only assets (mesh/target data files) were used; no MakeHuman program code (AGPL) was
used, copied or linked. Attribution is not legally required under CC0 but is given here.

- Source: https://github.com/makehumancommunity/makehuman
- Commit: `a8bc2d54ff0ac92e78ff71431b1023eda42bf482`
- Original copyright (released to CC0): Manuel Bastioni, Data Collection AB, Joel Palmius,
  Jonas Hauquier and the MakeHuman team.

## Files used (sha256)

- `makehuman/data/3dobjs/base.obj` - `8e761e6624b8f54536409135d1636da63b32486a90d4897f84e121d144f6fb4c`
- `makehuman/data/targets/macrodetails/caucasian-female-young.target` - `118379f6e8ba9266247fdb8788a20e1df40a239f97ced0b9905bcbcc74f6e820`
- `makehuman/data/targets/macrodetails/universal-female-young-averagemuscle-averageweight.target` - `4ba5396ddabda448ece15650a566fbebfbb10239256ccb201e8f883429e12249`
- `makehuman/data/targets/breast/female-young-averagemuscle-averageweight-mincup-averagefirmness.target` - `65e4ca16353f6c0e23b420cf5d9b31d8993943b76e6cd3a6961943371dd6bc2f`
- `makehuman/data/targets/breast/female-young-averagemuscle-averageweight-maxcup-averagefirmness.target` - `b5aaa1595cf4efcee69022fc60034d93ab6f6fc473e6f8e162e5526803b074bf`
- `makehuman/data/targets/breast/nipple-point-incr.target` - `2f9ac628bf9cf2b2e84bc2d93c81764f4d703405b49016bf6ae8b39901616781`
- `LICENSE.md` - `edd99571ca62698f78c943fd4fffb159a413c315fa0d4819d3cec5cadaf8b4f1`
- `LICENSE.ASSETS.md` - `f6089cba01cb570a24712b41ab8a586ccd3cc5ef53dc266ca50b95c288956d2c`

## Targets applied

- `caucasian-female-young.target` weight 1.0 (whole body)
- `universal-female-young-averagemuscle-averageweight.target` weight 1.0 (whole body)
- `female-young-averagemuscle-averageweight-mincup-averagefirmness.target` weight 1.0 (subject's right side only)
- Opposite breast: MakeHuman default (average cup, average firmness; no breast target).
- `maxcup-averagefirmness` and `nipple-point-incr` were read only to locate the breast
  region and the nipple (not applied).

## Processing (by `src/scripts/build_torso_asset.py`)

1. Kept only the `body` face group of `base.obj` (removed helper-* proxies: eyes, eyelashes,
   teeth, tongue, hair, genital, skirt, tights; and all joint-* helper cubes, which were used
   only as landmarks for cropping).
2. Phantom-side (right) breast flattened: vertices of the breast region (where the
   cup-size targets differ by > 10% of their max, plus a 45 mm
   disc around the nipple, dilated 1 ring, holes filled) were moved along the
   local chest normal onto a thin-plate-spline height field fitted to a 3-ring band of
   undisturbed skin around the region (vertex tangential positions kept).
3. 2 level(s) of Catmull-Clark subdivision.
4. Cropped to a torso (plane cuts below the neck, at the hips and across the upper arms),
   boundary vertices snapped to the cut planes, quads triangulated, normals recomputed.
5. Units converted from MakeHuman decimetres to millimetres (x100).

Result: 23474 vertices, 46368 triangles, bbox (mm) [[-241.6, 104.0, -63.6], [241.6, 524.4, 159.1]].

The derived `torso.json` is likewise offered under CC0 1.0.
