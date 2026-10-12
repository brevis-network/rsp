EIP-2537's own test vectors, from `assets/eip-2537/` in ethereum/EIPs at
faaa35fff856a109c40d291e612447fadc2095fe (the last commit to touch them), unmodified.

Every JSON file there is here except the two of valid MSM inputs, `msm_G1_bls.json` (3.6 MB) and
`msm_G2_bls.json` (6.5 MB). The `mul_*` files hold G1MUL's and G2MUL's vectors, which run on the
MSM precompiles as an MSM of one pair (see `test-vectors.md` there). MSMs of more pairs are covered
by the differential tests against revm's own backend in `tests/eip2537.rs`.
