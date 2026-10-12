EIP-2537's own test vectors, from `assets/eip-2537/` in ethereum/EIPs at
faaa35fff856a109c40d291e612447fadc2095fe (the last commit to touch them), unmodified.

Every file is here except the two passing MSM files, `msm_G1_bls.json` (3.6 MB) and
`msm_G2_bls.json` (6.5 MB); the rejected MSM inputs are kept. MSM results are covered instead by
the differential tests against revm's own backend in `tests/eip2537.rs`.
