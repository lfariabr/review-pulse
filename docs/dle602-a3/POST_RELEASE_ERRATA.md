# ReviewPulse v3.0.0 — post-release documentation errata

**Recorded:** 2026-08-15 AEST
**Immutable release:** `v3.0.0` at `c2ee52ab4c4415eb2ddc4223500040147b2a92b9`

This note explains why the post-release documentation commit does not move the tag or regenerate
the already recorded ZIPs. The release artifacts remain immutable; their `PACKAGE_MANIFEST.json`
files and the GitHub release are the authoritative integrity records.

## Corrections

- The final archives are 54,048,531 bytes (lightweight) and 301,254,100 bytes (all-models).
- Their SHA-256 digests are `935aabe3470082d0ecbb92596b60e65203e326caab91e46c3deb2609840ca9b9`
  and `0c773f444de2c1459d488ae4ab2c534c3025cbd5b445b530812421705ea7c17d`, respectively.
- The package manifests record `c2ee52a`; an older embedded `SUBMISSION_README.md` line referred
  to `7adb3ca`. The source documentation is corrected for future builds, but the published
  `v3.0.0` bytes and digests are not rewritten.
- The 11 package-test skips are three legacy Amazon corpus checks, six SemEval sample-provenance
  checks and two Git-metadata checks. The SemEval-derived predictions are excluded for licensing
  and provenance reasons, not file size.
- Python 3.12.10 is the project reference environment. The dependency constraints do not pin the
  interpreter; the Copilot verification additionally ran under Python 3.14.3.

## Scope

No model weights, application code, release tag or published checksum was changed by this erratum.
It corrects wording and status records only. The reproducibility report records the independently
verified package results; the historical checklist and QA documents retain their original evidence
with a final disposition note.
