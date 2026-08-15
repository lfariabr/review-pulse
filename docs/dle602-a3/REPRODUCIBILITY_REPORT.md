# ReviewPulse v3.0.0 (DLE602 A3) — Reproducibility & Usability Report

**Verified**: 2026-08-15 | **Reference Python**: 3.12.10 | **Verification Python**: 3.14.3 | **Commit**: c2ee52ab4c4415eb2ddc4223500040147b2a92b9

## Executive Summary

✅ **Build reproducibility**: Bit-identical reproduction achieved  
✅ **Package verification**: Both archives install, pass the recorded suite and pass the documented smoke checks
✅ **Release traceability**: The tag, manifests, archive sizes and published digests agree

---

## Part 1: Build Reproducibility

### Methodology

1. Cloned repository fresh from https://github.com/lfariabr/review-pulse
2. Checked out tag `v3.0.0` at commit `c2ee52ab4c4415eb2ddc4223500040147b2a92b9`
3. Ran `git lfs pull --include="*"` to fully resolve all LFS pointers
4. Rebuilt both archives using `python3 scripts/build_a3_package.py`
5. Compared byte sizes and SHA-256 digests against originals in `dist/v3.0.0/`

### Results

| Archive | Mode | Bytes | SHA-256 Digest |
|---------|------|-------|----------------|
| **Rebuilt** | lightweight | 54,048,531 | `935aabe3470082d0ecbb92596b60e65203e326caab91e46c3deb2609840ca9b9` |
| **Original** | lightweight | 54,048,531 | `935aabe3470082d0ecbb92596b60e65203e326caab91e46c3deb2609840ca9b9` |
| **Rebuilt** | all-models | 301,254,100 | `0c773f444de2c1459d488ae4ab2c534c3025cbd5b445b530812421705ea7c17d` |
| **Original** | all-models | 301,254,100 | `0c773f444de2c1459d488ae4ab2c534c3025cbd5b445b530812421705ea7c17d` |

### Verdict: ✅ **PASS — Deterministic Build Verified**

**Both archives are byte-for-byte identical.** The builder's determinism claims hold:
- Entry timestamps are reproducibly fixed to the source commit time
- Archive member paths are consistently sorted
- Two rebuilds under the documented verification procedure produced identical output

---

## Part 2: Package Usability & Security

### Lightweight Archive

**Setup & Installation**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -c constraints-a3.txt
```
✅ Clean installation; all dependencies resolve without conflict.

**Test Results**:
```text
355 passed, 11 skipped, 43 warnings in 63.28s
```
✅ All tests pass. Skips are intentional (documented below).

**Manifest Verification**:
- `source_commit`: `c2ee52ab4c4415eb2ddc4223500040147b2a92b9` ✅
- `artifact_mode`: `lightweight` ✅

**Content Security Scan**:
- ✅ No `.git`, `.venv`, `__pycache__`, `.pytest_cache`, `.env`, `.DS_Store`
- ✅ No `*.xml` files, `predictions.csv`, or LFS pointer files
- ✅ `data/semeval2014/` contains only `.gitkeep` (licensed data excluded)

### All-Models Archive

**Setup & Installation**: ✅ Identical to lightweight; all dependencies clean.

**Test Results**:
```text
355 passed, 11 skipped, 43 warnings in 66.95s
```
✅ Identical test outcomes. All model files present and functional.

**Manifest Verification**: ✅ Same source commit; `artifact_mode: all` correct.

**Content Security Scan**: ✅ All checks pass identically to lightweight.

### Verdict: ✅ **PASS — Reproducibility and package checks**

The checks found no excluded files, unresolved LFS pointers or package-level installation failures. This
is evidence for the recorded submission workflow, not a general security or production-readiness claim.

---

## Known-Intentional Behaviour (Not Defects)

### Test Skips (11 Total)

1. **Legacy Amazon checks (3 skips)**: These checks require the non-redistributed ISY503 `.review` corpus.
2. **Sample provenance tests (6 skips)**: These require `predictions.csv`, which contains row-level SemEval-derived
   text and gold labels and is excluded for licensing/provenance reasons.
3. **Package builder tests (2 skips)**: These require Git metadata (`.git/`), which ZIP archives do not carry.

### Other Intentional Absences

- **SemEval XML**: Licensed academic data. Deliberately excluded. Parser raises a controlled `FileNotFoundError` when accessed.
- **LFS pointer strings in source code**: Found only in `scripts/build_a3_package.py` and `tests/test_build_a3_package.py` as validation constants. No LFS pointer files (`version https://git-lfs.github.com/spec/v1`) are present in extracted archives.

### Python Version Note

- **Reference environment**: Python 3.12.10, as recorded in the project baseline.
- **Verification run**: Python 3.14.3.
- **Dependency constraints**: `constraints-a3.txt` pins package versions, not the Python interpreter.
- **Observed compatibility**: The verification environment passed the recorded package checks without changes.

---

## Differences from Expected Values

**Result: None.** All observed values match expected values exactly:
- ✅ Lightweight SHA-256 matches
- ✅ All-models SHA-256 matches
- ✅ Lightweight test count: 355 passed, 11 skipped
- ✅ All-models test count: 355 passed, 11 skipped
- ✅ Both manifests record correct source commit `c2ee52a`

---

## Conclusion

**ReviewPulse v3.0.0 (DLE602 A3) submission archives meet all reproducibility and usability criteria:**

1. **Reproducible**: Fresh clone + rebuild produces bit-identical archives
2. **Secure**: All forbidden content absent; manifests correct
3. **Functional**: All tests pass; all dependencies resolve cleanly
4. **Portable**: No environment-specific artifacts; verified on Python 3.14.3

**Recommendation**: The archives passed the recorded reproducibility and usability checks. Final submission
acceptance remains an academic delivery decision, separate from this technical verification.

---

## Appendix: Verification Commands

```bash
# Rebuild lightweight
python3 scripts/build_a3_package.py --artifact-mode lightweight --output lw.zip

# Rebuild all-models
python3 scripts/build_a3_package.py --artifact-mode all --output all.zip

# Verify checksums
shasum -a 256 lw.zip all.zip

# Extract and test
mkdir test && cd test && unzip -q ../lw.zip
cd ReviewPulse-v3.0.0
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt -c constraints-a3.txt
python3 -m pytest -q
```
