# SHA256-CDP: Cyclic Digit-sum Projection

**GPU-accelerated rainbow table implementation of the CDP structural analysis framework for SHA-256.**

Built on the findings of the CDP paper (DOI: [10.5281/zenodo.20627240](https://doi.org/10.5281/zenodo.20627240)), this repository provides a complete, working implementation of CDP-based rainbow chain tables with OpenCL build kernels and Vulkan compute query/verify pipelines targeting AMD RDNA4 hardware.

---

## What is CDP?

CDP (Cyclic Digit-sum Projection) is a structural analysis framework for SHA-256 that reveals previously undocumented mathematical properties of the hash function's output distribution.

The core observation: the hex-digit sum `W(H)` of any SHA-256 output, when iteratively re-hashed through `f(w) = W(SHA256(str(w)))`, converges deterministically into exactly **two closed cycles**:

```
C1: 476 ↔ 438          (2-cycle)
C2: 471 → 472 → 525 → 537 → 414 → 417 → 546 → 518 → 471  (8-cycle)
```

This cyclic structure, combined with a multi-component fingerprint `F(H) = (W, Wvec₁₆, cycle_entry, W₂–W₅, max_nibble, min_nibble)`, yields a **bijective mapping** over constrained input spaces — enabling O(1) preimage lookup via rainbow tables with zero reduction-function collisions.

Key proven properties:
- **Theorem 5.1** — Complement nibble sum invariant: `Σnibble(W₀) = 38` for all 256 complement pairs
- **Theorem 5.4** — Ergodic Markov property: `π_B = 17.00%`, independent of `K[i]`, `H₀`, and input class
- `W(H₀) = 502` — detectable structural signature of NIST initialization constants (+22.2 above equilibrium)
- Zero-collision bijection over 1.67M inputs across all tested input spaces

CDP does **not** break SHA-256. Preimage and collision resistance are unaffected. See Section 12 of the paper.

---

## Repository Structure

```
SHA256-CDP/
├── cdp_chain_builder.py   # Main entry point — build, crack, query CLI
├── cdp_query.comp         # Vulkan GLSL compute shader (query kernel)
├── cdp_verify.comp        # Vulkan GLSL compute shader (verify kernel)
├── vulkan_query.py        # Vulkan Python engine (ACO backend)
└── paper/
    └── cdp_v3.pdf         # CDP paper v3
```

---

## Requirements

- Python 3.10+
- PyOpenCL: `pip install pyopencl`
- NumPy: `pip install numpy`
- Vulkan (recommended): `pip install vulkan` + [Vulkan SDK](https://vulkan.lunarg.com/)
- AMD GPU with RDNA2+ (tested on RX 9070 XT / gfx1201)

> **Note:** Vulkan is required for correct query performance. Without it, the system falls back to CPU multiprocessing (significantly slower). The OpenCL query kernel has a known AMD PAL-LLVM optimizer bug on gfx1201 that produces incorrect results with `-cl-fast-relaxed-math`; the Vulkan ACO backend does not have this issue.

> **Windows users:** Add `python.exe`, `clinfo.exe`, and `explorer.exe` to Windows Defender Controlled Folder Access whitelist to prevent GPU memory access blocks during table operations.

---

## Installation

```bash
git clone https://github.com/JM00NJ/SHA256-CDP
cd SHA256-CDP
pip install pyopencl numpy vulkan
```

Verify GPU detection:
```bash
python cdp_chain_builder.py --validate
```

---

## Usage

### Build rainbow tables

```bash
# Single table (lowercase 8-char, chain_len=300,000)
python cdp_chain_builder.py --build --charset lower --length 8 --chain-len 300000 --output cdp_8.bin

# Multiple tables (higher coverage)
# n=1: 66.7%  n=2: 88.9%  n=3: 96.3%  n=5: 99.3%
python cdp_chain_builder.py --build-multi 3 --prefix cdp_8_bin --length 8 --chain-len 300000
```

### Crack a hash

```bash
# Single hash
python cdp_chain_builder.py --crack <sha256_hash> \
  --tables cdp_8_bin_1.bin,cdp_8_bin_2.bin,cdp_8_bin_3.bin \
  --length 8 --chain-len 300000

# Hash list (batch mode — tables loaded once)
python cdp_chain_builder.py --crack-list hashes.txt \
  --tables cdp_8_bin_1.bin,cdp_8_bin_2.bin,cdp_8_bin_3.bin \
  --length 8 --chain-len 300000
```

### End-to-end self-test

```bash
python cdp_chain_builder.py --self-test --length 7
```

### Validate GPU output

```bash
python cdp_chain_builder.py --validate
python cdp_chain_builder.py --validate-query <sha256_hash>
```

---

## Performance

Tested on **AMD Radeon RX 9070 XT** (gfx1201, 32 CUs), Windows 11, driver 3679.0 (PAL,LC):

| Operation | Throughput |
|---|---|
| Table build (ILP2 kernel) | ~2.5 GH/s |
| Query — first batch (shallow) | ~14 GH/s |
| Query — average across batches | ~5 GH/s |
| GPU verify (Vulkan ACO) | ~2s per 88k candidates |

**7-char lowercase, 3 tables (96.3% coverage):**
- Table size: 3 × 0.7 MB
- Build time: ~3 × 12s
- Crack time: ~43s per hash

**Kernel modes:**
- `--mode ilp2` — 2 chains/thread, interleaved SHA256 for ILP (default, fastest on AMD RDNA)
- `--mode vec4` — 4 chains/thread using `uint4` arithmetic
- `--mode scalar` — 1 chain/thread (baseline)

---

## Technical Notes

### Why Vulkan for query?

The AMD PAL-LLVM compiler backend used by OpenCL on Windows (gfx1201) has a documented optimizer bug: variable-start loops (`for step=pos; step<N`) produce incorrect results with `-cl-fast-relaxed-math`. The workaround (`-O0`) restores correctness but reduces throughput ~3-4×. The Vulkan ACO backend is a completely separate compiler pipeline and does not have this issue. Query and verify kernels use Vulkan; build kernels use OpenCL (unaffected by the bug).

### CDP bijective reduction

The standard PCG-seeded reduction function uses 64 bits of entropy from `digest[0:2]`. The CDP-injective reduction builds the seed from the full fingerprint `(W, Wvec₁₆, max_nibble, min_nibble)` — the same components proven injective over `SHA256(X)` by the CDP bijection theorem. This guarantees zero chain merges from the reduction function itself; remaining merges are birthday-paradox endpoint collisions (~33% per table, matching the expected 66.7% coverage).

### Binary table format

Tables use a compact binary format (`CDP1` magic, 39 bytes/entry) — approximately 3.4× smaller than JSON. Use `--convert` to convert existing JSON tables:

```bash
python cdp_chain_builder.py --convert table.json --output table.bin
```

---

## Charsets

| Name | Characters | Space (8-char) |
|---|---|---|
| `lower` | a–z (26) | 2.1 × 10¹¹ |
| `alnum` | a–z, 0–9 (36) | 2.8 × 10¹² |
| `full` | a–z, A–Z, 0–9, symbols (70) | 5.8 × 10¹⁴ |

---

## License

© 2026 Erenay Özkan (JM00NJ / Vesqer)

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-v3)** with the **Commons Clause**.

Under the Commons Clause, you may not sell this software or use it as part of a commercial product or service without explicit written permission from the author.

Open source use, research, and non-commercial applications are permitted under AGPL-v3 terms — modifications must be published under the same license.

For commercial licensing inquiries: [netacoding.com](https://netacoding.com)

See [LICENSE](LICENSE) for full terms.

---

## References
- Blog Post: https://netacoding.com/posts/cdp-sha256-structural-analysis/
- Erenay Özkan. *CDP: Cyclic Digit-sum Projection — Structural Analysis of SHA-256 Output Distribution, Ergodic Basin Pressure, and Input Class Fingerprinting.* v3, 2026. DOI: [10.5281/zenodo.20627240](https://doi.org/10.5281/zenodo.20627240)
- P. Oechslin. *Making a Faster Cryptanalytic Time-Memory Trade-Off.* CRYPTO 2003.
- M. Hellman. *A Cryptanalytic Time-Memory Trade-Off.* IEEE Trans. Inf. Theory, 1980.

---

*Built by [JM00NJ](https://github.com/JM00NJ) — [netacoding.com](https://netacoding.com)*
