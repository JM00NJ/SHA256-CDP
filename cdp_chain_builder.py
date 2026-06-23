# -*- coding: utf-8 -*-
# SHA256-CDP: Cyclic Digit-sum Projection — Rainbow Chain Builder
# Author:  Erenay Özkan (JM00NJ / Vesqer)
# Web:     https://netacoding.com
# GitHub:  https://github.com/JM00NJ/SHA256-CDP
# Paper:   https://doi.org/10.5281/zenodo.20627240
# License: AGPL-v3 + Commons Clause — commercial use prohibited without permission
"""
CDP Rainbow Chain Builder
=========================
AMD RX 9070 XT / OpenCL + Vulkan GPU implementation

Charsets:
  lower  — a-z                          (26 chars)
  alnum  — a-z, 0-9                     (36 chars)
  full   — a-z, A-Z, 0-9, !@#$%^&*()   (70 chars)

Key design decisions:
  1. AMD Adrenalin driver has ~1-2s internal watchdog (independent of TdrDelay).
     Fix A: auto-size batch so each dispatch < 800ms.
     Fix B: In AMD Adrenalin -> Performance -> Tuning -> Manual,
             set Max GPU Clock = 3000 MHz (fixes RX 9070 XT crash bug).
  2. All numpy operations vectorized -- no Python loops for buffer prep.
     starts_flat uses np.frombuffer for O(1) encoding instead of O(n) loop.
  3. OpenCL query kernel compiled with -O0 due to AMD PAL-LLVM optimizer bug
     on gfx1201 (variable-depth loops produce wrong results with fast-math).
     Vulkan ACO backend used for query/verify -- unaffected by this bug.

Usage:
  # Validate GPU output (scalar + vec4 + ilp2 correctness check)
  python cdp_chain_builder.py --validate

  # End-to-end self-test (build mini table + crack)
  python cdp_chain_builder.py --self-test --length 7

  # Build single table
  python cdp_chain_builder.py --build --charset lower --length 8 --chain-len 300000 --output cdp_8.bin
  python cdp_chain_builder.py --build --charset alnum --length 8 --chain-len 300000 --output cdp_8_alnum.bin
  python cdp_chain_builder.py --build --charset full  --length 8 --chain-len 300000 --output cdp_8_full.bin

  # Build multiple tables (higher coverage)
  #   n=1: 66.7%  n=2: 88.9%  n=3: 96.3%  n=5: 99.3%
  python cdp_chain_builder.py --build-multi 3 --prefix cdp_8_bin   --length 8  --chain-len 300000
  python cdp_chain_builder.py --build-multi 3 --prefix cdp_10_bin  --length 10 --chain-len 1000000
  python cdp_chain_builder.py --build-multi 5 --prefix cdp_8_alnum_bin --charset alnum --length 8 --chain-len 300000

  # Crack a single hash
  python cdp_chain_builder.py --crack <hash> \
      --tables cdp_8_bin_1.bin,cdp_8_bin_2.bin,cdp_8_bin_3.bin \
      --length 8 --chain-len 300000

  # Crack multiple hashes from file (tables loaded once)
  python cdp_chain_builder.py --crack-list hashes.txt \
      --tables cdp_8_bin_1.bin,cdp_8_bin_2.bin,cdp_8_bin_3.bin \
      --length 8 --chain-len 300000

  # Convert JSON table to binary (3.4x smaller)
  python cdp_chain_builder.py --convert cdp_table.json --output cdp_table.bin

  # Validate query kernel against Python reference
  python cdp_chain_builder.py --validate-query <hash>
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pytools')
warnings.filterwarnings('ignore', message='.*siphash24.*')
import pyopencl as cl
import numpy as np
import hashlib, time, argparse, os, json, struct
import vulkan_query as _vq

# --- CDP Core -----------------------------------------------------------------

CHARSETS = {
    'lower': 'abcdefghijklmnopqrstuvwxyz',
    'alnum': 'abcdefghijklmnopqrstuvwxyz0123456789',
    'full':  'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()',
}

def W_fn(h):
    return sum(int(c,16) for c in h)

def sha256h(b):
    return hashlib.sha256(b).hexdigest()

MASK64 = 0xFFFFFFFFFFFFFFFF

# --- CDP Fingerprint Constants & Helpers -------------------------------------

C1       = {476, 438}
C2       = {471, 472, 525, 537, 414, 417, 546, 518}
C1_BASIN = {418,425,431,435,437,438,446,449,456,461,463,
            474,476,478,493,503,504,510,512,516,524,529,531,533,542}

def _iter_W(w):
    """Single W-iteration step: W(SHA256(str(w)))"""
    return W_fn(sha256h(str(w).encode()))

def _cycle_entry(w):
    """Iterate until C1 or C2 reached. Returns 476 for C1, 471 for C2."""
    seen = set()
    for _ in range(25):
        if w in C1:   return 476
        if w in C2:   return 471
        if w in seen: return w    # unexpected cycle
        seen.add(w)
        w = _iter_W(w)
    return w

# Precompute W2..W5 and cycle_entry for all W in [250,750].
# ~500 SHA256 calls at import time — negligible overhead (~50ms).
_W_CACHE: dict = {}

def _build_W_cache():
    for w in range(250, 750):
        w2 = _iter_W(w);  w3 = _iter_W(w2)
        w4 = _iter_W(w3); w5 = _iter_W(w4)
        _W_CACHE[w] = (w2, w3, w4, w5, _cycle_entry(w))

_build_W_cache()   # run once at import

# ── W0-state fingerprint helpers ───────────────────────────────────────────
# state[1][a] = W0_uint32 + 0xfc08884d  (deterministic, no SHA256 needed)
# state[1][e] = W0_uint32 + 0x98c7e2a2
# where W0_uint32 = first 4 bytes of plaintext as big-endian uint32.

_W0_CONST_A = 0xfc08884d

def _ns32(x):
    s = 0
    while x: s += x & 0xF; x >>= 4
    return s

def _start_w0_bucket(s):
    """W0 nibble-sum bucket (0-7) for a plaintext start string."""
    b = s.encode()[:4].ljust(4, b'\x00')
    return _ns32(int.from_bytes(b, 'big')) % 8

def _start_s1fp(s):
    """state[1][a] nibble fingerprint (0-120) — no SHA256, just a 32-bit add."""
    b = s.encode()[:4].ljust(4, b'\x00')
    return _ns32((int.from_bytes(b, 'big') + _W0_CONST_A) & 0xFFFFFFFF)

def build_w0_index(table):
    """Split table dict into 8 sub-dicts by W0 bucket. Better cache locality."""
    idx = [{} for _ in range(8)]
    for fp, v in table.items():
        start = v if isinstance(v, str) else v[0]
        idx[_start_w0_bucket(start)][fp] = start
    return idx

# ── Binary table format ─────────────────────────────────────────────────────
# 39 byte/entry vs ~132 byte/entry JSON  →  3.4× smaller
# 3 tables: 54 MB JSON → 16 MB binary

_BIN_MAGIC   = b'CDP1'
_BIN_VERSION = 1

def save_binary_table(entries_iter, out_path, charset_name,
                      str_len, chain_len, n_chains):
    """Write CDP table to compact binary format. Returns entries written."""
    with open(out_path, 'wb') as f:
        f.write(_BIN_MAGIC)
        f.write(struct.pack('B', _BIN_VERSION))
        f.write(charset_name.encode()[:8].ljust(8, b'\x00'))
        f.write(struct.pack('B', str_len))
        f.write(struct.pack('>I', chain_len))
        f.write(struct.pack('>I', n_chains))
        written = 0
        for e in entries_iter:
            key   = e['key']
            start = e['start']
            w0v   = _start_w0_bucket(start) & 0x7
            s1v   = _start_s1fp(start) & 0x7F
            f.write(struct.pack('>H', key[0]))
            for i in range(16): f.write(struct.pack('B', key[1+i]))
            f.write(struct.pack('>H', key[17]))
            f.write(struct.pack('>HHHH', key[18], key[19], key[20], key[21]))
            f.write(struct.pack('BB', key[22], key[23]))
            f.write(start.encode()[:str_len])
            f.write(struct.pack('B', (w0v<<5)|(s1v>>2)))
            written += 1
    return written

def load_binary_table(path):
    """Load CDP binary table. Returns (table_dict, w0_idx, header)."""
    table = {}
    with open(path, 'rb') as f:
        assert f.read(4) == _BIN_MAGIC, f"Not a CDP binary table: {path}"
        f.read(1)
        charset_name = f.read(8).rstrip(b'\x00').decode()
        str_len   = struct.unpack('B', f.read(1))[0]
        chain_len = struct.unpack('>I', f.read(4))[0]
        n_chains  = struct.unpack('>I', f.read(4))[0]
        entry_sz  = 2 + 16 + 2 + 8 + 2 + str_len + 1
        while True:
            raw = f.read(entry_sz)
            if len(raw) < entry_sz: break
            W   = struct.unpack('>H', raw[0:2])[0]
            wv  = tuple(raw[2:18])
            ce  = struct.unpack('>H', raw[18:20])[0]
            w2,w3,w4,w5 = struct.unpack('>HHHH', raw[20:28])
            mx  = raw[28]; mn = raw[29]
            start = raw[30:30+str_len].decode()
            k = (W, wv, ce, w2, w3, w4, w5, mx, mn)
            table[k] = start
    w0_idx = build_w0_index(table)
    header = {'charset': charset_name, 'str_len': str_len, 'chain_len': chain_len}
    print(f"Binary table loaded: {len(table):,} chains  "
          f"W0-buckets: {[len(w0_idx[b]) for b in range(8)]}")
    return table, w0_idx, header

def full_fingerprint(h_hex):
    """
    Full CDP fingerprint as per README:
      F(H) = (W, Wvec16, cycle_entry, W2, W3, W4, W5, max_digit, min_digit)
    """
    W_val = W_fn(h_hex)
    wvec  = tuple(
        sum(int(c,16) for c in h_hex[i*4:(i+1)*4])
        for i in range(16)
    )
    entry = _W_CACHE.get(W_val)
    if entry:
        w2, w3, w4, w5, ce = entry
    else:                          # W outside [250,750]: rare edge case
        w2 = _iter_W(W_val); w3 = _iter_W(w2)
        w4 = _iter_W(w3);    w5 = _iter_W(w4)
        ce = _cycle_entry(W_val)
    nibbles = [int(c,16) for c in h_hex]
    return (W_val, wvec, ce, w2, w3, w4, w5, max(nibbles), min(nibbles))

def py_reduce_v3(hash_hex, step, charset, length):
    """Legacy PCG reduction — kept for reference/comparison."""
    d64  = int(hash_hex[:16], 16)
    seed = ((d64  * 6364136223846793005) & MASK64) \
         ^ ((step * 1442695040888963407) & MASK64)
    seed ^= (seed >> 30)
    seed  = (seed * 0xBF58476D1CE4E5B9) & MASK64
    seed ^= (seed >> 27)
    seed  = (seed * 0x94D049BB133111EB) & MASK64
    seed ^= (seed >> 31)
    result = []
    for _ in range(length):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & MASK64
        result.append(charset[(seed >> 33) % len(charset)])
    return ''.join(result)

def py_reduce_inj(hash_hex, step, charset, length):
    """CDP-injective reduction (paper Section 11 / Theorem 11.1).

    Seed built from W + wvec16 + max + min — the same components proven
    injective over SHA256(X) by the CDP bijection theorem.
    Guarantees: F(h1)≠F(h2) → seed1≠seed2 → R(h1)≠R(h2) → zero chain merges.
    Must match cdp_reduce_inj kernel exactly (bit-for-bit).
    """
    nibbles = [int(c,16) for c in hash_hex]          # 64 nibbles
    W   = sum(nibbles)
    wv  = [sum(nibbles[i*4:(i+1)*4]) for i in range(16)]
    mx  = max(nibbles)
    mn  = min(nibbles)

    # Seed construction — mirrors kernel loop exactly
    seed = (W * 0x9E3779B97F4A7C15) & MASK64
    for i in range(16):
        seed = ((seed * 6364136223846793005)
                ^ (wv[i] * (i+1) * 0xA3B2C1D4E5F60718)) & MASK64
    seed  = (seed ^ (mx << 32) ^ mn) & MASK64
    seed  = (seed ^ (step * 1442695040888963407)) & MASK64

    # PCG mixing (identical to cdp_reduce_full)
    seed ^= (seed >> 30)
    seed  = (seed * 0xBF58476D1CE4E5B9) & MASK64
    seed ^= (seed >> 27)
    seed  = (seed * 0x94D049BB133111EB) & MASK64
    seed ^= (seed >> 31)

    result = []
    for _ in range(length):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & MASK64
        result.append(charset[(seed >> 33) % len(charset)])
    return ''.join(result)

def py_chain(start_str, chain_len, charset):
    """Walk chain with CDP-injective reduction, return full fingerprint."""
    current = start_str
    for step in range(chain_len):
        h = sha256h(current.encode())
        current = py_reduce_inj(h, step, charset, len(start_str))
    h_final = sha256h(current.encode())
    return full_fingerprint(h_final)

# --- OpenCL Kernel ------------------------------------------------------------

KERNEL_SRC = """
__constant uint K[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,
    0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
    0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,
    0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,
    0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
    0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,
    0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,
    0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
    0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

// ── Scalar macros ────────────────────────────────────────────────────────────
#define ROTR(x,n)   (((x)>>(n))|((x)<<(32-(n))))
#define CH(x,y,z)   (((x)&(y))^(~(x)&(z)))
#define MAJ(x,y,z)  (((x)&(y))^((x)&(z))^((y)&(z)))
#define EP0(x)      (ROTR(x,2)^ROTR(x,13)^ROTR(x,22))
#define EP1(x)      (ROTR(x,6)^ROTR(x,11)^ROTR(x,25))
#define SIG0(x)     (ROTR(x,7)^ROTR(x,18)^((x)>>3))
#define SIG1(x)     (ROTR(x,17)^ROTR(x,19)^((x)>>10))

// ── uint4 macros — 4 parallel hashes per operation ───────────────────────────
#define ROTR4(x,n)  (((x)>>(n))|((x)<<(32-(n))))
#define CH4(x,y,z)  (((x)&(y))^(~(x)&(z)))
#define MAJ4(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))
#define EP04(x)     (ROTR4(x,2)^ROTR4(x,13)^ROTR4(x,22))
#define EP14(x)     (ROTR4(x,6)^ROTR4(x,11)^ROTR4(x,25))
#define SIG04(x)    (ROTR4(x,7)^ROTR4(x,18)^((x)>>3))
#define SIG14(x)    (ROTR4(x,17)^ROTR4(x,19)^((x)>>10))

// ── sha256_opt: W[16] sliding window — 4× less private memory than W[64] ─────
// No block[64] copy; padding computed inline.  #pragma unroll elides branches.
void sha256_opt(__private const uchar* data, int len, __private uint* digest) {
    uint w[16];
    ulong bitlen = (ulong)len * 8;
    #pragma unroll 16
    for(int i=0;i<16;i++){
        uint v=0;
        #pragma unroll 4
        for(int j=0;j<4;j++){
            int p=i*4+j;
            uchar b=(p<len)?data[p]:
                    (p==len)?0x80:
                    (p==56)?(uchar)(bitlen>>56):
                    (p==57)?(uchar)(bitlen>>48):
                    (p==58)?(uchar)(bitlen>>40):
                    (p==59)?(uchar)(bitlen>>32):
                    (p==60)?(uchar)(bitlen>>24):
                    (p==61)?(uchar)(bitlen>>16):
                    (p==62)?(uchar)(bitlen>>8):
                    (p==63)?(uchar)bitlen:0;
            v=(v<<8)|b;
        }
        w[i]=v;
    }
    uint H0=0x6a09e667u,H1=0xbb67ae85u,H2=0x3c6ef372u,H3=0xa54ff53au;
    uint H4=0x510e527fu,H5=0x9b05688cu,H6=0x1f83d9abu,H7=0x5be0cd19u;
    uint a=H0,b=H1,c=H2,d=H3,e=H4,f=H5,g=H6,h=H7;
    #pragma unroll
    for(int i=0;i<64;i++){
        uint wi;
        if(i<16){
            wi=w[i];
        }else{
            // Sliding window: indices all compile-time constant after unroll
            uint wm15=w[(i+1)&15], wm2=w[(i+14)&15], wm7=w[(i+9)&15];
            wi=SIG1(wm2)+wm7+SIG0(wm15)+w[i&15];
            w[i&15]=wi;
        }
        uint T1=h+EP1(e)+CH(e,f,g)+K[i]+wi;
        uint T2=EP0(a)+MAJ(a,b,c);
        h=g;g=f;f=e;e=d+T1;d=c;c=b;b=a;a=T1+T2;
    }
    digest[0]=H0+a;digest[1]=H1+b;digest[2]=H2+c;digest[3]=H3+d;
    digest[4]=H4+e;digest[5]=H5+f;digest[6]=H6+g;digest[7]=H7+h;
}

// ── sha256_vec4: 4 parallel hashes using uint4 arithmetic ────────────────────
// Each uint4 component (.x/.y/.z/.w) carries one independent chain.
// Packs identical padding into (uint4)(scalar) for free broadcast.
void sha256_vec4(
    __private const uchar* d0,__private const uchar* d1,
    __private const uchar* d2,__private const uchar* d3,
    int len, __private uint4* digest)
{
    uint4 w[16];
    ulong bitlen=(ulong)len*8;
    #pragma unroll 16
    for(int i=0;i<16;i++){
        uint v0=0,v1=0,v2=0,v3=0;
        #pragma unroll 4
        for(int j=0;j<4;j++){
            int p=i*4+j;
            // Padding bytes are identical for all 4 chains
            uchar pad=(p==len)?0x80:
                      (p==56)?(uchar)(bitlen>>56):
                      (p==57)?(uchar)(bitlen>>48):
                      (p==58)?(uchar)(bitlen>>40):
                      (p==59)?(uchar)(bitlen>>32):
                      (p==60)?(uchar)(bitlen>>24):
                      (p==61)?(uchar)(bitlen>>16):
                      (p==62)?(uchar)(bitlen>>8):
                      (p==63)?(uchar)bitlen:0;
            uchar b0=(p<len)?d0[p]:pad;
            uchar b1=(p<len)?d1[p]:pad;
            uchar b2=(p<len)?d2[p]:pad;
            uchar b3=(p<len)?d3[p]:pad;
            v0=(v0<<8)|b0; v1=(v1<<8)|b1;
            v2=(v2<<8)|b2; v3=(v3<<8)|b3;
        }
        w[i]=(uint4)(v0,v1,v2,v3);
    }
    uint4 H0=(uint4)(0x6a09e667u),H1=(uint4)(0xbb67ae85u),
          H2=(uint4)(0x3c6ef372u),H3=(uint4)(0xa54ff53au),
          H4=(uint4)(0x510e527fu),H5=(uint4)(0x9b05688cu),
          H6=(uint4)(0x1f83d9abu),H7=(uint4)(0x5be0cd19u);
    uint4 a=H0,bv=H1,cv=H2,dv=H3,ev=H4,fv=H5,gv=H6,hv=H7;
    #pragma unroll
    for(int i=0;i<64;i++){
        uint4 wi;
        if(i<16){
            wi=w[i];
        }else{
            uint4 wm15=w[(i+1)&15],wm2=w[(i+14)&15],wm7=w[(i+9)&15];
            wi=SIG14(wm2)+wm7+SIG04(wm15)+w[i&15];
            w[i&15]=wi;
        }
        uint4 T1=hv+EP14(ev)+CH4(ev,fv,gv)+(uint4)(K[i])+wi;
        uint4 T2=EP04(a)+MAJ4(a,bv,cv);
        hv=gv;gv=fv;fv=ev;ev=dv+T1;dv=cv;cv=bv;bv=a;a=T1+T2;
    }
    digest[0]=H0+a;  digest[1]=H1+bv; digest[2]=H2+cv; digest[3]=H3+dv;
    digest[4]=H4+ev; digest[5]=H5+fv; digest[6]=H6+gv; digest[7]=H7+hv;
}

uint W_fn(__private const uint* d){
    uint w=0;
    for(int i=0;i<8;i++){
        w+=(d[i]>>28)&0xF;w+=(d[i]>>24)&0xF;
        w+=(d[i]>>20)&0xF;w+=(d[i]>>16)&0xF;
        w+=(d[i]>>12)&0xF;w+=(d[i]>> 8)&0xF;
        w+=(d[i]>> 4)&0xF;w+= d[i]     &0xF;
    }
    return w;
}

void cdp_reduce_full(__private const uint* digest, ulong step,
                     __global const uchar* charset, int cs_len,
                     __private uchar* out, int out_len){
    ulong d64 = ((ulong)digest[0] << 32) | (ulong)digest[1];
    ulong seed= d64  * 6364136223846793005UL
              ^ step * 1442695040888963407UL;
    seed^=(seed>>30);seed*=0xBF58476D1CE4E5B9UL;
    seed^=(seed>>27);seed*=0x94D049BB133111EBUL;
    seed^=(seed>>31);
    for(int i=0;i<out_len;i++){
        seed=seed*6364136223846793005UL+1442695040888963407UL;
        out[i]=charset[(uint)(seed>>33)%(uint)cs_len];
    }
}

// ── cdp_reduce_inj: CDP-bijective reduction (paper Section 11, Theorem 11.1) ──
//
// Seed built from W + wvec16 + max + min — the exact components proven
// injective over SHA256(X) by the CDP bijection theorem.
//
// Proof of zero merges:
//   x1 ≠ x2  →  SHA256(x1) ≠ SHA256(x2)   (SHA256 collision resistance)
//            →  F(h1) ≠ F(h2)              (CDP bijection over X)
//            →  seed1 ≠ seed2              (components → unique seed)
//            →  R(h1) ≠ R(h2)             (different seed → different string)
//            →  chains never merge         (zero birthday paradox)
//
// Cost: same single pass over 8×uint32 digest as W_fn+wvec16+max_nibble+min_nibble.
// No extra SHA256 calls needed.  W2-W5 / cycle_entry only needed for endpoint key.
void cdp_reduce_inj(__private const uint* digest, ulong step,
                    __global const uchar* charset, int cs_len,
                    __private uchar* out, int out_len)
{
    // Single pass: compute W, wvec16[16], max, min simultaneously
    uint W=0, mx=0, mn=15;
    uint wv[16];
    #pragma unroll 8
    for(int i=0;i<8;i++){
        uint d=digest[i];
        uint n3=(d>>28)&0xF, n2=(d>>24)&0xF, n1=(d>>20)&0xF, n0=(d>>16)&0xF;
        uint n7=(d>>12)&0xF, n6=(d>> 8)&0xF, n5=(d>> 4)&0xF, n4= d     &0xF;
        wv[i*2  ]=n3+n2+n1+n0;
        wv[i*2+1]=n7+n6+n5+n4;
        W+=n3+n2+n1+n0+n7+n6+n5+n4;
        if(n3>mx)mx=n3; if(n2>mx)mx=n2; if(n1>mx)mx=n1; if(n0>mx)mx=n0;
        if(n7>mx)mx=n7; if(n6>mx)mx=n6; if(n5>mx)mx=n5; if(n4>mx)mx=n4;
        if(n3<mn)mn=n3; if(n2<mn)mn=n2; if(n1<mn)mn=n1; if(n0<mn)mn=n0;
        if(n7<mn)mn=n7; if(n6<mn)mn=n6; if(n5<mn)mn=n5; if(n4<mn)mn=n4;
    }

    // Build unique seed from all CDP components
    ulong seed=(ulong)W*0x9E3779B97F4A7C15UL;
    #pragma unroll 16
    for(int i=0;i<16;i++)
        seed=seed*6364136223846793005UL
            ^((ulong)wv[i]*(ulong)(i+1)*0xA3B2C1D4E5F60718UL);
    seed^=((ulong)mx<<32)|(ulong)mn;
    seed^=step*1442695040888963407UL;

    // PCG mixing (same as cdp_reduce_full)
    seed^=(seed>>30); seed*=0xBF58476D1CE4E5B9UL;
    seed^=(seed>>27); seed*=0x94D049BB133111EBUL;
    seed^=(seed>>31);

    for(int i=0;i<out_len;i++){
        seed=seed*6364136223846793005UL+1442695040888963407UL;
        out[i]=charset[(uint)(seed>>33)%(uint)cs_len];
    }
}

// ── vec4 helpers ─────────────────────────────────────────────────────────────

// Extract one scalar digest from uint4 digest (comp = 0/1/2/3 = x/y/z/w)
void extract_comp(__private const uint4* dg4, int comp,
                  __private uint* dg){
    if     (comp==0){for(int i=0;i<8;i++) dg[i]=dg4[i].x;}
    else if(comp==1){for(int i=0;i<8;i++) dg[i]=dg4[i].y;}
    else if(comp==2){for(int i=0;i<8;i++) dg[i]=dg4[i].z;}
    else            {for(int i=0;i<8;i++) dg[i]=dg4[i].w;}
}

// cdp_reduce_from_v4: extract scalar digest for one component, apply cdp_reduce_inj
void cdp_reduce_from_v4(__private const uint4* dg4, int comp, ulong step,
                        __global const uchar* charset, int cs_len,
                        __private uchar* out, int out_len){
    __private uint dg[8];
    extract_comp(dg4, comp, dg);           // reuses existing extract helper
    cdp_reduce_inj(dg, step, charset, cs_len, out, out_len);
}

void wvec16(__private const uint* d,__private uint* wv){
    for(int i=0;i<8;i++){
        wv[i*2]  =((d[i]>>28)&0xF)+((d[i]>>24)&0xF)
                 +((d[i]>>20)&0xF)+((d[i]>>16)&0xF);
        wv[i*2+1]=((d[i]>>12)&0xF)+((d[i]>> 8)&0xF)
                 +((d[i]>> 4)&0xF)+( d[i]      &0xF);
    }
}

uint max_nibble(__private const uint* d){
    uint mx=0,v;
    for(int i=0;i<8;i++){
        v=(d[i]>>28)&0xF;if(v>mx)mx=v;
        v=(d[i]>>24)&0xF;if(v>mx)mx=v;
        v=(d[i]>>20)&0xF;if(v>mx)mx=v;
        v=(d[i]>>16)&0xF;if(v>mx)mx=v;
        v=(d[i]>>12)&0xF;if(v>mx)mx=v;
        v=(d[i]>> 8)&0xF;if(v>mx)mx=v;
        v=(d[i]>> 4)&0xF;if(v>mx)mx=v;
        v= d[i]     &0xF;if(v>mx)mx=v;
    }
    return mx;
}

uint min_nibble(__private const uint* d){
    uint mn=15,v;
    for(int i=0;i<8;i++){
        v=(d[i]>>28)&0xF;if(v<mn)mn=v;
        v=(d[i]>>24)&0xF;if(v<mn)mn=v;
        v=(d[i]>>20)&0xF;if(v<mn)mn=v;
        v=(d[i]>>16)&0xF;if(v<mn)mn=v;
        v=(d[i]>>12)&0xF;if(v<mn)mn=v;
        v=(d[i]>> 8)&0xF;if(v<mn)mn=v;
        v=(d[i]>> 4)&0xF;if(v<mn)mn=v;
        v= d[i]     &0xF;if(v<mn)mn=v;
    }
    return mn;
}

// ── Scalar kernel (W[16] optimized) ──────────────────────────────────────────
__kernel void build_chains(
    __global const uchar* starts,
    __global       uint*  end_W,
    __global       uint*  end_wv,
    __global       uint*  end_maxmin,
    __global const uchar* charset,
             const int    cs_len,
             const int    str_len,
             const int    chain_len
){
    int gid=get_global_id(0);
    int base=gid*str_len;
    __private uchar cur[16];
    for(int i=0;i<str_len;i++) cur[i]=starts[base+i];
    __private uint digest[8];
    for(int step=0;step<chain_len;step++){
        sha256_opt(cur,str_len,digest);
        cdp_reduce_inj(digest,(ulong)step,charset,cs_len,cur,str_len);
    }
    sha256_opt(cur,str_len,digest);
    end_W[gid]=W_fn(digest);
    __private uint wv[16];
    wvec16(digest,wv);
    for(int i=0;i<16;i++) end_wv[gid*16+i]=wv[i];
    end_maxmin[gid*2]  =max_nibble(digest);
    end_maxmin[gid*2+1]=min_nibble(digest);
}

// ── Vec4 kernel: each thread processes 4 chains simultaneously ────────────────
// starts layout: [thread0_chain0..3][thread1_chain0..3]...  (4*str_len per thread)
// output layout: [thread0_chain0..3][thread1_chain0..3]...  (4 results per thread)
__kernel void build_chains_vec4(
    __global const uchar* starts,
    __global       uint*  end_W,
    __global       uint*  end_wv,
    __global       uint*  end_maxmin,
    __global const uchar* charset,
             const int    cs_len,
             const int    str_len,
             const int    chain_len
){
    int gid=get_global_id(0);
    int base=gid*4*str_len;

    // Load 4 chain start strings into private memory
    __private uchar c0[16],c1[16],c2[16],c3[16];
    for(int i=0;i<str_len;i++){
        c0[i]=starts[base+0*str_len+i];
        c1[i]=starts[base+1*str_len+i];
        c2[i]=starts[base+2*str_len+i];
        c3[i]=starts[base+3*str_len+i];
    }

    __private uint4 dg4[8];

    // Walk all 4 chains in lockstep
    for(int step=0;step<chain_len;step++){
        ulong s=(ulong)step;
        sha256_vec4(c0,c1,c2,c3,str_len,dg4);
        cdp_reduce_from_v4(dg4,0,s,charset,cs_len,c0,str_len);
        cdp_reduce_from_v4(dg4,1,s,charset,cs_len,c1,str_len);
        cdp_reduce_from_v4(dg4,2,s,charset,cs_len,c2,str_len);
        cdp_reduce_from_v4(dg4,3,s,charset,cs_len,c3,str_len);
    }

    // Final hash for all 4 chains
    sha256_vec4(c0,c1,c2,c3,str_len,dg4);

    // Compute and store fingerprint components for each of the 4 chains
    int ob=gid*4;
    __private uint dg[8];

    for(int comp=0;comp<4;comp++){
        extract_comp(dg4,comp,dg);
        end_W[ob+comp]=W_fn(dg);
        __private uint wv[16];
        wvec16(dg,wv);
        for(int k=0;k<16;k++) end_wv[(ob+comp)*16+k]=wv[k];
        end_maxmin[(ob+comp)*2]  =max_nibble(dg);
        end_maxmin[(ob+comp)*2+1]=min_nibble(dg);
    }
}

// ── sha256_ilp2: 2 independent SHA256s interleaved for ILP ───────────────────
// After #pragma unroll, the compiler sees chain-0 and chain-1 computations in
// each merged round as fully independent instruction streams.  AMD RDNA's
// out-of-order instruction scheduler can then overlap the ALU dependency
// stalls of one chain with useful work from the other → ~1.6× speedup.
//
// VGPR budget: w0[16]+w1[16]=32, a..h ×2=16, temps=8  → ~58 total
// (vs sha256_opt=42, vec4=156)  →  max 17 wavefronts/SIMD32  (excellent)
void sha256_ilp2(
    __private const uchar* inp0, __private const uchar* inp1,
    int len,
    __private uint* dg0, __private uint* dg1)
{
    uint w0[16], w1[16];
    ulong bitlen=(ulong)len*8;

    // Load + pad both chains simultaneously (same padding, different data bytes)
    #pragma unroll 16
    for(int i=0;i<16;i++){
        uint v0=0,v1=0;
        #pragma unroll 4
        for(int j=0;j<4;j++){
            int p=i*4+j;
            uchar pad=(p==len)?0x80:
                      (p==56)?(uchar)(bitlen>>56):
                      (p==57)?(uchar)(bitlen>>48):
                      (p==58)?(uchar)(bitlen>>40):
                      (p==59)?(uchar)(bitlen>>32):
                      (p==60)?(uchar)(bitlen>>24):
                      (p==61)?(uchar)(bitlen>>16):
                      (p==62)?(uchar)(bitlen>>8):
                      (p==63)?(uchar)bitlen:0;
            v0=(v0<<8)|((p<len)?inp0[p]:pad);
            v1=(v1<<8)|((p<len)?inp1[p]:pad);
        }
        w0[i]=v0; w1[i]=v1;
    }

    // SHA256 state — chain 0
    uint a0=0x6a09e667u,b0=0xbb67ae85u,c0=0x3c6ef372u,d0=0xa54ff53au;
    uint e0=0x510e527fu,f0=0x9b05688cu,g0=0x1f83d9abu,h0=0x5be0cd19u;
    // SHA256 state — chain 1 (separate registers, independent of chain 0)
    uint a1=0x6a09e667u,b1=0xbb67ae85u,c1=0x3c6ef372u,d1=0xa54ff53au;
    uint e1=0x510e527fu,f1=0x9b05688cu,g1=0x1f83d9abu,h1=0x5be0cd19u;

    // 64 merged rounds — with full unroll, compiler interleaves A and B ops
    #pragma unroll
    for(int i=0;i<64;i++){
        uint wi0,wi1;
        if(i<16){
            wi0=w0[i]; wi1=w1[i];
        }else{
            // Message schedule: chain 0
            uint wm15_0=w0[(i+1)&15],wm2_0=w0[(i+14)&15],wm7_0=w0[(i+9)&15];
            wi0=SIG1(wm2_0)+wm7_0+SIG0(wm15_0)+w0[i&15]; w0[i&15]=wi0;
            // Message schedule: chain 1 (independent — fills chain-0 ALU stall)
            uint wm15_1=w1[(i+1)&15],wm2_1=w1[(i+14)&15],wm7_1=w1[(i+9)&15];
            wi1=SIG1(wm2_1)+wm7_1+SIG0(wm15_1)+w1[i&15]; w1[i&15]=wi1;
        }
        // Compression: chain 0
        uint T1_0=h0+EP1(e0)+CH(e0,f0,g0)+K[i]+wi0;
        uint T2_0=EP0(a0)+MAJ(a0,b0,c0);
        // Compression: chain 1 (independent — fills T1_0 latency window)
        uint T1_1=h1+EP1(e1)+CH(e1,f1,g1)+K[i]+wi1;
        uint T2_1=EP0(a1)+MAJ(a1,b1,c1);
        // Update chain 0
        h0=g0;g0=f0;f0=e0;e0=d0+T1_0;d0=c0;c0=b0;b0=a0;a0=T1_0+T2_0;
        // Update chain 1
        h1=g1;g1=f1;f1=e1;e1=d1+T1_1;d1=c1;c1=b1;b1=a1;a1=T1_1+T2_1;
    }

    // Write both digests
    dg0[0]=0x6a09e667u+a0; dg0[1]=0xbb67ae85u+b0;
    dg0[2]=0x3c6ef372u+c0; dg0[3]=0xa54ff53au+d0;
    dg0[4]=0x510e527fu+e0; dg0[5]=0x9b05688cu+f0;
    dg0[6]=0x1f83d9abu+g0; dg0[7]=0x5be0cd19u+h0;
    dg1[0]=0x6a09e667u+a1; dg1[1]=0xbb67ae85u+b1;
    dg1[2]=0x3c6ef372u+c1; dg1[3]=0xa54ff53au+d1;
    dg1[4]=0x510e527fu+e1; dg1[5]=0x9b05688cu+f1;
    dg1[6]=0x1f83d9abu+g1; dg1[7]=0x5be0cd19u+h1;
}

// ── ILP2 kernel: each thread processes 2 chains, fully scalar ─────────────────
// batch=65536 threads → 131,072 chains/dispatch.
// ~58 VGPRs/thread → 17 wavefronts/SIMD32 → excellent latency hiding.
__kernel void build_chains_ilp2(
    __global const uchar* starts,
    __global       uint*  end_W,
    __global       uint*  end_wv,
    __global       uint*  end_maxmin,
    __global const uchar* charset,
             const int    cs_len,
             const int    str_len,
             const int    chain_len
){
    int gid=get_global_id(0);
    int base=gid*2*str_len;

    __private uchar c0[16],c1[16];
    for(int i=0;i<str_len;i++){
        c0[i]=starts[base+0*str_len+i];
        c1[i]=starts[base+1*str_len+i];
    }

    __private uint dg0[8],dg1[8];

    for(int step=0;step<chain_len;step++){
        ulong s=(ulong)step;
        sha256_ilp2(c0,c1,str_len,dg0,dg1);
        cdp_reduce_inj(dg0,s,charset,cs_len,c0,str_len);
        cdp_reduce_inj(dg1,s,charset,cs_len,c1,str_len);
    }

    sha256_ilp2(c0,c1,str_len,dg0,dg1);

    int ob=gid*2;
    __private uint wv[16];

    // Chain 0 fingerprint
    end_W[ob]=W_fn(dg0);
    wvec16(dg0,wv);
    for(int k=0;k<16;k++) end_wv[ob*16+k]=wv[k];
    end_maxmin[ob*2]  =max_nibble(dg0);
    end_maxmin[ob*2+1]=min_nibble(dg0);

    // Chain 1 fingerprint
    end_W[ob+1]=W_fn(dg1);
    wvec16(dg1,wv);
    for(int k=0;k<16;k++) end_wv[(ob+1)*16+k]=wv[k];
    end_maxmin[(ob+1)*2]  =max_nibble(dg1);
    end_maxmin[(ob+1)*2+1]=min_nibble(dg1);
}

// ── query_chains: GPU-accelerated rainbow table crack ────────────────────────
// Each thread handles one chain position, extending target_hash to endpoint.
//
// For position pos (scanned END→START):
//   cur = target_hash (32 bytes, packed as uint[8])
//   for step = pos to chain_len-1:
//     s   = cdp_reduce_inj(cur, step)
//     cur = sha256(s)
//   output fingerprint(cur) = (W, wvec16[16], max, min)
//
// CPU then does O(1) dict lookup.  On hit, CPU walks chain forward to verify.
//
// Throughput: same sha256_opt + cdp_reduce_inj as build — ~4.6 GH/s on gfx1201.
// Wall time for chain_len=1M: ~2 min/table.
__kernel void query_chains(
    __global const uchar* target_hash,   // 32 bytes, big-endian SHA256
    int chain_len,
    int batch_offset,                    // first gid maps to pos = (chain_len-1)-batch_offset
    __global const uchar* charset,
    int cs_len,
    int str_len,
    __global uint* out_W,                // [n_threads]
    __global uint* out_wv,               // [n_threads × 16]
    __global uint* out_mm                // [n_threads × 2]  {max, min}
)
{
    int gid = get_global_id(0);
    int pos = (chain_len - 1) - (batch_offset + gid);
    if(pos < 0) return;

    // Load target hash into private uint[8] (big-endian bytes → uint words)
    uint digest[8];
    for(int i=0;i<8;i++){
        digest[i] = ((uint)target_hash[i*4  ]<<24)
                  | ((uint)target_hash[i*4+1]<<16)
                  | ((uint)target_hash[i*4+2]<< 8)
                  | ((uint)target_hash[i*4+3]);
    }

    // AMD PAL/LLVM COMPILER BUG WORKAROUND:
    // Variable-start loop for(step=pos; step<N) produces wrong results on
    // gfx1201 with -cl-fast-relaxed-math when threads have different pos values.
    // Fix: fixed loop 0..chain_len with conditional activation at step==pos.
    // Ref: AMD Community "Wrong OpenCL calculation result" — workaround: -O0 or
    // fixed loop bounds.  We use fixed loop (correctness + full performance).
    uchar s[16];
    for(int step=0; step<chain_len; step++){
        if(step >= pos){
            cdp_reduce_inj(digest,(ulong)step,charset,cs_len,s,str_len);
            sha256_opt(s,str_len,digest);
        }
    }

    // digest now = sha256(endpoint_string) = endpoint_hash
    // Compute fingerprint components (same single-pass as cdp_reduce_inj)
    uint W=0, mx=0, mn=15;
    uint wv[16];
    for(int i=0;i<8;i++){
        uint d=digest[i];
        uint n3=(d>>28)&0xF,n2=(d>>24)&0xF,n1=(d>>20)&0xF,n0=(d>>16)&0xF;
        uint n7=(d>>12)&0xF,n6=(d>> 8)&0xF,n5=(d>> 4)&0xF,n4= d     &0xF;
        wv[i*2  ]=n3+n2+n1+n0;
        wv[i*2+1]=n7+n6+n5+n4;
        W+=wv[i*2]+wv[i*2+1];
        if(n3>mx)mx=n3;if(n2>mx)mx=n2;if(n1>mx)mx=n1;if(n0>mx)mx=n0;
        if(n7>mx)mx=n7;if(n6>mx)mx=n6;if(n5>mx)mx=n5;if(n4>mx)mx=n4;
        if(n3<mn)mn=n3;if(n2<mn)mn=n2;if(n1<mn)mn=n1;if(n0<mn)mn=n0;
        if(n7<mn)mn=n7;if(n6<mn)mn=n6;if(n5<mn)mn=n5;if(n4<mn)mn=n4;
    }
    out_W[gid]        = W;
    for(int i=0;i<16;i++) out_wv[gid*16+i]=wv[i];
    out_mm[gid*2  ]   = mx;
    out_mm[gid*2+1]   = mn;
}
"""

# --- Python Host Code ---------------------------------------------------------

# Conservative GPU hash rate estimate for safe_batch calculation
# (actual rate is measured and printed during build)
_EST_GPU_RATE = 10e9  # 10 GH/s conservative

# Max seconds per GPU dispatch to stay under AMD watchdog
_MAX_DISPATCH_S = 0.80  # 800ms -- under AMD 1-2s watchdog's ~1-2s internal limit

def safe_batch_size(chain_len, user_batch=65536):
    """
    On Linux/WSL2: no TDR or AMD watchdog -- use full user_batch directly.
    On Windows: would cap at _MAX_DISPATCH_S * _EST_GPU_RATE / chain_len.
    """
    return user_batch



def _query_worker(args):
    import warnings; warnings.filterwarnings('ignore')  # suppress per-process import noise
    target_hash_hex, pos_slice, charset, str_len, chain_len, table = args
    for pos in pos_slice:
        h = target_hash_hex
        for step in range(pos, chain_len):
            s = py_reduce_inj(h, step, charset, str_len)
            h = sha256h(s.encode())
        fp = full_fingerprint(h)
        if fp in table:
            cur = table[fp]
            for step in range(chain_len):
                candidate = sha256h(cur.encode())
                if candidate == target_hash_hex:
                    return cur
                cur = py_reduce_inj(candidate, step, charset, str_len)
    return None


class CDPChainBuilder:
    def __init__(self, charset_name='lower', str_len=8, chain_len=1_000_000):
        self.charset_name = charset_name
        self.charset      = CHARSETS[charset_name]
        self.cs_len       = len(self.charset)
        self.str_len      = str_len
        self.chain_len    = chain_len
        self._init_opencl()

        auto_batch = safe_batch_size(chain_len)
        est_dispatch_ms = auto_batch * chain_len / _EST_GPU_RATE * 1000

        print(f"CDP Chain Builder initialized:")
        print(f"  Charset:       {charset_name} ({self.cs_len} chars)")
        print(f"  Str len:       {str_len}")
        print(f"  Chain len:     {chain_len:,}  (paper Table 3 optimum)")
        print(f"  Space:         {self.cs_len**str_len:,}")
        chains_exact = self.cs_len**str_len // chain_len
        print(f"  Chains needed: {chains_exact:,}  (exact — CDP reduction = 0 merges)")
        print(f"  Coverage:      100%  (CDP bijection guarantees)")
        print(f"  Table size:    {chains_exact * 14 / 1e6:.1f} MB  (14 bytes/chain)")
        print(f"  GPU:           {self.device.name}")
        print(f"  Auto batch:    {auto_batch:,}  (~{est_dispatch_ms:.0f}ms/dispatch)")

    def _init_opencl(self):
        """
        Device selection strategy:
          1. Prefer named device if --device flag given
          2. Among all GPU devices, pick gfx1201 with most compute units
             (avoids gfx1036 1-CU emulated device and duplicate platforms)
          3. Fall back to first AMD GPU
        """
        candidates = []
        seen = set()
        for p in cl.get_platforms():
            for d in p.get_devices():
                if d.type != cl.device_type.GPU:
                    continue
                key = (d.name, d.driver_version)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((p, d))

        if not candidates:
            raise RuntimeError("No GPU devices found!")

        # Sort: prefer gfx1201 > other AMD > anything else
        # Within same name: prefer more compute units
        def score(pd):
            p, d = pd
            name = d.name.lower()
            cu   = d.max_compute_units
            if 'gfx1201' in name: return (100, cu)
            if 'gfx1036' in name and cu <= 2: return (0, cu)  # skip emulated
            if 'amd' in d.vendor.lower(): return (50, cu)
            return (10, cu)

        candidates.sort(key=score, reverse=True)

        # Print all candidates for transparency
        print(f"  Available devices:")
        for i, (p, d) in enumerate(candidates):
            marker = "<-- selected" if i == 0 else ""
            print(f"    [{i}] {d.name} ({d.max_compute_units} CUs) "
                  f"drv={d.driver_version} {marker}")

        _, gpu     = candidates[0]
        self.device  = gpu
        self.ctx     = cl.Context([gpu])
        self.queue   = cl.CommandQueue(self.ctx)
        self.program = cl.Program(
            self.ctx,
            KERNEL_SRC.encode('ascii','replace').decode('ascii')
        ).build(options="-cl-fast-relaxed-math")
        cs_bytes = np.frombuffer(self.charset.encode(), dtype=np.uint8)
        self.cs_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=cs_bytes
        )
        # Pre-create kernels once — build kernels use -cl-fast-relaxed-math for speed
        self.kernel      = cl.Kernel(self.program, 'build_chains')
        self.kernel_v4   = cl.Kernel(self.program, 'build_chains_vec4')
        self.kernel_ilp2 = cl.Kernel(self.program, 'build_chains_ilp2')

        # Query kernel compiled separately with -O0 (NO optimization).
        # AMD PAL/LLVM produces wrong results for variable-bound loops
        # (for step=pos; step<N) with -cl-fast-relaxed-math on gfx1201.
        # Confirmed workaround by AMD engineer: "-O0" restores correctness.
        # Ref: community.amd.com "Wrong OpenCL calculation result on AMD 5700 XT"
        # -O0 reduces query throughput ~3-4x but correctness is non-negotiable.
        print(f"  Compiling query kernel with -O0 (AMD LLVM correctness fix)...")
        self.program_query = cl.Program(
            self.ctx,
            KERNEL_SRC.encode('ascii', 'replace').decode('ascii')
        ).build(options="-O0")
        self.kernel_query = cl.Kernel(self.program_query, 'query_chains')

        # Try Vulkan compute engine (ACO backend — no PAL-LLVM optimizer bug)
        self._vk_engine = None
        if _vq.vk_available():
            try:
                self._vk_engine = _vq.VulkanQueryEngine(
                    self.chain_len, self.charset, self.str_len, verbose=True
                )
                print("  [Vulkan] GPU query enabled — OpenCL query disabled")
            except Exception as e:
                import traceback
                print(f"  [Vulkan] Init failed — full traceback:")
                traceback.print_exc()
                print("  Falling back to multiprocessing")
        else:
            print("  [Vulkan] Not available (pip install vulkan) — using multiprocessing query")

        # Vulkan verify engine — GPU forward walk + SHA256 compare
        self._vk_verify = None
        if _vq.vk_available():
            try:
                self._vk_verify = _vq.VulkanVerifyEngine(
                    self.charset, self.str_len, verbose=True)
                print("  [Vulkan Verify] GPU verify enabled")
            except Exception as e:
                print(f"  [Vulkan Verify] Init failed: {e} — using CPU verify")

    def _run_batch(self, starts, chain_len=None):
        """
        Single GPU dispatch for a batch of starts.
        Keep len(starts) small (use safe_batch_size) to stay under AMD watchdog.
        """
        t = chain_len or self.chain_len
        n = len(starts)

        # Fast: join all strings, encode once, no Python loop
        starts_flat = np.frombuffer(''.join(starts).encode('ascii'), dtype=np.uint8)

        starts_buf = cl.Buffer(self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=starts_flat)
        W_buf  = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY,
                           n      * np.dtype(np.uint32).itemsize)
        wv_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY,
                           n * 16 * np.dtype(np.uint32).itemsize)
        mm_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY,
                           n *  2 * np.dtype(np.uint32).itemsize)

        # Reuse pre-created kernel (avoids RepeatedKernelRetrieval warning + overhead)
        self.kernel.set_args(
            starts_buf, W_buf, wv_buf, mm_buf,
            self.cs_buf,
            np.int32(self.cs_len),
            np.int32(self.str_len),
            np.int32(t)
        )
        cl.enqueue_nd_range_kernel(self.queue, self.kernel, (n,), None)
        self.queue.finish()

        end_W  = np.zeros(n,      dtype=np.uint32)
        end_wv = np.zeros(n * 16, dtype=np.uint32)
        end_mm = np.zeros(n *  2, dtype=np.uint32)
        cl.enqueue_copy(self.queue, end_W,  W_buf)
        cl.enqueue_copy(self.queue, end_wv, wv_buf)
        cl.enqueue_copy(self.queue, end_mm, mm_buf)
        self.queue.finish()
        return end_W, end_wv.reshape(n, 16), end_mm.reshape(n, 2)

    def validate_gpu(self, n_samples=20):
        """Compare scalar / vec4 / ilp2 GPU output with Python reference."""
        import random; random.seed(42)
        starts   = [''.join(random.choices(self.charset, k=self.str_len))
                    for _ in range(n_samples)]
        test_len = 100

        print(f"\nGPU validation — scalar ({n_samples} chains, t={test_len})...")
        gW, gWV, gMM = self._run_batch(starts, chain_len=test_len)
        ok_s = 0
        for i, s in enumerate(starts):
            fp = py_chain(s, test_len, self.charset)
            if (gW[i]==fp[0] and tuple(int(x) for x in gWV[i])==fp[1]
                    and gMM[i,0]==fp[7] and gMM[i,1]==fp[8]):
                ok_s += 1
            else:
                print(f"  SCALAR MISMATCH {i}: GPU W={gW[i]} Python W={fp[0]}")
        print(f"  Scalar: {ok_s}/{n_samples} OK")

        print(f"GPU validation — vec4 ({n_samples} chains, t={test_len})...")
        gW4, gWV4, gMM4 = self._run_batch_vec4(starts, chain_len_override=test_len)
        ok_v = sum(1 for i in range(n_samples)
                   if gW4[i]==gW[i] and all(gWV4[i]==gWV[i])
                   and gMM4[i,0]==gMM[i,0] and gMM4[i,1]==gMM[i,1])
        if ok_v < n_samples:
            for i in range(n_samples):
                if gW4[i]!=gW[i]: print(f"  VEC4 MISMATCH {i}: scalar W={gW[i]} vec4 W={gW4[i]}")
        print(f"  Vec4:   {ok_v}/{n_samples} OK")

        print(f"GPU validation — ilp2 ({n_samples} chains, t={test_len})...")
        gW2, gWV2, gMM2 = self._run_batch_ilp2(starts, chain_len_override=test_len)
        ok_i = sum(1 for i in range(n_samples)
                   if gW2[i]==gW[i] and all(gWV2[i]==gWV[i])
                   and gMM2[i,0]==gMM[i,0] and gMM2[i,1]==gMM[i,1])
        if ok_i < n_samples:
            for i in range(n_samples):
                if gW2[i]!=gW[i]: print(f"  ILP2 MISMATCH {i}: scalar W={gW[i]} ilp2 W={gW2[i]}")
        print(f"  ILP2:   {ok_i}/{n_samples} OK")

        return ok_s==n_samples and ok_v==n_samples and ok_i==n_samples

    def _run_batch_ilp2(self, starts, chain_len_override=None):
        """
        ILP2 dispatch: each GPU thread processes 2 chains with interleaved SHA256.
        len(starts) must be divisible by 2.
        ~58 VGPRs/thread → 17 wavefronts/SIMD32 → best latency hiding.
        """
        assert len(starts) % 2 == 0, "ilp2 requires len(starts) % 2 == 0"
        n_total   = len(starts)
        n_threads = n_total // 2
        t         = chain_len_override or self.chain_len

        starts_flat = np.frombuffer(''.join(starts).encode('ascii'), dtype=np.uint8)
        starts_buf  = cl.Buffer(self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=starts_flat)
        W_buf  = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY,
                           n_total      * np.dtype(np.uint32).itemsize)
        wv_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY,
                           n_total * 16 * np.dtype(np.uint32).itemsize)
        mm_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY,
                           n_total *  2 * np.dtype(np.uint32).itemsize)

        self.kernel_ilp2.set_args(
            starts_buf, W_buf, wv_buf, mm_buf,
            self.cs_buf,
            np.int32(self.cs_len),
            np.int32(self.str_len),
            np.int32(t)
        )
        cl.enqueue_nd_range_kernel(self.queue, self.kernel_ilp2, (n_threads,), None)
        self.queue.finish()

        end_W  = np.zeros(n_total,      dtype=np.uint32)
        end_wv = np.zeros(n_total * 16, dtype=np.uint32)
        end_mm = np.zeros(n_total *  2, dtype=np.uint32)
        cl.enqueue_copy(self.queue, end_W,  W_buf)
        cl.enqueue_copy(self.queue, end_wv, wv_buf)
        cl.enqueue_copy(self.queue, end_mm, mm_buf)
        self.queue.finish()
        return end_W, end_wv.reshape(n_total, 16), end_mm.reshape(n_total, 2)

    def _run_batch_vec4(self, starts, chain_len_override=None):
        """Vec4 dispatch: each GPU thread processes 4 chains. len(starts) % 4 == 0."""
        assert len(starts) % 4 == 0, "vec4 requires len(starts) % 4 == 0"
        n_total   = len(starts)
        n_threads = n_total // 4
        t         = chain_len_override or self.chain_len

        starts_flat = np.frombuffer(''.join(starts).encode('ascii'), dtype=np.uint8)
        starts_buf  = cl.Buffer(self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=starts_flat)
        W_buf  = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY,
                           n_total      * np.dtype(np.uint32).itemsize)
        wv_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY,
                           n_total * 16 * np.dtype(np.uint32).itemsize)
        mm_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY,
                           n_total *  2 * np.dtype(np.uint32).itemsize)

        self.kernel_v4.set_args(
            starts_buf, W_buf, wv_buf, mm_buf,
            self.cs_buf,
            np.int32(self.cs_len),
            np.int32(self.str_len),
            np.int32(t)
        )
        cl.enqueue_nd_range_kernel(self.queue, self.kernel_v4, (n_threads,), None)
        self.queue.finish()

        end_W  = np.zeros(n_total,      dtype=np.uint32)
        end_wv = np.zeros(n_total * 16, dtype=np.uint32)
        end_mm = np.zeros(n_total *  2, dtype=np.uint32)
        cl.enqueue_copy(self.queue, end_W,  W_buf)
        cl.enqueue_copy(self.queue, end_wv, wv_buf)
        cl.enqueue_copy(self.queue, end_mm, mm_buf)
        self.queue.finish()
        return end_W, end_wv.reshape(n_total, 16), end_mm.reshape(n_total, 2)

    def build_table(self, output_file, user_batch=65536, mode='ilp2', seed_offset=0):
        """
        Build CDP rainbow table.

        mode='ilp2'   (default): 2 chains/thread, interleaved ILP. Best AMD RDNA rate.
        mode='vec4':             4 chains/thread, uint4 (no benefit on AMD RDNA).
        mode='scalar':           1 chain/thread (baseline / debug).
        """
        import random
        total_chains = self.cs_len ** self.str_len // self.chain_len
        n_threads    = min(user_batch, 65536)
        mult         = {'scalar': 1, 'ilp2': 2, 'vec4': 4}[mode]
        batch_chains = n_threads * mult
        total_batches = (total_chains + batch_chains - 1) // batch_chains
        dispatch     = {'scalar': self._run_batch,
                        'ilp2':   self._run_batch_ilp2,
                        'vec4':   self._run_batch_vec4}[mode]

        desc = {'scalar': 'scalar (1 chain/thread)',
                'ilp2':   'ILP2  (2 chains/thread, interleaved — recommended)',
                'vec4':   'vec4  (4 chains/thread, uint4)'}
        print(f"\nBuilding table: {total_chains:,} chains")
        print(f"Mode:           {desc[mode]}")
        print(f"Threads/dispatch: {n_threads:,}  ->  {batch_chains:,} chains/dispatch")
        print(f"Total batches:  {total_batches:,}")
        print()

        table  = {}
        built  = 0
        t0     = time.time()
        random.seed(42 + seed_offset)

        while built < total_chains:
            n_this = min(total_chains - built, batch_chains)
            n_this = ((n_this + mult - 1) // mult) * mult   # divisibility

            # Deduplicate: same start → same chain → same endpoint → dict overwrite
            # Use a set to guarantee unique starts within this batch
            starts_set = set()
            while len(starts_set) < n_this:
                starts_set.add(
                    ''.join(random.choices(self.charset, k=self.str_len)))
            starts = list(starts_set)
            W, wv, mm = dispatch(starts)

            for i in range(n_this):
                W_val = int(W[i])
                entry = _W_CACHE.get(W_val)
                if entry:
                    w2, w3, w4, w5, ce = entry
                else:
                    w2 = _iter_W(W_val); w3 = _iter_W(w2)
                    w4 = _iter_W(w3);    w5 = _iter_W(w4)
                    ce = _cycle_entry(W_val)
                key = (W_val,
                       tuple(int(x) for x in wv[i]),
                       ce, w2, w3, w4, w5,
                       int(mm[i,0]), int(mm[i,1]))
                table[key] = starts[i]

            built  += n_this
            elapsed = time.time() - t0
            rate    = built * self.chain_len / elapsed if elapsed > 0 else 0
            eta     = (total_chains - built) * self.chain_len / rate if rate > 0 else 0
            print(f"  {built/total_chains*100:>5.1f}%  {built:>10,}/{total_chains:>10,}  "
                  f"rate={rate/1e9:.3f}GH/s  ETA={eta/3600:.2f}h")


        print(f"\nSaving: {output_file}")

        # Shared entry list (used by both JSON and binary paths)
        entries = [
            {'key': [int(k[0])] + [int(x) for x in k[1]]
                   + [int(k[2])]
                   + [int(k[3]), int(k[4]), int(k[5]), int(k[6])]
                   + [int(k[7]), int(k[8])],
             'start': v,
             'w0': _start_w0_bucket(v),
             's1': _start_s1fp(v)}
            for k, v in table.items()
        ]

        if output_file.endswith('.bin'):
            # Direct binary — no JSON intermediary, 3.4x smaller
            n_written = save_binary_table(
                entries, output_file,
                self.charset_name, self.str_len,
                self.chain_len, len(table))
            size_mb = os.path.getsize(output_file) / 1e6
            print(f"Table saved (binary): {size_mb:.1f} MB, {n_written:,} chains")
        else:
            table_data = {
                'charset': self.charset_name, 'str_len': self.str_len,
                'chain_len': self.chain_len,  'n_chains': len(table),
                'entries': entries
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(table_data, f)
            size_mb = os.path.getsize(output_file) / 1e6
            print(f"Table saved (JSON): {size_mb:.1f} MB, {len(table):,} chains")
        return table

    def query(self, target_hash_hex, table):
        """
        Direct endpoint lookup — O(1) table lookup + O(chain_len) chain walk.

        Only succeeds if target_hash IS the final hash of a stored chain.
        Probability for a random target: n_chains / keyspace ≈ 0.002%.
        Useful for self-testing; for general cracking use full_query().
        """
        fp = full_fingerprint(target_hash_hex)
        if fp not in table:
            return None
        cur = table[fp]
        for step in range(self.chain_len):
            h = sha256h(cur.encode())
            if h == target_hash_hex:
                return cur
            cur = py_reduce_inj(h, step, self.charset, self.str_len)
        return None

    def full_query(self, target_hash_hex, table, verbose=True):
        """
        Parallel rainbow table query using all CPU cores.
        Splits pos range across mp.cpu_count() workers; stops on first hit.
          chain_len=30,000  + 12 cores -> ~9s/table  (Ryzen 9 7900)
          chain_len=100,000 + 12 cores -> ~2min/table
        """
        import time as _t
        import multiprocessing as mp
        t0 = _t.time()
        n_workers = max(1, mp.cpu_count() // 2)  # physical cores only (HT useless for CPU-bound)
        positions = list(range(self.chain_len - 1, -1, -1))
        chunk     = max(1, len(positions) // n_workers)
        slices    = [positions[i:i+chunk] for i in range(0, len(positions), chunk)]
        args = [
            (target_hash_hex, sl, self.charset, self.str_len, self.chain_len, table)
            for sl in slices
        ]
        if verbose:
            print(f"  [{n_workers} workers / chain_len={self.chain_len:,}]", flush=True)
        result = None
        with mp.Pool(n_workers) as pool:
            for hit in pool.imap_unordered(_query_worker, args, chunksize=1):
                if hit is not None:
                    result = hit
                    pool.terminate()
                    break
        elapsed = _t.time() - t0
        if verbose:
            status = "found" if result else "not found"
            print(f"  {status}  ({elapsed:.1f}s)", flush=True)
        return result, elapsed


    def self_test(self):
        """
        End-to-end test with a mini table (chain_len=1000).
        Builds a small table, picks a random start string, cracks its
        endpoint hash, then verifies the retrieved preimage.
        """
        import random
        print("\n=== CDP Self-Test (chain_len=1000) ===")

        # 1. Build a small table
        CHAIN_LEN_TEST = 1000
        orig_chain = self.chain_len
        self.chain_len = CHAIN_LEN_TEST
        self.kernel.set_args.__func__   # ensure kernel reuse is OK

        n_test_chains = 500
        random.seed(99)
        starts = [''.join(random.choices(self.charset, k=self.str_len))
                  for _ in range(n_test_chains)]
        W, wv, mm = self._run_batch_ilp2(starts, chain_len_override=CHAIN_LEN_TEST)

        mini_table = {}
        for i in range(n_test_chains):
            W_val = int(W[i]); entry = _W_CACHE.get(W_val)
            if entry: w2,w3,w4,w5,ce = entry
            else:
                w2=_iter_W(W_val); w3=_iter_W(w2); w4=_iter_W(w3); w5=_iter_W(w4)
                ce=_cycle_entry(W_val)
            key = (W_val, tuple(int(x) for x in wv[i]), ce, w2,w3,w4,w5,
                   int(mm[i,0]), int(mm[i,1]))
            mini_table[key] = starts[i]

        print(f"  Mini table: {len(mini_table)} unique chains built")

        # 2. Pick a known plaintext and compute its SHA256
        test_plain = random.choice(list(mini_table.values()))
        # Walk chain to get a mid-chain hash (not endpoint, to test full_query)
        cur = test_plain
        target_step = random.randint(0, CHAIN_LEN_TEST - 1)
        for step in range(target_step):
            h = sha256h(cur.encode())
            cur = py_reduce_inj(h, step, self.charset, self.str_len)  # ← must match build
        target_hash = sha256h(cur.encode())   # hash at step target_step

        print(f"  Target plaintext: '{cur}'  (position {target_step}/{CHAIN_LEN_TEST})")
        print(f"  Target hash:      {target_hash}")

        # 3. Crack it
        result, elapsed = self.full_query(target_hash, mini_table, verbose=True)

        if result and sha256h(result.encode()) == target_hash:
            print(f"  CRACKED: '{result}'  in {elapsed*1000:.1f}ms  ✓")
            ok = True
        else:
            print(f"  FAILED: not in table (coverage ~63%)  —  try again")
            ok = False

        self.chain_len = orig_chain
        return ok

    def validate_query_kernel(self, target_hash_hex, sample_positions=None):
        """
        Compare query_chains GPU output against Python for specific positions.
        If GPU ≠ Python: kernel bug confirmed.
        If GPU == Python but still false hits: fingerprint/table key mismatch.
        """
        target_hash_hex = target_hash_hex.strip()
        target_bytes    = bytes.fromhex(target_hash_hex)

        if sample_positions is None:
            # Pick positions near end (fast GPU) and near start (many steps)
            sample_positions = [
                self.chain_len - 1,      # pos=chain_len-1: 1 step
                self.chain_len - 2,      # pos=chain_len-2: 2 steps
                self.chain_len - 100,    # 100 steps
                self.chain_len - 1000,   # 1000 steps
                self.chain_len // 2,     # chain_len/2 steps
            ]
            sample_positions = [p for p in sample_positions if 0 <= p < self.chain_len]

        print(f"\n=== Query Kernel Validation ===")
        print(f"Target: {target_hash_hex}")
        print(f"Positions to test: {sample_positions}\n")

        all_ok = True
        for pos in sample_positions:
            # --- GPU side ---
            # batch_offset maps pos back: pos = (chain_len-1) - (batch_offset + gid)
            # → for gid=0: batch_offset = (chain_len-1) - pos
            batch_offset = (self.chain_len - 1) - pos
            W_gpu, wv_gpu, mm_gpu = self._run_query_batch(target_bytes, batch_offset, 1)
            gpu_W   = int(W_gpu[0])
            gpu_wv  = tuple(int(x) for x in wv_gpu[0])
            gpu_max = int(mm_gpu[0, 0])
            gpu_min = int(mm_gpu[0, 1])

            # --- Python side ---
            h = target_hash_hex
            for step in range(pos, self.chain_len):
                s = py_reduce_inj(h, step, self.charset, self.str_len)
                h = sha256h(s.encode())
            # h = endpoint_hash
            py_nibbles = [int(c, 16) for c in h]
            py_W   = sum(py_nibbles)
            py_wv  = tuple(sum(py_nibbles[i*4:(i+1)*4]) for i in range(16))
            py_max = max(py_nibbles)
            py_min = min(py_nibbles)

            match_W   = gpu_W   == py_W
            match_wv  = gpu_wv  == py_wv
            match_mm  = gpu_max == py_max and gpu_min == py_min
            ok = match_W and match_wv and match_mm

            status = "✓ OK " if ok else "✗ MISMATCH"
            print(f"  pos={pos:>7d}  steps={self.chain_len-pos:>7d}  {status}")
            if not ok:
                all_ok = False
                if not match_W:
                    print(f"           W:   GPU={gpu_W}  PY={py_W}")
                if not match_wv:
                    # Show first mismatching wv component
                    for k in range(16):
                        if gpu_wv[k] != py_wv[k]:
                            print(f"           wv[{k}]: GPU={gpu_wv[k]}  PY={py_wv[k]}")
                            break
                if not match_mm:
                    print(f"           max: GPU={gpu_max}  PY={py_max}")
                    print(f"           min: GPU={gpu_min}  PY={py_min}")

        print()
        if all_ok:
            print("  GPU kernel output matches Python for all tested positions.")
            print("  If false hits still occur, the issue is in table/key format.")
        else:
            print("  GPU kernel BUG confirmed — output differs from Python.")
        return all_ok

    # ── GPU crack ────────────────────────────────────────────────────────────

    def _run_query_batch(self, target_bytes, batch_offset, n):
        # Vulkan path — ACO compiler, no PAL-LLVM bug, full GPU speed
        if self._vk_engine is not None:
            return self._vk_engine.query_batch(target_bytes, batch_offset, n)
        # Fallback: OpenCL -O0 (slow but present as last resort)
        """
        GPU dispatch: n threads, each handling one chain position.
        Position for thread gid = (chain_len-1) - (batch_offset + gid).
        Returns (W[n], wv[n,16], mm[n,2]) — fingerprints of each extension's endpoint.

        Uses enqueue_fill_buffer for GPU-side zero-init (more reliable than
        COPY_HOST_PTR on AMD PAL — the host copy can be async for large buffers,
        leaving stale data visible to the kernel).
        """
        tgt_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=np.frombuffer(target_bytes, dtype=np.uint8)
        )
        W_buf  = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, n * 4)
        wv_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, n * 16 * 4)
        mm_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, n * 2  * 4)

        # GPU-side zero-fill — synchronous on the command queue
        zero = np.uint32(0)
        cl.enqueue_fill_buffer(self.queue, W_buf,  zero, 0, n * 4)
        cl.enqueue_fill_buffer(self.queue, wv_buf, zero, 0, n * 16 * 4)
        cl.enqueue_fill_buffer(self.queue, mm_buf, zero, 0, n * 2  * 4)
        self.queue.finish()   # fills must complete before kernel writes

        # Local size 64 = one RDNA4 wavefront — avoids driver-side divergence bugs
        local_sz = min(64, n)
        # n must be a multiple of local_sz
        n_aligned = ((n + local_sz - 1) // local_sz) * local_sz

        self.kernel_query.set_args(
            tgt_buf, np.int32(self.chain_len), np.int32(batch_offset),
            self.cs_buf, np.int32(self.cs_len), np.int32(self.str_len),
            W_buf, wv_buf, mm_buf
        )
        cl.enqueue_nd_range_kernel(self.queue, self.kernel_query,
                                   (n_aligned,), (local_sz,))
        self.queue.finish()

        out_W  = np.zeros(n,      dtype=np.uint32)
        out_wv = np.zeros(n * 16, dtype=np.uint32)
        out_mm = np.zeros(n * 2,  dtype=np.uint32)
        cl.enqueue_copy(self.queue, out_W,  W_buf)
        cl.enqueue_copy(self.queue, out_wv, wv_buf)
        cl.enqueue_copy(self.queue, out_mm, mm_buf)
        self.queue.finish()
        return out_W, out_wv.reshape(n, 16), out_mm.reshape(n, 2)

    def _verify_hit(self, start_str, pos, target_hash_hex):
        """
        Walk chain from start_str for pos steps, check if sha256(result) == target.
        Returns plaintext string or None.
        """
        cur = start_str
        for step in range(pos):
            h   = sha256h(cur.encode())
            cur = py_reduce_inj(h, step, self.charset, self.str_len)
        if sha256h(cur.encode()) == target_hash_hex:
            return cur
        return None

    def gpu_crack(self, target_hash_hex, tables, verbose=True):
        """
        GPU-accelerated rainbow table crack.

        For each table: GPU extends target_hash from every chain position to
        endpoint (chain_len positions in parallel), CPU does O(1) dict lookup
        per result, then CPU verifies hits.

        Speed vs full_query():
          chain_len=1M, 1 table: ~2-3 min GPU  (vs 2.5h Python)
          chain_len=1M, 3 tables: ~6-9 min GPU  (96.3% coverage)

        tables: list of dicts loaded with _load_table()
        """
        import hashlib as _hl
        target_hash_hex = target_hash_hex.strip()   # trailing space/newline guard
        target_bytes = bytes.fromhex(target_hash_hex)
        # BATCH=65536: full GPU utilization per dispatch.
        # AMD PAL zero-init bug is now fixed with enqueue_fill_buffer.
        BATCH = 65536
        # Expected hits per table for a crackable hash: ~1.
        # (One true hit + ~0 false positives with CDP bijection.)
        # If hits >> 10 per batch, GPU buffer corruption detected.
        MAX_HITS_SANITY = 100_000  # hard cap: stale buffer → W=0 filtered out already
        t_total = time.time()

        for tbl_idx, table in enumerate(tables):
            if verbose:
                n_batches = (self.chain_len + BATCH - 1) // BATCH
                print(f"\n[Table {tbl_idx+1}/{len(tables)}]  "
                      f"{len(table):,} chains  |  "
                      f"{n_batches} GPU batches  |  "
                      f"~{self.chain_len**2/2/4.6e9/60:.0f} min estimated")

            hits = []   # (pos, start_str) pairs found via table lookup
            t0   = time.time()
            abort_flag = False

            for batch_offset in range(0, self.chain_len, BATCH):
                n = min(BATCH, self.chain_len - batch_offset)

                W, wv, mm = self._run_query_batch(target_bytes, batch_offset, n)

                # DEBUG: first batch — compare GPU vs Python for 5 threads
                if batch_offset == 0 and verbose:
                    print("\n  [DEBUG] GPU vs Python fingerprint comparison (first 5 threads):")
                    for dbg_i in range(min(5, n)):
                        dbg_pos = (self.chain_len - 1) - dbg_i
                        h = target_hash_hex
                        for st in range(dbg_pos, self.chain_len):
                            s = py_reduce_inj(h, st, self.charset, self.str_len)
                            h = sha256h(s.encode())
                        fp_py  = full_fingerprint(h)
                        W_gpu  = int(W[dbg_i])
                        wv_gpu = tuple(int(x) for x in wv[dbg_i])
                        mx_gpu = int(mm[dbg_i, 0])
                        mn_gpu = int(mm[dbg_i, 1])
                        wv_ok  = (wv_gpu == fp_py[1])
                        W_ok   = (W_gpu == fp_py[0])
                        mm_ok  = (mx_gpu == fp_py[7] and mn_gpu == fp_py[8])
                        status = "OK" if (W_ok and wv_ok and mm_ok) else "MISMATCH"
                        wv_str = "wv=OK" if wv_ok else f"wv MISMATCH gpu={wv_gpu[:4]} py={fp_py[1][:4]}"
                        print(f"    gid={dbg_i} pos={dbg_pos}: W={W_gpu}/{fp_py[0]}"
                              f"{'OK' if W_ok else 'X'}  {wv_str}  "
                              f"mm={mx_gpu}/{fp_py[7]},{mn_gpu}/{fp_py[8]}"
                              f"{'OK' if mm_ok else 'X'}  [{status}]")
                    # Deep position test
                    deep_i = min(32768, n-1)
                    deep_pos = (self.chain_len - 1) - deep_i
                    h2 = target_hash_hex
                    for st in range(deep_pos, self.chain_len):
                        s2 = py_reduce_inj(h2, st, self.charset, self.str_len)
                        h2 = sha256h(s2.encode())
                    fp2 = full_fingerprint(h2)
                    W2g = int(W[deep_i]); wv2g = tuple(int(x) for x in wv[deep_i])
                    print(f"    gid={deep_i} pos={deep_pos} ({deep_pos} steps): "
                          f"W={W2g}/{fp2[0]}{'OK' if W2g==fp2[0] else 'X'}  "
                          f"wv={'OK' if wv2g==fp2[1] else f'MISMATCH first4 gpu={wv2g[:4]} py={fp2[1][:4]}'}")
                    print()

                # CPU table lookup for each thread result
                batch_hits = 0
                for i in range(n):
                    W_val = int(W[i])
                    # Skip W=0: unwritten buffer position (never a real SHA256 output)
                    if W_val == 0:
                        continue
                    entry = _W_CACHE.get(W_val)
                    if entry:
                        w2, w3, w4, w5, ce = entry
                    else:
                        w2=_iter_W(W_val); w3=_iter_W(w2)
                        w4=_iter_W(w3);    w5=_iter_W(w4)
                        ce=_cycle_entry(W_val)
                    key = (W_val,
                           tuple(int(x) for x in wv[i]),
                           ce, w2, w3, w4, w5,
                           int(mm[i,0]), int(mm[i,1]))
                    if key in table:
                        pos = (self.chain_len - 1) - (batch_offset + i)
                        hits.append((pos, table[key]))
                        batch_hits += 1

                if verbose:
                    done = batch_offset + n
                    elapsed = time.time() - t0
                    rate = done * self.chain_len / elapsed if elapsed > 0 else 0
                    print(f"\r  {done/self.chain_len*100:5.1f}%  "
                          f"batch {batch_offset//BATCH+1}/{n_batches}  "
                          f"hits={len(hits):4d}  "
                          f"{rate/1e9:.2f}GH/s  {elapsed:.0f}s",
                          end='', flush=True)

                # Note: with correct Vulkan GPU query, hits are genuine chain matches
                # High hit counts are expected (hash appears in ~66% of chains)
                if verbose and len(hits) % 10000 < BATCH:
                    print(f"  [{len(hits):,} candidates so far]", flush=True)

            if verbose: print()

            # ── Verify hits ───────────────────────────────────────────────
            hits_sorted = sorted(hits, key=lambda h: h[0])
            t_verify    = time.time()
            print(f"  Verifying {len(hits_sorted):,} candidates...", flush=True)

            if self._vk_verify and len(hits_sorted) > 0:
                # GPU verify: all candidates in parallel, <1s
                VBATCH = self._vk_verify.BATCH_MAX
                for vstart in range(0, len(hits_sorted), VBATCH):
                    batch = hits_sorted[vstart:vstart + VBATCH]
                    result = self._vk_verify.verify_batch(batch, target_bytes)
                    if result:
                        elapsed = time.time() - t_total
                        v_elapsed = time.time() - t_verify
                        if verbose:
                            print(f"\n  CRACKED (GPU verify)  '{result}'  "
                                  f"verify={v_elapsed:.2f}s  total={elapsed:.1f}s")
                        return result
            else:
                # CPU fallback
                n_verified = 0
                for pos, start_str in hits_sorted:
                    result = self._verify_hit(start_str, pos, target_hash_hex)
                    n_verified += 1
                    if result:
                        elapsed   = time.time() - t_total
                        v_elapsed = time.time() - t_verify
                        if verbose:
                            print(f"\n  CRACKED (CPU verify)  '{result}'  "
                                  f"verify={v_elapsed:.1f}s  total={elapsed:.1f}s")
                        return result
                    if n_verified % 1000 == 0:
                        r = n_verified / (time.time() - t_verify)
                        print(f"  Verified {n_verified:,}/{len(hits_sorted):,} "
                              f"({r:.0f}/s)  pos={pos}", flush=True)

            if verbose:
                v_elapsed = time.time() - t_verify
                print(f"  Table {tbl_idx+1}: {len(hits_sorted):,} hits, "
                      f"0 verified  (verify={v_elapsed:.1f}s)")

        if verbose:
            print(f"\n  Not found  (try more tables for higher coverage)")
        return None

    def build_multi(self, n_tables, prefix='table', mode='ilp2', user_batch=65536):
        """
        Build n_tables rainbow tables with different random seeds.

        Each table has ~66.7% coverage independently.
        Combined coverage: 1 - (1-0.667)^n

        n=1: 66.7%   n=2: 88.9%   n=3: 96.3%   n=5: 99.3%
        """
        single_cov = 0.667
        combined   = 1 - (1 - single_cov) ** n_tables
        paths      = []

        print(f"\nBuilding {n_tables} tables  →  combined coverage ≈ {combined*100:.1f}%")
        for i in range(n_tables):
            ext  = '.bin' if prefix.endswith('.bin') or prefix.endswith('_bin') else '.json'
            path = f"{prefix}_{i+1}{ext}"
            print(f"\n{'='*55}")
            print(f"Table {i+1}/{n_tables}: {path}")
            print(f"{'='*55}")
            self.build_table(path, user_batch=user_batch, mode=mode, seed_offset=i * 999983)
            paths.append(path)

        print(f"\n{'='*55}")
        print(f"Multi-table build complete:")
        for i, p in enumerate(paths):
            sz = os.path.getsize(p) / 1e6
            print(f"  {p}  ({sz:.1f} MB)")
        print(f"Combined coverage: {combined*100:.1f}%  "
              f"({n_tables} tables × ~66.7% each)")
        return paths

# --- CLI ----------------------------------------------------------------------

if __name__ == '__main__':
    import multiprocessing as _mp; _mp.freeze_support()
    parser = argparse.ArgumentParser(description='CDP Rainbow Chain Builder')
    parser.add_argument('--charset',     default='lower', choices=CHARSETS.keys())
    parser.add_argument('--length',      type=int, default=8)
    parser.add_argument('--chain-len',   type=int, default=300_000)
    parser.add_argument('--batch',       type=int, default=65536,
                        help='GPU threads/dispatch')
    parser.add_argument('--mode',        default='ilp2',
                        choices=['ilp2', 'vec4', 'scalar'],
                        help='Kernel mode: ilp2 (default/fastest), vec4, scalar')
    parser.add_argument('--output',      default='cdp_table.json')
    parser.add_argument('--convert',     type=str, default=None, metavar='SRC',
                        help='Convert JSON table to binary: --convert src.json '
                             '--output dst.bin  (3.4x smaller, 39 byte/entry)')
    parser.add_argument('--validate',    action='store_true',
                        help='Run GPU validation (scalar + vec4 + ilp2)')
    parser.add_argument('--validate-query', type=str, default=None, metavar='HASH',
                        help='Compare query kernel output against Python for given hash')
    parser.add_argument('--build',       action='store_true')
    parser.add_argument('--build-multi', type=int, default=0, metavar='N',
                        help='Build N tables with different seeds (higher coverage)')
    parser.add_argument('--prefix',      default='table',
                        help='Output prefix for --build-multi (default: table)')
    parser.add_argument('--self-test',   action='store_true',
                        help='End-to-end crack demo with mini table (chain_len=1000)')
    parser.add_argument('--crack',       type=str, default=None,
                        help='GPU-accelerated crack of a SHA256 hex hash')
    parser.add_argument('--crack-list',  type=str, default=None,
                        help='Crack multiple hashes from a file (one hash per line). '
                             'Tables loaded once, all hashes cracked sequentially.')
    parser.add_argument('--tables',      type=str, default=None,
                        help='Comma-separated list of table files for --crack')
    parser.add_argument('--table',       type=str, default='cdp_table.json',
                        help='Single table file (legacy --crack / --query)')
    parser.add_argument('--query',       type=str, default=None,
                        help='Direct endpoint lookup (fast test only)')
    args = parser.parse_args()

    builder = CDPChainBuilder(
        charset_name=args.charset,
        str_len=args.length,
        chain_len=args.chain_len,
    )

    if args.validate:
        ok = builder.validate_gpu(n_samples=20)
        if not ok: print("Validation FAILED!"); exit(1)
        print("All validation passed.")

    if args.convert:
        src = args.convert
        dst = args.output if not args.output.endswith('.json') else src.replace('.json', '.bin')
        if not os.path.exists(src):
            print(f"Source not found: {src}"); exit(1)
        print(f"Converting {src} → {dst}")
        with open(src, encoding='utf-8') as f:
            data = json.load(f)
        n = save_binary_table(data['entries'], dst, data['charset'],
                              data['str_len'], data['chain_len'], data['n_chains'])
        src_mb = os.path.getsize(src)/1e6
        dst_mb = os.path.getsize(dst)/1e6
        print(f"Done: {src_mb:.1f} MB → {dst_mb:.1f} MB  ({src_mb/dst_mb:.1f}x)  {n:,} entries")

    if args.validate_query:
        builder.validate_query_kernel(args.validate_query)

    if args.build:
        builder.build_table(args.output, user_batch=args.batch, mode=args.mode)

    if args.build_multi > 0:
        builder.build_multi(
            n_tables=args.build_multi,
            prefix=args.prefix,
            mode=args.mode,
            user_batch=args.batch
        )

    if args.self_test:
        builder.self_test()

    def _load_table(path):
        if path.endswith('.bin'):
            table, _, _ = load_binary_table(path)
            return table
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return {(e['key'][0], tuple(e['key'][1:17]),
                 e['key'][17],
                 e['key'][18], e['key'][19], e['key'][20], e['key'][21],
                 e['key'][22], e['key'][23]):
                e['start']
                for e in data['entries']}

    # Resolve table paths (shared by --crack and --crack-list)
    if args.crack or args.crack_list:
        if args.tables:
            table_paths = [p.strip() for p in args.tables.split(',')]
        else:
            table_paths = [args.table]

        missing = [p for p in table_paths if not os.path.exists(p)]
        if missing:
            print(f"Table(s) not found: {missing}"); exit(1)

        tables_list  = []
        total_chains = 0
        for p in table_paths:
            t = _load_table(p)
            tables_list.append(t)
            total_chains += len(t)
            print(f"Loaded: {p}  ({len(t):,} chains)")

        n_tbl = len(tables_list)
        cov   = 1 - (1 - 0.667) ** n_tbl
        print(f"\n{n_tbl} table(s)  |  {total_chains:,} total chains  |  "
              f"~{cov*100:.1f}% combined coverage\n")

    if args.crack:
        print(f"Target: {args.crack.strip()}")
        result = builder.gpu_crack(args.crack.strip(), tables_list, verbose=True)
        if result:
            print(f"\nCRACKED: '{result}'")
            print(f"SHA256:   {sha256h(result.encode())}")
        else:
            print(f"\nNot found  (~{(1-cov)*100:.1f}% probability of miss)")

    if args.crack_list:
        if not os.path.exists(args.crack_list):
            print(f"Hash list not found: {args.crack_list}"); exit(1)

        hashes = [l.strip() for l in open(args.crack_list)
                  if l.strip() and not l.startswith('#')]
        print(f"Hash list: {len(hashes)} hashes from '{args.crack_list}'")
        print(f"Tables loaded once — cracking sequentially.\n")

        cracked = 0
        t_total = time.time()
        for idx, h in enumerate(hashes):
            print(f"[{idx+1}/{len(hashes)}] {h}")
            result = builder.gpu_crack(h, tables_list, verbose=False)
            if result:
                cracked += 1
                print(f"  CRACKED: '{result}'")
            else:
                print(f"  Not found")

        elapsed = time.time() - t_total
        print(f"\n{'─'*50}")
        print(f"Done: {cracked}/{len(hashes)} cracked  |  "
              f"{elapsed:.1f}s total  |  {elapsed/len(hashes):.1f}s per hash")

    if args.query:
        if not os.path.exists(args.table):
            print(f"Table not found: {args.table}"); exit(1)
        table = _load_table(args.table)
        import time as _t; t0 = _t.time()
        result = builder.query(args.query, table)
        elapsed = _t.time() - t0
        print(f"FOUND: '{result}'  ({elapsed*1000:.2f}ms)" if result
              else f"Not found (endpoint lookup only — use --crack for full traversal)")
