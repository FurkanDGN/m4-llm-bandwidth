import sys, numpy as np
sys.path.insert(0, sys.argv[3])
import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Q

src, dst = sys.argv[1], sys.argv[2]
r = GGUFReader(src)

def deq_q4_0(raw, n_rows, n_k):
    b = raw.reshape(n_rows, n_k // 32, 18)
    d = b[:, :, :2].copy().view(np.float16).astype(np.float32)[:, :, 0]
    q = b[:, :, 2:]
    lo = (q & 0x0F).astype(np.int8) - 8
    hi = (q >> 4).astype(np.int8) - 8
    v = np.concatenate([lo, hi], axis=2).astype(np.float32) * d[:, :, None]
    return v.reshape(n_rows, n_k)

def quant_q4_0(x):
    n_rows, n_k = x.shape
    b = x.reshape(n_rows, n_k // 32, 32)
    amax_i = np.abs(b).argmax(axis=2)
    mx = np.take_along_axis(b, amax_i[:, :, None], axis=2)[:, :, 0]
    d = mx / -8.0
    idv = np.where(d != 0, 1.0 / np.where(d == 0, 1, d), 0.0)
    q = np.clip((b * idv[:, :, None] + 8.5).astype(np.int32), 0, 15).astype(np.uint8)
    lo, hi = q[:, :, :16], q[:, :, 16:]
    packed = (lo | (hi << 4)).astype(np.uint8)
    dh = d.astype(np.float16).view(np.uint8).reshape(n_rows, n_k // 32, 2)
    return np.concatenate([dh, packed], axis=2).reshape(n_rows, -1)

w = GGUFWriter(dst, r.fields["general.architecture"].parts[-1].tobytes().decode())
for k, f in r.fields.items():
    if k.startswith("GGUF."): continue
    if k == "general.architecture": continue
    val = f.contents()
    if f.types[0] == gguf.GGUFValueType.ARRAY:
        w.add_array(k, val)
    else:
        w.add_key_value(k, val, f.types[0])
n_done = 0
for t in r.tensors:
    arr = t.data
    shape = list(reversed([int(x) for x in t.shape]))
    if t.name.endswith("ffn_down.weight") and t.tensor_type == Q.Q4_0:
        n_rows, n_k = shape  # ne=[n_k, n_rows] -> numpy (n_rows, n_k)
        raw = np.asarray(arr).reshape(-1)
        x = deq_q4_0(raw, n_rows, n_k)          # (3584, 18944)
        xt = np.ascontiguousarray(x.T)          # (18944, 3584)
        qt = quant_q4_0(xt)                      # (18944, 3584*18/32)
        w.add_tensor(t.name, qt, raw_dtype=Q.Q4_0)
        n_done += 1
    else:
        a = np.asarray(arr)
        w.add_tensor(t.name, a, raw_dtype=(t.tensor_type if a.dtype == np.uint8 else None))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(progress=False); w.close()
print("transpoze edilen down sayisi:", n_done)
