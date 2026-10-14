# GGUF'a katman başına seçim ağırlığı tensörleri ekler: blk.N.ffn_sel_up.weight (ort|silu g|), blk.N.ffn_sel_down.weight (down satır normu)
import sys, struct, numpy as np
sys.path.insert(0, sys.argv[5])
import gguf
from gguf import GGUFReader, GGUFWriter
src, dst, ist_yol, norm_yol = sys.argv[1:5]

def oku(yol):
    f = open(yol, 'rb'); nk, ne0 = struct.unpack('<ii', f.read(8)); (n,) = struct.unpack('<d', f.read(8))
    out = []
    for k in range(nk):
        g = np.frombuffer(f.read(8*ne0), dtype=np.float64); u = np.frombuffer(f.read(8*ne0), dtype=np.float64); h = np.frombuffer(f.read(8*ne0), dtype=np.float64)
        out.append((g, u, h))
    return nk, ne0, n, out
nk, ne0, n, ist = oku(ist_yol)
nk2, ne02, _, nrm = oku(norm_yol)
assert nk == nk2 == 28 and ne0 == ne02 == 18944
r = GGUFReader(src)
w = GGUFWriter(dst, r.fields["general.architecture"].parts[-1].tobytes().decode())
for k, f in r.fields.items():
    if k.startswith("GGUF.") or k == "general.architecture": continue
    val = f.contents()
    if f.types[0] == gguf.GGUFValueType.ARRAY: w.add_array(k, val)
    else: w.add_key_value(k, val, f.types[0])
for t in r.tensors:
    a = np.asarray(t.data)
    w.add_tensor(t.name, a, raw_dtype=(t.tensor_type if a.dtype == np.uint8 else None))
for k in range(nk):
    sel_up = (ist[k][0] / n).astype(np.float32)
    sel_down = nrm[k][0].astype(np.float32)
    w.add_tensor(f"blk.{k}.ffn_sel_up.weight", sel_up)
    w.add_tensor(f"blk.{k}.ffn_sel_down.weight", sel_down)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(progress=False); w.close()
print("eklendi:", 2*nk, "tensor")
