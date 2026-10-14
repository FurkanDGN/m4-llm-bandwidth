import sys, struct, numpy as np
sys.path.insert(0, '~/claude-scratch/llamacpp-ablasyon-sb/gguf-py')
from gguf import GGUFReader
from gguf.quants import dequantize
r = GGUFReader('~/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf')
t = {x.name: x for x in r.tensors}
nk = 28; ne0 = 18944
out = open(sys.argv[1], 'wb')
out.write(struct.pack('<ii', nk, ne0)); out.write(struct.pack('<d', 1.0))
for k in range(nk):
    x = t[f'blk.{k}.ffn_down.weight']
    w = dequantize(x.data, x.tensor_type)   # shape (n_embd, n_ff)
    assert w.shape == (3584, ne0), w.shape
    nrm = np.sqrt((w.astype(np.float64)**2).sum(axis=0))
    one = np.ones(ne0)
    out.write(nrm.tobytes()); out.write(one.tobytes()); out.write(nrm.tobytes())
    print(k, float(nrm.mean()), float(nrm.min()), float(nrm.max()), flush=True)
out.close()
