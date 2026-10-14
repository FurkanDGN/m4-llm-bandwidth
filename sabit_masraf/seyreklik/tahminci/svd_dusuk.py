import sys, numpy as np
sys.path.insert(0,'~/claude-scratch/llamacpp-ablasyon-sb/gguf-py')
from gguf import GGUFReader
r=GGUFReader('~/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf')
def deq(raw,n_rows,n_k):
    b=raw.reshape(n_rows,n_k//32,18); d=b[:,:,:2].copy().view(np.float16).astype(np.float32)[:,:,0]; q=b[:,:,2:]
    lo=(q&0xF).astype(np.int8)-8; hi=(q>>4).astype(np.int8)-8
    return (np.concatenate([lo,hi],axis=2).astype(np.float32)*d[:,:,None]).reshape(n_rows,n_k)
T={t.name:t for t in r.tensors}
for kat in [int(k) for k in sys.argv[1].split(',')]:
    out={}
    for ad in ['gate','up']:
        t=T[f'blk.{kat}.ffn_{ad}.weight']; W=deq(np.asarray(t.data).reshape(-1),18944,3584)  # (18944, 3584)
        U,S,Vt=np.linalg.svd(W,full_matrices=False)
        for rk in [128,256,512]:
            out[f'{ad}_A{rk}']=(Vt[:rk].T*S[:rk]).astype(np.float32)  # (3584, rk)
            out[f'{ad}_B{rk}']=U[:,:rk].astype(np.float32)             # (18944, rk)
        print(f"kat {kat} {ad} tekil deger payi r128 %.3f r256 %.3f r512 %.3f"%((S[:128]**2).sum()/(S**2).sum(),(S[:256]**2).sum()/(S**2).sum(),(S[:512]**2).sum()/(S**2).sum()),flush=True)
    np.savez(f'~/claude-scratch/tahminci/agirlik/svd_k{kat}.npz',**out)
