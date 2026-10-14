import sys, numpy as np
NX,NF=3584,18944
kat=int(sys.argv[1]); veri='~/claude-scratch/tahminci/veri_h'
x=np.fromfile(f'{veri}/x_k{kat}.f16',dtype=np.float16).reshape(-1,NX).astype(np.float32)[-1000:]
h=np.fromfile(f'{veri}/h_k{kat}.f16',dtype=np.float16).reshape(-1,NF).astype(np.float32)[-1000:]; a=np.abs(h)
sys.path.insert(0,'~/claude-scratch/llamacpp-ablasyon-sb/gguf-py')
from gguf import GGUFReader
r=GGUFReader('~/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf'); T={t.name:t for t in r.tensors}
def deq(raw,n_rows,n_k):
    b=raw.reshape(n_rows,n_k//32,18); d=b[:,:,:2].copy().view(np.float16).astype(np.float32)[:,:,0]; q=b[:,:,2:]
    lo=(q&0xF).astype(np.int8)-8; hi=(q>>4).astype(np.int8)-8
    return (np.concatenate([lo,hi],axis=2).astype(np.float32)*d[:,:,None]).reshape(n_rows,n_k)
Wu=deq(np.asarray(T[f'blk.{kat}.ffn_up.weight'].data).reshape(-1),NF,NX); Wg=deq(np.asarray(T[f'blk.{kat}.ffn_gate.weight'].data).reshape(-1),NF,NX)
Wd=deq(np.asarray(T[f'blk.{kat}.ffn_down.weight'].data).reshape(-1),3584,NF)   # (3584, 18944)
u=x@Wu.T; g=x@Wg.T; au=np.abs(u); ag=np.abs(g/(1+np.exp(-g)))
y=h@Wd.T; ny=np.linalg.norm(y,axis=1)
def sec(score,frac):
    k=int(NF*(1-frac)); th=np.partition(score,NF-k,axis=1)[:,NF-k][:,None]; return score>=th
def rapor(ad,sel):
    hp=np.where(sel,h,0); yp=hp@Wd.T
    top=sec(a,0.995)  # en büyük %0.5
    print(f"{ad:28s} h hata {np.mean(np.linalg.norm(h-hp,axis=1)/np.linalg.norm(h,axis=1)):.3f}  cikis hata {np.mean(np.linalg.norm(y-yp,axis=1)/ny):.3f}  en buyuk %0.5 kapsama {(sel&top).sum()/top.sum():.3f}")
rapor("kahin %50", sec(a,0.5)); rapor("kahin %60", sec(a,0.6)); rapor("kahin %70", sec(a,0.7))
rapor("up %40", sec(au,0.4)); rapor("up %50", sec(au,0.5))
rapor("gate %50", sec(ag,0.5))
rapor("up %50 + gate %50 birlesim", sec(au,0.5)|sec(ag,0.5))
rapor("up*? |u|*|g| sirasi %50", sec(au*ag,0.5))
# down satır normu ağırlıklı kahin: |h_i|*||D_i||
dn=np.linalg.norm(Wd,axis=0)
rapor("kahin |h|*||D_i|| %50", sec(a*dn[None,:],0.5))
rapor("up |u|*||D_i|| %50", sec(au*dn[None,:],0.5))
