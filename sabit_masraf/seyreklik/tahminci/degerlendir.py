# Seçim yöntemlerinin |h| kütlesini ne kadar yakaladığını ölçer (seçilen %50).
import sys, numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
NX,NF=3584,18944
kat=int(sys.argv[1]); veri='~/claude-scratch/tahminci/veri_h'
x=np.fromfile(f'{veri}/x_k{kat}.f16',dtype=np.float16).reshape(-1,NX).astype(np.float32)
h=np.fromfile(f'{veri}/h_k{kat}.f16',dtype=np.float16).reshape(-1,NF).astype(np.float32)
a=np.abs(h); n=x.shape[0]; nv=n//5
print(f"kat {kat} token {n}")
def kutle(score, idx):   # score: (m, NF) büyük = seç
    k=NF//2; th=np.partition(score,NF-k,axis=1)[:,NF-k][:,None]; sel=score>=th
    aa=a[idx]; return ((aa*sel).sum(1)/aa.sum(1)).mean(), (sel & (aa>=np.partition(aa,NF-k,axis=1)[:,NF-k][:,None])).sum(1).mean()/k
vi=np.arange(n-nv,n)
print("kahin        kutle %.3f kapsama %.3f"%kutle(a[vi],vi))
print("rastgele     kutle %.3f kapsama %.3f"%kutle(np.random.rand(nv,NF),vi))
# tam gate (üst sınır: gate'i tam hesaplayınca)
sys.path.insert(0,'~/claude-scratch/llamacpp-ablasyon-sb/gguf-py')
from gguf import GGUFReader
r=GGUFReader('~/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf'); T={t.name:t for t in r.tensors}
def deq(raw,n_rows,n_k):
    b=raw.reshape(n_rows,n_k//32,18); d=b[:,:,:2].copy().view(np.float16).astype(np.float32)[:,:,0]; q=b[:,:,2:]
    lo=(q&0xF).astype(np.int8)-8; hi=(q>>4).astype(np.int8)-8
    return (np.concatenate([lo,hi],axis=2).astype(np.float32)*d[:,:,None]).reshape(n_rows,n_k)
Wg=deq(np.asarray(T[f'blk.{kat}.ffn_gate.weight'].data).reshape(-1),NF,NX); Wu=deq(np.asarray(T[f'blk.{kat}.ffn_up.weight'].data).reshape(-1),NF,NX)
g=x[vi]@Wg.T; u=x[vi]@Wu.T
print("tam gate     kutle %.3f kapsama %.3f"%kutle(np.abs(g/(1+np.exp(-g))),vi))
print("tam gate*up  kutle %.3f kapsama %.3f (doğrulama: kahin olmalı)"%kutle(np.abs(g/(1+np.exp(-g))*u),vi))
# SVD düşük rank
try:
    S=np.load(f'~/claude-scratch/tahminci/agirlik/svd_k{kat}.npz')
    for rk in [128,256,512]:
        gl=(x[vi]@S[f'gate_A{rk}'])@S[f'gate_B{rk}'].T; ul=(x[vi]@S[f'up_A{rk}'])@S[f'up_B{rk}'].T
        print(f"svd r{rk}     kutle %.3f kapsama %.3f"%kutle(np.abs(gl/(1+np.exp(-gl))*ul),vi))
except FileNotFoundError: print("svd yok")
# eğitimli MLP: hedef log|h|, MSE
for rk in [256]:
    class M(nn.Module):
        def __init__(s): super().__init__(); s.l1=nn.Linear(NX,rk); s.l2=nn.Linear(rk,NF)
        def __call__(s,x): return s.l2(nn.relu(s.l1(x)))
    m=M(); opt=optim.Adam(learning_rate=1e-3)
    y=np.log(a+1e-3).astype(np.float32); ym=y[:-nv].mean(); ys=y[:-nv].std()
    def lf(m,xb,yb): return nn.losses.mse_loss(m(xb),yb)
    st=nn.value_and_grad(m,lf); xs=x/(x.std()+1e-6)
    for ep in range(6):
        perm=np.random.permutation(n-nv)
        for i in range(0,n-nv-255,256):
            idx=perm[i:i+256]; l,gr=st(m,mx.array(xs[idx]),mx.array((y[idx]-ym)/ys)); opt.update(m,gr); mx.eval(m.parameters(),opt.state)
    sc=np.array(m(mx.array(xs[vi])))
    print(f"mlp r{rk} log|h| kutle %.3f kapsama %.3f"%kutle(sc,vi))
