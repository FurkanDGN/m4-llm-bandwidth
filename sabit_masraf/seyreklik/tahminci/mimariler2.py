import sys, time, numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
NX,NF=3584,18944
kat=int(sys.argv[1]); veri='~/claude-scratch/tahminci/veri_h'
x=np.fromfile(f'{veri}/x_k{kat}.f16',dtype=np.float16).reshape(-1,NX).astype(np.float32)
h=np.fromfile(f'{veri}/h_k{kat}.f16',dtype=np.float16).reshape(-1,NF).astype(np.float32)
a=np.abs(h); n=x.shape[0]; nv=n//5; vi=np.arange(n-nv,n); ti=np.arange(0,n-nv)
def kutle(score, idx):
    k=NF//2; th=np.partition(score,NF-k,axis=1)[:,NF-k][:,None]; sel=score>=th
    aa=a[idx]; return ((aa*sel).sum(1)/aa.sum(1)).mean()
print(f"kat {kat} token {n}  kahin %.3f"%kutle(a[vi],vi), flush=True)
y=np.log(a+1e-3).astype(np.float32); wgt=(a/a[ti].mean()).astype(np.float32)

sys.path.insert(0,'~/claude-scratch/llamacpp-ablasyon-sb/gguf-py')
from gguf import GGUFReader
r=GGUFReader('~/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf'); T={t.name:t for t in r.tensors}
def deq(raw,n_rows,n_k):
    b=raw.reshape(n_rows,n_k//32,18); d=b[:,:,:2].copy().view(np.float16).astype(np.float32)[:,:,0]; q=b[:,:,2:]
    lo=(q&0xF).astype(np.int8)-8; hi=(q>>4).astype(np.int8)-8
    return (np.concatenate([lo,hi],axis=2).astype(np.float32)*d[:,:,None]).reshape(n_rows,n_k)
t0=time.time()
Wg=deq(np.asarray(T[f'blk.{kat}.ffn_gate.weight'].data).reshape(-1),NF,NX); Wu=deq(np.asarray(T[f'blk.{kat}.ffn_up.weight'].data).reshape(-1),NF,NX)
Ug,Sg,Vgt=np.linalg.svd(Wg,full_matrices=False); Uu,Su,Vut=np.linalg.svd(Wu,full_matrices=False)
print("svd %d s"%(time.time()-t0), flush=True)

class Bilin(nn.Module):
    def __init__(s,r): super().__init__(); s.a1=nn.Linear(NX,r,bias=False); s.a2=nn.Linear(r,NF,bias=False); s.b1=nn.Linear(NX,r,bias=False); s.b2=nn.Linear(r,NF,bias=False)
    def __call__(s,x):
        g=s.a2(s.a1(x)); u=s.b2(s.b1(x)); return nn.silu(g)*u   # ham h tahmini
def svd_init(m,rk):
    m.a1.weight=mx.array((Vgt[:rk]*Sg[:rk,None]).astype(np.float32)); m.a2.weight=mx.array(Ug[:,:rk].astype(np.float32))
    m.b1.weight=mx.array((Vut[:rk]*Su[:rk,None]).astype(np.float32)); m.b2.weight=mx.array(Uu[:,:rk].astype(np.float32))
    mx.eval(m.parameters()); return m
def skor(m):
    return np.concatenate([np.abs(np.array(m(mx.array(x[vi[i:i+512]])))) for i in range(0,nv,512)])
def egit(m, kayip, epochs, lr, ad, bs=128):
    t0=time.time(); opt=optim.Adam(learning_rate=lr); st=nn.value_and_grad(m,kayip)
    print(f"{ad:44s} baslangic kutle %.3f"%kutle(skor(m),vi), flush=True)
    for ep in range(epochs):
        perm=np.random.permutation(len(ti))
        for i in range(0,len(ti)-bs+1,bs):
            idx=ti[perm[i:i+bs]]
            l,gr=st(m,mx.array(x[idx]),mx.array(h[idx]),mx.array(wgt[idx])); opt.update(m,gr); mx.eval(m.parameters(),opt.state)
        print(f"{ad:44s} epoch {ep} kutle %.3f (%d s)"%(kutle(skor(m),vi), time.time()-t0), flush=True)
# kayıplar: ham h'ye MSE (büyükler doğal ağırlıklı); log büyüklük ağırlıklı MSE
mse_h=lambda m,xb,hb,wb: nn.losses.mse_loss(m(xb),hb)
logw =lambda m,xb,hb,wb: mx.mean(wb*(mx.log(mx.abs(m(xb))+1e-3)-mx.log(mx.abs(hb)+1e-3))**2)
for rk in [512,1024]:
    egit(svd_init(Bilin(rk),rk), mse_h, 3, 1e-4, f"cift dogrusal r{rk} SVD baslangic, h MSE")
egit(svd_init(Bilin(1024),1024), logw, 3, 1e-4, "cift dogrusal r1024 SVD, log agirlikli")
rk=2048; m=svd_init(Bilin(rk),rk); print(f"cift dogrusal r2048 yalniz SVD                 kutle %.3f"%kutle(skor(m),vi), flush=True)
