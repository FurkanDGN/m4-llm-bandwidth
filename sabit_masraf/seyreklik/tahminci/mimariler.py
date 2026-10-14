import sys, time, numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
NX,NF=3584,18944
kat=int(sys.argv[1]); veri='~/claude-scratch/tahminci/veri_h'
x=np.fromfile(f'{veri}/x_k{kat}.f16',dtype=np.float16).reshape(-1,NX).astype(np.float32)
h=np.fromfile(f'{veri}/h_k{kat}.f16',dtype=np.float16).reshape(-1,NF).astype(np.float32)
a=np.abs(h); n=x.shape[0]; nv=n//5; vi=np.arange(n-nv,n); ti=np.arange(0,n-nv)
xs=(x/(x.std()+1e-6)).astype(np.float32)
def kutle(score, idx):
    k=NF//2; th=np.partition(score,NF-k,axis=1)[:,NF-k][:,None]; sel=score>=th
    aa=a[idx]; return ((aa*sel).sum(1)/aa.sum(1)).mean()
print(f"kat {kat} token {n}  kahin %.3f rastgele 0.500"%kutle(a[vi],vi), flush=True)

# ağırlıklı hedef: w = |h| / ortalama, kayıp = w * (s - log|h|)^2  ("büyükleri doğru sırala")
y=np.log(a+1e-3).astype(np.float32); ym=y[ti].mean(); ys=y[ti].std(); yn=(y-ym)/ys
wgt=(a/ a[ti].mean()).astype(np.float32)

class MLP(nn.Module):
    def __init__(s,r): super().__init__(); s.l1=nn.Linear(NX,r); s.l2=nn.Linear(r,NF)
    def __call__(s,x): return s.l2(nn.relu(s.l1(x)))
class Bilin(nn.Module):
    def __init__(s,r): super().__init__(); s.a1=nn.Linear(NX,r,bias=False); s.a2=nn.Linear(r,NF,bias=False); s.b1=nn.Linear(NX,r,bias=False); s.b2=nn.Linear(r,NF,bias=False)
    def __call__(s,x):
        g=s.a2(s.a1(x)); u=s.b2(s.b1(x)); return mx.log(mx.abs(nn.silu(g)*u)+1e-3)
class BilinGenis(nn.Module):  # gate ve up ayrı ayrı MLP ile yaklaşık, sonra çarpım
    def __init__(s,r): super().__init__(); s.a1=nn.Linear(NX,r); s.a2=nn.Linear(r,NF); s.b1=nn.Linear(NX,r); s.b2=nn.Linear(r,NF)
    def __call__(s,x):
        g=s.a2(nn.relu(s.a1(x))); u=s.b2(nn.relu(s.b1(x))); return mx.log(mx.abs(nn.silu(g)*u)+1e-3)

def egit(model, kayip, epochs=6, lr=1e-3, bs=256, ad=""):
    t0=time.time(); opt=optim.Adam(learning_rate=lr); st=nn.value_and_grad(model,kayip)
    for ep in range(epochs):
        perm=np.random.permutation(len(ti))
        for i in range(0,len(ti)-bs+1,bs):
            idx=ti[perm[i:i+bs]]
            l,gr=st(model,mx.array(xs[idx]),mx.array(yn[idx]),mx.array(wgt[idx])); opt.update(model,gr); mx.eval(model.parameters(),opt.state)
    sc=np.concatenate([np.array(model(mx.array(xs[vi[i:i+1024]]))) for i in range(0,nv,1024)])
    print(f"{ad:40s} kutle %.3f  (%d s)"%(kutle(sc,vi), time.time()-t0), flush=True)

mse=lambda m,xb,yb,wb: nn.losses.mse_loss(m(xb),yb)
wmse=lambda m,xb,yb,wb: mx.mean(wb*(m(xb)-yb)**2)
egit(MLP(1024), mse, ad="MLP r1024, log|h| MSE")
egit(MLP(1024), wmse, ad="MLP r1024, agirlikli MSE")
egit(Bilin(256), mse, ad="cift dogrusal r256, MSE")
egit(Bilin(512), mse, ad="cift dogrusal r512, MSE")
egit(Bilin(512), wmse, ad="cift dogrusal r512, agirlikli MSE")
egit(BilinGenis(512), mse, ad="gate/up ayri MLP r512 carpim, MSE")

# düşük bitli ön izleme: gate ve up ağırlıklarını 32'lik bloklarla 2 bite indir, yaklaşık h ile seç
sys.path.insert(0,'~/claude-scratch/llamacpp-ablasyon-sb/gguf-py')
from gguf import GGUFReader
r=GGUFReader('~/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf'); T={t.name:t for t in r.tensors}
def deq(raw,n_rows,n_k):
    b=raw.reshape(n_rows,n_k//32,18); d=b[:,:,:2].copy().view(np.float16).astype(np.float32)[:,:,0]; q=b[:,:,2:]
    lo=(q&0xF).astype(np.int8)-8; hi=(q>>4).astype(np.int8)-8
    return (np.concatenate([lo,hi],axis=2).astype(np.float32)*d[:,:,None]).reshape(n_rows,n_k)
def qbits(W,bits):
    b=W.reshape(W.shape[0],-1,32); mx_=np.abs(b).max(2,keepdims=True); lv=2**(bits-1)-0.5
    d=mx_/lv; q=np.clip(np.round(b/np.where(d==0,1,d)),-lv-0.5,lv-0.5); return (q*d).reshape(W.shape)
Wg=deq(np.asarray(T[f'blk.{kat}.ffn_gate.weight'].data).reshape(-1),NF,NX); Wu=deq(np.asarray(T[f'blk.{kat}.ffn_up.weight'].data).reshape(-1),NF,NX)
xv=x[vi]
for bits in [2,3]:
    g=xv@qbits(Wg,bits).T; u=xv@qbits(Wu,bits).T
    print(f"{bits} bit on izleme (gate*up)                 kutle %.3f"%kutle(np.abs(g/(1+np.exp(-g))*u),vi), flush=True)
g=xv@qbits(Wg,2).T; u=xv@Wu.T
print("2 bit gate * tam up                        kutle %.3f"%kutle(np.abs(g/(1+np.exp(-g))*u),vi))
