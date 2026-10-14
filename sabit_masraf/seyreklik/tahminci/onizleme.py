import sys, numpy as np
NX,NF=3584,18944
kat=int(sys.argv[1]); veri='~/claude-scratch/tahminci/veri_h'
x=np.fromfile(f'{veri}/x_k{kat}.f16',dtype=np.float16).reshape(-1,NX).astype(np.float32)
h=np.fromfile(f'{veri}/h_k{kat}.f16',dtype=np.float16).reshape(-1,NF).astype(np.float32)
a=np.abs(h); n=x.shape[0]; vi=np.arange(n-2000,n); xv=x[vi]; av=a[vi]; hv=h[vi]
sys.path.insert(0,'~/claude-scratch/llamacpp-ablasyon-sb/gguf-py')
from gguf import GGUFReader
r=GGUFReader('~/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf'); T={t.name:t for t in r.tensors}
def deq(raw,n_rows,n_k):
    b=raw.reshape(n_rows,n_k//32,18); d=b[:,:,:2].copy().view(np.float16).astype(np.float32)[:,:,0]; q=b[:,:,2:]
    lo=(q&0xF).astype(np.int8)-8; hi=(q>>4).astype(np.int8)-8
    return (np.concatenate([lo,hi],axis=2).astype(np.float32)*d[:,:,None]).reshape(n_rows,n_k)
Wg=deq(np.asarray(T[f'blk.{kat}.ffn_gate.weight'].data).reshape(-1),NF,NX); Wu=deq(np.asarray(T[f'blk.{kat}.ffn_up.weight'].data).reshape(-1),NF,NX)
def kutle(score):
    k=NF//2; th=np.partition(score,NF-k,axis=1)[:,NF-k][:,None]; sel=score>=th
    return ((av*sel).sum(1)/av.sum(1)).mean(), sel
def hata(sel, hyak):   # seçilmeyenler sıfır / ön izleme değeri: göreli L2 hata
    h0=np.where(sel,hv,0.0); h1=np.where(sel,hv,hyak)
    nrm=np.linalg.norm(hv,axis=1)
    return (np.linalg.norm(hv-h0,axis=1)/nrm).mean(), (np.linalg.norm(hv-h1,axis=1)/nrm).mean()
def q_bits(W,bits,blk):
    b=W.reshape(W.shape[0],-1,blk)
    if bits==1:
        s=np.abs(b).mean(2,keepdims=True); return (np.sign(b)*s).reshape(W.shape)
    if bits==1.58:
        s=np.abs(b).mean(2,keepdims=True); t=0.7*s; q=np.where(b>t,1,np.where(b<-t,-1,0)); sc=np.where((q!=0).sum(2,keepdims=True)>0,(np.abs(b)*(q!=0)).sum(2,keepdims=True)/np.maximum((q!=0).sum(2,keepdims=True),1),s); return (q*sc).reshape(W.shape)
    lv=2**(bits-1)-0.5; mx_=np.abs(b).max(2,keepdims=True); d=mx_/lv
    q=np.clip(np.round(b/np.where(d==0,1,d)),-lv-0.5,lv-0.5); return (q*d).reshape(W.shape)
def bayt(bits,blk):  # MB, iki matris, ölçek f16
    return 2*NF*NX*(bits/8 + 2/blk)/1e6
m0,_=kutle(av); print(f"kat {kat} kahin kutle {m0:.3f}; yoğun gate+up 76 MB")
for bits,blk in [(1,32),(1,128),(1,3584),(1.58,32),(1.58,128),(2,32),(2,128),(3,32),(4,32)]:
    g=xv@q_bits(Wg,bits,blk).T; u=xv@q_bits(Wu,bits,blk).T; hy=g/(1+np.exp(-g))*u
    m,sel=kutle(np.abs(hy)); e0,e1=hata(sel,hy)
    print(f"{bits:5} bit blok {blk:5d}: on izleme {bayt(bits,blk):5.1f} MB  kutle {m:.3f}  hata(sifir) {e0:.3f}  hata(onizleme) {e1:.3f}", flush=True)
# kıyas: kahin seçimde hata(sıfır)
_,sel=kutle(av); e0,_=hata(sel,np.zeros_like(hv)); print(f"kahin secim hata(sifir) {e0:.3f}")
print("--- seçimsiz, tüm gate/up düşük bitle (kıyas: saf kuantizasyon) ---")
for bits,blk in [(2,32),(3,32),(1.58,32)]:
    g=xv@q_bits(Wg,bits,blk).T; u=xv@q_bits(Wu,bits,blk).T; hy=g/(1+np.exp(-g))*u
    print(f"{bits:5} bit blok {blk}: {bayt(bits,blk):5.1f} MB  tam hata {(np.linalg.norm(hv-hy,axis=1)/np.linalg.norm(hv,axis=1)).mean():.3f}", flush=True)
print("--- karışık: 1 bit ön izleme ile seç, seçilenler q4 tam, kalanlar q2/q3 ---")
g1=xv@q_bits(Wg,1,128).T; u1=xv@q_bits(Wu,1,128).T; _,sel=kutle(np.abs(g1/(1+np.exp(-g1))*u1))
for bits in [2,3]:
    g=xv@q_bits(Wg,bits,32).T; u=xv@q_bits(Wu,bits,32).T; hy=g/(1+np.exp(-g))*u
    h1=np.where(sel,hv,hy); mb=19.1+38.2+bayt(bits,32)/2
    print(f"1bit sec + q4 secili + q{bits} kalan: ~{mb:5.1f} MB  hata {(np.linalg.norm(hv-h1,axis=1)/np.linalg.norm(hv,axis=1)).mean():.3f}", flush=True)
print("--- tek matrisle seçim ---")
u=xv@Wu.T; m,_=kutle(np.abs(u)); print(f"yalniz tam up: kutle {m:.3f}")
g=xv@Wg.T; m,_=kutle(np.abs(g/(1+np.exp(-g)))); print(f"yalniz tam gate: kutle {m:.3f}")
print("--- up'a göre seçim: hata ve oran ---")
u=xv@Wu.T; au=np.abs(u)
for frac in [0.4,0.5,0.6]:
    k=int(NF*(1-frac)); th=np.partition(au,NF-k,axis=1)[:,NF-k][:,None]; sel=au>=th
    h0=np.where(sel,hv,0.0); e=(np.linalg.norm(hv-h0,axis=1)/np.linalg.norm(hv,axis=1)).mean()
    print(f"up secim, sifirlanan {frac:.1f}: kutle {((av*sel).sum(1)/av.sum(1)).mean():.3f} hata {e:.3f}")
ak=av
for frac in [0.5,0.6]:
    k=int(NF*(1-frac)); th=np.partition(ak,NF-k,axis=1)[:,NF-k][:,None]; sel=ak>=th
    h0=np.where(sel,hv,0.0); print(f"kahin, sifirlanan {frac:.1f}: hata {(np.linalg.norm(hv-h0,axis=1)/np.linalg.norm(hv,axis=1)).mean():.3f}")
