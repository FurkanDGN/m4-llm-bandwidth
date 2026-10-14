import sys, os, time, numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim

veri = sys.argv[1]; katlar = [int(k) for k in sys.argv[2].split(',')]; r = int(sys.argv[3]) if len(sys.argv) > 3 else 128
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
NX, NF = 3584, 18944

def yukle(k):
    x = np.fromfile(f'{veri}/x_k{k}.f16', dtype=np.float16).reshape(-1, NX)
    mb = np.fromfile(f'{veri}/m_k{k}.bits', dtype=np.uint8).reshape(-1, NF // 8)
    m = np.unpackbits(mb, axis=1, bitorder='little')[:, :NF]
    assert x.shape[0] == m.shape[0]
    return x, m

class Tahminci(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(NX, r); self.l2 = nn.Linear(r, NF)
    def __call__(self, x):
        return self.l2(nn.relu(self.l1(x)))

def kapsama(model, x, m, bs=1024):
    tot = 0.0; n = 0
    for i in range(0, x.shape[0], bs):
        s = model(mx.array(x[i:i+bs].astype(np.float32)))
        s = np.array(s)
        k = NF // 2
        th = np.partition(s, NF - k, axis=1)[:, NF - k][:, None]
        pred = s >= th
        mm = m[i:i+bs].astype(bool)
        tot += ((pred & mm).sum(1) / mm.sum(1)).sum(); n += mm.shape[0]
    return tot / n

for k in katlar:
    t0 = time.time()
    x, m = yukle(k)
    n = x.shape[0]; nv = max(512, n // 10)
    xt, mt, xv, mv = x[:-nv], m[:-nv], x[-nv:], m[-nv:]
    model = Tahminci()
    opt = optim.Adam(learning_rate=1e-3)
    def loss_fn(model, xb, mb):
        s = model(xb)
        return nn.losses.binary_cross_entropy(s, mb, with_logits=True, reduction='mean')
    step = nn.value_and_grad(model, loss_fn)
    bs = 256
    for ep in range(epochs):
        perm = np.random.permutation(xt.shape[0]); tl = 0; nb = 0
        for i in range(0, xt.shape[0] - bs + 1, bs):
            idx = perm[i:i+bs]
            xb = mx.array(xt[idx].astype(np.float32)); mb = mx.array(mt[idx].astype(np.float32))
            l, g = step(model, xb, mb); opt.update(model, g); mx.eval(model.parameters(), opt.state)
            tl += l.item(); nb += 1
        if ep == 0 or ep == epochs - 1:
            print(f"kat {k} epoch {ep} loss {tl/nb:.4f} dogrulama_kapsama {kapsama(model, xv, mv):.4f} ({time.time()-t0:.0f} s)", flush=True)
    W1 = np.array(model.l1.weight); b1 = np.array(model.l1.bias); W2 = np.array(model.l2.weight); b2 = np.array(model.l2.bias)
    np.savez(f'{os.path.dirname(veri)}/agirlik/k{k}_r{r}.npz', W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"kat {k} tamam, token {n}, rastgele taban 0.5", flush=True)
