import xml.etree.ElementTree as ET, sys, statistics as st
yol = sys.argv[1]
idmap = {}
def deger(el):
    if el is None: return None
    r = el.get('ref')
    if r is not None: return idmap.get((el.tag, r))
    i = el.get('id')
    v = el.get('fmt') if el.tag in ('formatted-label', 'process') else (el.text or el.get('fmt') or '')
    if el.tag == 'formatted-label': v = el.get('fmt') or ''
    if i is not None: idmap[(el.tag, i)] = v
    return v
rows = []
for ev, el in ET.iterparse(yol, events=('end',)):
    if el.tag != 'row': continue
    start = dur = None; kanal = ''; etiket = ''
    for c in el:
        v = deger(c)
        if c.tag == 'start-time' and start is None: start = int(v)
        elif c.tag == 'duration' and dur is None: dur = int(v)
        elif c.tag == 'gpu-channel-name' and not kanal: kanal = v or ''
        elif c.tag == 'formatted-label' and not etiket: etiket = v or ''
    rows.append((start, dur, kanal, etiket)); el.clear()
ll = sorted([(s, d) for s, d, k, e in rows if 'llama-speculative' in e and 'Compute' in k and s and d])
print(f"{yol}: llama-speculative Compute aralik sayisi {len(ll)}")
# uzun (hedef dogrulama, >12 ms) ve diger aralik siniflari
def sinif(d):
    ms = d / 1e6
    return 'uzun>12ms' if ms > 12 else ('orta3-12ms' if ms > 3 else 'kisa<3ms')
sayac = {}; sure = {}; bosluk_once = {}
prev_end = None
for s, d in ll:
    c = sinif(d)
    sayac[c] = sayac.get(c, 0) + 1
    sure.setdefault(c, []).append(d / 1e6)
    if prev_end is not None and s > prev_end:
        bosluk_once.setdefault(c, []).append((s - prev_end) / 1e6)
    prev_end = max(prev_end or 0, s + d)
pencere = (ll[-1][0] + ll[-1][1] - ll[0][0]) / 1e9
print(f"  pencere {pencere:.3f} s")
for c in ('uzun>12ms', 'orta3-12ms', 'kisa<3ms'):
    if c in sure:
        b = bosluk_once.get(c, [0])
        print(f"  {c:11s} adet {sayac[c]:4d} | ortalama sure {st.mean(sure[c]):7.3f} ms (medyan {st.median(sure[c]):7.3f}) | oncesindeki bosluk ortalama {st.mean(b):6.3f} ms (toplam {sum(b):7.1f} ms)")
