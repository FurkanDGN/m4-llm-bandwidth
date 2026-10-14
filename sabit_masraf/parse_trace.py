#!/usr/bin/env python3
"""metal-gpu-intervals dökümünden: llama-bench'in Compute kanalında GPU meşgul/boşluk analizi.
Soru (E1): token başına 1.63 ms sabit masraf GPU boşluğu mu, kernel içi mi?"""
import xml.etree.ElementTree as ET
import sys

yol = sys.argv[1]
# ref/id sözlüğü: aynı tag+id ilk görüldüğünde değer taşır, sonra ref ile anılır
idmap = {}

def deger(el):
    if el is None:
        return None
    r = el.get('ref')
    if r is not None:
        return idmap.get((el.tag, r))
    i = el.get('id')
    v = el.get('fmt') if el.tag in ('formatted-label', 'process') else (el.text or el.get('fmt') or '')
    if el.tag == 'formatted-label':
        v = el.get('fmt') or ''
    if i is not None:
        idmap[(el.tag, i)] = v
    return v

rows = []
for ev, el in ET.iterparse(yol, events=('end',)):
    if el.tag != 'row':
        continue
    start = dur = None
    kanal = ''
    etiket = ''
    for c in el:
        v = deger(c)
        if c.tag == 'start-time' and start is None:
            start = int(v)
        elif c.tag == 'duration' and dur is None:
            dur = int(v)
        elif c.tag == 'gpu-channel-name' and not kanal:
            kanal = v or ''
        elif c.tag == 'formatted-label' and not etiket:
            etiket = v or ''
    rows.append((start, dur, kanal, etiket))
    el.clear()

ll = [(s, d) for s, d, k, e in rows if 'llama-bench' in e and 'Compute' in k and s and d]
print(f"toplam satır {len(rows)}, llama-bench Compute {len(ll)}")
if not ll:
    kanallar = {}
    for s, d, k, e in rows[:200]:
        kanallar[k] = kanallar.get(k, 0) + 1
    print("kanallar:", kanallar)
    ornek = [e for s, d, k, e in rows if 'llama' in e.lower()][:3]
    print("llama etiketi örnekleri:", ornek)
    sys.exit(0)

ll.sort()
# aralık birleşimi (üst üste binenleri tek say)
bussy = 0
cs, ce = ll[0][0], ll[0][0] + ll[0][1]
for s, d in ll[1:]:
    e = s + d
    if s <= ce:
        ce = max(ce, e)
    else:
        bussy += ce - cs
        cs, ce = s, e
bussy += ce - cs
span = ll[-1][0] + ll[-1][1] - ll[0][0]
print(f"pencere: {span/1e9:.3f} s, GPU compute meşgul: {bussy/1e9:.3f} s, boşluk: {(span-bussy)/1e9:.3f} s ({100*(span-bussy)/span:.1f}%)")
print(f"encoder aralığı sayısı: {len(ll)}, ortalama süre: {bussy/len(ll)/1e6:.3f} ms")
# boşluk dağılımı
gaps = []
prev_end = None
for s, d in ll:
    if prev_end is not None and s > prev_end:
        gaps.append(s - prev_end)
    prev_end = max(prev_end or 0, s + d)
gaps.sort(reverse=True)
print(f"boşluk sayısı: {len(gaps)}, en büyük 5: {[f'{g/1e6:.2f} ms' for g in gaps[:5]]}")
if gaps:
    import statistics
    print(f"boşluk medyanı: {statistics.median(gaps)/1e3:.1f} µs, toplamı: {sum(gaps)/1e6:.1f} ms")
