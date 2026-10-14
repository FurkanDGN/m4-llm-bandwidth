import re, sys, collections
# kullanim: graf_say2.py hata_dosyasi
# GRAPH_DEBUG her komut tamponunu ayri yazar ve her biri node[0] ile baslar. Bir karar adimi iki tampondur.
# Son iki tampon birlestirilip tek-token karar adiminin GPU cagrilari sayilir (noop satirlari sayilmaz).
tamponlar, cur = [], None
for s in open(sys.argv[1], errors='ignore').read().splitlines():
    m = re.search(r'node\[\s*(\d+)\]\s+-\s+(\S+)\s+(\S+)?(.*)$', s)
    if not m:
        continue
    if int(m.group(1)) == 0:
        cur = []
        tamponlar.append(cur)
    if cur is not None and '(noop)' not in m.group(4):
        cur.append(m.group(2))
adim = tamponlar[-2] + tamponlar[-1]
c = collections.Counter(adim)
print(f"{sys.argv[1].split('/')[-1]}: token basina GPU cagrisi {len(adim)} | MUL_MAT {c['MUL_MAT']} | kalan {len(adim) - c['MUL_MAT']}")
print("  " + ", ".join(f"{op} {n}" for op, n in c.most_common()))
