import re, sys, collections
# kullanim: graf_say.py hata_dosyasi
# Bir karar adimi kayitta iki parca olarak gorunur (ilk parca 'norm-0' ile baslar, ikinci parca devamidir).
# Son iki grup birlestirilip tek-token karar adiminin GPU cagrilari sayilir.
satirlar = open(sys.argv[1], errors='ignore').read().splitlines()
gruplar, cur = [], None
for s in satirlar:
    m = re.search(r'node\[\s*(\d+)\]\s+-\s+(\S+)\s+(\S+)?(.*)$', s)
    if not m:
        continue
    op, ad, rest = m.group(2), m.group(3) or '', m.group(4)
    if op == 'RMS_NORM' and ad == 'norm-0':
        cur = []
        gruplar.append(cur)
    if cur is not None:
        cur.append((op, rest))
adim = gruplar[-2] + gruplar[-1]
calisan = [x for x in adim if '(noop)' not in x[1]]
c = collections.Counter(x[0] for x in calisan)
print(f"{sys.argv[1].split('/')[-1]}: token basina GPU cagrisi {len(calisan)} | MUL_MAT {c['MUL_MAT']} | kalan {len(calisan) - c['MUL_MAT']}")
print("  " + ", ".join(f"{op} {n}" for op, n in c.most_common()))
