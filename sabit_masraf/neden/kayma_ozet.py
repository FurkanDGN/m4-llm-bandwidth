import csv, statistics as st
d = {}
for r in csv.DictReader(open("kayma_tek2.csv")):
    if r["tok_s"] is None or r["shift"] is None or not r["shift"].isdigit(): continue
    d.setdefault((r["model"], int(r["shift"])), []).append(float(r["tok_s"]))
for m in ("05", "15"):
    if (m, 0) not in d: continue
    base = d[(m, 0)]
    print("model", m)
    for sh in (0, 512, 1024, 1536):
        v = d.get((m, sh), [])
        if not v: continue
        n = min(len(v), len(base))
        hizli = sum(1 for x, y in zip(base, v) if y > x)
        print("  kayma %4d  n=%d  ort %.2f tok/s (%.3f ms/token)  fark %+.1f%%  kayma 0'dan hizli %d/%d tur  tur %s" % (sh, len(v), st.mean(v), 1000/st.mean(v), (st.mean(v)/st.mean(base)-1)*100, hizli, n, [round(x, 1) for x in v]))
