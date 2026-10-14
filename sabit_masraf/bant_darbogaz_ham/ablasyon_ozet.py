import csv, statistics as st, sys
rows = [r for r in csv.DictReader(open('ablasyon_sonuc.csv')) if r.get('ms_token') and r['model'] != 'BITTI' and r['ms_token'][0].isdigit()]
modeller = {'05': '0.5B q4_0', '7': '7B q4_0'}
sira_ad = []
for r in rows:
    if r['yapilandirma'] not in sira_ad:
        sira_ad.append(r['yapilandirma'])
for m, ad in modeller.items():
    print(f"\n=== {ad} ===")
    taban = [float(r['ms_token']) for r in rows if r['model'] == m and r['yapilandirma'] == 'taban']
    tb = st.median(taban)
    print(f"taban: {[round(x,3) for x in taban]} ms/token (medyan {tb:.3f}; en kucuk-en buyuk fark %{(max(taban)/min(taban)-1)*100:.1f})")
    for y in sira_ad:
        if y == 'taban':
            continue
        v = [float(r['ms_token']) for r in rows if r['model'] == m and r['yapilandirma'] == y]
        if not v:
            continue
        med = st.median(v)
        print(f"{y:12s} {[round(x,3) for x in v]}  medyan {med:.3f} ms  | tabandan dusen {tb-med:+.3f} ms (%{(tb-med)/tb*100:.1f})")
