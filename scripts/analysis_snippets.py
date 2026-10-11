import csv
import re
import statistics

# Bu dosya, oturum sirasinda satir ici calistirilan analiz parcalarinin kaydidir (orijinaller kayboldu, yeniden yazildi).


# --- llama-speculative / llama-lookup ciktisindan sayilar ---
def speculative_stats(path):
    t = open(path, errors="replace").read()
    enc = re.search(r"encoded\s+(\d+) tokens in\s+([\d.]+) seconds", t)
    n = int(re.search(r"decoded\s+(\d+) tokens", t)[1])
    sec = float(re.search(r"decoded\s+\d+ tokens in\s+([\d.]+) seconds", t)[1])
    ac = int(re.search(r"n_accept\s+=\s+(\d+)", t)[1])
    dr = int(re.search(r"n_drafted\s+=\s+(\d+)", t)[1])
    pe = re.search(r"prompt eval time =\s+([\d.]+) ms", t)
    rounds = n - ac                      # her turda 1 hedef-ornekli token + kabul edilenler
    verify_ms = (float(pe[1]) - float(enc[2]) * 1000) / rounds
    draft_ms = sec * 1000 / rounds - verify_ms
    return dict(tok_s=n / sec, yield_per_round=n / rounds, verify_ms=verify_ms, draft_ms=draft_ms, accept=ac / dr if dr else None)


# --- llama-bench csv: parti suresi ms = n/ts*1000 ---
def batch_ms(path):
    return {int(r["n_prompt"]): int(r["n_prompt"]) / float(r["avg_ts"]) * 1000 for r in csv.DictReader(open(path))}


# --- test-backend-ops perf: sonuc satiri, kendisinden onceki MUL_MAT satirina aittir (derleme loglari araya girer) ---
def perf_us(path):
    t = re.sub(r"\x1b\[[0-9;]*m", "", open(path, errors="replace").read())
    out, cur = {}, None
    for line in t.split("\n"):
        m = re.search(r"MUL_MAT\(type_a=(\w+),type_b=f32,m=(\d+),n=(\d+),k=(\d+)", line)
        if m:
            cur = (m[1], int(m[2]), int(m[3]), int(m[4]))
        r = re.search(r"(\d+) runs -\s+([\d.]+) us/run", line)
        if r and cur:
            out[cur] = float(r[2])
            cur = None
    return out


# --- projeksiyon: olculen llama.cpp tur bilesenlerini, bizim cekirdek egrisiyle olcekle ---
meas = {1: (1.68, 38.9, 5.4), 2: (2.19, 54.8, 9.3), 3: (2.41, 76.3, 12.9), 4: (2.63, 111.5, 16.7),
        5: (2.77, 107.0, 20.3), 6: (2.90, 141.0, 24.1), 7: (2.97, 152.0, 27.4), 8: (3.11, 96.9, 30.5)}   # k: (tok/tur, dogrulama ms, taslak ms)
llama = {2: 1.92, 3: 2.77, 4: 3.75, 5: 5.59, 6: 5.34, 7: 7.00, 8: 7.47, 9: 5.09}                      # 7B, tek-token'a oran (llama.cpp olculen)
ours = {2: 1.00, 3: 1.09, 4: 1.66, 5: 1.66, 6: 1.66, 7: 1.66, 8: 1.66, 9: 3.10}                       # bizim en iyi okuma (mikro-olcum)
base_tok_s = 54.5


def projection():
    rows = []
    for k, (y, v, d) in meas.items():
        B = k + 1
        measured = y / ((v + d) / 1000)
        proj = y / ((v * ours[B] / llama[B] + d) / 1000)
        rows.append((k, B, measured, proj, proj / base_tok_s))
    return rows


# --- en iyi-of-iki okuma egrisi (bench11/15 ciktisindan): skaler OKU, MMA tm4/tm2/tm1 sutunlarinin minimumu, B=1'e oran ---
def best_of(rows):
    # rows: [(B, a, skaler_oku, tm4, tm2, tm1, skaler_turev, mma_turev, skaler_rast, mma_rast)] ms
    b1 = min(x for x in rows[0][2:6] if x)
    return {r[0]: min(x for x in r[2:6] if x) / b1 for r in rows}


# --- sihirli sayi kimligi: nibble n (0..15) -> half 0x6400|n = 1024+n (tam), cikarma 1032 -> n-8 (tam) ---
