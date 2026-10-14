import importlib.util, os, re, sys, collections
D = os.path.expanduser("~/claude-scratch/bant-darbogaz/neden_kanit/")
spec = importlib.util.spec_from_file_location("pg", D + "parse_gd.py")
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)

def load(path):
    gs = [g for g in p.parse(D + path) if p.is_target_3col(g)]
    return gs[-1]

def table(path, layer=5):
    g = load(path)
    lo = min(a[1] for n in g["nodes"] for a in n["adds"])
    rows = []
    for n in g["nodes"]:
        if n["op"] != "MUL_MAT" or len(n["adds"]) < 3:
            continue
        if not (n["name"].endswith("-%d" % layer) or n["name"] in ("node_%d" % (26 + 34 * layer),)):
            continue
        s1 = n["adds"][1]; ds = n["adds"][2]
        rows.append((n["name"], s1[1] - lo, s1[2] - s1[1], ds[1] - lo, ds[2] - ds[1]))
    return rows

for path in sys.argv[1:]:
    print("==", path)
    for r in table(path):
        print("  %-12s src1=%10d (%7d B) dst=%10d (%7d B)  src1%%256=%3d dst%%256=%3d" % (r[0], r[1], r[2], r[3], r[4], r[1] % 256, r[3] % 256))
