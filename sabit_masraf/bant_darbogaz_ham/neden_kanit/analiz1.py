import importlib.util, os, collections, re
D = os.path.expanduser("~/claude-scratch/bant-darbogaz/neden_kanit/")
spec = importlib.util.spec_from_file_location("pg", D + "parse_gd.py")
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)

def mm_align(path):
    gs = [g for g in p.parse(D + path) if p.is_target_3col(g)]
    g = gs[-1]
    stats = collections.defaultdict(lambda: collections.Counter())
    for n in g["nodes"]:
        if n["op"] != "MUL_MAT" or len(n["adds"]) < 3:
            continue
        kind = re.sub(r"-\d+$", "", n["name"])
        src1 = n["adds"][1][1]
        dst = n["adds"][2][1]
        stats[kind][("src1", src1 % 4096 == 0, src1 % 256 == 0, src1 % 128 == 0, src1 % 64 == 0)] += 1
        stats[kind][("dst", dst % 4096 == 0, dst % 256 == 0, dst % 128 == 0, dst % 64 == 0)] += 1
    return stats

import sys
for path in sys.argv[1:]:
    print("==", path, "(p0 % 4096==0, %256==0, %128==0, %64==0)")
    st = mm_align(path)
    for kind in sorted(st):
        print("  ", kind, dict(st[kind]))
