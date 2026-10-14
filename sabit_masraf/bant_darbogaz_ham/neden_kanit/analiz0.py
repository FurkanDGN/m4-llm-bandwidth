import importlib.util, os
spec = importlib.util.spec_from_file_location("pg", os.path.expanduser("~/claude-scratch/bant-darbogaz/neden_kanit/parse_gd.py"))
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
gs = p.parse(os.path.expanduser("~/claude-scratch/bant-darbogaz/neden_kanit/gd_512.err"))
for i,g in enumerate(gs):
    n=g["nodes"]
    print(i, len(n), n[-1]["idx"] if n else None, n[0]["op"] if n else None, n[0]["srcs"][:1] if n else None)
