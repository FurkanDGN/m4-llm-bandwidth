import importlib.util, os, re, sys, collections
D = os.path.expanduser("~/claude-scratch/bant-darbogaz/neden_kanit/")
spec = importlib.util.spec_from_file_location("pg", D + "parse_gd.py")
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
for path in sys.argv[1:]:
    gs = [g for g in p.parse(D + path) if p.is_target_3col(g)]
    g = gs[-1]
    st = collections.defaultdict(collections.Counter)
    for n in g["nodes"]:
        if n["op"] != "MUL_MAT" or len(n["adds"]) < 3:
            continue
        kind = re.sub(r"-\d+$", "", n["name"])
        if kind.startswith("node_"):
            kind = "attn_o"
        st[kind][n["adds"][1][1] % 1024] += 1
    print("==", path, "src1 adresi % 1024")
    for k in ("Qcur", "Kcur", "Vcur", "attn_o", "ffn_gate", "ffn_up", "ffn_out"):
        print("  %-9s %s" % (k, dict(st[k])))
