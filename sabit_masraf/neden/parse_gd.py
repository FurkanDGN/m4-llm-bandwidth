import re, sys, collections

RE_NODE = re.compile(r"ggml_metal_op_encode_impl: node\[\s*(\d+)\] - (\S+)\s+(\S*)\s*(\(concurrent\))?")
RE_SRC = re.compile(r"ggml_metal_op_encode_impl: src(\d) -\s+(\S+)\s+\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\] \[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\], (\d), (\S+)")
RE_DST = re.compile(r"ggml_metal_op_encode_impl: node\s+-\s+(\S+)\s+\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\] \[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\], 1, (\S+)")
RE_ADD = re.compile(r"ggml_mem_ranges_add_(src|dst): add \w+ range buf=(\d+), \[(\d+), (\d+)\)")
RE_OV = re.compile(r"ggml_mem_ranges_check: the (src|dst) range buf=(\d+), \[(\d+), (\d+)\) overlaps with a previous (src|dst) range buf=(\d+), \[(\d+), (\d+)\)")
RE_FUSE = re.compile(r"fuse (\d+) ops")


def parse(path):
    graphs = []
    cur = None
    node = None
    pending_ov = []
    for line in open(path, errors="ignore"):
        m = RE_NODE.search(line)
        if m:
            idx = int(m.group(1))
            if idx == 0 and (cur is None or cur["nodes"]):
                cur = {"nodes": []}
                graphs.append(cur)
            if cur is None:
                cur = {"nodes": []}
                graphs.append(cur)
            node = {"idx": idx, "op": m.group(2), "name": m.group(3), "conc": bool(m.group(4)),
                    "srcs": [], "dst": None, "adds": [], "ov": pending_ov, "fuse": 1}
            pending_ov = []
            cur["nodes"].append(node)
            continue
        m = RE_OV.search(line)
        if m:
            pending_ov.append((m.group(1), int(m.group(3)), int(m.group(4)), m.group(5), int(m.group(7)), int(m.group(8))))
            continue
        if node is None:
            continue
        m = RE_SRC.search(line)
        if m:
            node["srcs"].append((int(m.group(1)), m.group(12), (int(m.group(3)), int(m.group(4)))))
            continue
        m = RE_DST.search(line)
        if m:
            node["dst"] = (m.group(10), (int(m.group(2)), int(m.group(3))))
            continue
        m = RE_FUSE.search(line)
        if m:
            node["fuse"] = int(m.group(1))
            continue
        m = RE_ADD.search(line)
        if m:
            node["adds"].append((m.group(1), int(m.group(3)), int(m.group(4))))
    return graphs


def is_target_3col(g):
    if not g["nodes"]:
        return False
    n = g["nodes"][0]
    return g["nodes"][-1]["idx"] > 600 and n["srcs"] and n["srcs"][0][2] == (3584, 3)


def simulate(g, strict):
    ranges = []
    barriers = []
    for n in g["nodes"]:
        if not n["adds"]:
            continue
        conflict = None
        for t, p0, p1 in n["adds"]:
            for rt, q0, q1 in ranges:
                if t == "src" and rt == "src":
                    continue
                hit = (p0 < q1 and p1 > q0) if strict else (p0 < q1 and p1 >= q0)
                if hit:
                    conflict = (t, p0, p1, rt, q0, q1)
                    break
            if conflict:
                break
        if conflict:
            barriers.append((n, conflict))
            ranges = []
        ranges.extend(n["adds"])
    return barriers


if __name__ == "__main__":
    for path in sys.argv[1:]:
        gs = parse(path)
        t3 = [g for g in gs if is_target_3col(g)]
        print("==", path, "graf:", len(gs), "hedef 3 sutun (>900 dugum):", len(t3))
        for i, g in enumerate(t3):
            marks = sum(1 for n in g["nodes"] if n["adds"] and not n["conc"])
            print("  graf", i, "dugum", len(g["nodes"]), "concurrent isaretsiz (adds olan):", marks,
                  "sim>=:", len(simulate(g, False)), "sim>:", len(simulate(g, True)))
