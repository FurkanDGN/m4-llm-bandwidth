import sys

# kullanim: python3 gen_mv9_instances.py v9|w  >> kernels_mv9.metal (bir kopyasina eklenir)
# v9: bench12.swift ad bicimi (v9_f{F}_g{G}_{rnd,p2,p3})
# w : bench13.swift ad bicimi (w_f{F}_g{G}_{rnd,m{gmode}p{P}})

mode = sys.argv[1]
out = []
if mode == "v9":
    for F in (0, 1, 2):
        out.append(f'template [[host_name("v9_f{F}_g0")]] kernel decltype(mv9<{F}, 8, 0, 2, 3>) mv9<{F}, 8, 0, 2, 3>;')
        for G in range(1, 5):
            out.append(f'template [[host_name("v9_f{F}_g{G}_rnd")]] kernel decltype(mv9<{F}, 8, {G}, 2, 3>) mv9<{F}, 8, {G}, 2, 3>;')
            for P in (2, 3):
                out.append(f'template [[host_name("v9_f{F}_g{G}_p{P}")]] kernel decltype(mv9<{F}, 8, {G}, 1, {P}>) mv9<{F}, 8, {G}, 1, {P}>;')
elif mode == "w":
    for F in (0, 1, 2):
        out.append(f'template [[host_name("w_f{F}_g0")]] kernel decltype(mv9<{F}, 8, 0, 2, 3>) mv9<{F}, 8, 0, 2, 3>;')
        for G in (2, 3, 4):
            for (gm, P) in [(1, 2), (3, 2), (4, 2), (5, 2), (6, 2), (1, 3), (3, 3), (4, 3), (2, 3)]:
                tag = "rnd" if gm == 2 else f"m{gm}p{P}"
                out.append(f'template [[host_name("w_f{F}_g{G}_{tag}")]] kernel decltype(mv9<{F}, 8, {G}, {gm}, {P}>) mv9<{F}, 8, {G}, {gm}, {P}>;')
else:
    raise SystemExit("v9 veya w")
print("\n".join(out))
