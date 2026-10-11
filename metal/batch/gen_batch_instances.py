import sys

# kullanim: python3 gen_batch_instances.py k6|k7|k8|k8r|k14  >> ilgili .metal dosyasinin bir kopyasina eklenir
# k6  : kernels6.metal  -> s_b{B}_m{0,1}_g0 (+ s_b1_m0_g1)
# k7  : kernels7.metal  -> m7_b{B}_{m0,m1p3,m2} (+ m7_b1_m1p{1,2,4})
# k8  : kernels8 (kernels14.metal'in dot()'lu ve MODE 3'suz hali) -> m8_b{B}_{m0,m1p3,m2} (+ m8_b1_m1p{1,2,4})
# k8r : R taramasi -> m8_b{B}_r{R}_{m0,m1p3,m2}
# k14 : kernels14.metal -> m8_b{B}_r{R}_{m0,m1p3,m2,m3}

mode = sys.argv[1]
out = []


def Rof(B):
    return 8 if B <= 8 else (4 if B <= 12 else 2)


if mode == "k6":
    for B in range(1, 17):
        R = 8 if B <= 8 else 4
        for M in (0, 1):
            out.append(f'template [[host_name("s_b{B}_m{M}_g0")]] kernel decltype(mm_s<{B}, {R}, {M}, 0>) mm_s<{B}, {R}, {M}, 0>;')
    out.append('template [[host_name("s_b1_m0_g1")]] kernel decltype(mm_s<1, 8, 0, 1>) mm_s<1, 8, 0, 1>;')
elif mode == "k7":
    for B in range(1, 17):
        R = Rof(B)
        out.append(f'template [[host_name("m7_b{B}_m0")]] kernel decltype(mm7<{B}, {R}, 0, 3>) mm7<{B}, {R}, 0, 3>;')
        out.append(f'template [[host_name("m7_b{B}_m1p3")]] kernel decltype(mm7<{B}, {R}, 1, 3>) mm7<{B}, {R}, 1, 3>;')
        out.append(f'template [[host_name("m7_b{B}_m2")]] kernel decltype(mm7<{B}, {R}, 2, 3>) mm7<{B}, {R}, 2, 3>;')
    for P in (1, 2, 4):
        out.append(f'template [[host_name("m7_b1_m1p{P}")]] kernel decltype(mm7<1, 8, 1, {P}>) mm7<1, 8, 1, {P}>;')
elif mode == "k8":
    for B in range(1, 17):
        R = Rof(B)
        out.append(f'template [[host_name("m8_b{B}_m0")]] kernel decltype(mm8<{B}, {R}, 0, 3>) mm8<{B}, {R}, 0, 3>;')
        out.append(f'template [[host_name("m8_b{B}_m1p3")]] kernel decltype(mm8<{B}, {R}, 1, 3>) mm8<{B}, {R}, 1, 3>;')
        out.append(f'template [[host_name("m8_b{B}_m2")]] kernel decltype(mm8<{B}, {R}, 2, 3>) mm8<{B}, {R}, 2, 3>;')
    for P in (1, 2, 4):
        out.append(f'template [[host_name("m8_b1_m1p{P}")]] kernel decltype(mm8<1, 8, 1, {P}>) mm8<1, 8, 1, {P}>;')
elif mode in ("k8r", "k14"):
    modes = [(0, "m0"), (1, "m1p3"), (2, "m2")] + ([(3, "m3")] if mode == "k14" else [])
    for B in range(1, 17):
        for R in (1, 2, 4, 8):
            if R * B > 64:
                continue
            for M, nm in modes:
                out.append(f'template [[host_name("m8_b{B}_r{R}_{nm}")]] kernel decltype(mm8<{B}, {R}, {M}, 3>) mm8<{B}, {R}, {M}, 3>;')
else:
    raise SystemExit("k6|k7|k8|k8r|k14")
print("\n".join(out))
