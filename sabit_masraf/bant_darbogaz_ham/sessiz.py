import fcntl, subprocess, sys, time
fs = [open(f"/private/tmp/atolye-slot-{i}.lock", "w") for i in range(1, 6)]
t0 = time.time()
for f in fs:
    fcntl.flock(f, fcntl.LOCK_EX)
print(f"kilitler alindi, bekleme {time.time()-t0:.0f} sn", flush=True)
time.sleep(20)
try:
    r = subprocess.run(sys.argv[1:], timeout=1150)
finally:
    for f in fs:
        f.close()
    print(f"kilitler birakildi, pencere {time.time()-t0:.0f} sn", flush=True)
