// FFN-only timing harness: 28 layers of (gate, up, down) q4_0, residual chain, Metal backend.
// mode: dense | down | up   (sparse paths are selected by the Metal backend env knobs; this only builds the graph shape)
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const int NE = 3584, NF = 18944;

int main(int argc, char ** argv) {
    std::string mode = argc > 1 ? argv[1] : "dense";
    const int nl    = argc > 2 ? atoi(argv[2]) : 28;
    const int iters = argc > 3 ? atoi(argv[3]) : 20;
    const int kat_max = getenv("SEYREK_UP_KAT_MAX") ? atoi(getenv("SEYREK_UP_KAT_MAX")) : 1000;

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { fprintf(stderr, "GPU yok\n"); return 1; }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);

    const size_t n_tensors = (size_t) nl*3 + 1;
    ggml_init_params ip = { ggml_tensor_overhead()*(n_tensors + 1) + 1024, nullptr, true };
    ggml_context * wctx = ggml_init(ip);
    std::vector<ggml_tensor *> G(nl), U(nl), D(nl);
    for (int l = 0; l < nl; ++l) {
        G[l] = ggml_new_tensor_2d(wctx, GGML_TYPE_Q4_0, NE, NF);
        U[l] = ggml_new_tensor_2d(wctx, GGML_TYPE_Q4_0, NE, NF);
        D[l] = (mode == "dense") ? ggml_new_tensor_2d(wctx, GGML_TYPE_Q4_0, NF, NE)
                                 : ggml_new_tensor_2d(wctx, GGML_TYPE_Q4_0, NE, NF);   // transposed: neuron rows
    }
    ggml_tensor * x0 = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, NE);
    ggml_tensor * hbuf = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, NF);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    if (!wbuf) { fprintf(stderr, "ağırlık tamponu yok\n"); return 1; }

    // random q4_0 bytes, d = fixed small half (0x1800 ≈ 0.00049)
    {
        std::mt19937 rng(42);
        const size_t nbytes = (size_t) NE*NF/32*18;
        std::vector<uint8_t> buf(nbytes);
        for (size_t i = 0; i < nbytes; ++i) buf[i] = (uint8_t) rng();
        for (size_t b = 0; b < nbytes; b += 18) { buf[b] = 0x00; buf[b+1] = 0x18; }
        for (int l = 0; l < nl; ++l) {
            for (size_t i = 0; i < nbytes; i += 4096) buf[i] = (uint8_t) rng();   // per-layer variation (not needed for timing)
            ggml_backend_tensor_set(G[l], buf.data(), 0, nbytes);
            ggml_backend_tensor_set(U[l], buf.data(), 0, nbytes);
            ggml_backend_tensor_set(D[l], buf.data(), 0, nbytes);
        }
        std::vector<float> xv(NE);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto & v : xv) v = nd(rng);
        ggml_backend_tensor_set(x0, xv.data(), 0, NE*sizeof(float));
        std::vector<float> hv(NF);
        for (auto & v : hv) v = nd(rng);
        ggml_backend_tensor_set(hbuf, hv.data(), 0, NF*sizeof(float));
    }

    ggml_init_params gp = { ggml_tensor_overhead()*(size_t) nl*16 + ggml_graph_overhead_custom(nl*16, false) + 4096, nullptr, true };
    ggml_context * gctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, nl*16, false);
    ggml_tensor * cur = x0;
    for (int l = 0; l < nl; ++l) {
        ggml_tensor * out;
        if (mode == "mv") {                       // yalnız up matvec (yoğun çekirdek)
            out = ggml_view_1d(gctx, ggml_mul_mat(gctx, U[l], cur), NE, 0);
        } else if (mode == "mvhist") {            // up matvec + histogram epilogu
            ggml_tensor * u = ggml_mul_mat(gctx, U[l], cur); u->op_params[1] = 0x5E7ED;
            out = ggml_view_1d(gctx, u, NE, 0);
        } else if (mode == "glu") {               // birleşik gate+up+swiglu
            ggml_tensor * g = ggml_mul_mat(gctx, G[l], cur);
            ggml_tensor * u = ggml_mul_mat(gctx, U[l], cur);
            out = ggml_view_1d(gctx, ggml_swiglu_split(gctx, g, u), NE, 0);
        } else if (mode == "upgate") {            // up-hist + seyrek gate, down yok
            ggml_tensor * u = ggml_mul_mat(gctx, U[l], cur); u->op_params[1] = 0x5E7ED;
            ggml_tensor * g = ggml_mul_mat(gctx, G[l], cur); g->op_params[1] = 0x5E7EE; g->src[2] = u;
            out = ggml_view_1d(gctx, g, NE, 0);
        } else if (mode == "downonly") {          // yoğun GLU + seyrek down (= down modu); burada: GLU yok, h = sabit vektör
            ggml_tensor * v = ggml_view_2d(gctx, D[l], NF, NE, (NF/32)*18, 0);
            out = ggml_mul_mat(gctx, v, hbuf); out->op_params[1] = 0x5E7EC;
        } else if (mode == "up" && l <= kat_max) {
            ggml_tensor * u = ggml_mul_mat(gctx, U[l], cur); u->op_params[1] = 0x5E7ED;
            ggml_tensor * g = ggml_mul_mat(gctx, G[l], cur); g->op_params[1] = 0x5E7EE; g->src[2] = u;
            ggml_tensor * v = ggml_view_2d(gctx, D[l], NF, NE, (NF/32)*18, 0);
            out = ggml_mul_mat(gctx, v, g); out->op_params[1] = 0x5E7EC;
        } else {
            ggml_tensor * g = ggml_mul_mat(gctx, G[l], cur);
            ggml_tensor * u = ggml_mul_mat(gctx, U[l], cur);
            ggml_tensor * h = ggml_swiglu_split(gctx, g, u);
            if (mode == "dense") {
                out = ggml_mul_mat(gctx, D[l], h);
            } else {
                ggml_tensor * v = ggml_view_2d(gctx, D[l], NF, NE, (NF/32)*18, 0);
                out = ggml_mul_mat(gctx, v, h); out->op_params[1] = 0x5E7EC;
            }
        }
        cur = ggml_add(gctx, x0, ggml_scale(gctx, out, 1e-4f));   // bağımlılık zinciri, büyüklük sabit
    }
    ggml_build_forward_expand(gf, cur);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(ga, gf)) { fprintf(stderr, "çizge ayrılamadı\n"); return 1; }

    for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    std::vector<double> ms;
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        auto t1 = std::chrono::high_resolution_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size()/2];
    std::vector<float> outv(NE);
    ggml_backend_tensor_get(cur, outv.data(), 0, NE*sizeof(float));
    double s = 0; for (float v : outv) s += v;
    printf("SONUC mode=%s nl=%d ortanca_ms=%.3f katman_us=%.1f min_ms=%.3f cikti_toplam=%.4g\n", mode.c_str(), nl, med, 1000.0*med/nl, ms[0], s);
    return 0;
}
