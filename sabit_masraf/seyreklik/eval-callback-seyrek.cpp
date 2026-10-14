#include "arg.h"
#include "common.h"
#include "debug.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cstdlib>
#include <string>
#include <vector>


#include <cmath>
#include <cstring>
#include <algorithm>
#include <map>

struct seyrek_istat { double n = 0; double alt[4] = {0,0,0,0}; double enerji_yari = 0; double satir = 0; };
static std::map<std::string, seyrek_istat> g_istat;
static const float g_esik[4] = {0.01f, 0.05f, 0.1f, 0.2f};

static bool seyrek_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    (void) user_data;
    const bool ilgili = strncmp(t->name, "ffn_swiglu", 10) == 0;
    if (ask) return ilgili;
    if (!ilgili) return true;
    if (std::getenv("SEYREK_MASKE")) {
        const int kat = atoi(t->name + 11);
        if (kat % 4 == 0) {
            const int64_t ne0 = t->ne[0], ne1 = t->ne[1];
            std::vector<float> buf(ne0 * ne1);
            for (int64_t j = 0; j < ne1; ++j) ggml_backend_tensor_get(t, buf.data() + j * ne0, j * t->nb[1], ne0 * sizeof(float));
            char fn[256]; snprintf(fn, sizeof(fn), "%s/maske_k%d.bin", std::getenv("SEYREK_MASKE"), kat);
            FILE * f = fopen(fn, "wb");
            std::vector<float> a(ne0), b(ne0);
            for (int64_t j = 0; j < ne1; ++j) {
                for (int64_t i = 0; i < ne0; ++i) a[i] = fabsf(buf[j*ne0 + i]);
                b = a; std::nth_element(b.begin(), b.begin() + ne0/2, b.end());
                const float th = b[ne0/2];
                std::vector<uint8_t> m(ne0);
                for (int64_t i = 0; i < ne0; ++i) m[i] = a[i] >= th ? 1 : 0;
                fwrite(m.data(), 1, ne0, f);
            }
            fclose(f);
        }
    }
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1];
    std::vector<float> buf(ne0 * ne1);
    for (int64_t j = 0; j < ne1; ++j) {
        ggml_backend_tensor_get(t, buf.data() + j * ne0, j * t->nb[1], ne0 * sizeof(float));
    }
    auto & st = g_istat[t->name];
    std::vector<float> a(ne0);
    for (int64_t j = 0; j < ne1; ++j) {
        const float * r = buf.data() + j * ne0;
        float mx = 0; double top = 0;
        for (int64_t i = 0; i < ne0; ++i) { a[i] = fabsf(r[i]); mx = std::max(mx, a[i]); top += a[i]; }
        for (int64_t i = 0; i < ne0; ++i) for (int e = 0; e < 4; ++e) if (a[i] < g_esik[e] * mx) st.alt[e] += 1;
        std::vector<float> b(a); std::nth_element(b.begin(), b.begin() + ne0/2, b.end(), std::greater<float>());
        double ust = 0; for (int64_t i = 0; i < ne0/2; ++i) ust += b[i];
        st.enerji_yari += ust / top;
        st.n += ne0; st.satir += 1;
    }
    return true;
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);

    if (tokens.empty()) {
        LOG_ERR("%s : there are not input tokens to process - (try to provide a prompt with '-p')\n", __func__);
        return false;
    }

    LOG_INF("number of input tokens = %zu\n", tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        LOG_INF("  %d\n", tokens[i]);
    }

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_debug_cb_user_data cb_data;

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = std::getenv("SEYREK") ? seyrek_cb : common_debug_cb_eval;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    // init
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }
    if (std::getenv("SEYREK")) {
        printf("katman,satir,alt_0.01,alt_0.05,alt_0.1,alt_0.2,ust_yari_enerji\n");
        for (auto & kv : g_istat) {
            auto & st = kv.second;
            printf("%s,%.0f,%.3f,%.3f,%.3f,%.3f,%.3f\n", kv.first.c_str(), st.satir, st.alt[0]/st.n, st.alt[1]/st.n, st.alt[2]/st.n, st.alt[3]/st.n, st.enerji_yari/st.satir);
        }
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
