// MoE expert activation profiler for llama.cpp
//
// Records per-token expert routing with router confidence, plus token
// probabilities during generation. Outputs a JSON profile file.
//
// Usage:
//   llama-moe-profile -m model.gguf -p "prompt" [-f corpus.txt] [-ngl N] [-c N] [-n N]
//
// -n N controls how many tokens to generate after processing the prompt.
// Output: profile-DDMMYYYY-HHMMSS.json (path printed to stderr)

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"

#include <algorithm>
#include <cinttypes>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ── data structures ────────────────────────────────────────────────

struct top_expert {
    int32_t id;
    float   prob;
};

struct token_candidate {
    std::string text;
    float       prob;
};

struct frame_data {
    std::map<int, std::vector<top_expert>> router_top16;
    std::map<int, std::vector<int32_t>> selected;
    std::vector<token_candidate> candidates;
    bool is_generated = false;
    std::string token_text;
};

struct moe_profile_data {
    std::vector<frame_data> frames;
    int n_expert_used = 0;
    int n_expert      = 0;
    size_t batch_frame_offset = 0;
    std::vector<int32_t> scratch_i32;
    std::vector<float>   scratch_f32;
};

// ── eval callback ──────────────────────────────────────────────────

static bool moe_profile_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    const bool is_topk  = (strncmp(t->name, "ffn_moe_topk", 12) == 0);
    const bool is_probs = (strncmp(t->name, "ffn_moe_probs", 13) == 0);

    if (!is_topk && !is_probs) return false;
    if (ask) return true;

    auto * d = static_cast<moe_profile_data *>(user_data);

    const char * dash = strchr(t->name, '-');
    if (!dash) return true;
    int il = atoi(dash + 1);

    if (is_topk) {
        int n_eu  = (int)t->ne[0];
        int n_tok = (int)t->ne[1];
        if (d->n_expert_used == 0) d->n_expert_used = n_eu;

        size_t n = ggml_nelements(t);
        d->scratch_i32.resize(n);
        ggml_backend_tensor_get(t, d->scratch_i32.data(), 0, n * sizeof(int32_t));

        for (int tok = 0; tok < n_tok; ++tok) {
            size_t fi = d->batch_frame_offset + tok;
            if (fi >= d->frames.size()) break;
            auto & sel = d->frames[fi].selected[il];
            for (int k = 0; k < n_eu; ++k) {
                int32_t eid = d->scratch_i32[k + tok * n_eu];
                sel.push_back(eid);
                if (eid >= d->n_expert) d->n_expert = eid + 1;
            }
        }
    }

    if (is_probs) {
        int ne  = (int)t->ne[0];
        int n_tok = (int)t->ne[1];

        size_t n = ggml_nelements(t);
        d->scratch_f32.resize(n);
        ggml_backend_tensor_get(t, d->scratch_f32.data(), 0, n * sizeof(float));

        for (int tok = 0; tok < n_tok; ++tok) {
            size_t fi = d->batch_frame_offset + tok;
            if (fi >= d->frames.size()) break;

            std::vector<std::pair<float, int>> ranked;
            ranked.reserve(ne);
            for (int e = 0; e < ne; ++e) {
                float p = d->scratch_f32[e + tok * ne];
                ranked.push_back({p, e});
            }
            std::partial_sort(ranked.begin(), ranked.begin() + std::min(16, (int)ranked.size()),
                              ranked.end(), std::greater<>());

            auto & top16 = d->frames[fi].router_top16[il];
            int n_keep = std::min(16, (int)ranked.size());
            for (int i = 0; i < n_keep; ++i) {
                top16.push_back({ranked[i].second, ranked[i].first});
            }
        }
    }

    return true;
}

// ── run inference ──────────────────────────────────────────────────

static bool run(llama_context * ctx, const common_params & params,
                moe_profile_data & profile_data) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::string formatted_prompt;
    {
        char arch[64] = {};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        std::string arch_str(arch);

        if (arch_str == "gemma4" || arch_str == "gemma3" || arch_str == "gemma2") {
            formatted_prompt = "<|turn>user\n" + params.prompt + "<turn|>\n<|turn>model\n";
            LOG_INF("applied %s chat template\n", arch);
        } else {
            const char * tmpl = llama_model_chat_template(model, nullptr);
            std::vector<llama_chat_message> chat = {{"user", params.prompt.c_str()}};
            int len = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, nullptr, 0);
            if (len > 0) {
                formatted_prompt.resize(len + 1);
                llama_chat_apply_template(tmpl, chat.data(), chat.size(), true,
                                          formatted_prompt.data(), formatted_prompt.size());
                formatted_prompt.resize(len);
                LOG_INF("applied chat template (%d chars)\n", len);
            } else {
                formatted_prompt = params.prompt;
                LOG_INF("no chat template, using raw prompt\n");
            }
        }
    }

    std::vector<llama_token> tokens = common_tokenize(ctx, formatted_prompt, add_bos, true);
    if (tokens.empty()) {
        LOG_ERR("no input tokens\n");
        return false;
    }

    LOG_INF("prefill: processing %zu tokens...\n", tokens.size());

    for (size_t i = 0; i < tokens.size(); ++i) {
        frame_data fr;
        fr.is_generated = false;
        char buf[128];
        int len = llama_token_to_piece(vocab, tokens[i], buf, sizeof(buf) - 1, 0, true);
        if (len < 0) len = 0;
        buf[len] = 0;
        fr.token_text = buf;
        profile_data.frames.push_back(std::move(fr));
    }

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_batch = llama_n_batch(ctx);
    int n_past = 0;

    for (size_t i = 0; i < tokens.size(); ) {
        size_t n = std::min((size_t)n_batch, tokens.size() - i);
        if ((int)(n_past + n) > n_ctx) {
            n = (size_t)n_ctx - n_past;
            if (n == 0) break;
        }

        profile_data.batch_frame_offset = i;
        llama_batch batch = llama_batch_get_one(tokens.data() + i, (int)n);
        if (llama_decode(ctx, batch)) {
            LOG_ERR("decode failed at offset %zu\n", i);
            return false;
        }
        n_past += (int)n;
        i += n;
    }

    LOG_INF("prefill: done (%zu tokens)\n", tokens.size());

    int n_gen = params.n_predict;
    if (n_gen <= 0) n_gen = 64;

    LOG_INF("decode: generating %d tokens...\n", n_gen);

    auto sparams = params.sampling;
    auto * smpl = common_sampler_init(model, sparams);

    for (int g = 0; g < n_gen && n_past < n_ctx; ++g) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        int n_vocab = llama_vocab_n_tokens(vocab);

        std::vector<std::pair<float, int>> logit_pairs;
        logit_pairs.reserve(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            logit_pairs.push_back({logits[i], i});
        }
        std::partial_sort(logit_pairs.begin(), logit_pairs.begin() + 10,
                          logit_pairs.end(), std::greater<>());

        float max_logit = logit_pairs[0].first;
        float sum_exp = 0;
        for (int i = 0; i < std::min(10, n_vocab); ++i) {
            sum_exp += expf(logit_pairs[i].first - max_logit);
        }

        llama_token next_token = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, next_token, true);

        frame_data fr;
        fr.is_generated = true;
        char buf[128];
        int len = llama_token_to_piece(vocab, next_token, buf, sizeof(buf) - 1, 0, true);
        if (len < 0) len = 0;
        buf[len] = 0;
        fr.token_text = buf;

        for (int i = 0; i < std::min(10, n_vocab); ++i) {
            char tbuf[128];
            int tlen = llama_token_to_piece(vocab, logit_pairs[i].second,
                                            tbuf, sizeof(tbuf) - 1, 0, true);
            if (tlen < 0) tlen = 0;
            tbuf[tlen] = 0;
            float prob = expf(logit_pairs[i].first - max_logit) / sum_exp;
            fr.candidates.push_back({tbuf, prob});
        }

        profile_data.frames.push_back(std::move(fr));
        profile_data.batch_frame_offset = profile_data.frames.size() - 1;

        if (llama_vocab_is_eog(vocab, next_token)) {
            LOG_INF("decode: hit EOS at token %d\n", g + 1);
            break;
        }

        llama_batch batch = llama_batch_get_one(&next_token, 1);
        if (llama_decode(ctx, batch)) {
            LOG_ERR("decode failed at generation step %d\n", g);
            return false;
        }
        n_past++;
    }

    common_sampler_free(smpl);
    LOG_INF("decode: done (total frames: %zu)\n", profile_data.frames.size());
    return true;
}

// ── JSON helpers ───────────────────────────────────────────────────

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    out += ' ';
                } else if (c >= 0x80) {
                    // Validate UTF-8 sequence; skip invalid bytes
                    int len = 0;
                    if      ((c & 0xE0) == 0xC0) len = 2;
                    else if ((c & 0xF0) == 0xE0) len = 3;
                    else if ((c & 0xF8) == 0xF0) len = 4;
                    else { out += '?'; break; } // invalid lead byte
                    if (i + len > s.size()) { out += '?'; break; }
                    bool valid = true;
                    for (int j = 1; j < len; ++j) {
                        if (((unsigned char)s[i + j] & 0xC0) != 0x80) { valid = false; break; }
                    }
                    if (valid) {
                        out.append(s, i, len);
                        i += len - 1;
                    } else {
                        out += '?';
                    }
                } else {
                    out += (char)c;
                }
        }
    }
    return out;
}

static std::string make_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    struct tm tm;
    gmtime_r(&t, &tm);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

static std::string make_filename() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    struct tm tm;
    localtime_r(&t, &tm);
    char buf[64];
    strftime(buf, sizeof(buf), "profile-%d%m%Y-%H%M%S.json", &tm);
    return buf;
}

// ── JSON emitter ───────────────────────────────────────────────────

static void emit_json(const moe_profile_data & data, const char * model_desc,
                      const std::string & prompt, const common_params & params,
                      const char * path) {
    std::ofstream f(path);
    if (!f) {
        LOG_ERR("failed to open %s for writing\n", path);
        return;
    }

    f << std::fixed;
    f.precision(4);

    const int n_expert = data.n_expert;
    const int n_frames = (int)data.frames.size();

    // Collect layer IDs
    std::vector<int> layer_ids;
    for (auto & frame : data.frames) {
        if (!frame.selected.empty()) {
            for (auto & [lid, _] : frame.selected) {
                layer_ids.push_back(lid);
            }
            break;
        }
    }

    int n_prefill = 0, n_generated = 0;
    for (auto & fr : data.frames) {
        if (fr.is_generated) n_generated++; else n_prefill++;
    }

    // Get quantization from model desc
    std::string quant(model_desc);

    f << "{\n";
    f << "  \"model\": \"" << json_escape(model_desc) << "\",\n";
    f << "  \"prompt\": \"" << json_escape(prompt) << "\",\n";
    f << "  \"timestamp\": \"" << make_timestamp() << "\",\n";
    f << "  \"context_size\": " << params.n_ctx << ",\n";
    f << "  \"temperature\": " << params.sampling.temp << ",\n";
    f << "  \"n_expert\": " << n_expert << ",\n";
    f << "  \"n_expert_used\": " << data.n_expert_used << ",\n";
    f << "  \"n_prefill\": " << n_prefill << ",\n";
    f << "  \"n_generated\": " << n_generated << ",\n";

    // Layers array
    f << "  \"layers\": [";
    for (size_t i = 0; i < layer_ids.size(); ++i) {
        if (i > 0) f << ",";
        f << layer_ids[i];
    }
    f << "],\n";

    // Frames array
    f << "  \"frames\": [\n";
    for (int fi = 0; fi < n_frames; ++fi) {
        if (fi > 0) f << ",\n";
        auto & fr = data.frames[fi];

        f << "    {\"gen\":" << (fr.is_generated ? "true" : "false");
        f << ",\"token\":\"" << json_escape(fr.token_text) << "\"";

        // Selected experts per layer
        f << ",\"sel\":[";
        bool first_l = true;
        for (int lid : layer_ids) {
            if (!first_l) f << ",";
            first_l = false;
            auto it = fr.selected.find(lid);
            if (it != fr.selected.end()) {
                f << "[";
                for (size_t k = 0; k < it->second.size(); ++k) {
                    if (k > 0) f << ",";
                    f << it->second[k];
                }
                f << "]";
            } else {
                f << "[]";
            }
        }
        f << "]";

        // Router top-16 per layer
        f << ",\"top\":[";
        first_l = true;
        for (int lid : layer_ids) {
            if (!first_l) f << ",";
            first_l = false;
            auto it = fr.router_top16.find(lid);
            if (it != fr.router_top16.end()) {
                f << "[";
                for (size_t k = 0; k < it->second.size(); ++k) {
                    if (k > 0) f << ",";
                    f << "[" << it->second[k].id << "," << it->second[k].prob << "]";
                }
                f << "]";
            } else {
                f << "[]";
            }
        }
        f << "]";

        // Token candidates
        f << ",\"cands\":[";
        for (size_t ci = 0; ci < fr.candidates.size(); ++ci) {
            if (ci > 0) f << ",";
            f << "{\"t\":\"" << json_escape(fr.candidates[ci].text)
              << "\",\"p\":" << fr.candidates[ci].prob << "}";
        }
        f << "]";

        f << "}";
    }
    f << "\n  ]\n";
    f << "}\n";

    f.close();
    LOG_INF("JSON profile written to %s\n", path);
}

// ── main ───────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    common_params params;
    moe_profile_data profile_data;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    params.cb_eval           = moe_profile_cb;
    params.cb_eval_user_data = &profile_data;
    params.warmup            = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (!model || !ctx) {
        LOG_ERR("failed to init model/context\n");
        return 1;
    }

    char desc[256];
    llama_model_desc(model, desc, sizeof(desc));
    LOG_INF("model: %s\n", desc);
    LOG_INF("n_layer: %d\n", llama_model_n_layer(model));
    LOG_INF("\n");

    if (!run(ctx, params, profile_data)) {
        return 1;
    }

    if (profile_data.frames.empty()) {
        LOG_ERR("No frames recorded.\n");
        llama_backend_free();
        return 1;
    }

    int n_prefill = 0, n_generated = 0;
    for (auto & fr : profile_data.frames) {
        if (fr.is_generated) n_generated++; else n_prefill++;
    }

    LOG_INF("\nn_expert = %d, n_expert_used = %d, prefill = %d, generated = %d\n",
            profile_data.n_expert, profile_data.n_expert_used, n_prefill, n_generated);

    std::string filename = make_filename();
    std::string path = "/tmp/" + filename;
    emit_json(profile_data, desc, params.prompt, params, path.c_str());

    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
