// MoE expert activation profiler for llama.cpp
//
// Records per-token expert routing with router confidence, plus token
// probabilities during generation. Outputs an animated HTML visualisation.
//
// Usage:
//   llama-moe-profile -m model.gguf -p "prompt" [-f corpus.txt] [-ngl N] [-c N] [-n N]
//
// -n N controls how many tokens to generate after processing the prompt.

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
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
    // Per layer: top-16 experts with their router probabilities
    std::map<int, std::vector<top_expert>> router_top16;
    // Per layer: the 8 actually selected expert IDs
    std::map<int, std::vector<int32_t>> selected;

    // Token probability candidates (decode phase only)
    std::vector<token_candidate> candidates; // top-10

    bool is_generated = false;
    std::string token_text;
};

struct moe_profile_data {
    std::vector<frame_data> frames;

    int n_expert_used = 0;
    int n_expert      = 0;

    // Track how many frames existed before the current batch started
    // Set by the run() function before each llama_decode()
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
        // Shape: [n_expert_used, n_tokens]
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
        // Shape: [n_expert, n_tokens]
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

    // Apply chat template for instruction-tuned models.
    // llama_chat_apply_template doesn't handle complex Jinja templates (like Gemma4's),
    // so we detect the architecture and apply the format directly.
    std::string formatted_prompt;
    {
        char arch[64] = {};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        std::string arch_str(arch);

        if (arch_str == "gemma4" || arch_str == "gemma3" || arch_str == "gemma2") {
            // Gemma chat format: <|turn>user\n{prompt}<turn|>\n<|turn>model\n
            formatted_prompt = "<|turn>user\n" + params.prompt + "<turn|>\n<|turn>model\n";
            LOG_INF("applied %s chat template\n", arch);
        } else {
            // Try the generic template API
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

    // ── Prefill phase ──────────────────────────────────────────────

    LOG_INF("prefill: processing %zu tokens...\n", tokens.size());

    // Pre-create frames for the prompt tokens
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

    // ── Decode phase (generation) ──────────────────────────────────

    int n_gen = params.n_predict;
    if (n_gen <= 0) n_gen = 64; // default: generate 64 tokens

    LOG_INF("decode: generating %d tokens...\n", n_gen);

    // Use the built-in sampler chain (respects --temp, --top-k, --repeat-penalty, etc.)
    auto sparams = params.sampling;
    auto * smpl = common_sampler_init(model, sparams);

    for (int g = 0; g < n_gen && n_past < n_ctx; ++g) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        int n_vocab = llama_vocab_n_tokens(vocab);

        // Find top-10 candidates from raw logits (before sampling) for display
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

        // Sample using the proper sampler chain
        llama_token next_token = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, next_token, true);

        // Create frame for this generated token
        frame_data fr;
        fr.is_generated = true;
        char buf[128];
        int len = llama_token_to_piece(vocab, next_token, buf, sizeof(buf) - 1, 0, true);
        if (len < 0) len = 0;
        buf[len] = 0;
        fr.token_text = buf;

        // Record top-10 token candidates
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

        // Check for EOS
        if (llama_vocab_is_eog(vocab, next_token)) {
            LOG_INF("decode: hit EOS at token %d\n", g + 1);
            break;
        }

        // Decode the generated token (triggers expert routing callback)
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

// ── HTML emitter ───────────────────────────────────────────────────

static void emit_html(const moe_profile_data & data, const char * model_desc,
                      const char * path) {
    std::ofstream f(path);
    if (!f) {
        LOG_ERR("failed to open %s for writing\n", path);
        return;
    }

    const int n_expert = data.n_expert;
    const int n_expert_used = data.n_expert_used;
    const int n_frames = (int)data.frames.size();

    // Count layers from first frame that has data
    std::vector<int> layer_ids;
    for (auto & frame : data.frames) {
        if (!frame.selected.empty()) {
            for (auto & [lid, _] : frame.selected) {
                layer_ids.push_back(lid);
            }
            break;
        }
    }
    const int n_layers = (int)layer_ids.size();

    // Count prefill vs generated
    int n_prefill = 0, n_generated = 0;
    for (auto & fr : data.frames) {
        if (fr.is_generated) n_generated++; else n_prefill++;
    }

    // ── Build JSON ─────────────────────────────────────────────────

    // Frames: per token, per layer: {sel: [8 ids], top: [{id, p}, ...]}
    std::ostringstream frames_json;
    frames_json << std::fixed;
    frames_json.precision(4);
    frames_json << "[";
    for (int fi = 0; fi < n_frames; ++fi) {
        if (fi > 0) frames_json << ",";
        auto & fr = data.frames[fi];
        frames_json << "{\"gen\":" << (fr.is_generated ? "true" : "false");

        // Selected experts per layer
        frames_json << ",\"sel\":[";
        bool first_l = true;
        for (int lid : layer_ids) {
            if (!first_l) frames_json << ",";
            first_l = false;
            auto it = fr.selected.find(lid);
            if (it != fr.selected.end()) {
                frames_json << "[";
                for (size_t k = 0; k < it->second.size(); ++k) {
                    if (k > 0) frames_json << ",";
                    frames_json << it->second[k];
                }
                frames_json << "]";
            } else {
                frames_json << "[]";
            }
        }
        frames_json << "]";

        // Router top-16 per layer
        frames_json << ",\"top\":[";
        first_l = true;
        for (int lid : layer_ids) {
            if (!first_l) frames_json << ",";
            first_l = false;
            auto it = fr.router_top16.find(lid);
            if (it != fr.router_top16.end()) {
                frames_json << "[";
                for (size_t k = 0; k < it->second.size(); ++k) {
                    if (k > 0) frames_json << ",";
                    frames_json << "[" << it->second[k].id << "," << it->second[k].prob << "]";
                }
                frames_json << "]";
            } else {
                frames_json << "[]";
            }
        }
        frames_json << "]";

        // Token candidates (decode only)
        frames_json << ",\"cands\":[";
        for (size_t ci = 0; ci < fr.candidates.size(); ++ci) {
            if (ci > 0) frames_json << ",";
            // Escape the token text for JSON
            frames_json << "{\"t\":\"";
            for (char c : fr.candidates[ci].text) {
                if (c == '"') frames_json << "\\\"";
                else if (c == '\\') frames_json << "\\\\";
                else if (c == '\n') frames_json << "\\n";
                else if (c == '\r') frames_json << "\\r";
                else if (c == '\t') frames_json << "\\t";
                else if ((unsigned char)c < 0x20) frames_json << " ";
                else frames_json << c;
            }
            frames_json << "\",\"p\":" << fr.candidates[ci].prob << "}";
        }
        frames_json << "]";

        frames_json << "}";
    }
    frames_json << "]";

    // Token texts
    std::ostringstream tokens_json;
    tokens_json << "[";
    for (int i = 0; i < n_frames; ++i) {
        if (i > 0) tokens_json << ",";
        tokens_json << "\"";
        for (char c : data.frames[i].token_text) {
            if (c == '"') tokens_json << "\\\"";
            else if (c == '\\') tokens_json << "\\\\";
            else if (c == '\n') tokens_json << "\\n";
            else if (c == '\r') tokens_json << "\\r";
            else if (c == '\t') tokens_json << "\\t";
            else if ((unsigned char)c < 0x20) tokens_json << " ";
            else tokens_json << c;
        }
        tokens_json << "\"";
    }
    tokens_json << "]";

    // Layer IDs
    std::ostringstream layers_json;
    layers_json << "[";
    for (size_t i = 0; i < layer_ids.size(); ++i) {
        if (i > 0) layers_json << ",";
        layers_json << layer_ids[i];
    }
    layers_json << "]";

    // ── Write HTML ─────────────────────────────────────────────────

    f << R"(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MoE Expert Routing</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
       background: #0a0a0f; color: #e0e0e8; padding: 12px 20px; display: flex; gap: 20px;
       overflow: hidden; }

.main { flex: 1; min-width: 0; margin-right: 280px; }
.sidebar { width: 260px; flex-shrink: 0; position: fixed; right: 20px; top: 20px; bottom: 20px;
           display: flex; flex-direction: column; }

h1 { font-size: 16px; font-weight: 600; margin-bottom: 2px; }
.sub { font-size: 11px; color: #666; margin-bottom: 8px; }

.player { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; flex-wrap: wrap; }
.player button { background: #1a1a24; color: #e0e0e8; border: 1px solid #333; padding: 5px 12px;
                 border-radius: 4px; cursor: pointer; font-size: 12px; }
.player button:hover { background: #2a2a3a; }
.player button.active { background: #1a2a3a; border-color: #7dd3fc; }
.player input[type=range] { flex: 1; min-width: 150px; accent-color: #7dd3fc; }
.player .info { font-size: 12px; color: #888; min-width: 100px; }
.speed-ctrl { font-size: 11px; color: #666; display: flex; align-items: center; gap: 4px; }
.speed-ctrl select { background: #1a1a24; color: #ccc; border: 1px solid #333;
                     padding: 2px 4px; border-radius: 3px; font-size: 11px; }

.token-bar { font-size: 13px; padding: 8px 12px; background: #12121a; border: 1px solid #1a1a24;
             border-radius: 6px; font-family: 'SF Mono', 'Cascadia Code', monospace;
             height: 40px; line-height: 1.4; overflow: hidden; white-space: nowrap;
             margin-bottom: 6px; display: flex; align-items: center; }
.token-bar .past { color: #555; }
.token-bar .cur { color: #7dd3fc; background: #1a2a3a; padding: 1px 2px; border-radius: 2px; font-weight: 600; }
.token-bar .fut { color: #333; }
.token-bar .gen { color: #4ade80; }
.token-bar .gen.cur { background: #1a3a2a; color: #4ade80; }

.phase-badge { display: inline-block; font-size: 10px; padding: 2px 8px; border-radius: 3px;
               margin-left: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.phase-prefill { background: #1a2a3a; color: #7dd3fc; }
.phase-decode  { background: #1a3a2a; color: #4ade80; }

.grid-wrap { }
.grid-outer { display: flex; gap: 0; }
.ylabels { display: flex; flex-direction: column; gap: 1px; flex-shrink: 0; }
.ylbl { height: 16px; font-size: 9px; color: #444; display: flex; align-items: center;
        justify-content: flex-end; padding-right: 4px; width: 24px; }
.grid { display: grid; gap: 1px; width: 100%; }
.cell { height: 16px; border-radius: 1px; }

.tooltip { position: fixed; background: #1e1e2a; border: 1px solid #444; border-radius: 6px;
           padding: 6px 10px; font-size: 11px; pointer-events: none; z-index: 100;
           box-shadow: 0 4px 16px rgba(0,0,0,0.5); display: none; }
.tooltip .v { font-weight: 700; color: #7dd3fc; }

/* sidebar */
.sidebar h2 { font-size: 14px; font-weight: 600; margin-bottom: 10px; }
.cand-list { list-style: none; }
.cand-item { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 12px; }
.cand-bar-bg { flex: 1; height: 14px; background: #151520; border-radius: 3px; overflow: hidden; position: relative; }
.cand-bar { height: 100%; border-radius: 3px; transition: width 0.15s; }
.cand-bar.top { background: #4ade80; }
.cand-bar.other { background: #334; }
.cand-tok { min-width: 70px; font-family: monospace; color: #ccc; text-align: right;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cand-pct { min-width: 40px; text-align: right; color: #888; font-size: 11px; }
.no-cands { color: #444; font-size: 12px; font-style: italic; }

.router-detail { margin-top: 20px; }
.router-detail h2 { font-size: 14px; font-weight: 600; margin-bottom: 8px; }
</style>
</head>
<body>

<div class="main">
<h1>MoE Expert Routing</h1>
)";

    f << "<div class=\"sub\">" << model_desc
      << " &middot; " << n_layers << " layers &times; " << n_expert << " experts"
      << " &middot; top-" << n_expert_used
      << " &middot; " << n_prefill << " prefill + " << n_generated << " generated</div>\n";

    f << R"(
<div class="player">
  <button id="btnPlay">Play</button>
  <button id="btnStep">&rarr;</button>
  <input type="range" id="scrub" min="0" value="0">
  <div class="info" id="finfo">0 / 0</div>
  <span class="phase-badge" id="phaseBadge"></span>
  <div class="speed-ctrl">
    <select id="spd">
      <option value="400">Slow</option>
      <option value="120" selected>Normal</option>
      <option value="40">Fast</option>
      <option value="10">Max</option>
    </select>
  </div>
</div>

<div class="token-bar" id="tokBar"></div>

<div class="grid-wrap">
  <div class="grid-outer">
    <div class="ylabels" id="yl"></div>
    <div class="grid" id="gr"></div>
  </div>
</div>
<div class="tooltip" id="tt"></div>

<div style="font-size:11px;color:#555;margin:8px 0 4px;" id="cumInfo">Cumulative Expert Usage (decode)</div>
<div class="grid-wrap">
  <div class="grid-outer">
    <div class="ylabels" id="ylCum"></div>
    <div class="grid" id="grCum"></div>
  </div>
</div>
</div>

<div class="sidebar">
  <h2>Token Predictions</h2>
  <ul class="cand-list" id="candList">
    <li class="no-cands">Visible during generation phase</li>
  </ul>
  <div style="margin-top:auto;">
    <div style="font-size:11px;color:#888;margin-bottom:6px;">Decode Stats</div>
    <div id="decodeStats" style="font-size:12px;color:#555;">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
        <span>Avg Gini</span><span id="statGini" style="color:#7dd3fc;font-weight:600;">—</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
        <span>Avg expert overlap</span><span id="statOverlap" style="color:#7dd3fc;font-weight:600;">—</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
        <span>Unique experts used</span><span id="statUnique" style="color:#7dd3fc;font-weight:600;">—</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
        <span>Unique (last 64 tokens)</span><span id="statRecent" style="color:#7dd3fc;font-weight:600;">—</span>
      </div>
    </div>
    <div style="font-size:11px;color:#888;margin-bottom:4px;">Hottest Experts (per layer)</div>
    <div id="topExperts" style="font-size:11px;font-family:monospace;color:#aaa;"></div>
  </div>
</div>

<script>
)";

    f << "const F=" << frames_json.str() << ";\n";
    f << "const TK=" << tokens_json.str() << ";\n";
    f << "const LY=" << layers_json.str() << ";\n";
    f << "const NE=" << n_expert << ";\n";
    f << "const NEU=" << n_expert_used << ";\n";
    f << "const NL=" << n_layers << ";\n";
    f << "const NF=" << n_frames << ";\n";
    f << "const NP=" << n_prefill << ";\n";

    f << R"(
const gr=document.getElementById('gr');
const yl=document.getElementById('yl');
const tt=document.getElementById('tt');
const scrub=document.getElementById('scrub');
const finfo=document.getElementById('finfo');
const tokBar=document.getElementById('tokBar');
const badge=document.getElementById('phaseBadge');
const candList=document.getElementById('candList');
const statGini=document.getElementById('statGini');
const statOverlap=document.getElementById('statOverlap');
const statUnique=document.getElementById('statUnique');
const statRecent=document.getElementById('statRecent');
const topExperts=document.getElementById('topExperts');
const grCum=document.getElementById('grCum');
const ylCum=document.getElementById('ylCum');
const cumInfo=document.getElementById('cumInfo');

gr.style.gridTemplateColumns=`repeat(${NE},1fr)`;
grCum.style.gridTemplateColumns=`repeat(${NE},1fr)`;
scrub.max=NF-1;

// Pre-compute final cumulative max (decode only) for fixed colour scale
let cumFinalMax=0;
{
  const tmp=[];
  for(let li=0;li<NL;li++) tmp[li]=new Uint32Array(NE);
  for(let i=0;i<NF;i++){
    if(!F[i].gen) continue;
    for(let li=0;li<NL;li++){
      const sel=F[i].sel[li]||[];
      for(const eid of sel) if(eid>=0&&eid<NE){ tmp[li][eid]++; if(tmp[li][eid]>cumFinalMax) cumFinalMax=tmp[li][eid]; }
    }
  }
}

// Build cumulative grid cells
const cumCells=[];
const cumCounts=[]; // cumCounts[li][ei] = count so far
for(let li=0;li<NL;li++){
  const l=document.createElement('div');
  l.className='ylbl';l.textContent=LY[li];
  ylCum.appendChild(l);
  cumCells[li]=[];
  cumCounts[li]=new Uint32Array(NE);
  for(let ei=0;ei<NE;ei++){
    const c=document.createElement('div');
    c.className='cell';
    c.style.background='#0e0e16';
    grCum.appendChild(c);
    cumCells[li][ei]=c;
  }
}

// Build main grid cells
const cells=[];
for(let li=0;li<NL;li++){
  const l=document.createElement('div');
  l.className='ylbl';l.textContent=LY[li];
  yl.appendChild(l);
  cells[li]=[];
  for(let ei=0;ei<NE;ei++){
    const c=document.createElement('div');
    c.className='cell';
    c.style.background='#0e0e16';
    c.addEventListener('mouseenter',()=>{
      tt.style.display='block';
      const fi=parseInt(scrub.value);
      const fr=F[fi];
      const sel=fr&&fr.sel[li]?fr.sel[li]:[];
      const isActive=sel.includes(ei);
      // Find probability
      let prob=0;
      if(fr&&fr.top[li]){
        for(const[id,p]of fr.top[li]){if(id===ei){prob=p;break;}}
      }
      tt.innerHTML=`L<span class="v">${LY[li]}</span> E<span class="v">${ei}</span> `
        +(isActive?`<span style="color:#4ade80">ACTIVE</span>`:`<span style="color:#555">idle</span>`)
        +(prob>0?` p=<span class="v">${(prob*100).toFixed(1)}%</span>`:'');
    });
    c.addEventListener('mousemove',e=>{tt.style.left=(e.clientX+10)+'px';tt.style.top=(e.clientY+10)+'px';});
    c.addEventListener('mouseleave',()=>tt.style.display='none');
    gr.appendChild(c);
    cells[li][ei]=c;
  }
}

// No histogram setup needed — stats are computed in render()

// Render frame
let prev=new Set();

function render(fi){
  const fr=F[fi];
  if(!fr)return;

  const now=new Set();

  // Update grid: use router probabilities for intensity
  // First build a probability map for this frame
  const probMap={}; // key: "li_ei" -> prob
  for(let li=0;li<NL;li++){
    if(fr.top[li]){
      for(const[id,p]of fr.top[li]){
        probMap[li+'_'+id]=p;
      }
    }
    if(fr.sel[li]){
      for(const eid of fr.sel[li]){
        if(eid>=0) now.add(li*NE+eid);
      }
    }
  }

  // Fade old, light new
  for(const key of prev){
    if(!now.has(key)){
      const li=Math.floor(key/NE),ei=key%NE;
      cells[li][ei].style.background='#161622';
    }
  }
  for(const key of now){
    const li=Math.floor(key/NE),ei=key%NE;
    const p=probMap[li+'_'+ei]||0;
    // Map probability to brightness: higher prob = brighter
    const bright=Math.floor(80+175*Math.min(p*NEU,1));
    cells[li][ei].style.background=`rgb(${Math.floor(bright*0.3)},${Math.floor(bright*0.85)},${bright})`;
  }

  // Clear faded after delay
  const oldPrev=new Set(prev);
  setTimeout(()=>{
    for(const key of oldPrev){
      if(!now.has(key)){
        const li=Math.floor(key/NE),ei=key%NE;
        const bg=cells[li][ei].style.background;
        if(bg.includes('22,')|| bg==='#161622'||bg==='rgb(22, 22, 34)')
          cells[li][ei].style.background='#0e0e16';
      }
    }
  },250);
  prev=now;

  // Token bar
  const ctx=25;
  const s=Math.max(0,fi-ctx),e=Math.min(NF,fi+ctx+1);
  let h='';
  for(let i=s;i<e;i++){
    const txt=(TK[i]||'').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'\u21b5').replace(/ /g,'\u00b7');
    const g=F[i]&&F[i].gen;
    if(i<fi) h+=`<span class="${g?'gen past':'past'}">${txt}</span>`;
    else if(i===fi) h+=`<span class="${g?'gen cur':'cur'}">${txt}</span>`;
    else h+=`<span class="fut">${txt}</span>`;
  }
  tokBar.innerHTML=h;

  // Phase badge
  if(fr.gen){badge.textContent='decode';badge.className='phase-badge phase-decode';}
  else{badge.textContent='prefill';badge.className='phase-badge phase-prefill';}

  // Frame info
  finfo.textContent=`${fi+1} / ${NF}`;
  scrub.value=fi;

  // Sidebar: token candidates
  if(fr.cands&&fr.cands.length>0){
    candList.innerHTML='';
    for(let i=0;i<fr.cands.length;i++){
      const c=fr.cands[i];
      const li=document.createElement('li');
      li.className='cand-item';
      const pct=(c.p*100).toFixed(1);
      li.innerHTML=`<span class="cand-tok">${c.t.replace(/</g,'&lt;')}</span>`
        +`<div class="cand-bar-bg"><div class="cand-bar ${i===0?'top':'other'}" style="width:${(c.p*100).toFixed(1)}%"></div></div>`
        +`<span class="cand-pct">${pct}%</span>`;
      candList.appendChild(li);
    }
  }else{
    candList.innerHTML='<li class="no-cands">Prefill token (no predictions)</li>';
  }

  // Expert usage histogram — rebuild up to current frame (decode only)
  // Rebuild cumulative heatmap counts
  for(let li=0;li<NL;li++) cumCounts[li].fill(0);
  let decTokens=0;
  for(let i=0;i<=fi;i++){
    if(!F[i].gen) continue;
    decTokens++;
    for(let li=0;li<NL;li++){
      const sel=F[i].sel[li]||[];
      for(const eid of sel) if(eid>=0&&eid<NE) cumCounts[li][eid]++;
    }
  }

  // Update sidebar stats
  if(decTokens>0){
    // Avg Gini coefficient across layers (0 = uniform, 1 = concentrated)
    let giniSum=0;
    for(let li=0;li<NL;li++){
      const c=cumCounts[li];
      let total=0;
      for(let ei=0;ei<NE;ei++) total+=c[ei];
      if(total>0){
        const sorted=[...c].sort((a,b)=>a-b);
        let s=0;
        for(let i=0;i<NE;i++) s+=(2*(i+1)-NE-1)*sorted[i];
        giniSum+=s/(NE*total);
      }
    }
    const avgGini=giniSum/NL;
    statGini.textContent=avgGini.toFixed(3);
    statGini.style.color=avgGini>0.3?'#f87171':avgGini>0.1?'#fbbf24':'#4ade80';

    // Expert overlap with previous token
    if(fi>0 && F[fi].gen && F[fi-1].gen){
      let totalOverlap=0;
      for(let li=0;li<NL;li++){
        const prev=new Set(F[fi-1].sel[li]||[]);
        const curr=F[fi].sel[li]||[];
        for(const e of curr) if(prev.has(e)) totalOverlap++;
      }
      statOverlap.textContent=`${(totalOverlap/NL).toFixed(1)}/${NEU}`;
    }

    // Unique experts used per layer (so far in decode)
    let totalUniq=0;
    for(let li=0;li<NL;li++){
      let used=0;
      for(let ei=0;ei<NE;ei++) if(cumCounts[li][ei]>0) used++;
      totalUniq+=used;
    }
    statUnique.textContent=`${totalUniq}/${NE*NL}`;

    // Unique experts in last 64 decode tokens
    const window=64;
    let recentUniq=0;
    for(let li=0;li<NL;li++){
      const seen=new Uint8Array(NE);
      const start=Math.max(0,fi-window+1);
      for(let i=start;i<=fi;i++){
        if(!F[i].gen) continue;
        const sel=F[i].sel[li]||[];
        for(const eid of sel) if(eid>=0&&eid<NE) seen[eid]=1;
      }
      let u=0;
      for(let ei=0;ei<NE;ei++) u+=seen[ei];
      recentUniq+=u;
    }
    statRecent.textContent=`${recentUniq}/${NE*NL}`;

    // Top experts per layer — find the single hottest expert per layer
    let html='';
    for(let li=0;li<NL;li++){
      let topE=-1,topC=0;
      for(let ei=0;ei<NE;ei++){
        if(cumCounts[li][ei]>topC){topC=cumCounts[li][ei];topE=ei;}
      }
      const pct=decTokens>0?(topC/decTokens*100).toFixed(0):'0';
      const bar=decTokens>0?Math.round(topC/decTokens*60):0;
      html+=`<div style="display:flex;gap:4px;align-items:center;margin-bottom:1px;">`
        +`<span style="color:#555;width:18px;text-align:right;">${LY[li]}</span>`
        +`<span style="color:#7dd3fc;width:28px;">e${topE}</span>`
        +`<div style="width:${bar}px;height:6px;background:#7dd3fc;border-radius:2px;"></div>`
        +`<span style="color:#555;">${pct}%</span></div>`;
    }
    topExperts.innerHTML=html;
  }else{
    statGini.textContent='\u2014';statGini.style.color='#7dd3fc';
    statOverlap.textContent='\u2014';
    statUnique.textContent='\u2014';
    statRecent.textContent='\u2014';
    topExperts.innerHTML='<span style="color:#444;">Decode phase only</span>';
  }

  if(decTokens>0){
    for(let li=0;li<NL;li++){
      for(let ei=0;ei<NE;ei++){
        const v=cumCounts[li][ei];
        if(v===0){
          cumCells[li][ei].style.background='#0e0e16';
        }else{
          const t=v/Math.max(cumFinalMax,1);
          // dark -> cyan -> white (two-stop lerp)
          let r,g,b;
          if(t<0.5){
            const s=t*2; // 0..1 over first half
            r=Math.round(10+s*20);   // 10 -> 30
            g=Math.round(14+s*186);  // 14 -> 200
            b=Math.round(22+s*230);  // 22 -> 252
          }else{
            const s=(t-0.5)*2; // 0..1 over second half
            r=Math.round(30+s*225);  // 30 -> 255
            g=Math.round(200+s*55);  // 200 -> 255
            b=Math.round(252+s*3);   // 252 -> 255
          }
          cumCells[li][ei].style.background=`rgb(${r},${g},${b})`;
        }
      }
    }
    cumInfo.textContent=`${decTokens} decode tokens accumulated \u2022 max hits: ${cumFinalMax}`;
  }else{
    cumInfo.textContent='Scrub into decode phase to see accumulation';
  }
}

// Playback
let playing=false,timer=null,frame=0;
function play(){playing=true;document.getElementById('btnPlay').textContent='Pause';
  document.getElementById('btnPlay').classList.add('active');tick();}
function pause(){playing=false;if(timer){clearTimeout(timer);timer=null;}
  document.getElementById('btnPlay').textContent='Play';
  document.getElementById('btnPlay').classList.remove('active');}
function tick(){if(!playing)return;render(frame);frame=(frame+1)%NF;
  timer=setTimeout(tick,parseInt(document.getElementById('spd').value));}

document.getElementById('btnPlay').addEventListener('click',()=>{if(playing)pause();else play();});
document.getElementById('btnStep').addEventListener('click',()=>{pause();render(frame);frame=(frame+1)%NF;});
scrub.addEventListener('input',()=>{pause();frame=parseInt(scrub.value);render(frame);});

render(0);
</script>
</body>
</html>
)";

    f.close();
    LOG_INF("HTML written to %s\n", path);
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

    LOG_INF("\nn_expert = %d, n_expert_used = %d, frames = %zu\n",
            profile_data.n_expert, profile_data.n_expert_used, profile_data.frames.size());

    emit_html(profile_data, desc, "/tmp/moe-profile.html");

    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
