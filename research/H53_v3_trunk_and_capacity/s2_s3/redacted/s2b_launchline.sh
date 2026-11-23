#!/usr/bin/env bash
# trainer-lane: post the S2b launch line BEFORE the fire (Amendment 6 condition 5), from state/s2b_prep.json + the manifest.
#   s2b_launchline.sh <medium SHA256SUMS sha256>
set -u
ST=/mnt/xbabe3-external/jsr/trainer-lane/s2/state; MSUM=$1
read -r BW WG RULE A_LS A_SM A_MD G_MD H_MD A_LL CHK <<< "$(python3 -c '
import json; j = json.load(open("'$ST'/s2b_prep.json")); b = j["bw"]
print(b["extra_base_weight"], j["chosen_window_groups"], j["rule"].replace(" ", "_"), round(b["librispeech"]["a"], 4), round(b["small"]["a"], 4),
      round(b["medium"]["a"], 4), b["medium"]["groups"], round(b["medium"]["hours"], 2), round(b["a_LL_pooled"], 4), b["ls_audio_share_check"])')"
AUD=$(python3 -c '
import json; j = json.load(open("'$ST'/s2b_prep.json"))
print("\n".join("| %d | %.4f | %.4f | %.3f / **%.3f** | %s |" % (a["window_groups"], a["ll_audio_share_pooled"], a["ll_share_median"], a["ll_share_p5"], a["ll_share_p95"], "PASS" if a["pass"] else "fail") for a in j["audits"]))')
/mnt/xbabe3-external/jsr/trainer-lane/s2/tools/post.sh "S2b LAUNCH LINE (Amendment 6 condition 5, posted BEFORE the fire): H53_S2b_medium_s7919 at WORLD 6, extra_base_weight $BW, window_groups $WG @claude-lead @eval-lane @h53-trunk @h52-librilight @codex-full-fleet" <<B
- **Conditions:** (1) medium finalised by h52 (v1.1 labels only, amendment 4's NaN rule); (2) sealed SHA256SUMS **${MSUM:0:16}…** identical and \`sha256sum -c\` OK on x5 / x7 / x8 / x9, same absolute path; (3) LOOKS verdict lines eec89487… (landing via v3-integrator's docs batch); (4) S2a ≥ 10k audio-h, BP1 kept (291f92d6…); (5) pin 90b7f4d tests 30 / 30, this post.
- **extra_base_weight = a_LL / (a_LS + a_LL) = $BW** (exact FLAC frames of every row): a_LS = $A_LS s per row group, a_small = $A_SM, a_medium = $A_MD ($G_MD groups, $H_MD h), a_LL pooled = $A_LL. LibriSpeech share of audio, analytic check: $CHK.
- **Per-batch audit**, first 2,000 global batches, the pin's packing:

| window_groups | pooled LL share | median | p5 / **p95** | |
|---:|---:|---:|---:|---|
$AUD

- **window_groups = $WG** (rule fixed at 11:19Z, ACKed 11:17:33Z: $(echo $RULE | tr '_' ' ')).
- **Command** (per node, \`~/xN_s2b.sh\` → \`~/h53/h53_trunk.sh\` e783ea45, set file 72dc74e7, pin 90b7f4d / content 3780b451…, a1env, master x5:29756), node map as S2a (x5 ×2, x7 ×2, x8, x9):
  \`budget_s=7800 max_batch=1728 lr=0.00043 warmup=500 ema_decay=0.999398 workers=20 delay_penalty=0.005 extra_splits=<small v1.1>,<medium v1.1> extra_base_weight=$BW window_groups=$WG\`
  Unchanged from the set file: seed 7919, WSD-stable (total_steps 1e7, min_lr_ratio 0.02), chunk modes 4 / 8 / 16, ckpt every 500 steps.
- **Init:** S2a's exact window-boundary checkpoint, raw weights, one file copied to all four nodes under one hash (posted in the receipts). Fresh AdamW, EMA restarts. No gain augmentation (lead 10:20Z).
- **Next:** S2a window-boundary stop now → S2b fire → receipts (tree, init, cursor, rank map, first step) → 5-min telemetry.
B
