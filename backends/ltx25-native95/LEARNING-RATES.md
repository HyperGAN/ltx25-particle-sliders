# Warm-sunset illumination: the matched RAW32 results

Both tested rates tied on the one declared finished-video case. The frozen model instrument gave a shared OFF mean of0.4117647058823529 and ON mean0.5 for each arm, so both gains are exactly0.08823529411764708. Retain both saved recipes; this comparison selects no unique rate or automatic default.

| Saved arm | G | D multiplier | Actual D | Unused prior multiplier | Completed updates | Intended-effect gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G012 |0.012 |1.5 |0.018000000000000002 |2.5 |32 |+0.08823529411764708 |
| G001 |0.001 |18 |0.018000000000000002 |2.5 |32 |+0.08823529411764708 |

The [portable recipe JSON](recipes/warm-sunset-RAW32-provisional.json) records both tuples with `rate_overrides:null`, `provisional_rate_overrides:null`, `selected_unique_rate:null` and `ema_decay:null`. There is no actual particle bank in this model. The nominal prior rates0.03 and0.0025 are unused. Preserve the exact stored D float; do not independently round or redesign the fixed-D comparison.

Each arm has32 completed updates under the original horizon128, batch2, rank4/alpha4 and seed7. Inference used strength1 and the original8 coupled AV steps, CFG1, STG0, modality scale1, width768/height512,17 frames at24fps and native48k stereo audio. The target is warm low-angle sunset illumination in finished video. All17 frames have equal weight1/17; their normalized illumination levels form one clip mean, then matched ON minus shared OFF. This is one matched case, not17 independent cases. Audio, generic beauty, preservation, confidence and teacher SSE are not secondary selectors.

The frozen Qwen3-VL-8B v4 instrument is provisional and has no formal trait qualification. Whole joint-AV catastrophic eligibility remains unmeasured. The tie is the measured result under this declared instrument and case; it does not issue a catastrophic PASS, FULL32 guard, automatic promotion, ordinary Generate readiness or released weights. It does not require rerating existing first observations, rerendering the completed OFF or running an extra confirmation/search.

Both arms reused the same completed four-teacher/two-initial-row cache. The fresh G001 arm matched original initialized components and non-rate RNG before update1. G012 used exact saved continuations and an explicitly recovered completed latent boundary; G001 used the public latent-return phase before the original decoder tail. Official nonreentrant checkpointing and frozen/inactive CPU storage are recorded execution profiles. The full original Gemma/connectors/5D video response and joint AV arithmetic remain; different execution histories preclude an LR-only causal or bitwise-equivalence claim.

Use the values with their source profile `ltx25-native-gemma4-te4-video-connectors-r4-a4-convnd-first-row-recovery-v4`, captured product commit `edcf9c3b121191f95e2b0153bf5e7797afb6ca1a`, original model revision `4791a1100e0cf97f7ed5c55c12dd71dce13f9dd3`, and the exact ConvNd PG/core prerequisites described in [the owner API](README.md). They are not interchangeable settings for the historical UNI/dummy route or a stock/newer shared-core installation.

For exact continuation, load the authentic session and typed FULL together and preserve its cohort, cache, paths, horizon, initialization receipt, optimizers and RNG. Do not apply either rate tuple as an override to an already saved cohort. Fresh work binds scoped initialization before optimizer creation and preserves pretrained weights/zero-UP leaves; it uses `resume=None` and its own declared cohort. RAW inference uses `selection='raw'` and explicit `ema_decay=None`. Completed teachers, updates and saved output observations are reused.

The new owner wrapper's local wheel build, installed-target tests and generated console refusal passed four software checks. That is separate evidence from the captured native previews and the matched32 comparison. Public exact ConvNd dependency artifacts and full native execution through this installed wheel have not been established by that software check.
