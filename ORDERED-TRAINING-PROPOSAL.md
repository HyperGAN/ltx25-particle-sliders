# LTX 2.5 ordered integration and training proposal

Status: proposal only. The earliest unmet stage is native integration with the
existing shared particle-sliders lifecycle. The current separately named modern
cohort admits only a dummy CPU fixture. It is an adapter GAN without a routed
particle bank. It has a framewise spatial critic, which the bounded CPU probe
shows is insensitive to frame order. Neither that fixture nor shared synthetic
CUDA evidence qualifies a pretrained model, an order-sensitive video critic,
media rendering or concept quality.

The source-only publication is now explicitly authorized. That authorization
does not approve this training proposal or native execution. The shared
core source target is `a794e3b4a8813b012f2c4faa3da82b78d4a90214`; historical
UNI and all old packet source identities remain unchanged. The readiness branch
and its CPU diagnostic have not received a confirmed Claude RETURN review.

## 1. Implement and qualify the shared integration

First obtain a reviewed native LTX 2.5 loader and staged-model identity, exact
immutable model/component/shard and dependency receipts, and a borrowed-model
factory that validates source, geometry and ownership before hooks. Preserve the
current modern proposal's TE-last-four/video-connector hosts, rank 4 / alpha 4,
frozen base and frozen neutral audio conditioning. Keep the full genuine native
video field and masks. Prove native packing and time/scale conventions from the
actual model. Native model/source/data access and root-owned scheduling are
prerequisites, not instructions to download or launch anything here.

The current modern experiment uses frozen-plus video velocity as its teacher;
it is not the historical working post-connector embedding UNI objective. The
historical documentation reports nearly identical plus/neutral velocity and a
failed velocity UNI experiment. Therefore measure the native teacher separation
on fixed held-out rows before considering a GAN run. If the signal is negligible,
stop for a reviewed objective/domain decision. Do not silently replace the
teacher, flatten token fields, bypass shared-domain admission or treat adversarial
loss as proof of editing.

A true embedding teacher would require a reviewed shared `[B,T,D]` token-response
contract with masks, valid-token penalty semantics, source/domain admission and
checkpoint identity. Temporal video qualification also needs a reviewed,
identity-bound temporal critic/graph and matched output, VJP and second-derivative
checks. Coordinate these with the existing shared-core and temporal owners via root.
No temporal architecture or GAN mathematics is invented in this proposal.

Before any tuning, root must obtain Claude actual-diff and RETURN review and a
bounded GPU allocation. The first proposed two-device qualification is OFF
identity; base/adapter gradient ownership and one update; raw/EMA responses only
if genuinely supported; exact split versus uninterrupted checkpoint/resume; and
cleanup of hooks, modes, borrowed state and RNG. The earlier readiness proposal
caps this at six total paired updates and twenty minutes, subject to stricter root
budgets. Failure, nonfinite state, unexpected base gradients, OOM, mismatch or
budget conflict stops the run; there is no automatic fallback or retry.

## 2. Learning-rate and metric proposal — not a result

The actual shared `bispf_optimized` preset in the qualified PG 0.9.4 source
inherits `lr = 0.012` and `d_lr_mult = 1.5` from `bcap`, with a cosine schedule.
The current LTX dummy factory does not expose an LR option. Historical LTX UNI
cards use AdamW at `0.0002` for 700 steps; that is a different optimizer and
objective, not an optimal GAN learning rate.

After native admission, teacher-separation and quality instrumentation pass,
propose an independently approved pilot at multipliers 0.5, 1 and 2 of the
actual shared preset LR: `0.006`, `0.012` and `0.024`. This is an untested search
bracket, not a recommendation that any point is safe. Keep the shared preset,
critic-to-generator LR ratio, batch, schedule, data, initialization, seeds and
budget fixed. Set the shared schedule horizon to `total_steps=1024` for every
pilot and for the final run; pre-register a common earlier pilot stop point.
Otherwise a short cosine schedule is not comparable to the eventual run.
Because the critic LR is derived from the base LR, the bracket changes both
rates at the preset ratio. The native factory must expose and bind
this shared public `recipe_overrides` option in its new cohort configuration;
the existing dummy factory must not be modified to impersonate native support.

Select an LR using a pre-registered, blinded, paired video evaluation against
OFF on held-out subjects, prompts and seeds: target direction improvement together
with acceptable identity, composition, motion and non-target preservation. Record
the same fixed evaluation protocol, invalid/capped outputs and uncertainty for
every candidate. Require independent media/rendering qualification first. With
only the numerical response API currently available, this metric is unavailable
and no LR can yet be selected. Thresholds and the pilot budget need review before
allocation; do not choose them after seeing outcomes.

Teacher separation, response alignment, finite gradients, GAN losses and critic
penalties are diagnostic proxies and safety checks. They are not native concept
quality or a substitute for blinded held-out evaluation. If no candidate meets
the pre-registered preservation criteria, select none.

## 3. Direction candidates — not selected sliders

The source contains two smile rows and two chiaroscuro rows. These are meaningful
candidate concepts, not enough independent material for training and held-out
selection. Smile uses closed mouth versus visible teeth while holding subject,
clothing, camera, motion and sound; chiaroscuro varies lighting while holding
subject or still-life composition. Historical embedding-UNI results report weak
mid-scale identity for smile and weak composition preservation for chiaroscuro.
Those observations do not establish modern GAN quality.

After the integration and evaluation gates, create and independently review
additional subject-disjoint and scene-disjoint prompt pairs plus neutral and
unrelated controls. Keep audio assessment with its separate owner. Choose a
direction only after its frozen teacher has a measured signal and its paired
video results pass the same held-out target/preservation criteria. A weak or
confounded teacher is a stop, not a reason to claim a direction was selected.

## 4. Full run and sampling — contingent on earlier results

Only after an LR and direction are actually selected, propose a separately
reviewed 1024-update run with immutable data/source/model/optimizer identities,
root-owned two-device allocation and explicit wall-time and per-device memory
caps. Save exact resumable checkpoints at steps 0, 64, 128, 256, 512, 768 and
1024, plus any approved failure receipt. Do not reset scheduler or optimizer
state when resuming. OFF, raw and EMA numerical checks apply only where the
reviewed native factory supports them. A rendering/sampling bridge must be
separately qualified; the present numerical dummy API cannot produce native media.

Stop on invalid identities or geometry, failed OFF, nonfinite state, detached
adapter, base gradient, temporal/mask violation, degraded held-out controls,
checkpoint mismatch, incomplete cleanup, timeout, OOM or root occupancy conflict.
Do not launch, claim a card, download weights, publish outputs or auto-retry from
this plan. Root owns scheduling, Claude review, any later allocation and outputs.
