# Lexi — Obrenna's voice edge

Lexi is the **ears and mouth** for [Obrenna](https://llm.alex-dyakin.com). It runs
on **aragon** (alongside Lexicon/Alchemy) and holds the mic/speaker plus all the
voice engines and logic. Obrenna's **brain** (the LLM) stays on **alison**.

**Only text crosses the network.** Lexi does mic → wake word → VAD → Whisper (STT)
locally, sends the transcript to Obrenna's existing `POST /api/chat/stream`,
streams the answer tokens back, sentence-chunks them, and speaks them with Piper
— all locally. No audio leaves aragon, and **nothing calls a cloud service**:
every engine is local inference on local model files.

```
aragon (Lexi):  mic → wake → VAD → STT ─┐
                                          │  text + account_id  (LAN → tunnel)
   Piper ← sentence-chunk ← tokens (SSE) ◄┼──────────────────────────────────►  alison: Obrenna gateway
                                          │                                        → POST /api/chat/stream
   speaker ◄─────────────────────────────┘                                        → small voice model
```

## Layers

- **engines/** — `stt` (faster-whisper), `tts` (Piper), `wake` (openWakeWord),
  `vad` (webrtcvad). Local, CPU, lazy-loaded.
- **logic** — `pipeline` (the turn: wake→VAD→STT→identity→brain→chunk→TTS→speak),
  `session` (state + barge-in cancel token), `audio` (sentence chunker + mic/IO).
- **identity** — resolves the speaking **Lexicon** user → `account_id` the brain
  scopes memory by (per-user memory).
- **obrenna_client** — talks to the brain: LAN-primary/tunnel-fallback, gateway
  shared-secret auth, SSE token streaming.

## Install

Base (client + logic only — enough to run text mode):

```bash
pip install -e .
```

Full engine stack (on the aragon device, with a mic + PortAudio):

```bash
pip install -e ".[engines]"
python scripts/fetch_models.py   # one-time; afterwards Lexi runs offline
```

## Configure

```bash
cp config.example.toml config.toml   # edit alison's LAN IP, models, etc.
```

The Obrenna gateway shared secret is **not** in the config. Provide it via env or
file (same token the gateway expects in `OBRENNA_AGENT_TOKEN`):

```bash
export OBRENNA_AGENT_TOKEN="$(cat /path/to/shared/token)"   # or ./.agent_token
```

Optional per-device Lexicon session for identity (else `default_account_id`):

```bash
export LEXICON_SESSION="<JSESSIONID>"
```

## Run

Start with **text mode** — it exercises the whole aragon→alison path (identity +
gateway auth + brain streaming) without a mic:

```bash
python -m lexi --no-speak        # type a message; prints the brain's answer
python -m lexi --say "what time is it" --no-speak
python -m lexi                   # text in, spoken out (needs engines)
python -m lexi --voice           # full mic loop: wake → listen → answer → speak
```

## Brain side (alison) — already landed

Obrenna reuses its existing streaming route; Lexi passes two optional fields:
`account_id` (per-user memory) and `orchestrator` (small voice model, e.g.
`qwen3.5-4b-claude-opus-reasoning-distilled-v2`). No new brain route.

## Gateway route (required companion change, on alison)

The Obrenna gateway currently authenticates `/api/chat/*` with a **browser
cookie**. Lexi is headless, so add a Caddy route that authenticates the voice
call with the same **shared secret** the codebase-agent uses
(`/_auth/verify-agent`, `X-Obrenna-Agent-Token`). In the gateway `Caddyfile`,
mirror the codebase-agent block for the chat-stream path:

```caddy
handle /api/chat/stream {
    forward_auth localhost:9100 {
        uri /_auth/verify-agent
        copy_headers Authorization X-Obrenna-Agent-Token
    }
    reverse_proxy localhost:8000
}
```

(Browser chat continues to use the cookie-checked `handle {}` block for the
non-streaming route. Keep `OBRENNA_AGENT_TOKEN` ≥ 32 chars; the verifier fails
closed.)

## Add Lexi as a submodule of `full-back-end-server` (on aragon)

```bash
cd /path/to/full-back-end-server
git submodule add git@github.com:ablepacifist/Lexi.git Lexi
git commit -m "Add Lexi voice-edge submodule"
```

To clone the whole stack later: `git clone --recurse-submodules …`, or
`git submodule update --init --recursive` in an existing checkout.
