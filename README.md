# Lexi — Obrenna's voice edge

Lexi is the **ears and mouth** for [Obrenna](https://llm.alex-dyakin.com). It runs
on the voice-edge device — a **Raspberry Pi** (mic + speaker) — and holds all the
voice engines and logic. Obrenna's **brain** (the LLM) stays on **alison**.
**aragon** hosts Lexicon/Alchemy and can optionally run a lightweight STT-offload
service for Lexi (see below) — it does not run the voice loop itself.

**Only text crosses the network.** Lexi does mic → wake word → VAD → Whisper (STT)
locally (or offloaded to aragon — still just a transcript over the LAN), sends
the transcript to Obrenna's existing `POST /api/chat/stream`, streams the answer
tokens back, sentence-chunks them, and speaks them with Piper — all locally. No
audio leaves the Pi, and **nothing calls a cloud service**: every engine is local
inference on local model files.

```
Pi (Lexi):      mic → wake → VAD → STT ─┐
                                          │  text + account_id  (LAN → tunnel)
   Piper ← sentence-chunk ← tokens (SSE) ◄┼──────────────────────────────────►  alison: Obrenna gateway
                                          │                                        → POST /api/chat/stream
   speaker ◄─────────────────────────────┘                                        → small voice model

                 (optional) STT offload: Pi --[PCM over LAN]--> aragon: lexi.server --stt
```

## Layers

- **engines/** — `stt` (faster-whisper, or `stt_remote` to offload it — see
  below), `tts` (Piper), `wake` (openWakeWord), `vad` (webrtcvad). Local, CPU,
  lazy-loaded.
- **logic** — `pipeline` (the turn: wake→VAD→STT→identity→media/brain→chunk→
  TTS→speak), `session` (state + barge-in cancel token), `audio` (sentence
  chunker + mic/IO), `intents` (local device commands — see Music & media).
- **media** — talks to the Lexicon library directly (play/playlist/live-stream/
  skip/pause/resume via mpv), so music works even with the brain offline.
- **history** — a local, TTL'd JSONL transcript of every turn, for debugging.
- **identity** — resolves the speaking **Lexicon** user → `account_id` the brain
  scopes memory by (per-user memory).
- **obrenna_client** — talks to the brain: LAN-primary/tunnel-fallback, gateway
  shared-secret auth, SSE token streaming.

## Install

Base (client + logic only — enough to run text mode):

```bash
pip install -e .
```

Full engine stack (on the device with the mic + speaker — normally the Pi):

```bash
pip install -e ".[engines]"
python scripts/fetch_models.py   # one-time; afterwards Lexi runs offline
```

### Python 3.13 (e.g. Raspberry Pi OS / Debian 13) — wake word note

openWakeWord's `tflite-runtime` dependency has **no Python 3.13 wheel**, so
`.[engines]` fails on 3.13. Install the rest, then add openWakeWord without its
tflite dep and run it on **onnxruntime** (already pulled by faster-whisper/piper):

```bash
pip install faster-whisper==1.2.1 piper-tts==1.7.0 webrtcvad-wheels==2.0.14 \
            sounddevice==0.5.6 numpy==2.4.6
pip install --no-deps openwakeword==0.6.0
pip install requests tqdm scipy scikit-learn      # openWakeWord's real runtime deps
python scripts/fetch_models.py
```

`engines/wake.py` loads the model with `inference_framework="onnx"`, so no
tflite-runtime is needed at runtime.

## Configure

```bash
cp config.example.toml config.toml   # edit engine settings (models, wake word, etc.)
cp .env.example .env                 # optional: edit destinations for a standalone run
```

**Destinations** (bind host/port, Obrenna's brain URLs, the Lexicon URL, the
tool token) come from the monorepo's one destination registry, not from
`config.toml`: `.env.example` documents the keys (`LEXI_HOST`, `LEXI_PORT`,
`LEXICON_PORT`, `ALISON_LAN_IP`, `BRAIN_LAN_PORT`, `PUBLIC_BRAIN_URL`,
`PUBLIC_LEXICON_URL`, `LEXI_TOOL_TOKEN`, `LEXI_STT_REMOTE_URL`), with safe
standalone defaults so a lone Lexi checkout runs on its own. Layering (lowest
to highest): dataclass defaults/`config.toml` < Lexi's own `.env` < the
monorepo root `.env` < real process environment. See `.env.example` for
details and `lexi/config.py`'s module docstring for how it is loaded.

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

Start with **text mode** — it exercises the whole Pi→alison path (identity +
gateway auth + brain streaming) without a mic:

```bash
python -m lexi --no-speak        # type a message; prints the brain's answer
python -m lexi --say "what time is it" --no-speak
python -m lexi                   # text in, spoken out (needs engines)
python -m lexi --voice           # full mic loop: wake → listen → answer → speak
```

## Running it continuously (systemd --user service)

In production this runs as a per-user systemd service so it survives logout and
starts on boot, not as a manually-run foreground command:

```bash
# ~/.config/systemd/user/lexi-voice.service
[Unit]
Description=Lexi voice assistant
After=network-online.target sound.target

[Service]
WorkingDirectory=%h/Lexi
ExecStart=%h/Lexi/.venv/bin/python -m lexi --voice
Restart=on-failure

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now lexi-voice
loginctl enable-linger "$USER"   # so it keeps running after you log out / on boot
```

`__main__.py` already guards against two instances fighting over the mic
(`_claim_single_instance`, via a pidfile) — if an orphaned `--voice` process is
still holding the device, starting a new one kills it first.

## Music & media (Lexicon library)

Spoken music commands ("play X", "stop", "skip", "shuffle my music", "play the
live stream") are matched **locally** by `intents.py` before anything reaches
the brain — so music works even when alison is off. `media.py` talks to
Lexicon directly and drives playback through **mpv**, controlled live over its
JSON IPC socket (so skip/pause/resume don't require killing the process).

Lexicon also serves a synchronized **live stream** (`/api/livestream/*`) that
every listener hears at the same position; `play_livestream` joins it and a
background thread follows Lexicon's SSE update feed so the Pi switches tracks
when the communal stream advances.

Whether Lexi can speak over playing music (pause, speak, resume) or has to stop
it first depends on the device's audio routing — see **Pi audio** below.

## Conversation history

`history.py` keeps a local, TTL'd JSONL log of every voice turn (what Whisper
heard, and what Lexi did about it) under `logs/turns.jsonl` — useful for
debugging without a live log tail. It's gitignored and separate from the
brain's own per-user memory on alison; a logging failure never breaks a turn.

## STT offload (optional): running Whisper on aragon instead of the Pi

The Pi 4's CPU is the pipeline's slowest step. `lexi/server.py` can run as a
**second, STT-only service** on a stronger box on the LAN (aragon):

```bash
python -m lexi.server --stt --host 0.0.0.0 --port 8765
```

This builds *only* a `WhisperStt` — no TTS/wake/VAD/brain client — so the
offload host needs just `faster-whisper` installed, nothing else from
`.[engines]`. It accepts raw int16 mono PCM on `POST /stt` (not an encoded
clip), which keeps it on faster-whisper's numpy decode path — **no PyAV
needed**, which matters on Windows hosts where PyAV can be a fight to install.

On the Pi, point `LEXI_STT_REMOTE_URL` at it (e.g.
`http://192.168.1.4:8765`) and set `LEXI_TOOL_TOKEN` to the same value on both
machines. `engines/stt_remote.py`'s `RemoteStt` then POSTs each utterance there
and falls back to a local Whisper model automatically if the remote is
unreachable or errors — aragon is often started by hand, so voice must never
hard-depend on it being up. `--host 0.0.0.0` is required for the Pi to reach it
(the default, `127.0.0.1`, is loopback-only).

## Pi audio: mixing music + speech on one output (PipeWire)

The Pi's 3.5mm jack is a single-open ALSA device: a raw `plughw` player (mpv)
holds it exclusively, so TTS can't speak while music plays. Route **both** through
PipeWire (already running on Raspberry Pi OS) so they mix — this is what lets Lexi
pause music on wake, speak a reply, and resume, and gives audible skip
confirmations. One-time setup on the Pi:

```bash
sudo apt install -y pipewire-alsa          # ALSA → PipeWire bridge (adds pcm.pipewire)
wpctl status                                # confirm the default sink is the 3.5mm jack
                                            # ("bcm2835 Headphones") and the source is your mic
```

Then make the ALSA **default playback** go through PipeWire while capture stays on
the raw USB mic (`/etc/asound.conf`):

```
pcm.!default { type asym; playback.pcm "pipewire"; capture.pcm "plughw:CARD=M3,DEV=0" }
ctl.!default { type hw; card M3 }
```

and set in `config.toml`:

```toml
[engines]
mpv_audio_device = "pipewire"   # mpv's native PipeWire output (not raw ALSA)
audio_shared = true             # pause + speak-over + resume; without this, music
                                # is stopped before a spoken reply (safe fallback)
```

Audio is reached via the per-user PipeWire socket under `XDG_RUNTIME_DIR`;
interactive/SSH sessions set it, and Lexi defaults it to `/run/user/<uid>` if a
lean launcher (cron/systemd) leaves it unset.

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

## Deployment note

In practice this runs as a **standalone checkout** on the Pi (`~/Lexi`, cloned
directly from GitHub, deployed via `git fetch && git reset --hard origin/dev`),
not as a git submodule of `full-back-end-server`. The submodule path below
still works if you want aragon-side integration instead:

```bash
cd /path/to/full-back-end-server
git submodule add git@github.com:ablepacifist/Lexi.git Lexi
git commit -m "Add Lexi voice-edge submodule"
```

To clone the whole stack later: `git clone --recurse-submodules …`, or
`git submodule update --init --recursive` in an existing checkout.

## Lights: voice control of the Tapo bulbs

"Turn on the lamp", "make the bedroom light blue", "set both lights to 30 percent",
"dimmer", "make it red" — handled **locally** (`lexi/lights/`), matched *before*
media so `put on the lamp` is never taken for a song, and working with the brain
off.

```
Pi: wake → STT → light grammar ──► MQTT bridge (a home-automation hub) ──► bulbs   (preferred)
                              └──► direct from the Pi ──► bulbs                     (fallback if the hub is off)
```

**Set up (each machine that touches the bulbs — the voice device, and the hub for
the bridge):**

```bash
pip install '.[lights]'                    # python-kasa + paho-mqtt (pinned)
cp lights.example.toml lights.toml         # then edit: your names + each bulb's MAC
```
Put the secrets in `.env` (git-ignored; keys are in `.env.example`): `TAPO_USERNAME`/
`TAPO_PASSWORD` (the TP-Link account the bulbs were set up under — used **locally** to
derive the bulbs' auth key, no cloud calls) and, to prefer a bridge, `MQTT_HOST` (+
`MQTT_USERNAME`/`MQTT_PASSWORD`). No `MQTT_HOST` = direct control only.

```bash
python -m lexi.lights discover             # find the bulbs; prints MACs + a lights.toml snippet
python -m lexi.lights identify lamp        # blink one so you can see which bulb it is
python -m lexi.lights say "make the lamp blue"   # the voice grammar, without the voice
python -m lexi.lights say --direct "turn off the lights"   # skip the bridge
python -m lexi.lights state                # what the bulbs report
python -m lexi.lights bridge               # run the bridge (the hub's service)
```

Bulbs are found by **MAC**, not IP — the router hands out IPs and they change, so no
DHCP reservation is needed. To turn a MAC into an address, `TapoBackend` tries, in
order: the last-known address (saved to `logs/light_hosts.json`, connected to
directly), **ARP** (`/proc/net/arp` — the network layer's own MAC→IP table, reliable),
and only then UDP broadcast discovery, retried once — broadcast replies are
frequently lost on Wi-Fi, so it's last on purpose. If a bulb moves, the saved
address heals itself on the next command.

**Newer bulb firmware:** some Tapo bulbs ship firmware using TP-Link's newer `TPAP`
protocol, which `python-kasa` doesn't support yet. `discover` prints these as
`UNSUPPORTED`. The fix that has worked in practice: remove the bulb from the Tapo
app and re-add it from scratch (a full re-pair) — this has reliably brought a bulb
back onto the older, supported protocol even when the "Third-Party Compatibility"
toggle (Me → Third-Party Services, in the Tapo app) did not. Don't block the
bulbs' internet access either way: on some firmware that makes them reboot-loop.

**The hub's broker:** Mosquitto needs a LAN listener with authentication (its
default listener is localhost-only) — add `/etc/mosquitto/conf.d/lexi.conf`:

```
listener 1883
allow_anonymous false
password_file /etc/mosquitto/passwd
acl_file /etc/mosquitto/acl
```
```
# /etc/mosquitto/acl — the voice device may command lights and read their state;
# only the bridge may write acks, state and its own status.
user lexi
topic write home/lights/+/set
topic read  home/lights/#

user lights-bridge
topic read  home/lights/+/set
topic write home/lights/+/ack
topic write home/lights/+/state
topic write home/lights/bridge/status
```

MQTT topics: `home/lights/<id>/set` (command), `.../ack`, retained `.../state`, and
a retained `home/lights/bridge/status` (`online`/`offline`, via MQTT Last Will) —
this is how the voice device knows instantly whether to skip the bridge, so a
powered-off hub costs zero extra latency on top of a normal command. Node-RED (or
anything else) can publish/subscribe the same topics for schedules and scenes.
