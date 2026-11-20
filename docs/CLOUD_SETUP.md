# Cloud ASR / TTS setup (optional)

SONJ runs fully offline by default. Cloud backends are an **option** you can switch on per stage in the **Pipeline view** (rail item ⇶): click ASR or TTS and pick a cloud provider, then paste your API key. Keys are kept **in memory for the session only** (never written to disk) and the app **falls back to local** if a cloud call fails or the key is missing.

> Privacy note: cloud ASR sends your microphone audio to the provider; cloud TTS sends the reply text. For a private/standalone build (e.g. the Raspberry Pi), keep everything local. Prices below are approximate — verify at signup.

---

## Your $5-per-account budget — recommendation

| Provider | Best for | Cost reality on $5 | Verdict |
| --- | --- | --- | --- |
| **Deepgram** | ASR (speech-to-text) | **$200 free credit** on signup; then ~$0.0043/min (Nova-3). $5 ≈ 1,160 min | **Best value** — the free credit alone covers months of testing |
| **OpenAI** | ASR + TTS | Pay-as-you-go, $5 minimum credit. Whisper-1 $0.006/min; tts-1 $0.015/1K chars. $5 ≈ 830 min ASR or ~330K chars TTS | Flexible, one key for both stages |
| **ElevenLabs** | TTS (premium voices) | Free 10K credits/mo; **Starter $5/mo** (often $1 first month) ≈ 30K chars (~30 min speech) | Best voice quality; $5/mo recurring |

**Suggested split on your budget:** Deepgram for ASR (free $200 credit → effectively $0), and either OpenAI tts-1 or ElevenLabs Starter ($5) for TTS. You do not need all three.

---

## 1) Deepgram (ASR)

1. Go to **https://console.deepgram.com/signup** → sign up (email or GitHub/Google). No card needed for the free credit.
2. You get **$200 free credit** automatically.
3. In the console: **Settings → API Keys → Create a new API key** (name it `sonj`, permission `Member`). Copy the key (shown once).
4. In SONJ: **Pipeline view → ASR → Deepgram (cloud)** → paste the key.
5. Cost after credit: Nova-3 ~**$0.0043/min** (~$0.26/hour). $5 ≈ 1,160 minutes.

## 2) OpenAI (ASR + TTS)

1. Go to **https://platform.openai.com/signup** → create an account (phone verification required).
2. **Settings → Billing → Add payment method**, then **Add credits** (minimum **$5**). Pay-as-you-go; no subscription.
3. **Settings → API keys → Create new secret key** → copy it (shown once).
4. In SONJ: **Pipeline view → ASR → OpenAI (cloud)** and/or **TTS → OpenAI (cloud)** → paste the same key.
5. Costs: **Whisper-1 $0.006/min** ASR; **tts-1 $0.015/1K characters**, tts-1-hd $0.030/1K. $5 ≈ 830 min ASR or ~330K chars TTS.
   - Set a **usage limit** in Billing so you cannot overspend.

## 3) ElevenLabs (TTS)

1. Go to **https://elevenlabs.io/sign-up** → create an account.
2. Free tier: **10,000 credits/month** (~10 min speech). To go further: **Subscription → Starter ($5/month**, frequently **$1 for the first month**) → 30,000 credits/month.
3. Click your **profile icon → API Keys** → copy the key.
4. In SONJ: **Pipeline view → TTS → ElevenLabs (cloud)** → paste the key.
5. Cost: Starter **$5/mo** recurring (~30K chars ≈ 30 min). Flash v2.5 models use fewer credits per character. Cancel anytime; unused credits do not roll over.

---

## Switching backends (Pipeline view)

- **ASR stage**: `parakeet` (local, default, ~0.14s/utterance) · `whisper`/`moonshine` (local) · `deepgram`/`openai` (cloud).
- **TTS stage**: local neural voices (lessac fast / Kokoro personas) · `openai`/`elevenlabs` (cloud).
- **LLM stage**: your Copilot/OpenRouter model (cloud) — the local router is a separate helper, see below.
- Any cloud stage with no key, or that errors, **falls back to local automatically**.

## The local router (and the Raspberry Pi)

The "local router" was a small local helper model (qwen2.5-coder:7b via Ollama) used only for cheap sub-tasks: picking acknowledgement words, hinting turn type, judging whether you finished a thought (smart-turn), and summarizing long recaps. It is **not** the conversation brain (that is Front on Copilot/OpenRouter).

**It is OFF by default and should stay OFF on a Raspberry Pi 4 (4GB).** A 7B model needs ~6GB and will not fit alongside the voice pipeline. SONJ's **keyword routing** covers acknowledgements and turn hints with zero RAM, which is the right design for the Pi. The `router_enabled` toggle exists for a laptop with spare RAM if you want smart-turn back, but keep it off for the Pi build.
