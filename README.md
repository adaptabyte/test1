---
title: Chatterbox TTS
emoji: 🍿
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
short_description: Expressive Zeroshot TTS
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## External Models

This demo requires external models for speech recognition and language
generation. Speech is transcribed using the Parakeet `nvidia/parakeet-tdt-0.6b-v2`
model while replies are generated with `Qwen/Qwen3-0.6B`. Both models are
downloaded on first use by the `transformers` library.

The app records speech from your microphone, transcribes it with Parakeet and
feeds the result to Qwen3 to generate a reply. The reply is then synthesized
back using your own voice as the reference prompt.
