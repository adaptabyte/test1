import random
import numpy as np
import torch
from chatterbox.src.chatterbox.tts import ChatterboxTTS
import gradio as gr
import spaces
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    pipeline,
)
import soundfile as sf
import tempfile

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# --- Global Model Initialization ---
MODEL = None
STT_MODEL = None
LLM_MODEL = None
TOKENIZER = None

def get_or_load_model():
    """Loads the ChatterboxTTS model if it hasn't been loaded already,
    and ensures it's on the correct device."""
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        try:
            MODEL = ChatterboxTTS.from_pretrained(DEVICE)
            if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
                MODEL.to(DEVICE)
            print(f"Model loaded successfully. Internal device: {getattr(MODEL, 'device', 'N/A')}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return MODEL

def get_or_load_stt():
    """Loads the Parakeet speech recognition pipeline."""
    global STT_MODEL
    if STT_MODEL is None:
        model_name = "nvidia/parakeet-tdt-0.6b-v2"
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(DEVICE)
        STT_MODEL = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0 if DEVICE == "cuda" else -1,
        )
    return STT_MODEL

def get_or_load_llm():
    global LLM_MODEL, TOKENIZER
    if LLM_MODEL is None or TOKENIZER is None:
        model_name = "Qwen/Qwen3-0.6B"
        TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        LLM_MODEL = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    return LLM_MODEL, TOKENIZER

# Attempt to load the model at startup.
try:
    get_or_load_model()
except Exception as e:
    print(f"CRITICAL: Failed to load model on startup. Application may not function. Error: {e}")

def set_seed(seed: int):
    """Sets the random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

@spaces.GPU
def generate_tts_audio(
    text_input: str,
    audio_prompt_path_input: str,
    exaggeration_input: float,
    temperature_input: float,
    seed_num_input: int,
    cfgw_input: float
) -> tuple[int, np.ndarray]:
    """
    Generates TTS audio using the ChatterboxTTS model.

    Args:
        text_input: The text to synthesize (max 300 characters).
        audio_prompt_path_input: Path to the reference audio file.
        exaggeration_input: Exaggeration parameter for the model.
        temperature_input: Temperature parameter for the model.
        seed_num_input: Random seed (0 for random).
        cfgw_input: CFG/Pace weight.

    Returns:
        A tuple containing the sample rate (int) and the audio waveform (numpy.ndarray).
    """
    current_model = get_or_load_model()

    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    print(f"Generating audio for text: '{text_input[:50]}...'")
    wav = current_model.generate(
        text_input[:300],  # Truncate text to max chars
        audio_prompt_path=audio_prompt_path_input,
        exaggeration=exaggeration_input,
        temperature=temperature_input,
        cfg_weight=cfgw_input,
    )
    print("Audio generation complete.")
    return (current_model.sr, wav.squeeze(0).numpy())

@spaces.GPU
def generate_conversation_response(mic_audio_path: str) -> tuple[int, np.ndarray, str, str]:
    """Converts a microphone recording into a conversational reply using
    STT, an LLM, and TTS.

    Args:
        mic_audio_path: Path to the recorded microphone audio.

    Returns:
        Tuple containing sample rate, generated waveform, transcript and LLM reply text.
    """
    if not mic_audio_path:
        raise ValueError("No audio provided")

    stt = get_or_load_stt()
    llm, tokenizer = get_or_load_llm()
    current_model = get_or_load_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        data, sr = sf.read(mic_audio_path)
        sf.write(tmp.name, data, sr)
        audio_path = tmp.name

    # Speech to text
    stt_output = stt(audio_path)
    transcript = stt_output["text"] if isinstance(stt_output, dict) else stt_output

    # LLM response
    inputs = tokenizer(transcript, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        output_ids = llm.generate(**inputs, max_new_tokens=128)
    reply_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # TTS using the user's voice as prompt
    wav = current_model.generate(
        reply_text,
        audio_prompt_path=audio_path,
    )
    return current_model.sr, wav.squeeze(0).numpy(), transcript, reply_text

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Chatterbox Conversational Demo
        Speak into the microphone and let the model respond in your own voice.
        """
    )

    mic = gr.Audio(sources=["microphone"], type="filepath", label="Speak")

    with gr.Row():
        transcript_box = gr.Textbox(label="Transcript")
        reply_box = gr.Textbox(label="Model Reply")

    audio_output = gr.Audio(label="Synthesized Response")
    run_btn = gr.Button("Talk")

    run_btn.click(
        fn=generate_conversation_response,
        inputs=[mic],
        outputs=[audio_output, transcript_box, reply_box],
    )

demo.launch()
