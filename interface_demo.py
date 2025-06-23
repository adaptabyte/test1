import numpy as np
import gradio as gr


def dummy_generate_audio(
    text_input: str,
    audio_prompt_path_input: str,
    exaggeration_input: float,
    temperature_input: float,
    seed_num_input: int,
    cfgw_input: float,
) -> tuple[int, np.ndarray]:
    """Return a short tone as a placeholder for real TTS output."""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), False)
    tone = np.sin(2 * np.pi * 440 * t)
    return sr, tone


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Chatterbox TTS Demo (Interface Only)
        This file shows only the Gradio interface without loading the full model.
        """
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Hello world!",
                label="Text to synthesize (max chars 300)",
                max_lines=5,
            )
            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio File (Optional)",
            )
            exaggeration = gr.Slider(
                0.25,
                2,
                step=0.05,
                label="Exaggeration (Neutral = 0.5, extreme values can be unstable)",
                value=0.5,
            )
            cfg_weight = gr.Slider(0.2, 1, step=0.05, label="CFG/Pace", value=0.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=0.05, label="Temperature", value=0.8)

            run_btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    run_btn.click(
        fn=dummy_generate_audio,
        inputs=[text, ref_wav, exaggeration, temp, seed_num, cfg_weight],
        outputs=[audio_output],
    )

if __name__ == "__main__":
    demo.launch()
