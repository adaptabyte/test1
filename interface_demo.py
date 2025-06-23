import gradio as gr


def respond(message: str, history: list[tuple[str, str]]) -> str:
    """Simple echo bot used to demonstrate a conversational interface."""
    return f"You said: {message}"


demo = gr.ChatInterface(
    fn=respond,
    title="Chatterbox Speech Chat Demo",
    description=(
        "Speak into your microphone and hear the demo repeat your words. "
        "This uses Gradio's built-in Whisper and gTTS for speech input and output."
    ), 
    input_audio="microphone",
    output_audio=True, 
)

if __name__ == "__main__":
    demo.launch()
