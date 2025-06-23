import gradio as gr


def respond(message: str, history: list[tuple[str, str]]) -> str:
    """Simple echo bot used to demonstrate a conversational interface."""
    return f"You said: {message}"


demo = gr.ChatInterface(
    fn=respond,
    title="Chatterbox Chat Demo",
    description=(
        "This is a minimal conversational interface that echoes back the user "
        "message. Replace the `respond` function with a real TTS model for a "
        "fully interactive experience."
    ),
)

if __name__ == "__main__":
    demo.launch()
