import gradio as gr


def respond(message: str, history: list[tuple[str, str]]) -> str:
    """Simple echo bot used to demonstrate a conversational interface."""
    return f"You said: {message}"


demo = gr.ChatInterface(
    fn=respond,
    title="Chatterbox Chat Demo",
    description=(
        "Type into the textbox and see the demo repeat your words."
    ),
)

if __name__ == "__main__":
    demo.launch()
