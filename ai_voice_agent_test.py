import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a friendly, concise voice assistant on a company website.
Keep answers short (1-3 sentences) since they will be spoken aloud.
Answer questions about the business helpfully and naturally.
Do not use markdown, bullet points, or emojis — this text will be converted to speech.
If you don't know something, say so plainly and offer to connect them with a human.
"""


MODEL = "llama-3.3-70b-versatile"


def get_reply(user_message: str, history: list[dict] | None = None) -> str:
    """Send a user message to Groq and get back the assistant's spoken reply."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=150,
    )
    return response.choices[0].message.content


def chat_loop():
    """Interactive test loop — type messages like a visitor talking to the widget."""
    print(f"Voice agent test (model: {MODEL}). Type 'quit' to exit.\n")
    history: list[dict] = []

    while True:
        user_input = input("Visitor: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        reply = get_reply(user_input, history)
        print(f"Agent:   {reply}\n")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        history = history[-10:]


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print(" Set your GROQ_API_KEY environment variable first.")
        print('   export GROQ_API_KEY="your-key-here"\n')
    chat_loop()
