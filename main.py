import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODELS = [
    "openrouter/free",
    "qwen/qwen3.6-27b",
    "cohere/north-mini-code:free",
    "nex-agi/nex-n2-pro:free",
]

client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=os.environ["OPENROUTER_API_KEY"],
)


def generate_post(topic: str) -> str:
    last_error = None
    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You write Telegram posts for a DevOps audience. "
                            "Write the main text in Persian. "
                            "Keep technical terms in English only when they are standard technical words, "
                            "for example: Docker, Kubernetes, CI/CD, Pipeline, GitLab, Runner, Cache, "
                            "Registry, Image, Container, Observability, Monitoring, Logging, Metrics, "
                            "Deployment, Rollback, SRE, DevSecOps. "
                            "Do not overuse English. "
                            "Use a practical, direct, non-hype style. "
                            "Do not invent facts, versions, CVEs, release dates, or commands. "
                            "Keep it suitable for Telegram. "
                            "Do not add an AI disclosure footer; the application will add it."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Write one useful daily DevOps Telegram post about this topic: {topic}. "
                            "Structure it with a short hook, one practical explanation, and one actionable takeaway."
                        ),
                    },
                ],
                temperature=0.7,
                max_tokens=900,
            )

            post = response.choices[0].message.content.strip()
            return post
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"All OpenRouter models failed. Last error: {last_error}")


def send_telegram(text: str) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHANNEL_ID"]

    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": False,
        },
        timeout=30,
    )
    response.raise_for_status()


if __name__ == "__main__":
    topic = os.getenv(
        "POST_TOPIC", "A practical DevOps lesson from Docker, CI/CD, Linux, Kubernetes, monitoring, or security"
    )

    post = generate_post(topic)
    send_telegram(post)
