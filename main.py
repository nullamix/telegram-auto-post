from pathlib import Path
from string import Template
import argparse
import os

import requests
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODELS = [
    "openrouter/free",
    "qwen/qwen3.6-27b",
    "cohere/north-mini-code:free",
    "nex-agi/nex-n2-pro:free",
]


def read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def render_prompt(template_text: str, variables: dict[str, str]) -> str:
    return Template(template_text).safe_substitute(variables)


def load_prompt_set(prompt_set: str, variables: dict[str, str]) -> tuple[str, str]:
    prompt_dir = Path("prompts") / prompt_set

    system_template = read_text_file(prompt_dir / "system.txt")
    user_template = read_text_file(prompt_dir / "user.txt")

    system_prompt = render_prompt(system_template, variables)
    user_prompt = render_prompt(user_template, variables)

    return system_prompt, user_prompt


def generate_post(client: OpenAI, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int = 900) -> str:
    last_error = None
    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            post_content = response.choices[0].message.content
            if post_content:
                return post_content.strip()
        except OpenAIError as e:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt-set",
        default=os.getenv("PROMPT_SET", "example_daily"),
        help="Prompt folder name inside prompts/",
    )

    parser.add_argument(
        "--audience",
        default=os.getenv("POST_AUDIENCE", "daily readers interested in something"),
        help="Target audience",
    )

    parser.add_argument(
        "--tone",
        default=os.getenv("POST_TONE", "direct, practical, concise"),
        help="Writing tone",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("TEMPERATURE", "0.7")),
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("MAX_TOKENS", "900")),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print post instead of sending to Telegram",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    variables = {
        "audience": args.audience,
        "tone": args.tone,
    }

    system_prompt, user_prompt = load_prompt_set(args.prompt_set, variables)

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    post = generate_post(
        client=client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if args.dry_run:
        print(post)
        return

    send_telegram(post)


if __name__ == "__main__":
    main()
