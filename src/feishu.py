import os
import requests


def _webhook() -> str:
    url = os.environ.get("FEISHU_WEBHOOK_URL_ASTOCK")
    if not url:
        raise RuntimeError("FEISHU_WEBHOOK_URL_ASTOCK not set")
    return url


def send_text(text: str) -> None:
    resp = requests.post(_webhook(), json={"msg_type": "text", "content": {"text": text}}, timeout=10)
    resp.raise_for_status()


def send_card(title: str, content: str, color: str = "blue") -> None:
    card = {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": color,
            },
            "elements": [
                {"tag": "markdown", "content": content}
            ],
        },
    }
    resp = requests.post(_webhook(), json=card, timeout=10)
    resp.raise_for_status()
