
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from rules_based import search_policies

# Load secret tokens from .env
load_dotenv(Path(__file__).resolve().parent / ".env")

# Create Slack bot
app = App(token=os.environ["SLACK_BOT_TOKEN"])


@app.event("app_mention")
def handle_policy_question(event, say):
    # Remove Slack mention from the question
    question = re.sub(
        r"<@[^>]+>",
        "",
        event.get("text", "")
    ).strip()

    if not question:
        say(
            text="Please ask me a company policy question.",
            thread_ts=event["ts"]
        )
        return

    try:
        # Use the existing rules-based policy search
        answer, policy = search_policies(question)

        # Send the answer to Slack
        say(
            text=(
                f"*Question:* {question}\n"
                f"*Answer:* {answer}\n"
                f"*Relevant Policy:* {policy}"
            ),
            thread_ts=event["ts"]
        )

    except Exception as error:
        print("Error:", error)
        say(
            text="Sorry, I couldn't search the company policies.",
            thread_ts=event["ts"]
        )


if __name__ == "__main__":
    print("Starting Company Policy Assistant Bot...")

    SocketModeHandler(
        app,
        os.environ["SLACK_APP_TOKEN"]
    ).start()

