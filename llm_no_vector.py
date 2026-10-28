
import csv
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def ask_llm(question):
    start_time = time.perf_counter()

    with open("data/company_policies.csv", newline="", encoding="utf-8-sig") as file:
        policies = list(csv.DictReader(file))

    policy_database = json.dumps(policies, ensure_ascii=False)

    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=(
            "You are a company policy assistant. "
            "Answer questions using ONLY the supplied company policies. "
            "Identify the exact relevant policy title. "
            "If the database does not contain the answer, say so. "
            "Never invent company rules. "
            "Return exactly two lines: "
            "Answer: <answer> and Relevant Policy: <policy title or None>."
        ),
        input=(
            f"Company policy database:\n{policy_database}\n\n"
            f"Employee question: {question}"
        ),
        max_output_tokens=250
    )

    elapsed = time.perf_counter() - start_time

    return response, elapsed


if __name__ == "__main__":
    question = input("Ask a company policy question: ")

    try:
        response, elapsed = ask_llm(question)

        print("\n" + response.output_text)
        print(f"\nResponse Time: {elapsed:.3f} seconds")
        print(f"Input Tokens: {response.usage.input_tokens}")
        print(f"Output Tokens: {response.usage.output_tokens}")
        print(f"Total Tokens: {response.usage.total_tokens}")

    except Exception as error:
        print("\nError:", error)

