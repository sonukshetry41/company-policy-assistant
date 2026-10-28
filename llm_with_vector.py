
import csv
import time
import os
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
client = OpenAI()

# Load the company policies
with open("data/company_policies.csv", newline="", encoding="utf-8-sig") as file:
    policies = list(csv.DictReader(file))

# Build a local TF-IDF vector index
documents = [
    p["title"] + " " + p["department"] + " " +
    p["category"] + " " + p["policy_text"]
    for p in policies
]

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
policy_vectors = vectorizer.fit_transform(documents)


def ask_with_vector(question):
    start_time = time.perf_counter()

    # Convert question into a vector
    question_vector = vectorizer.transform([question])

    # Find the most similar policies
    similarities = cosine_similarity(
        question_vector, policy_vectors
    )[0]

    # Select the three highest-scoring policies
    top_indices = similarities.argsort()[::-1][:3]

    # Reject questions with no keyword overlap
    if similarities[top_indices[0]] <= 0:
        elapsed = time.perf_counter() - start_time
        return (
            "Answer: No relevant policy found.\n"
            "Relevant Policy: None",
            elapsed,
            0,
            0,
            0
        )

    selected_policies = "\n\n".join(
        f"Policy: {policies[i]['title']}\n"
        f"Department: {policies[i]['department']}\n"
        f"Category: {policies[i]['category']}\n"
        f"Text: {policies[i]['policy_text']}"
        for i in top_indices
        if similarities[i] > 0
    )

    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=(
            "You are a company policy assistant. "
            "Answer ONLY using the supplied policy excerpts. "
            "Identify the exact relevant policy title. "
            "If the excerpts do not support an answer, "
            "say the information is unavailable. "
            "Do not invent company rules. "
            "Return exactly two lines: "
            "Answer: <answer> and "
            "Relevant Policy: <policy title or None>."
        ),
        input=(
            f"Relevant company policies:\n{selected_policies}\n\n"
            f"Employee question: {question}"
        ),
        max_output_tokens=250
    )

    elapsed = time.perf_counter() - start_time

    return (
        response.output_text,
        elapsed,
        response.usage.input_tokens,
        response.usage.output_tokens,
        response.usage.total_tokens
    )


if __name__ == "__main__":
    question = input("Ask a company policy question: ")

    try:
        answer, elapsed, input_tokens, output_tokens, total_tokens = (
            ask_with_vector(question)
        )

        print("\n" + answer)
        print(f"\nResponse Time: {elapsed:.3f} seconds")
        print(f"Input Tokens: {input_tokens}")
        print(f"Output Tokens: {output_tokens}")
        print(f"Total Tokens: {total_tokens}")

    except Exception as error:
        print("\nError:", error)

