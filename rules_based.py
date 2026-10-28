import csv
import re

def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def search_policies(question):
    with open("data/company_policies.csv", newline="", encoding="utf-8-sig") as file:
        policies = list(csv.DictReader(file))

    question_words = tokenize(question)

    stop_words = {
        "what", "is", "the", "a", "an", "of", "to",
        "for", "in", "on", "how", "does", "do",
        "are", "can", "i", "my", "company", "policy"
    }

    question_words -= stop_words

    best_policy = None
    best_score = 0

    for policy in policies:
        title_words = tokenize(policy["title"])
        text_words = tokenize(policy["policy_text"])

        score = (
            3 * len(question_words & title_words)
            + len(question_words & text_words)
        )

        if score > best_score:
            best_score = score
            best_policy = policy

    if best_policy is None:
        return "No relevant policy found.", "None"

    return best_policy["policy_text"], best_policy["title"]


if __name__ == "__main__":
    question = input("Ask a company policy question: ")

    answer, policy = search_policies(question)

    print("\nAnswer:", answer)
    print("Relevant Policy:", policy)
