"""
Synonym Expansion — LLM proposes, human approves.

Reads concepts.json (your manually-seeded business concepts and their
confirmed synonyms), asks a LOCALLY-RUNNING Ollama model to propose
additional plausible synonyms per concept, and writes the suggestions
to synonym_candidates.json for review.

No external API, no API key, no data leaving your machine — this
talks to Ollama's local server only (default: http://localhost:11434).

Nothing here is written back to concepts.json automatically — see
merge_approved.py for that step, which only merges what you've marked
approved.

Requirements:
  1. Ollama installed and running (https://ollama.com)
  2. A model pulled, e.g.:  ollama pull llama3.1
  3. pip install requests
"""

import json
import requests
from pathlib import Path

CONCEPTS_FILE = Path("concepts.json")
CANDIDATES_FILE = Path("synonym_candidates.json")

OLLAMA_URL = "C:\Users\nadir.ali\.ollama"
OLLAMA_MODEL = "llama3.1"  # change to whatever model you've pulled locally

# Context helps the LLM stay on-domain rather than guessing generically.
# Adjust this to describe your actual business briefly.
DOMAIN_CONTEXT = (
    "This is an insurance / Takaful (family takaful) business database "
    "used by Pak Qatar Family Takaful. Concepts relate to policies, "
    "participants/clients, agents, branches, premiums/contributions, "
    "claims, and underwriting."
)


DEFAULT_CONCEPTS = {
    "CLIENT": {
        "synonyms": ["Client Name", "Participant Name", "Customer", "Client"],
        "source": "manual",
        "status": "confirmed",
    },
    "NATIONAL_ID": {
        "synonyms": ["CNIC Number", "CNIC", "National Identity Card Number", "NIC Number", "Identity Card No"],
        "source": "manual",
        "status": "confirmed",
    },
}


def load_concepts() -> dict:
    if not CONCEPTS_FILE.exists():
        print(f"{CONCEPTS_FILE} not found — creating it with starter concepts (CLIENT, NATIONAL_ID).")
        with open(CONCEPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONCEPTS, f, indent=2)
    with open(CONCEPTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(concept_name: str, seed_synonyms: list) -> str:
    return f"""{DOMAIN_CONTEXT}

A database column has been manually tagged as belonging to the business
concept "{concept_name}", with these confirmed synonym terms a user
might use when asking a question:

{", ".join(seed_synonyms)}

Suggest up to 8 ADDITIONAL plausible synonym terms a business user in
this domain might use to refer to the same concept, that are NOT
already in the list above. Only include terms that mean the SAME
thing — do not include related-but-different concepts (for example,
"Beneficiary" is NOT a synonym for "Client", it's a different role).

Respond ONLY with a JSON array of objects, no other text, no markdown
fences. Each object must have exactly these fields:
- "term": the suggested synonym
- "reasoning": one short sentence on why it likely means the same thing

If you cannot confidently suggest any additional terms, return an
empty array [].
"""


def get_llm_suggestions(concept_name: str, seed_synonyms: list) -> list:
    prompt = build_prompt(concept_name, seed_synonyms)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Could not reach Ollama at " + OLLAMA_URL + ". "
            "Is Ollama running? Start it, then try again."
        )

    raw_text = response.json().get("response", "").strip()

    # Strip accidental markdown fences if the model adds them anyway
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        suggestions = json.loads(raw_text)
    except json.JSONDecodeError:
        print(f"  [WARN] Could not parse model response for {concept_name}, skipping.")
        print(f"  Raw response: {raw_text[:300]}")
        return []

    return suggestions


def run():
    concepts = load_concepts()

    all_candidates = {}

    for concept_name, data in concepts.items():
        seed_synonyms = data.get("synonyms", [])
        print(f"Expanding concept: {concept_name} ({len(seed_synonyms)} seed synonyms)")

        suggestions = get_llm_suggestions(concept_name, seed_synonyms)

        all_candidates[concept_name] = {
            "seed_synonyms": seed_synonyms,
            "candidates": [
                {
                    "term": s.get("term"),
                    "reasoning": s.get("reasoning"),
                    "approved": None,  # to be set True/False during review
                }
                for s in suggestions
            ],
        }

        print(f"  -> {len(suggestions)} candidate(s) proposed")

    with open(CANDIDATES_FILE, "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, indent=2)

    print(f"\nReview candidates in: {CANDIDATES_FILE.resolve()}")
    print("Set \"approved\": true or false on each candidate, then run merge_approved.py")


if __name__ == "__main__":
    run()