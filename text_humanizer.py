"""
Text Humanizer Script
=====================
This script takes an essay and iteratively rewrites it to reduce its AI
detection probability.  It uses:
  - OpenAI (with the 'clude' model) for paraphrasing.
  - GPTZero API for AI-content detection.

The loop runs for a maximum of 10 iterations.  If the AI probability score
drops to 0.05 (5%) or below the loop exits early.
"""

import openai
import requests

# ---------------------------------------------------------------------------
# API Keys – replace the placeholder values with your actual keys
# ---------------------------------------------------------------------------
OPENAI_API_KEY = "your_key_here"
GPTZERO_API_KEY = "your_key_here"

# Maximum number of rewrite attempts
MAX_ITERATIONS = 10

# AI probability threshold (5%)
AI_THRESHOLD = 0.05

# GPTZero API endpoint for document-level AI detection
GPTZERO_API_URL = "https://api.gptzero.me/v2/predict/text"


# ---------------------------------------------------------------------------
# Paraphrase Function
# ---------------------------------------------------------------------------
def paraphrase_text(text: str) -> str:
    """Rewrite *text* so that it reads as naturally human-written.

    Uses the OpenAI Python library with the 'clude' model and a strong
    system prompt that encourages burstiness, perplexity, varied sentence
    length, natural transitions, and avoidance of common AI buzzwords.

    Returns the rewritten text, or the original text if the API call fails.
    """

    # Build the OpenAI client with the configured key
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = (
        "You are an expert human writer and editor. Your task is to rewrite "
        "the provided text so it sounds authentically human. Follow these "
        "guidelines strictly:\n"
        "1. Increase burstiness: mix very short punchy sentences with longer, "
        "more complex ones.\n"
        "2. Increase perplexity: use varied and occasionally unexpected word "
        "choices.\n"
        "3. Vary sentence length and structure throughout the text.\n"
        "4. Use natural, conversational transitions (avoid robotic connectors "
        "like 'Furthermore', 'Moreover', 'In conclusion').\n"
        "5. Avoid common AI-associated buzzwords such as 'delve', 'crucial', "
        "'landscape', 'foster', 'underpinning', 'realm', 'holistic'.\n"
        "6. Preserve the original meaning and all factual content.\n"
        "7. Return ONLY the rewritten text with no extra commentary."
    )

    try:
        response = client.chat.completions.create(
            model="clude",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=1.0,
        )
        rewritten = response.choices[0].message.content
        return rewritten.strip()
    except openai.APIConnectionError as exc:
        print(f"[Paraphrase] Connection error: {exc}")
        return text
    except openai.AuthenticationError as exc:
        print(f"[Paraphrase] Authentication error – check OPENAI_API_KEY: {exc}")
        return text
    except openai.APITimeoutError as exc:
        print(f"[Paraphrase] Request timed out: {exc}")
        return text
    except openai.APIError as exc:
        print(f"[Paraphrase] API error: {exc}")
        return text


# ---------------------------------------------------------------------------
# Detection Function (GPTZero)
# ---------------------------------------------------------------------------
def detect_ai_probability(text: str) -> float:
    """Return the overall AI probability score for *text* using the GPTZero API.

    Makes a POST request to the GPTZero endpoint and parses the JSON
    response to extract the score as a float (0.0 – 1.0).

    Returns 1.0 (worst case) if the request fails so the loop continues.
    """

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "x-api-key": GPTZERO_API_KEY,
    }

    payload = {
        "document": text,
    }

    try:
        response = requests.post(
            GPTZERO_API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        # GPTZero returns the overall probability under
        # documents[0].completely_generated_prob
        score = float(
            data["documents"][0]["completely_generated_prob"]
        )
        return score
    except requests.exceptions.Timeout:
        print("[Detection] Request timed out.")
        return 1.0
    except requests.exceptions.ConnectionError as exc:
        print(f"[Detection] Connection error: {exc}")
        return 1.0
    except requests.exceptions.HTTPError as exc:
        print(f"[Detection] HTTP error – check GPTZERO_API_KEY: {exc}")
        return 1.0
    except (KeyError, IndexError, ValueError) as exc:
        print(f"[Detection] Unexpected response format: {exc}")
        return 1.0
    except requests.exceptions.RequestException as exc:
        print(f"[Detection] Request error: {exc}")
        return 1.0


# ---------------------------------------------------------------------------
# Main Humanisation Loop
# ---------------------------------------------------------------------------
def humanize(essay: str) -> str:
    """Iteratively rewrite *essay* until the AI score is at or below the
    threshold, or the maximum number of iterations is reached.

    Returns the final version of the text.
    """

    current_text = essay

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---")

        # Step 1: Check the current AI detection score
        score = detect_ai_probability(current_text)
        print(f"AI probability score: {score:.4f}")

        # Step 2: If the score is at or below 5%, we are done
        if score <= AI_THRESHOLD:
            print(5)
            print("\nFinal rewritten text:\n")
            print(current_text)
            return current_text

        # Step 3: Rewrite the text to make it sound more human
        print("Score above threshold – paraphrasing…")
        current_text = paraphrase_text(current_text)

    # If we exhausted all iterations without passing the threshold
    print("\nMax iterations reached. Returning best version so far.\n")
    print(current_text)
    return current_text


# ---------------------------------------------------------------------------
# Entry Point – paste your essay in the variable below and run the script
# ---------------------------------------------------------------------------
original_essay = """Paste your essay text here."""

if __name__ == "__main__":
    humanize(original_essay)
