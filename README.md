# text-humanizer-app

A Python script that iteratively rewrites an essay until it passes an AI detection test (GPTZero). It uses OpenAI for paraphrasing and GPTZero for detection scoring.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Open `text_humanizer.py` and set your API keys at the top of the file:

```python
OPENAI_API_KEY = "your_key_here"
GPTZERO_API_KEY = "your_key_here"
```

## Usage

1. Paste your essay into the `original_essay` variable at the bottom of `text_humanizer.py`.
2. Run the script:

```bash
python text_humanizer.py
```

The script will repeatedly check the essay's AI probability score via GPTZero and rewrite it using OpenAI until the score drops to 5% or below (or 10 iterations are reached).

## Running Tests

```bash
python -m pytest test_text_humanizer.py -v
```