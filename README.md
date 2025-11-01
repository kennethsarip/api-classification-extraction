# corrections-classification

News article classification and extraction using Groq API.

## Setup

### Prerequisites

- Python 3.8 or higher
- Poetry (install from [poetry.python.org](https://python-poetry.org/docs/#installation))

### Installation

1. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Create a `.env` file with your Groq API key:
   ```bash
   GROQ_API_KEY=your_api_key_here
   ```

## Usage

Run the classification script:
```bash
poetry run python classification.py
```

Or activate the Poetry shell first:
```bash
poetry shell
python classification.py
```

## Configuration

Edit the configuration variables at the top of `classification.py`:
- `MODEL_NAME`: The Groq model to use (e.g., `"openai/gpt-oss-20b"`)
- `ARTICLE_FILES`: List of article files to process
- `GROUND_TRUTH_DATA`: Expected results for evaluation

## Adding New Articles

1. Create a new text file (e.g., `news-article-3.txt`)
2. Add it to the `ARTICLE_FILES` list in `classification.py`
3. Optionally add ground truth data to `GROUND_TRUTH_DATA`
