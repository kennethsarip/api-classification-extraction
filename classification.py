# Model Configuration
MODEL_NAME = "openai/gpt-oss-20b"
TEMPERATURE = 0
MAX_COMPLETION_TOKENS = 8192
TOP_P = 1
REASONING_EFFORT = "medium"
STREAM_RESPONSE = True

# Article Files - Add more article files here to test
ARTICLE_FILES = [
    "news-article-1.txt",
    "news-article-2.txt",
]

# Ground Truth Data - Add ground truth for each article file
GROUND_TRUTH_DATA = {
    "news-article-1.txt": {
        "classification": {"topic": "politics", "sentiment": "neutral", "urgency": "high"},
        "extraction": {
            "namesOfPeople": ["Kwame Raoul", "JB Pritzker", "Dick Durbin"],
            "location": ["Chicago", "Illinois", "Texas"],
            "day": "October 6, 2025",
            "event": "lawsuit against National Guard deployment",
            "effects": ["unrest", "economic harm", "constitutional violation"]
        }
    },
    "news-article-2.txt": {
        "classification": {"topic": "politics", "sentiment": "neutral", "urgency": "medium"},
        "extraction": {
            "namesOfPeople": ["Ahmed al-Sharaa", "Tom Barrack", "Trump"],
            "location": ["Washington, D.C.", "Syria", "Saudi Arabia"],
            "day": "November 1, 2025",
            "event": "Syrian president meeting with Trump at White House",
            "effects": ["diplomatic relations", "sanctions lifted", "rebuilding"]
        }
    },
}


from groq import Groq
import json
import os
import re
from pathlib import Path
from pydantic import BaseModel
from enum import Enum
from dotenv import load_dotenv

class Topic(str, Enum):
    politics = "politics"
    business = "business"
    crime = "crime"
    sports = "sports"
    entertainment = "entertainment"
    tech = "tech"
    world = "world"
    local = "local"
    other = "other"

class Sentiment(str, Enum):
    negative = "negative"
    neutral = "neutral"
    positive = "positive"

class Urgency(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class NewsClassification(BaseModel):
    topic: Topic
    sentiment: Sentiment
    urgency: Urgency

class NewsExtraction(BaseModel):
    namesOfPeople: list[str]
    location: list[str]
    day: str
    event: str
    effects: list[str]

class AnalyzeResult(BaseModel):
    classification: NewsClassification
    extraction: NewsExtraction


def read_article_file(filepath: str) -> str:
    """Read article content from a text file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Article file not found: {filepath}")
    return path.read_text().strip()

def extract_json_from_text(text: str) -> str:
    """Return the first top-level JSON object or array found in text.
    Handles fenced code blocks like ```json ...``` and extra prose.
    """
    cleaned = text.strip()
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\n", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n```$", "", cleaned)

    # Try to locate a JSON object
    brace_start = cleaned.find("{")
    bracket_start = cleaned.find("[")
    starts = [s for s in [brace_start, bracket_start] if s != -1]
    if not starts:
        raise ValueError("No JSON start delimiter found in model output")
    start = min(starts)

    # Walk and find matching closing bracket/brace
    stack = []
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if ch in "[{":
            stack.append(ch)
        elif ch in "]}":
            if not stack:
                continue
            open_ch = stack.pop()
            if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                raise ValueError("Mismatched JSON delimiters in model output")
            if not stack:
                # Found the end of top-level JSON
                return cleaned[start:idx+1]
    # If we exit loop with stack not empty, JSON was incomplete
    raise ValueError("Incomplete JSON found in model output")

def f1_score(predicted: list, ground_truth: list) -> float:
    """Calculate F1 score between predicted and ground truth lists.
    
    F1 score is the harmonic mean of precision and recall:
    - Precision = TP / (TP + FP) - how many predictions are correct
    - Recall = TP / (TP + FN) - how many ground truth items were found
    - F1 = 2 × (precision × recall) / (precision + recall)
    
    Args:
        predicted: List of predicted items
        ground_truth: List of ground truth items
    
    Returns:
        F1 score between 0 and 1
    """
    pred_set = set(predicted)
    truth_set = set(ground_truth)
    
    # True Positives: items in both lists
    tp = len(pred_set & truth_set)
    
    # False Positives: items in predicted but not in ground truth
    fp = len(pred_set - truth_set)
    
    # False Negatives: items in ground truth but not in predicted
    fn = len(truth_set - pred_set)
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Calculate F1 score (harmonic mean)
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def evaluate_similarity_percentage(predicted: dict, ground_truth: dict) -> float:
    """Calculate overall similarity percentage between predicted and ground truth."""
    scores = {}
    
    # Classification (exact match for enums)
    scores["topic"] = 1.0 if predicted["classification"]["topic"] == ground_truth["classification"]["topic"] else 0.0
    scores["sentiment"] = 1.0 if predicted["classification"]["sentiment"] == ground_truth["classification"]["sentiment"] else 0.0
    scores["urgency"] = 1.0 if predicted["classification"]["urgency"] == ground_truth["classification"]["urgency"] else 0.0
    
    # Extraction (F1 score)
    scores["namesOfPeople"] = f1_score(
        predicted["extraction"]["namesOfPeople"], 
        ground_truth["extraction"]["namesOfPeople"]
    )
    scores["location"] = f1_score(
        predicted["extraction"]["location"], 
        ground_truth["extraction"]["location"]
    )
    scores["effects"] = f1_score(
        predicted["extraction"]["effects"], 
        ground_truth["extraction"]["effects"]
    )
    
    # Convert to percentage
    average_score = sum(scores.values()) / len(scores)
    return round(average_score * 100, 2)


# Analysis Functions
def analyze_article(client: Groq, text: str) -> dict:
    """Analyze a news article and return the classification and extraction results."""
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": f"""You are a news article analyzer. Return ONLY valid JSON (no prose, no code fences) for this schema: {AnalyzeResult.model_json_schema()}

Labeling rules:
- politics: government, elections, policy, lawsuits against government officials
- crime: criminal activity, arrests, charges, court cases between private parties
- business: companies, markets, finance
- local: city/state-specific events without national scope
- If unsure, set topic to 'other'
- urgency: low/medium/high based on immediacy and consequences described

Analyze the following article:
{text}"""
            }
        ],
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        top_p=TOP_P,
        reasoning_effort=REASONING_EFFORT,
        stream=STREAM_RESPONSE,
    )

    # Stream model response
    output_chunks: list[str] = []
    for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        output_chunks.append(content)
        print(content, end="")

    # Parse and validate JSON
    raw_output = "".join(output_chunks)
    try:
        json_str = extract_json_from_text(raw_output)
        data = json.loads(json_str)
        validated = AnalyzeResult.model_validate(data)
        return validated.model_dump()
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from model output: {e}")

def process_article(client: Groq, article_file: str, ground_truth: dict = None):
    """Process a single article file and optionally compare with ground truth."""
    print(f"\n{'='*80}")
    print(f"Processing: {article_file}")
    print(f"{'='*80}\n")
    
    # Read article
    text = read_article_file(article_file)
    
    # Analyze article
    print("Analysis Result:")
    result = analyze_article(client, text)
    
    # Compare with ground truth if available
    if ground_truth:
        similarity = evaluate_similarity_percentage(result, ground_truth)
        print(f"\n{'='*80}")
        print(f"Similarity Score: {similarity}%")
        print(f"{'='*80}\n")
        return result, similarity
    else:
        print(f"\n{'='*80}\n")
        return result, None

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main function to process all articles."""
    # Load environment variables
    load_dotenv()
    
    # Initialize client
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    # Process each article
    results = []
    for article_file in ARTICLE_FILES:
        ground_truth = GROUND_TRUTH_DATA.get(article_file)
        result, similarity = process_article(client, article_file, ground_truth)
        results.append({
            "file": article_file,
            "result": result,
            "similarity": similarity
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for r in results:
        print(f"\n{r['file']}:")
        print(f"  Topic: {r['result']['classification']['topic']}")
        print(f"  Sentiment: {r['result']['classification']['sentiment']}")
        print(f"  Urgency: {r['result']['classification']['urgency']}")
        if r['similarity'] is not None:
            print(f"  Similarity: {r['similarity']}%")
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()