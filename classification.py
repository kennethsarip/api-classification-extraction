# Model Configuration
OPENAI_MODEL_NAME = "gpt-4o"
DEEPSEEK_MODEL_NAME = "deepseek-chat"
TEMPERATURE = 0  # Temperature=0 for deterministic, reproducible results (important for evaluation)
MAX_COMPLETION_TOKENS = 8192
TOP_P = 1
STREAM_RESPONSE = True

# Evaluation Configuration
NUM_EVALUATION_RUNS = 1  # Number of times to run each article (set to 1 for single comparison, higher for reliability testing)
EVALUATION_MODE = False  # Set to True to run reliability evaluation instead of single run

# Reliability Configuration
MAX_RETRIES = 3  # Maximum number of retries for failed API calls
INITIAL_RETRY_DELAY = 1.0  # Initial delay in seconds for exponential backoff
MAX_RETRY_DELAY = 60.0  # Maximum delay in seconds
RATE_LIMIT_RETRY_DELAY = 30.0  # Initial delay for rate limit errors (increased to avoid hitting limits)
RATE_LIMIT_EXPONENTIAL_BASE = 2.0  # Exponential backoff multiplier for rate limits
DELAY_BETWEEN_RUNS = 2.0  # Delay in seconds between evaluation runs to avoid rate limits (increase if hitting limits)

# Article Files - Add more article files here to test
ARTICLE_FILES = [
    "news-article-1.txt",
    "news-article-2.txt",
    "news-article-3.txt",
    "news-article-4.txt",
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
    "news-article-3.txt": {
        "classification": {"topic": "world", "sentiment": "neutral", "urgency": "high"},
        "extraction": {
            "namesOfPeople": ["Trump", "Pete Hegseth", "Bola Ahmed Tinubu", "Ted Cruz"],
            "location": ["Nigeria", "Washington, D.C.", "U.S."],
            "day": "November 1, 2025",
            "event": "Trump threatens to cut off aid to Nigeria and orders Pentagon to prepare for possible action over persecution of Christians",
            "effects": ["diplomatic tensions", "potential sanctions", "threat of military action", "religious freedom concerns"]
        }
    },
    "news-article-4.txt": {
        "classification": {"topic": "politics", "sentiment": "neutral", "urgency": "medium"},
        "extraction": {
            "namesOfPeople": ["Mark Carney", "Trump", "Doug Ford", "Ronald Reagan"],
            "location": ["Canada", "Ontario", "U.S.", "Washington, D.C."],
            "day": "November 1, 2025",
            "event": "Canada's Prime Minister apologizes to Trump over anti-tariff ad",
            "effects": ["trade tensions", "tariff hikes", "diplomatic relations", "trade talks ended"]
        }
    },
}


from openai import OpenAI, RateLimitError, APIConnectionError, APIError
import json
import os
import re
import time
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from pydantic import BaseModel
from enum import Enum
from dotenv import load_dotenv
from typing import Optional, Tuple

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
def analyze_article_openai_with_retry(client: OpenAI, text: str, silent: bool = False, error_tracker: dict = None) -> Tuple[dict, dict]:
    """Analyze a news article using OpenAI with retry logic and error tracking.
    
    Returns:
        Tuple of (result_dict, error_info_dict)
        error_info_dict contains: retries, rate_limited, errors
    """
    error_info = {
        "retries": 0,
        "rate_limited": False,
        "errors": [],
        "total_time": 0.0
    }
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            start_time = time.time()
            completion = client.chat.completions.create(
                model=OPENAI_MODEL_NAME,
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
                max_tokens=MAX_COMPLETION_TOKENS,
                top_p=TOP_P,
                stream=STREAM_RESPONSE,
            )

            # Stream model response
            output_chunks: list[str] = []
            for chunk in completion:
                content = chunk.choices[0].delta.content or ""
                output_chunks.append(content)
                if not silent:
                    print(content, end="")

            # Parse and validate JSON
            raw_output = "".join(output_chunks)
            json_str = extract_json_from_text(raw_output)
            data = json.loads(json_str)
            validated = AnalyzeResult.model_validate(data)
            error_info["total_time"] = time.time() - start_time
            return validated.model_dump(), error_info
            
        except RateLimitError as e:
            error_info["rate_limited"] = True
            error_info["errors"].append(f"RateLimitError (attempt {attempt + 1}): {str(e)}")
            if attempt < MAX_RETRIES:
                # Exponential backoff for rate limits: 30s, 60s, 120s
                delay = RATE_LIMIT_RETRY_DELAY * (RATE_LIMIT_EXPONENTIAL_BASE ** attempt)
                delay = min(delay, MAX_RETRY_DELAY * 2)  # Cap at 120 seconds
                if not silent:
                    print(f"\n⚠️  Rate limit hit (OpenAI). Waiting {delay:.1f}s before retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(delay)
                error_info["retries"] += 1
            else:
                raise
        except (APIConnectionError, APIError) as e:
            error_info["errors"].append(f"APIError (attempt {attempt + 1}): {type(e).__name__}: {str(e)}")
            if attempt < MAX_RETRIES:
                delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                # Silent retry - details shown in summary
                time.sleep(delay)
                error_info["retries"] += 1
            else:
                raise
        except (ValueError, RuntimeError) as e:
            # JSON parsing errors - don't retry
            error_info["errors"].append(f"ParseError: {str(e)}")
            raise RuntimeError(f"Failed to parse JSON from OpenAI model output: {e}")
        except Exception as e:
            error_info["errors"].append(f"UnknownError (attempt {attempt + 1}): {type(e).__name__}: {str(e)}")
            if attempt < MAX_RETRIES:
                delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                time.sleep(delay)
                error_info["retries"] += 1
            else:
                raise
    
    # Should never reach here, but just in case
    raise RuntimeError("Max retries exceeded")

def analyze_article_openai(client: OpenAI, text: str, silent: bool = False) -> dict:
    """Analyze a news article using OpenAI and return the classification and extraction results."""
    result, _ = analyze_article_openai_with_retry(client, text, silent)
    return result

def analyze_article_deepseek_with_retry(client: OpenAI, text: str, silent: bool = False, error_tracker: dict = None) -> Tuple[dict, dict]:
    """Analyze a news article using DeepSeek with retry logic and error tracking.
    
    Returns:
        Tuple of (result_dict, error_info_dict)
    """
    error_info = {
        "retries": 0,
        "rate_limited": False,
        "errors": [],
        "total_time": 0.0
    }
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            start_time = time.time()
            completion = client.chat.completions.create(
                model=DEEPSEEK_MODEL_NAME,
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
                max_tokens=MAX_COMPLETION_TOKENS,
                top_p=TOP_P,
                stream=False,  # DeepSeek API
            )

            # Get response (not streaming for DeepSeek)
            raw_output = completion.choices[0].message.content
            if not silent:
                print(raw_output)
            
            # Parse and validate JSON
            json_str = extract_json_from_text(raw_output)
            data = json.loads(json_str)
            validated = AnalyzeResult.model_validate(data)
            error_info["total_time"] = time.time() - start_time
            return validated.model_dump(), error_info
            
        except RateLimitError as e:
            error_info["rate_limited"] = True
            error_info["errors"].append(f"RateLimitError (attempt {attempt + 1}): {str(e)}")
            if attempt < MAX_RETRIES:
                # Exponential backoff for rate limits: 30s, 60s, 120s
                delay = RATE_LIMIT_RETRY_DELAY * (RATE_LIMIT_EXPONENTIAL_BASE ** attempt)
                delay = min(delay, MAX_RETRY_DELAY * 2)  # Cap at 120 seconds
                if not silent:
                    print(f"\n⚠️  Rate limit hit (DeepSeek). Waiting {delay:.1f}s before retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(delay)
                error_info["retries"] += 1
            else:
                raise
        except (APIConnectionError, APIError) as e:
            error_info["errors"].append(f"APIError (attempt {attempt + 1}): {type(e).__name__}: {str(e)}")
            if attempt < MAX_RETRIES:
                delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                # Silent retry - details shown in summary
                time.sleep(delay)
                error_info["retries"] += 1
            else:
                raise
        except (ValueError, RuntimeError) as e:
            # JSON parsing errors - don't retry
            error_info["errors"].append(f"ParseError: {str(e)}")
            raise RuntimeError(f"Failed to parse JSON from DeepSeek model output: {e}")
        except Exception as e:
            error_info["errors"].append(f"UnknownError (attempt {attempt + 1}): {type(e).__name__}: {str(e)}")
            if attempt < MAX_RETRIES:
                delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                time.sleep(delay)
                error_info["retries"] += 1
            else:
                raise
    
    raise RuntimeError("Max retries exceeded")

def analyze_article_deepseek(client: OpenAI, text: str, silent: bool = False) -> dict:
    """Analyze a news article using DeepSeek and return the classification and extraction results."""
    result, _ = analyze_article_deepseek_with_retry(client, text, silent)
    return result

def process_article_pair(openai_client: OpenAI, deepseek_client: OpenAI, article_file: str, silent: bool = False):
    """Process a single article file with both models and compare their outputs.
    
    Args:
        openai_client: OpenAI client instance (Model A)
        deepseek_client: DeepSeek/OpenAI client instance (Model B)
        article_file: Path to article file
        silent: If True, suppress output (useful for evaluation runs)
    
    Returns:
        Tuple of (openai_result, deepseek_result, similarity_score, openai_time, deepseek_time, openai_error_info, deepseek_error_info)
    """
    if not silent:
        print(f"\n{'='*80}")
        print(f"Processing: {article_file}")
        print(f"{'='*80}\n")
    
    # Read article
    text = read_article_file(article_file)
    
    # Analyze with OpenAI (Model A) - with error tracking
    if not silent:
        print("Model A (OpenAI) Analysis:")
    openai_result, openai_error_info = analyze_article_openai_with_retry(openai_client, text, silent=silent)
    openai_time = openai_error_info["total_time"]
    
    # Analyze with DeepSeek (Model B) - with error tracking
    if not silent:
        print(f"\nModel B (DeepSeek) Analysis:")
    deepseek_result, deepseek_error_info = analyze_article_deepseek_with_retry(deepseek_client, text, silent=silent)
    deepseek_time = deepseek_error_info["total_time"]
    
    # Compare the two model outputs
    similarity = evaluate_similarity_percentage(openai_result, deepseek_result)
    
    if not silent:
        print(f"\n{'='*80}")
        print(f"Model A vs Model B Similarity: {similarity}%")
        print(f"  OpenAI time:  {openai_time:.2f}s")
        print(f"  DeepSeek time: {deepseek_time:.2f}s")
        if openai_error_info["retries"] > 0 or deepseek_error_info["retries"] > 0:
            print(f"  Retries - OpenAI: {openai_error_info['retries']}, DeepSeek: {deepseek_error_info['retries']}")
        print(f"{'='*80}\n")
    
    return openai_result, deepseek_result, similarity, openai_time, deepseek_time, openai_error_info, deepseek_error_info

def compare_results(result1: dict, result2: dict) -> dict:
    """Compare two analysis results and return consistency metrics.
    
    Returns:
        Dictionary with consistency scores:
        - exact_match: Whether results are identical
        - classification_match: Whether classifications match exactly
        - topic_match: Whether topics match
        - sentiment_match: Whether sentiments match
        - urgency_match: Whether urgencies match
        - extraction_similarity: Average F1 score across extraction fields
    """
    # Classification comparison
    topic_match = result1["classification"]["topic"] == result2["classification"]["topic"]
    sentiment_match = result1["classification"]["sentiment"] == result2["classification"]["sentiment"]
    urgency_match = result1["classification"]["urgency"] == result2["classification"]["urgency"]
    classification_match = topic_match and sentiment_match and urgency_match
    
    # Extraction comparison using F1 scores
    names_f1 = f1_score(
        result1["extraction"]["namesOfPeople"],
        result2["extraction"]["namesOfPeople"]
    )
    location_f1 = f1_score(
        result1["extraction"]["location"],
        result2["extraction"]["location"]
    )
    effects_f1 = f1_score(
        result1["extraction"]["effects"],
        result2["extraction"]["effects"]
    )
    
    # Compare event and day (exact match for these)
    event_match = result1["extraction"]["event"] == result2["extraction"]["event"]
    day_match = result1["extraction"]["day"] == result2["extraction"]["day"]
    
    extraction_similarity = (names_f1 + location_f1 + effects_f1) / 3.0
    
    exact_match = (
        classification_match and
        names_f1 == 1.0 and
        location_f1 == 1.0 and
        effects_f1 == 1.0 and
        event_match and
        day_match
    )
    
    return {
        "exact_match": exact_match,
        "classification_match": classification_match,
        "topic_match": topic_match,
        "sentiment_match": sentiment_match,
        "urgency_match": urgency_match,
        "extraction_similarity": extraction_similarity,
        "names_f1": names_f1,
        "location_f1": location_f1,
        "effects_f1": effects_f1,
        "event_match": event_match,
        "day_match": day_match,
    }

def evaluate_model_comparison(openai_client: OpenAI, deepseek_client: OpenAI, article_file: str, num_runs: int = NUM_EVALUATION_RUNS):
    """Run multiple evaluations comparing two LLMs on the same article.
    
    Args:
        openai_client: OpenAI client instance (Model A)
        deepseek_client: DeepSeek/OpenAI client instance (Model B)
        article_file: Path to article file
        num_runs: Number of times to run the comparison
    
    Returns:
        Dictionary with evaluation statistics
    """
    # Get ground truth for this article
    ground_truth = GROUND_TRUTH_DATA.get(article_file)
    
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON EVALUATION: {article_file}")
    print(f"Model A: {OPENAI_MODEL_NAME} (OpenAI)")
    print(f"Model B: {DEEPSEEK_MODEL_NAME} (DeepSeek)")
    if ground_truth:
        print(f"Ground Truth: Available (topic={ground_truth['classification']['topic']}, sentiment={ground_truth['classification']['sentiment']}, urgency={ground_truth['classification']['urgency']})")
    else:
        print(f"Ground Truth: Not available for this article")
    if num_runs > 1:
        print(f"Running {num_runs} iterations for reliability testing...")
        if DELAY_BETWEEN_RUNS > 0:
            print(f"Delay between runs: {DELAY_BETWEEN_RUNS}s")
    else:
        print(f"Running single comparison...")
    print(f"{'='*80}\n")
    
    openai_results = []
    deepseek_results = []
    similarities = []  # Model A vs Model B similarity
    openai_ground_truth_similarities = []  # Model A vs Ground Truth
    deepseek_ground_truth_similarities = []  # Model B vs Ground Truth
    openai_timings = []
    deepseek_timings = []
    
    # Reliability tracking
    openai_errors = defaultdict(int)
    deepseek_errors = defaultdict(int)
    openai_retry_counts = []
    deepseek_retry_counts = []
    openai_rate_limited_count = 0
    deepseek_rate_limited_count = 0
    successful_runs = 0
    failed_runs = 0
    error_details = []
    
    # Track successful calls before rate limits
    openai_successful_calls_before_rate_limit = []
    deepseek_successful_calls_before_rate_limit = []
    openai_total_successful_calls = 0
    deepseek_total_successful_calls = 0
    
    # Run multiple times comparing both models
    for run_num in range(1, num_runs + 1):
        print(f"Run {run_num}/{num_runs}...", end=" ", flush=True)
        try:
            openai_result, deepseek_result, similarity, openai_time, deepseek_time, openai_error_info, deepseek_error_info = process_article_pair(
                openai_client, deepseek_client, article_file, silent=True
            )
            openai_results.append(openai_result)
            deepseek_results.append(deepseek_result)
            similarities.append(similarity)
            openai_timings.append(openai_time)
            deepseek_timings.append(deepseek_time)
            openai_retry_counts.append(openai_error_info["retries"])
            deepseek_retry_counts.append(deepseek_error_info["retries"])
            
            # Compare each model to ground truth
            if ground_truth:
                openai_gt_similarity = evaluate_similarity_percentage(openai_result, ground_truth)
                deepseek_gt_similarity = evaluate_similarity_percentage(deepseek_result, ground_truth)
                openai_ground_truth_similarities.append(openai_gt_similarity)
                deepseek_ground_truth_similarities.append(deepseek_gt_similarity)
            
            # Track successful calls
            openai_total_successful_calls += 1
            deepseek_total_successful_calls += 1
            
            # If rate limited on this run, record how many calls were made before
            if openai_error_info["rate_limited"]:
                openai_rate_limited_count += 1
                openai_successful_calls_before_rate_limit.append({
                    "run": run_num,
                    "calls_before": openai_total_successful_calls,
                    "retries": openai_error_info["retries"]
                })
            if deepseek_error_info["rate_limited"]:
                deepseek_rate_limited_count += 1
                deepseek_successful_calls_before_rate_limit.append({
                    "run": run_num,
                    "calls_before": deepseek_total_successful_calls,
                    "retries": deepseek_error_info["retries"]
                })
            
            successful_runs += 1
            status = "✓"
            retry_info = ""
            if openai_error_info["retries"] > 0 or deepseek_error_info["retries"] > 0:
                retry_info = f" [Retries: OpenAI={openai_error_info['retries']}, DeepSeek={deepseek_error_info['retries']}]"
            
            similarity_info = f"Similarity: {similarity:.2f}%"
            if ground_truth:
                openai_gt = openai_ground_truth_similarities[-1]
                deepseek_gt = deepseek_ground_truth_similarities[-1]
                similarity_info += f" | GT: OpenAI={openai_gt:.2f}%, DeepSeek={deepseek_gt:.2f}%"
            print(f"{status} (OpenAI: {openai_time:.2f}s, DeepSeek: {deepseek_time:.2f}s, {similarity_info}){retry_info}")
            
            # Add delay between runs to avoid rate limits
            if run_num < num_runs and DELAY_BETWEEN_RUNS > 0:
                time.sleep(DELAY_BETWEEN_RUNS)
            
        except RateLimitError as e:
            failed_runs += 1
            error_type = type(e).__name__
            # Show simplified error - full details in summary
            print(f"✗ RateLimitError (calls before: OpenAI={openai_total_successful_calls}, DeepSeek={deepseek_total_successful_calls})")
            error_details.append({
                "run": run_num, 
                "error": error_type, 
                "message": str(e),
                "openai_calls_before": openai_total_successful_calls,
                "deepseek_calls_before": deepseek_total_successful_calls
            })
            # Try to determine which API had the issue
            if "openai" in str(e).lower():
                openai_errors[error_type] += 1
                # Record failed rate limit with call count
                openai_successful_calls_before_rate_limit.append({
                    "run": run_num,
                    "calls_before": openai_total_successful_calls,
                    "failed": True
                })
            else:
                deepseek_errors[error_type] += 1
                deepseek_successful_calls_before_rate_limit.append({
                    "run": run_num,
                    "calls_before": deepseek_total_successful_calls,
                    "failed": True
                })
        except Exception as e:
            failed_runs += 1
            error_type = type(e).__name__
            print(f"✗ Error: {error_type}: {e}")
            error_details.append({"run": run_num, "error": error_type, "message": str(e)})
            # Try to categorize error
            if "openai" in str(e).lower() or "OpenAI" in error_type:
                openai_errors[error_type] += 1
            else:
                deepseek_errors[error_type] += 1
    
    if not openai_results or not deepseek_results:
        print("No successful runs!")
        return None
    
    # Calculate consistency within each model across runs
    openai_inter_similarities = []
    deepseek_inter_similarities = []
    for i in range(len(openai_results)):
        for j in range(i + 1, len(openai_results)):
            openai_pair_sim = evaluate_similarity_percentage(openai_results[i], openai_results[j])
            openai_inter_similarities.append(openai_pair_sim)
            
            deepseek_pair_sim = evaluate_similarity_percentage(deepseek_results[i], deepseek_results[j])
            deepseek_inter_similarities.append(deepseek_pair_sim)
    
    # Calculate overall similarity (average across all runs)
    overall_similarity = statistics.mean(similarities) if similarities else 0.0
    overall_openai_gt_similarity = statistics.mean(openai_ground_truth_similarities) if openai_ground_truth_similarities else None
    overall_deepseek_gt_similarity = statistics.mean(deepseek_ground_truth_similarities) if deepseek_ground_truth_similarities else None
    
    # Calculate detailed comparison metrics
    comparison_metrics = []
    for i in range(len(openai_results)):
        comparison = compare_results(openai_results[i], deepseek_results[i])
        comparison_metrics.append(comparison)
    
    # Aggregate comparison statistics
    exact_match_rate = sum(m["exact_match"] for m in comparison_metrics) / len(comparison_metrics) if comparison_metrics else 0.0
    classification_match_rate = sum(m["classification_match"] for m in comparison_metrics) / len(comparison_metrics) if comparison_metrics else 0.0
    avg_extraction_similarity = sum(m["extraction_similarity"] for m in comparison_metrics) / len(comparison_metrics) if comparison_metrics else 0.0
    
    # Timing statistics
    openai_timing_stats = {
        "mean": statistics.mean(openai_timings),
        "median": statistics.median(openai_timings),
        "stdev": statistics.stdev(openai_timings) if len(openai_timings) > 1 else 0.0,
        "min": min(openai_timings),
        "max": max(openai_timings),
    }
    
    deepseek_timing_stats = {
        "mean": statistics.mean(deepseek_timings),
        "median": statistics.median(deepseek_timings),
        "stdev": statistics.stdev(deepseek_timings) if len(deepseek_timings) > 1 else 0.0,
        "min": min(deepseek_timings),
        "max": max(deepseek_timings),
    }
    
    # Similarity statistics
    similarity_stats = {
        "mean": statistics.mean(similarities),
        "median": statistics.median(similarities),
        "stdev": statistics.stdev(similarities) if len(similarities) > 1 else 0.0,
        "min": min(similarities),
        "max": max(similarities),
    }
    
    # Model internal consistency
    openai_internal_consistency = statistics.mean(openai_inter_similarities) if openai_inter_similarities else 0.0
    deepseek_internal_consistency = statistics.mean(deepseek_inter_similarities) if deepseek_inter_similarities else 0.0
    
    # Reliability metrics
    total_attempts = successful_runs + failed_runs
    success_rate = (successful_runs / total_attempts * 100) if total_attempts > 0 else 0.0
    avg_openai_retries = statistics.mean(openai_retry_counts) if openai_retry_counts else 0.0
    avg_deepseek_retries = statistics.mean(deepseek_retry_counts) if deepseek_retry_counts else 0.0
    max_openai_retries = max(openai_retry_counts) if openai_retry_counts else 0
    max_deepseek_retries = max(deepseek_retry_counts) if deepseek_retry_counts else 0
    
    # Calculate coefficient of variation for timing (stability metric)
    openai_cv = (openai_timing_stats['stdev'] / openai_timing_stats['mean']) if openai_timing_stats['mean'] > 0 else 0.0
    deepseek_cv = (deepseek_timing_stats['stdev'] / deepseek_timing_stats['mean']) if deepseek_timing_stats['mean'] > 0 else 0.0
    
    # Print evaluation report
    print(f"\n{'='*80}")
    print("MODEL COMPARISON REPORT")
    print(f"{'='*80}")
    
    # Summary Scores
    print(f"\nSUMMARY SCORES:")
    print(f"  Model A vs Model B Similarity: {overall_similarity:.2f}%")
    print(f"    (Average similarity across {len(similarities)} successful runs)")
    
    if ground_truth and overall_openai_gt_similarity is not None and overall_deepseek_gt_similarity is not None:
        print(f"\n  Model A (OpenAI) vs Ground Truth: {overall_openai_gt_similarity:.2f}%")
        print(f"    (Average similarity across {len(openai_ground_truth_similarities)} successful runs)")
        print(f"  Model B (DeepSeek) vs Ground Truth: {overall_deepseek_gt_similarity:.2f}%")
        print(f"    (Average similarity across {len(deepseek_ground_truth_similarities)} successful runs)")
    
    # Reliability Metrics
    print(f"\nRELIABILITY METRICS:")
    print(f"  Success Rate:              {success_rate:.1f}% ({successful_runs}/{total_attempts} runs)")
    print(f"  Failed Runs:               {failed_runs}")
    print(f"  Total Successful API Calls - OpenAI: {openai_total_successful_calls}")
    print(f"  Total Successful API Calls - DeepSeek: {deepseek_total_successful_calls}")
    print(f"  Model A (OpenAI) Rate Limits: {openai_rate_limited_count}")
    print(f"  Model B (DeepSeek) Rate Limits: {deepseek_rate_limited_count}")
    print(f"  Average Retries - OpenAI:  {avg_openai_retries:.2f} (max: {max_openai_retries})")
    print(f"  Average Retries - DeepSeek: {avg_deepseek_retries:.2f} (max: {max_deepseek_retries})")
    
    # Show calls before rate limits
    if openai_successful_calls_before_rate_limit:
        calls_counts = [r["calls_before"] for r in openai_successful_calls_before_rate_limit]
        avg_calls_before = statistics.mean(calls_counts) if calls_counts else 0
        min_calls_before = min(calls_counts) if calls_counts else 0
        max_calls_before = max(calls_counts) if calls_counts else 0
        print(f"\n  OpenAI Rate Limit Analysis:")
        print(f"    Calls before rate limit: avg={avg_calls_before:.1f}, min={min_calls_before}, max={max_calls_before}")
        print(f"    Rate limits occurred at runs: {[r['run'] for r in openai_successful_calls_before_rate_limit]}")
    
    if deepseek_successful_calls_before_rate_limit:
        calls_counts = [r["calls_before"] for r in deepseek_successful_calls_before_rate_limit]
        avg_calls_before = statistics.mean(calls_counts) if calls_counts else 0
        min_calls_before = min(calls_counts) if calls_counts else 0
        max_calls_before = max(calls_counts) if calls_counts else 0
        print(f"\n  DeepSeek Rate Limit Analysis:")
        print(f"    Calls before rate limit: avg={avg_calls_before:.1f}, min={min_calls_before}, max={max_calls_before}")
        print(f"    Rate limits occurred at runs: {[r['run'] for r in deepseek_successful_calls_before_rate_limit]}")
    
    if openai_errors or deepseek_errors:
        print(f"\nERROR BREAKDOWN:")
        if openai_errors:
            print(f"  Model A (OpenAI) Errors:")
            for error_type, count in openai_errors.items():
                print(f"    {error_type}: {count}")
        if deepseek_errors:
            print(f"  Model B (DeepSeek) Errors:")
            for error_type, count in deepseek_errors.items():
                print(f"    {error_type}: {count}")
    
    print(f"\nPerformance Metrics (Timing):")
    print(f"  Model A (OpenAI):")
    print(f"    Mean:   {openai_timing_stats['mean']:.2f}s")
    print(f"    Median: {openai_timing_stats['median']:.2f}s")
    print(f"    StdDev: {openai_timing_stats['stdev']:.2f}s")
    print(f"    CV:     {openai_cv*100:.1f}% (lower = more stable)")
    print(f"    Min:    {openai_timing_stats['min']:.2f}s")
    print(f"    Max:    {openai_timing_stats['max']:.2f}s")
    print(f"  Model B (DeepSeek):")
    print(f"    Mean:   {deepseek_timing_stats['mean']:.2f}s")
    print(f"    Median: {deepseek_timing_stats['median']:.2f}s")
    print(f"    StdDev: {deepseek_timing_stats['stdev']:.2f}s")
    print(f"    CV:     {deepseek_cv*100:.1f}% (lower = more stable)")
    print(f"    Min:    {deepseek_timing_stats['min']:.2f}s")
    print(f"    Max:    {deepseek_timing_stats['max']:.2f}s")
    
    print(f"\nSimilarity Statistics (Model A vs Model B):")
    print(f"  Mean:   {similarity_stats['mean']:.2f}%")
    print(f"  Median: {similarity_stats['median']:.2f}%")
    print(f"  StdDev: {similarity_stats['stdev']:.2f}%")
    print(f"  Min:    {similarity_stats['min']:.2f}%")
    print(f"  Max:    {similarity_stats['max']:.2f}%")
    
    # Ground truth similarity statistics
    if ground_truth and openai_ground_truth_similarities and deepseek_ground_truth_similarities:
        openai_gt_stats = {
            "mean": statistics.mean(openai_ground_truth_similarities),
            "median": statistics.median(openai_ground_truth_similarities),
            "stdev": statistics.stdev(openai_ground_truth_similarities) if len(openai_ground_truth_similarities) > 1 else 0.0,
            "min": min(openai_ground_truth_similarities),
            "max": max(openai_ground_truth_similarities),
        }
        deepseek_gt_stats = {
            "mean": statistics.mean(deepseek_ground_truth_similarities),
            "median": statistics.median(deepseek_ground_truth_similarities),
            "stdev": statistics.stdev(deepseek_ground_truth_similarities) if len(deepseek_ground_truth_similarities) > 1 else 0.0,
            "min": min(deepseek_ground_truth_similarities),
            "max": max(deepseek_ground_truth_similarities),
        }
        
        print(f"\nGround Truth Similarity Statistics:")
        print(f"  Model A (OpenAI) vs Ground Truth:")
        print(f"    Mean:   {openai_gt_stats['mean']:.2f}%")
        print(f"    Median: {openai_gt_stats['median']:.2f}%")
        print(f"    StdDev: {openai_gt_stats['stdev']:.2f}%")
        print(f"    Min:    {openai_gt_stats['min']:.2f}%")
        print(f"    Max:    {openai_gt_stats['max']:.2f}%")
        print(f"  Model B (DeepSeek) vs Ground Truth:")
        print(f"    Mean:   {deepseek_gt_stats['mean']:.2f}%")
        print(f"    Median: {deepseek_gt_stats['median']:.2f}%")
        print(f"    StdDev: {deepseek_gt_stats['stdev']:.2f}%")
        print(f"    Min:    {deepseek_gt_stats['min']:.2f}%")
        print(f"    Max:    {deepseek_gt_stats['max']:.2f}%")
    else:
        openai_gt_stats = None
        deepseek_gt_stats = None
    
    print(f"\nInternal Consistency (within each model across runs):")
    print(f"  Model A (OpenAI):      {openai_internal_consistency:.2f}%")
    print(f"  Model B (DeepSeek):   {deepseek_internal_consistency:.2f}%")
    
    print(f"\nDetailed Comparison Metrics:")
    print(f"  Exact Match Rate:        {exact_match_rate*100:.1f}%")
    print(f"  Classification Match:    {classification_match_rate*100:.1f}%")
    print(f"  Avg Extraction Similarity: {avg_extraction_similarity*100:.1f}%")
    
    print(f"\n{'='*80}\n")
    
    # Prepare ground truth stats for return
    if ground_truth and openai_ground_truth_similarities and deepseek_ground_truth_similarities:
        openai_gt_stats_for_return = {
            "mean": statistics.mean(openai_ground_truth_similarities),
            "median": statistics.median(openai_ground_truth_similarities),
            "stdev": statistics.stdev(openai_ground_truth_similarities) if len(openai_ground_truth_similarities) > 1 else 0.0,
            "min": min(openai_ground_truth_similarities),
            "max": max(openai_ground_truth_similarities),
        }
        deepseek_gt_stats_for_return = {
            "mean": statistics.mean(deepseek_ground_truth_similarities),
            "median": statistics.median(deepseek_ground_truth_similarities),
            "stdev": statistics.stdev(deepseek_ground_truth_similarities) if len(deepseek_ground_truth_similarities) > 1 else 0.0,
            "min": min(deepseek_ground_truth_similarities),
            "max": max(deepseek_ground_truth_similarities),
        }
    else:
        openai_gt_stats_for_return = None
        deepseek_gt_stats_for_return = None
    
    return {
        "article_file": article_file,
        "num_runs": len(openai_results),
        "total_attempts": successful_runs + failed_runs,
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "success_rate": success_rate,
        "overall_similarity": overall_similarity,  # Model A vs Model B
        "overall_openai_gt_similarity": overall_openai_gt_similarity,  # Model A vs Ground Truth
        "overall_deepseek_gt_similarity": overall_deepseek_gt_similarity,  # Model B vs Ground Truth
        "openai_timing_stats": openai_timing_stats,
        "deepseek_timing_stats": deepseek_timing_stats,
        "similarity_stats": similarity_stats,  # Model A vs Model B stats
        "openai_ground_truth_stats": openai_gt_stats_for_return,
        "deepseek_ground_truth_stats": deepseek_gt_stats_for_return,
        "openai_internal_consistency": openai_internal_consistency,
        "deepseek_internal_consistency": deepseek_internal_consistency,
        "reliability_metrics": {
            "openai_rate_limited_count": openai_rate_limited_count,
            "deepseek_rate_limited_count": deepseek_rate_limited_count,
            "openai_total_successful_calls": openai_total_successful_calls,
            "deepseek_total_successful_calls": deepseek_total_successful_calls,
            "openai_successful_calls_before_rate_limit": openai_successful_calls_before_rate_limit,
            "deepseek_successful_calls_before_rate_limit": deepseek_successful_calls_before_rate_limit,
            "avg_openai_retries": avg_openai_retries,
            "avg_deepseek_retries": avg_deepseek_retries,
            "max_openai_retries": max_openai_retries,
            "max_deepseek_retries": max_deepseek_retries,
            "openai_errors": dict(openai_errors),
            "deepseek_errors": dict(deepseek_errors),
        },
        "comparison_metrics": {
            "exact_match_rate": exact_match_rate,
            "classification_match_rate": classification_match_rate,
            "avg_extraction_similarity": avg_extraction_similarity,
        },
    }

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main function to process all articles."""
    # Load environment variables
    load_dotenv()
    
    # Initialize clients
    openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    deepseek_client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com"
    )
    
    if EVALUATION_MODE:
        # Run model comparison evaluation
        evaluation_results = []
        for article_file in ARTICLE_FILES:
            eval_result = evaluate_model_comparison(
                openai_client, deepseek_client, article_file, num_runs=NUM_EVALUATION_RUNS
            )
            if eval_result:
                evaluation_results.append(eval_result)
        
        # Overall summary
        if evaluation_results:
            print(f"\n{'='*80}")
            print("OVERALL EVALUATION SUMMARY")
            print(f"{'='*80}")
            
            avg_openai_time = statistics.mean([r["openai_timing_stats"]["mean"] for r in evaluation_results])
            avg_deepseek_time = statistics.mean([r["deepseek_timing_stats"]["mean"] for r in evaluation_results])
            avg_similarity = statistics.mean([r["overall_similarity"] for r in evaluation_results])
            avg_exact_match = statistics.mean([r["comparison_metrics"]["exact_match_rate"] for r in evaluation_results])
            avg_classification_match = statistics.mean([r["comparison_metrics"]["classification_match_rate"] for r in evaluation_results])
            
            # Reliability summary
            total_attempts = sum(r["total_attempts"] for r in evaluation_results)
            total_successful = sum(r["successful_runs"] for r in evaluation_results)
            total_failed = sum(r["failed_runs"] for r in evaluation_results)
            overall_success_rate = (total_successful / total_attempts * 100) if total_attempts > 0 else 0.0
            total_openai_rate_limits = sum(r["reliability_metrics"]["openai_rate_limited_count"] for r in evaluation_results)
            total_deepseek_rate_limits = sum(r["reliability_metrics"]["deepseek_rate_limited_count"] for r in evaluation_results)
            avg_openai_retries = statistics.mean([r["reliability_metrics"]["avg_openai_retries"] for r in evaluation_results])
            avg_deepseek_retries = statistics.mean([r["reliability_metrics"]["avg_deepseek_retries"] for r in evaluation_results])
            
            print(f"\nAcross all articles:")
            print(f"  Average processing time (Model A - OpenAI):    {avg_openai_time:.2f}s")
            print(f"  Average processing time (Model B - DeepSeek): {avg_deepseek_time:.2f}s")
            print(f"  Average similarity (Model A vs Model B):      {avg_similarity:.2f}%")
            print(f"  Average exact match rate:                     {avg_exact_match*100:.1f}%")
            print(f"  Average classification match rate:            {avg_classification_match*100:.1f}%")
            
            print(f"\nOverall Reliability:")
            print(f"  Overall Success Rate:            {overall_success_rate:.1f}% ({total_successful}/{total_attempts})")
            print(f"  Total Failed Runs:                {total_failed}")
            total_openai_calls = sum(r["reliability_metrics"]["openai_total_successful_calls"] for r in evaluation_results)
            total_deepseek_calls = sum(r["reliability_metrics"]["deepseek_total_successful_calls"] for r in evaluation_results)
            print(f"  Total Successful API Calls - OpenAI: {total_openai_calls}")
            print(f"  Total Successful API Calls - DeepSeek: {total_deepseek_calls}")
            print(f"  Total Rate Limit Events - OpenAI: {total_openai_rate_limits}")
            print(f"  Total Rate Limit Events - DeepSeek: {total_deepseek_rate_limits}")
            print(f"  Average Retries - OpenAI:         {avg_openai_retries:.2f}")
            print(f"  Average Retries - DeepSeek:       {avg_deepseek_retries:.2f}")
            
            # Aggregate rate limit analysis
            all_openai_calls_before = []
            all_deepseek_calls_before = []
            for r in evaluation_results:
                all_openai_calls_before.extend([x["calls_before"] for x in r["reliability_metrics"].get("openai_successful_calls_before_rate_limit", [])])
                all_deepseek_calls_before.extend([x["calls_before"] for x in r["reliability_metrics"].get("deepseek_successful_calls_before_rate_limit", [])])
            
            if all_openai_calls_before:
                avg_calls = statistics.mean(all_openai_calls_before)
                print(f"\n  OpenAI Rate Limit Pattern:")
                print(f"    Average calls before rate limit: {avg_calls:.1f}")
                print(f"    Range: {min(all_openai_calls_before)} - {max(all_openai_calls_before)} calls")
            
            if all_deepseek_calls_before:
                avg_calls = statistics.mean(all_deepseek_calls_before)
                print(f"\n  DeepSeek Rate Limit Pattern:")
                print(f"    Average calls before rate limit: {avg_calls:.1f}")
                print(f"    Range: {min(all_deepseek_calls_before)} - {max(all_deepseek_calls_before)} calls")
            
            print(f"\n{'='*80}\n")
    else:
        # Single run mode - compare both models
        results = []
        for article_file in ARTICLE_FILES:
            openai_result, deepseek_result, similarity, openai_time, deepseek_time, openai_error_info, deepseek_error_info = process_article_pair(
                openai_client, deepseek_client, article_file
            )
            results.append({
                "file": article_file,
                "openai_result": openai_result,
                "deepseek_result": deepseek_result,
                "similarity": similarity,
                "openai_time": openai_time,
                "deepseek_time": deepseek_time,
                "openai_retries": openai_error_info["retries"],
                "deepseek_retries": deepseek_error_info["retries"]
            })
        
        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        for r in results:
            print(f"\n{r['file']}:")
            print(f"  Model A (OpenAI) - Topic: {r['openai_result']['classification']['topic']}, "
                  f"Sentiment: {r['openai_result']['classification']['sentiment']}, "
                  f"Urgency: {r['openai_result']['classification']['urgency']} ({r['openai_time']:.2f}s)")
            print(f"  Model B (DeepSeek) - Topic: {r['deepseek_result']['classification']['topic']}, "
                  f"Sentiment: {r['deepseek_result']['classification']['sentiment']}, "
                  f"Urgency: {r['deepseek_result']['classification']['urgency']} ({r['deepseek_time']:.2f}s)")
            print(f"  Similarity: {r['similarity']}%")
        print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()