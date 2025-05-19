from __future__ import annotations
from autogen import ConversableAgent, register_function
import os, sys, re, ast, textwrap
from typing import Dict, List, get_type_hints

SCORE_KEYWORDS: dict[int, list[str]] = {
    1: ["awful", "horrible", "disgusting"],
    2: ["bad", "unpleasant", "offensive"],
    # 3: ["average", "okay", "uninspiring", "forgettable"],
    # 4: ["good", "nice", "great", "enjoyable", "satisfying", "delightful", "pleasant"],
    # 5: ["awesome", "incredible", "amazing", "fantastic", "blew my mind", "outstanding", "exceptional", "superb"],
    3: ["average", "uninspiring", "forgettable"],
    4: ["good", "enjoyable", "satisfying"],
    5: ["awesome", "incredible", "amazing"]
}

# ────────────────────────────────────────────────────────────────
# 0. OpenAI API key setup ── *Do **not** modify this block.*
# ────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    sys.exit("❗ Set the OPENAI_API_KEY environment variable first.")
LLM_CFG = {"config_list": [{"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}]}

# ────────────────────────────────────────────────────────────────
# 1. Utility data structures & helper functions
# ────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()

def fetch_restaurant_data(restaurant_name: str) -> dict[str, list[str]]:
    data = {}
    target = normalize(restaurant_name)
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            name, review = line.split('.', 1)
            if normalize(name) == target:
                data.setdefault(name.strip(), []).append(review.strip())
    return data


def calculate_overall_score(restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]) -> dict[str, str]:
    """Geometric-mean rating rounded to 3 dp."""
    n = len(food_scores)
    if n == 0 or n != len(customer_service_scores):
        raise ValueError("food_scores and customer_service_scores must be non-empty and same length")
    total = sum(((f**2 * s)**0.5) * (1 / (n * (125**0.5))) * 10 for f, s in zip(food_scores, customer_service_scores))
    return {restaurant_name: f"{total:.3f}"}

# register functions
fetch_restaurant_data.__annotations__ = get_type_hints(fetch_restaurant_data)
calculate_overall_score.__annotations__ = get_type_hints(calculate_overall_score)

# ──────────────────────────────────────────────
# 2. Agent setup
# ──────────────────────────────────────────────

def build_agent(name, msg):
    return ConversableAgent(name=name, system_message=msg, llm_config=LLM_CFG)

DATA_FETCH = build_agent(
    "fetch_agent",
    'Return JSON {"call":"fetch_restaurant_data","args":{"restaurant_name":"<name>"}}'
)

ADJ_EXTRACTOR = build_agent(
    "adjective_extractor_agent",
    textwrap.dedent(f"""\
    Input format is:
    Review 1: ...(text)... [END]
    Review 2: ...(text)... [END]
    ...
    Review n: ...(text)... [END]

    Please extract 'one adjective' that describe food and 'one adjective' that describe customer for each review.
    Consider context, negation, sarcasm, and subtle sentiment indicators. Focus on the overall tone and intent of the sentence rather than specific keywords. Some reviews may use humor, exaggeration, or idioms - interpret them appropriately.
    These known keywords could be extracted first:
    {SCORE_KEYWORDS}
                    
    Output format:
    {{
        1: {{"food": "average", "service": "incredible", "review": "..."}},
        2: {{"food": ..., "service": ..., "review": "..."}},
        ...
        n: {{"food": ..., "service": ..., "review": "..."}}
    }}

    If a review has no adjectives for food or service, return empty lists for them.
    Only return valid Python dict, no explanation, no Markdown.
""")
)

ANALYZER = build_agent(
    "review_analyzer_agent",
    textwrap.dedent(f"""\
    Input is :
    {{
        1: {{"food": ["average"], "service": ["incredible"], "review": "..."}},
        2: {{"food": [...], "service": [...], "review": "..."}},
        ...
        n: {{"food": ["..."], "service": ["..."], "review": "..."}}
    }}

    There are two extracted adjectives from each review: one for food and one for service.
    Please score from 1 to 5 based on how positively or negatively the food or service is described.
    Scoring guide:
    {SCORE_KEYWORDS}
    Do not rely solely on exact keywords. Consider context, negation, sarcasm, and subtle sentiment indicators. Focus on the overall tone and intent of the sentence rather than specific keywords. Some reviews may use humor, exaggeration, or idioms - interpret them appropriately.
    It is quite possible that the adjectives are not in the list. Please use your judgment to determine its meaning and score it.

    If the word in the review isn't exactly listed, estimate the score by semantic similarity:
    - "terrible" → score 1 (similar to "awful")
    - "not great" → score 2 (similar to "bad")
    - "mediocre" → score 3 (similar to "average")
    - "delightful" → score 4 (similar to "enjoyable")
    - "fantastic" → score 5 (similar to "amazing")
    
    General rules:
    - If the description is strongly negative → score 1
    - If it is negative but not superlative → score 2
    - If it is neutral or mixed sentiment → score 3
    - If it is clearly positive but not superlative → score 4
    - If it is extremely positive → score 5
    - Ignore irrelevant or off-topic adjectives

    Output should be :
    {{
        "food_scores": {{
            1: 5,
            2: 4,
            ...
        }},
        "customer_service_scores": {{
            1: 4,
            2: 4,
            ...
        }}
    }}

    Example:
    {{"food": "not great", "service": "lovely"}}
    food_scores = 2 (double negative), customer_service_scores = 4 (synonyms)

    Use your judgment to map each review to a score from 1 (worst) to 5 (best).
    Only return valid Python dict, no explanation, no Markdown.
""")
)
SCORER = build_agent(
    "scoring_agent",
    "Given name + two lists. Reply only: the final returned score of the function: 'calculate_overall_score(...)'."
)
ENTRY = build_agent("entry", "Coordinator")

# register functions
register_function(
    fetch_restaurant_data,
    caller=DATA_FETCH,
    executor=ENTRY,
    name="fetch_restaurant_data",
    description="Fetch reviews from specified data file by name.",
)
register_function(
    calculate_overall_score,
    caller=SCORER,
    executor=ENTRY,
    name="calculate_overall_score",
    description="Compute final rating via geometric mean.",
)

# ────────────────────────────────────────────────────────────────
# 3. Conversation helpers
# ────────────────────────────────────────────────────────────────

def format_reviews_for_prompt(reviews_dict: dict[str, list[str]]) -> str:
    if not reviews_dict:
        return "No reviews found."

    formatted = []
    for restaurant, reviews in reviews_dict.items():
        formatted.append(f"Restaurant: {restaurant}\n")
        for i, review in enumerate(reviews, 1):
            clean_review = review.replace('"', '')
            formatted.append(f"Review {i}: {clean_review} [END]")
        formatted.append("")

    return "\n".join(formatted)

def collect_values_from_dict(d: dict) -> list:
    return [value for value in d.values()]

def run_chat_sequence(entry: ConversableAgent, sequence: list[dict]) -> str:
    ctx = {**getattr(entry, "_initiate_chats_ctx", {})}
    for step in sequence:
        msg = step["message"].format(**ctx)
        chat = entry.initiate_chat(
            step["recipient"], message=msg,
            summary_method=step.get("summary_method", "last_msg"),
            max_turns=step.get("max_turns", 2),
        )
        out = chat.summary
        # Data fetch output
        if step["recipient"] is DATA_FETCH:
            for past in reversed(chat.chat_history):
                try:
                    data = ast.literal_eval(past["content"])
                    if isinstance(data, dict) and data and not ("call" in data):
                        ctx.update({"reviews_dict": data, "restaurant_name": next(iter(data)), "formatted_reviews": format_reviews_for_prompt(data)})
                        break
                except:
                    continue
        elif step["recipient"] is ADJ_EXTRACTOR:
            ctx["adj_extractor_output"] = out
            out_dicts = ast.literal_eval(out)
            ctx["adjectives_and_reviews"] = out_dicts
        # Analyzer output passed directly
        elif step["recipient"] is ANALYZER:
            ctx["analyzer_output"] = out
            out_dicts = ast.literal_eval(out)
            ctx["food_scores"] = collect_values_from_dict(out_dicts["food_scores"])
            ctx["customer_service_scores"] = collect_values_from_dict(out_dicts["customer_service_scores"])
    return out

ConversableAgent.initiate_chats = lambda self, seq: run_chat_sequence(self, seq)

# ──────────────────────────────────────────────
# 4. Main entry
# ──────────────────────────────────────────────

def main(user_query: str, data_path: str = "restaurant-data.txt"):
    global DATA_PATH
    DATA_PATH = data_path
    agents = {"data_fetch": DATA_FETCH, "adj_extractor": ADJ_EXTRACTOR, "analyzer": ANALYZER, "scorer": SCORER}
    chat_sequence = [
        {"recipient": agents["data_fetch"], 
         "message": "Find reviews for this query: {user_query}", 
         "summary_method": "last_msg", 
         "max_turns": 2},

        {"recipient": agents["adj_extractor"],
         "message": "Here are the reviews:\n{formatted_reviews}\n\nExtract adjectives that describe food and service respectively.", 
         "summary_method": "last_msg", 
         "max_turns": 1},

        {"recipient": agents["analyzer"], 
         "message": "Here are the extracted adjectives from the reviews:\n{adjectives_and_reviews}\n\nExtract exactly one food score and one service score for each review.", 
         "summary_method": "last_msg", 
         "max_turns": 1},

        {"recipient": agents["scorer"], 
         "message": "restaurant_name : {restaurant_name}, analyzer_output :\nfood_scores : {food_scores}\ncustomer_service_scores : {customer_service_scores}", 
         "summary_method": "last_msg", 
         "max_turns": 2},
    ]
    ENTRY._initiate_chats_ctx = {"user_query": user_query}
    result = ENTRY.initiate_chats(chat_sequence)
    print(f"result: {result}")
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python main.py path/to/data.txt "How good is Subway?" ')
        sys.exit(1)

    path = sys.argv[1]
    query = sys.argv[2]
    main(query, path)