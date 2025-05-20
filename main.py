from __future__ import annotations
from autogen import ConversableAgent, register_function
import os, sys, re, ast, textwrap
from typing import Dict, List, get_type_hints

SCORE_KEYWORDS: dict[int, list[str]] = {
    1: ["awful", "horrible", "disgusting"],
    2: ["bad", "unpleasant", "offensive"],
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

# def score_reviews(adjectives: dict[int, dict[str, str]], score_keywords: dict[int, list[str]]) ->dict[str, dict[str, int]]:
#     food_scores = []
#     customer_service_scores = []
#     for i, adj in adjectives.items():
#         food_adj = adj["food"]
#         service_adj = adj["service"]
#         food_score = next((score for score, words in score_keywords.items() if food_adj in words), 0)
#         service_score = next((score for score, words in score_keywords.items() if service_adj in words), 0)
#         food_scores.append(food_score)
#         customer_service_scores.append(service_score)
#     return {"food_scores": food_scores, "customer_service_scores": customer_service_scores}

def calculate_overall_score(restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]) -> dict[str, str]:
    """Geometric-mean rating rounded to 3 dp."""
    n = len(food_scores)
    if n == 0 or n != len(customer_service_scores):
        raise ValueError("food_scores and customer_service_scores must be non-empty and same length")
    total = sum(((f**2 * s)**0.5) * (1 / (n * (125**0.5))) * 10 for f, s in zip(food_scores, customer_service_scores))
    return {restaurant_name: f"{total:.3f}"}

# register functions
fetch_restaurant_data.__annotations__ = get_type_hints(fetch_restaurant_data)
# score_reviews.__annotations__ = get_type_hints(score_reviews)
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

    Please extract 'one expression' that describe the food and 'one expression' that describe the customer service for each review.
    These known keywords could be extracted first:
    {SCORE_KEYWORDS}
    If there are no specific known keywords, consider context, negation, sarcasm, and subtle sentiment indicators. Some reviews may use humor, exaggeration, or idioms.
                    
    Output format:
    {{
        1: {{"food": "average", "service": "incredibly friendly and efficient"}},
        2: {{"food": "...", "service": "..."}},
        ...
        n: {{"food": "...", "service": "..."}}
    }}

    If a review has no adjectives for food or service, return empty lists for them.
    Only return valid Python dict, no explanation, no Markdown.
""")
)

CLASSIFIER = build_agent(
    "classifier_agent",
    textwrap.dedent(f"""\
    Input is :
    ["adjective1", "adjective2", ...]
    
    Below is the dict of already known keywords and their scores:                
    SCORE_KEYWORDS:
    {SCORE_KEYWORDS}

    The input adjectivess scores are not in the dict, please determine a score for it from 1 to 5 using your judgment. 

    The general rules for estimating scores for unknown adjectives:
    - If the adjective is strongly negative → score 1
    - If it is negative but not superlative → score 2
    - If it is neutral or mixed sentiment → score 3
    - If it is clearly positive but not superlative → score 4
    - If it is extremely positive → score 5
    - Ignore irrelevant or off-topic adjectives
    - You may need to notice that some adverbs may strengthen or weaken the meaning of an adjective. For example, "above average" are positive expression,

    Output format (updated SCORE_KEYWORDS) (Please keep the original adjectives and their scores):
    {{
        "1": [...],
        "2": [...],
        ...
        "5": [...]
    }}

    Only return a valid Python dict, no explanation, no Markdown.
""")
)

ANALYZER = build_agent(
    "review_analyzer_agent",
    textwrap.dedent(f"""\
    Input is :
    adjectives:
    {{
        1: {{"food": "average", "service": "incredible"}},
        2: {{"food": ..., "service": ...}},
        ...
        n: {{"food": "...", "service": "..."}}
    }}
    SCORE_KEYWORDS:
    {{
        1: [...],
        2: [...],
        3: [...],
        4: [...],
        5: [...]
    }}

    Please assign scores to each review's food and service using the SCORE_KEYWORDS.

    Output format:
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

    Only return a valid Python dict. No explanation, no Markdown.
""")
)

# ANALYZER = build_agent(
#     "analyzer_agent",
#     "Given adjectives and score_keywords. Reply only: the final returned dict of the function: 'score_reviews(...)'."
# )

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

# register_function(
#     score_reviews,
#     caller=ANALYZER,
#     executor=ENTRY,
#     name="score_reviews",
#     description="Score reviews based on adjectives and keywords.",
# )

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

def collect_adjectives(d: dict) -> list:
    excluded_words = set()
    for word_list in SCORE_KEYWORDS.values():
        excluded_words.update(word_list)

    unique_adjectives = set()
    for entry in d.values():
        for adj in entry.values():
            if adj not in excluded_words:
                unique_adjectives.add(adj)
    return list(unique_adjectives)

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
            ctx["extracted_adjectives"] = ast.literal_eval(out)
            ctx["new_adjectives"] = collect_adjectives(ctx["extracted_adjectives"])
        elif step["recipient"] is CLASSIFIER:
            ctx["updated_score_keywords"] = ast.literal_eval(out)
        elif step["recipient"] is ANALYZER:
            out_dicts = ast.literal_eval(out)
            ctx["food_scores"] = out_dicts["food_scores"]
            ctx["customer_service_scores"] = out_dicts["customer_service_scores"]
    return out

ConversableAgent.initiate_chats = lambda self, seq: run_chat_sequence(self, seq)

# ──────────────────────────────────────────────
# 4. Main entry
# ──────────────────────────────────────────────

def main(user_query: str, data_path: str = "restaurant-data.txt"):
    global DATA_PATH
    DATA_PATH = data_path
    agents = {"data_fetch": DATA_FETCH, "adj_extractor": ADJ_EXTRACTOR, "classifier": CLASSIFIER, "analyzer": ANALYZER, "scorer": SCORER}
    chat_sequence = [
        {"recipient": agents["data_fetch"], 
         "message": "Find reviews for this query: {user_query}", 
         "summary_method": "last_msg", 
         "max_turns": 2},

        {"recipient": agents["adj_extractor"],
         "message": "Here are the reviews:\n{formatted_reviews}\n\nExtract adjectives that describe food and service respectively.", 
         "summary_method": "last_msg", 
         "max_turns": 1},

        {"recipient": agents["classifier"],
        "message": "Here are the new extracted adjectives from the reviews:\n{new_adjectives}\n\nUpdate the SCORE_KEYWORDS list with new adjectives and their scores.",
        "summary_method": "last_msg",
        "max_turns": 1},

        {"recipient": agents["analyzer"], 
        #  "message": "adjectives: {extracted_adjectives}\nscore_keywords: {updated_score_keywords}",
         "message": "Here are some adjectives and a SCORE_KEYWORDS list.\nadjectives: {extracted_adjectives}\nSCORE_KEYWORDS: {updated_score_keywords}\n\nPlease assign scores to each review's food and service using the SCORE_KEYWORDS.",
         "summary_method": "last_msg", 
         "max_turns": 1},

        {"recipient": agents["scorer"], 
         "message": "restaurant_name : {restaurant_name}\nfood_scores : {food_scores}\ncustomer_service_scores : {customer_service_scores}", 
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