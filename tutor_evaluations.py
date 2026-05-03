import re
import math
from difflib import SequenceMatcher
from typing import Dict, Any, List, Tuple

# -----------------------------
# Helpers: normalization + matching
# -----------------------------
def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s

def fuzzy_ratio(a: str, b: str) -> float:
    a, b = normalize_text(a), normalize_text(b)
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()

def extract_number(text: str):
    """
    Pull the first reasonable numeric token from text.
    Supports: 35, 35.5, -1.2, 1e-3, 35–36 (returns first), 35-36 (returns first)
    """
    if not text:
        return None
    t = text.replace("–", "-")  # normalize en-dash
    m = re.search(r"([-+]?\d+(\.\d+)?([eE][-+]?\d+)?)", t)
    if not m:
        return None
    try:
        return float(m.group(1))
    except:
        return None

def contains_any(text: str, keywords: List[str]) -> bool:
    t = normalize_text(text)
    return any(normalize_text(k) in t for k in keywords)

# -----------------------------
# Scoring dimensions
# -----------------------------
def score_correctness(q: Dict[str, Any], answer: str) -> Tuple[float, str]:
    """
    Returns (0..1, rationale)
    - For numeric: compare to expected numeric if possible (or accept ranges if specified)
    - For true/false: exact match on true/false
    - For MC: checks expected token
    - For conceptual: uses fuzzy match + key phrases
    """
    qtype = q.get("type", "").lower()
    expected = q.get("expected_answer", "")

    ans_norm = normalize_text(answer)
    exp_norm = normalize_text(expected)

    # If expected says "depends", we can't do correctness automatically
    if "depends" in exp_norm:
        return 0.5, "Expected answer is context-dependent; correctness requires human review."

    # True/False — search anywhere in the answer, not just the first character
    if qtype in ["true_false", "tf", "boolean"]:
        has_true  = bool(re.search(r"\btrue\b",  ans_norm))
        has_false = bool(re.search(r"\bfalse\b", ans_norm))
        if has_true and not has_false:
            ans_val = "true"
        elif has_false and not has_true:
            ans_val = "false"
        elif has_false and has_true:
            # Both words present (e.g. "X is false, Y is true") — take the one
            # that appears as the final verdict word nearest "is" or "statement is"
            last_true  = ans_norm.rfind("true")
            last_false = ans_norm.rfind("false")
            ans_val = "true" if last_true > last_false else "false"
        else:
            return 0.0, "Could not find 'true' or 'false' in answer."
        if ans_val == exp_norm:
            return 1.0, "Correct True/False."
        return 0.0, f"Incorrect True/False: answer contains '{ans_val}', expected '{exp_norm}'."

    # Numeric — also handle binary strings and hex values
    if qtype == "numeric":
        exp_stripped = re.sub(r"[\s]", "", expected)   # remove spaces from expected
        ans_stripped = re.sub(r"[\s]", "", answer)     # remove spaces from answer

        # Binary string match (e.g. "00011111" vs "0001 1111" or "11111")
        if re.fullmatch(r"[01]{4,}", exp_stripped):
            if exp_stripped in ans_stripped:
                return 1.0, "Binary string matched after stripping spaces."
            # Also accept if any binary token in the answer has the same numeric value
            # (handles missing leading zeros, e.g. "11111" == "00011111")
            exp_val = int(exp_stripped, 2)
            for token in re.findall(r"\b[01]{4,}\b", ans_stripped):
                if int(token, 2) == exp_val:
                    return 1.0, f"Binary value matches (leading zeros differ): '{token}' == '{exp_stripped}'."
            return 0.0, f"Binary string '{exp_stripped}' not found in answer."

        # Hex match — must run before extract_number which would misparse "0x1F" as 0
        exp_hex = re.search(r"\b0x([0-9a-f]+)\b", expected, re.IGNORECASE)
        if exp_hex:
            target = exp_hex.group(1)
            # Accept with or without 0x prefix in answer
            ans_hex = re.search(r"(?:0x)?" + re.escape(target) + r"\b", answer, re.IGNORECASE)
            if ans_hex:
                return 1.0, "Hex value matched."
            return 0.0, f"Hex value '{target}' not found in answer."

        exp_num = extract_number(expected)
        ans_num = extract_number(answer)
        if exp_num is None or ans_num is None:
            r = fuzzy_ratio(answer, expected)
            return (1.0 if r >= 0.85 else 0.5 if r >= 0.65 else 0.0), "Numeric parse failed; used fuzzy match."
        rel_tol = q.get("rel_tol", 0.05)
        abs_tol = q.get("abs_tol", 0.05)
        diff = abs(ans_num - exp_num)
        ok = diff <= max(abs_tol, rel_tol * abs(exp_num))
        return (1.0 if ok else 0.0), f"Parsed numeric answer={ans_num}, expected={exp_num}, diff={diff}."

    # Multiple-choice (simple)
    if qtype in ["multiple_choice", "mc"]:
        # For single-letter expected answers (A/B/C/D), use whole-word matching only.
        # Substring match (e.g. "a" in "launch and...") gives false positives.
        if len(exp_norm) == 1 and exp_norm in "abcd":
            # Find all standalone MC letters mentioned in the answer
            letters_mentioned = set(re.findall(r"\b([abcd])\b", ans_norm))
            if exp_norm in letters_mentioned and len(letters_mentioned) == 1:
                return 1.0, f"Matched expected MC letter '{exp_norm}' (only letter mentioned)."
            if exp_norm in letters_mentioned and len(letters_mentioned) > 1:
                return 0.3, f"Expected letter '{exp_norm}' present but answer also mentions {letters_mentioned - {exp_norm}}."
            return 0.0, f"Expected MC letter '{exp_norm}' not found as standalone letter in answer."
        # For multi-word expected answers (e.g. "Secondary"), require whole-phrase match
        if exp_norm in ans_norm:
            return 1.0, "Expected option/label appears in answer."
        # Fallback: letter extraction
        m = re.search(r"\b([abcd])\b", ans_norm)
        if m and m.group(1) == exp_norm:
            return 1.0, "Matched expected MC letter."
        return 0.0, "Did not match expected multiple-choice answer."

    # Matching / conceptual / code_reasoning: fuzzy + keyphrase heuristic
    # If the question provides keyphrases, use them.
    keyphrases = q.get("keyphrases", [])
    if keyphrases:
        hit_rate = sum(1 for kp in keyphrases if normalize_text(kp) in ans_norm) / max(1, len(keyphrases))
        # combine hit_rate with fuzzy
        r = fuzzy_ratio(answer, expected)
        score = max(0.0, min(1.0, 0.6 * hit_rate + 0.4 * r))
        return score, f"Keyphrase hit rate={hit_rate:.2f}, fuzzy={r:.2f}."
    else:
        r = fuzzy_ratio(answer, expected)
        return (1.0 if r >= 0.85 else 0.7 if r >= 0.70 else 0.4 if r >= 0.55 else 0.0), f"Fuzzy similarity={r:.2f}."

def score_reasoning(answer: str) -> Tuple[float, str]:
    """
    Rough proxy: does the answer show steps?
    Looks for step markers, equations, or structured explanation.
    """
    t = normalize_text(answer)
    indicators = [
        "step", "first", "second", "third", "because", "therefore", "so ",
        "=", "->", "thus", "for example", "we know", "calculate", "compute"
    ]
    hits = sum(1 for ind in indicators if ind in t)
    # cap at 1.0
    score = min(1.0, hits / 6.0)
    return score, f"Reasoning indicators hit={hits}."

def score_pedagogy(answer: str) -> Tuple[float, str]:
    """
    Tutor-like behavior:
    - asks guiding questions or suggests what to check
    - offers an approach rather than just final answer
    """
    t = normalize_text(answer)
    pedagogy_signals = [
        "try", "you can", "look for", "check", "hint", "remember", "think about",
        "a good way", "start by", "common mistake", "walk through", "let's"
    ]
    hits = sum(1 for s in pedagogy_signals if s in t)
    score = min(1.0, hits / 5.0)
    return score, f"Pedagogy signals hit={hits}."

def score_grounding(answer: str) -> Tuple[float, str]:
    """
    Did they cite course materials? Since your course LLM should point to lecture/slide/page.
    Heuristic: look for tokens like 'Lecture', 'Slide', 'Page', 'Quiz', or patterns like 'L#', 'S#'.
    """
    t = answer or ""
    patterns = [
        r"\blecture\b", r"\bslides?\b", r"\bpage\b", r"\bsection\b", r"\bmodule\b",
        r"\bquiz\b", r"\bweek\b", r"\blesson\b",
        r"\bslide\s*\d+\b", r"\bpage\s*\d+\b", r"\blecture\s*\d+\b"
    ]
    hits = sum(1 for p in patterns if re.search(p, t, flags=re.IGNORECASE))
    # Require at least 1 explicit reference to get non-zero
    score = min(1.0, hits / 2.0)  # 0.5 for one hit, 1.0 for 2+
    return score, f"Grounding reference hits={hits}."

# -----------------------------
# Main evaluator
# -----------------------------
def evaluate_answer(q: Dict[str, Any], answer: str) -> Dict[str, Any]:
    corr, corr_note = score_correctness(q, answer)
    reas, reas_note = score_reasoning(answer)
    peda, peda_note = score_pedagogy(answer)
    grnd, grnd_note = score_grounding(answer)

    # Weighted overall (tune as you like)
    weights = q.get("weights", {"correctness": 0.45, "reasoning": 0.25, "pedagogy": 0.20, "grounding": 0.10})
    overall = (
        weights["correctness"] * corr +
        weights["reasoning"] * reas +
        weights["pedagogy"] * peda +
        weights["grounding"] * grnd
    )

    return {
        "id": q.get("id"),
        "topic": q.get("topic"),
        "type": q.get("type"),
        "question": q.get("question"),
        "expected_answer": q.get("expected_answer"),
        "llm_answer": answer,
        "scores": {
            "correctness": round(corr, 3),
            "reasoning": round(reas, 3),
            "pedagogy": round(peda, 3),
            "grounding": round(grnd, 3),
            "overall": round(overall, 3),
        },
        "notes": {
            "correctness": corr_note,
            "reasoning": reas_note,
            "pedagogy": peda_note,
            "grounding": grnd_note,
        }
    }

def evaluate_batch(agent, questions):
    """
    agent: your RAG agent (rag_agent)
    questions: list of question dictionaries
    """

    results = []

    for q in questions:

        question_text = q["question"]

        # Call the agent the same way you normally do
        agent_result = agent.invoke({
            "messages": [("human", question_text)]
        })

        # Extract final response text
        llm_answer = agent_result["messages"][-1].content

        # Score the answer
        graded = evaluate_answer(q, llm_answer)

        results.append({
            "id": q["id"],
            "topic": q["topic"],
            "question": question_text,
            "expected_answer": q["expected_answer"],
            "llm_answer": llm_answer,
            "scores": graded["scores"],
            "notes": graded["notes"]
        })

    return results


# -----------------------------
# Display helper
# -----------------------------
def display_results(results: List[Dict[str, Any]]) -> None:
    """
    Print a clean summary of evaluation results showing:
    - Question asked
    - Expected answer
    - LLM's actual response
    - Whether the answer is considered correct (based on correctness score >= 0.7)
    """
    divider = "=" * 70
    thin_divider = "-" * 70

    correct_count = 0

    for r in results:
        is_correct = r["scores"]["correctness"] >= 0.7
        correct_label = "✅ CORRECT" if is_correct else "❌ INCORRECT"
        if is_correct:
            correct_count += 1

        print(divider)
        print(f"[{r['id']}] {r['topic']}")
        print(thin_divider)
        print(f"Question:        {r['question']}")
        print(f"Expected Answer: {r['expected_answer']}")
        print(f"LLM Response:    {r['llm_answer']}")
        print(f"Result:          {correct_label}  (correctness score: {r['scores']['correctness']:.2f})")
        print()

    print(divider)
    total = len(results)
    print(f"SUMMARY: {correct_count}/{total} correct ({100 * correct_count / total:.0f}%)")
    print(divider)