"""Investigation assistant: "Why was transaction 3487632 flagged?"

Safety rule: the numbers (risk score, decision, factors) are written by our code, never by the LLM.
The LLM (optional) only writes a short plain-English explanation from those facts.
No API key? We use a template, so the assistant always works.

LLM backend: Groq (qwen/qwen3-27b).  Set GROQ_API_KEY to enable.
"""
import os
import re

GROQ_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3-27b")

SYSTEM = ("You help fraud analysts. Explain in 2-3 short sentences why a transaction was flagged. "
          "Use ONLY the facts you are given. Do not invent numbers, names or reasons. "
          "Factor values are z-scores (how far from the typical value). Mention that this is a model "
          "signal, not proof of fraud.")


def find_transaction_id(question):
    """Take the transaction id from the question (ex: 'why was #3487632 flagged?')."""
    numbers = re.findall(r"\d{4,}", question)
    return int(numbers[0]) if numbers else None


def value_phrase(factor):
    """Turn a z-score into a short phrase (very high / high / typical / low / missing)."""
    z = factor["value_z"]
    if z is None:
        return "missing"
    if z >= 2:
        return "very high"
    if z >= 1:
        return "high"
    if z <= -2:
        return "very low"
    if z <= -1:
        return "low"
    return "typical value"


def factors_text(result):
    """Numbered list of the main risk factors (only the ones that RAISE the risk)."""
    risky = [f for f in result["factors"] if f["shap"] > 0] or result["factors"]
    return "\n".join(f"{i}. {f['description']} - {value_phrase(f)}" for i, f in enumerate(risky, 1))


def template_explanation(result):
    """Fallback explanation when no API key is set or the LLM call fails."""
    risky = [f for f in result["factors"] if f["shap"] > 0][:3]
    if result["decision"] == "APPROVE":
        return "The model sees no strong fraud signals for this transaction."
    reasons = "; ".join(f"{f['description']} ({value_phrase(f)})" for f in risky)
    return f"The model raised the risk mainly because of: {reasons}. This is a model signal, not proof of fraud."


def llm_explanation(result, question):
    """Ask Groq for a short plain-English explanation.

    Uses streaming internally and collects the full response before returning.
    Returns None if GROQ_API_KEY is not set or the call fails, so the caller
    can fall back to template_explanation().
    """
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None

    facts = (
        f"Transaction {result['txn_id']}\n"
        f"Risk score: {result['fraud_prob']:.2f}\n"
        f"Decision: {result['decision']}\n"
        "Main factors:\n"
        + "\n".join(
            f"- {f['description']} (effect: {f['effect']}, z={f['value_z']})"
            for f in result["factors"]
        )
    )

    try:
        from groq import Groq

        client = Groq(api_key=key, timeout=20)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Question: {question}\n\nFacts:\n{facts}"},
            ],
            temperature=0.6,
            max_completion_tokens=300,
            top_p=0.95,
            reasoning_effort="default",
            stream=True,
            stop=None,
        )

        # Collect streamed chunks into a single string
        parts = []
        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta:
                parts.append(delta)
        return "".join(parts).strip() or None

    except Exception:
        return None


def answer(question, result):
    """Final text for the analyst.

    Returns (text, source) where source is 'llm' or 'template' so the caller
    can track which path was used (e.g. for Prometheus metrics).
    """
    flagged = result["decision"] != "APPROVE"
    head = f"Transaction #{result['txn_id']} was " + ("flagged as high risk." if flagged else "approved (low risk).")
    llm_text   = llm_explanation(result, question)
    source     = "llm" if llm_text else "template"
    explanation = llm_text or template_explanation(result)
    body = f"{head}\n\n{explanation}\n"
    if flagged:
        body += f"\nMain factors:\n{factors_text(result)}\n"
    text = body + f"\nRisk Score: {result['fraud_prob']:.2f}\nDecision: {result['decision']}"
    return text, source
