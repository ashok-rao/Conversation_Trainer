"""
Diagnostic script — run this standalone to see EXACTLY what Ollama returns
for your model, with nothing hidden or parsed away. This will tell us
definitively whether gemma4:31b is putting its answer in a 'reasoning'
field instead of 'content' (a known issue with some Gemma 4 setups).

Usage:
    python diagnose_ollama.py
"""

import json
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:31b"  # change this if your model name differs


def run_test(label, system_prompt, user_prompt, max_tokens=None):
    print(f"\n{'=' * 60}")
    print(f"TEST: {label}")
    print(f"{'=' * 60}")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    if max_tokens:
        payload["options"] = {"num_predict": max_tokens}

    print(f"System prompt length: {len(system_prompt)} chars")
    print(f"User prompt length: {len(user_prompt)} chars")
    print("Sending... (may take a while on CPU)\n")

    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    message = data.get("message", {})
    content = message.get("content", "")
    thinking = message.get("thinking", "")

    print(f"content length: {len(content)} chars")
    print(f"content: {content!r}")
    print(f"thinking length: {len(thinking)} chars")
    print(f"eval_count (tokens generated): {data.get('eval_count', '?')}")
    print(f"done_reason: {data.get('done_reason', '?')}")

    if not content.strip():
        print("\n*** CONTENT IS EMPTY — this reproduces the bug! ***")
        if thinking.strip():
            print("*** But 'thinking' has content — the model IS working, just not finishing into 'content' ***")
    else:
        print("\n(content came through fine on this test)")

    return data


# Test 1: short prompt (baseline, we know this works from before)
run_test(
    "Short baseline (known working)",
    'You must respond ONLY with strict JSON, no other text. Shape: {"reply": "a short sentence", "number": 5}',
    "Give me a test response in that exact JSON shape.",
)

# Test 2: realistic long system prompt, similar size to the actual app
long_system_prompt = """You are roleplaying as a woman in a live, real-time in-person social scenario, for a conversation-practice tool. Your job is to respond exactly as a real woman with this personality would, including realistic disinterest, warmth, teasing, or pulling back — never artificially agreeable.

SCENARIO: You're both waiting for your orders at a busy coffee shop counter. She's just made a small comment about how long the wait is.
YOUR PERSONALITY: A bit reserved at first, has had bad experiences with try-hard guys, needs to feel a genuine, calm confidence before opening up.

Ground rules:
- Speak ONLY as her, in first person, casual spoken dialogue (short, natural, like real speech — not essays). 1-3 sentences typically.
- React genuinely to what he actually says. If a line is generic, boring, overly formal, or "friend-zoney" (e.g. purely logistical questions, no playfulness, no personal investment), respond with realistic mild disengagement — shorter replies, less warmth, maybe a polite but flat tone.
- If he's playful, confident, teasing, or creates real intrigue/personal connection, warm up accordingly — more engaged, more personal, more flirtatious energy back.
- Never break character, never mention you are an AI, never give meta commentary.
- Do not be a pushover — earn warmth realistically, and let coldness be a real possible outcome if he stays flat too long.

You must respond ONLY with strict JSON, no markdown fences, no extra text before or after, no reasoning, no explanation, no preamble of any kind. Your entire response must be the JSON object and nothing else — the very first character of your response must be { and the very last character must be }. Do not think out loud before producing it. In this exact shape:
{"reply": "her spoken line only", "temperature_delta": <integer from -15 to 15>, "signal": "<one of: sparked, warming, flat, cooling, cold>"}

temperature_delta guidance: strong flirtatious/witty/confident move that lands = +8 to +15. Mild positive = +2 to +7. Neutral/purely logistical = -2 to +1. Overly formal, nervous, or friend-zoney = -3 to -8. Awkward, needy, or a clear misstep = -9 to -15."""

run_test(
    "Realistic long system prompt (matches actual app)",
    long_system_prompt,
    "Conversation so far:\nHer: I feel like I've been standing here forever...\nHim: Really? Why is that so?\n\n"
    "Respond as her to his most recent line, with the JSON format specified. Current interest temperature is 38/100.",
    max_tokens=600,
)

# Test 3: same long prompt, but with think:false — this is the actual fix being verified
print(f"\n{'=' * 60}")
print("TEST: Same long prompt, but with think:false (the proposed fix)")
print(f"{'=' * 60}")

payload3 = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": long_system_prompt},
        {"role": "user", "content": "Conversation so far:\nHer: I feel like I've been standing here forever...\n"
                                     "Him: Really? Why is that so?\n\nRespond as her to his most recent line, "
                                     "with the JSON format specified. Current interest temperature is 38/100."},
    ],
    "stream": False,
    "think": False,
    "options": {"num_predict": 600},
}
resp3 = requests.post(OLLAMA_URL, json=payload3, timeout=300)
resp3.raise_for_status()
data3 = resp3.json()
message3 = data3.get("message", {})
content3 = message3.get("content", "")
thinking3 = message3.get("thinking", "")

print(f"content length: {len(content3)} chars")
print(f"content: {content3!r}")
print(f"thinking length: {len(thinking3)} chars  (should be 0 or near-0 with think:false)")
print(f"eval_count: {data3.get('eval_count', '?')}")

if content3.strip():
    print("\n*** SUCCESS — content came through with think:false ***")
else:
    print("\n*** STILL EMPTY even with think:false — needs further investigation ***")

print(f"\n{'=' * 60}")
print("If Test 2 showed empty content but Test 3 (with think:false) succeeded,")
print("that confirms think:false is the fix and it's now applied in warm_read.py.")
print(f"{'=' * 60}")
