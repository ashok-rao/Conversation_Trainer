"""
Warm Read — Offline Conversation Trainer
==========================================
A terminal-based, fully local conversation-practice tool for building
quick-thinking, in-person flirting/conversation skills.

Voice input (your speech -> text) via faster-whisper (local, offline).
Voice output (her replies spoken aloud) via pyttsx3 (local, offline).
Conversational partner powered by a local Ollama model.

No internet connection required after initial setup (model downloads).
No data leaves your machine.
"""

import json
import random
import re
import sys
import threading
import time
import wave
from dataclasses import dataclass, field

import requests

try:
    import pyttsx3
except ImportError:
    print("Missing dependency 'pyttsx3'. Run: pip install pyttsx3")
    sys.exit(1)

try:
    import sounddevice as sd
    import numpy as np
except ImportError:
    print("Missing dependency 'sounddevice' or 'numpy'. Run: pip install sounddevice numpy")
    sys.exit(1)

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Missing dependency 'faster-whisper'. Run: pip install faster-whisper")
    sys.exit(1)


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma4:31b"

SAMPLE_RATE = 16000
WHISPER_MODEL_SIZE = "base.en"  # good balance of speed/accuracy on CPU

RESPONSE_WINDOW_SECONDS = 8  # default; user can change at setup


SCENARIOS = [
    {
        "id": "coffee",
        "label": "Coffee shop",
        "setting": "You're both waiting for your orders at a busy coffee shop counter. "
                   "She's just made a small comment about how long the wait is.",
    },
    {
        "id": "party",
        "label": "Party, mutual friend group",
        "setting": "You're at a house party. A mutual friend just introduced you both in "
                   "passing and then walked off to grab drinks, leaving you two standing together.",
    },
    {
        "id": "dating-app",
        "label": "Dating app, first meetup",
        "setting": "You matched on a dating app a few days ago and this is the first few "
                   "minutes of meeting in person at a casual bar, sitting down together for the first time.",
    },
    {
        "id": "bookstore",
        "label": "Bookstore / everyday encounter",
        "setting": "You're both reaching for the same book on a low shelf in a bookstore, "
                   "and just made eye contact and laughed a little.",
    },
]

PERSONAS = [
    {
        "id": "playful",
        "label": "Playful & teasing",
        "desc": "Quick with banter, enjoys teasing, gets bored by flat or overly polite "
                "responses, warms up fast to wit.",
    },
    {
        "id": "guarded",
        "label": "Guarded, warms slowly",
        "desc": "A bit reserved at first, has had bad experiences with try-hard guys, needs "
                "to feel a genuine, calm confidence before opening up.",
    },
    {
        "id": "confident",
        "label": "Confident & direct",
        "desc": "Says what she thinks, has low patience for hesitation or over-explaining, "
                "respects someone who can hold their own and match her energy.",
    },
    {
        "id": "shy",
        "label": "Shy but interested",
        "desc": "Quieter, gives shorter responses not out of disinterest but nerves, needs "
                "the guy to carry more of the conversational energy and give her easy ways to engage.",
    },
]


# ============================================================
# Terminal color helpers (ANSI — works in Windows Terminal / modern cmd)
# ============================================================

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    WARM = "\033[38;5;209m"
    WARM_HOT = "\033[38;5;202m"
    COOL = "\033[38;5;103m"
    DANGER = "\033[38;5;167m"
    GOOD = "\033[38;5;179m"
    FAINT = "\033[38;5;243m"


def enable_windows_ansi():
    """Enable ANSI escape code support on older Windows terminals."""
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)


# ============================================================
# Local AI (Ollama) helper
# ============================================================

def check_ollama():
    """Verify Ollama is running and the model is available."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        matches = [m for m in models if m.startswith(OLLAMA_MODEL)]
        if not matches:
            print(f"{C.DANGER}Ollama is running, but model '{OLLAMA_MODEL}' isn't pulled yet.{C.RESET}")
            print(f"Run this first: {C.BOLD}ollama pull {OLLAMA_MODEL}{C.RESET}")
            return False
        return True
    except requests.exceptions.ConnectionError:
        print(f"{C.DANGER}Can't reach Ollama at localhost:11434.{C.RESET}")
        print(f"Make sure Ollama is installed and running (open the Ollama app, or run 'ollama serve').")
        return False
    except Exception as e:
        print(f"{C.DANGER}Unexpected error checking Ollama: {e}{C.RESET}")
        return False


def ask_ollama(system_prompt, user_prompt, max_tokens=600, timeout=300):
    """Send a chat request to the local Ollama model and return its text response.

    Local CPU inference can be slow, especially on the very first request after
    the model loads into memory — 300s gives real headroom for larger models
    (this app currently targets a ~30B-class model, which is slower than an 8B).

    'think: False' disables/minimizes Gemma 4's built-in reasoning mode — without
    this, the model spends a chunk of its token budget on an internal 'thinking'
    trace before ever writing the actual JSON reply, which was causing empty or
    truncated responses on longer, more complex prompts (like the real conversation
    history). We don't need the reasoning trace for this use case, only the reply.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "options": {"num_predict": max_tokens},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def ask_ollama_with_feedback(system_prompt, user_prompt, max_tokens=600):
    """Wraps ask_ollama with a visible 'thinking' spinner (so long CPU inference
    doesn't look like a hang) and graceful handling of timeouts/connection drops
    so a slow response doesn't crash the whole session."""
    stop_spinner = threading.Event()
    result = {}

    def spin():
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        start = time.time()
        while not stop_spinner.is_set():
            elapsed = time.time() - start
            note = "  (larger models can take a couple minutes on CPU)" if elapsed > 20 else ""
            sys.stdout.write(f"\r{C.FAINT}{frames[i % len(frames)]} she's thinking... ({elapsed:.0f}s){note}{C.RESET}   ")
            sys.stdout.flush()
            i += 1
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * 70 + "\r")
        sys.stdout.flush()

    def worker():
        try:
            result["raw"] = ask_ollama(system_prompt, user_prompt, max_tokens=max_tokens)
        except requests.exceptions.ReadTimeout:
            result["error"] = ("timeout", "The local model took too long to respond (over 5 minutes). "
                                           "This can happen on the very first request while the model "
                                           "loads, or if your machine is under heavy load. Try again — "
                                           "it's usually faster on the next attempt.")
        except requests.exceptions.ConnectionError:
            result["error"] = ("connection", "Lost connection to Ollama. Make sure it's still running "
                                              "(check your system tray, or run 'ollama serve').")
        except Exception as e:
            result["error"] = ("unknown", f"Unexpected error talking to the model: {e}")

    spinner_thread = threading.Thread(target=spin)
    worker_thread = threading.Thread(target=worker)
    spinner_thread.start()
    worker_thread.start()
    worker_thread.join()
    stop_spinner.set()
    spinner_thread.join()

    if "error" in result:
        kind, msg = result["error"]
        print(f"{C.DANGER}⚠ {msg}{C.RESET}")
        return None
    return result.get("raw")


def safe_parse_json(raw):
    """Extract a JSON object from the model's raw text response, tolerating
    markdown fences, extra prose, and common local-model formatting quirks
    (smart quotes, trailing commas, single-quoted keys/values)."""
    text = raw.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    def try_load(candidate):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    result = try_load(text)
    if result is not None:
        return result

    # Narrow to the outermost { ... } block in case there's prose around it
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text

    result = try_load(candidate)
    if result is not None:
        return result

    # Common local-model quirks: smart quotes instead of straight quotes,
    # trailing commas before a closing brace/bracket, single quotes.
    cleaned = candidate
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)  # trailing commas

    result = try_load(cleaned)
    if result is not None:
        return result

    # Last resort: single quotes around keys/values instead of double quotes
    single_to_double = re.sub(r"'([^']*)'", r'"\1"', cleaned)
    result = try_load(single_to_double)
    if result is not None:
        return result

    return None


# ============================================================
# Text-to-speech
# ============================================================

class Voice:
    def __init__(self):
        # Store the voice preference so speak() can re-apply it on each fresh
        # engine instance, without needing to re-detect the female voice every time.
        self._voice_id = None
        probe_engine = pyttsx3.init()
        voices = probe_engine.getProperty("voices")
        female = next((v for v in voices if "female" in v.name.lower() or "zira" in v.name.lower()), None)
        if female:
            self._voice_id = female.id
        probe_engine.stop()

    def speak(self, text):
        # pyttsx3's SAPI5 event loop is effectively single-use per engine instance —
        # reusing one engine across multiple speak() calls causes every call after
        # the first to silently produce no audio. Re-initializing fresh each time
        # is the documented workaround.
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        if self._voice_id:
            engine.setProperty("voice", self._voice_id)
        engine.say(text)
        engine.runAndWait()
        engine.stop()


# ============================================================
# Speech-to-text (push-to-talk recording + Whisper transcription)
# ============================================================

class Listener:
    def __init__(self):
        print(f"{C.FAINT}Loading local speech recognition model (first run may take a moment)...{C.RESET}")
        self.model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

    def record_and_transcribe(self):
        """Press Enter to start recording, press Enter again to stop, then transcribe.
        Typing something (like /end) instead of a bare Enter is honored directly —
        useful since '/end' typed here previously got silently discarded."""
        pre_input = input(f"{C.WARM}Press ENTER to start speaking (or type /end to finish): {C.RESET}").strip()
        if pre_input:
            # User typed something instead of just pressing Enter — treat it as
            # their full response rather than silently discarding it and recording anyway.
            return pre_input

        print(f"{C.DANGER}● Recording... press ENTER to stop.{C.RESET}")

        frames = []
        stop_flag = threading.Event()

        def callback(indata, frame_count, time_info, status):
            frames.append(indata.copy())

        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback)
        stream.start()

        input()  # blocks until Enter pressed again
        stream.stop()
        stream.close()

        if not frames:
            print(f"{C.DIM}No audio captured.{C.RESET}")
            return ""

        audio = np.concatenate(frames, axis=0).flatten().astype(np.float32) / 32768.0
        print(f"{C.FAINT}Transcribing...{C.RESET}")

        segments, _ = self.model.transcribe(audio, language="en")
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text


# ============================================================
# Conversation state & flow
# ============================================================

@dataclass
class ConvoState:
    scenario: dict
    persona: dict
    timer_seconds: int
    messages: list = field(default_factory=list)  # list of (role, text)
    temperature: int = 50
    temp_history: list = field(default_factory=lambda: [50])


def build_system_prompt(state: ConvoState):
    return f"""You are roleplaying as a woman in a live, real-time in-person social scenario, for a conversation-practice tool. Your job is to respond exactly as a real woman with this personality would, including realistic disinterest, warmth, teasing, or pulling back — never artificially agreeable.

SCENARIO: {state.scenario['setting']}
YOUR PERSONALITY: {state.persona['desc']}

Ground rules:
- Speak ONLY as her, in first person, casual spoken dialogue (short, natural, like real speech — not essays). 1-3 sentences typically.
- React genuinely to what he actually says. If a line is generic, boring, overly formal, or "friend-zoney" (e.g. purely logistical questions, no playfulness, no personal investment), respond with realistic mild disengagement — shorter replies, less warmth, maybe a polite but flat tone.
- If he's playful, confident, teasing, or creates real intrigue/personal connection, warm up accordingly — more engaged, more personal, more flirtatious energy back.
- Never break character, never mention you are an AI, never give meta commentary.
- Do not be a pushover — earn warmth realistically, and let coldness be a real possible outcome if he stays flat too long.

You must respond ONLY with strict JSON, no markdown fences, no extra text before or after, no reasoning, no explanation, no preamble of any kind. Your entire response must be the JSON object and nothing else — the very first character of your response must be {{ and the very last character must be }}. Do not think out loud before producing it. In this exact shape:
{{"reply": "her spoken line only", "temperature_delta": <integer from -15 to 15>, "signal": "<one of: sparked, warming, flat, cooling, cold>"}}

temperature_delta guidance: strong flirtatious/witty/confident move that lands = +8 to +15. Mild positive = +2 to +7. Neutral/purely logistical = -2 to +1. Overly formal, nervous, or friend-zoney = -3 to -8. Awkward, needy, or a clear misstep = -9 to -15."""


def temp_label(temp):
    if temp >= 75:
        return "Sparking", C.WARM_HOT
    if temp >= 55:
        return "Warming", C.WARM
    if temp >= 40:
        return "Neutral", C.WARM
    if temp >= 20:
        return "Cooling", C.COOL
    return "Cold", C.COOL


def print_temp_bar(state: ConvoState):
    label, color = temp_label(state.temperature)
    filled = int(state.temperature / 100 * 30)
    bar = "█" * filled + "░" * (30 - filled)
    print(f"\n{C.FAINT}INTEREST TEMPERATURE{C.RESET}  {color}{C.BOLD}{label} · {state.temperature}{C.RESET}")
    print(f"{color}{bar}{C.RESET}\n")


def run_countdown_display(seconds):
    """Blocking countdown that runs to completion BEFORE the input/recording
    prompt appears — no threading, so there's no race condition with input().
    Returns nothing; just gives a visible sense of pressure before your turn."""
    start = time.time()
    while True:
        elapsed = time.time() - start
        remaining = max(0, seconds - elapsed)
        sys.stdout.write(f"\r{C.FAINT}Get ready to respond: {remaining:4.1f}s{C.RESET}   ")
        sys.stdout.flush()
        if remaining <= 0:
            sys.stdout.write(f"\r{C.DANGER}Go — respond now (a slow start reads as hesitation).{C.RESET}\n")
            sys.stdout.flush()
            break
        time.sleep(0.1)


def run_conversation(state: ConvoState, voice: Voice, listener: Listener, use_voice_input: bool):
    system_prompt = build_system_prompt(state)

    print(f"\n{C.BOLD}{C.WARM}{state.scenario['label']} · {state.persona['label']}{C.RESET}\n")

    # Opening line — retry on failure since this is the very first thing that happens
    raw = None
    while raw is None:
        raw = ask_ollama_with_feedback(system_prompt, "Give your opening line to start this scene naturally — "
                                        "the very first thing she says to kick things off. "
                                        "Respond only with the JSON format specified.")
        if raw is None:
            retry = input(f"{C.WARM}Try again? (y/n): {C.RESET}").strip().lower()
            if retry != "y":
                print(f"{C.FAINT}Ending session.{C.RESET}")
                return state

    parsed = safe_parse_json(raw)
    opener = (parsed or {}).get("reply", "Hey.")
    state.messages.append(("her", opener))

    print(f"{C.ITALIC}HER: {opener}{C.RESET}")
    voice.speak(opener)

    while True:
        print_temp_bar(state)

        run_countdown_display(state.timer_seconds)

        start_time = time.time()
        if use_voice_input:
            user_text = listener.record_and_transcribe()
        else:
            user_text = input(f"{C.WARM}You: {C.RESET}")
        elapsed = time.time() - start_time

        if not user_text.strip():
            print(f"{C.DIM}(nothing captured, try again){C.RESET}")
            continue

        if user_text.strip().lower() in ("/end", "end", "quit", "exit"):
            break

        print(f"{C.WARM}You said: {user_text}{C.RESET}")
        state.messages.append(("him", user_text))

        took_too_long = elapsed > state.timer_seconds
        timing_note = ""
        if took_too_long:
            timing_note = (f" He took a noticeably long pause before responding ({elapsed:.1f}s, "
                            f"past the natural window) — real hesitation, like he froze or was "
                            f"searching for something to say. Let that read as a small awkward "
                            f"beat unless his line is strong enough to fully recover it.")
        else:
            timing_note = f" He responded quickly ({elapsed:.1f}s), keeping natural conversational pace."

        transcript = "\n".join(f"{'Him' if r == 'him' else 'Her'}: {t}" for r, t in state.messages)
        user_prompt = (f"Conversation so far:\n{transcript}\n\nRespond as her to his most recent line, "
                        f"with the JSON format specified. Current interest temperature is "
                        f"{state.temperature}/100.{timing_note}")

        print(f"{C.FAINT}she's responding...{C.RESET}")
        raw = ask_ollama_with_feedback(system_prompt, user_prompt)
        if raw is None:
            # Remove the message we just added so the transcript stays in sync —
            # otherwise the next turn would have two "him" lines in a row.
            state.messages.pop()
            print(f"{C.DIM}Try your line again.{C.RESET}")
            continue

        parsed = safe_parse_json(raw)
        if not parsed:
            state.messages.pop()
            if raw is not None and raw.strip() == "":
                print(f"{C.DANGER}(the model returned an empty response — nothing to parse){C.RESET}")
                print(f"{C.DIM}This can happen if num_predict cut it off too early, or the model "
                      f"produced only whitespace/control tokens. If this keeps happening, we may "
                      f"need to raise max_tokens or adjust the prompt.{C.RESET}")
            else:
                print(f"{C.DANGER}(couldn't parse a response as JSON — here's what the model actually said:){C.RESET}")
                print(f"{C.DIM}{raw!r}{C.RESET}")
            print(f"{C.DIM}Try your line again.{C.RESET}")
            continue

        reply = parsed.get("reply", "...")
        delta = parsed.get("temperature_delta", 0)
        try:
            delta = int(delta)
        except (TypeError, ValueError):
            delta = 0

        state.temperature = max(0, min(100, state.temperature + delta))
        state.temp_history.append(state.temperature)
        state.messages.append(("her", reply))

        print(f"\n{C.ITALIC}HER: {reply}{C.RESET}")
        voice.speak(reply)

    return state


def generate_feedback(state: ConvoState):
    transcript = "\n".join(f"{'Him' if r == 'him' else 'Her'}: {t}" for r, t in state.messages)

    feedback_system = """You are a sharp, honest social-skills coach reviewing a practice conversation. The user is working on: he tends to run out of things to say, freezes up, and ends conversations on friendly notes rather than sparking romantic interest. Be specific and concrete, cite actual lines from the transcript, and never generic ("be more confident"). Be encouraging but direct — do not sugarcoat missed opportunities.

Respond ONLY with strict JSON, no markdown fences, no extra text, in this exact shape:
{
  "overall": "2-3 sentence honest summary of how this conversation went for him",
  "strengths": ["specific moment with a short quote and why it worked"],
  "misses": ["specific moment with a short quote where he went flat/friendly/ran out of steam, and why"],
  "rewrites": [{"his_line": "a flat/weak line he actually said", "better_line": "an improved version"}],
  "pattern": "1-2 sentences naming the recurring pattern across the conversation",
  "next_focus": "one single concrete thing to focus on next time"
}"""

    user_prompt = (f"Scenario: {state.scenario['setting']}\nHer personality: {state.persona['desc']}\n\n"
                    f"Full transcript:\n{transcript}\n\nFinal interest temperature: "
                    f"{state.temperature}/100 (started at 50). Give the feedback report.")

    print(f"\n{C.FAINT}Reading back through the conversation...{C.RESET}\n")
    raw = ask_ollama_with_feedback(feedback_system, user_prompt, max_tokens=700)

    print(f"{C.BOLD}{C.WARM}── DEBRIEF ──{C.RESET}\n")

    if raw is None:
        print(f"{C.DANGER}Couldn't reach the model to generate a debrief this time.{C.RESET}")
        return

    report = safe_parse_json(raw)

    if not report:
        print(f"{C.DANGER}Couldn't generate a structured report this time. Raw model output:{C.RESET}")
        print(raw)
        return

    print(f"{report.get('overall', '')}\n")

    if report.get("strengths"):
        print(f"{C.GOOD}{C.BOLD}WHAT WORKED{C.RESET}")
        for s in report["strengths"]:
            print(f"  • {s}")
        print()

    if report.get("misses"):
        print(f"{C.DANGER}{C.BOLD}WHERE IT COOLED{C.RESET}")
        for s in report["misses"]:
            print(f"  • {s}")
        print()

    if report.get("rewrites"):
        print(f"{C.WARM}{C.BOLD}TRY THIS INSTEAD{C.RESET}")
        for r in report["rewrites"]:
            print(f"  {C.FAINT}You said:{C.RESET} \"{r.get('his_line', '')}\"")
            print(f"  {C.WARM}Try:{C.RESET} \"{r.get('better_line', '')}\"\n")

    if report.get("pattern"):
        print(f"{C.COOL}{C.BOLD}THE PATTERN{C.RESET}")
        print(f"  {report['pattern']}\n")

    if report.get("next_focus"):
        print(f"{C.BOLD}NEXT FOCUS{C.RESET}")
        print(f"  {report['next_focus']}\n")


# ============================================================
# Setup / menu flow
# ============================================================

def choose_from_list(prompt, options, label_key="label"):
    print(f"\n{C.BOLD}{prompt}{C.RESET}")
    print(f"  0. Surprise me")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt[label_key]}")
    while True:
        choice = input(f"{C.WARM}> {C.RESET}").strip()
        if choice == "0" or choice == "":
            return random.choice(options)
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print(f"{C.DIM}Enter a number from the list.{C.RESET}")


def main():
    enable_windows_ansi()

    print(f"{C.BOLD}{C.WARM}")
    print("╔══════════════════════════════════════╗")
    print("║           WARM READ — OFFLINE          ║")
    print("╚══════════════════════════════════════╝")
    print(f"{C.RESET}")
    print(f"{C.FAINT}Local, offline conversation trainer. No data leaves your machine.{C.RESET}\n")

    if not check_ollama():
        sys.exit(1)

    use_voice_input = input(f"\n{C.WARM}Use voice input? (y/n, default y): {C.RESET}").strip().lower() != "n"

    listener = None
    if use_voice_input:
        listener = Listener()

    voice = Voice()

    while True:
        scenario = choose_from_list("Choose a setting:", SCENARIOS)
        persona = choose_from_list("Choose her personality:", PERSONAS)

        timer_input = input(f"\n{C.WARM}Response window in seconds (default 8): {C.RESET}").strip()
        try:
            timer_seconds = int(timer_input) if timer_input else RESPONSE_WINDOW_SECONDS
        except ValueError:
            timer_seconds = RESPONSE_WINDOW_SECONDS

        print(f"\n{C.FAINT}Type '/end' at the 'Press ENTER to start speaking' prompt (before you start "
              f"recording) to finish and get feedback.{C.RESET}")

        state = ConvoState(scenario=scenario, persona=persona, timer_seconds=timer_seconds)
        state = run_conversation(state, voice, listener, use_voice_input)

        if len(state.messages) > 1:
            generate_feedback(state)

        again = input(f"\n{C.WARM}Practice again? (y/n): {C.RESET}").strip().lower()
        if again != "y":
            break

    print(f"\n{C.FAINT}Session ended. Good work.{C.RESET}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C.FAINT}Interrupted. Bye.{C.RESET}")
