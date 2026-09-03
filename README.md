# Warm Read — Offline Conversation Trainer

A fully local, offline conversation-practice tool for in-person conversation
and flirting skills. Runs entirely on your machine — no data leaves your
computer, no internet needed once set up.

Real voice input (your speech, transcribed locally) and real voice output
(her replies, spoken aloud) — no browser permission restrictions, because
this runs as a native Python app, not inside a browser sandbox.

---

## What you're installing

- **Ollama** — runs a local AI language model on your machine (this powers
  her conversational responses)
- **Python packages** — for speech-to-text, text-to-speech, and talking to
  Ollama

Total setup time: roughly 15-20 minutes, most of it spent waiting for
downloads.

---

## Setup (Windows)

### 1. Install Ollama

Download and run the installer from **https://ollama.com/download** —
choose Windows. Run the installer normally, no special options needed.

### 2. Pull the model

Open **Command Prompt** (press Windows key, type `cmd`, press Enter) and run:

```
ollama pull llama3.1
```

This downloads the model (a few GB — grab a coffee). Once it finishes,
verify Ollama is working:

```
ollama list
```

You should see `llama3.1` in the list.

### 3. Install Python (if you don't already have it)

Check first:

```
python --version
```

If that fails, download Python from **https://python.org/downloads** —
during install, **make sure to tick "Add Python to PATH"**.

### 4. Install the required Python packages

In Command Prompt, navigate to the folder where you saved these files
(e.g. `cd Downloads\warm-read-desktop`), then run:

```
pip install -r requirements.txt
```

This installs everything needed for speech recognition, text-to-speech,
and talking to Ollama. The first line (`faster-whisper`) is the biggest
download — it includes the local speech recognition model.

### 5. Run it

Make sure Ollama is running in the background (it usually starts
automatically after install — check for its icon in your system tray).
Then, in the same Command Prompt window:

```
python warm_read.py
```

---

## How to use it

1. It'll check Ollama is reachable, then ask if you want voice input
   (recommended — that's the whole point).
2. Pick a scenario and a personality for your practice partner (or hit
   Enter / "0" for a random one).
3. Set your response window in seconds — how long you have to reply
   before it's counted as a hesitant pause. Start around 8s, and try
   pushing down to 5s once you're comfortable.
4. **Press Enter to start recording**, speak your line, **press Enter
   again to stop**. It transcribes what you said and sends it to her.
5. She replies — you'll see the text and hear it spoken aloud.
6. Say or type `/end` at any point to finish and get your debrief —
   a breakdown of what worked, where the conversation cooled off, and
   specific rewrites for your weaker lines.
7. Choose to practice again with a new scenario, or quit.

---

## Notes on the local model

This uses **Llama 3.1 8B** running entirely on your machine — it's a
genuinely capable model, but it's not as sharp as some cloud-based
models at picking up on subtle conversational tone or staying tightly
in character over a long conversation. If replies start feeling
repetitive, generic, or occasionally break character, that's a known
limitation of running locally rather than a bug. If your laptop has a
dedicated NVIDIA GPU, Ollama will automatically use it and responses
will be noticeably faster; on CPU alone, expect a few seconds of
"thinking" time before each reply.

If you want a different quality/speed trade-off later, you can swap
the model by changing `OLLAMA_MODEL` near the top of `warm_read.py` to
another model you've pulled via `ollama pull <model-name>` — larger
models (e.g. `llama3.1:70b`) are noticeably better but need much more
RAM and a good GPU to run at a usable speed.

## Troubleshooting

- **"Can't reach Ollama"** — make sure the Ollama app is actually
  running (check your system tray), or run `ollama serve` manually in
  a separate Command Prompt window and leave it open.
- **No microphone input detected** — check Windows' Sound settings
  (Settings → System → Sound → Input) to confirm your microphone is
  selected as the default input device.
- **Voice sounds robotic/wrong** — `pyttsx3` uses whatever voices are
  installed on your Windows system. You can install additional voices
  via Settings → Time & Language → Speech.
- **Responses are slow** — this is normal on CPU-only machines,
  especially on first use while the model loads into memory. Later
  responses in the same session are usually faster.
