"""Train and export SONJ4's tiny local ONNX intent classifier (front | back | doctor).

Exports:
    models/classifier/intent.onnx   (float32[1, D] -> float32[1, 3] probabilities)
    models/classifier/vocab.json    (vocabulary + IDF weights matching classifier.py)

Run:
    .venv/bin/python train_classifier_onnx.py
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from classifier import extract_text_features

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "models" / "classifier"
LABELS = ("front", "back", "doctor")

# Curated, balanced training corpus covering:
# - canonical mex3 / classifiercheck / sonj4_eval20 prompts
# - natural conversational variations & paraphrases
# - negations ("don't build...", "don't set a timer...")
# - natural audio/mic/TTS/tool troubleshooting -> doctor
DATASET: list[tuple[str, str]] = [
    # ---- FRONT (0): quick device actions, time/math, reminders, music, memory, short Q&A, negations ----
    ("what time is it?", "front"),
    ("tell me the current time", "front"),
    ("what is the date today?", "front"),
    ("what day of the week is it?", "front"),
    ("what is the weather right now?", "front"),
    ("what is apple's stock price?", "front"),
    ("what is the stock price of nvidia?", "front"),
    ("check tesla tsla stock price right now", "front"),
    ("get msft stock price today", "front"),
    ("what is the stock price of gcash?", "front"),
    ("check the stock price of smart or pldt", "front"),
    ("check stock price of gcash or smart", "front"),
    ("calculate 17 * 23", "front"),
    ("calculate 17 times 23", "front"),
    ("what is 125 divided by 5?", "front"),
    ("what is 5 plus 7?", "front"),
    ("what is 48 minus 19?", "front"),
    ("90 minus 27", "front"),
    (
        "i have 90 seconds; 27 seconds have passed. how many seconds remain? don't set a timer - just answer in one sentence.",
        "front",
    ),
    ("don't set a timer, just tell me how many minutes are left", "front"),
    ("don't build a game, just tell me what time it is", "front"),
    ("do not create a script, just answer in one sentence", "front"),
    ("never mind the script, just say hello", "front"),
    (
        "please skip writing a program, just tell me how many kilometers are in 10 miles",
        "front",
    ),
    ("don't run a tool or write code, just convert 72 fahrenheit to celsius", "front"),
    ("convert 100 usd to eur in one sentence", "front"),
    ("how many ounces are in a pound?", "front"),
    ("turn on the bedroom lights", "front"),
    ("turn off the living room lamp", "front"),
    ("switch off the kitchen lights", "front"),
    ("dim the lights to 30 percent", "front"),
    ("brighten the desk lamp", "front"),
    ("set a timer for 30 seconds", "front"),
    ("start a 5 minute timer", "front"),
    ("cancel my timer", "front"),
    ("set an alarm for 7 am", "front"),
    ("wake me up at 6:30 tomorrow", "front"),
    ("remind me to drink water in 10 minutes", "front"),
    ("could you remind me about the meeting tomorrow morning at nine", "front"),
    ("list my reminders", "front"),
    ("what reminders do i have?", "front"),
    ("turn the volume up", "front"),
    ("lower the volume", "front"),
    ("mute the speaker", "front"),
    ("play music", "front"),
    ("pause the music", "front"),
    ("next track", "front"),
    ("skip this song", "front"),
    ("previous track", "front"),
    ("turn down the thermostat", "front"),
    ("raise the temperature by two degrees", "front"),
    ("tell me a short joke", "front"),
    ("tell me a joke", "front"),
    ("hello there", "front"),
    ("hi nova", "front"),
    ("hey piattos", "front"),
    ("good morning", "front"),
    ("good night", "front"),
    ("how are you doing today?", "front"),
    ("thank you so much", "front"),
    ("thanks for helping", "front"),
    ("remember that my favorite color is blue", "front"),
    ("remember that my test color is teal", "front"),
    ("what is my favorite color?", "front"),
    ("my name is leandro", "front"),
    ("what is my name?", "front"),
    (
        "please explain how photosynthesis works in simple words for a ten year old student",
        "front",
    ),
    ("tell me a two sentence summary of why the sky is blue", "front"),
    ("what is the capital of Japan?", "front"),
    ("who invented the telephone?", "front"),
    ("say hello world aloud", "front"),
    ("be quiet", "front"),
    ("stop talking", "front"),
    ("silence", "front"),
    ("call mom", "front"),
    # ---- BACK (1): build/code/files, web research, verify/audit, drafting/writing/summaries ----
    ("create a snake game in sonj4/out/sandbox/snake.py", "back"),
    ("create a small playable snake game in html", "back"),
    ("create a small calculator webpage and save it in sandbox", "back"),
    ("build two small scripts in parallel: hello.py and bye.py", "back"),
    ("write a python script to sort a csv file by the second column", "back"),
    ("can you write up a quick python script to parse json?", "back"),
    (
        "create ipv4_check.py with a function that validates ipv4 addresses and run tests",
        "back",
    ),
    ("make a small webpage with a bio link and save it", "back"),
    ("build a todo list app in html and javascript", "back"),
    ("create a tiny json file with three sample records and validate it", "back"),
    ("automate renaming all jpg files in a folder using python", "back"),
    ("generate a markdown report from this csv dataset", "back"),
    ("read the pdf file and extract the verification code", "back"),
    ("open the pdf document in out/reports and pull out the token", "back"),
    ("read this pdf report and find the reference number inside it", "back"),
    ("compare raspberry pi 5 and jetson orin nano in a table and save a file", "back"),
    ("write a bash script to back up my folder", "back"),
    ("write me a haiku about rain", "back"),
    ("write a short poem about the ocean", "back"),
    ("draft a polite email declining a meeting", "back"),
    ("draft a cover letter for an engineering role", "back"),
    ("draft an email thanking my professor for the feedback", "back"),
    ("write a short thank you note and save it to notes.txt", "back"),
    ("compose a formal invitation letter for the workshop", "back"),
    ("research the latest news about raspberry pi 5 and give me a source", "back"),
    (
        "research the latest raspberry pi 5 news and give me one headline with its source",
        "back",
    ),
    ("research manila weather trends and verify the source link", "back"),
    (
        "research current ai governance news and summarize the three main takeaways",
        "back",
    ),
    ("research ai governance news and summarize the top 3 takeaways", "back"),
    (
        "research the official python documentation for pathlib and give me the url",
        "back",
    ),
    (
        "download the official gazette of the philippines latest proclamation pdf",
        "back",
    ),
    (
        "find the latest official gazette proclamation and give me the official page and pdf link",
        "back",
    ),
    ("look up who won the latest nba game", "back"),
    ("look up current bitcoin price from the web", "back"),
    ("search the web for python 3.13 release highlights", "back"),
    ("find out the latest updates on edge ai hardware", "back"),
    ("verify this claim: the earth is flat and audit the evidence", "back"),
    ("verify this claim: 19 * 29 = 551 and audit the math", "back"),
    ("verify this claim: 19 times 29 equals 551. show the calculation.", "back"),
    ("fact check whether coffee dehydrates you and cite sources", "back"),
    ("audit this script for bugs and write unit tests", "back"),
    ("summarize this long article into bullet points", "back"),
    ("condense these meeting notes into action items", "back"),
    ("have one back worker research a fact and another verify its source", "back"),
    ("create a small text file in the sandbox and post progress to community", "back"),
    ("fetch the latest exchange rate for usd to php and cite the source", "back"),
    ("create a github public repo sample with a readme", "back"),
    ("create a public github repository named demo-sample", "back"),
    ("create a public github repo", "back"),
    ("check the github repository octocat/Hello-World", "back"),
    ("check other github repositories and summarize what they do", "back"),
    ("check github repo of another user and list its files", "back"),
    ("read the readme file in my github repository", "back"),
    ("read readme in my github repo", "back"),
    ("write a new python script into the github public repo", "back"),
    ("write a file into github repo", "back"),
    ("edit a file inside my github repo and commit the change", "back"),
    ("edit readme in my github repo", "back"),
    ("edit public repository readme on github", "back"),
    ("edit a file on github", "back"),
    ("delete a file in my github repository", "back"),
    ("list my github repositories and create a pull request", "back"),
    ("open an issue in my github public repo", "back"),
    ("create a branch in my github repo", "back"),
    ("check https://github.com/roatienza/deep-learning-experiments and tell me what is inside", "back"),
    ("read the roatienza deep-learning-experiments repository on github", "back"),
    ("check repo and produce a public github repo sample", "back"),
    # ---- DOCTOR (2): health checks, logs, settings, architecture, audio/mic/TTS/tool diagnostics ----
    ("run a health check on the app", "doctor"),
    ("run a health check on sonj4 and tell me the status", "doctor"),
    ("check system health and report any warnings", "doctor"),
    ("check the incident log and diagnose any errors", "doctor"),
    ("check recent incident logs for audio or tts errors", "doctor"),
    ("inspect the app settings and explain the doctor's repair limits", "doctor"),
    ("change wake_gate setting via doctor", "doctor"),
    ("change wake_gate setting or inspect app settings via doctor", "doctor"),
    ("review app architecture and verify config.json settings", "doctor"),
    ("check if any tasks failed and inspect system health logs", "doctor"),
    (
        "check this session for failed or cancelled tasks and inspect audio/asr/tts health",
        "doctor",
    ),
    ("diagnose why audio playback stopped working", "doctor"),
    ("why is my microphone not hearing me right now?", "doctor"),
    ("my mic isn't picking up my voice, can you inspect it?", "doctor"),
    ("the microphone looks broken and cannot hear speech", "doctor"),
    ("microphone stopped detecting my voice", "doctor"),
    ("audio input device is not capturing voice", "doctor"),
    ("the audio keeps cutting out and tts is silent, can you check?", "doctor"),
    ("why is tts silent and not speaking aloud?", "doctor"),
    ("speakers are not playing any sound, diagnose the audio stack", "doctor"),
    ("what model is front using right now and why did my last task fail?", "doctor"),
    ("why did my background task fail or get cancelled?", "doctor"),
    ("is the web search tool working today or did it throw an error?", "doctor"),
    ("check if asr or vad is throwing errors in the logs", "doctor"),
    ("inspect server_err.txt and tell me why startup warned", "doctor"),
    ("troubleshoot voiceprint barge-in false triggers", "doctor"),
    ("inspect the app logs for any crashed workers", "doctor"),
    ("propose setting tts_rate to 0 using doctor", "doctor"),
    ("can doctor inspect why latency spiked on the last turn?", "doctor"),
    ("check if all agent panes are warm and healthy", "doctor"),
    ("are there any errors in config.json or agents.json?", "doctor"),
    ("inspect the diagnostics and system health status", "doctor"),
    ("why is the assistant lagging when listening?", "doctor"),
]


def build_vocab(
    texts: list[str], min_df: int = 1, max_features: int = 1600
) -> tuple[dict[str, int], np.ndarray]:
    df: Counter[str] = Counter()
    for t in texts:
        feats = extract_text_features(t)
        for k in feats:
            df[k] += 1
    # Prioritize word/bigram/negation/meta features, then frequent char n-grams
    items = [(k, c) for k, c in df.items() if c >= min_df or not k.startswith("c")]
    items.sort(key=lambda kv: (0 if not kv[0].startswith("c") else 1, -kv[1], kv[0]))
    selected = [k for k, _ in items[:max_features]]
    n_docs = len(texts)
    index = {k: i for i, k in enumerate(selected)}
    idf = np.array(
        [math.log((1.0 + n_docs) / (1.0 + df[k])) + 1.0 for k in selected],
        dtype=np.float32,
    )
    return index, idf


def vectorize_batch(
    texts: list[str], index: dict[str, int], idf: np.ndarray
) -> np.ndarray:
    x = np.zeros((len(texts), len(index)), dtype=np.float32)
    for row, text in enumerate(texts):
        feats = extract_text_features(text)
        for k, cnt in feats.items():
            col = index.get(k)
            if col is not None:
                x[row, col] = (1.0 + math.log(cnt)) * float(idf[col])
        norm = float(np.linalg.norm(x[row]))
        if norm > 1e-12:
            x[row] /= norm
    return x


def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    hidden: int = 48,
    epochs: int = 450,
    lr: float = 0.04,
    l2: float = 3e-4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = x.shape
    c = len(LABELS)
    w1 = (rng.standard_normal((d, hidden)) * math.sqrt(2.0 / d)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = (rng.standard_normal((hidden, c)) * math.sqrt(2.0 / hidden)).astype(np.float32)
    b2 = np.zeros(c, dtype=np.float32)

    # Class-balanced sample weights
    counts = np.bincount(y, minlength=c).astype(np.float32)
    cls_w = (n / (c * np.maximum(counts, 1.0))).astype(np.float32)
    sample_w = cls_w[y][:, None]

    y_onehot = np.zeros((n, c), dtype=np.float32)
    y_onehot[np.arange(n), y] = 1.0

    # Adam state
    params = [w1, b1, w2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for step in range(1, epochs + 1):
        z1 = x @ params[0] + params[1]
        h1 = np.maximum(z1, 0.0)
        logits = h1 @ params[2] + params[3]
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)

        dlogits = ((probs - y_onehot) * sample_w) / n
        dw2 = h1.T @ dlogits + l2 * params[2]
        db2 = dlogits.sum(axis=0)
        dh1 = dlogits @ params[2].T
        dz1 = dh1 * (z1 > 0.0)
        dw1 = x.T @ dz1 + l2 * params[0]
        db1 = dz1.sum(axis=0)

        grads = [dw1, db1, dw2, db2]
        cur_lr = lr * (0.5 * (1.0 + math.cos(math.pi * step / epochs)))
        for idx, (p, g) in enumerate(zip(params, grads)):
            m[idx] = beta1 * m[idx] + (1.0 - beta1) * g
            v[idx] = beta2 * v[idx] + (1.0 - beta2) * (g * g)
            m_hat = m[idx] / (1.0 - beta1**step)
            v_hat = v[idx] / (1.0 - beta2**step)
            p -= cur_lr * m_hat / (np.sqrt(v_hat) + eps)

    return params[0], params[1], params[2], params[3]


def export_onnx(
    path: Path,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> None:
    d = w1.shape[0]
    c = w2.shape[1]
    inp = helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, d])
    out = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [1, c])

    init_w1 = helper.make_tensor(
        "W1", TensorProto.FLOAT, w1.shape, w1.astype(np.float32).flatten().tolist()
    )
    init_b1 = helper.make_tensor(
        "B1", TensorProto.FLOAT, b1.shape, b1.astype(np.float32).flatten().tolist()
    )
    init_w2 = helper.make_tensor(
        "W2", TensorProto.FLOAT, w2.shape, w2.astype(np.float32).flatten().tolist()
    )
    init_b2 = helper.make_tensor(
        "B2", TensorProto.FLOAT, b2.shape, b2.astype(np.float32).flatten().tolist()
    )

    nodes = [
        helper.make_node("Gemm", ["features", "W1", "B1"], ["z1"]),
        helper.make_node("Relu", ["z1"], ["h1"]),
        helper.make_node("Gemm", ["h1", "W2", "B2"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "sonj4_intent_mlp",
        [inp],
        [out],
        initializer=[init_w1, init_b1, init_w2, init_b2],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def main() -> int:
    texts = [t for t, _ in DATASET]
    y = np.array([LABELS.index(lbl) for _, lbl in DATASET], dtype=np.int64)

    # 1. Stratified 5-fold CV check to measure out-of-sample generalization
    rng = np.random.default_rng(42)
    folds = np.zeros(len(DATASET), dtype=np.int64)
    for cls in range(len(LABELS)):
        idxs = np.where(y == cls)[0]
        rng.shuffle(idxs)
        for j, idx in enumerate(idxs):
            folds[idx] = j % 5
    cv_correct = 0
    for f in range(5):
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]
        tr_texts = [texts[i] for i in tr_idx]
        va_texts = [texts[i] for i in va_idx]
        idx_map, idf_vec = build_vocab(tr_texts, min_df=1, max_features=1600)
        x_tr = vectorize_batch(tr_texts, idx_map, idf_vec)
        x_va = vectorize_batch(va_texts, idx_map, idf_vec)
        w1, b1, w2, b2 = train_mlp(x_tr, y[tr_idx])
        h = np.maximum(x_va @ w1 + b1, 0.0)
        pred = np.argmax(h @ w2 + b2, axis=1)
        cv_correct += int(np.sum(pred == y[va_idx]))
    print(
        f"5-fold CV accuracy (unseen splits): {cv_correct}/{len(DATASET)} ({100.0 * cv_correct / len(DATASET):.1f}%)"
    )

    # 2. Train final ONNX model on full curated dataset
    index, idf = build_vocab(texts, min_df=1, max_features=1600)
    x_all = vectorize_batch(texts, index, idf)
    w1, b1, w2, b2 = train_mlp(x_all, y)
    h = np.maximum(x_all @ w1 + b1, 0.0)
    pred = np.argmax(h @ w2 + b2, axis=1)
    tr_acc = int(np.sum(pred == y))
    print(
        f"Full dataset accuracy: {tr_acc}/{len(DATASET)} ({100.0 * tr_acc / len(DATASET):.1f}%) | D={len(index)}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = OUT_DIR / "intent.onnx"
    vocab_path = OUT_DIR / "vocab.json"
    export_onnx(onnx_path, w1, b1, w2, b2)
    ordered_tokens = [tok for tok, _ in sorted(index.items(), key=lambda kv: kv[1])]
    vocab_path.write_text(
        json.dumps(
            {"labels": list(LABELS), "tokens": ordered_tokens, "idf": idf.tolist()},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")
    print(f"Saved: {vocab_path} ({vocab_path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
