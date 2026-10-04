"""
dashboard.py — next-word prediction, with a health check on the result.

Type some text, press Predict, and the page tells you both the predicted word
and how much that prediction can be trusted.

Run:
    streamlit run dashboard.py
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import streamlit as st

HERE = Path(__file__).resolve().parent
MODEL_FILE = HERE / "lstm_model.h5"
TOKENIZER_FILE = HERE / "tokenizer.pkl"
MAXLEN_FILE = HERE / "max_len.pkl"

st.set_page_config(page_title="Next Word Prediction", page_icon="🧠", layout="centered")

import h5py  # noqa: E402  (ships with tensorflow)


# ── loading ────────────────────────────────────────────────────────────────
# The checkpoint was saved by Keras 2.x and its config uses keys that current
# Keras rejects, so `load_model()` fails on it. Read the architecture out of the
# HDF5, rebuild the same graph, and copy the weights across by name.

def read_config(path):
    with h5py.File(path, "r") as f:
        cfg = f.attrs["model_config"]
        return json.loads(cfg.decode("utf-8") if isinstance(cfg, bytes) else cfg)


def read_width(path):
    for layer in read_config(path)["config"]["layers"]:
        if layer["class_name"] == "InputLayer":
            shape = layer["config"].get("batch_shape") or layer["config"].get("batch_input_shape")
            if shape:
                return int(shape[-1])
    return None


def load_model_file(path, width):
    import keras

    cfg = {l["class_name"]: l["config"] for l in read_config(path)["config"]["layers"]}
    emb, lstm, dense = cfg["Embedding"], cfg["LSTM"], cfg["Dense"]

    model = keras.Sequential([
        keras.layers.Input(shape=(width,), dtype="int32"),
        keras.layers.Embedding(emb["input_dim"], emb["output_dim"], name=emb["name"]),
        keras.layers.LSTM(
            lstm["units"], activation=lstm["activation"],
            recurrent_activation=lstm["recurrent_activation"],
            use_bias=lstm["use_bias"], unit_forget_bias=lstm["unit_forget_bias"],
            return_sequences=lstm["return_sequences"], name=lstm["name"],
        ),
        keras.layers.Dense(dense["units"], activation=dense["activation"],
                           use_bias=dense["use_bias"], name=dense["name"]),
    ])

    # Paths look like `lstm/sequential_1/lstm/lstm_cell/kernel`; `lstm` and
    # `dense_1` both end in `kernel`, so key on (layer, weight name).
    names = {"embedding_1": ("embeddings",),
             "lstm": ("kernel", "recurrent_kernel", "bias"),
             "dense_1": ("kernel", "bias")}
    found = {}
    with h5py.File(path, "r") as f:
        f["model_weights"].visititems(
            lambda n, o: found.__setitem__((n.split("/")[0], n.split("/")[-1]), np.asarray(o))
            if isinstance(o, h5py.Dataset) else None
        )
    for layer in model.layers:
        layer.set_weights([found[(layer.name, w)] for w in names[layer.name]])
    return model


def pad(ids, width):
    row = np.zeros(width, dtype="int32")
    ids = list(ids)[-width:]  # keep the most recent tokens
    if ids:
        row[width - len(ids):] = ids
    return row


def shape_of(layer):
    """Keras 3 layers have no .output_shape attribute; read it off the tensor."""
    try:
        return str(tuple(layer.output.shape))
    except Exception:
        return "?"


# ── load once, keep for the whole session ──────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def get_model():
    t0 = time.perf_counter()
    tokenizer = pickle.load(open(TOKENIZER_FILE, "rb"))
    max_len = int(pickle.load(open(MAXLEN_FILE, "rb")))
    width = read_width(MODEL_FILE) or max_len
    model = load_model_file(MODEL_FILE, width)
    load_secs = time.perf_counter() - t0

    index_word = getattr(tokenizer, "index_word", None) or {
        i: w for w, i in tokenizer.word_index.items()}

    def predict(text):
        """Returns (tokens_matched, probabilities, milliseconds)."""
        ids = tokenizer.texts_to_sequences([text])[0]
        t = time.perf_counter()
        probs = model.predict(pad(ids, width)[None, :], verbose=0)[0]
        return ids, probs, (time.perf_counter() - t) * 1000

    return {
        "predict": predict, "tokenizer": tokenizer, "index_word": index_word,
        "max_len": max_len, "width": width, "load_secs": load_secs,
        "params": int(model.count_params()),
        "vocab": len(tokenizer.word_index),
        "out_width": int(np.asarray(model.output_shape).ravel()[-1]),
        "file_mb": MODEL_FILE.stat().st_size / 1e6,
        "weights_mb": model.count_params() * 4 / 1e6,
        "layers": [(l.__class__.__name__, shape_of(l), int(l.count_params()))
                   for l in model.layers],
    }


M = get_model()
word_for = lambda i: M["index_word"].get(int(i), str(i))  # noqa: E731


# ── model facts (always visible) ───────────────────────────────────────────

st.title("🧠 Next Word Prediction")
st.caption(f"`{MODEL_FILE.name}` · loaded in {M['load_secs']:.1f}s")

f1, f2, f3, f4 = st.columns(4)
f1.metric("Parameters", f"{M['params']:,}")
f2.metric("Vocabulary", f"{M['vocab']:,} words")
f3.metric("Context window", f"{M['width']} tokens")
f4.metric("Model file", f"{M['file_mb']:.1f} MB")

st.divider()

# ── input ──────────────────────────────────────────────────────────────────

st.subheader("Enter your text")
text = st.text_input("Type a sentence", value="the president of the united states",
                     label_visibility="collapsed")

if st.button("Predict next word", type="primary"):
    if not text.strip():
        st.warning("Please type something first.")
    else:
        ids, probs, ms = M["predict"](text)
        st.session_state["last"] = (text, ids, probs, ms)

# ── result + health of that result ─────────────────────────────────────────

if "last" in st.session_state:
    text, ids, probs, ms = st.session_state["last"]
    st.divider()

    if not ids:
        st.error("**The model recognised none of your words.** Every word here is outside "
                 "its 8,978-word vocabulary, so there is nothing to predict from.")
        st.info("Try plain English like *\"i am going to the\"*.")
    else:
        order = np.argsort(probs[: M["out_width"]])[::-1]
        top1, top2 = int(order[0]), int(order[1])
        conf = float(probs[top1])
        margin = conf / max(float(probs[top2]), 1e-9)
        words = text.split()
        coverage = len(ids) / max(1, len(words))
        dropped = max(0, len(ids) - M["width"])
        nz = probs[: M["out_width"]][probs[: M["out_width"]] > 0]
        entropy = float(-(nz * np.log2(nz)).sum())

        st.subheader("Prediction")
        # Large, unmissable display
        st.metric(label="Predicted next word", value=word_for(top1),
                  delta=f"{conf:.1%} confident", delta_color="normal")
        st.caption(f"took {ms:.0f} ms · {len(ids)} of your {len(words)} words recognised")

        st.subheader("Prediction health")
        checks = [
            ("Understood your text",
             "fail" if coverage < 0.5 else "warn" if coverage < 0.85 else "pass",
             f"{coverage:.0%} of words recognised ({len(ids)}/{len(words)})"),
            ("Enough context to predict",
             "warn" if dropped else "pass",
             f"used {min(len(ids), M['width'])} of {M['width']} token slots"
             + (f", dropped {dropped} older words" if dropped else "")),
            ("Confident in the answer",
             "fail" if conf < 0.10 else "warn" if conf < 0.25 else "pass",
             f"{conf:.1%}" + (" — close to guessing" if conf < 0.25 else "")),
            ("Clear winner over runner-up",
             "warn" if margin < 3 else "pass",
             f"{word_for(top1)} beats `{word_for(top2)}` by {margin:.1f}×"),
            ("Distribution is focused",
             "warn" if entropy > 10 else "pass",
             f"entropy {entropy:.2f} bits"),
            ("Fast enough to use",
             "warn" if ms > 400 else "pass",
             f"{ms:.0f} ms"),
        ]
        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
        for name, status, detail in checks:
            st.write(f"{icon[status]} **{name}** — {detail}")

        worst = min(s for _, s, _ in checks)
        if worst == "fail":
            st.error("**This prediction is not trustworthy** — the model did not understand "
                     "your text, or has nothing to go on.")
        elif worst == "warn":
            st.warning("**Treat this prediction as a guess.** The model ran, but the input "
                       "or the output distribution is weak.")
        else:
            st.success("**Healthy prediction** — the model understood the input and was "
                       "confident in its answer.")

        with st.expander("Other candidates"):
            for i in reversed(order[:5]):
                st.progress(float(probs[i]), text=f"{word_for(i)} — {probs[i]:.1%}")

with st.expander("Model details"):
    st.dataframe([{"layer": n, "output shape": s, "params": f"{p:,}"}
                  for n, s, p in M["layers"]],
                 width="stretch", hide_index=True)
    st.write(f"- Weights: {M['weights_mb']:.1f} MB of the {M['file_mb']:.1f} MB file "
             "(the rest is saved optimizer state)\n"
             f"- The final layer predicts {M['out_width']:,} words but the tokenizer only "
             f"emits up to index {max(M['index_word']):,}, so "
             f"{M['out_width'] - max(M['index_word']) - 1:,} outputs are unreachable.\n"
             f"- `max_len.pkl`: {M['max_len']}")