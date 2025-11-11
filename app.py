# app.py — Multimodal Grammar Scorer (Audio → Text → Score)
import os, io, re, math, warnings, tempfile, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa
import joblib
import soundfile as sf

import streamlit as st

MIC_REC_OK = False
try:
    from streamlit_mic_recorder import mic_recorder
    MIC_REC_OK = True
except Exception:
    pass

from pathlib import Path
from typing import Optional, List
# from transformers import DebertaConfig, DebertaModel
# from transformers import DebertaTokenizer
# --- Mic recorder (optional) ---
try:
    from st_audiorec import st_audiorec
    MIC_OK = True
except Exception:
    MIC_OK = False

# --- ASR backends (optional) ---
try:
    from faster_whisper import WhisperModel as FWModel
    FW_OK = True
except Exception:
    FW_OK = False

NEMO_OK = False
try:
    import nemo.collections.asr as nemo_asr
    NEMO_OK = True
except Exception:
    pass

# ------------------ Pipeline hyperparams / paths ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUDIO_SR = 16000
TEXT_MODEL = "microsoft/deberta-v3-small"
MAX_LEN = 256
CLIP_RANGE = (0.0, 5.0)

ARTIFACTS_DIR = Path("artifacts")
TEXT_HEAD_PT = ARTIFACTS_DIR / "text_head.pt"
AUDIO_MLP_PT = ARTIFACTS_DIR / "audio_mlp.pt"
LGBM_TXT     = ARTIFACTS_DIR / "rules_lgbm.txt"
RF_PKL       = ARTIFACTS_DIR / "rules_rf.pkl"
W_VAL_NPY    = ARTIFACTS_DIR / "w_val.npy"
W_RANK_NPY   = ARTIFACTS_DIR / "w_rank.npy"
ISO_PKL      = ARTIFACTS_DIR / "iso.pkl"

# ------------------ Rule features ------------------
DISFLUENCIES = {"uh","um","erm","hmm","you know","like","sort of"}
BANNER_PAT = re.compile(r"(Get .*?in your inbox|Join Medium.*|Read more on Medium.*)", re.I)

def clean_text(s: str) -> str:
    s = BANNER_PAT.sub("", s or "")
    s = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_rule_feats(texts: List[str]) -> pd.DataFrame:
    rows = []
    for t in texts:
        s = t or ""
        toks = s.split()
        tok_n = len(toks)
        chars = len(s)
        avg_tok = (chars / max(1, tok_n))
        commas = s.count(","); periods = s.count("."); qmarks = s.count("?"); exc = s.count("!")
        caps_ratio = sum(ch.isupper() for ch in s) / max(1, len(s))
        repeats = sum(1 for i in range(1, tok_n) if toks[i].lower()==toks[i-1].lower())
        disfluency_hits = sum(1 for w in DISFLUENCIES if w in s.lower())
        rows.append(dict(
            tok_n=tok_n, chars=chars, avg_tok=avg_tok,
            commas=commas, periods=periods, qmarks=qmarks, exclam=exc,
            caps_ratio=caps_ratio, repeats=repeats, disfluencies=disfluency_hits
        ))
    df = pd.DataFrame(rows)
    # enrich
    df["punct_rate"] = (df["commas"]+df["periods"]+df["qmarks"]+df["exclam"]) / np.maximum(1, df["tok_n"])
    df["repeat_rate"] = df["repeats"] / np.maximum(1, df["tok_n"])
    df["disfluency_rate"] = df["disfluencies"] / np.maximum(1, df["tok_n"])
    df["chars_per_tok"] = df["chars"] / np.maximum(1, df["tok_n"])
    df["caps_x_punct"] = df["caps_ratio"] * df["punct_rate"]
    df["avgTok_x_punct"] = df["avg_tok"] * df["punct_rate"]
    return df

# ------------------ Audio featurizer (Whisper-tiny encoder + prosody/pitch) ------------------
import whisper as openai_whisper
class AudioFeaturizer:
    def __init__(self, device="cpu"):
        self.wm = openai_whisper.load_model("tiny", device=device)
        self.device = device

    def _encode(self, wav, sr):
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        wav = openai_whisper.pad_or_trim(torch.tensor(wav))
        mel = openai_whisper.log_mel_spectrogram(wav).to(self.device)
        with torch.no_grad():
            hs = self.wm.encoder(mel.unsqueeze(0))  # [1,T,384]
        return hs.squeeze(0).float().cpu().mean(dim=0)  # [384]

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        wav = librosa.util.normalize(wav)
        wav, _ = librosa.effects.trim(wav, top_db=20)
        dur = len(wav) / AUDIO_SR
        rms = float(librosa.feature.rms(y=wav).mean())
        zcr = float(librosa.feature.zero_crossing_rate(y=wav).mean())
        try:
            f0 = librosa.yin(wav, fmin=80, fmax=400, sr=AUDIO_SR)
            f0 = f0[np.isfinite(f0)]
            f0_mean = float(np.nanmean(f0)) if f0.size else 0.0
            f0_std  = float(np.nanstd(f0))  if f0.size else 0.0
            voiced_ratio = float(np.mean((f0 > 0).astype(float))) if f0.size else 0.0
        except Exception:
            f0_mean = f0_std = voiced_ratio = 0.0
        enc = self._encode(wav, sr)  # [384]
        vec = torch.cat([enc, torch.tensor([dur, rms, zcr, f0_mean, f0_std, voiced_ratio], dtype=torch.float32)], dim=0)  # [390]
        return vec.numpy()

# ------------------ Text model (DeBERTa) + head ------------------
from transformers import AutoTokenizer, AutoModel

from transformers import AutoTokenizer, AutoModel

PREF_MODELS = [
    "microsoft/deberta-v3-small",      # needs sentencepiece
    "roberta-base",                    # no sentencepiece
    "distilbert-base-uncased",         # no sentencepiece
]

class TextRegressor(nn.Module):
    def __init__(self, model_name=None):
        super().__init__()
        self.model_name = model_name or PREF_MODELS[0]
        last_err = None
        for name in ([self.model_name] + [m for m in PREF_MODELS if m != self.model_name]):
            try:
                # Use slow tokenizer for DeBERTa to avoid convert_slow_tokenizer issues
                use_fast = False if "deberta" in name else True
                self.tok = AutoTokenizer.from_pretrained(name, use_fast=use_fast, trust_remote_code=False)
                self.txt = AutoModel.from_pretrained(name, trust_remote_code=False)
                self.model_name = name
                break
            except Exception as e:
                last_err = e
                continue
        if not hasattr(self, "txt"):
            raise RuntimeError(f"Could not load any text backbone. Last error: {last_err}")

        hid = self.txt.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(2*hid, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, texts):
        tokd = self.tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        out = self.txt(**tokd).last_hidden_state
        cls  = out[:, 0]
        mean = out.mean(dim=1)
        emb  = torch.cat([cls, mean], dim=1)
        return self.head(emb).squeeze(-1)



# ------------------ Audio MLP ------------------
class AudioMLP(nn.Module):
    def __init__(self, in_dim=390):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

# ------------------ Meta helpers ------------------
def clip01_5(x): return np.clip(x, CLIP_RANGE[0], CLIP_RANGE[1])
def rank_scale(x):
    idx = np.argsort(np.argsort(x))
    return idx.astype(np.float32) / max(1, len(x)-1)

# ------------------ Cached loaders ------------------
@st.cache_resource(show_spinner=False)
def load_text_model():
    m = TextRegressor().to(DEVICE)
    if TEXT_HEAD_PT.exists():
        m.head.load_state_dict(torch.load(TEXT_HEAD_PT, map_location=DEVICE), strict=True)
        return m, True
    return m, False

@st.cache_resource(show_spinner=False)
def load_audio_model():
    a = AudioMLP().to(DEVICE)
    if AUDIO_MLP_PT.exists():
        a.load_state_dict(torch.load(AUDIO_MLP_PT, map_location=DEVICE), strict=True)
        return a, True
    return a, False

@st.cache_resource(show_spinner=False)
def load_rules_models():
    lgbm = None; rf = None; ok = False
    if LGBM_TXT.exists():
        try:
            import lightgbm as lgb
            lgbm = lgb.Booster(model_file=str(LGBM_TXT)); ok = True
        except Exception as e:
            st.warning(f"Could not load LightGBM: {e}")
    if not ok and RF_PKL.exists():
        try:
            rf = joblib.load(RF_PKL); ok = True
        except Exception as e:
            st.warning(f"Could not load RF: {e}")
    return lgbm, rf, ok

@st.cache_resource(show_spinner=False)
def load_meta():
    w_val = np.load(W_VAL_NPY) if W_VAL_NPY.exists() else None
    w_rank = np.load(W_RANK_NPY) if W_RANK_NPY.exists() else None
    iso = joblib.load(ISO_PKL) if ISO_PKL.exists() else None
    return w_val, w_rank, iso

# ASR models cached
@st.cache_resource(show_spinner=False)
def load_nemo_asr():
    if NEMO_OK:
        try:
            return nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
        except Exception as e:
            st.warning(f"NeMo ASR load failed: {e}")
    return None

@st.cache_resource(show_spinner=False)
def load_fw():
    if FW_OK:
        try:
            return FWModel("tiny", device=DEVICE, compute_type="float16" if DEVICE=="cuda" else "int8")
        except Exception as e:
            st.warning(f"faster-whisper load failed: {e}")
    return None

@st.cache_resource(show_spinner=False)
def load_owhisper():
    try:
        return openai_whisper.load_model("tiny", device=DEVICE)
    except Exception as e:
        st.warning(f"openai-whisper load failed: {e}")
        return None

# ------------------ ASR: prefer NeMo → FW → openai-whisper ------------------
def asr_transcribe(wav: np.ndarray, sr: int, backend: str = "auto") -> str:
    wav = librosa.util.normalize(wav)

    choice = backend
    if backend == "auto":
        choice = "nemo" if NEMO_OK else ("faster_whisper" if FW_OK else "openai_whisper")

    # NeMo (your CSV pipeline)
    if choice == "nemo" and NEMO_OK:
        model = load_nemo_asr()
        if model is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                sf.write(tf.name, wav, sr)
                tf.flush()
                try:
                    result = model.transcribe([tf.name], channel_selector=0)
                    text = (result[0].text or "").strip()
                    if text:
                        return text
                finally:
                    try: os.remove(tf.name)
                    except Exception: pass

    # Faster-Whisper (try VAD on → off)
    if choice in ("faster_whisper", "auto") and FW_OK:
        fw = load_fw()
        if fw is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                sf.write(tf.name, wav, sr)
                tf.flush()
                try:
                    segs, _ = fw.transcribe(tf.name, vad_filter=True, language="en")
                    text = " ".join([s.text for s in segs]).strip()
                    if not text:
                        segs, _ = fw.transcribe(tf.name, vad_filter=False, language="en")
                        text = " ".join([s.text for s in segs]).strip()
                    if text:
                        return text
                finally:
                    try: os.remove(tf.name)
                    except Exception: pass

    # openai-whisper (final fallback)
    ow = load_owhisper()
    if ow is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            sf.write(tf.name, wav, sr)
            tf.flush()
            try:
                res = ow.transcribe(tf.name, language="en")
                text = (res.get("text") or "").strip()
                if text:
                    return text
            finally:
                try: os.remove(tf.name)
                except Exception: pass

    return ""

# ------------------ Scoring (single sample) ------------------
def score_one(text_model, text_ready, audio_mlp, audio_ready, lgbm, rf, rules_ready, w_val, w_rank, iso,
              wav: np.ndarray, sr: int, asr_backend: str = "auto"):
    # 1) ASR
    text = asr_transcribe(wav, sr, backend=asr_backend)
    text_clean = clean_text(text)

    # 2) Features
    rules_df = extract_rule_feats([text_clean])
    rules_vec = rules_df.values.astype(np.float32)
    afe = AudioFeaturizer(device="cpu")
    audio_vec = afe(wav, sr).reshape(1, -1).astype(np.float32)

    # 3) Base predictions
    preds = {}
    if text_ready:
        text_model.eval()
        with torch.no_grad():
            p = text_model([text_clean]).detach().cpu().numpy().ravel()[0]
            preds["text"] = float(p)
    if audio_ready:
        audio_mlp.eval()
        with torch.no_grad():
            p = audio_mlp(torch.tensor(audio_vec, dtype=torch.float32, device=DEVICE)).detach().cpu().numpy().ravel()[0]
            preds["audio"] = float(p)
    if rules_ready:
        if lgbm is not None:
            p = lgbm.predict(rules_vec); preds["rules"] = float(p[0])
        elif rf is not None:
            p = rf.predict(rules_vec);   preds["rules"] = float(p[0])

    # 4) Meta blend
    if (w_val is not None) and (w_rank is not None) and (len(preds) >= 2):
        cols = ["text","audio","rules"]
        present = [c for c in cols if c in preds]
        x = np.array([preds[c] for c in present], dtype=np.float32)[None, :]

        def slice_norm(w_full):
            w = np.array([w_full[i] for i,c in enumerate(cols) if c in present], dtype=np.float32)
            s = w.sum()
            return w/s if s>0 else np.ones_like(w)/len(w)

        y_val  = float((x @ slice_norm(w_val).reshape(-1,1)).ravel()[0])
        y_rank = y_val  # single-sample approx
        y_pred = (1-0.3)*y_val + 0.3*y_rank
    else:
        # heuristic fallback
        order = [("text", 0.5), ("audio", 0.4), ("rules", 0.1)]
        wsum, total = 0.0, 0.0
        for k,w in order:
            if k in preds: total += preds[k]*w; wsum += w
        if wsum == 0:
            base = 3.0
            if rules_df["disfluency_rate"].iloc[0] > 0.02: base -= 0.4
            if rules_df["repeat_rate"].iloc[0] > 0.02:     base -= 0.2
            if rules_df["punct_rate"].iloc[0]   < 0.03:    base -= 0.2
            y_pred = base
        else:
            y_pred = total/wsum

    if iso is not None:
        y_pred = float(iso.predict([y_pred])[0])

    y_pred = float(np.clip(y_pred, *CLIP_RANGE))
    return {
        "transcript": text,
        "text_pred": preds.get("text"),
        "audio_pred": preds.get("audio"),
        "rules_pred": preds.get("rules"),
        "final_pred": y_pred
    }

def get_audio_from_mic() -> Optional[bytes]:
    """
    Try st_audiorec first; if unavailable, try streamlit-mic-recorder.
    Returns WAV bytes or None.
    """
    # A) st_audiorec path
    if MIC_OK:
        st.markdown("**Record with mic**")
        rec = st_audiorec()  # returns WAV bytes or None
        if rec is not None and len(rec) > 0:
            return rec

    # B) streamlit-mic-recorder path (no sample_rate arg supported)
    if MIC_REC_OK:
        st.markdown("**or record with the mic**")
        # Minimal, compatible signature
        data = mic_recorder(
            start_prompt="🎙️ Start recording",
            stop_prompt="⏹️ Stop",
            just_once=False,
            use_container_width=True,
            key="micrec",
        )
        # Handle multiple possible payload shapes
        if data is not None:
            # 1) Newer builds often return dict with raw .wav bytes
            if isinstance(data, dict):
                # Prefer direct WAV bytes if present
                if "bytes" in data and data["bytes"]:
                    return data["bytes"]

                # Some variants return PCM array + sample_rate
                sr = data.get("sample_rate") or data.get("sr")
                arr = data.get("audio") or data.get("array")
                if arr is not None and sr:
                    import numpy as np, io, soundfile as sf
                    buf = io.BytesIO()
                    # ensure float32 mono
                    arr = np.asarray(arr, dtype=np.float32)
                    sf.write(buf, arr, int(sr), format="WAV", subtype="PCM_16")
                    return buf.getvalue()

            # 2) Some versions return raw bytes directly
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)

    return None


# ------------------ UI ------------------
st.set_page_config(page_title="Multimodal Grammar Scorer (Audio → Text → Score)", layout="wide")
st.title("🎤 Multimodal Grammar Scorer (Audio → Text → Score)")
st.caption("Record or upload speech, auto-transcribe to text, and predict grammar quality on [1,5].")

with st.sidebar:
    st.subheader("Settings")
    asr_choice = st.selectbox(
        "ASR backend",
        ["auto", "nemo" if NEMO_OK else "(install NeMo)", "faster_whisper" if FW_OK else "(install faster-whisper)", "openai_whisper"],
        index=0
    )
    st.markdown("**Artifacts in `artifacts/`**: `text_head.pt`, `audio_mlp.pt`, `rules_lgbm.txt`/`rules_rf.pkl`, `w_val.npy`, `w_rank.npy`, `iso.pkl`")
    show_details = st.checkbox("Show per-branch details", value=False)

# Load models / artifacts
text_model, text_ready = load_text_model()
audio_mlp, audio_ready = load_audio_model()
lgbm, rf, rules_ready  = load_rules_models()
w_val, w_rank, iso     = load_meta()

cols = st.columns([1, 1])
with cols[0]:
    st.subheader("1) Provide Audio")
    wav_bytes = None

    # Mic path
    wav_bytes = get_audio_from_mic()
    if wav_bytes:
        st.audio(wav_bytes, format="audio/wav")

    # Upload path
    st.markdown("**…or upload `.wav`**")
    up = st.file_uploader("Upload WAV", type=["wav"], accept_multiple_files=False)
    if up is not None:
        wav_bytes = up.read()
        st.audio(wav_bytes, format="audio/wav")



with cols[1]:
    st.subheader("2) Predict")
    run = st.button("Transcribe & Score", type="primary")

if run:
    if not wav_bytes:
        st.error("Please record or upload a WAV first.")
        st.stop()

    # Decode bytes → numpy wav @ 16k
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tf.write(wav_bytes)
        tf.flush()
        wav, sr = librosa.load(tf.name, sr=AUDIO_SR, mono=True)
    try: os.remove(tf.name)
    except Exception: pass

    details = score_one(
        text_model, text_ready, audio_mlp, audio_ready, lgbm, rf, rules_ready, w_val, w_rank, iso,
        wav, sr, asr_backend=asr_choice
    )

    st.success(f"Predicted Grammar Score: **{details['final_pred']:.2f}**  (1–5)")

    with st.expander("Show transcript", expanded=True):
        if (details["transcript"] or "").strip():
            st.write(details["transcript"])
        else:
            st.info("No transcript available for this audio (ASR returned empty). Try switching ASR backend in the sidebar.")

    if show_details:
        c1,c2,c3 = st.columns(3)
        c1.metric("Text branch",  f"{details['text_pred']:.2f}"  if details["text_pred"]  is not None else "—")
        c2.metric("Audio branch", f"{details['audio_pred']:.2f}" if details["audio_pred"] is not None else "—")
        c3.metric("Rules branch", f"{details['rules_pred']:.2f}" if details["rules_pred"] is not None else "—")

with col2:
    st.subheader("📘 Grammar Score Rubric")
    
    st.markdown("""
    <style>
    .table-container {
        margin-top: 4px;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #ddd;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    thead th {
        background-color: #4f8bf9;
        color: white;
        padding: 6px;
        text-align: center;
    }
    tbody td {
        border: 1px solid #ddd;
        padding: 6px;
        vertical-align: top;
    }
    </style>
    """, unsafe_allow_html=True)

    rubric_html = """
    <div class="table-container">
    <table>
    <thead>
    <tr>
        <th>Score</th>
        <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td><b>1</b></td>
        <td>Struggles with sentence structure; very limited grammar control.</td>
    </tr>
    <tr>
        <td><b>2</b></td>
        <td>Frequent basic grammatical mistakes; incomplete sentences.</td>
    </tr>
    <tr>
        <td><b>3</b></td>
        <td>Decent grasp of structure but noticeable grammar or syntax errors.</td>
    </tr>
    <tr>
        <td><b>4</b></td>
        <td>Strong grammar control; only minor, non-disruptive errors.</td>
    </tr>
    <tr>
        <td><b>5</b></td>
        <td>Highly accurate grammar; handles complex structures well.</td>
    </tr>
    </tbody>
    </table>
    </div>
    """

    st.markdown(rubric_html, unsafe_allow_html=True)

