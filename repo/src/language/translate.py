"""STEP 24: Minimal downstream demonstration (spec section 25).

The recognition model (src/models + a frozen checkpoint) decides the
sign. This module only maps an ACCEPTED recognized class to a canonical
text string and optionally speaks it. It has no authority over what was
recognized and performs no smoothing/correction of the recognition
output -- that would blur the line the spec explicitly draws.

canonical_text_map.json is built from data/processed/vocabulary.csv's
labels the first time this module is used; edit it by hand for any
glosses that need a more natural display string than the raw label.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DEFAULT_MAP_PATH = Path("data/processed/canonical_text_map.json")


def build_default_map(vocabulary_csv: str = "data/processed/vocabulary.csv",
                       out_path: Path = DEFAULT_MAP_PATH) -> dict:
    vocab = pd.read_csv(vocabulary_csv)
    selected = vocab.loc[vocab["selected"], "label"].tolist()
    mapping = {label: str(label).replace("_", " ") for label in selected}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
    return mapping


def recognized_sign_to_text(recognized_class: str, map_path: Path = DEFAULT_MAP_PATH) -> str:
    if not map_path.exists():
        build_default_map(out_path=map_path)
    mapping = json.loads(map_path.read_text())
    return mapping.get(recognized_class, recognized_class)


def speak(text: str) -> None:
    """Optional TTS. Fails silently (prints instead) if pyttsx3 / a local
    speech engine is unavailable -- this is a demo convenience, not part
    of the research contribution, per spec section 25.
    """
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:  # noqa: BLE001
        print(f"[translate] TTS unavailable ({e}); text output only: {text}")
