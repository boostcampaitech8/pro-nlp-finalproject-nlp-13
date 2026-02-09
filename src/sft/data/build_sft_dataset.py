from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class DatasetBuildConfig:
    enabled: bool = False

    train_root: str = ""
    val_root: str = ""
    topic: str = "여행, 관광 및 명소"

    k_turn: int = 3
    max_context_chars: int = 1200
    seed: int = 42

    ratio_core: float = 0.70
    ratio_chat: float = 0.20
    ratio_ask: float = 0.10

    min_core_chars: int = 25
    min_chat_chars: int = 8
    min_ask_chars: int = 8
    min_other_chars: int = 8

    out_dir: str = "data/persona_data"
    out_prefix: str = "sft_reqinfo_mix"


def is_missing(value) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "null", "none", "nan"}


def build_context_from_pairs(
    pairs: List[Dict],
    idx: int,
    k_turn: int = 3,
    max_chars: int = 1200,
) -> str:
    # Keep previous completed pairs + current user turn.
    lines: List[str] = []
    start = max(0, idx - k_turn)
    for j in range(start, idx):
        lines.append(f"user: {pairs[j]['a_text']}")
        lines.append(f"assistant: {pairs[j]['b_text']}")
    lines.append(f"user: {pairs[idx]['a_text']}")
    context = "\n".join(lines)
    if len(context) > max_chars:
        context = context[-max_chars:]
    return context


def extract_ab_pairs(dialog_obj: Dict, topic: str) -> List[Dict]:
    info = dialog_obj.get("info", {})
    speaker_info = info.get("speaker", {})

    if info.get("topic") != topic:
        return []
    if not is_missing(speaker_info.get("speakerCId")):
        return []

    utts: List[Dict] = []
    for i, utter in enumerate(dialog_obj.get("utterances", [])):
        speaker = utter.get("speaker")
        if speaker not in {"speakerA", "speakerB"}:
            continue
        utts.append(
            {
                "speaker": speaker,
                "text": (utter.get("text") or "").strip(),
                "speech_act": (utter.get("speech_act") or "").strip(),
                "slot_cnt": len(utter.get("slot") or []),
                "turn_id": utter.get("turn_id"),
                "utterance_id": utter.get("utterance_id"),
                "utt_order": i,
            }
        )

    speakers = {u["speaker"] for u in utts}
    if speakers != {"speakerA", "speakerB"}:
        return []

    pairs: List[Dict] = []
    i = 0
    while i < len(utts) - 1:
        a = utts[i]
        b = utts[i + 1]

        if a["speaker"] == "speakerA" and b["speaker"] == "speakerB":
            same_turn = True
            if a.get("turn_id") and b.get("turn_id"):
                same_turn = a["turn_id"] == b["turn_id"]
            if same_turn and a["text"] and b["text"]:
                pairs.append(
                    {
                        "dialog_id": info.get("id"),
                        "keyword": info.get("keyword"),
                        "a_text": a["text"],
                        "a_speech_act": a["speech_act"],
                        "a_slot_cnt": a["slot_cnt"],
                        "a_turn_id": a["turn_id"],
                        "a_utterance_id": a["utterance_id"],
                        "b_text": b["text"],
                        "b_speech_act": b["speech_act"],
                        "b_slot_cnt": b["slot_cnt"],
                        "b_turn_id": b["turn_id"],
                        "b_utterance_id": b["utterance_id"],
                    }
                )
                i += 2
                continue
        i += 1

    return pairs


def collect_pair_samples(
    roots: Dict[str, Path],
    topic: str,
    k_turn: int,
    max_context_chars: int,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for split, root in roots.items():
        for fp in root.rglob("*.json"):
            with fp.open("r", encoding="utf-8") as f:
                dialog = json.load(f)

            pairs = extract_ab_pairs(dialog, topic=topic)
            if not pairs:
                continue

            for idx, p in enumerate(pairs):
                row = dict(p)
                row["split"] = split
                row["pair_idx"] = idx
                row["context"] = build_context_from_pairs(
                    pairs,
                    idx,
                    k_turn=k_turn,
                    max_chars=max_context_chars,
                )
                row["target"] = p["b_text"]
                row["target_len"] = len(p["b_text"])
                rows.append(row)

    return pd.DataFrame(rows)


def add_bucket(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["bucket"] = "other"
    x.loc[
        (x["a_speech_act"] == "정보 요청") & (x["b_speech_act"] == "정보 제공"),
        "bucket",
    ] = "core_req_info"
    x.loc[x["b_speech_act"] == "친교 및 잡담", "bucket"] = "chat"
    x.loc[x["b_speech_act"] == "정보 요청", "bucket"] = "ask_back"
    return x


def sample_by_ratio(
    df: pd.DataFrame,
    ratio: Dict[str, float],
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    available = df["bucket"].value_counts().to_dict()
    use = {k: v for k, v in ratio.items() if available.get(k, 0) > 0}
    if not use:
        return df.iloc[0:0].copy(), {}, available

    ratio_sum = sum(use.values())
    use = {k: v / ratio_sum for k, v in use.items()}

    scale = min(available[k] / use[k] for k in use)
    quota = {k: int(scale * use[k]) for k in use}

    parts: List[pd.DataFrame] = []
    for bucket, n in quota.items():
        if n <= 0:
            continue
        group = df[df["bucket"] == bucket]
        parts.append(group.sample(n=n, random_state=seed, replace=False))

    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out, quota, available


def make_ratio_mixed_dataset(
    pair_df: pd.DataFrame,
    ratio: Dict[str, float],
    min_chars_by_bucket: Dict[str, int],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    x = add_bucket(pair_df)

    keep = pd.Series(True, index=x.index)
    for bucket, min_len in min_chars_by_bucket.items():
        keep &= ~((x["bucket"] == bucket) & (x["target_len"] < min_len))
    x = x[keep].copy()

    train = x[x["split"] == "train"].copy()
    val = x[x["split"] == "validation"].copy()

    train_mix, train_quota, train_available = sample_by_ratio(train, ratio, seed=seed)
    val_mix, val_quota, val_available = sample_by_ratio(val, ratio, seed=seed)

    debug = {
        "train_available": train_available,
        "train_quota": train_quota,
        "val_available": val_available,
        "val_quota": val_quota,
    }
    return train_mix, val_mix, debug


def to_sft_messages_df(df: pd.DataFrame, start_id: int = 0) -> pd.DataFrame:
    rows: List[Dict] = []
    for i, row in enumerate(df.itertuples(index=False), start=start_id):
        messages = [
            {"role": "user", "content": row.context},
            {"role": "assistant", "content": row.target},
        ]
        rows.append(
            {
                "conversation_id": i,
                "messages": messages,
                "split": row.split,
                "bucket": row.bucket,
                "dialog_id": row.dialog_id,
                "pair_idx": row.pair_idx,
                "target_len": row.target_len,
            }
        )
    return pd.DataFrame(rows)


def run_build(cfg: DatasetBuildConfig) -> Dict[str, object]:
    roots = {
        "train": Path(cfg.train_root),
        "validation": Path(cfg.val_root),
    }
    for split, root in roots.items():
        if not root.exists():
            raise FileNotFoundError(f"{split} root not found: {root}")

    pair_df = collect_pair_samples(
        roots=roots,
        topic=cfg.topic,
        k_turn=cfg.k_turn,
        max_context_chars=cfg.max_context_chars,
    )
    if pair_df.empty:
        raise RuntimeError("No pair samples were extracted. Check root paths and filters.")

    ratio = {
        "core_req_info": cfg.ratio_core,
        "chat": cfg.ratio_chat,
        "ask_back": cfg.ratio_ask,
    }
    min_chars_by_bucket = {
        "core_req_info": cfg.min_core_chars,
        "chat": cfg.min_chat_chars,
        "ask_back": cfg.min_ask_chars,
        "other": cfg.min_other_chars,
    }

    train_mix, val_mix, debug = make_ratio_mixed_dataset(
        pair_df=pair_df,
        ratio=ratio,
        min_chars_by_bucket=min_chars_by_bucket,
        seed=cfg.seed,
    )

    train_sft = to_sft_messages_df(train_mix, start_id=0)
    val_sft = to_sft_messages_df(val_mix, start_id=len(train_sft))
    full_sft = pd.concat([train_sft, val_sft], axis=0).reset_index(drop=True)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = out_dir / f"{cfg.out_prefix}_train.csv"
    val_csv = out_dir / f"{cfg.out_prefix}_val.csv"
    full_csv = out_dir / f"{cfg.out_prefix}.csv"
    pair_csv = out_dir / f"{cfg.out_prefix}_pairs.csv"
    debug_json = out_dir / f"{cfg.out_prefix}_debug.json"

    train_sft.to_csv(train_csv, index=False, encoding="utf-8-sig")
    val_sft.to_csv(val_csv, index=False, encoding="utf-8-sig")
    full_sft.to_csv(full_csv, index=False, encoding="utf-8-sig")
    pair_df.to_csv(pair_csv, index=False, encoding="utf-8-sig")
    debug_json.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "full_csv": str(full_csv),
        "pair_csv": str(pair_csv),
        "debug_json": str(debug_json),
        "pair_shape": tuple(pair_df.shape),
        "train_mix_shape": tuple(train_mix.shape),
        "val_mix_shape": tuple(val_mix.shape),
        "debug": debug,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT dataset from AIHub SNS JSON files.")
    parser.add_argument("--train-root", type=str, required=True)
    parser.add_argument("--val-root", type=str, required=True)
    parser.add_argument("--topic", type=str, default="여행, 관광 및 명소")
    parser.add_argument("--k-turn", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--ratio-core", type=float, default=0.70)
    parser.add_argument("--ratio-chat", type=float, default=0.20)
    parser.add_argument("--ratio-ask", type=float, default=0.10)

    parser.add_argument("--min-core-chars", type=int, default=25)
    parser.add_argument("--min-chat-chars", type=int, default=8)
    parser.add_argument("--min-ask-chars", type=int, default=8)
    parser.add_argument("--min-other-chars", type=int, default=8)

    parser.add_argument("--out-dir", type=str, default="data/persona_data")
    parser.add_argument("--out-prefix", type=str, default="sft_reqinfo_mix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DatasetBuildConfig(
        enabled=True,
        train_root=args.train_root,
        val_root=args.val_root,
        topic=args.topic,
        k_turn=args.k_turn,
        max_context_chars=args.max_context_chars,
        seed=args.seed,
        ratio_core=args.ratio_core,
        ratio_chat=args.ratio_chat,
        ratio_ask=args.ratio_ask,
        min_core_chars=args.min_core_chars,
        min_chat_chars=args.min_chat_chars,
        min_ask_chars=args.min_ask_chars,
        min_other_chars=args.min_other_chars,
        out_dir=args.out_dir,
        out_prefix=args.out_prefix,
    )

    result = run_build(cfg)

    print("[Done] Saved files:")
    print(f"- {result['train_csv']}")
    print(f"- {result['val_csv']}")
    print(f"- {result['full_csv']}")
    print(f"- {result['pair_csv']}")
    print(f"- {result['debug_json']}")
    print()
    print("[Stats]")
    print(f"pair_df: {result['pair_shape']}")
    print(f"train_mix: {result['train_mix_shape']}")
    print(f"val_mix: {result['val_mix_shape']}")


if __name__ == "__main__":
    main()
