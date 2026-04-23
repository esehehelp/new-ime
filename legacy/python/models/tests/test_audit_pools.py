import json

from scripts.audit_pools import audit_pool, build_eval_indices


def test_audit_pool_basic(tmp_path):
    pool_path = tmp_path / "pool.jsonl"
    eval_path = tmp_path / "eval.jsonl"

    pool_rows = [
        {"reading": "かな", "surface": "仮名", "context": "前文"},
        {"reading": "へんかん", "surface": "変換", "context": ""},
    ]
    eval_rows = [
        {"reading": "かな", "surface": "仮名", "context": "前文"},
    ]

    pool_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in pool_rows),
        encoding="utf-8",
    )
    eval_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in eval_rows),
        encoding="utf-8",
    )

    lexical, sixgrams = build_eval_indices([str(eval_path)])
    stats = audit_pool(str(pool_path), lexical, sixgrams)

    assert stats.lines == 2
    assert stats.lexical_overlap == 1
    assert stats.kanji_surface_ratio > 0
