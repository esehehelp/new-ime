# Benchmark

## 実行

```bash
cd /mnt/d/Dev/new-ime
PYTHONPATH=src uv run python -m new_ime.cli.bench
```

`configs/bench/*.toml` がすべて実行される。絞る場合だけ `-m` / `-t` を使う。

```bash
PYTHONPATH=src uv run python -m new_ime.cli.bench -m suiko-v1-small-kenlm-6gram-q8-moe -t probe_v3
```

主判断は `probe_v3`。`ajimee_jwtd` は n=200 のため補助指標として見る。

## 比較対象

| config | type | source |
|---|---|---|
| `suiko-v1-small-kenlm-6gram-q8-moe` | ctc-nat + KenLM MoE | checkpoint + general/tech 6-gram q8 + entity 4-gram (alpha=0.2 beta=0.6) |
| `zenz-v2.5-xsmall` | zenz-v2.5 | `references/zenz-v2.5-xsmall/` |
| `zenz-v2.5-small` | zenz-v2.5 | `references/zenz-v2.5-small/` |
| `zenz-v2.5-medium` | zenz-v2.5 | `references/zenz-v2.5-medium/` |
| `zenz-v3.1-small` | zenz-v3.1 | `references/zenz-v3.1-small/` |
| `jinen-v1-xsmall` | jinen-v1 | HF `togatogah/jinen-v1-xsmall` |
| `jinen-v1-small` | jinen-v1 | HF `togatogah/jinen-v1-small` |

## 結果

probe_v3:

| config | EM1 | EM5 | CharAcc | p50 ms |
|---|---:|---:|---:|---:|
| `zenz-v2.5-medium` | 0.7470 | 0.8760 | 0.9660 | 1173 |
| `zenz-v3.1-small` | 0.7180 | 0.8560 | 0.9590 | 417 |
| `zenz-v2.5-small` | 0.7130 | 0.8480 | 0.9590 | 376 |
| `zenz-v2.5-xsmall` | 0.6950 | 0.8130 | 0.9530 | 118 |
| `suiko-v1-small-kenlm-6gram-q8-moe` | 0.6782 | 0.7874 | 0.9495 | 14.0 |
| `jinen-v1-small` | 0.6720 | 0.7760 | 0.9440 | 278 |
| `jinen-v1-xsmall` | 0.6090 | 0.7470 | 0.9290 | 115 |

ajimee_jwtd:

| config | EM1 | EM5 | CharAcc | p50 ms |
|---|---:|---:|---:|---:|
| `zenz-v2.5-medium` | 0.8750 | 0.9700 | 0.9820 | 1361 |
| `zenz-v3.1-small` | 0.8600 | 0.9300 | 0.9830 | 470 |
| `zenz-v2.5-small` | 0.8400 | 0.9550 | 0.9770 | 418 |
| `zenz-v2.5-xsmall` | 0.6950 | 0.8450 | 0.9530 | 139 |
| `suiko-v1-small-kenlm-6gram-q8-moe` | 0.6800 | 0.8150 | 0.9563 | 17.7 |
| `jinen-v1-small` | 0.6550 | 0.8350 | 0.9520 | 309 |
| `jinen-v1-xsmall` | 0.3950 | 0.5250 | 0.9170 | 124 |
