---
status: current
last_updated: 2026-04-18
---

# ベンチマーク比較結果 (2026-04-18)

## 全9モデル × 3ベンチ

| Model | Params | manual EM | ajimee EM | eval_v3 EM | 最遅p50 |
|-------|--------|-----------|-----------|------------|---------|
| zenz-v2.5-medium greedy | 310M | 0.900 | 0.787 | 0.412 | 1853ms |
| zenz-v2.5-small greedy | 91M | 0.890 | 0.750 | 0.375 | 581ms |
| zenz-v2.5-xsmall greedy | 26M | 0.880 | 0.588 | 0.312 | 242ms |
| ar_v3_vast greedy | 32M | 0.800 | 0.450 | 0.412 | 191ms |
| ar_v3_vast beam10 | 32M | 0.800 | 0.450 | 0.450 | 1082ms |
| ar_v3_local greedy | 32M | 0.780 | 0.400 | 0.325 | 233ms |
| ar_v3_local beam10 | 32M | 0.790 | 0.412 | 0.350 | 1095ms |
| ar_v3_chunks greedy | 30M | 0.590 | 0.212 | 0.013 | 132ms |
| ar_v3_chunks beam10 | 30M | 0.590 | 0.188 | 0.013 | 1086ms |

## 同規模対決: ar_v3_vast (32M) vs zenz-xsmall (26M)

| ベンチ | ar_v3_vast EM | zenz-xsmall EM | 勝者 |
|--------|--------------|----------------|------|
| manual_test | 0.800 | 0.880 | zenz +8pt |
| ajimee_jwtd | 0.450 | 0.588 | zenz +14pt |
| eval_v3_dev | **0.412** | 0.312 | **自前 +10pt** |

## 重要な知見

1. **自前分布 (eval_v3) では 310M zenz-medium と互角** (EM 0.412 = 同値)
2. **汎化ベンチ (AJIMEE) では xsmall にも負ける** → データ多様性不足
3. **zenz 系列のスケーリング**: manual はサチ (0.88-0.90), AJIMEE は明確にスケール
4. **CTC-NAT で速度優位を取る戦略は有効**: zenz-xsmall 75-242ms に対し CTC-NAT 10-30ms を狙える
5. **beam search は EM を改善しない** (greedy とほぼ同等)、ただし CharAcc は微改善

## Phase 3 への示唆

- 精度で zenz-xsmall を超えるには: データ多様性 + モデル規模拡大
- 速度で差別化するには: CTC-NAT 並列生成
- 200M CTC-NAT 1.58-bit なら: zenz-small (91M) に精度で並び、速度で 10倍��上高速
