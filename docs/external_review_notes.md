---
status: current
last_updated: 2026-04-17
---

# 外部レビュー議論からの知見整理 (2026-04-17)

別 LLM との議論から抽出した、プロジェクトに応用可能な知見。
優先度順に整理。

---

## 即座に応用すべき (今のフェーズで関係する)

### 1. KenLM shallow fusion — CTC beam search への統合

**核心**: CTC 推論時に外部 n-gram LM のスコアを加算するだけで精度が底上げできる。
学習パイプラインに触らず、推論時のみで完結。

```
score(y|x) = log P_CTC(y|x) + λ · log P_LM(y)
```

**実装経路**:
- Wikipedia コーパスで KenLM (4-gram) を学習 → 数時間
- C++ サーバー側: KenLM C++ API (`kenlm/lm/model.hh`) を直接統合
- CTC prefix beam search に LM スコア加算を追加 (~100行の修正)
- `alpha` (LM重み) と `beta` (word insertion penalty) を dev セットで調整

**注意**: トークン単位を CTC 出力と揃える必要あり (文字単位 CTC → 文字単位 KenLM)。

**タイミング**: Phase 3 (CTC-NAT 本体) と同時に導入可能。Phase 2 不要。

**ROI**: 高。実装コスト小、精度改善の可能性大。ASR では標準手法で実績豊富。

### 2. 評価フレームワークの先行構築

**核心**: モデル学習前に評価セットと指標を固めておくと、全フェーズが指標ドリブンになる。

**最小構成 (今すぐ作るべき)**:
- `src/eval/metrics.py`: Top-K 文字精度 (Character Accuracy) の計算
- `src/eval/run_eval.py`: 全指標を一括実行するスケルトン
- dev セット: 1000-2000 文 (Wikipedia + 青空文庫から層化サンプリング)
- test セット: 5000-10000 文 (封印、v1.0 リリースまで使わない)
- mozc ベースライン測定

**精度指標** (優先順):
1. 文字精度 Top-1 / Top-5 / Top-10
2. 文節境界精度 (P/R/F1)
3. 同音異義語精度 (サブセット別集計)
4. ドメイン別精度 (日常/技術/文学/ニュース)

**速度指標**:
1. 推論レイテンシ p50/p95/p99 (目標: p95 < 100ms)
2. 入力長依存性 (CTC-NAT の定数時間推論を実証)
3. E2E 遅延 (IPC 込み)

**BLEU は使わない** — IME は短文・単一参照なので文字精度で十分。

### 3. 評価データの汚染防止

**核心**: 学習データと評価データの分離を作品/記事レベルで行う。

- Wikipedia: 記事 ID で分離 (ランダム文分割ではダメ、記事単位)
- 青空文庫: 作品単位で分離 (同一作品の文が学習と評価に跨がらない)
- ゴールドサブセット (100-500文): 人手で読み付与を検証済み、絶対精度の天井測定用

---

## Phase 3 で考慮すべき

### 4. T_in >= T_out 制約の検証

**核心**: CTC は入力長 >= 出力長が必要。かな→漢字は通常 T_in > T_out だが、
出力トークナイザの粒度次第で逆転しうる。

**検証方法**: 学習データの全ペアで `len(reading_tokens) >= len(surface_tokens)` を確認。
違反率が高ければ入力側にパディング or upsampling 層を追加。

**既に ROADMAP のリスクテーブルに記載済み** だが、実データでの違反率を数値化すべき。

### 5. 同音異義語がCTC-NATの誤り集中点

**核心**: CTC の conditional independence 仮定は同音異義語に弱い。
「こうしょう」→「交渉/考証/高尚」の選択は文脈依存で、CTC の各位置独立出力と相性が悪い。

**対策** (Phase 3):
- Mask-CTC refinement が第一防衛線 (低信頼位置を再予測)
- KenLM shallow fusion が第二防衛線 (n-gram 文脈で曖昧さ解消)
- それでも不足なら (d) ハイブリッド案: 高信頼は CTC、低信頼時のみ AR で再ランキング

### 6. CTC-NAT の現実的な目標設定

**核心**: 「精度は同等、速度で大きく勝つ」が現実的目標。精度で zenz-v1 を超えるのは構造的に難しい可能性がある。

**撤退判断基準** (Phase 3 Go/No-Go):
- (a) 速度優先で CTC-NAT 採用: 精度差 1-2 ポイント以内なら
- (b) AR + 投機的デコードに撤退: 精度差 3+ ポイントなら
- (c) Imputer 的反復 CTC: 精度差 2-3 ポイントで速度に余裕があれば
- (d) ハイブリッド (CTC + AR 再ランキング): 実装複雑だが最高精度

---

## 候補ランキングパイプライン (Phase 4-5 で実装)

### 7. ランキングの5層分離

モデル内部の N-best 生成とは独立に、以下を**別機構**で扱う:

1. **CTC beam search + KenLM**: モデル側 (Phase 3)
2. **ユーザ学習 (頻度・最近性)**: SQLite、指数減衰
3. **ユーザ辞書**: ハード制約 (辞書にあれば必ずトップ)
4. **文脈依存適応**: v2 送り (最初はアプリ別学習データ分離のみ)
5. **CVAE パーソナライゼーション**: v2 送り

**最終スコア**:
```
final_score(y) = model_score(y) + α·user_freq(y) + β·recency(y)
```

**SQLite スキーマ** (提案通り):
```sql
CREATE TABLE user_history (
    reading TEXT, context_hash INTEGER, candidate TEXT,
    count INTEGER DEFAULT 1, last_used TIMESTAMP,
    PRIMARY KEY (reading, context_hash, candidate)
);
CREATE TABLE user_dict (
    reading TEXT, candidate TEXT, priority INTEGER DEFAULT 0,
    PRIMARY KEY (reading, candidate)
);
```

### 8. アプリ別学習データ分離

fcitx5 はアプリ名を取得可能。アプリごとに別のユーザ学習データを持つ。
工数小・リターン大の可能性。v1.0 に入れる価値あり。

---

## データ品質の追加改善 (次サイクルで検討)

### 9. 青空文庫のルビ情報活用

《》で囲まれた読み情報は MeCab 読み付与の**検証用教師信号**として使える。
現在の process_aozora.py は形態素解析済み CSV を使っているが、
原文からルビを抽出して読み付与の正解率を測定できる。

### 10. 文長分布の補正

IME 実入力は短文節 (5-30文字) が典型。学習データは長文偏り。
**文内ランダム切り出し**でデータ拡張する案。Phase 2 で検討。

### 11. n-gram 重複除去

Wikipedia 内のテンプレート由来の定型句を 6-gram 完全一致で検出・除去。
モデルの定型句への過適合防止。

---

## 確認済み・変更不要

- **1.58-bit を外す判断**: 正しい。int8 で十分な可能性高い。
- **CVAE を v2 送り**: 正しい。v1.0 の経験がないと設計できない。
- **撤退経路 (AR + 投機的デコード)**: 維持。Phase 2 のベースラインが安全弁。
