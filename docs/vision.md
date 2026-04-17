# new-ime: 最善構成ビジョン (工数度外視)

## 設計原則

- タスクの本質に合致した構造を選ぶ
- 精度・速度・サイズ・適応性の全てで最高点
- 将来拡張性を持たせる

## アーキテクチャ: CTC-NAT Encoder-Decoder

| コンポーネント | 構成 |
|---------------|------|
| Encoder | Transformer 12層, hidden 512, BERT-japanese-char 初期化 |
| Decoder | NAT Transformer 8層, hidden 512, 双方向 self-attention |
| CTC Head | 並列出力 + Mask-CTC iterative refinement (2-3回) |
| 合計 | 40-60M params |

### CTC-NAT を選ぶ理由

- 日本語読み→表記は単調アラインメント → CTC の前提と合致
- 並列生成でレイテンシが入力長に依存しない
- AR の構造崩壊 (「人口のうの」) が原理的に発生しない
- Beam search の exposure bias 問題がない
- Phase 2 の実験で beam 内に 95% 正解があることを確認済み

## 量子化: 1.58-bit QAT

### 段階的移行

1. **fp16 完全学習** → 安定精度確保、KD 教師として使用
2. **Continual QAT (fp16→1.58-bit)** → 中央値スケーリング (Nielsen 2024)
3. **Activation 8-bit 量子化** → weight 1.58-bit + activation 8-bit 混合

### 最終サイズ

- 40-60M × 1.58-bit ≈ **10-15MB** (モデル本体)
- Activation + KV cache 込みで **30-40MB** (実メモリ)

### 根拠

- Nielsen 2024 "BitNet b1.58 Reloaded": 100K-48M で検証済み
- encoder-decoder (BERT/T5系) でも検証済み
- 暗黙的正則化効果で fp16 を超える場合もあり

## ユーザ適応: 階層的 CVAE

### 潜在変数設計

```
z = (z_writer[32], z_domain[16], z_session[16]) = 64次元
```

- **z_writer**: 書き手の文体・語彙選好 (ユーザ固有、オンライン更新)
- **z_domain**: ビジネス/カジュアル/技術 (アプリ別に推定)
- **z_session**: 直近入力の傾向 (セッション内で動的更新)

### 推論時の適応

- ユーザが数十〜数百文入力 → z_writer をローカルでオンライン更新
- モデル本体は固定、z のみ更新 → プライバシー保護
- データは端末外に出ない

## データ戦略

### 目標: 100-200億トークン

| ソース | 推定規模 | 特徴 |
|--------|---------|------|
| Wikipedia v3 | 2012万文 | 百科事典 |
| 青空文庫 | 242万文 | 文学・文体多様性 |
| Swallow Corpus v3 | 数億トークン | 大規模 Web コーパス |
| 日本語日常対話コーパス | 口語補強 | 会話体 |
| おーぷん2ちゃんねる | ネット口語 | 非標準表記含む |
| 国会議事録 | 公文書 | フォーマル |
| Qiita/Zenn 抜粋 | 技術文書 | IT用語 |
| Livedoor + ニュース | ニュース | 報道体 |
| チャンクデータ | 1億+ | 文節単位の短文 |
| ゴールドデータ | 10-20K | 人手検証済み |

### 読み付与の精緻化

- UniDic features[17] (仮名形出現形) を基本
- NEologd 併用でクロスバリデーション
- 青空文庫のルビ情報を教師信号に
- 数万文の人手検証サブセット

## 推論パイプライン

```
ひらがな入力 + 左文脈 + z (writer/domain/session)
    │
    ▼ Encoder (並列, ~5ms)
    │
    ▼ Decoder (NAT並列, ~5ms)
    │
    ▼ CTC beam search (beam=20) + KenLM shallow fusion
    │
    ▼ Mask-CTC refinement (2-3回, ~5ms each)
    │
    ▼ ユーザ辞書ハード制約 + ユーザ学習スコア加算
    │
    最終 top-K 候補 (K=10)
```

### 目標速度

| 入力長 | 目標レイテンシ |
|--------|-------------|
| 短文 (5文字) | < 20ms |
| 中文 (15文字) | < 30ms |
| 長文 (30文字) | < 50ms |

## 配布プラットフォーム

| プラットフォーム | 方式 | 状態 |
|----------------|------|------|
| Linux (fcitx5) | エンジンプラグイン | 設計済み |
| Windows (TSF) | mozc フォーク + DLL 差し替え | **DLL 動作確認済み** |
| macOS (IMKit) | 将来拡張 | |
| モバイル (iOS/Android) | 将来拡張 | |

## 学習ロードマップ

### Phase 3a: CTC-NAT fp16

1. BERT encoder 初期化 + NAT decoder スクラッチ
2. CTC loss + GLAT 学習
3. KD from Phase 2 AR model
4. Mask-CTC refinement 追加
5. KenLM shallow fusion 統合

### Phase 3b: CVAE 統合

1. z_writer/z_domain/z_session 導入
2. 書き手別データで VAE 学習
3. posterior collapse 対策 (KL annealing)

### Phase 3c: 1.58-bit QAT

1. fp16 → 1.58-bit Continual QAT
2. 中央値スケーリング
3. bitnet.cpp ベースの推論カーネル
4. int8 fallback 用意

## 目標精度

| ベンチ | Phase 2 (32M AR) | Phase 3 目標 |
|--------|------------------|-------------|
| manual_test EM | 0.800 | 0.900+ |
| AJIMEE EM | 0.450 | 0.800+ |
| gold_1k EM | 0.660 | 0.900+ |
| eval_v3 EM | 0.412 | 0.500+ |

## 撤退経路

| リスク | 撤退先 |
|--------|--------|
| CTC-NAT 精度不足 | AR + 投機的デコード |
| 1.58-bit 品質劣化 | int8 量子化 (30-40MB) |
| CVAE posterior collapse | 条件付きモデル (非変分版) |
| KenLM 効果薄 | ユーザ辞書強化のみ |

## この構成の独自価値

既存 IME (mozc, zenz, Google, ATOK) のどれも持っていない組み合わせ:

- **小型だが高性能** (zenz-v2.5-small 水準)
- **極小サイズ** (1.58-bit で 15MB)
- **ユーザ適応可能** (CVAE)
- **並列生成で高速** (CTC-NAT)
- **多ドメイン対応** (100-200億トークン)
- **プライバシー保護** (ローカル適応)
