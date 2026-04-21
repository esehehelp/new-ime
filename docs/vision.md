---
status: current
last_updated: 2026-04-21
---

# new-ime Vision

## 目的

`new-ime` の目的は、**CPU で実用になる neural IME の中核技術を作ること**です。完成品の IME を広く配ることよりも、次を明確にすることを優先します。

- どのモデル系が日本語かな漢字変換に向いているか
- どの評価条件を canonical とするか
- どこまでが C++ 側の責務で、どこからを UI 統合側に委ねるか

## 現在の判断

### 1. 主力モデルは CTC-NAT 30M 系

現時点での基準線は `ctc-nat-30m-student` step160k です。

理由は単純です。

- 30M 帯でも精度が実用域に近い
- KenLM 併用で同音異義語と文脈依存候補が大きく改善する
- CPU でのレイテンシが小さい
- AR 系よりも per-input のコストが読みやすい

この repo は当面、この系統を土台に比較と改善を進めます。

### 2. 比較対象は zenz と jinen

現実的な比較対象は次です。

- `zenz-v2.5-xsmall`
- `zenz-v2.5-small`
- `zenz-v3.1-small`
- `jinen-v1-xsmall`
- `jinen-v1-small`

この比較で見たいのは「最高精度」だけではありません。

- 同サイズ帯でどこまで迫れるか
- CPU レイテンシがどれだけ違うか
- KenLM や ONNX int8 を含めた運用形でどう見えるか

## スコープ

### In scope

- CTC-NAT 系モデルの改善
- KenLM shallow fusion
- ONNX / int8 を含む CPU 推論
- CLI と benchmark を中心にした検証
- Rust ベースの Windows 統合

## アーキテクチャ方針

### 推論の中心

現在の基本形は次です。

1. かな入力を受ける
2. CTC-NAT で候補を出す
3. beam search で列挙する
4. KenLM で再順位付けする
5. top-k 候補を返す

この構成は、短い入力単位が多い IME ではかなり相性が良いと見ています。

### C++ 側の責務

C++ 側には次を残します。

- 推論エンジン
- KenLM 連携
- CLI / FFI
- composing / preedit の基本ロジック

逆に、Windows の UI 統合を C++ TSF で抱え込む方針は捨てます。ここは不安定性が高く、検証コストに対して得られる前進が小さいためです。ただし，Rustで再実装する可能性があります．

### Windows 統合の方針

Windows では **Rust TSF を正系** とします。

- C++ TSF は削除する
- engine は C++ / Rust のどちらからでも使える形に寄せる
- UI 統合は Rust 側で安全性を高めながら進める

## データ戦略

この repo の価値は、モデルだけでなくデータの持ち方にもあります。

- `probe_v3` を phrase-level canonical bench とする
- `AJIMEE JWTD_v2` を外部比較用の固定ベンチとする
- 学習データは広く持つが、評価データ汚染は強く避ける

短期では、巨大化よりも **edge / names / numeric** の改善に効く追加データや LM の見直しを優先します。

## 近い将来にやること

### 1. 30M 系の底上げ

- edge category の改善
- names / numeric の再学習
- KenLM-MoE の安定化
- ONNX int8 の production 条件確定

### 2. benchmark の一本化

比較条件は [benchmark_comparison.md](D:/Dev/new-ime/docs/benchmark_comparison.md) を唯一の基準とし、そこから外れた測定は参考値として扱います。

### 3. Windows 統合の整理

- C++ TSF を完全削除
- Rust TSF に一本化
- C++ は engine / demo / FFI に集中

## 何を成功とみなすか

当面の成功条件は次です。

- `ctc-nat-30m-student` 系が canonical bench で安定して再現できる
- 30M 帯で `zenz-v2.5-xsmall` に近い精度を保ちながら、CPU レイテンシで優位を維持する
- Windows 統合が C++ TSF に依存しない形に整理される

この 3 点が揃えば、この repo は「実験の寄せ集め」ではなく、明確な研究線を持った IME 基盤になります。
