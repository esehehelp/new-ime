# Third-Party License Notices

本リポジトリのコードは [MIT](LICENSE)。モデル/データ成果物は [CC BY-SA 4.0](MODEL_LICENSE)。
加えて、以下の 3rd-party ライブラリに依存しており、それぞれのライセンス義務が
別途発生する。

## KenLM (LGPL 2.1)

- リポジトリ: https://github.com/kpu/kenlm
- ライセンス: **GNU Lesser General Public License 2.1**
- 本リポジトリでの扱い:
  - `engine/server/third_party/kenlm/` は `.gitignore` 済、**GitHub にソースは
    含まれない**。user が `engine/server/third_party/setup_kenlm.sh` 経由で
    clone する
  - `engine/server/src/lm_scorer_kenlm.{h,cpp}` は KenLM API を呼び出す
    **ラッパーコード** (本リポジトリの MIT コード、KenLM 本体には含めない)
  - `engine/win32/interactive_ctc.cpp` は KenLM を **静的リンク** してビルド
    (`build_ctc.bat` 経由)

### LGPL 義務の発生タイミング

| 使用形態 | 義務 |
|---|---|
| Local でビルド・実行のみ | **義務なし** (自分用の linked binary) |
| KenLM を含まない source 配布 (GitHub 現状) | **義務なし** |
| 静的リンクされた binary を公開配布 | **義務あり**: LGPL notice 同梱 + 差替えを可能にするため object file or source 提供 |
| 動的リンクされた binary + kenlm.dll 別途配布 | **義務軽減**: LGPL notice 同梱のみ、user は kenlm.dll を入れ替え可能 |

### 現在の運用

- **research/local demo 用途**: LGPL 義務は発生していない
- **将来の配布** (例: `interactive_ctc.exe` の release) では下記のどちらかを選択:
  1. 動的リンク (kenlm.dll 別ファイル) + LGPL notice → 最も簡単
  2. 静的リンク + object file 同梱 + LGPL notice → Windows では面倒
  3. KenLM を自作 4-gram (MIT) に置換 → プロプライエタリ配布したい場合

### 代替案

将来 LGPL を避けたい場合の候補:

- 自作 4-gram scorer (datacore Rust に実装、~2-3 日、MIT)
- KenLM を training/eval のみに使い、runtime 推論では使わない

## その他の依存

### PyTorch (BSD-3-Clause)

- ライセンス: BSD 3-Clause
- 本リポジトリでの扱い: training / eval で依存、ランタイムは user 環境にインストール

### onnxruntime (MIT)

- ライセンス: MIT
- 本リポジトリでの扱い: `tools/onnxruntime-win-x64-*/` として静的 DLL 利用

### Ginza / SudachiDict (Apache 2.0)

- ライセンス: Apache 2.0
- 本リポジトリでの扱い: `tools/corpus_v2/bunsetsu_split.py` で使用、ランタイム依存

### Fugashi / unidic-lite (MIT + BSD style)

- ライセンス: MIT (Fugashi), BSD-style (unidic-lite)
- 本リポジトリでの扱い: 読み生成パイプラインで使用

### mozc-ut dict (不問 / Public Domain 相当)

- 参照: `tools/old/build-train-mix-v2.py` で間接参照
- `tools/dict/import_mozc_dict.py` で処理、成果物の `fixed_dict_mozc_ut*.tsv` は
  `.gitignore` 済

## 上流データソース

学習データに含まれる source のライセンス詳細は [DATA_LICENSES.md](DATA_LICENSES.md)
参照 (Wikipedia CC-BY-SA 3.0、青空文庫 PD、HPLT CC0、FineWeb-2 ODC-By、
zenz-v2.5-dataset ODC-By、Wiktionary CC-BY-SA 3.0、Tatoeba CC-BY 2.0 FR、
Wikibooks CC-BY-SA 3.0、Wikinews CC-BY 2.5 など)。

これら **データを含む成果物 (モデル重み / 学習 JSONL)** は `MODEL_LICENSE`
(CC BY-SA 4.0) 配下、上流帰属義務 (ATTRIBUTION.md) 適用。GitHub には
入らず HF / Release 経由で配布。
