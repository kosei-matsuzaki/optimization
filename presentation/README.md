# presentation/

進捗発表デッキ置き場。**発表日ごとに 1 ディレクトリ**（`YYYYMMDD/`）。

```
presentation/
├── 20260519/            # 進捗報告 #1（PDF のみ）
└── 20260714/            # 進捗報告 #2（旧 20260707。発表日変更に伴い改名）
    ├── 20260714.pptx    # 最終成果物（build_deck.py の出力）
    ├── 20260714.pdf     # 共有用 PDF（build/render/ からコピー）
    ├── outline.md       # 構成メモ・データソース（results/ の参照元）
    ├── 発表原稿.md      # 読み上げ原稿
    ├── 解説.md          # 全ページ解説
    └── build/           # ビルドパイプライン一式（パス解決は相対なので丸ごと移動可）
        ├── figs.py         # 1. 図を figs/<pNN_slug>/<panel>.svg に生成
        ├── convert.py      # 2. SVG→EMF 変換（soffice / inkscape をパネル別に自動選択）
        ├── build_deck.py   # 3. python-pptx で組版 → ../20260714.pptx
        ├── capture_*.py    # 図データ (figs/*.npz) の再現・取得スクリプト
        ├── figs/           # 生成図（ページ別サブフォルダの SVG/EMF/PNG ＋ npz）
        └── render/         # 検証用 pptx→pdf→png（soffice ＋ PyMuPDF、png/ は使い捨て）
```

## ビルド手順（20260714/build/ で実行）

```sh
python3 figs.py        # 図の変更時のみ
python3 convert.py     # 〃
python3 build_deck.py  # 組版（レイアウトのみの変更ならここから）
cd .. && soffice --headless --convert-to pdf --outdir build/render 20260714.pptx
cp build/render/20260714.pdf 20260714.pdf
```

スライド修正後は **pdf→png（PyMuPDF）まで通して目視確認するまで完了としない**。
デザイン規約・ベクター画像のハマりどころはメモリ
（`project_deck_design_system` / `project_slide_vector_pipeline`）と
`.claude/slide_design_notes.md` を参照。

## 新しいデッキの始め方

新しい `YYYYMMDD/` を作り、直前デッキの `build/` をコピーして開始する。
`figs.py`・`capture_*.py` は repo ルートを `Path(__file__).parents[3]` で解決している
（`presentation/<date>/build/*.py` の深さ前提）。
