use mecab::Tagger;

fn kata_to_hira(s: &str) -> String {
    s.chars()
        .map(|c| {
            if ('\u{30A1}'..='\u{30FA}').contains(&c) {
                char::from_u32(c as u32 - 0x60).unwrap_or(c)
            } else {
                c
            }
        })
        .collect()
}

/// unidic-lite output format (tab-separated):
/// Surface\t書字形\t仮名形\t原形\t品詞\t活用型\t活用形\tアクセント
/// We want: 仮名形 (index 1 after splitting feature by \t... but actually
/// the mecab-rs crate gives us surface separately, and feature as one string)
///
/// With `-d unidic_lite/dicdir`, the feature string is:
/// 書字形出現形\t仮名形出現形\t原形\t品詞\t活用型\t活用形\tアクセント
/// Note: the default node output format differs from csv output format.
/// In the mecab-rs Node, `feature` is everything after the surface+tab.

fn main() {
    // Use unidic-lite dictionary
    let dict_path = std::env::var("UNIDIC_DIR")
        .unwrap_or_else(|_| {
            "/home/esehe/.local/lib/python3.12/site-packages/unidic_lite/dicdir".to_string()
        });

    let arg = format!("-d {}", dict_path);
    let mut tagger = Tagger::new(&arg);

    let tests = vec![
        "今日はいい天気ですね",
        "東京都渋谷区",
        "学校に行く",
        "漢字変換の精度を評価する",
        "散歩した交渉が難航している",
        "形成される子供独自の文化である",
    ];

    for text in &tests {
        println!("Input: {}", text);

        let result = tagger.parse_to_node(*text);
        for node in result.iter_next() {
            if node.stat as i32 == 2 || node.stat as i32 == 3 {
                continue;
            }

            let surface = &(node.surface)[..node.length as usize];
            let feature = node.feature;

            // Feature fields are tab-separated in unidic-lite output
            // But mecab-rs may give them comma-separated depending on config
            // Let's check both
            let parts: Vec<&str> = if feature.contains('\t') {
                feature.split('\t').collect()
            } else {
                feature.split(',').collect()
            };

            // For unidic-lite default output:
            // parts[0] = 書字形出現形 (katakana)
            // parts[1] = 仮名形出現形 (katakana, conjugated) ← WANT THIS
            // parts[2] = 原形
            // parts[3] = 品詞 (hyphenated)
            let kana_form = if parts.len() > 1 && parts[1] != "*" {
                parts[1]
            } else if !parts.is_empty() && parts[0] != "*" {
                parts[0]
            } else {
                ""
            };

            let pos = if parts.len() > 3 {
                parts[3].split('-').next().unwrap_or("?")
            } else {
                "?"
            };

            let reading = kata_to_hira(kana_form);

            println!(
                "  {:8} pos={:6} kana={:12} → {}",
                surface, pos, kana_form, reading
            );
        }
        println!();
    }
}
