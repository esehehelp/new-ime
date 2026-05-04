//! data-synth-name: template-based generation of sentences containing common
//! Japanese proper nouns (surnames, given names, place names, org names).
//!
//! Purpose: probe的な弱点である `names` カテゴリ (固有名詞の読み → 表記) を
//! 学習 mix にスパイクさせる。NEologd 等の大規模辞書の取得を強制せず、
//! built-in の最小セットで bootstrap し、`--names-csv` / `--places-csv`
//! で拡張可能にする。
//!
//! Output schema: bunsetsu-compatible. Particle/context are included so the
//! model sees the name in natural surroundings.

use anyhow::{Context, Result};
use clap::Parser;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "data-synth-name", about = "Generate named-entity synth rows")]
struct Cli {
    #[arg(long)]
    output: PathBuf,
    /// CSV (surface,reading) of additional surnames. Optional.
    #[arg(long)]
    surnames_csv: Option<PathBuf>,
    /// CSV (surface,reading) of additional given names. Optional.
    #[arg(long)]
    given_csv: Option<PathBuf>,
    /// CSV (surface,reading) of additional place names. Optional.
    #[arg(long)]
    places_csv: Option<PathBuf>,
    /// CSV (surface,reading) of company / organisation names. Optional.
    #[arg(long)]
    orgs_csv: Option<PathBuf>,
    #[arg(long, default_value_t = 2_500_000)]
    target_size: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Serialize)]
struct Row<'a> {
    reading: String,
    surface: String,
    left_context_surface: String,
    left_context_reading: String,
    span_bunsetsu: u32,
    source: &'a str,
    sentence_id: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let mut surnames = builtin_surnames();
    extend_csv(&mut surnames, cli.surnames_csv.as_deref(), "surnames")?;
    let mut given = builtin_given_names();
    extend_csv(&mut given, cli.given_csv.as_deref(), "given_names")?;
    let mut places = builtin_places();
    extend_csv(&mut places, cli.places_csv.as_deref(), "places")?;
    let mut orgs = builtin_orgs();
    extend_csv(&mut orgs, cli.orgs_csv.as_deref(), "orgs")?;

    eprintln!(
        "[data-synth-name] lexicon: surnames={} given={} places={} orgs={}",
        surnames.len(),
        given.len(),
        places.len(),
        orgs.len(),
    );

    if let Some(parent) = cli.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut out = BufWriter::with_capacity(
        8 * 1024 * 1024,
        File::create(&cli.output).with_context(|| format!("create {}", cli.output.display()))?,
    );

    let mut rng = StdRng::seed_from_u64(cli.seed);
    let mut idx = 0usize;
    let mut emitted = 0usize;

    // Template allocation: roughly
    //   40%  person_single_bunsetsu  (山田さんは / 佐藤が / 鈴木太郎の)
    //   20%  person_full_name        (山田太郎 / 佐藤花子さん)
    //   20%  place_phrase            (東京で / 大阪の / 横浜に)
    //   10%  org_phrase              (株式会社〜 / 〜大学)
    //   10%  person_in_place         ("東京の山田さん") as left_context
    let target_person_single = cli.target_size * 40 / 100;
    let target_person_full = cli.target_size * 20 / 100;
    let target_place = cli.target_size * 20 / 100;
    let target_org = cli.target_size * 10 / 100;
    let target_person_place =
        cli.target_size - target_person_single - target_person_full - target_place - target_org;

    eprintln!(
        "[allocation] single={} full={} place={} org={} person_in_place={}",
        target_person_single, target_person_full, target_place, target_org, target_person_place
    );

    // ----- person single bunsetsu -----
    let person_suffixes: &[(&str, &str)] = &[
        ("さん", "さん"),
        ("さんは", "さんは"),
        ("さんが", "さんが"),
        ("さんに", "さんに"),
        ("さんの", "さんの"),
        ("さんと", "さんと"),
        ("さんも", "さんも"),
        ("氏", "し"),
        ("氏は", "しは"),
        ("氏の", "しの"),
        ("先生", "せんせい"),
        ("先生は", "せんせいは"),
        ("先生の", "せんせいの"),
        ("部長", "ぶちょう"),
        ("部長は", "ぶちょうは"),
        ("社長", "しゃちょう"),
        ("社長の", "しゃちょうの"),
        ("課長", "かちょう"),
        ("教授", "きょうじゅ"),
        ("教授の", "きょうじゅの"),
        ("博士", "はかせ"),
        ("くん", "くん"),
        ("ちゃん", "ちゃん"),
        // bare surname (no suffix) — the model must output just the kanji
        ("", ""),
    ];
    for _ in 0..target_person_single {
        let (name_s, name_r) = surnames.choose(&mut rng).unwrap();
        let (suf_s, suf_r) = person_suffixes.choose(&mut rng).unwrap();
        let surface = format!("{}{}", name_s, suf_s);
        let reading = format!("{}{}", name_r, suf_r);
        write_row(
            &mut out,
            Row {
                reading,
                surface,
                left_context_surface: String::new(),
                left_context_reading: String::new(),
                span_bunsetsu: 1,
                source: "synth_name",
                sentence_id: format!("synth_name:{}", idx),
            },
        )?;
        idx += 1;
        emitted += 1;
    }

    // ----- full name -----
    for _ in 0..target_person_full {
        let (sn_s, sn_r) = surnames.choose(&mut rng).unwrap();
        let (gn_s, gn_r) = given.choose(&mut rng).unwrap();
        let suffix_idx = rng.gen_range(0..5);
        let (suf_s, suf_r) = match suffix_idx {
            0 => ("さん", "さん"),
            1 => ("氏", "し"),
            2 => ("", ""),
            3 => ("さんは", "さんは"),
            _ => ("さんの", "さんの"),
        };
        let surface = format!("{}{}{}", sn_s, gn_s, suf_s);
        let reading = format!("{}{}{}", sn_r, gn_r, suf_r);
        write_row(
            &mut out,
            Row {
                reading,
                surface,
                left_context_surface: String::new(),
                left_context_reading: String::new(),
                span_bunsetsu: 1,
                source: "synth_name_full",
                sentence_id: format!("synth_name_full:{}", idx),
            },
        )?;
        idx += 1;
        emitted += 1;
    }

    // ----- place phrase -----
    let place_suffixes: &[(&str, &str)] = &[
        ("", ""),
        ("で", "で"),
        ("に", "に"),
        ("の", "の"),
        ("へ", "へ"),
        ("から", "から"),
        ("まで", "まで"),
        ("を", "を"),
        ("と", "と"),
        ("市", "し"),
        ("市の", "しの"),
        ("県", "けん"),
        ("県の", "けんの"),
        ("都", "と"),
        ("都の", "との"),
        ("府", "ふ"),
        ("府の", "ふの"),
        ("区", "く"),
        ("区の", "くの"),
        ("町", "ちょう"),
        ("町の", "ちょうの"),
    ];
    for _ in 0..target_place {
        let (p_s, p_r) = places.choose(&mut rng).unwrap();
        let (suf_s, suf_r) = place_suffixes.choose(&mut rng).unwrap();
        let surface = format!("{}{}", p_s, suf_s);
        let reading = format!("{}{}", p_r, suf_r);
        write_row(
            &mut out,
            Row {
                reading,
                surface,
                left_context_surface: String::new(),
                left_context_reading: String::new(),
                span_bunsetsu: 1,
                source: "synth_place",
                sentence_id: format!("synth_place:{}", idx),
            },
        )?;
        idx += 1;
        emitted += 1;
    }

    // ----- org phrase -----
    let org_prefixes: &[(&str, &str)] = &[
        ("株式会社", "かぶしきがいしゃ"),
        ("有限会社", "ゆうげんがいしゃ"),
        ("合同会社", "ごうどうがいしゃ"),
        ("", ""),
    ];
    let org_suffixes: &[(&str, &str)] = &[
        ("", ""),
        ("は", "は"),
        ("が", "が"),
        ("の", "の"),
        ("に", "に"),
        ("大学", "だいがく"),
        ("大学の", "だいがくの"),
        ("株式会社", "かぶしきがいしゃ"),
        ("研究所", "けんきゅうじょ"),
        ("病院", "びょういん"),
        ("銀行", "ぎんこう"),
        ("新聞", "しんぶん"),
    ];
    for _ in 0..target_org {
        let (o_s, o_r) = orgs.choose(&mut rng).unwrap();
        let (pre_s, pre_r) = org_prefixes.choose(&mut rng).unwrap();
        let (suf_s, suf_r) = org_suffixes.choose(&mut rng).unwrap();
        let surface = format!("{}{}{}", pre_s, o_s, suf_s);
        let reading = format!("{}{}{}", pre_r, o_r, suf_r);
        write_row(
            &mut out,
            Row {
                reading,
                surface,
                left_context_surface: String::new(),
                left_context_reading: String::new(),
                span_bunsetsu: 1,
                source: "synth_org",
                sentence_id: format!("synth_org:{}", idx),
            },
        )?;
        idx += 1;
        emitted += 1;
    }

    // ----- person in place (uses left_context) -----
    for _ in 0..target_person_place {
        let (place_s, place_r) = places.choose(&mut rng).unwrap();
        let (name_s, name_r) = surnames.choose(&mut rng).unwrap();
        let suffix_idx = rng.gen_range(0..5);
        let (suf_s, suf_r) = match suffix_idx {
            0 => ("さんは", "さんは"),
            1 => ("さんの", "さんの"),
            2 => ("さんが", "さんが"),
            3 => ("氏は", "しは"),
            _ => ("社長は", "しゃちょうは"),
        };
        let lc_idx = rng.gen_range(0..3);
        let (lc_s, lc_r) = match lc_idx {
            0 => (format!("{}の", place_s), format!("{}の", place_r)),
            1 => (format!("{}に住む", place_s), format!("{}にすむ", place_r)),
            _ => (
                format!("{}出身の", place_s),
                format!("{}しゅっしんの", place_r),
            ),
        };
        let surface = format!("{}{}", name_s, suf_s);
        let reading = format!("{}{}", name_r, suf_r);
        write_row(
            &mut out,
            Row {
                reading,
                surface,
                left_context_surface: lc_s,
                left_context_reading: lc_r,
                span_bunsetsu: 1,
                source: "synth_name_context",
                sentence_id: format!("synth_name_context:{}", idx),
            },
        )?;
        idx += 1;
        emitted += 1;
    }

    out.flush()?;
    eprintln!(
        "[data-synth-name] wrote {} rows to {}",
        emitted,
        cli.output.display()
    );
    Ok(())
}

fn write_row(out: &mut dyn Write, row: Row) -> Result<()> {
    struct PyFmt;
    impl serde_json::ser::Formatter for PyFmt {
        fn begin_object_key<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first {
                Ok(())
            } else {
                w.write_all(b", ")
            }
        }
        fn begin_object_value<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
        ) -> std::io::Result<()> {
            w.write_all(b": ")
        }
        fn begin_array_value<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first {
                Ok(())
            } else {
                w.write_all(b", ")
            }
        }
    }
    let mut buf = Vec::with_capacity(256);
    {
        let mut ser = serde_json::Serializer::with_formatter(&mut buf, PyFmt);
        row.serialize(&mut ser).context("serialize row")?;
    }
    out.write_all(&buf)?;
    out.write_all(b"\n")?;
    Ok(())
}

fn extend_csv(
    target: &mut Vec<(String, String)>,
    path: Option<&std::path::Path>,
    label: &str,
) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    let reader = BufReader::new(
        File::open(path).with_context(|| format!("open {} csv {}", label, path.display()))?,
    );
    let before = target.len();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let mut parts = line.splitn(2, ',');
        let surface = parts.next().unwrap_or("").trim();
        let reading = parts.next().unwrap_or("").trim();
        if surface.is_empty() || reading.is_empty() {
            continue;
        }
        target.push((surface.to_string(), reading.to_string()));
        if i > 0 && i % 100_000 == 0 {
            eprintln!("  [{}] loaded {}", label, i);
        }
    }
    eprintln!(
        "[csv] {} +{} rows from {}",
        label,
        target.len() - before,
        path.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Built-in fallback lexicons. Small but high-confidence. Extend via CSV args.
// ---------------------------------------------------------------------------

fn owned(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(s, r)| (s.to_string(), r.to_string()))
        .collect()
}

fn builtin_surnames() -> Vec<(String, String)> {
    owned(&[
        ("佐藤", "さとう"),
        ("鈴木", "すずき"),
        ("高橋", "たかはし"),
        ("田中", "たなか"),
        ("伊藤", "いとう"),
        ("渡辺", "わたなべ"),
        ("山本", "やまもと"),
        ("中村", "なかむら"),
        ("小林", "こばやし"),
        ("加藤", "かとう"),
        ("吉田", "よしだ"),
        ("山田", "やまだ"),
        ("佐々木", "ささき"),
        ("山口", "やまぐち"),
        ("松本", "まつもと"),
        ("井上", "いのうえ"),
        ("木村", "きむら"),
        ("林", "はやし"),
        ("斎藤", "さいとう"),
        ("清水", "しみず"),
        ("山崎", "やまざき"),
        ("森", "もり"),
        ("池田", "いけだ"),
        ("橋本", "はしもと"),
        ("阿部", "あべ"),
        ("石川", "いしかわ"),
        ("山下", "やました"),
        ("中島", "なかじま"),
        ("石井", "いしい"),
        ("小川", "おがわ"),
        ("前田", "まえだ"),
        ("岡田", "おかだ"),
        ("長谷川", "はせがわ"),
        ("藤田", "ふじた"),
        ("後藤", "ごとう"),
        ("近藤", "こんどう"),
        ("村上", "むらかみ"),
        ("遠藤", "えんどう"),
        ("青木", "あおき"),
        ("坂本", "さかもと"),
        ("斉藤", "さいとう"),
        ("福田", "ふくだ"),
        ("太田", "おおた"),
        ("西村", "にしむら"),
        ("藤井", "ふじい"),
        ("金子", "かねこ"),
        ("岡本", "おかもと"),
        ("中野", "なかの"),
        ("藤原", "ふじわら"),
        ("三浦", "みうら"),
        ("原田", "はらだ"),
        ("中川", "なかがわ"),
        ("松田", "まつだ"),
        ("竹内", "たけうち"),
        ("小野", "おの"),
        ("田村", "たむら"),
        ("中山", "なかやま"),
        ("藤本", "ふじもと"),
        ("原", "はら"),
        ("和田", "わだ"),
        ("上田", "うえだ"),
        ("武田", "たけだ"),
        ("村田", "むらた"),
        ("谷口", "たにぐち"),
        ("上野", "うえの"),
        ("工藤", "くどう"),
        ("宮崎", "みやざき"),
        ("酒井", "さかい"),
        ("大野", "おおの"),
        ("今井", "いまい"),
        ("高木", "たかぎ"),
        ("丸山", "まるやま"),
        ("増田", "ますだ"),
        ("杉山", "すぎやま"),
        ("野口", "のぐち"),
        ("大塚", "おおつか"),
        ("小島", "こじま"),
        ("千葉", "ちば"),
        ("久保", "くぼ"),
        ("平野", "ひらの"),
        ("岩崎", "いわさき"),
        ("新井", "あらい"),
        ("桜井", "さくらい"),
        ("大西", "おおにし"),
        ("松井", "まつい"),
        ("野村", "のむら"),
        ("菊地", "きくち"),
        ("松尾", "まつお"),
        ("大橋", "おおはし"),
        ("石田", "いしだ"),
        ("高田", "たかだ"),
        ("関", "せき"),
    ])
}

fn builtin_given_names() -> Vec<(String, String)> {
    owned(&[
        ("太郎", "たろう"),
        ("次郎", "じろう"),
        ("三郎", "さぶろう"),
        ("一郎", "いちろう"),
        ("健", "けん"),
        ("健一", "けんいち"),
        ("健太", "けんた"),
        ("大輔", "だいすけ"),
        ("和夫", "かずお"),
        ("正男", "まさお"),
        ("明", "あきら"),
        ("誠", "まこと"),
        ("光男", "みつお"),
        ("博", "ひろし"),
        ("進", "すすむ"),
        ("昇", "のぼる"),
        ("豊", "ゆたか"),
        ("勉", "つとむ"),
        ("隆", "たかし"),
        ("剛", "つよし"),
        ("修", "おさむ"),
        ("学", "まなぶ"),
        ("翔太", "しょうた"),
        ("蓮", "れん"),
        ("陽翔", "はると"),
        ("悠真", "ゆうま"),
        ("颯太", "そうた"),
        ("湊", "みなと"),
        ("大和", "やまと"),
        ("樹", "いつき"),
        ("花子", "はなこ"),
        ("美香", "みか"),
        ("恵美", "えみ"),
        ("由美", "ゆみ"),
        ("洋子", "ようこ"),
        ("幸子", "さちこ"),
        ("京子", "きょうこ"),
        ("和子", "かずこ"),
        ("久美子", "くみこ"),
        ("真由美", "まゆみ"),
        ("智子", "ともこ"),
        ("直子", "なおこ"),
        ("美樹", "みき"),
        ("裕子", "ゆうこ"),
        ("陽子", "ようこ"),
        ("美穂", "みほ"),
        ("美紀", "みき"),
        ("愛", "あい"),
        ("葵", "あおい"),
        ("結衣", "ゆい"),
        ("凛", "りん"),
        ("陽菜", "ひな"),
        ("美咲", "みさき"),
        ("さくら", "さくら"),
        ("明日香", "あすか"),
        ("ゆり", "ゆり"),
        ("ひまり", "ひまり"),
        ("凜", "りん"),
    ])
}

fn builtin_places() -> Vec<(String, String)> {
    owned(&[
        ("東京", "とうきょう"),
        ("大阪", "おおさか"),
        ("京都", "きょうと"),
        ("名古屋", "なごや"),
        ("横浜", "よこはま"),
        ("神戸", "こうべ"),
        ("札幌", "さっぽろ"),
        ("福岡", "ふくおか"),
        ("仙台", "せんだい"),
        ("広島", "ひろしま"),
        ("千葉", "ちば"),
        ("埼玉", "さいたま"),
        ("川崎", "かわさき"),
        ("北海道", "ほっかいどう"),
        ("青森", "あおもり"),
        ("岩手", "いわて"),
        ("宮城", "みやぎ"),
        ("秋田", "あきた"),
        ("山形", "やまがた"),
        ("福島", "ふくしま"),
        ("茨城", "いばらき"),
        ("栃木", "とちぎ"),
        ("群馬", "ぐんま"),
        ("新潟", "にいがた"),
        ("富山", "とやま"),
        ("石川", "いしかわ"),
        ("福井", "ふくい"),
        ("山梨", "やまなし"),
        ("長野", "ながの"),
        ("岐阜", "ぎふ"),
        ("静岡", "しずおか"),
        ("愛知", "あいち"),
        ("三重", "みえ"),
        ("滋賀", "しが"),
        ("兵庫", "ひょうご"),
        ("奈良", "なら"),
        ("和歌山", "わかやま"),
        ("鳥取", "とっとり"),
        ("島根", "しまね"),
        ("岡山", "おかやま"),
        ("山口", "やまぐち"),
        ("徳島", "とくしま"),
        ("香川", "かがわ"),
        ("愛媛", "えひめ"),
        ("高知", "こうち"),
        ("佐賀", "さが"),
        ("長崎", "ながさき"),
        ("熊本", "くまもと"),
        ("大分", "おおいた"),
        ("宮崎", "みやざき"),
        ("鹿児島", "かごしま"),
        ("沖縄", "おきなわ"),
        ("渋谷", "しぶや"),
        ("新宿", "しんじゅく"),
        ("池袋", "いけぶくろ"),
        ("品川", "しながわ"),
        ("秋葉原", "あきはばら"),
        ("銀座", "ぎんざ"),
        ("上野", "うえの"),
        ("浅草", "あさくさ"),
        ("六本木", "ろっぽんぎ"),
        ("恵比寿", "えびす"),
        ("原宿", "はらじゅく"),
        ("表参道", "おもてさんどう"),
        ("梅田", "うめだ"),
        ("難波", "なんば"),
        ("天王寺", "てんのうじ"),
        ("心斎橋", "しんさいばし"),
        ("中野", "なかの"),
        ("西新宿", "にしんしんじゅく"),
        ("東京駅", "とうきょうえき"),
        ("羽田", "はねだ"),
        ("成田", "なりた"),
        ("関西", "かんさい"),
        ("関東", "かんとう"),
        ("東北", "とうほく"),
        ("九州", "きゅうしゅう"),
        ("四国", "しこく"),
        ("中部", "ちゅうぶ"),
        ("日本", "にほん"),
        ("アメリカ", "あめりか"),
        ("中国", "ちゅうごく"),
        ("韓国", "かんこく"),
        ("イギリス", "いぎりす"),
        ("フランス", "ふらんす"),
        ("ドイツ", "どいつ"),
        ("イタリア", "いたりあ"),
    ])
}

fn builtin_orgs() -> Vec<(String, String)> {
    owned(&[
        ("東京", "とうきょう"),
        ("日本", "にっぽん"),
        ("三菱", "みつびし"),
        ("三井", "みつい"),
        ("住友", "すみとも"),
        ("ソニー", "そにー"),
        ("ホンダ", "ほんだ"),
        ("トヨタ", "とよた"),
        ("日立", "ひたち"),
        ("富士通", "ふじつう"),
        ("キヤノン", "きやのん"),
        ("ニコン", "にこん"),
        ("任天堂", "にんてんどう"),
        ("楽天", "らくてん"),
        ("野村", "のむら"),
        ("大和", "だいわ"),
        ("東海", "とうかい"),
        ("西日本", "にしにほん"),
        ("東日本", "ひがしにほん"),
        ("早稲田", "わせだ"),
        ("慶應", "けいおう"),
        ("京都", "きょうと"),
        ("北海道", "ほっかいどう"),
        ("東北", "とうほく"),
        ("九州", "きゅうしゅう"),
        ("筑波", "つくば"),
        ("名古屋", "なごや"),
        ("朝日", "あさひ"),
        ("読売", "よみうり"),
        ("毎日", "まいにち"),
        ("日経", "にっけい"),
        ("産経", "さんけい"),
        ("NHK", "えぬえいちけー"),
        ("フジ", "ふじ"),
    ])
}
