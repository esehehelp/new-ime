# probe_v3 失敗カテゴリ分析

LLM (DeepInfra Gemma 4 31B-it) による失敗分類。総コスト $0.0049。

カテゴリ定義は `scripts/llm_failure_analysis.py` の CATEGORIES 参照。


## suiko-v1-greedy (n=142 failures)

| failure_category | count | % |
|---|---:|---:|
| homophone_choice | 56 | 39.4% |
| model_garbage | 26 | 18.3% |
| named_entity | 14 | 9.9% |
| katakana_loanword | 10 | 7.0% |
| kana_kanji_split | 8 | 5.6% |
| okurigana | 6 | 4.2% |
| context_misuse | 6 | 4.2% |
| numeric_format | 5 | 3.5% |
| reference_ambiguous | 5 | 3.5% |
| other | 3 | 2.1% |
| punctuation | 2 | 1.4% |
| rare_kanji | 1 | 0.7% |

### probe_v3 カテゴリ別の失敗内訳

- **edge**: model_garbage(9), reference_ambiguous(4), katakana_loanword(3), homophone_choice(2), punctuation(1)
- **general**: homophone_choice(12), model_garbage(7), named_entity(4), kana_kanji_split(3), katakana_loanword(3), okurigana(2), reference_ambiguous(1)
- **homophone**: homophone_choice(13), context_misuse(6), kana_kanji_split(1)
- **names**: homophone_choice(11), named_entity(7), model_garbage(2), katakana_loanword(2), kana_kanji_split(1)
- **numeric**: homophone_choice(10), numeric_format(5), model_garbage(4), okurigana(3), named_entity(3), kana_kanji_split(2), katakana_loanword(1), rare_kanji(1)
- **particle**: other(3), homophone_choice(1)
- **tech**: homophone_choice(7), model_garbage(4), punctuation(1), okurigana(1), katakana_loanword(1), kana_kanji_split(1)

### サンプル (各失敗カテゴリ 上位 3 例)


#### context_misuse

- reading=`きしゃがきしゃできしゃした` context=`新聞社の` refs=`記者が汽車で帰社した` top1=`記者が記社で記社した`
- reading=`かがくのぶんせきをおこなった` context=`研究室では` refs=`化学の分析を行った` top1=`科学の分析を行った`
- reading=`かんじょうてきになってかんじょうをもちくずした` context=`会計の前に` refs=`感情的になって勘定を持ち崩した` top1=`感情的になって感情を持ち崩した`

#### homophone_choice

- reading=`はいたつはなのかごにとうちゃくします` context=`` refs=`配達は七日後に到着します / 配達は7日後に到着します` top1=`配達花のか籠に到着します`
- reading=`えーふぉーのようしをじゅうまいいんさつした` context=`` refs=`A4の用紙を10枚印刷した / A4の用紙を十枚印刷した` top1=`A4の用紙を従妹印刷した`
- reading=`こうえんでこうえんをきいた` context=`休日の午後は` refs=`公園で講演を聞いた` top1=`公園で公演を聞いた`

#### kana_kanji_split

- reading=`きょうはねむくてしごとがはかどらない` context=`昨晩は遅くまで残業だったので、` refs=`今日は眠くて仕事が捗らない / 今日は眠くて仕事がはかどらない` top1=`今日は眠くて仕事がはどらない`
- reading=`ははにあまえてはははなしをきかない` context=`子供が` refs=`母に甘えては話を聞かない` top1=`母に甘えて母話を聞かない`
- reading=`こころざしちくはちゅうごくのおんみょうごぎょうせつのえいきょうものこしていた。` context=`は游気を水蒸気の意味でも用いており、今日の空気と全く同じというわけではなかった。` refs=`志築は中国の陰陽五行説の影響も残していた。` top1=`志竹区は中国の陰陽五行説の影響も残していた。`

#### katakana_loanword

- reading=`ごーるまえひゃくめーとるでぬかれた` context=`` refs=`ゴール前100メートルで抜かれた / ゴール前百メートルで抜かれた / ゴール前100mで抜かれた` top1=`ゴール前100メトルで抜かれた`
- reading=`としょかんでべんきょうしているあいだにうとうとした` context=`` refs=`図書館で勉強している間にうとうとした` top1=`図書館で勉強している間にウとうとした`
- reading=`ずーむをいんすとーるした` context=`会議のために` refs=`Zoomをインストールした` top1=`Zomをインストールした`

#### model_garbage

- reading=`にちようひんのうりあげがはちじゅっぱーせんとをしめる` context=`` refs=`日用品の売上が80パーセントを占める / 日用品の売り上げが80%を占める` top1=`日用品の売上が80<0x25>を占める`
- reading=`ことしのうりあげはきょねんひひゃくにじゅっぱーせんとだ` context=`` refs=`今年の売上は去年比120パーセントだ / 今年の売上は去年比120%だ` top1=`今年の売上は去年比120<0x25>だ`
- reading=`あんけーとのかいとうりつはろくじゅっぱーせんとだった` context=`発表によると` refs=`アンケートの回答率は60パーセントだった / アンケートの回答率は60%だった` top1=`アンケートの回答率は60<0x25>だった`

#### named_entity

- reading=`「としまそんし」および「こくせいちょうさ」によればくちのしまのじんこうのせんいはかきのとおりである。` context=`南限の植物` refs=`「十島村誌」及び「国勢調査」によれば口之島の人口の遷移は下記のとおりである。` top1=`「豊島村市」および「国勢調査」によれば口の島の人口の繊維は下記の通りである。`
- reading=`えんしゅつはみいけたかし、きゃくほんはりりー・ふらんきー、しゅえんはいちかわえびぞう。` context=`主演は松平健。` refs=`演出は三池崇史、脚本はリリー・フランキー、主演は市川海老蔵。` top1=`演出は三池隆、脚本はリリー・フランキー、主演は市川海老蔵。`
- reading=`ほんみょうはさんこうぎん（みよししろがね）。` context=`静岡県出身。` refs=`本名は三好銀（みよし しろがね）。` top1=`本名は三光銀（三よししろがね）。`

#### numeric_format

- reading=`さんぜんこのしょうひんがほかんされている` context=`倉庫には` refs=`3000個の商品が保管されている / 三千個の商品が保管されている` top1=`0千個の商品が保管されている`
- reading=`きおんがさんじゅうごどをこえた` context=`` refs=`気温が35度を超えた / 気温が35℃を超えた` top1=`気温が三十5度を超えた`
- reading=`ろくせんまんえんだった` context=`新築マンションの価格は` refs=`6000万円だった / 六千万円だった` top1=`600万円だった`

#### okurigana

- reading=`ぴーえいちななのすいようえきをつくる` context=`` refs=`pH7の水溶液を作る` top1=`PH7の水溶液を作くる`
- reading=`いぬがげんきにしっぽをふっていた` context=`家に帰ると` refs=`犬が元気に尻尾を振っていた / 犬が元気にしっぽを振っていた` top1=`犬が元気に尻ぽを振っていた`
- reading=`おーぷんそーすこみゅにてぃにこんとりびゅーとする` context=`` refs=`オープンソースコミュニティにコントリビュートする` top1=`オープンソースコミュニティにコントリビューとする`

#### other

- reading=`かれにこのしょるいをわたしてください` context=`` refs=`彼にこの書類を渡してください` top1=`彼にこの書類を渡して下さい`
- reading=`あかいぺんだけでかいてください` context=`` refs=`赤いペンだけで書いてください` top1=`赤いペンだけで書いて下さい`
- reading=`わからないてんがあればしつもんしてください` context=`` refs=`分からない点があれば質問してください` top1=`分からない点があれば質問して下さい`

#### punctuation

- reading=`もじこーどがゆーてぃーえふえいとになった` context=`` refs=`文字コードがUTF-8になった` top1=`文字コードがUTF8になった`
- reading=`しーあいしーでぃーぱいぷらいんをこうちくする` context=`` refs=`CI/CDパイプラインを構築する` top1=`CICDパイプラインを構築する`

#### rare_kanji

- reading=`はままつしはまなくのせいぶ、みっかびちくのなんぶにいちする。` context=`住居表示未実施。` refs=`浜松市浜名区の西部、三ヶ日地区の南部に位置する。` top1=`浜松市浜名区の西部、三ケ日地区の南部に位置する。`

#### reference_ambiguous

- reading=`ちいさいころからえをかくのがすきだった` context=`` refs=`小さい頃から絵を描くのが好きだった` top1=`小さい頃から絵をかくのが好きだった`
- reading=`まっくぶっくぷろでさぎょうをしている` context=`` refs=`MacBook Proで作業をしている / マックブックプロで作業をしている` top1=`MacBookプロで作業をしている`
- reading=`すたっくおーばーふろーでかいけつほうをさがした` context=`` refs=`スタックオーバーフローで解決方法を探した` top1=`スタックオーバーフローで解決法を探した`

## suiko-v1-kenlm-moe (n=112 failures)

| failure_category | count | % |
|---|---:|---:|
| homophone_choice | 44 | 39.3% |
| model_garbage | 13 | 11.6% |
| context_misuse | 12 | 10.7% |
| katakana_loanword | 8 | 7.1% |
| okurigana | 8 | 7.1% |
| kana_kanji_split | 7 | 6.2% |
| named_entity | 7 | 6.2% |
| reference_ambiguous | 5 | 4.5% |
| numeric_format | 3 | 2.7% |
| punctuation | 3 | 2.7% |
| other | 2 | 1.8% |

### probe_v3 カテゴリ別の失敗内訳

- **edge**: katakana_loanword(3), reference_ambiguous(3), model_garbage(3), homophone_choice(2), punctuation(1)
- **general**: homophone_choice(13), model_garbage(4), kana_kanji_split(3), named_entity(2), other(1), katakana_loanword(1), punctuation(1), okurigana(1)
- **homophone**: context_misuse(12), homophone_choice(5), reference_ambiguous(1), kana_kanji_split(1)
- **names**: homophone_choice(8), named_entity(5), okurigana(1), reference_ambiguous(1), other(1), punctuation(1)
- **numeric**: homophone_choice(12), model_garbage(4), numeric_format(3), kana_kanji_split(2), okurigana(1)
- **particle**: okurigana(4)
- **tech**: katakana_loanword(4), homophone_choice(4), model_garbage(2), okurigana(1), kana_kanji_split(1)

### サンプル (各失敗カテゴリ 上位 3 例)


#### context_misuse

- reading=`きしゃがきしゃできしゃした` context=`新聞社の` refs=`記者が汽車で帰社した` top1=`記者が貴社で記者した`
- reading=`かがくのぶんせきをおこなった` context=`研究室では` refs=`化学の分析を行った` top1=`科学の分析を行った`
- reading=`かれのきぎょうけいかくがせいこうした` context=`新聞によると` refs=`彼の起業計画が成功した` top1=`彼の企業計画が成功した`

#### homophone_choice

- reading=`えーふぉーのようしをじゅうまいいんさつした` context=`` refs=`A4の用紙を10枚印刷した / A4の用紙を十枚印刷した` top1=`A4の用紙を従妹印刷した`
- reading=`こうえんでこうえんをきいた` context=`休日の午後は` refs=`公園で講演を聞いた` top1=`公園で公演を聞いた`
- reading=`かみにおがみをささげる` context=`神社の` refs=`神に拝みを捧げる` top1=`神に拝神を捧げる`

#### kana_kanji_split

- reading=`はいたつはなのかごにとうちゃくします` context=`` refs=`配達は七日後に到着します / 配達は7日後に到着します` top1=`配達花の後に到着します`
- reading=`ははにあまえてはははなしをきかない` context=`子供が` refs=`母に甘えては話を聞かない` top1=`母に甘えて母話を聞かない`
- reading=`こころざしちくはちゅうごくのおんみょうごぎょうせつのえいきょうものこしていた。` context=`は游気を水蒸気の意味でも用いており、今日の空気と全く同じというわけではなかった。` refs=`志築は中国の陰陽五行説の影響も残していた。` top1=`志地区は中国の陰陽五行説の影響も残していた。`

#### katakana_loanword

- reading=`あたらしいぷろじぇくとがはじまるのがたのしみだ` context=`` refs=`新しいプロジェクトが始まるのが楽しみだ` top1=`新しいPojectが始まるのが楽しみだ`
- reading=`おーぷんえーあいのえーぴーあいをつかう` context=`` refs=`OpenAIのAPIを使う` top1=`オープンAIのAPIを使う`
- reading=`まっくぶっくぷろでさぎょうをしている` context=`` refs=`MacBook Proで作業をしている / マックブックプロで作業をしている` top1=`MacBookプロで作業をしている`

#### model_garbage

- reading=`さんぜんこのしょうひんがほかんされている` context=`倉庫には` refs=`3000個の商品が保管されている / 三千個の商品が保管されている` top1=`0千個の商品が保管されている`
- reading=`にちようひんのうりあげがはちじゅっぱーせんとをしめる` context=`` refs=`日用品の売上が80パーセントを占める / 日用品の売り上げが80%を占める` top1=`日用品の売上が80<0x25>を占める`
- reading=`じぇーそんぺーろーどをぱーすする` context=`` refs=`JSONペイロードをパースする` top1=`じぇーそんぺーろーどをぱーすする`

#### named_entity

- reading=`そのときけんしんもうさるるやう` context=`川中島に出陣あり` refs=`そのとき謙信申さるるやう` top1=`その時検信申さるるやう`
- reading=`とやすたろうにとわれて、おすずはまたあかくなって、くびをふった。` context=`って出かける時は、チョウチンもつけていたろうから、年かっこうぐらい見えたろうね」` refs=`と保太郎に問われて、お鈴はまた赤くなって、首をふった。` top1=`と安太郎に問われて、お鈴はまた赤くなって、首を振った。`
- reading=`あなんしつばきちょうのむじんとうでたちばなわんないにいちする。` context=`` refs=`阿南市椿町の無人島で橘湾内に位置する。` top1=`阿南市椿町の無人島で花湾内に位置する。`

#### numeric_format

- reading=`きおんがさんじゅうごどをこえた` context=`` refs=`気温が35度を超えた / 気温が35℃を超えた` top1=`気温が三十五度を超えた`
- reading=`ついかとうしでにせんおくえんがひつようだ` context=`` refs=`追加投資で2000億円が必要だ / 追加投資で二千億円が必要だ` top1=`追加投資で200億円が必要だ`
- reading=`しゅうりひはさんまんにせんえんだった` context=`` refs=`修理費は3万2千円だった / 修理費は三万二千円だった / 修理費は32,000円だった` top1=`修理費は3万2000円だった`

#### okurigana

- reading=`ふじさんはせかいいさんにとうろくされている` context=`` refs=`富士山は世界遺産に登録されている` top1=`富士さんは世界遺産に登録されている`
- reading=`かれにこのしょるいをわたしてください` context=`` refs=`彼にこの書類を渡してください` top1=`彼にこの書類を渡して下さい`
- reading=`あかいぺんだけでかいてください` context=`` refs=`赤いペンだけで書いてください` top1=`赤いペンだけで書いて下さい`

#### other

- reading=`ちいさいころからえをかくのがすきだった` context=`` refs=`小さい頃から絵を描くのが好きだった` top1=`小さい頃から絵をかくのが好きだった`
- reading=`にゅーよーくのせんとらるぱーくをさんぽした` context=`` refs=`ニューヨークのセントラルパークを散歩した` top1=`NYのセントラルパークを散歩した`

#### punctuation

- reading=`もじこーどがゆーてぃーえふえいとになった` context=`` refs=`文字コードがUTF-8になった` top1=`文字コードがUTF8になった`
- reading=`おっととともにらーめんてんをきりもりしている。` context=`蓮太郎の母。` refs=`夫とともにラーメン店を切り盛りしている。` top1=`夫と共にラーメン店を切り盛りしている。`
- reading=`わきみさきびーちろっくわきみさきのびーちろっくながさきけんのぶんかざい` context=`樺島灯台公園` refs=`脇岬ビーチロック脇岬のビーチロック 長崎県の文化財` top1=`脇岬ビーチロック脇岬のビーチロック長崎県の文化財`

#### reference_ambiguous

- reading=`きかいのきかいをうかがった` context=`会議の後で` refs=`機械の機会をうかがった` top1=`機械の機会を伺った`
- reading=`ゆーちゅーぶでおんがくをきいている` context=`` refs=`YouTubeで音楽を聞いている / ユーチューブで音楽を聞いている` top1=`YouTubeで音楽を聴いている`
- reading=`すらっくのちゃんねるでこうちくをそうだんする` context=`` refs=`Slackのチャンネルで構築を相談する` top1=`スラックのチャンネルで構築を相談する`

## hatsuyume-greedy (n=0 failures)

| failure_category | count | % |
|---|---:|---:|

### probe_v3 カテゴリ別の失敗内訳


### サンプル (各失敗カテゴリ 上位 3 例)


## hatsuyume-kenlm-moe (n=0 failures)

| failure_category | count | % |
|---|---:|---:|

### probe_v3 カテゴリ別の失敗内訳


### サンプル (各失敗カテゴリ 上位 3 例)
