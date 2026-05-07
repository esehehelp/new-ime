# probe_v3 paste-friendly (答えなし)

frontier LLM を chat UI で手動 bench する用。各 item を 1 つずつコピペして投げ、上の system prompt + 下の reading/context を user message として送る。出力は 1 行 1 surface のみ。記録は `docs/probe_v3_with_answers.md` 側で答えと突き合わせ。

## 共通 system prompt
```
あなたは日本語のかな漢字変換アシスタント。ユーザが与える読みを、文脈に合う最も自然な漢字混じり日本語に 1 通りだけ変換し、余計な説明・引用符・ピリオド・改行なしで surface のみ 1 行で返してください。
```

## 共通 user template
```
前文脈: {context}
読み: {reading}

変換結果 (漢字混じり surface のみ、1 行):
```

---

## edge (n=40)

### edge_001_hand
- reading: `あいふぉーんとあんどろいどをひかくする`
- context: `(無し)`

### edge_002_hand
- reading: `らいんのめっせーじをうけとった`
- context: `(無し)`

### edge_003_hand
- reading: `ぐーぐるまっぷでけんさくする`
- context: `(無し)`

### edge_004_hand
- reading: `ぎっとはぶにりぽじとりをさくせいした`
- context: `(無し)`

### edge_005_hand
- reading: `ぱいそんのすくりぷとをかく`
- context: `(無し)`

### edge_006_hand
- reading: `こーひーをすたーばっくすでかう`
- context: `毎朝の`

### edge_007_hand
- reading: `あまぞんぷらいむでどうがをみる`
- context: `(無し)`

### edge_008_hand
- reading: `ねっとふりっくすのしんさくをみた`
- context: `(無し)`

### edge_009_hand
- reading: `おーぷんえーあいのえーぴーあいをつかう`
- context: `(無し)`

### edge_010_hand
- reading: `ずーむをいんすとーるした`
- context: `会議のために`

### edge_011_hand
- reading: `ゆーちゅーぶでおんがくをきいている`
- context: `(無し)`

### edge_012_hand
- reading: `すたーとあっぷにじょいんした`
- context: `(無し)`

### edge_013_hand
- reading: `まっくぶっくぷろでさぎょうをしている`
- context: `(無し)`

### edge_014_hand
- reading: `うぇぶさいとのゆーあいがあたらしくなった`
- context: `(無し)`

### edge_015_hand
- reading: `しすてむのあーるぴーえーをどうにゅうする`
- context: `(無し)`

### edge_016_hand
- reading: `すたっくおーばーふろーでかいけつほうをさがした`
- context: `(無し)`

### edge_017_hand
- reading: `もじこーどがゆーてぃーえふえいとになった`
- context: `(無し)`

### edge_018_hand
- reading: `でーたをしーえすぶいでえくすぽーとする`
- context: `(無し)`

### edge_019_hand
- reading: `じぇーそんぺーろーどをぱーすする`
- context: `(無し)`

### edge_020_hand
- reading: `でぃーえぬえすせっていをこうしんした`
- context: `(無し)`

### edge_021_hand
- reading: `ぶいぴーえぬせつぞくにじかんがかかる`
- context: `(無し)`

### edge_022_hand
- reading: `そーしゃるめでぃあぐるーぷをかんりする`
- context: `(無し)`

### edge_023_hand
- reading: `でじたるとらんすふぉーめーしょんをすいしんする`
- context: `(無し)`

### edge_024_hand
- reading: `ふろんとえんどふれーむわーくはりあくとをつかう`
- context: `(無し)`

### edge_025_hand
- reading: `すくらむかいはつをどうにゅうした`
- context: `(無し)`

### edge_026_hand
- reading: `はっかそんにさんかしてぷろとたいぷをつくった`
- context: `(無し)`

### edge_027_hand
- reading: `ぶろっくちぇーんぎじゅつにきょうみがある`
- context: `(無し)`

### edge_028_hand
- reading: `くらうどふぁんでぃんぐでしきんをあつめる`
- context: `(無し)`

### edge_029_hand
- reading: `えすえぬえすでばずってとれんどいりした`
- context: `(無し)`

### edge_030_hand
- reading: `りもーとわーくがすたんだーどになった`
- context: `(無し)`

### edge_031_hand
- reading: `えすでぃーじーずのもくひょうをたっせいする`
- context: `(無し)`

### edge_032_hand
- reading: `めたばーすでばーちゃるかいぎをひらく`
- context: `(無し)`

### edge_033_hand
- reading: `ちゃっとじーぴーてぃーをぎょうむにかつようする`
- context: `(無し)`

### edge_034_hand
- reading: `きゅーあーるこーどでけっさいする`
- context: `(無し)`

### edge_035_hand
- reading: `あぷりでちぇっくする`
- context: `朝の交通状況を`

### edge_036_hand
- reading: `えっくすじょうでとうこうがばずった`
- context: `(無し)`

### edge_037_hand
- reading: `すらっくのちゃんねるでこうちくをそうだんする`
- context: `(無し)`

### edge_038_hand
- reading: `てんそるふろーよりぱいとーちをこのむ`
- context: `(無し)`

### edge_039_hand
- reading: `あーるえすえすふぃーどでにゅーすをよむ`
- context: `(無し)`

### edge_040_hand
- reading: `やふーのとぷべーじがこうしんされた`
- context: `(無し)`

## general (n=75)

### general_001_hand
- reading: `あたらしいすまーとふぉんをこうにゅうした`
- context: `(無し)`

### general_002_hand
- reading: `しゅうまつはかぞくでこうえんへでかけた`
- context: `(無し)`

### general_003_hand
- reading: `きょうはねむくてしごとがはかどらない`
- context: `昨晩は遅くまで残業だったので、`

### general_004_hand
- reading: `あしたはゆきになるかもしれない`
- context: `天気予報によると、`

### general_005_hand
- reading: `としょかんでべんきょうしているあいだにうとうとした`
- context: `(無し)`

### general_006_hand
- reading: `えきまえのかふぇでまちあわせる`
- context: `友人との約束で`

### general_007_hand
- reading: `こんばんのこんだてをかんがえる`
- context: `冷蔵庫の中身を見て、`

### general_008_hand
- reading: `かのじょはえいごをねっしんにべんきょうしている`
- context: `(無し)`

### general_009_hand
- reading: `すきなしゅみのじかんがとれない`
- context: `最近は忙しくて、`

### general_010_hand
- reading: `ちいさいころからえをかくのがすきだった`
- context: `(無し)`

### general_011_hand
- reading: `こうえんでたのしそうにあそんでいる`
- context: `子供たちが`

### general_012_hand
- reading: `あめがつよくふっているのでかさをさす`
- context: `(無し)`

### general_013_hand
- reading: `えきのかいさつでともだちとわかれた`
- context: `(無し)`

### general_014_hand
- reading: `まいにちおそくまでべんきょうしている`
- context: `試験が近いので、`

### general_015_hand
- reading: `つうきんでんしゃはいつもこんでいる`
- context: `(無し)`

### general_016_hand
- reading: `しんせんなやさいをすーぱーでかってきた`
- context: `(無し)`

### general_017_hand
- reading: `だいどころでりょうりをしているにおいがする`
- context: `(無し)`

### general_018_hand
- reading: `じてんしゃでじゅっぷんかかる`
- context: `駅まで`

### general_019_hand
- reading: `やまみちをのぼるとけしきがひろがった`
- context: `(無し)`

### general_020_hand
- reading: `うみでおよいでひやけした`
- context: `夏休みに`

### general_021_hand
- reading: `あたらしいぷろじぇくとがはじまるのがたのしみだ`
- context: `(無し)`

### general_022_hand
- reading: `かれのはなしはいつもおもしろい`
- context: `(無し)`

### general_023_hand
- reading: `にわにきれいなはながさいている`
- context: `(無し)`

### general_024_hand
- reading: `かぞくでおんせんりょこうをけいかくしている`
- context: `(無し)`

### general_025_hand
- reading: `しんゆうとひさしぶりにさけをのんだ`
- context: `昔からの`

### general_026_hand
- reading: `あさのじょぎんぐをしゅうかんにしている`
- context: `(無し)`

### general_027_hand
- reading: `ゆうはんのあとにかるいさんぽをする`
- context: `(無し)`

### general_028_hand
- reading: `ことしのなつはとくべつあつかった`
- context: `(無し)`

### general_029_hand
- reading: `こどものときのおもいでがよみがえる`
- context: `(無し)`

### general_030_hand
- reading: `しょうがっこうのおんしにさいかいした`
- context: `(無し)`

### general_031_hand
- reading: `いぬがげんきにしっぽをふっていた`
- context: `家に帰ると`

### general_032_hand
- reading: `あねはらいねんけっこんするよていだ`
- context: `(無し)`

### general_033_hand
- reading: `りょうしんにあたらしいしょくばをほうこくした`
- context: `(無し)`

### general_034_hand
- reading: `かぞくのきゅうじつがそろうのはひさしぶりだ`
- context: `(無し)`

### general_035_hand
- reading: `さいきんえいがかんへいくことがへった`
- context: `(無し)`

### general_036_hand
- reading: `やけいがとてもきれいでゆうめいだ`
- context: `この辺りは`

### general_037_hand
- reading: `ひさしぶりにじっかへかえることにした`
- context: `(無し)`

### general_038_hand
- reading: `あさからばんまでかいぎづめでつかれた`
- context: `(無し)`

### general_039_hand
- reading: `きのうのしあいはひきわけにおわった`
- context: `(無し)`

### general_040_hand
- reading: `あたらしくはじめたしゅみがたのしい`
- context: `(無し)`

### general_041_wiki
- reading: `くわえて、りょうしゅの、じゅうみんにたいするせきにんこういはさほどおおくなかった。`
- context: `た場合、物価の上昇や、時代の遷移にかかわらず、条件を変更することはできなかった。`

### general_042_wiki
- reading: `そんなになったからにはいきばっていてはいけないよ。`
- context: `まあ大変に窶れているじゃあないか。`

### general_043_wiki
- reading: `てったいするときは、ぜんそくりょくで。`
- context: `最悪なのは、ものごとにこだわりすぎ、致命傷になるまで深追いしてしまうことだ。`

### general_044_wiki
- reading: `おっととともにらーめんてんをきりもりしている。`
- context: `蓮太郎の母。`

### general_045_wiki
- reading: `えんたーぷらいずごうのこうほうし。`
- context: `地球人男性。`

### general_046_wiki
- reading: `そのときけんしんもうさるるやう`
- context: `川中島に出陣あり`

### general_047_wiki
- reading: `やどやのてんしゅは、いつものようにあたらしいふつうのわいんをかわずこうかなものをえらんだりゆうはなんかときいた。`
- context: `奴隷はいつも以上に丁寧にワインを試飲し、より良い品質のものを注文した。`

### general_048_wiki
- reading: `ろうぎのすいそくはじぶんだけのこころにしかわからなかったのであろう。`
- context: `僅少の貯蓄で夫妻が冷たくなろうとは思われる理由がない。`

### general_049_wiki
- reading: `わたしはおとぎばなしでもきくようなきになってこのはなしをきいていた。`
- context: `いいか……忘れるな……」`

### general_050_wiki
- reading: `「もちろん、さようでございます。`
- context: `「それでは、父が、厭じゃと申した時に、何うもならんではないか？」`

### general_051_wiki
- reading: `おゆきは、ごむのながぐつであさつゆをふくんだしだをふみながらわたしののちをついふてきた。`
- context: `ほんとうにさつき雉を見たんだから……」`

### general_052_wiki
- reading: `かのじょがつとめていたじぶんにもでんわをかけると、じょうって、おんなしゅのこえでれいたんに、`
- context: `、その家へ電話をかけて女主人の都合を問い合わすと、いつも留守という返事であった。`

### general_053_wiki
- reading: `すいてんろうし（すいてんろうし）`
- context: `死に際に拳法書を村の男に託す。`

### general_054_wiki
- reading: `このようにひわいなのうりょくをもっているが、はるひさじしんのせいてきなものへのかんしんはうすい。`
- context: `また、絶頂除霊以外の術技は最低レベルで、まともに使えない。`

### general_055_wiki
- reading: `ちゅうごく・たいわんにんににっぽんのやっきょくをしょうかいしたらだいにんきに！`
- context: `台湾各地でセミナー、日本国内のメーカーに対する訪日コンサルティングを行っている。`

### general_056_wiki
- reading: `よるになりつきがしずんであんやとなるとつきをまねきよせ、つきよとしたとのでんせつがある。`
- context: `地で泊まった空海（弘法大師）が、水がない衆生の不便を感じて加持し清水を湧かせた。`

### general_057_wiki
- reading: `えだまめのむねあてとよばれるきみどりいろのみずたまぶらじゃーをちゃくよう。`
- context: `なお、後頭部には様々なシールが貼られていることが多い。`

### general_058_wiki
- reading: `そのごしゅをのみほうろうしているところを、おーしゃんさいどじゅうみんによってひそかにさつがいされる。`
- context: `アーロンの負傷を招いて追放される。`

### general_059_wiki
- reading: `「こういうときには、のべちゃんもきをきかして、さけてくれればかいに」`
- context: `とお倉も姉娘の後に附いて言った。`

### general_060_wiki
- reading: `ていまいたちのはなしでは、すうがくがとくにとくいだったとのこと。`
- context: `勉強も優秀であり、学生時代は首席だった。`

### general_061_wiki
- reading: `いきものをなんでもにんぎょうにできるにんぎょうしょくにん。`
- context: `フリジッタ`

### general_062_wiki
- reading: `ぷらんてんをだえんけいにかっとし、かねちゃいろになるまであげたものである。`
- context: `プランテンは西アフリカでも一般的な食材である。`

### general_063_wiki
- reading: `しかし、くのうときのうしょうがいはつねにかんれんしているわけではない。`
- context: `るが、個人の生活の中で機能障害を正確に認識するために有用と評価することもできる。`

### general_064_wiki
- reading: `とやすたろうにとわれて、おすずはまたあかくなって、くびをふった。`
- context: `って出かける時は、チョウチンもつけていたろうから、年かっこうぐらい見えたろうね」`

### general_065_wiki
- reading: `なるほど、それはたくみいおもいつきだ。`
- context: `ではお前はここまでお話しを買いに来たのか。`

### general_066_wiki
- reading: `にじめぐりにじさんじのまちめぐり`
- context: `コラボミニゲームが配信。`

### general_067_wiki
- reading: `しょうぎぶいんながらこうじゅつのかこからいごもつよく、なみのちゅうがくせいあいてならあっとうするじつりょくをもつ。`
- context: `葉瀬中内では「泣く子も黙る加賀」と呼ばれ恐れられる。`

### general_068_wiki
- reading: `こういうはなしぶりのまに、もすくではしんしがめぐりあうことのなかったゆるやかさですもーりぬぃでのときがけいかした。`
- context: `誰からそのことをききましたか？」`

### general_069_wiki
- reading: `にゅーすえぶりぃない「えぶりぃとくしゅう・しょくひせつやくせいかつ」（にっぽんてれび）`
- context: `ニュースエブリィ内「エブリィ特集・お取り寄せ物産展」`

### general_070_wiki
- reading: `「それでそのしょてんのほうはぶじなのかね」`
- context: `咄嗟の間に死んだ女の所天の事が聞いて見たくなる。`

### general_071_wiki
- reading: `にとろぜっかじょうをかんじゃがのみこんだ。`
- context: `兵庫区薬剤師会ホームページ・トピックス`

### general_072_wiki
- reading: `いくらかのぶっしをたいひさせたあとにひをつけ、あめりかぐんじょうりくぶたいがふねをかくほできるまえにばくはつした。`
- context: `で劣勢になった戦隊が港に留まり、さらに強力なものを建造するというパターンだった。`

### general_073_wiki
- reading: `あいすほっけーかいのえいきゅうけつばん`
- context: `ズ、ダラス・カウボーイズ、ラスベガス・レイダースは永久欠番制度を導入していない。`

### general_074_wiki
- reading: `げんどうきつけじてんしゃ（こうばんようすくーたーなど）`
- context: `警ら用自転車`

### general_075_wiki
- reading: `また、いほうばっさいやしんりんかさい、びょうがいちゅうやがいらいしゅなどにより、しんりんのしつやきのうがていかしている。`
- context: `の主な原因は、農業や牧畜などのための開発や転用であり、特に熱帯地域で顕著である。`

## homophone (n=37)

### homophone_001_hand
- reading: `いしからくすりをしょほうされた`
- context: `風邪を引いたので、`

### homophone_002_hand
- reading: `かれはつよいいしをもってけつだんした`
- context: `プロジェクトの成功に向けて`

### homophone_003_hand
- reading: `そうほうのいしのそつうがじゅうようだ`
- context: `契約交渉では`

### homophone_004_hand
- reading: `にわのちゅうしんにおおきないしがある`
- context: `我が家の`

### homophone_005_hand
- reading: `きしゃがきしゃできしゃした`
- context: `新聞社の`

### homophone_006_hand
- reading: `かがくのじっけんをおこなった`
- context: `学校では`

### homophone_007_hand
- reading: `かがくのぶんせきをおこなった`
- context: `研究室では`

### homophone_008_hand
- reading: `こうえんでこうえんをきいた`
- context: `休日の午後は`

### homophone_009_hand
- reading: `かのじょはこうえんかいでこうえんした`
- context: `演台に立った`

### homophone_010_hand
- reading: `かれのきぎょうけいかくがせいこうした`
- context: `新聞によると`

### homophone_011_hand
- reading: `ゆうめいなきぎょうにしゅうしょくがきまった`
- context: `大学卒業後、`

### homophone_012_hand
- reading: `かんしんなことにかんしんをもった`
- context: `社会問題に対して`

### homophone_013_hand
- reading: `しゅうりょうはしゅうりょうまでにおえた`
- context: `新商品の`

### homophone_014_hand
- reading: `じこかいけつができないじこもある`
- context: `台風のため`

### homophone_015_hand
- reading: `いじょうがないかいじょうもみせてくださいといった`
- context: `医師は念のため`

### homophone_016_hand
- reading: `きしょうよほうをきいてきしょうした`
- context: `天気予報では`

### homophone_017_hand
- reading: `かれはきしょうもきしょうもわるい`
- context: `最近の`

### homophone_018_hand
- reading: `しんじつとじじつがちがうばあいがある`
- context: `調査によると`

### homophone_019_hand
- reading: `きかいがきかいをうしなった`
- context: `約束の時間に`

### homophone_020_hand
- reading: `おなじかんじのかんじをうけた`
- context: `試験を受けて`

### homophone_021_hand
- reading: `かんじょうてきになってかんじょうをもちくずした`
- context: `会計の前に`

### homophone_022_hand
- reading: `じょうけんとじょうこうのいみがちがう`
- context: `法律の条文で`

### homophone_023_hand
- reading: `はしをつかってはしをわたす`
- context: `料理の時は`

### homophone_024_hand
- reading: `はしをかけるこうじがはじまった`
- context: `川の近くに`

### homophone_025_hand
- reading: `こうかいするこうかいをこうかいした`
- context: `選手は`

### homophone_026_hand
- reading: `かみにおがみをささげる`
- context: `神社の`

### homophone_027_hand
- reading: `かみをきってもらう`
- context: `髪型を変えたいので`

### homophone_028_hand
- reading: `きかいのきかいをうかがった`
- context: `会議の後で`

### homophone_029_hand
- reading: `じぶんのいじをつらぬくいしがためされる`
- context: `面接では`

### homophone_030_hand
- reading: `ははにあまえてはははなしをきかない`
- context: `子供が`

### homophone_031_hand
- reading: `せんせいのいこうをうかがっていこうをかためた`
- context: `試験前に`

### homophone_032_hand
- reading: `しんごうをみてしんこうをやめた`
- context: `交差点で`

### homophone_033_hand
- reading: `いっせいにせいそうがはじまった`
- context: `大会では`

### homophone_034_hand
- reading: `こうていをかくにんしておく`
- context: `山登りの前に`

### homophone_035_hand
- reading: `こうていのこうていをほうもんした`
- context: `王室の`

### homophone_036_hand
- reading: `こうえいがおこなうこうえんのほしゅう`
- context: `地域の`

### homophone_037_hand
- reading: `むらのあるじがきじをきじょうにおいた`
- context: `昔話では`

## names (n=55)

### names_001_hand
- reading: `とうきょうだいがくにごうかくしたしんゆう`
- context: `(無し)`

### names_002_hand
- reading: `きょうとのきんかくじをけんがくした`
- context: `(無し)`

### names_003_hand
- reading: `ほっかいどうのさっぽろしではゆきまつりがある`
- context: `(無し)`

### names_004_hand
- reading: `なごやしのめいてつひゃっかてんでかいものした`
- context: `(無し)`

### names_005_hand
- reading: `おおさかじょうからしがいをみおろした`
- context: `(無し)`

### names_006_hand
- reading: `ふじさんはせかいいさんにとうろくされている`
- context: `(無し)`

### names_007_hand
- reading: `おきなわけんのしゅりじょうをほうもんした`
- context: `(無し)`

### names_008_hand
- reading: `かながわけんのよこはまえきではじめてあった`
- context: `(無し)`

### names_009_hand
- reading: `くまもとけんのあそさんでとざんをたのしんだ`
- context: `(無し)`

### names_010_hand
- reading: `とうきょうとしぶやくはちこうまえであいましょう`
- context: `(無し)`

### names_011_hand
- reading: `ひろしまへいわきねんこうえんをおとずれた`
- context: `(無し)`

### names_012_hand
- reading: `ならけんのとうだいじでだいぶつをおがんだ`
- context: `(無し)`

### names_013_hand
- reading: `いばらきけんのつくばだいがくにしんがくした`
- context: `(無し)`

### names_014_hand
- reading: `かごしまけんのさくらじまがふんかした`
- context: `(無し)`

### names_015_hand
- reading: `せんだいしのたなばたまつりがゆうめいだ`
- context: `(無し)`

### names_016_hand
- reading: `とっとりさきゅうでらくだにのった`
- context: `(無し)`

### names_017_hand
- reading: `いしかわけんのけんろくえんへいきたい`
- context: `来年は`

### names_018_hand
- reading: `あおもりけんのねぶたまつりがみたい`
- context: `(無し)`

### names_019_hand
- reading: `しんおおさかからしんよこはままでいどうした`
- context: `新幹線で`

### names_020_hand
- reading: `ながのけんのまつもとじょうをけんがくした`
- context: `(無し)`

### names_021_hand
- reading: `おりんぴっくがとうきょうでかいさいされた`
- context: `(無し)`

### names_022_hand
- reading: `さんふらんしすこのぐーぐるほんしゃをほうもんした`
- context: `(無し)`

### names_023_hand
- reading: `にゅーよーくのせんとらるぱーくをさんぽした`
- context: `(無し)`

### names_024_hand
- reading: `ぱりのえっふぇるとうはよぞらにひかっていた`
- context: `(無し)`

### names_025_hand
- reading: `ろんどんのばっきんがむきゅうでんをけんがくした`
- context: `(無し)`

### names_026_hand
- reading: `あめりかのしりこんばれーではたらく`
- context: `(無し)`

### names_027_hand
- reading: `とよたじどうしゃがでんきじどうしゃをはっぴょうした`
- context: `(無し)`

### names_028_hand
- reading: `そにーぐるーぷのしんしゃちょうがしゅうにんした`
- context: `(無し)`

### names_029_hand
- reading: `みついふどうさんのきじがけいさいされた`
- context: `業界紙に`

### names_030_hand
- reading: `こくれんそうかいでこうえんをおこなった`
- context: `日本の総理は`

### names_031_wiki
- reading: `あなんしつばきちょうのむじんとうでたちばなわんないにいちする。`
- context: `(無し)`

### names_032_wiki
- reading: `だいとしいがいはおおがきちくしょうぼうくみあいがある。`
- context: `導入に合わせ、東京消防庁、京都市消防局、名古屋市消防局など大都市圏に配備された。`

### names_033_wiki
- reading: `しほろちょう（しほろちょう）は、ほっかいどうかとうぐんにあるまち。`
- context: `(無し)`

### names_034_wiki
- reading: `ふくとめむら（ふくとめむら）は、いしかわけんいしかわぐんにあったむら。`
- context: `(無し)`

### names_035_wiki
- reading: `しかし、このくににはけんりょく、せいふ、せんそう、ほうりつ、けいばつなどということばがまるでないのです。`
- context: `、商業のこと、工業のこと、学術のことなど、知っていることを全部話してやりました。`

### names_036_wiki
- reading: `ほっかいどうなかしべつのうぎょうこうとうがっこう`
- context: `計根別郵便局`

### names_037_wiki
- reading: `いてんしたせいいずみてらいがいはげんそんする。`
- context: `幕末には、坂上に次のような寺院が建ち並んでいた。`

### names_038_wiki
- reading: `えのもとせいき（もとやまがたけんつるおかしちょう）`
- context: `足達兼一郎（元山形県鶴岡市長）`

### names_039_wiki
- reading: `さっぽろしりつふくいのなかがっこう`
- context: `札幌市立発寒中学校`

### names_040_wiki
- reading: `こじまかずし（けんぽう、もととうほくだいがくきょうじゅ）`
- context: `黒沼悦郎（商法、早稲田大学大学院法務研究科教授）`

### names_041_wiki
- reading: `げんしょくはちゅうおうだいがくぶんがくぶじんぶんしゃかいがっかきょうじゅ。`
- context: `サムネイル`

### names_042_wiki
- reading: `のーざん・ぱしふぃっくてつどうのとうぶではしゅにんぎしになった。`
- context: `戦後の行動`

### names_043_wiki
- reading: `えすかーどうしくえきまえはいつだんち`
- context: `竹園第二団地`

### names_044_wiki
- reading: `ついめぐる（おいまわし）は、あきたけんよこてしのまちてい。`
- context: `(無し)`

### names_045_wiki
- reading: `きこくご、なむさんだいがくこうし。`
- context: `同年、南山大学助手。`

### names_046_wiki
- reading: `また、ないぶのどうくつはみずがたまるためすべりやすく、じょせいではしょうしょうくろうするほどのけわしいみちとなっている。`
- context: `定期的に浮き沈みを繰り返す特徴を持つため発見は困難。`

### names_047_wiki
- reading: `わきみさきびーちろっくわきみさきのびーちろっくながさきけんのぶんかざい`
- context: `樺島灯台公園`

### names_048_wiki
- reading: `いつおのようねんき、てじまかはながおかぐんこくふむらにうつり、ちょうなん・だいちょうがいぎょうでかけいをささえた。`
- context: `武士・手嶋俊蔵（増魚）の三男であった。`

### names_049_wiki
- reading: `さかがみたむらまろがえぞせいばつのさい、てきをおびきよせるため、ねぶたをうみにながしたことにゆらいするというせつ。`
- context: `坂上田村麻呂説`

### names_050_wiki
- reading: `この「げんざいのまいそうちのこうほう」にはまちのきょうどうぼちはふくまれていなかった。`
- context: `ェイとチェンバーズ・ストリートの角からフォーリー・スクエアまで北東に伸びていた。`

### names_051_wiki
- reading: `けたいのゆえに、みやこのじょうよりはなれてかれらすいぐんの`
- context: `さるを今はた＊總帥の過失、並びに士卒らの`

### names_052_wiki
- reading: `ふなばししりつまえばるちゅうがっこう「まえばら」（「まいはら」からかいしょう）`
- context: `船橋市立前原小学校「まえはら」`

### names_053_wiki
- reading: `おおやむら（おおやむら）は、さいたまけんいるまぐんにそんざいしたむら。`
- context: `(無し)`

### names_054_wiki
- reading: `せいふやたのせいりょくがひそかにそれをしょうれいしているようなみつゆだって、あるかもしれないのよ」`
- context: `ら、あなたのカンは正しいかも知れないけど、密輸品にもピンからキリまであるのです。`

### names_055_wiki
- reading: `ぼしょはとうきょうとみなとくのたつはるてら。`
- context: `東京府生まれ。`

## numeric (n=65)

### numeric_001_hand
- reading: `さんじゅうにんがしゅっせきするよていだ`
- context: `会議の案内を見ると、`

### numeric_002_hand
- reading: `しがつついたちにかいてんする`
- context: `新しい店舗は、`

### numeric_003_hand
- reading: `じゅうごにんのえんじにあがさんかしている`
- context: `プロジェクトには`

### numeric_004_hand
- reading: `にじゅうよじかんたいせいでたいおうする`
- context: `(無し)`

### numeric_005_hand
- reading: `ひゃくにんをこえていた`
- context: `駅前の行列は`

### numeric_006_hand
- reading: `せんきゅうひゃくはちじゅうごねんにうまれた`
- context: `(無し)`

### numeric_007_hand
- reading: `さんぜんこのしょうひんがほかんされている`
- context: `倉庫には`

### numeric_008_hand
- reading: `いちまんえんさつをごまいりょうがえした`
- context: `(無し)`

### numeric_009_hand
- reading: `いちまんにんのらんなーがさんかした`
- context: `今回のマラソンには`

### numeric_010_hand
- reading: `よんひゃくごじゅうみりりっとるのみずをのんだ`
- context: `(無し)`

### numeric_011_hand
- reading: `ごまんにんにたっした`
- context: `試合の観客は`

### numeric_012_hand
- reading: `きおんがさんじゅうごどをこえた`
- context: `(無し)`

### numeric_013_hand
- reading: `ろくせんまんえんだった`
- context: `新築マンションの価格は`

### numeric_014_hand
- reading: `さんびゃくごじゅうめーとるあるらしい`
- context: `この湖の深さは`

### numeric_015_hand
- reading: `ひゃくにじゅうばんのばすにのった`
- context: `(無し)`

### numeric_016_hand
- reading: `ごーるまえひゃくめーとるでぬかれた`
- context: `(無し)`

### numeric_017_hand
- reading: `こうそくどうろをひゃくじゅっきろでそうこうする`
- context: `(無し)`

### numeric_018_hand
- reading: `ついかとうしでにせんおくえんがひつようだ`
- context: `(無し)`

### numeric_019_hand
- reading: `さんじゅっきろぐらむをこえた`
- context: `荷物の総重量は`

### numeric_020_hand
- reading: `さんじかんはんのえいがをみた`
- context: `(無し)`

### numeric_021_hand
- reading: `よねんかんのだいがくせいかつをおえた`
- context: `(無し)`

### numeric_022_hand
- reading: `にちようひんのうりあげがはちじゅっぱーせんとをしめる`
- context: `(無し)`

### numeric_023_hand
- reading: `ことしのうりあげはきょねんひひゃくにじゅっぱーせんとだ`
- context: `(無し)`

### numeric_024_hand
- reading: `あんけーとのかいとうりつはろくじゅっぱーせんとだった`
- context: `発表によると`

### numeric_025_hand
- reading: `しゃちょうのねんれいはろくじゅうきゅうさいだ`
- context: `(無し)`

### numeric_026_hand
- reading: `あかちゃんのたいじゅうはさんきろだった`
- context: `(無し)`

### numeric_027_hand
- reading: `はいたつはなのかごにとうちゃくします`
- context: `(無し)`

### numeric_028_hand
- reading: `じゅうがつにはんきけっさんをおこなう`
- context: `(無し)`

### numeric_029_hand
- reading: `にじゅっせいきのおわりまでにけいざいせいちょうがつづいた`
- context: `(無し)`

### numeric_030_hand
- reading: `えーふぉーのようしをじゅうまいいんさつした`
- context: `(無し)`

### numeric_031_hand
- reading: `ぴーえいちななのすいようえきをつくる`
- context: `(無し)`

### numeric_032_hand
- reading: `しゅうりひはさんまんにせんえんだった`
- context: `(無し)`

### numeric_033_hand
- reading: `ふたつめのぎだいにすすみましょう`
- context: `(無し)`

### numeric_034_hand
- reading: `ろくがつよっかにしけんがある`
- context: `(無し)`

### numeric_035_hand
- reading: `はちじはっぷんちゃくでよていどおりだった`
- context: `電車は`

### numeric_036_hand
- reading: `ほんをひゃくさつよみたい`
- context: `(無し)`

### numeric_037_hand
- reading: `むいかめのれんぞくきんむがきつい`
- context: `(無し)`

### numeric_038_hand
- reading: `だうんろーどじかんはにびょうだった`
- context: `(無し)`

### numeric_039_hand
- reading: `えーあいのおうとうじかんはじゅうびょういないだ`
- context: `(無し)`

### numeric_040_hand
- reading: `にじゅっぷんしかない`
- context: `試験の制限時間は`

### numeric_041_wiki
- reading: `それこそぼくいちにんのめいわくではありませんからね。」`
- context: `ければ、僕も喜んで話しますが——万一秘密の洩れた事が、山県公にでも知れて見給え。`

### numeric_042_wiki
- reading: `くろまじゅつぶのこういってん。`
- context: `黒魔術部。`

### numeric_043_wiki
- reading: `そしてしきりにそういちろうのしたいをのぞきこんでいた。`
- context: `そのとき検事と署長とは、踏台の上に抱き合うようにして乗っていた。`

### numeric_044_wiki
- reading: `こころざしちくはちゅうごくのおんみょうごぎょうせつのえいきょうものこしていた。`
- context: `は游気を水蒸気の意味でも用いており、今日の空気と全く同じというわけではなかった。`

### numeric_045_wiki
- reading: `おおしおはちまんぐう（ふくいけんえちぜんし）`
- context: `若宮八幡宮（福井県白山市）`

### numeric_046_wiki
- reading: `「たはちはすぐにまじめさうなかおをするんでいやになるな。`
- context: `さうに腕組をすると、さつき何か云ひ出さうとした見るからに元気者らしい剽軽な男は、`

### numeric_047_wiki
- reading: `そのご、いちじてきにかーとれーすへふっきし「すぺいん・かーとせんしゅけん」へしゅつじょうする。`
- context: `さらに「フォーミュラ・ルノー・ユーロカップ」への参戦も決まる。`

### numeric_048_wiki
- reading: `「としまそんし」および「こくせいちょうさ」によればくちのしまのじんこうのせんいはかきのとおりである。`
- context: `南限の植物`

### numeric_049_wiki
- reading: `これをかいこんしようというくわだてがさいさんしじんによっておこなわれたが、いずれもしっぱいした。`
- context: `例証されているが、この地は、マン僧正によれば１）、本来最も乾燥した砂地であった。`

### numeric_050_wiki
- reading: `ごがつ（しりーずてーま「いさんそうぞく」）`
- context: `弥生は看護師を辞めてハナを野田家に引き取り自分で介護をすることに決める。`

### numeric_051_wiki
- reading: `「あしがまんぞくであったころには、ごにんであろうと、じゅうにんであろうと、なでぎりにしたものだったが」`
- context: `と撞木杖の武士は事も無げに、`

### numeric_052_wiki
- reading: `とみれば、みちばたのしばのじょうにおかれたけんじゅつのどうぐいちくみ。`
- context: `お松は秋の情景をほしいままにして、山と畑との勾配ゆるやかな道を歩みました。`

### numeric_053_wiki
- reading: `ひばくち・ひろしまの「きおくのけいしょう」をてーまに、「かこ」「げんざい」「みらい」のさんぶこうせいでてんかいしていく。`
- context: `共演に松永有紘、有坂心花、入山杏奈、平岡祐太ら。`

### numeric_054_wiki
- reading: `じぶんのへやといううちに、このこはへやをふたつもっている。`
- context: `こう言って、茂太郎は、おとなしく、自分の部屋に戻りました。`

### numeric_055_wiki
- reading: `これは、なにせんねんかむかしのできごとであるとどうじに、また、このしゅんかんのげんじつごとでもあった。`
- context: `そして、それはいま、タミル族の碩学ヤトラカン・サミ博士に伝わっているのだ。`

### numeric_056_wiki
- reading: `……しかしはっぽうとりまかれてしまった」`
- context: `……どうともして早くここを遁れ。`

### numeric_057_wiki
- reading: `こうそうびるぐんにあるまんしょんのいっしつでちかしゃげきじょうなどはそんざいしない。`
- context: `中、パイソンとレッドホークが一緒に登場するときがあるがおそらくこれはミスである。`

### numeric_058_wiki
- reading: `えんしゅつはみいけたかし、きゃくほんはりりー・ふらんきー、しゅえんはいちかわえびぞう。`
- context: `主演は松平健。`

### numeric_059_wiki
- reading: `だいとくさんねん、もっごじょうかわかんぐんしつなみいりしょうかい」。`
- context: `」の総称で、漢人・女直人・契丹人といった旧金朝領の住民を主体とする軍団であった。`

### numeric_060_wiki
- reading: `もでるがんはけんじゅうがたやらいふるがた、ましんがんがたなどじゅうすうちょうがおうしゅうされた。`
- context: `訓練された人間の撃ち方といえる」と評している。`

### numeric_061_wiki
- reading: `そのご、おすやすしんくせいふは、しんきのふどうさんはんばいをいちじてきにきんしした。`
- context: `新区の発表直後、この地域の不動産価格が急速に高騰している。`

### numeric_062_wiki
- reading: `ちゅうごくかほくしょうのばんりのちょうじょうにほどちかい、たいがにめんしたむら。`
- context: `ストーリー`

### numeric_063_wiki
- reading: `ほんみょうはさんこうぎん（みよししろがね）。`
- context: `静岡県出身。`

### numeric_064_wiki
- reading: `はかんとくのはらたつのりからきたいされ、かいまくいちぐんめんばーいりをした。`
- context: `千葉ロッテマリーンズなどが獲得を検討したものの指名球団はなく日本生命に進む。`

### numeric_065_wiki
- reading: `はままつしはまなくのせいぶ、みっかびちくのなんぶにいちする。`
- context: `住居表示未実施。`

## particle (n=32)

### particle_001_hand
- reading: `かれはうまれつきのさいのうがある`
- context: `(無し)`

### particle_002_hand
- reading: `たいふうがひがしへすすんでいる`
- context: `(無し)`

### particle_003_hand
- reading: `わたしはにほんごをべんきょうしています`
- context: `(無し)`

### particle_004_hand
- reading: `こどもがとしょかんへほんをかえしにいった`
- context: `(無し)`

### particle_005_hand
- reading: `かれにこのしょるいをわたしてください`
- context: `(無し)`

### particle_006_hand
- reading: `きょうとにりょこうするのがたのしみだ`
- context: `(無し)`

### particle_007_hand
- reading: `とうきょうからおおさかへいどうする`
- context: `(無し)`

### particle_008_hand
- reading: `あめにもまけずかぜにもまけない`
- context: `(無し)`

### particle_009_hand
- reading: `ともだちとえいがをみにいった`
- context: `(無し)`

### particle_010_hand
- reading: `いぬをさんぽにつれていく`
- context: `(無し)`

### particle_011_hand
- reading: `かのじょはかれよりもせがたかい`
- context: `(無し)`

### particle_012_hand
- reading: `ねこばかりがきにいっている`
- context: `(無し)`

### particle_013_hand
- reading: `あかいぺんだけでかいてください`
- context: `(無し)`

### particle_014_hand
- reading: `ともだちやかぞくもしょうたいした`
- context: `(無し)`

### particle_015_hand
- reading: `しごとのあとにすぽーつじむへいく`
- context: `(無し)`

### particle_016_hand
- reading: `りょうしんのためにりょこうをぷれぜんとした`
- context: `(無し)`

### particle_017_hand
- reading: `あめがふっているのでかさがひつようだ`
- context: `(無し)`

### particle_018_hand
- reading: `こうこうせいなのにうんてんめんきょはもっていない`
- context: `(無し)`

### particle_019_hand
- reading: `じこではなくこしょうでちこくした`
- context: `(無し)`

### particle_020_hand
- reading: `あなたがきめるならわたしもさんせいします`
- context: `(無し)`

### particle_021_hand
- reading: `ほんやくされなくてもいみはわかる`
- context: `(無し)`

### particle_022_hand
- reading: `かいぎがおわったらすぐにかえる`
- context: `(無し)`

### particle_023_hand
- reading: `ねつがあがったのでびょういんへいった`
- context: `(無し)`

### particle_024_hand
- reading: `でんしゃがおくれたせいでちこくした`
- context: `(無し)`

### particle_025_hand
- reading: `はるになるとさくらがさく`
- context: `(無し)`

### particle_026_hand
- reading: `あめがふればしあいはちゅうしだ`
- context: `(無し)`

### particle_027_hand
- reading: `べんきょうすればするほどおもしろくなる`
- context: `(無し)`

### particle_028_hand
- reading: `ちちはさけをのまないのでくるまをうんてんできる`
- context: `(無し)`

### particle_029_hand
- reading: `じかんがなくてもあさごはんはたべる`
- context: `(無し)`

### particle_030_hand
- reading: `かれはいっしょうけんめいはたらいている`
- context: `(無し)`

### particle_031_hand
- reading: `わからないてんがあればしつもんしてください`
- context: `(無し)`

### particle_032_hand
- reading: `てつだってもらえるとたすかる`
- context: `(無し)`

## tech (n=44)

### tech_001_hand
- reading: `ぷろぐらみんぐげんごのぱいそんをまなんでいる`
- context: `(無し)`

### tech_002_hand
- reading: `でーたべーすのすきーまをせっけいする`
- context: `(無し)`

### tech_003_hand
- reading: `りれーしょなるでーたべーすでじょいんくえりーをかく`
- context: `(無し)`

### tech_004_hand
- reading: `きかいがくしゅうのもでるをふぁいんちゅーにんぐする`
- context: `(無し)`

### tech_005_hand
- reading: `てんそるふろーでにゅーらるねっとわーくをこうちくした`
- context: `(無し)`

### tech_006_hand
- reading: `こうばいこうかほうでさいてきかする`
- context: `(無し)`

### tech_007_hand
- reading: `ばっくぷろぱげーしょんのりろんをりかいする`
- context: `(無し)`

### tech_008_hand
- reading: `ちゅうたたみこみそうのぱらめーたをちょうせいする`
- context: `(無し)`

### tech_009_hand
- reading: `しぜんげんごしょりではとらんすふぉーまーがしゅりゅうだ`
- context: `最近の`

### tech_010_hand
- reading: `あてんしょんきこうがせいのうのかぎになる`
- context: `(無し)`

### tech_011_hand
- reading: `にゅーらるねっとのそうすうをふやしてじっけんした`
- context: `(無し)`

### tech_012_hand
- reading: `どっかーこんてなでかいはつかんきょうをこうちくする`
- context: `(無し)`

### tech_013_hand
- reading: `くばねてぃすくらすたーにでぷろいした`
- context: `(無し)`

### tech_014_hand
- reading: `まいくろさーびすあーきてくちゃをどうにゅうする`
- context: `(無し)`

### tech_015_hand
- reading: `ろーどばらんさーがかどうしていない`
- context: `(無し)`

### tech_016_hand
- reading: `しーあいしーでぃーぱいぷらいんをこうちくする`
- context: `(無し)`

### tech_017_hand
- reading: `こーどれびゅーでばぐをみつけた`
- context: `(無し)`

### tech_018_hand
- reading: `ゆにっとてすとのかばれっじをあげる`
- context: `(無し)`

### tech_019_hand
- reading: `りふぁくたりんぐでこーどのじょうちょうせいをへらした`
- context: `(無し)`

### tech_020_hand
- reading: `ぷるりくえすとがまーじされた`
- context: `(無し)`

### tech_021_hand
- reading: `おーぷんそーすこみゅにてぃにこんとりびゅーとする`
- context: `(無し)`

### tech_022_hand
- reading: `けんげんかんりしすてむをてすとした`
- context: `(無し)`

### tech_023_hand
- reading: `ぶんさんしすてむのかようせいをけんしょうする`
- context: `(無し)`

### tech_024_hand
- reading: `でっどろっくをけんしゅつするあるごりずむ`
- context: `(無し)`

### tech_025_hand
- reading: `いんめもりきゃっしゅでそくどをあげるせっけいをおこなう`
- context: `(無し)`

### tech_026_hand
- reading: `くらうどほすてぃんぐにいこうしている`
- context: `会社のサーバは`

### tech_027_hand
- reading: `あいぴーあどれすとぽーとばんごうをかくにんする`
- context: `(無し)`

### tech_028_hand
- reading: `ろぎんぐらいぶらりをつかってでばっぐする`
- context: `(無し)`

### tech_029_hand
- reading: `ぷろふぁいらーでぼとるねっくをしらべる`
- context: `(無し)`

### tech_030_hand
- reading: `ぶらんちせんりゃくにぎっとふろーをさいようした`
- context: `(無し)`

### tech_031_wiki
- reading: `しばしば、こんぴゅーたーのきどうじに、じどうてきにきどうされるようにせっていをしてりようするものがおおい。`
- context: `オープンソース 向けの利用にも対応しているものもある。`

### tech_032_wiki
- reading: `どうりょくはすいへいたいこうぴすとんしきでぃーぜるえんじん。`
- context: `ゆるい傾斜の前面以外はほぼ垂直に近い箱組車体である。`

### tech_033_wiki
- reading: `しょうにんのねっとわーくではーれむにしょうひんをとりよせるやくめをたんとうしている。`
- context: `カーリーのご利益に預かっているが、それを抜きにしても彼女へ強い好意を抱いている。`

### tech_034_wiki
- reading: `しちょうそんはいかとなる（しちょうそんこーどじゅん）。`
- context: `おおむね広義の「山原（やんばる）」と呼ばれる地域に相当する。`

### tech_035_wiki
- reading: `このため、すーぱーどるふぃーのぼでぃとくみあわせたさくれいがいくつかいんたーねっとじょうにこうかいされている。`
- context: `スリーサイズが異なるため、衣装の互換性はない。`

### tech_036_wiki
- reading: `けっかとして、そふとうぇあのしんらいせいにかんするようきゅうしようがうまれた。`
- context: `その原因はユーザインタフェースの貧弱さから直接的なバグまで様々である。`

### tech_037_wiki
- reading: `こーどほかん、せいせい、ちゃっときのうなどをていきょう。`
- context: `プロジェクト全体の解析やターミナル操作が可能。`

### tech_038_wiki
- reading: `くどうしすてむはちゅうくうじくへいこうかるだんをさいようした。`
- context: `異なるが、磁気回路の工夫などにより出力特性が極力同一となるように設計されている。`

### tech_039_wiki
- reading: `とくしまぶんりだいがくこうがくぶのじょうほうしすてむこうがくかをでんしじょうほうこうがくかにめいしょうへんこう。`
- context: `徳島文理大学工学部の機械電子工学科を機械創造工学科に名称変更。`

### tech_040_wiki
- reading: `ぶれーきはぶれんぼのぶれーきしすてむをさいようすることでせいどうせいもたかめている。`
- context: `スペンションには専用チューニングを施しており、動力性能を支えるように調整された。`

### tech_041_wiki
- reading: `いけだけーぶるねっとわーくをのぞくぜんけーぶるてれびきょく（いけだはせとないかいほうそうをさいほうそう）`
- context: `日本海ケーブルネットワーク（鳥取市・岩美町）`

### tech_042_wiki
- reading: `じゅうようしょうにんをほごするしょうにんほごぷろぐらむのえーじぇんと。`
- context: `ジョン・クルーガー`

### tech_043_wiki
- reading: `なお、でじたるのあさひかわきょくだしふぉんとはさっぽろきょくだしのものよりわずかにかげがこい。`
- context: `され、それ以外では東京または札幌局送出（デジタルは札幌局送出のみ）のものを表示。`

### tech_044_wiki
- reading: `てきおうせいにすぐれたでーたせんたーをめざす`
- context: `インダストリー事業`

