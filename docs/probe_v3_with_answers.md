# probe_v3 with references (答えあり、採点用)

各 item の `expected` に正解候補。frontier LLM の出力を `top1` 列に手書きで埋めて scoring。NFKC 一致 (NFKC 正規化後に references のいずれかと一致) で EM1 算出。

## システム情報

- 元データ: datasets/eval/probe/probe.json (348 items)
- カテゴリ: edge / general / homophone / names / numeric / particle / tech
- スコアリング: `expected` のいずれかと NFKC 一致で正解 (em1_nfkc)

---

## edge (n=40)

### edge_001_hand
- reading: `あいふぉーんとあんどろいどをひかくする`
- context: `(無し)`
- expected:
  - `iPhoneとAndroidを比較する`
  - `アイフォーンとアンドロイドを比較する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_002_hand
- reading: `らいんのめっせーじをうけとった`
- context: `(無し)`
- expected:
  - `LINEのメッセージを受け取った`
  - `ラインのメッセージを受け取った`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_003_hand
- reading: `ぐーぐるまっぷでけんさくする`
- context: `(無し)`
- expected:
  - `Googleマップで検索する`
  - `グーグルマップで検索する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_004_hand
- reading: `ぎっとはぶにりぽじとりをさくせいした`
- context: `(無し)`
- expected:
  - `GitHubにリポジトリを作成した`
  - `ギットハブにリポジトリを作成した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_005_hand
- reading: `ぱいそんのすくりぷとをかく`
- context: `(無し)`
- expected:
  - `Pythonのスクリプトを書く`
  - `パイソンのスクリプトを書く`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_006_hand
- reading: `こーひーをすたーばっくすでかう`
- context: `毎朝の`
- expected:
  - `コーヒーをスターバックスで買う`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_007_hand
- reading: `あまぞんぷらいむでどうがをみる`
- context: `(無し)`
- expected:
  - `Amazonプライムで動画を見る`
  - `アマゾンプライムで動画を見る`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_008_hand
- reading: `ねっとふりっくすのしんさくをみた`
- context: `(無し)`
- expected:
  - `Netflixの新作を見た`
  - `ネットフリックスの新作を見た`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_009_hand
- reading: `おーぷんえーあいのえーぴーあいをつかう`
- context: `(無し)`
- expected:
  - `OpenAIのAPIを使う`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_010_hand
- reading: `ずーむをいんすとーるした`
- context: `会議のために`
- expected:
  - `Zoomをインストールした`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_011_hand
- reading: `ゆーちゅーぶでおんがくをきいている`
- context: `(無し)`
- expected:
  - `YouTubeで音楽を聞いている`
  - `ユーチューブで音楽を聞いている`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_012_hand
- reading: `すたーとあっぷにじょいんした`
- context: `(無し)`
- expected:
  - `スタートアップにジョインした`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_013_hand
- reading: `まっくぶっくぷろでさぎょうをしている`
- context: `(無し)`
- expected:
  - `MacBook Proで作業をしている`
  - `マックブックプロで作業をしている`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_014_hand
- reading: `うぇぶさいとのゆーあいがあたらしくなった`
- context: `(無し)`
- expected:
  - `ウェブサイトのUIが新しくなった`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_015_hand
- reading: `しすてむのあーるぴーえーをどうにゅうする`
- context: `(無し)`
- expected:
  - `システムのRPAを導入する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_016_hand
- reading: `すたっくおーばーふろーでかいけつほうをさがした`
- context: `(無し)`
- expected:
  - `スタックオーバーフローで解決方法を探した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_017_hand
- reading: `もじこーどがゆーてぃーえふえいとになった`
- context: `(無し)`
- expected:
  - `文字コードがUTF-8になった`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_018_hand
- reading: `でーたをしーえすぶいでえくすぽーとする`
- context: `(無し)`
- expected:
  - `データをCSVでエクスポートする`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_019_hand
- reading: `じぇーそんぺーろーどをぱーすする`
- context: `(無し)`
- expected:
  - `JSONペイロードをパースする`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_020_hand
- reading: `でぃーえぬえすせっていをこうしんした`
- context: `(無し)`
- expected:
  - `DNS設定を更新した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_021_hand
- reading: `ぶいぴーえぬせつぞくにじかんがかかる`
- context: `(無し)`
- expected:
  - `VPN接続に時間がかかる`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_022_hand
- reading: `そーしゃるめでぃあぐるーぷをかんりする`
- context: `(無し)`
- expected:
  - `ソーシャルメディアグループを管理する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_023_hand
- reading: `でじたるとらんすふぉーめーしょんをすいしんする`
- context: `(無し)`
- expected:
  - `デジタルトランスフォーメーションを推進する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_024_hand
- reading: `ふろんとえんどふれーむわーくはりあくとをつかう`
- context: `(無し)`
- expected:
  - `フロントエンドフレームワークはReactを使う`
  - `フロントエンドフレームワークはリアクトを使う`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_025_hand
- reading: `すくらむかいはつをどうにゅうした`
- context: `(無し)`
- expected:
  - `スクラム開発を導入した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_026_hand
- reading: `はっかそんにさんかしてぷろとたいぷをつくった`
- context: `(無し)`
- expected:
  - `ハッカソンに参加してプロトタイプを作った`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_027_hand
- reading: `ぶろっくちぇーんぎじゅつにきょうみがある`
- context: `(無し)`
- expected:
  - `ブロックチェーン技術に興味がある`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_028_hand
- reading: `くらうどふぁんでぃんぐでしきんをあつめる`
- context: `(無し)`
- expected:
  - `クラウドファンディングで資金を集める`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_029_hand
- reading: `えすえぬえすでばずってとれんどいりした`
- context: `(無し)`
- expected:
  - `SNSでバズってトレンド入りした`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_030_hand
- reading: `りもーとわーくがすたんだーどになった`
- context: `(無し)`
- expected:
  - `リモートワークがスタンダードになった`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_031_hand
- reading: `えすでぃーじーずのもくひょうをたっせいする`
- context: `(無し)`
- expected:
  - `SDGsの目標を達成する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_032_hand
- reading: `めたばーすでばーちゃるかいぎをひらく`
- context: `(無し)`
- expected:
  - `メタバースでバーチャル会議を開く`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_033_hand
- reading: `ちゃっとじーぴーてぃーをぎょうむにかつようする`
- context: `(無し)`
- expected:
  - `ChatGPTを業務に活用する`
  - `チャットGPTを業務に活用する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_034_hand
- reading: `きゅーあーるこーどでけっさいする`
- context: `(無し)`
- expected:
  - `QRコードで決済する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_035_hand
- reading: `あぷりでちぇっくする`
- context: `朝の交通状況を`
- expected:
  - `アプリでチェックする`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_036_hand
- reading: `えっくすじょうでとうこうがばずった`
- context: `(無し)`
- expected:
  - `X上で投稿がバズった`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_037_hand
- reading: `すらっくのちゃんねるでこうちくをそうだんする`
- context: `(無し)`
- expected:
  - `Slackのチャンネルで構築を相談する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_038_hand
- reading: `てんそるふろーよりぱいとーちをこのむ`
- context: `(無し)`
- expected:
  - `TensorFlowよりPyTorchを好む`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_039_hand
- reading: `あーるえすえすふぃーどでにゅーすをよむ`
- context: `(無し)`
- expected:
  - `RSSフィードでニュースを読む`
- top1: <!-- frontier LLM の出力をここに記入 -->

### edge_040_hand
- reading: `やふーのとぷべーじがこうしんされた`
- context: `(無し)`
- expected:
  - `Yahoo!のトップページが更新された`
- top1: <!-- frontier LLM の出力をここに記入 -->

## general (n=75)

### general_001_hand
- reading: `あたらしいすまーとふぉんをこうにゅうした`
- context: `(無し)`
- expected:
  - `新しいスマートフォンを購入した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_002_hand
- reading: `しゅうまつはかぞくでこうえんへでかけた`
- context: `(無し)`
- expected:
  - `週末は家族で公園へ出かけた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_003_hand
- reading: `きょうはねむくてしごとがはかどらない`
- context: `昨晩は遅くまで残業だったので、`
- expected:
  - `今日は眠くて仕事が捗らない`
  - `今日は眠くて仕事がはかどらない`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_004_hand
- reading: `あしたはゆきになるかもしれない`
- context: `天気予報によると、`
- expected:
  - `明日は雪になるかもしれない`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_005_hand
- reading: `としょかんでべんきょうしているあいだにうとうとした`
- context: `(無し)`
- expected:
  - `図書館で勉強している間にうとうとした`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_006_hand
- reading: `えきまえのかふぇでまちあわせる`
- context: `友人との約束で`
- expected:
  - `駅前のカフェで待ち合わせる`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_007_hand
- reading: `こんばんのこんだてをかんがえる`
- context: `冷蔵庫の中身を見て、`
- expected:
  - `今晩の献立を考える`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_008_hand
- reading: `かのじょはえいごをねっしんにべんきょうしている`
- context: `(無し)`
- expected:
  - `彼女は英語を熱心に勉強している`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_009_hand
- reading: `すきなしゅみのじかんがとれない`
- context: `最近は忙しくて、`
- expected:
  - `好きな趣味の時間が取れない`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_010_hand
- reading: `ちいさいころからえをかくのがすきだった`
- context: `(無し)`
- expected:
  - `小さい頃から絵を描くのが好きだった`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_011_hand
- reading: `こうえんでたのしそうにあそんでいる`
- context: `子供たちが`
- expected:
  - `公園で楽しそうに遊んでいる`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_012_hand
- reading: `あめがつよくふっているのでかさをさす`
- context: `(無し)`
- expected:
  - `雨が強く降っているので傘をさす`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_013_hand
- reading: `えきのかいさつでともだちとわかれた`
- context: `(無し)`
- expected:
  - `駅の改札で友達と別れた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_014_hand
- reading: `まいにちおそくまでべんきょうしている`
- context: `試験が近いので、`
- expected:
  - `毎日遅くまで勉強している`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_015_hand
- reading: `つうきんでんしゃはいつもこんでいる`
- context: `(無し)`
- expected:
  - `通勤電車はいつも混んでいる`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_016_hand
- reading: `しんせんなやさいをすーぱーでかってきた`
- context: `(無し)`
- expected:
  - `新鮮な野菜をスーパーで買ってきた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_017_hand
- reading: `だいどころでりょうりをしているにおいがする`
- context: `(無し)`
- expected:
  - `台所で料理をしている匂いがする`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_018_hand
- reading: `じてんしゃでじゅっぷんかかる`
- context: `駅まで`
- expected:
  - `自転車で10分かかる`
  - `自転車で十分かかる`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_019_hand
- reading: `やまみちをのぼるとけしきがひろがった`
- context: `(無し)`
- expected:
  - `山道を登ると景色が広がった`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_020_hand
- reading: `うみでおよいでひやけした`
- context: `夏休みに`
- expected:
  - `海で泳いで日焼けした`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_021_hand
- reading: `あたらしいぷろじぇくとがはじまるのがたのしみだ`
- context: `(無し)`
- expected:
  - `新しいプロジェクトが始まるのが楽しみだ`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_022_hand
- reading: `かれのはなしはいつもおもしろい`
- context: `(無し)`
- expected:
  - `彼の話はいつも面白い`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_023_hand
- reading: `にわにきれいなはながさいている`
- context: `(無し)`
- expected:
  - `庭に綺麗な花が咲いている`
  - `庭にきれいな花が咲いている`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_024_hand
- reading: `かぞくでおんせんりょこうをけいかくしている`
- context: `(無し)`
- expected:
  - `家族で温泉旅行を計画している`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_025_hand
- reading: `しんゆうとひさしぶりにさけをのんだ`
- context: `昔からの`
- expected:
  - `親友と久しぶりに酒を飲んだ`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_026_hand
- reading: `あさのじょぎんぐをしゅうかんにしている`
- context: `(無し)`
- expected:
  - `朝のジョギングを習慣にしている`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_027_hand
- reading: `ゆうはんのあとにかるいさんぽをする`
- context: `(無し)`
- expected:
  - `夕飯の後に軽い散歩をする`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_028_hand
- reading: `ことしのなつはとくべつあつかった`
- context: `(無し)`
- expected:
  - `今年の夏は特別暑かった`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_029_hand
- reading: `こどものときのおもいでがよみがえる`
- context: `(無し)`
- expected:
  - `子供の時の思い出が蘇る`
  - `子供の時の思い出がよみがえる`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_030_hand
- reading: `しょうがっこうのおんしにさいかいした`
- context: `(無し)`
- expected:
  - `小学校の恩師に再会した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_031_hand
- reading: `いぬがげんきにしっぽをふっていた`
- context: `家に帰ると`
- expected:
  - `犬が元気に尻尾を振っていた`
  - `犬が元気にしっぽを振っていた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_032_hand
- reading: `あねはらいねんけっこんするよていだ`
- context: `(無し)`
- expected:
  - `姉は来年結婚する予定だ`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_033_hand
- reading: `りょうしんにあたらしいしょくばをほうこくした`
- context: `(無し)`
- expected:
  - `両親に新しい職場を報告した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_034_hand
- reading: `かぞくのきゅうじつがそろうのはひさしぶりだ`
- context: `(無し)`
- expected:
  - `家族の休日が揃うのは久しぶりだ`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_035_hand
- reading: `さいきんえいがかんへいくことがへった`
- context: `(無し)`
- expected:
  - `最近映画館へ行くことが減った`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_036_hand
- reading: `やけいがとてもきれいでゆうめいだ`
- context: `この辺りは`
- expected:
  - `夜景がとても綺麗で有名だ`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_037_hand
- reading: `ひさしぶりにじっかへかえることにした`
- context: `(無し)`
- expected:
  - `久しぶりに実家へ帰ることにした`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_038_hand
- reading: `あさからばんまでかいぎづめでつかれた`
- context: `(無し)`
- expected:
  - `朝から晩まで会議詰めで疲れた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_039_hand
- reading: `きのうのしあいはひきわけにおわった`
- context: `(無し)`
- expected:
  - `昨日の試合は引き分けに終わった`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_040_hand
- reading: `あたらしくはじめたしゅみがたのしい`
- context: `(無し)`
- expected:
  - `新しく始めた趣味が楽しい`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_041_wiki
- reading: `くわえて、りょうしゅの、じゅうみんにたいするせきにんこういはさほどおおくなかった。`
- context: `た場合、物価の上昇や、時代の遷移にかかわらず、条件を変更することはできなかった。`
- expected:
  - `加えて、領主の、住民に対する責任行為はさほど多くなかった。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_042_wiki
- reading: `そんなになったからにはいきばっていてはいけないよ。`
- context: `まあ大変に窶れているじゃあないか。`
- expected:
  - `そんなになったからには息張っていては行けないよ。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_043_wiki
- reading: `てったいするときは、ぜんそくりょくで。`
- context: `最悪なのは、ものごとにこだわりすぎ、致命傷になるまで深追いしてしまうことだ。`
- expected:
  - `撤退する時は、全速力で。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_044_wiki
- reading: `おっととともにらーめんてんをきりもりしている。`
- context: `蓮太郎の母。`
- expected:
  - `夫とともにラーメン店を切り盛りしている。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_045_wiki
- reading: `えんたーぷらいずごうのこうほうし。`
- context: `地球人男性。`
- expected:
  - `エンタープライズ号の航法士。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_046_wiki
- reading: `そのときけんしんもうさるるやう`
- context: `川中島に出陣あり`
- expected:
  - `そのとき謙信申さるるやう`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_047_wiki
- reading: `やどやのてんしゅは、いつものようにあたらしいふつうのわいんをかわずこうかなものをえらんだりゆうはなんかときいた。`
- context: `奴隷はいつも以上に丁寧にワインを試飲し、より良い品質のものを注文した。`
- expected:
  - `宿屋の店主は、いつものように新しい普通のワインを買わず高価なものを選んだ理由は何かと聞いた。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_048_wiki
- reading: `ろうぎのすいそくはじぶんだけのこころにしかわからなかったのであろう。`
- context: `僅少の貯蓄で夫妻が冷たくなろうとは思われる理由がない。`
- expected:
  - `老妓の推測は自分だけの心にしかわからなかったのであろう。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_049_wiki
- reading: `わたしはおとぎばなしでもきくようなきになってこのはなしをきいていた。`
- context: `いいか……忘れるな……」`
- expected:
  - `私はお伽噺でも聞くような気になってこの話を聞いていた。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_050_wiki
- reading: `「もちろん、さようでございます。`
- context: `「それでは、父が、厭じゃと申した時に、何うもならんではないか？」`
- expected:
  - `「勿論、左様でございます。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_051_wiki
- reading: `おゆきは、ごむのながぐつであさつゆをふくんだしだをふみながらわたしののちをついふてきた。`
- context: `ほんとうにさつき雉を見たんだから……」`
- expected:
  - `お雪は、ゴムの長靴で朝露を含んだ歯朶を踏みながら私の後を追ふて来た。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_052_wiki
- reading: `かのじょがつとめていたじぶんにもでんわをかけると、じょうって、おんなしゅのこえでれいたんに、`
- context: `、その家へ電話をかけて女主人の都合を問い合わすと、いつも留守という返事であった。`
- expected:
  - `彼女が勤めていた時分にも電話をかけると、定って、女衆の声で冷淡に、`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_053_wiki
- reading: `すいてんろうし（すいてんろうし）`
- context: `死に際に拳法書を村の男に託す。`
- expected:
  - `水天老師（すいてんろうし）`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_054_wiki
- reading: `このようにひわいなのうりょくをもっているが、はるひさじしんのせいてきなものへのかんしんはうすい。`
- context: `また、絶頂除霊以外の術技は最低レベルで、まともに使えない。`
- expected:
  - `このように卑猥な能力を持っているが、晴久自身の性的なものへの関心は薄い。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_055_wiki
- reading: `ちゅうごく・たいわんにんににっぽんのやっきょくをしょうかいしたらだいにんきに！`
- context: `台湾各地でセミナー、日本国内のメーカーに対する訪日コンサルティングを行っている。`
- expected:
  - `中国・台湾人に日本の薬局を紹介したら大人気に！`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_056_wiki
- reading: `よるになりつきがしずんであんやとなるとつきをまねきよせ、つきよとしたとのでんせつがある。`
- context: `地で泊まった空海（弘法大師）が、水がない衆生の不便を感じて加持し清水を湧かせた。`
- expected:
  - `夜になり月が沈んで闇夜となると月を招き寄せ、月夜としたとの伝説がある。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_057_wiki
- reading: `えだまめのむねあてとよばれるきみどりいろのみずたまぶらじゃーをちゃくよう。`
- context: `なお、後頭部には様々なシールが貼られていることが多い。`
- expected:
  - `枝豆の胸当てと呼ばれる黄緑色の水玉ブラジャーを着用。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_058_wiki
- reading: `そのごしゅをのみほうろうしているところを、おーしゃんさいどじゅうみんによってひそかにさつがいされる。`
- context: `アーロンの負傷を招いて追放される。`
- expected:
  - `その後酒を飲み放浪している所を、オーシャンサイド住民によって密かに殺害される。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_059_wiki
- reading: `「こういうときには、のべちゃんもきをきかして、さけてくれればかいに」`
- context: `とお倉も姉娘の後に附いて言った。`
- expected:
  - `「こういう時には、延ちゃんも気を利かして、避けてくれれば可いに」`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_060_wiki
- reading: `ていまいたちのはなしでは、すうがくがとくにとくいだったとのこと。`
- context: `勉強も優秀であり、学生時代は首席だった。`
- expected:
  - `弟妹たちの話では、数学が特に得意だったとのこと。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_061_wiki
- reading: `いきものをなんでもにんぎょうにできるにんぎょうしょくにん。`
- context: `フリジッタ`
- expected:
  - `生き物を何でも人形にできる人形職人。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_062_wiki
- reading: `ぷらんてんをだえんけいにかっとし、かねちゃいろになるまであげたものである。`
- context: `プランテンは西アフリカでも一般的な食材である。`
- expected:
  - `プランテンを楕円形にカットし、金茶色になるまで揚げたものである。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_063_wiki
- reading: `しかし、くのうときのうしょうがいはつねにかんれんしているわけではない。`
- context: `るが、個人の生活の中で機能障害を正確に認識するために有用と評価することもできる。`
- expected:
  - `しかし、苦悩と機能障害は常に関連しているわけではない。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_064_wiki
- reading: `とやすたろうにとわれて、おすずはまたあかくなって、くびをふった。`
- context: `って出かける時は、チョウチンもつけていたろうから、年かっこうぐらい見えたろうね」`
- expected:
  - `と保太郎に問われて、お鈴はまた赤くなって、首をふった。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_065_wiki
- reading: `なるほど、それはたくみいおもいつきだ。`
- context: `ではお前はここまでお話しを買いに来たのか。`
- expected:
  - `成る程、それは巧い思い付きだ。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_066_wiki
- reading: `にじめぐりにじさんじのまちめぐり`
- context: `コラボミニゲームが配信。`
- expected:
  - `にじめぐり にじさんじの街めぐり`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_067_wiki
- reading: `しょうぎぶいんながらこうじゅつのかこからいごもつよく、なみのちゅうがくせいあいてならあっとうするじつりょくをもつ。`
- context: `葉瀬中内では「泣く子も黙る加賀」と呼ばれ恐れられる。`
- expected:
  - `将棋部員ながら後述の過去から囲碁も強く、並の中学生相手なら圧倒する実力を持つ。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_068_wiki
- reading: `こういうはなしぶりのまに、もすくではしんしがめぐりあうことのなかったゆるやかさですもーりぬぃでのときがけいかした。`
- context: `誰からそのことをききましたか？」`
- expected:
  - `こういう話しぶりの間に、モスクでは伸子がめぐり合うことのなかったゆるやかさでスモーリヌィでの時が経過した。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_069_wiki
- reading: `にゅーすえぶりぃない「えぶりぃとくしゅう・しょくひせつやくせいかつ」（にっぽんてれび）`
- context: `ニュースエブリィ内「エブリィ特集・お取り寄せ物産展」`
- expected:
  - `ニュースエブリィ内「エブリィ特集・食費節約生活」（日本テレビ）`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_070_wiki
- reading: `「それでそのしょてんのほうはぶじなのかね」`
- context: `咄嗟の間に死んだ女の所天の事が聞いて見たくなる。`
- expected:
  - `「それでその所天の方は無事なのかね」`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_071_wiki
- reading: `にとろぜっかじょうをかんじゃがのみこんだ。`
- context: `兵庫区薬剤師会ホームページ・トピックス`
- expected:
  - `ニトロ舌下錠を患者が飲み込んだ。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_072_wiki
- reading: `いくらかのぶっしをたいひさせたあとにひをつけ、あめりかぐんじょうりくぶたいがふねをかくほできるまえにばくはつした。`
- context: `で劣勢になった戦隊が港に留まり、さらに強力なものを建造するというパターンだった。`
- expected:
  - `幾らかの物資を退避させた後に火を付け、アメリカ軍上陸部隊が船を確保できる前に爆発した。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_073_wiki
- reading: `あいすほっけーかいのえいきゅうけつばん`
- context: `ズ、ダラス・カウボーイズ、ラスベガス・レイダースは永久欠番制度を導入していない。`
- expected:
  - `アイスホッケー界の永久欠番`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_074_wiki
- reading: `げんどうきつけじてんしゃ（こうばんようすくーたーなど）`
- context: `警ら用自転車`
- expected:
  - `原動機付自転車（交番用スクーターなど）`
- top1: <!-- frontier LLM の出力をここに記入 -->

### general_075_wiki
- reading: `また、いほうばっさいやしんりんかさい、びょうがいちゅうやがいらいしゅなどにより、しんりんのしつやきのうがていかしている。`
- context: `の主な原因は、農業や牧畜などのための開発や転用であり、特に熱帯地域で顕著である。`
- expected:
  - `また、違法伐採や森林火災、病害虫や外来種などにより、森林の質や機能が低下している。`
- top1: <!-- frontier LLM の出力をここに記入 -->

## homophone (n=37)

### homophone_001_hand
- reading: `いしからくすりをしょほうされた`
- context: `風邪を引いたので、`
- expected:
  - `医師から薬を処方された`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_002_hand
- reading: `かれはつよいいしをもってけつだんした`
- context: `プロジェクトの成功に向けて`
- expected:
  - `彼は強い意志を持って決断した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_003_hand
- reading: `そうほうのいしのそつうがじゅうようだ`
- context: `契約交渉では`
- expected:
  - `双方の意思の疎通が重要だ`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_004_hand
- reading: `にわのちゅうしんにおおきないしがある`
- context: `我が家の`
- expected:
  - `庭の中心に大きな石がある`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_005_hand
- reading: `きしゃがきしゃできしゃした`
- context: `新聞社の`
- expected:
  - `記者が汽車で帰社した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_006_hand
- reading: `かがくのじっけんをおこなった`
- context: `学校では`
- expected:
  - `科学の実験を行った`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_007_hand
- reading: `かがくのぶんせきをおこなった`
- context: `研究室では`
- expected:
  - `化学の分析を行った`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_008_hand
- reading: `こうえんでこうえんをきいた`
- context: `休日の午後は`
- expected:
  - `公園で講演を聞いた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_009_hand
- reading: `かのじょはこうえんかいでこうえんした`
- context: `演台に立った`
- expected:
  - `彼女は講演会で講演した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_010_hand
- reading: `かれのきぎょうけいかくがせいこうした`
- context: `新聞によると`
- expected:
  - `彼の起業計画が成功した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_011_hand
- reading: `ゆうめいなきぎょうにしゅうしょくがきまった`
- context: `大学卒業後、`
- expected:
  - `有名な企業に就職が決まった`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_012_hand
- reading: `かんしんなことにかんしんをもった`
- context: `社会問題に対して`
- expected:
  - `感心なことに関心を持った`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_013_hand
- reading: `しゅうりょうはしゅうりょうまでにおえた`
- context: `新商品の`
- expected:
  - `修了は終了までに終えた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_014_hand
- reading: `じこかいけつができないじこもある`
- context: `台風のため`
- expected:
  - `自己解決ができない事故もある`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_015_hand
- reading: `いじょうがないかいじょうもみせてくださいといった`
- context: `医師は念のため`
- expected:
  - `異常がない異状も見せてくださいと言った`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_016_hand
- reading: `きしょうよほうをきいてきしょうした`
- context: `天気予報では`
- expected:
  - `気象予報を聞いて起床した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_017_hand
- reading: `かれはきしょうもきしょうもわるい`
- context: `最近の`
- expected:
  - `彼は気性も気象も悪い`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_018_hand
- reading: `しんじつとじじつがちがうばあいがある`
- context: `調査によると`
- expected:
  - `真実と事実が違う場合がある`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_019_hand
- reading: `きかいがきかいをうしなった`
- context: `約束の時間に`
- expected:
  - `機会が機械を失った`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_020_hand
- reading: `おなじかんじのかんじをうけた`
- context: `試験を受けて`
- expected:
  - `同じ漢字の感じを受けた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_021_hand
- reading: `かんじょうてきになってかんじょうをもちくずした`
- context: `会計の前に`
- expected:
  - `感情的になって勘定を持ち崩した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_022_hand
- reading: `じょうけんとじょうこうのいみがちがう`
- context: `法律の条文で`
- expected:
  - `条件と条項の意味が違う`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_023_hand
- reading: `はしをつかってはしをわたす`
- context: `料理の時は`
- expected:
  - `箸を使って端を渡す`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_024_hand
- reading: `はしをかけるこうじがはじまった`
- context: `川の近くに`
- expected:
  - `橋を架ける工事が始まった`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_025_hand
- reading: `こうかいするこうかいをこうかいした`
- context: `選手は`
- expected:
  - `後悔する航海を公開した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_026_hand
- reading: `かみにおがみをささげる`
- context: `神社の`
- expected:
  - `神に拝みを捧げる`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_027_hand
- reading: `かみをきってもらう`
- context: `髪型を変えたいので`
- expected:
  - `髪を切ってもらう`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_028_hand
- reading: `きかいのきかいをうかがった`
- context: `会議の後で`
- expected:
  - `機械の機会をうかがった`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_029_hand
- reading: `じぶんのいじをつらぬくいしがためされる`
- context: `面接では`
- expected:
  - `自分の意地を貫く意志が試される`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_030_hand
- reading: `ははにあまえてはははなしをきかない`
- context: `子供が`
- expected:
  - `母に甘えては話を聞かない`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_031_hand
- reading: `せんせいのいこうをうかがっていこうをかためた`
- context: `試験前に`
- expected:
  - `先生の意向を伺って意向を固めた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_032_hand
- reading: `しんごうをみてしんこうをやめた`
- context: `交差点で`
- expected:
  - `信号を見て進行をやめた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_033_hand
- reading: `いっせいにせいそうがはじまった`
- context: `大会では`
- expected:
  - `一斉に清掃が始まった`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_034_hand
- reading: `こうていをかくにんしておく`
- context: `山登りの前に`
- expected:
  - `行程を確認しておく`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_035_hand
- reading: `こうていのこうていをほうもんした`
- context: `王室の`
- expected:
  - `皇帝の高弟を訪問した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_036_hand
- reading: `こうえいがおこなうこうえんのほしゅう`
- context: `地域の`
- expected:
  - `公営が行う公園の補修`
- top1: <!-- frontier LLM の出力をここに記入 -->

### homophone_037_hand
- reading: `むらのあるじがきじをきじょうにおいた`
- context: `昔話では`
- expected:
  - `村の主が記事を机上に置いた`
- top1: <!-- frontier LLM の出力をここに記入 -->

## names (n=55)

### names_001_hand
- reading: `とうきょうだいがくにごうかくしたしんゆう`
- context: `(無し)`
- expected:
  - `東京大学に合格した親友`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_002_hand
- reading: `きょうとのきんかくじをけんがくした`
- context: `(無し)`
- expected:
  - `京都の金閣寺を見学した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_003_hand
- reading: `ほっかいどうのさっぽろしではゆきまつりがある`
- context: `(無し)`
- expected:
  - `北海道の札幌市では雪祭りがある`
  - `北海道の札幌市では雪まつりがある`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_004_hand
- reading: `なごやしのめいてつひゃっかてんでかいものした`
- context: `(無し)`
- expected:
  - `名古屋市の名鉄百貨店で買い物した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_005_hand
- reading: `おおさかじょうからしがいをみおろした`
- context: `(無し)`
- expected:
  - `大阪城から市街を見下ろした`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_006_hand
- reading: `ふじさんはせかいいさんにとうろくされている`
- context: `(無し)`
- expected:
  - `富士山は世界遺産に登録されている`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_007_hand
- reading: `おきなわけんのしゅりじょうをほうもんした`
- context: `(無し)`
- expected:
  - `沖縄県の首里城を訪問した`
  - `沖縄県の首里城を訪れた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_008_hand
- reading: `かながわけんのよこはまえきではじめてあった`
- context: `(無し)`
- expected:
  - `神奈川県の横浜駅で初めて会った`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_009_hand
- reading: `くまもとけんのあそさんでとざんをたのしんだ`
- context: `(無し)`
- expected:
  - `熊本県の阿蘇山で登山を楽しんだ`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_010_hand
- reading: `とうきょうとしぶやくはちこうまえであいましょう`
- context: `(無し)`
- expected:
  - `東京都渋谷区ハチ公前で会いましょう`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_011_hand
- reading: `ひろしまへいわきねんこうえんをおとずれた`
- context: `(無し)`
- expected:
  - `広島平和記念公園を訪れた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_012_hand
- reading: `ならけんのとうだいじでだいぶつをおがんだ`
- context: `(無し)`
- expected:
  - `奈良県の東大寺で大仏を拝んだ`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_013_hand
- reading: `いばらきけんのつくばだいがくにしんがくした`
- context: `(無し)`
- expected:
  - `茨城県の筑波大学に進学した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_014_hand
- reading: `かごしまけんのさくらじまがふんかした`
- context: `(無し)`
- expected:
  - `鹿児島県の桜島が噴火した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_015_hand
- reading: `せんだいしのたなばたまつりがゆうめいだ`
- context: `(無し)`
- expected:
  - `仙台市の七夕祭りが有名だ`
  - `仙台市の七夕まつりが有名だ`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_016_hand
- reading: `とっとりさきゅうでらくだにのった`
- context: `(無し)`
- expected:
  - `鳥取砂丘でラクダに乗った`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_017_hand
- reading: `いしかわけんのけんろくえんへいきたい`
- context: `来年は`
- expected:
  - `石川県の兼六園へ行きたい`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_018_hand
- reading: `あおもりけんのねぶたまつりがみたい`
- context: `(無し)`
- expected:
  - `青森県のねぶた祭が見たい`
  - `青森県のねぶた祭りが見たい`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_019_hand
- reading: `しんおおさかからしんよこはままでいどうした`
- context: `新幹線で`
- expected:
  - `新大阪から新横浜まで移動した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_020_hand
- reading: `ながのけんのまつもとじょうをけんがくした`
- context: `(無し)`
- expected:
  - `長野県の松本城を見学した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_021_hand
- reading: `おりんぴっくがとうきょうでかいさいされた`
- context: `(無し)`
- expected:
  - `オリンピックが東京で開催された`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_022_hand
- reading: `さんふらんしすこのぐーぐるほんしゃをほうもんした`
- context: `(無し)`
- expected:
  - `サンフランシスコのGoogle本社を訪問した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_023_hand
- reading: `にゅーよーくのせんとらるぱーくをさんぽした`
- context: `(無し)`
- expected:
  - `ニューヨークのセントラルパークを散歩した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_024_hand
- reading: `ぱりのえっふぇるとうはよぞらにひかっていた`
- context: `(無し)`
- expected:
  - `パリのエッフェル塔は夜空に光っていた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_025_hand
- reading: `ろんどんのばっきんがむきゅうでんをけんがくした`
- context: `(無し)`
- expected:
  - `ロンドンのバッキンガム宮殿を見学した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_026_hand
- reading: `あめりかのしりこんばれーではたらく`
- context: `(無し)`
- expected:
  - `アメリカのシリコンバレーで働く`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_027_hand
- reading: `とよたじどうしゃがでんきじどうしゃをはっぴょうした`
- context: `(無し)`
- expected:
  - `トヨタ自動車が電気自動車を発表した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_028_hand
- reading: `そにーぐるーぷのしんしゃちょうがしゅうにんした`
- context: `(無し)`
- expected:
  - `ソニーグループの新社長が就任した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_029_hand
- reading: `みついふどうさんのきじがけいさいされた`
- context: `業界紙に`
- expected:
  - `三井不動産の記事が掲載された`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_030_hand
- reading: `こくれんそうかいでこうえんをおこなった`
- context: `日本の総理は`
- expected:
  - `国連総会で講演を行った`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_031_wiki
- reading: `あなんしつばきちょうのむじんとうでたちばなわんないにいちする。`
- context: `(無し)`
- expected:
  - `阿南市椿町の無人島で橘湾内に位置する。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_032_wiki
- reading: `だいとしいがいはおおがきちくしょうぼうくみあいがある。`
- context: `導入に合わせ、東京消防庁、京都市消防局、名古屋市消防局など大都市圏に配備された。`
- expected:
  - `大都市以外は大垣地区消防組合がある。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_033_wiki
- reading: `しほろちょう（しほろちょう）は、ほっかいどうかとうぐんにあるまち。`
- context: `(無し)`
- expected:
  - `士幌町（しほろちょう）は、北海道河東郡にある町。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_034_wiki
- reading: `ふくとめむら（ふくとめむら）は、いしかわけんいしかわぐんにあったむら。`
- context: `(無し)`
- expected:
  - `福留村（ふくとめむら）は、石川県石川郡にあった村。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_035_wiki
- reading: `しかし、このくににはけんりょく、せいふ、せんそう、ほうりつ、けいばつなどということばがまるでないのです。`
- context: `、商業のこと、工業のこと、学術のことなど、知っていることを全部話してやりました。`
- expected:
  - `しかし、この国には権力、政府、戦争、法律、刑罰などという言葉がまるでないのです。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_036_wiki
- reading: `ほっかいどうなかしべつのうぎょうこうとうがっこう`
- context: `計根別郵便局`
- expected:
  - `北海道中標津農業高等学校`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_037_wiki
- reading: `いてんしたせいいずみてらいがいはげんそんする。`
- context: `幕末には、坂上に次のような寺院が建ち並んでいた。`
- expected:
  - `移転した正泉寺以外は現存する。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_038_wiki
- reading: `えのもとせいき（もとやまがたけんつるおかしちょう）`
- context: `足達兼一郎（元山形県鶴岡市長）`
- expected:
  - `榎本政規（元山形県鶴岡市長）`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_039_wiki
- reading: `さっぽろしりつふくいのなかがっこう`
- context: `札幌市立発寒中学校`
- expected:
  - `札幌市立福井野中学校`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_040_wiki
- reading: `こじまかずし（けんぽう、もととうほくだいがくきょうじゅ）`
- context: `黒沼悦郎（商法、早稲田大学大学院法務研究科教授）`
- expected:
  - `小嶋和司（憲法、元東北大学教授）`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_041_wiki
- reading: `げんしょくはちゅうおうだいがくぶんがくぶじんぶんしゃかいがっかきょうじゅ。`
- context: `サムネイル`
- expected:
  - `現職は中央大学文学部人文社会学科教授。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_042_wiki
- reading: `のーざん・ぱしふぃっくてつどうのとうぶではしゅにんぎしになった。`
- context: `戦後の行動`
- expected:
  - `ノーザン・パシフィック鉄道の東部では主任技師になった。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_043_wiki
- reading: `えすかーどうしくえきまえはいつだんち`
- context: `竹園第二団地`
- expected:
  - `エスカード牛久駅前ハイツ団地`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_044_wiki
- reading: `ついめぐる（おいまわし）は、あきたけんよこてしのまちてい。`
- context: `(無し)`
- expected:
  - `追廻（おいまわし）は、秋田県横手市の町丁。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_045_wiki
- reading: `きこくご、なむさんだいがくこうし。`
- context: `同年、南山大学助手。`
- expected:
  - `帰国後、南山大学講師。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_046_wiki
- reading: `また、ないぶのどうくつはみずがたまるためすべりやすく、じょせいではしょうしょうくろうするほどのけわしいみちとなっている。`
- context: `定期的に浮き沈みを繰り返す特徴を持つため発見は困難。`
- expected:
  - `また、内部の洞窟は水がたまるため滑りやすく、女性では少々苦労する程の険しい道となっている。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_047_wiki
- reading: `わきみさきびーちろっくわきみさきのびーちろっくながさきけんのぶんかざい`
- context: `樺島灯台公園`
- expected:
  - `脇岬ビーチロック脇岬のビーチロック 長崎県の文化財`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_048_wiki
- reading: `いつおのようねんき、てじまかはながおかぐんこくふむらにうつり、ちょうなん・だいちょうがいぎょうでかけいをささえた。`
- context: `武士・手嶋俊蔵（増魚）の三男であった。`
- expected:
  - `厳夫の幼年期、手嶋家は長岡郡国府村に移り、長男・大長が医業で家計を支えた。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_049_wiki
- reading: `さかがみたむらまろがえぞせいばつのさい、てきをおびきよせるため、ねぶたをうみにながしたことにゆらいするというせつ。`
- context: `坂上田村麻呂説`
- expected:
  - `坂上田村麻呂が蝦夷征伐の際、敵をおびき寄せるため、ねぶたを海に流したことに由来するという説。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_050_wiki
- reading: `この「げんざいのまいそうちのこうほう」にはまちのきょうどうぼちはふくまれていなかった。`
- context: `ェイとチェンバーズ・ストリートの角からフォーリー・スクエアまで北東に伸びていた。`
- expected:
  - `この「現在の埋葬地の後方」には町の共同墓地は含まれていなかった。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_051_wiki
- reading: `けたいのゆえに、みやこのじょうよりはなれてかれらすいぐんの`
- context: `さるを今はた＊總帥の過失、並びに士卒らの`
- expected:
  - `懈怠の故に、都城より離れて彼ら水軍の`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_052_wiki
- reading: `ふなばししりつまえばるちゅうがっこう「まえばら」（「まいはら」からかいしょう）`
- context: `船橋市立前原小学校「まえはら」`
- expected:
  - `船橋市立前原中学校「まえばら」（「まいはら」から改称）`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_053_wiki
- reading: `おおやむら（おおやむら）は、さいたまけんいるまぐんにそんざいしたむら。`
- context: `(無し)`
- expected:
  - `大家村（おおやむら）は、埼玉県入間郡に存在した村。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_054_wiki
- reading: `せいふやたのせいりょくがひそかにそれをしょうれいしているようなみつゆだって、あるかもしれないのよ」`
- context: `ら、あなたのカンは正しいかも知れないけど、密輸品にもピンからキリまであるのです。`
- expected:
  - `政府や他の勢力がひそかにそれを奨励しているような密輸だって、あるかも知れないのよ」`
- top1: <!-- frontier LLM の出力をここに記入 -->

### names_055_wiki
- reading: `ぼしょはとうきょうとみなとくのたつはるてら。`
- context: `東京府生まれ。`
- expected:
  - `墓所は東京都港区の龍原寺。`
- top1: <!-- frontier LLM の出力をここに記入 -->

## numeric (n=65)

### numeric_001_hand
- reading: `さんじゅうにんがしゅっせきするよていだ`
- context: `会議の案内を見ると、`
- expected:
  - `30人が出席する予定だ`
  - `三十人が出席する予定だ`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_002_hand
- reading: `しがつついたちにかいてんする`
- context: `新しい店舗は、`
- expected:
  - `4月1日に開店する`
  - `四月一日に開店する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_003_hand
- reading: `じゅうごにんのえんじにあがさんかしている`
- context: `プロジェクトには`
- expected:
  - `15人のエンジニアが参加している`
  - `十五人のエンジニアが参加している`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_004_hand
- reading: `にじゅうよじかんたいせいでたいおうする`
- context: `(無し)`
- expected:
  - `24時間体制で対応する`
  - `二十四時間体制で対応する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_005_hand
- reading: `ひゃくにんをこえていた`
- context: `駅前の行列は`
- expected:
  - `100人を超えていた`
  - `百人を超えていた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_006_hand
- reading: `せんきゅうひゃくはちじゅうごねんにうまれた`
- context: `(無し)`
- expected:
  - `1985年に生まれた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_007_hand
- reading: `さんぜんこのしょうひんがほかんされている`
- context: `倉庫には`
- expected:
  - `3000個の商品が保管されている`
  - `三千個の商品が保管されている`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_008_hand
- reading: `いちまんえんさつをごまいりょうがえした`
- context: `(無し)`
- expected:
  - `1万円札を5枚両替した`
  - `一万円札を五枚両替した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_009_hand
- reading: `いちまんにんのらんなーがさんかした`
- context: `今回のマラソンには`
- expected:
  - `1万人のランナーが参加した`
  - `一万人のランナーが参加した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_010_hand
- reading: `よんひゃくごじゅうみりりっとるのみずをのんだ`
- context: `(無し)`
- expected:
  - `450ミリリットルの水を飲んだ`
  - `450mlの水を飲んだ`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_011_hand
- reading: `ごまんにんにたっした`
- context: `試合の観客は`
- expected:
  - `5万人に達した`
  - `五万人に達した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_012_hand
- reading: `きおんがさんじゅうごどをこえた`
- context: `(無し)`
- expected:
  - `気温が35度を超えた`
  - `気温が35℃を超えた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_013_hand
- reading: `ろくせんまんえんだった`
- context: `新築マンションの価格は`
- expected:
  - `6000万円だった`
  - `六千万円だった`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_014_hand
- reading: `さんびゃくごじゅうめーとるあるらしい`
- context: `この湖の深さは`
- expected:
  - `350メートルあるらしい`
  - `350mあるらしい`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_015_hand
- reading: `ひゃくにじゅうばんのばすにのった`
- context: `(無し)`
- expected:
  - `120番のバスに乗った`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_016_hand
- reading: `ごーるまえひゃくめーとるでぬかれた`
- context: `(無し)`
- expected:
  - `ゴール前100メートルで抜かれた`
  - `ゴール前百メートルで抜かれた`
  - `ゴール前100mで抜かれた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_017_hand
- reading: `こうそくどうろをひゃくじゅっきろでそうこうする`
- context: `(無し)`
- expected:
  - `高速道路を110キロで走行する`
  - `高速道路を110kmで走行する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_018_hand
- reading: `ついかとうしでにせんおくえんがひつようだ`
- context: `(無し)`
- expected:
  - `追加投資で2000億円が必要だ`
  - `追加投資で二千億円が必要だ`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_019_hand
- reading: `さんじゅっきろぐらむをこえた`
- context: `荷物の総重量は`
- expected:
  - `30キログラムを超えた`
  - `30kgを超えた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_020_hand
- reading: `さんじかんはんのえいがをみた`
- context: `(無し)`
- expected:
  - `3時間半の映画を見た`
  - `三時間半の映画を見た`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_021_hand
- reading: `よねんかんのだいがくせいかつをおえた`
- context: `(無し)`
- expected:
  - `4年間の大学生活を終えた`
  - `四年間の大学生活を終えた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_022_hand
- reading: `にちようひんのうりあげがはちじゅっぱーせんとをしめる`
- context: `(無し)`
- expected:
  - `日用品の売上が80パーセントを占める`
  - `日用品の売り上げが80%を占める`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_023_hand
- reading: `ことしのうりあげはきょねんひひゃくにじゅっぱーせんとだ`
- context: `(無し)`
- expected:
  - `今年の売上は去年比120パーセントだ`
  - `今年の売上は去年比120%だ`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_024_hand
- reading: `あんけーとのかいとうりつはろくじゅっぱーせんとだった`
- context: `発表によると`
- expected:
  - `アンケートの回答率は60パーセントだった`
  - `アンケートの回答率は60%だった`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_025_hand
- reading: `しゃちょうのねんれいはろくじゅうきゅうさいだ`
- context: `(無し)`
- expected:
  - `社長の年齢は69歳だ`
  - `社長の年齢は六十九歳だ`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_026_hand
- reading: `あかちゃんのたいじゅうはさんきろだった`
- context: `(無し)`
- expected:
  - `赤ちゃんの体重は3キロだった`
  - `赤ちゃんの体重は三キロだった`
  - `赤ちゃんの体重は3kgだった`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_027_hand
- reading: `はいたつはなのかごにとうちゃくします`
- context: `(無し)`
- expected:
  - `配達は七日後に到着します`
  - `配達は7日後に到着します`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_028_hand
- reading: `じゅうがつにはんきけっさんをおこなう`
- context: `(無し)`
- expected:
  - `10月に半期決算を行う`
  - `十月に半期決算を行う`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_029_hand
- reading: `にじゅっせいきのおわりまでにけいざいせいちょうがつづいた`
- context: `(無し)`
- expected:
  - `20世紀の終わりまでに経済成長が続いた`
  - `二十世紀の終わりまでに経済成長が続いた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_030_hand
- reading: `えーふぉーのようしをじゅうまいいんさつした`
- context: `(無し)`
- expected:
  - `A4の用紙を10枚印刷した`
  - `A4の用紙を十枚印刷した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_031_hand
- reading: `ぴーえいちななのすいようえきをつくる`
- context: `(無し)`
- expected:
  - `pH7の水溶液を作る`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_032_hand
- reading: `しゅうりひはさんまんにせんえんだった`
- context: `(無し)`
- expected:
  - `修理費は3万2千円だった`
  - `修理費は三万二千円だった`
  - `修理費は32,000円だった`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_033_hand
- reading: `ふたつめのぎだいにすすみましょう`
- context: `(無し)`
- expected:
  - `2つ目の議題に進みましょう`
  - `二つ目の議題に進みましょう`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_034_hand
- reading: `ろくがつよっかにしけんがある`
- context: `(無し)`
- expected:
  - `6月4日に試験がある`
  - `六月四日に試験がある`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_035_hand
- reading: `はちじはっぷんちゃくでよていどおりだった`
- context: `電車は`
- expected:
  - `8時8分着で予定通りだった`
  - `08:08着で予定通りだった`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_036_hand
- reading: `ほんをひゃくさつよみたい`
- context: `(無し)`
- expected:
  - `本を100冊読みたい`
  - `本を百冊読みたい`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_037_hand
- reading: `むいかめのれんぞくきんむがきつい`
- context: `(無し)`
- expected:
  - `6日目の連続勤務がきつい`
  - `六日目の連続勤務がきつい`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_038_hand
- reading: `だうんろーどじかんはにびょうだった`
- context: `(無し)`
- expected:
  - `ダウンロード時間は2秒だった`
  - `ダウンロード時間は二秒だった`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_039_hand
- reading: `えーあいのおうとうじかんはじゅうびょういないだ`
- context: `(無し)`
- expected:
  - `AIの応答時間は10秒以内だ`
  - `AIの応答時間は十秒以内だ`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_040_hand
- reading: `にじゅっぷんしかない`
- context: `試験の制限時間は`
- expected:
  - `20分しかない`
  - `二十分しかない`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_041_wiki
- reading: `それこそぼくいちにんのめいわくではありませんからね。」`
- context: `ければ、僕も喜んで話しますが——万一秘密の洩れた事が、山県公にでも知れて見給え。`
- expected:
  - `それこそ僕一人の迷惑ではありませんからね。」`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_042_wiki
- reading: `くろまじゅつぶのこういってん。`
- context: `黒魔術部。`
- expected:
  - `黒魔術部の紅一点。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_043_wiki
- reading: `そしてしきりにそういちろうのしたいをのぞきこんでいた。`
- context: `そのとき検事と署長とは、踏台の上に抱き合うようにして乗っていた。`
- expected:
  - `そしてしきりに総一郎の屍体を覗きこんでいた。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_044_wiki
- reading: `こころざしちくはちゅうごくのおんみょうごぎょうせつのえいきょうものこしていた。`
- context: `は游気を水蒸気の意味でも用いており、今日の空気と全く同じというわけではなかった。`
- expected:
  - `志築は中国の陰陽五行説の影響も残していた。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_045_wiki
- reading: `おおしおはちまんぐう（ふくいけんえちぜんし）`
- context: `若宮八幡宮（福井県白山市）`
- expected:
  - `大塩八幡宮（福井県越前市）`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_046_wiki
- reading: `「たはちはすぐにまじめさうなかおをするんでいやになるな。`
- context: `さうに腕組をすると、さつき何か云ひ出さうとした見るからに元気者らしい剽軽な男は、`
- expected:
  - `「田八は直ぐに真面目さうな顔をするんで厭になるな。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_047_wiki
- reading: `そのご、いちじてきにかーとれーすへふっきし「すぺいん・かーとせんしゅけん」へしゅつじょうする。`
- context: `さらに「フォーミュラ・ルノー・ユーロカップ」への参戦も決まる。`
- expected:
  - `その後、一時的にカートレースへ復帰し「スペイン・カート選手権」へ出場する。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_048_wiki
- reading: `「としまそんし」および「こくせいちょうさ」によればくちのしまのじんこうのせんいはかきのとおりである。`
- context: `南限の植物`
- expected:
  - `「十島村誌」及び「国勢調査」によれば口之島の人口の遷移は下記のとおりである。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_049_wiki
- reading: `これをかいこんしようというくわだてがさいさんしじんによっておこなわれたが、いずれもしっぱいした。`
- context: `例証されているが、この地は、マン僧正によれば１）、本来最も乾燥した砂地であった。`
- expected:
  - `これを開墾しようという企てが再三私人によって行われたが、いずれも失敗した。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_050_wiki
- reading: `ごがつ（しりーずてーま「いさんそうぞく」）`
- context: `弥生は看護師を辞めてハナを野田家に引き取り自分で介護をすることに決める。`
- expected:
  - `五月（シリーズテーマ「遺産相続」）`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_051_wiki
- reading: `「あしがまんぞくであったころには、ごにんであろうと、じゅうにんであろうと、なでぎりにしたものだったが」`
- context: `と撞木杖の武士は事も無げに、`
- expected:
  - `「足が満足であった頃には、五人であろうと、十人であろうと、撫斬りにしたものだったが」`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_052_wiki
- reading: `とみれば、みちばたのしばのじょうにおかれたけんじゅつのどうぐいちくみ。`
- context: `お松は秋の情景をほしいままにして、山と畑との勾配ゆるやかな道を歩みました。`
- expected:
  - `と見れば、道ばたの芝の上に置かれた剣術の道具一組。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_053_wiki
- reading: `ひばくち・ひろしまの「きおくのけいしょう」をてーまに、「かこ」「げんざい」「みらい」のさんぶこうせいでてんかいしていく。`
- context: `共演に松永有紘、有坂心花、入山杏奈、平岡祐太ら。`
- expected:
  - `被爆地・広島の「記憶の継承」をテーマに、「過去」「現在」「未来」の三部構成で展開していく。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_054_wiki
- reading: `じぶんのへやといううちに、このこはへやをふたつもっている。`
- context: `こう言って、茂太郎は、おとなしく、自分の部屋に戻りました。`
- expected:
  - `自分の部屋といううちに、この子は部屋を二つ持っている。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_055_wiki
- reading: `これは、なにせんねんかむかしのできごとであるとどうじに、また、このしゅんかんのげんじつごとでもあった。`
- context: `そして、それはいま、タミル族の碩学ヤトラカン・サミ博士に伝わっているのだ。`
- expected:
  - `これは、何千年か昔のできごとであると同時に、また、この瞬間の現実事でもあった。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_056_wiki
- reading: `……しかしはっぽうとりまかれてしまった」`
- context: `……どうともして早くここを遁れ。`
- expected:
  - `……しかし八方取りまかれてしまった」`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_057_wiki
- reading: `こうそうびるぐんにあるまんしょんのいっしつでちかしゃげきじょうなどはそんざいしない。`
- context: `中、パイソンとレッドホークが一緒に登場するときがあるがおそらくこれはミスである。`
- expected:
  - `高層ビル群にあるマンションの一室で地下射撃場などは存在しない。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_058_wiki
- reading: `えんしゅつはみいけたかし、きゃくほんはりりー・ふらんきー、しゅえんはいちかわえびぞう。`
- context: `主演は松平健。`
- expected:
  - `演出は三池崇史、脚本はリリー・フランキー、主演は市川海老蔵。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_059_wiki
- reading: `だいとくさんねん、もっごじょうかわかんぐんしつなみいりしょうかい」。`
- context: `」の総称で、漢人・女直人・契丹人といった旧金朝領の住民を主体とする軍団であった。`
- expected:
  - `大徳三年、以五条河漢軍悉並入称海」。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_060_wiki
- reading: `もでるがんはけんじゅうがたやらいふるがた、ましんがんがたなどじゅうすうちょうがおうしゅうされた。`
- context: `訓練された人間の撃ち方といえる」と評している。`
- expected:
  - `モデルガンは拳銃型やライフル型、マシンガン型など十数丁が押収された。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_061_wiki
- reading: `そのご、おすやすしんくせいふは、しんきのふどうさんはんばいをいちじてきにきんしした。`
- context: `新区の発表直後、この地域の不動産価格が急速に高騰している。`
- expected:
  - `その後、雄安新区政府は、新規の不動産販売を一時的に禁止した。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_062_wiki
- reading: `ちゅうごくかほくしょうのばんりのちょうじょうにほどちかい、たいがにめんしたむら。`
- context: `ストーリー`
- expected:
  - `中国河北省の万里の長城にほど近い、大河に面した村。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_063_wiki
- reading: `ほんみょうはさんこうぎん（みよししろがね）。`
- context: `静岡県出身。`
- expected:
  - `本名は三好銀（みよし しろがね）。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_064_wiki
- reading: `はかんとくのはらたつのりからきたいされ、かいまくいちぐんめんばーいりをした。`
- context: `千葉ロッテマリーンズなどが獲得を検討したものの指名球団はなく日本生命に進む。`
- expected:
  - `は監督の原辰徳から期待され、開幕一軍メンバー入りをした。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### numeric_065_wiki
- reading: `はままつしはまなくのせいぶ、みっかびちくのなんぶにいちする。`
- context: `住居表示未実施。`
- expected:
  - `浜松市浜名区の西部、三ヶ日地区の南部に位置する。`
- top1: <!-- frontier LLM の出力をここに記入 -->

## particle (n=32)

### particle_001_hand
- reading: `かれはうまれつきのさいのうがある`
- context: `(無し)`
- expected:
  - `彼は生まれつきの才能がある`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_002_hand
- reading: `たいふうがひがしへすすんでいる`
- context: `(無し)`
- expected:
  - `台風が東へ進んでいる`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_003_hand
- reading: `わたしはにほんごをべんきょうしています`
- context: `(無し)`
- expected:
  - `私は日本語を勉強しています`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_004_hand
- reading: `こどもがとしょかんへほんをかえしにいった`
- context: `(無し)`
- expected:
  - `子供が図書館へ本を返しに行った`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_005_hand
- reading: `かれにこのしょるいをわたしてください`
- context: `(無し)`
- expected:
  - `彼にこの書類を渡してください`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_006_hand
- reading: `きょうとにりょこうするのがたのしみだ`
- context: `(無し)`
- expected:
  - `京都に旅行するのが楽しみだ`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_007_hand
- reading: `とうきょうからおおさかへいどうする`
- context: `(無し)`
- expected:
  - `東京から大阪へ移動する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_008_hand
- reading: `あめにもまけずかぜにもまけない`
- context: `(無し)`
- expected:
  - `雨にも負けず風にも負けない`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_009_hand
- reading: `ともだちとえいがをみにいった`
- context: `(無し)`
- expected:
  - `友達と映画を見に行った`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_010_hand
- reading: `いぬをさんぽにつれていく`
- context: `(無し)`
- expected:
  - `犬を散歩に連れていく`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_011_hand
- reading: `かのじょはかれよりもせがたかい`
- context: `(無し)`
- expected:
  - `彼女は彼よりも背が高い`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_012_hand
- reading: `ねこばかりがきにいっている`
- context: `(無し)`
- expected:
  - `猫ばかりが気に入っている`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_013_hand
- reading: `あかいぺんだけでかいてください`
- context: `(無し)`
- expected:
  - `赤いペンだけで書いてください`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_014_hand
- reading: `ともだちやかぞくもしょうたいした`
- context: `(無し)`
- expected:
  - `友達や家族も招待した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_015_hand
- reading: `しごとのあとにすぽーつじむへいく`
- context: `(無し)`
- expected:
  - `仕事の後にスポーツジムへ行く`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_016_hand
- reading: `りょうしんのためにりょこうをぷれぜんとした`
- context: `(無し)`
- expected:
  - `両親のために旅行をプレゼントした`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_017_hand
- reading: `あめがふっているのでかさがひつようだ`
- context: `(無し)`
- expected:
  - `雨が降っているので傘が必要だ`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_018_hand
- reading: `こうこうせいなのにうんてんめんきょはもっていない`
- context: `(無し)`
- expected:
  - `高校生なのに運転免許は持っていない`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_019_hand
- reading: `じこではなくこしょうでちこくした`
- context: `(無し)`
- expected:
  - `事故ではなく故障で遅刻した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_020_hand
- reading: `あなたがきめるならわたしもさんせいします`
- context: `(無し)`
- expected:
  - `あなたが決めるなら私も賛成します`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_021_hand
- reading: `ほんやくされなくてもいみはわかる`
- context: `(無し)`
- expected:
  - `翻訳されなくても意味は分かる`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_022_hand
- reading: `かいぎがおわったらすぐにかえる`
- context: `(無し)`
- expected:
  - `会議が終わったらすぐに帰る`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_023_hand
- reading: `ねつがあがったのでびょういんへいった`
- context: `(無し)`
- expected:
  - `熱が上がったので病院へ行った`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_024_hand
- reading: `でんしゃがおくれたせいでちこくした`
- context: `(無し)`
- expected:
  - `電車が遅れたせいで遅刻した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_025_hand
- reading: `はるになるとさくらがさく`
- context: `(無し)`
- expected:
  - `春になると桜が咲く`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_026_hand
- reading: `あめがふればしあいはちゅうしだ`
- context: `(無し)`
- expected:
  - `雨が降れば試合は中止だ`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_027_hand
- reading: `べんきょうすればするほどおもしろくなる`
- context: `(無し)`
- expected:
  - `勉強すればするほど面白くなる`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_028_hand
- reading: `ちちはさけをのまないのでくるまをうんてんできる`
- context: `(無し)`
- expected:
  - `父は酒を飲まないので車を運転できる`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_029_hand
- reading: `じかんがなくてもあさごはんはたべる`
- context: `(無し)`
- expected:
  - `時間がなくても朝ごはんは食べる`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_030_hand
- reading: `かれはいっしょうけんめいはたらいている`
- context: `(無し)`
- expected:
  - `彼は一生懸命働いている`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_031_hand
- reading: `わからないてんがあればしつもんしてください`
- context: `(無し)`
- expected:
  - `分からない点があれば質問してください`
- top1: <!-- frontier LLM の出力をここに記入 -->

### particle_032_hand
- reading: `てつだってもらえるとたすかる`
- context: `(無し)`
- expected:
  - `手伝ってもらえると助かる`
- top1: <!-- frontier LLM の出力をここに記入 -->

## tech (n=44)

### tech_001_hand
- reading: `ぷろぐらみんぐげんごのぱいそんをまなんでいる`
- context: `(無し)`
- expected:
  - `プログラミング言語のPythonを学んでいる`
  - `プログラミング言語のパイソンを学んでいる`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_002_hand
- reading: `でーたべーすのすきーまをせっけいする`
- context: `(無し)`
- expected:
  - `データベースのスキーマを設計する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_003_hand
- reading: `りれーしょなるでーたべーすでじょいんくえりーをかく`
- context: `(無し)`
- expected:
  - `リレーショナルデータベースでJOINクエリを書く`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_004_hand
- reading: `きかいがくしゅうのもでるをふぁいんちゅーにんぐする`
- context: `(無し)`
- expected:
  - `機械学習のモデルをファインチューニングする`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_005_hand
- reading: `てんそるふろーでにゅーらるねっとわーくをこうちくした`
- context: `(無し)`
- expected:
  - `TensorFlowでニューラルネットワークを構築した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_006_hand
- reading: `こうばいこうかほうでさいてきかする`
- context: `(無し)`
- expected:
  - `勾配降下法で最適化する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_007_hand
- reading: `ばっくぷろぱげーしょんのりろんをりかいする`
- context: `(無し)`
- expected:
  - `バックプロパゲーションの理論を理解する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_008_hand
- reading: `ちゅうたたみこみそうのぱらめーたをちょうせいする`
- context: `(無し)`
- expected:
  - `畳み込み層のパラメータを調整する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_009_hand
- reading: `しぜんげんごしょりではとらんすふぉーまーがしゅりゅうだ`
- context: `最近の`
- expected:
  - `自然言語処理ではTransformerが主流だ`
  - `自然言語処理ではトランスフォーマーが主流だ`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_010_hand
- reading: `あてんしょんきこうがせいのうのかぎになる`
- context: `(無し)`
- expected:
  - `アテンション機構が性能の鍵になる`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_011_hand
- reading: `にゅーらるねっとのそうすうをふやしてじっけんした`
- context: `(無し)`
- expected:
  - `ニューラルネットの層数を増やして実験した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_012_hand
- reading: `どっかーこんてなでかいはつかんきょうをこうちくする`
- context: `(無し)`
- expected:
  - `Dockerコンテナで開発環境を構築する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_013_hand
- reading: `くばねてぃすくらすたーにでぷろいした`
- context: `(無し)`
- expected:
  - `Kubernetesクラスタにデプロイした`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_014_hand
- reading: `まいくろさーびすあーきてくちゃをどうにゅうする`
- context: `(無し)`
- expected:
  - `マイクロサービスアーキテクチャを導入する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_015_hand
- reading: `ろーどばらんさーがかどうしていない`
- context: `(無し)`
- expected:
  - `ロードバランサーが稼働していない`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_016_hand
- reading: `しーあいしーでぃーぱいぷらいんをこうちくする`
- context: `(無し)`
- expected:
  - `CI/CDパイプラインを構築する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_017_hand
- reading: `こーどれびゅーでばぐをみつけた`
- context: `(無し)`
- expected:
  - `コードレビューでバグを見つけた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_018_hand
- reading: `ゆにっとてすとのかばれっじをあげる`
- context: `(無し)`
- expected:
  - `ユニットテストのカバレッジを上げる`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_019_hand
- reading: `りふぁくたりんぐでこーどのじょうちょうせいをへらした`
- context: `(無し)`
- expected:
  - `リファクタリングでコードの冗長性を減らした`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_020_hand
- reading: `ぷるりくえすとがまーじされた`
- context: `(無し)`
- expected:
  - `プルリクエストがマージされた`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_021_hand
- reading: `おーぷんそーすこみゅにてぃにこんとりびゅーとする`
- context: `(無し)`
- expected:
  - `オープンソースコミュニティにコントリビュートする`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_022_hand
- reading: `けんげんかんりしすてむをてすとした`
- context: `(無し)`
- expected:
  - `権限管理システムをテストした`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_023_hand
- reading: `ぶんさんしすてむのかようせいをけんしょうする`
- context: `(無し)`
- expected:
  - `分散システムの可用性を検証する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_024_hand
- reading: `でっどろっくをけんしゅつするあるごりずむ`
- context: `(無し)`
- expected:
  - `デッドロックを検出するアルゴリズム`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_025_hand
- reading: `いんめもりきゃっしゅでそくどをあげるせっけいをおこなう`
- context: `(無し)`
- expected:
  - `インメモリキャッシュで速度を上げる設計を行う`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_026_hand
- reading: `くらうどほすてぃんぐにいこうしている`
- context: `会社のサーバは`
- expected:
  - `クラウドホスティングに移行している`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_027_hand
- reading: `あいぴーあどれすとぽーとばんごうをかくにんする`
- context: `(無し)`
- expected:
  - `IPアドレスとポート番号を確認する`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_028_hand
- reading: `ろぎんぐらいぶらりをつかってでばっぐする`
- context: `(無し)`
- expected:
  - `ロギングライブラリを使ってデバッグする`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_029_hand
- reading: `ぷろふぁいらーでぼとるねっくをしらべる`
- context: `(無し)`
- expected:
  - `プロファイラーでボトルネックを調べる`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_030_hand
- reading: `ぶらんちせんりゃくにぎっとふろーをさいようした`
- context: `(無し)`
- expected:
  - `ブランチ戦略にgit-flowを採用した`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_031_wiki
- reading: `しばしば、こんぴゅーたーのきどうじに、じどうてきにきどうされるようにせっていをしてりようするものがおおい。`
- context: `オープンソース 向けの利用にも対応しているものもある。`
- expected:
  - `しばしば、コンピューターの起動時に、自動的に起動されるように設定をして利用するものが多い。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_032_wiki
- reading: `どうりょくはすいへいたいこうぴすとんしきでぃーぜるえんじん。`
- context: `ゆるい傾斜の前面以外はほぼ垂直に近い箱組車体である。`
- expected:
  - `動力は水平対向ピストン式ディーゼルエンジン。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_033_wiki
- reading: `しょうにんのねっとわーくではーれむにしょうひんをとりよせるやくめをたんとうしている。`
- context: `カーリーのご利益に預かっているが、それを抜きにしても彼女へ強い好意を抱いている。`
- expected:
  - `商人のネットワークでハーレムに商品を取り寄せる役目を担当している。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_034_wiki
- reading: `しちょうそんはいかとなる（しちょうそんこーどじゅん）。`
- context: `おおむね広義の「山原（やんばる）」と呼ばれる地域に相当する。`
- expected:
  - `市町村は以下となる（市町村コード順）。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_035_wiki
- reading: `このため、すーぱーどるふぃーのぼでぃとくみあわせたさくれいがいくつかいんたーねっとじょうにこうかいされている。`
- context: `スリーサイズが異なるため、衣装の互換性はない。`
- expected:
  - `このため、スーパードルフィーのボディと組み合わせた作例がいくつかインターネット上に公開されている。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_036_wiki
- reading: `けっかとして、そふとうぇあのしんらいせいにかんするようきゅうしようがうまれた。`
- context: `その原因はユーザインタフェースの貧弱さから直接的なバグまで様々である。`
- expected:
  - `結果として、ソフトウェアの信頼性に関する要求仕様が生まれた。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_037_wiki
- reading: `こーどほかん、せいせい、ちゃっときのうなどをていきょう。`
- context: `プロジェクト全体の解析やターミナル操作が可能。`
- expected:
  - `コード補完、生成、チャット機能などを提供。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_038_wiki
- reading: `くどうしすてむはちゅうくうじくへいこうかるだんをさいようした。`
- context: `異なるが、磁気回路の工夫などにより出力特性が極力同一となるように設計されている。`
- expected:
  - `駆動システムは中空軸平行カルダンを採用した。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_039_wiki
- reading: `とくしまぶんりだいがくこうがくぶのじょうほうしすてむこうがくかをでんしじょうほうこうがくかにめいしょうへんこう。`
- context: `徳島文理大学工学部の機械電子工学科を機械創造工学科に名称変更。`
- expected:
  - `徳島文理大学工学部の情報システム工学科を電子情報工学科に名称変更。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_040_wiki
- reading: `ぶれーきはぶれんぼのぶれーきしすてむをさいようすることでせいどうせいもたかめている。`
- context: `スペンションには専用チューニングを施しており、動力性能を支えるように調整された。`
- expected:
  - `ブレーキはブレンボのブレーキシステムを採用することで制動性も高めている。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_041_wiki
- reading: `いけだけーぶるねっとわーくをのぞくぜんけーぶるてれびきょく（いけだはせとないかいほうそうをさいほうそう）`
- context: `日本海ケーブルネットワーク（鳥取市・岩美町）`
- expected:
  - `池田ケーブルネットワークを除く全ケーブルテレビ局（池田は瀬戸内海放送を再放送）`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_042_wiki
- reading: `じゅうようしょうにんをほごするしょうにんほごぷろぐらむのえーじぇんと。`
- context: `ジョン・クルーガー`
- expected:
  - `重要証人を保護する証人保護プログラムのエージェント。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_043_wiki
- reading: `なお、でじたるのあさひかわきょくだしふぉんとはさっぽろきょくだしのものよりわずかにかげがこい。`
- context: `され、それ以外では東京または札幌局送出（デジタルは札幌局送出のみ）のものを表示。`
- expected:
  - `なお、デジタルの旭川局出しフォントは札幌局出しのものよりわずかに影が濃い。`
- top1: <!-- frontier LLM の出力をここに記入 -->

### tech_044_wiki
- reading: `てきおうせいにすぐれたでーたせんたーをめざす`
- context: `インダストリー事業`
- expected:
  - `適応性に優れたデータセンターを目指す`
- top1: <!-- frontier LLM の出力をここに記入 -->

