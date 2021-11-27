DNN-HMMで音声認識を行う。
実験は以下の流れでおこなっていく。

1 状態アライメントの推定(DNNの正解ラベル作成)
2 DNNの学習
3 評価データの特徴量抽出
4 DNN-HMMでの孤立単語音声認識


【事前準備】
項目1や項目4では、学習済みのHMMが必要となる。これはGMM-HMMを使用する。
まず、"data"や"00prepare"のフォルダと同じ階層に"04dnn_hmm"というフォルダを作成する。
以降で紹介するソースコードは、"04dnn_hmm"フォルダ内で実行することを想定している。


【実行手順】
"00_state_alignment.py"→"01_count_states.py"→"my_dataset.py"→"my_model.py"→"initializetation.py"→"03_dnn_recognize.py"→"hmmfunc.py"


【コード解説】
"00_state_alignment.py"
今回は、音素ではなく状態という、さらに細かい単位でのアライメントを行う。その状態アライメント推定を行うソースコード。
このコードではGMM-HMMである"03gmm_hmm/exp/model_3state_2mix/10.hmm"を使用して、学習データおよび開発データに対して状態アライメントの推定を行う。
学習データの特徴量リスト"01compute_features/mfcc/train_small/feats.scp"と音素ラベル(数値表記)"03gmm_hmm/exp/data/train_small/text_int"を読み込み"state_alignment"関数を用いて
各特徴量の状態アライメントを推定し、結果を"exp/data/train_small/alignment"に出力している。


"01_count_states.py"
各状態の出現回数をカウントするソースコード。
学習データの正解ラベル(アライメントファイル)"exp/data/train_small/alignment"を読み込み、各状態番号の出現回数をカウントし、その結果を"exp/model_dnn/state_counts"に保存する。


"my_dataset.py"
音声特徴量とラベルデータを扱うDatasetクラスの定義をするソースコード。

Datasetのクラスの名前は、"SequenceDataset"としている。ここでは特徴量リストファイルとラベルファイルのパスを"__init__"関数に与えています。"__init__"関数内では、
特徴量ファイルのパスとラベルを読み込み、それぞれリスト("self.feat_list"、"self.label_list")を作成している。また"__init__"関数の後半では、"self.label_list"に対して、パディングという処理を行なっています。
本プログラムでは、1サンプル=1発話として扱います。Pytorchにおいて、ミニバッチデータは各サンプルのサイズが統一されている必要があるため、ラベル長や特徴量のフレーム数が統一されている必要がある。
そこで、フレーム数の最大値、ラベル長の最大値をそれぞれ求めておき、各特徴量やラベルの末尾を0などの値で埋めることで長さを最大値に統一する。これをパディング処理と呼ぶ。
本プログラムでは、1発話を1サンプルとするため、"__len__"関数では発話数"self.num_utts"を出力している。"__getitem__"関数では、特徴量ファイルリスト"self.feat_list"から指定された"idx"の
ファイルパスを取得し、その特徴量を読み込む。その後、特徴量が平地0、分散1になるように正規化処理を行なった上でスプライシング処理を行う。また、パディング処理を特徴量に対しても行う。
ただし、ここでは"pad_index"ではなく0でパディングを行う。


"my_model.py"
DNN-モデルクラスのソースコード。
このコードではReLU関数を活性化関数に用いたDNNを定義している。隠れ層の数を"num_layers"で定義できるようにしていて、"num_layers"の数だけ隠れ層(線形層)とReLU関数("nn.ReLU")をリストに加えている。
その後、リストをModuleListというPytorchのDNN用リストに変換している。また、DNNの構造を定義した後、LeCunの初期化をおこなっている。


"initializetation.py"
LeCunのパラメータ初期化を行うソースコード。
線形層の処理に担当するのは"elif dim == 2"の部分で、"param in model.parameters[]"で取り出したモデルのパラメータ"param"に対して、"param.data.size[1]"とすることで、
パラメータ数(=入力されるノードの数)を得ている。そして"param.data.normal_[0,std]"とすることで、正規分布に従う初期値を生成している。
入力特徴量のスプライシング処理は、前後5フレームを用いて行う設定としている。(次元数は11倍となる。)オプティマイザはモメンタムSGD("torch.optim.SGD", "momentum=0.99")を用いている。
損失関数はクロスエントロピーを用いる("torch.n.CrossEntropyLoss")。"CrossEntropyLoss"を用いる際の注意点として。この関数はsoftmax活性化関数の計算が内部に含まれているためDNNモデルクラスには、
softmaxの処理を出力そうで行う必要はない。また引数の"ignore_index=pad_index"も重要である。
今回データのパディング処理をしているため、デフォルト設定のままだとパディングされているフレームまで損失関数が計算されてしまい、学習結果に影響を与えてしまう。そこで、ラベルデータのパディングに用いている値"pad_index"を
引数""ignore_indexに与えておくことで、ラベルが"pad_index"となっているフレームの損失値が計算されず、学習結果に影響が出なくなる。


"03_dnn_recognize.py"
DNN-HMMによる孤立単語音声認識を行うメインプログラムのソースコード。
本コードでは学習時に使用したDNNモデルと同じモデルを"model = MyDNN[...]"として定義し、その後"model.load_state_dict[torch.load[dnn_file]]"とすることで、保存した学習モデルのパラメータを読み込んでいる。
また"state_counts"を読み込み、各状態の事前確率を計算している。次に、GMM-HMMでの孤立単語音声認識と同様に辞書を読み込む。音声特徴量は読み込まれた後、DNNの学習時と同様に平均0、分散1での
正規化とスプライシング処理を行う。その後DNNモデルに入力し、その出力を得る。注意点として、定義したDNNモデルにはsoftmax関数の処理が入っていない。そのためDNNの出力に対して、"torch.nn.functional.softmax"を
用いて、softmax関数を計算して確率に変換している。その後、認識を行う。


"hmmfunc.py"
DNNが計算した尤度を用いた認識処理部のソースコード。
本コードの"set_out_prob"関数は、与えられた確率値をHMMの各状態の出力確率に代入する関数。"recognize_with_dnn"関数は"set_out_prob"関数を用いてDNNが計算した尤度を用いている。
そしてビタビアルゴリズムによって認識をおこなっている。




本プログラムでは、1発話を1サンプルとするため、"__len__"関数では発話数"self.num_utts"を出力している。"__getitem__"関数では、特徴量ファイルリスト"self.feat_list"から指定された"idx"のtえいぎで
本プログラムでは、1発話を1サンプルとするため、"__len__"関数では発話数"self.num_utts"を出力している。"__getitem__"関数では、特徴量ファイルリスト"self.feat_list"から指定された"idx"の
