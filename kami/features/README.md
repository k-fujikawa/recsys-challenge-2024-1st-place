candidate は impression x inview_article がユニークなキーとなる行
impression に user がひも付く


一つの手法で複数作成することもあり得るので番号の組み合わせで管理

- i: impression
    - impression ごとの特徴量を格納
- a: article
    - 対象となる article ごとの特徴量を格納
- u: user
    - 対象となる user ごとの特徴量を格納
- x: article x impression
    - articleとimpressionのkeyごとの特徴量を格納
- y: article x user
    - articleとuserのkeyごとの特徴量を格納
- c: candidateと同じ長さの特徴量を格納し横方向に結合するだけでOKにする

最初は demo データで処理をしてデバッグする

rename_dir_limit=20