同じディレクトリにpng画像を入れてこのプログラムを実行すると、以下の動作をします

1.  同じディレクトリのpngファイルを検索し、それぞれに対して手順2~4を行う
2.  輪郭を抽出する
    画像によって輪郭抽出条件が変わるので注意してください（デフォルトは赤成分に関する二値化）
3.  輪郭抽出結果の画像を出力する
4.  抽出した輪郭をなぞる点群データを出力する

出力した点群データはDS MechanicalというCADソフトで読み込み可能です（他のソフトでも可能かもしれません）。