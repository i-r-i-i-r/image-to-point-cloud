

import glob
import cv2
import os
import numpy as np
import pandas as pd

# DSMechに点群データを入力するための、データ変換と出力
# 引数は np.array[点数, 2or3]（[x,y], or [x,y,z]）
def output_point_cloud(arr0, name):
    type_ = str(type(arr0[0,0])).split("numpy.")[1].split("'")[0]
    if arr0.shape[1] ==2:
        arr_zxy = np.hstack([ np.zeros((arr0.shape[0],1),type_), arr0 ]) # z座標の追加
    else:
        arr_zxy = np.hstack([ arr0[:,2:], arr0[:,:2] ]) # xyz -> zxy
    
    df_main = pd.DataFrame(arr_zxy, columns=["3d=true","",""]) # メインのデータ

    df_last = pd.DataFrame([arr_zxy[0]], columns=df_main.columns) # メインのデータの最初の行(最後に一つ足して一周するようにする)
    df_2ndrow = pd.DataFrame([["polyline=false","",""]], columns=df_main.columns) # (index行を含めて)2行目
    df_out = pd.concat([df_2ndrow, df_main, df_last]) 
    df_out.to_csv( name+".txt", index = False, sep = "\t")


# メイン処理
def main():
    
    img_name_list = sorted(glob.glob("*.png")) # 画像ファイルの検索
    
    for img_name in img_name_list:
        img_0 = cv2.imread(img_name, -1) # 画像の読み込み
        index = np.where(img_0[:, :, 3] == 0)
        # 透過部分を白塗りする
        img_0[index] = [255, 255, 255, 255]
        
        # 出力フォルダの生成
        main_path = os.getcwd() # 現在の作業中フォルダのパスの取得
        output_path = main_path + '\output_'+img_name.split(".")[0] # 出力フォルダ名の設定
        os.makedirs(output_path, exist_ok=True) # ない場合は作る
        os.chdir(output_path) # 出力フォルダへの移動
        
        
        # 輪郭検出のための前処理
        # ここはケースによっていじらなくてはいけない
        img_1 = img_0[:,:,2] # BGRからR成分のみを抽出
        threshold = 100
        ret, img_2 = cv2.threshold(img_1, threshold, 255, cv2.THRESH_BINARY)
        img_3 = 255-img_2 # 白黒反転

        # 輪郭検出
        contours, hierarchy = cv2.findContours(img_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(filter(lambda x: cv2.contourArea(x) > 10, contours))# 小さい輪郭は誤検出として削除

        # 輪郭画像の出力（結果確認のため）
        if img_0.shape[2]==3:
            c_ = (0, 0, 255)
        else:
            c_ = (0, 0, 255, 255)
        
        img_5 = cv2.drawContours(img_0, contours, -1, color=c_, thickness=2)
        cv2.imwrite("contour.png",img_5)

        for i_contour in range(len(contours)):
            contour_list = contours[i_contour]
            contour_array = contour_list.reshape(contour_list.shape[0], 2) # np.array化
            output_point_cloud(contour_array, "contour"+str(i_contour) ) # データの変換と出力
        
        os.chdir(main_path)

main()


