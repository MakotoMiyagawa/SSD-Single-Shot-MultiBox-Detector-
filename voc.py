# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:35:12 2022

@author: makoto.miyagawa
"""

'''
1.訓練、検証のイメージとアノテーションのファイルパスのリストを作成する関数

'''
import os.path as osp

def make_filepath_list(rootpath):
    '''データのパスを格納したリストを作成する
    
    Parameters:
        rootpath(str): データフォルダーのルートパス
    Returns:
        train_img_list : 訓練用イメージのパスリスト
        train_anno_list: 訓練用アノテーションのパスリスト
        val_img_list   : 検証用イメージのパスリスト
        val_anno_list  : 検証用アノテーションのパスリスト
    '''
    #画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')
    
    # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
    train_id_names = osp.join(rootpath + 'ImageSets/Main/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Main/val.txt')
    
    # 訓練データの画像ファイルとアノテーションファイルへのパスを保存するリスト
    train_img_list = list()
    train_anno_list = list()
    
    for line in open(train_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        # %sをファイルIDに置き換えて画像のパスを作る
        img_path = (imgpath_template % file_id)
        # %sをファイルIDに置き換えてアノテーションのパスを作る
        anno_path = (annopath_template % file_id)
        train_img_list.append(img_path)     # train_img_listに追加
        train_anno_list.append(anno_path)   # train_anno_listに追加
    
    # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = list()
    val_anno_list = list()
    
    for line in open(val_id_names):
        file_id = line.strip()                     # 空白スペースと改行を
        img_path = (imgpath_template % file_id)    # 画像の
        anno_path = (annopath_template % file_id)  # アノテーションのパス
        val_img_list.append(img_path)              # val_img_listに追加
        val_anno_list.append(anno_path)            # val_anno_listに追加
    
    return train_img_list, train_anno_list, val_img_list, val_anno_list

"""
２．バウンディングボックスの座標と正解ラベルをリスト化するクラス

"""
import xml.etree.ElementTree as ElementTree  # XMLを処理するライブラリ
import numpy as np  # NumPy

class GetBBoxAndLabel(object):
    """
    1枚の画像のアノテーション
    （BBoxの座標、ラベルのインデックス）をNumPy配列で返す
    
    Attributes:
        classese(list): VOCのクラス名(str)を格納したリスト
    """
    def __init__(self, classes):  # __init__はコンストラクタ　インスタンス変数の初期化
        """インスタンス変数にクラスのリストを格納する
        
        Parameters:
            classes(list): VOCのクラス名(str)を格納したリスト
        """
        self.classes = classes
        
    def __call__(self, xml_path, width, height):
        """インスタンスから実行されるメソッド
        
        １枚の画像のアノテーションデータをリスト化して多重リストにまとめる
        バウンディングボックスの各座標は画像サイズで割り算して正規化する
        
        Parameters:
            xml_path(str): xmlファイルのパス
            width(int): イメージの幅(正規化に必要)
            height(int): イメージの高さ(正規化に必要)
            
        Returns(ndarray):  # NumPyのN次元配列
            [[xmin, ymin, xmax, ymax, ラベルのインデックス], ...]
            要素数は画像内に存在するobjectの数と同じ
        """
        # 画像内のすべての物体のアノテーションを格納するリスト
        annotation = []
        
        # アノテーションのxmlファイルを読み込む
        xml = ElementTree.parse(xml_path).getroot()
        
        # イメージの中の物体(object)の数だけループする
        for obj in xml.iter('object'):
            # --アノテーションで検知がdifficultのものは除外--
            # difficultの値(0または1)をtextで取得してint型に変換
            # difficult==1の物体は処理せずにfor文の先頭に戻る
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue
            
            # 1つの物体に対するアノテーションを格納するリスト
            bndbox = []
            
            # <name>の要素(物体名)を抽出
            # 小文字に変換後、両端の空白削除
            name = obj.find('name').text.lower().strip()
            # <bndbox>を取得
            bbox = obj.find('bndbox')
            
            # アノテーションの xmin, ymin, xmax, ymaxを取得し、0~1に規格化
            grid = ['xmin', 'ymin', 'xmax', 'ymax']
            
            for gr in (grid):
                # バウンディングボックスの座標<xmin><ymin><xmax><ymax>を取得
                # VOCは原点が(1,1)なので1を引き算して
                # 各オフセットの原点を(0,0)の状態にする
                axis_value = int(bbox.find(gr).text) - 1
                # バウンディングボックスの座標を正規化
                if gr == 'xmin' or gr == 'xmax':
                    # xmin、xmaxの値をイメージの幅で割り算
                    axis_value /= width
                else:
                    # ymin、ymaxの値をイメージの高さで割り算
                    axis_value /= height
                # 'xmin' 'ymin' 'xmax' 'ymax'の値を順にbndboxに追加
                bndbox.append(axis_value)
            
            # 物体名のインデックスを取得
            label_idx = self.classes.index(name)
            # bndboxにインデックスを追加して物体のアノテーションリストを完成
            bndbox.append(label_idx)
            
            # すべてのアノテーションリストをannotationに格納
            annotation += [bndbox]
        # 多重リスト[xmin, ymin, xmax, ymax, 正解ラベルのインデックス], ...]
        # を2次元のNumPy配列(ndarray)に変換
        return np.array(annotation)
