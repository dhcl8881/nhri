# 肝臟細胞核值比計算
## 環境設置
1. python 3.12.10
1. opencv-python 4.12.0.88
1. pillow 11.0.0
1. pytorch 2.7.1+cu128
1. torchvision 0.22.1
1. pandas 2.3.0
1. cuda 12.8
需要安裝cuda  
## 流程圖
![圖片](/cell/流程圖.jpg)
## 使用方法
先下載模型參數<br>
stage1模型參數 https://drive.google.com/file/d/1XNjVBBhplQ0WfAXGde6miDq-deYH4gTz/view?usp=sharing <br>
stage2模型參數 https://drive.google.com/file/d/1tXKcYGA_5wFxG2nL_ByL_E4cYYiAj3U8/view?usp=sharing <br>
下載後解壓到cell資料夾下

```
cd ./nhri/cell  #進入cell資料夾
./main.py --mix_dir 圖片位置
#注意 圖片檔名不能有空白 ex: channel red.jpg 檔名有空白會報錯
#example python main.py --mix_dir ./4/4_channel_mix.tif
```
會在sample資料夾產生每個細胞的細胞質分割範圍<br>
![圖片](/cell/sample/0.jpg)<br>
會在test資料夾產生整張圖片細胞的細胞質分割範圍和csv檔，csv檔內有每個細胞的細胞核面積和細胞質面積和胞質比。csv檔內最後一項為平均值
![圖片](/cell/test/test.jpg)<br>
## 實驗結果
![圖片](/cell/result.jpg)<br>
