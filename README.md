## 說明文件

### 本說明文件是關於SPN及SPN_V2(KGNA)兩個模型對於ICBEB(CPSC2018)和PTBXL兩個dataset的使用說明，兩個模型主要是對於ECG data中Arrhythmia的及早分類，關於程式執行步驟如下

1. `chmod +x get_data.sh` (賦予該執行檔執行權限)
2. `./get_data.sh` (產生ECG-data中的ICBEB/PTBXL兩個dataset, 並解壓縮模型)

### 模型樹狀圖如下
```
Production  - - - - - -模型目錄
  │
  ├─CNNLSTM  - - - - - -SPN-V2 backbone模型
  │  │─ICBEB ─ 1 ─ pretrained.pt
  │  └─PTBXL ─ 1 ─ pretrained.pt
  │
  ├─models  －－－－－－最終模型
  │  |─KGEVO  - - - - - -SPN-V2 最終模型
  |  |  |─ICBEB ─ 1 ─ model.pkl
  |  |  └─PTBXL ─ 1 ─ model.pkl
  |  └─SPN_V1  - - - - - -SPN 最終模型
  |     |─ICBEB ─ 1 ─ model.pt
  |     └─PTBXL ─ 1 ─ model.pt
  │
  ├─state  －－－－－－SPN-V2 state
  │  │─CNNLSTM ─     ICBEB ─ 1 ─ state.pkl
  |  └─CNNLSTM-500 ─ PTBXL ─ 1 ─ state.pkl
  │  
  │
  └─weights  －－－－－－SPN-V2 模型末端參數
  │  │─ICBEB ─ 1 ─ weight.pkl
  |  └─PTBXL ─ 1 ─ weight.pkl
```

3. `pip3 install -r requirements.txt` (安裝環境套件)
4. `cd Code-final` 
5. `python3 SPN_Online_Infernece.py ICBEB` (SPN執行第一個dataset)
6. `python3 SPN_Online_Infernece.py PTBXL` (SPN執行第二個dataset)
7. `python3 SPN_V2_Online_Infernece.py ICBEB` (SPN_V2執行第一個dataset)
8. `python3 SPN_V2_Online_Infernece.py PTBXL` (SPN_V2執行第二個dataset)


## Acknowledgement

This project was supported by Ministry of Science and Technology, Taiwan, under grant no. MOST 110-2634-FA49-002.
