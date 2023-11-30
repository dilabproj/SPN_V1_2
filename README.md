## Documentation

### This documentation is about the use of two models, SPN and SPN_V2 (KGNA), for the ICBEB (CPSC2018) and PTBXL datasets. These models are primarily for the early classification of Arrhythmia in ECG data. The steps to execute the program are as follows:

1. `chmod +x get_data.sh` (Grant execute permission to the script)
2. `./get_data.sh` (Generate ICBEB/PTBXL datasets in ECG-data and decompress the model)

### The structure is as follows:

```
Production  - - - - - -Model Directory
  │
  ├─CNNLSTM  - - - - - -SPN-V2 backbone model
  │  │─ICBEB ─ 1 ─ pretrained.pt
  │  └─PTBXL ─ 1 ─ pretrained.pt
  │
  ├─models  －－－－－－Final Models
  │  |─KGEVO  - - - - - -SPN-V2 Final Model
  |  |  |─ICBEB ─ 1 ─ model.pkl
  |  |  └─PTBXL ─ 1 ─ model.pkl
  |  └─SPN_V1  - - - - - -SPN Final Model
  |     |─ICBEB ─ 1 ─ model.pt
  |     └─PTBXL ─ 1 ─ model.pt
  │
  ├─state  －－－－－－SPN-V2 state
  │  │─CNNLSTM ─     ICBEB ─ 1 ─ state.pkl
  |  └─CNNLSTM-500 ─ PTBXL ─ 1 ─ state.pkl
  │  
  │
  └─weights  －－－－－－SPN-V2 End Parameters
  │  │─ICBEB ─ 1 ─ weight.pkl
  |  └─PTBXL ─ 1 ─ weight.pkl
```

3. `pip3 install -r requirements.txt` (Install environment packages)
4. `cd Code-final` 
5. `python3 SPN_Online_Infernece.py ICBEB` (Run SPN with the first dataset)
6. `python3 SPN_Online_Infernece.py PTBXL` (Run SPN with the second dataset)
7. `python3 SPN_V2_Online_Infernece.py ICBEB` (Run SPN_V2 with the first dataset))
8. `python3 SPN_V2_Online_Infernece.py PTBXL` (Run SPN_V2 with the second dataset)


## Acknowledgement

This project was supported by Ministry of Science and Technology, Taiwan, under grant no. MOST 110-2634-FA49-002.
