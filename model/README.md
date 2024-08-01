## Contains 1 quantization file
__quantcommon.py__  
It is the setting and combination of some quantitative means in Brevitas library

## Contains 8 network files

### Two benchmark networks

- __model_Full.py__  
  Full precision network: Lite-L3FNet(FP)
- __model_Quant.py__  
  Quantization network: Lite-L3FNet

### Six ablation implementation networks

#### Four strategies are necessary to prove the ablation network

- __model_None.py__  
  represents the network with 9*9 LF images, using 3D convolutions without DPP operations: Net_None, and it is referred to as `Net' in the paper.
- __model_77.py__  
  the model with disparity partitioning moved back to CC stage: Net_Undpp, and it is referred to as `Net' in the paper.
- __model_DPP.py__  
  the model using 3D convolutions: Net_3D, and it is referred to as `Net' in the paper.



#### Four strategies proved the effectiveness of the ablation network

- __model_Lp.py__  
  models that have slightly expanded channel and layer numbers for FE, CC, CA stages: Net_Lp.
- __model_w8bit.py__  
  only the quantized network whose weights are quantized using 8bit: Net_w8bit.
- __model_Op.py__  
  models that have slightly reduced channel and layer numbers for FE, CC, CA stages: Net_Op.
- __model_w2bit.py__  
  only the quantized network whose weights are quantized using 2bit: Net_w2bit
