## Store all the experimental results, including pfm files, disparity maps, and BadPix maps.
Each folder corresponds to each experimental group:
- Net_Quant: main experimental group, quantized version of RE-LFDE.
- Net_Full: The main experimental group, the full precision version of RE-LFDE.
- Net_None/mse/81: ablation experimental group, ```baseline```, the algorithm after adding our tailoring strategy on top of the advanced algorithm, is used as baseline.
- Net_None/mse/49: ablation experimental group, ```baseline+7*7```, using 7*7 inputs on top of baseline.
- Net_DPP: Ablation experimental group, ```baseline+7*7+DPP```, adding our proposed DPP operation on baseline+7*7.
- Net_Quant_LP: ablation experimental group, ```Net(Lp)```, reducing the pruning strength on the basis of Net_Quant.
- Net_Quant_OP: ablate the experimental group, ```Net(Op)```, and increase the pruning strength on the basis of Net_Quant.
- Net_Quant_w8bit: ablation experimental group, ```Net(w8bit)```,, using 8bit weights to quantify the network on the basis of Net_Quant.
- Net_Quant_w2bit: ablation experimental group, ```Net(w2bit)```,, which quantifies the network using 2bit weights based on Net_Quant.
