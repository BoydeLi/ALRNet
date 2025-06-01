# [ICASSP 2025] Camouflaged Object Detection via Neural Architecture Search
Xin Li, Keren Fu, Qijun Zhao<br />

## ✈ Abstract
The core challenge in camouflaged object detection (COD) is
identifying objects that blend seamlessly with their surroundings. Existing
methods emulate the strategies biological organisms break camouflage
by manually constructing modules with expert knowledge from existing
segmentation tasks, making it difficult to accurately understand complex
and unique camouflage semantics. We are the first to apply neural
architecture search (NAS) to COD, introducing an automatic localization
and refinement network called ALRNet. It explores a large search
space to discover more effective camouflage-specific modules. Specifically,
we propose a search-based automatic receptive field block (ARFB)
to adaptively excavate hierarchical discriminative cues and decouple
features in a multi-branch architecture. Moreover, we introduce an
edge-assisted explicit and implicit refinement (EEIR) module, combining
explicit priors with implicit search to create a dual-task structure for
edge and segmentation knowledge interaction.

<img src="imgs/ALRNet.png">

## ✈ Environmental Setups
`PyTorch 2.2.0 + CUDA 12.1`. Please install corresponding PyTorch and CUDA versions.
To create anaconda environment directly, please run flowing commands.
```
conda create -n ALRNet python=3.9.20
conda activate ALRNet
pip install -r requirements.txt
```
For the requirements.txt, please click [HERE](https://drive.google.com/drive/folders/1fE_DCGKU3WA-HmZnqRzaazHe44Lx9RP2?usp=sharing).
For more detail Vmamba environment configuration, please see [HERE]().

## ✈ Dataset and Training
Please download the [DIS-5K dataset](https://github.com/xuebinqin/DIS) first and place them in the "**COD_dataset**" directory. The structure of the "**COD_dataset**" folder should be as follows:

```
IS-Net
└──DIS5K
    └── DIS-TE1
    ├── DIS-TE2
    ├── DIS-TE3
    ├── DIS-TE4
    ├── DIS-TR
    └── DIS-VD
    	├──im
    	├──gt
    	└──mask #SAM MASK
```

To train or validate the ALRNet, please run:
```
python train.py
```

VMamba-S backbone weights：[[baidu](https://pan.baidu.com/s/199p0p9OfkQXqWVGaxco1lg)，提取码：c5t4]

Full Samba weights：[[baidu](https://pan.baidu.com/s/15787DVEmW59ftztopv-yMg)，提取码：bkvw]

## ✈ Quantitative Results
<img src="imgs/results.png" style="width: 80%;"/>

## ✈ Visual Results
<img src="imgs/visual results.png" style="width: 80%;"/>

## ✈ Citation
If you use Samba in your research or wish to refer our work, please use the following BibTeX entry.
```
@inproceedings{li2025camouflaged,
  title={Camouflaged Object Detection via Neural Architecture Search},
  author={Li, Xin and Fu, Keren and Zhao, Qijun},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}

## ✈ The search results of ALRNet:
<div align=center>
<img src="imgs/ARFB1.png" width="60%">
<img src="imgs/ARFB2.png" width="60%">
<img src="imgs/ARFB3.png" width="60%">
<img src="imgs/ARFB4.png" width="60%">
<img src="imgs/ARFB5.png" width="60%">
<img src="imgs/ACLM34.png" width="60%">
<img src="imgs/ACLM45.png" width="60%">
<img src="imgs/ACSF.png" width="60%">
<img src="imgs/SBR.png" width="60%">
<img src="imgs/EBR.png" width="60%">
<img src="imgs/IM.png" width="60%">
<img src="imgs/LowSBR.png" width="60%">
<img src="imgs/LowEBR.png" width="60%">
<img src="imgs/LowIM.png" width="60%">
</div>
