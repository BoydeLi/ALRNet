# [ICASSP 2025] Camouflaged Object Detection via Neural Architecture Search
Xin Li, Keren Fu, Qijun Zhao<br />

## ✈ Overview
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

<img src="imgs/ARFB1.png">

The search results of ALRNet:
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
