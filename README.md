# GNR638 A3

# Paper: 
- SSD: Single Shot MultiBox Detector (https://arxiv.org/pdf/1512.02325v5)

# Blog:
- https://medium.com/@dheerajkumarmaradana/ssd-single-shot-multibox-detector-d7d570bbbe6f

# Main files:
- `ssd_from_scratch.py`: Implementation of SSD from scratch
- `train_voc.py`: Training script for SSD on VOC dataset
- `train_voc_torchvision.py`: Training script for SSD using torchvision's implementation on VOC dataset
- `test_voc.py`: Evaluation script for SSD on VOC dataset
- `utils.py`: Utility functions for data loading, augmentation, and evaluation
- `ssd_orig.py`: Original SSD implementation (not the complete code, just the model part)

# Setup Instructions:
- wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar   
- wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar        
- tar xf VOCtrainval_06-Nov-2007.tar
- tar xf VOCtest_06-Nov-2007.tar
- data is stored in VOCdevkit/VOC2007/

# Training:
- To train using from scratch implementation: `python3 train_voc.py`
- To train using torchvision's implementation: `python3 train_voc_torchvision.py`

# Evaluation:
- To evaluate the from scratch model: `python3 test_voc.py --checkpoint checkpoints/ssd_best.pth`
- To evaluate the torchvision model: `python3 test_voc.py --checkpoint checkpoints_tv/ssd_best.pth --torchvision`
- With visualizations: `python3 test_voc.py --checkpoint checkpoints/ssd_best.pth --save-vis --num-vis 50`
