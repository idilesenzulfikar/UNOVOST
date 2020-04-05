## **UnOVOST: Unsupervised Offline Video Object Segmentation and Tracking**

### Overview
![Teaser](images/teaser.jpg#overview)

### Paper
[Jonahon Luiten*](https://www.vision.rwth-aachen.de/person/216/), Idil Esen Zulfikar*, [Bastian Leibe](https://www.vision.rwth-aachen.de/person/1/), "[UnOVOST: Unsupervised Offline Video Object Segmentation and Tracking](https://arxiv.org/abs/2001.05425)", WACV 2020

### Requirements

This code is written in Python 3.6 and the following modules are required:
- cv2
- PIL
- pycocotools
- scipy
- yaml
- pytorch

### Usage

Before running this code, prepare JSON files that contain mask, optical flow vector and ReID vector for each object proposal in a sequence.
- To generate masks for object proposals, we use this [Mask R-CNN implementation](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN).
- To extract Optical flow vector and ReID vector,we use [ReID network](https://github.com/JonathonLuiten/PReMVOS/tree/master/code) and [Optical-flow network](https://github.com/JonathonLuiten/PReMVOS/tree/master/code/optical_flow_net-PWC-Net) in [PremVOS](https://github.com/JonathonLuiten/PReMVOS).

An example json file can be downloaded from [this link](https://drive.google.com/file/d/1XdVZacEWJgh1ZfovAtg0NQXlSKu4jMUP/view?usp=sharing).

Afterwards, check your directory with JSON files to match this expected format:
    
    proposals/
        val/
            bike-packing/
                00000.json
                ...
                00079.json
                

Finally, run the code:
    
    python main.py --proposal_dir ../proposals/ --output_dir ../results/ --config  ../configs/unovost.yaml  
    

### Citation
```
@inproceedings{luiten2020unovost,
  title={UnOVOST: Unsupervised Offline Video Object Segmentation and Tracking},
  author={Luiten, Jonathon and Zulfikar, Idil Esen and Leibe, Bastian},
  booktitle={Proceedings of the IEEE Winter Conference on Applications in Computer Vision},
  year={2020}
}
```

### Contact
If you encounter any problems within the code or have any questions, please get in touch with Idil Esen Zulfikar (idil dot esen dot zuelfikar at rwth-aachen dot de) or [Jonathon Luiten](https://www.vision.rwth-aachen.de/person/216/)  (luiten at vision dot rwth-aachen dot de).
