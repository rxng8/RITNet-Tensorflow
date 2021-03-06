# Replication of RITNet

## About the project
* The source paper and github project of RITNet is here:
  * Source: https://arxiv.org/abs/1910.00694
  * Git: https://github.com/AayushKrChaudhary/RITnet

```
@inproceedings{chaudhary2019ritnet,
  title={RITnet: real-time semantic segmentation of the eye for gaze tracking},
  author={Chaudhary, Aayush K and Kothari, Rakshit and Acharya, Manoj and Dangi, Shusil and Nair, Nitinraj and Bailey, Reynold and Kanan, Christopher and Diaz, Gabriel and Pelz, Jeff B},
  booktitle={2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW)},
  pages={3698--3702},
  year={2019},
  organization={IEEE}
}
```

## Requirements:
* Anaconda
* Conda environment with Python 3.8

## Results:
* This is the result trained from 1 epoch of 8000 objects (~34,000 objects in the training set). These are trained with 1 GPU NVIDIA GTX 1060 Max-Q 6 Gb. Trained with normal CE loss.
  <div align="center">
    <p>Input:</p>
    <img src="./docs/figures/training_config_1_SimpleUNet2/example_input.png" width=300/>
    
    <p>Inference</p>
    <img src="./docs/figures/training_config_1_SimpleUNet2/example_output.png" width=300/>
  </div>

* This is the result trained from 10 epoch of 34,000 objects per epoch. These are trained with 1 GPU NVIDIA RTX 3080 12 Gb. Trained with normal CE loss
  <div align="center">
    <p>History</p>
    <img src="./docs/figures/training_config_1_SimpleUNet2/simpleUNet2_training_result.png" width=300/>

    <p>Input</p>
    <img src="./docs/figures/training_config_1_SimpleUNet2/example_input_2.png" width=300/>
    
    <p>Inference</p>
    <img src="./docs/figures/training_config_1_SimpleUNet2/example_output_2.png" width=300/>
  </div>

* This is the result trained from 10 epoch of 34,000 objects per epoch. These are trained with 1 GPU NVIDIA RTX 3080 12 Gb. Trained with normal CE and GDL loss.
  <div align="center">
    <p>History</p>
    <img src="./docs/figures/training_config_2_SimpleUNet2/history.png" width=300/>

    <p>Input</p>
    <img src="./docs/figures/training_config_2_SimpleUNet2/example_input_1.png" width=300/>
    
    <p>Inference</p>
    <img src="./docs/figures/training_config_2_SimpleUNet2/example_output_1.png" width=300/>
  </div>

* This is the result trained from 10 epoch of 34,000 objects per epoch. These are trained with 1 GPU NVIDIA RTX 3080 12 Gb. Trained with normal CE, GDL loss, and Surface Loss.
  <div align="center">
    <p>History</p>
    <img src="./docs/figures/training_config_3_SimpleUNet2/history.png" width=300/>

    <p>Input</p>
    <img src="./docs/figures/training_config_3_SimpleUNet2/input.png" width=300/>
    
    <p>Inference</p>
    <img src="./docs/figures/training_config_3_SimpleUNet2/output.png" width=300/>
  </div>

* Surface Loss: This is the example result of generating distance matrix based on preprocessed label.
  <div align="center">
    <p>Input (not preprocessed label)</p>
    <img src="./docs/figures/dist_matrix/input.png" width=300/>
    
    <p>Distance matrix</p>
    <img src="./docs/figures/dist_matrix/dist_mat.png" width=300/>
  </div>

* BAL Loss:
  * This is the example result of canny edge detection based on preprocessed label.
  <div align="center">
    <p>Input (not preprocessed label)</p>
    <img src="./docs/figures/bal/input.png" width=300/>
    
    <p>Canny</p>
    <img src="./docs/figures/bal/canny.png" width=300/>
  </div>

  * Model with BAL Loss integration is still in debugging stage and has not been trained.
  
## Misc
* Install TensorRT
```
python -m pip install C:\Users\GBURG-4\Documents\TensorRT-7.2.2.3.Windows10.x86_64.cuda-11.1.cudnn8.0\TensorRT-7.2.2.3\graphsurgeon\graphsurgeon-0.4.5-py2.py3-none-any.whl

python -m pip install C:\Users\GBURG-4\Documents\TensorRT-7.2.2.3.Windows10.x86_64.cuda-11.1.cudnn8.0\TensorRT-7.2.2.3\uff\uff-0.6.9-py2.py3-none-any.whl

python -m pip install C:\Users\GBURG-4\Documents\TensorRT-7.2.2.3.Windows10.x86_64.cuda-11.1.cudnn8.0\TensorRT-7.2.2.3\onnx_graphsurgeon\onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl
```