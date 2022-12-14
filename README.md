# RetinaFace-Emotion-Detection

Facial Expression Recognition with RetinaFace as face detector and Region of Interest (RoI) based ResNet as Emotion Detector

### Docs
- [Presentation - EN](./docs/Report-EN.pptx)
- [Technical Report - FA](./docs/Report-FA.pdf)


### Classes

1. Angry
2. Disgust
3. Fear
4. Happy
5. Sad
6. Surprise
7. Neutral

## Model Checkpoints

You need to place checkpoints in ```checkpoints``` folder.
- RetinaFace model weights: [[Download]](https://drive.google.com/drive/folders/1kyIUlhB38igMCPTRgfZh536fUpAi3uOK?usp=sharing)
- Dlib landmark detector model weights [[Download]](https://drive.google.com/file/d/1CiwW6vJjFl22dQHR4fRC0lN2_pfsGVmM/view?usp=sharing)
- RoI ResNet56 trained on FER2013 model weights [[Download]](https://drive.google.com/file/d/1At3fv5F_47z8yidllaTveoS13itIqgkf/view?usp=sharing)




## How to run

Change directory to code folder
```bash
cd code
```

Install dependencies
```bash
conda env create -f environment.yml
```
```bash
conda activate retinaface-roi
```

Image and results will be saved in results folder
```bash
python run.py --img_path="../photos/test1.jpg"
```

Webcam
```bash
python run.py --webcam=True
```


## Results

Input

![input](./results/test1/test1.jpg)

Detected faces using RetinaFace

![face1](./results/test1/Faces-Processed/1_proc.jpg)
![face2](./results/test1/Faces-Processed/2_proc.jpg)
![face3](./results/test1/Faces-Processed/3_proc.jpg)
![face3](./results/test1/Faces-Processed/3_proc.jpg)
![face4](./results/test1/Faces-Processed/4_proc.jpg)
![face5](./results/test1/Faces-Processed/5_proc.jpg)

RoI Mask

![roi1](./results/test1/RoI-Processed/1_RoI_proc.jpg)
![roi2](./results/test1/RoI-Processed/2_RoI_proc.jpg)
![roi3](./results/test1/RoI-Processed/3_RoI_proc.jpg)
![roi3](./results/test1/RoI-Processed/3_RoI_proc.jpg)
![roi4](./results/test1/RoI-Processed/4_RoI_proc.jpg)
![roi5](./results/test1/RoI-Processed/5_RoI_proc.jpg)

Output (from left to right)
1. Happy
2. Happy
3. Neutral
4. Surprise
5. Surprise

## Webcam Demo

```bash
python run.py --webcam=True
```

![webcam](./results/webcam.jpg)


## References

https://github.com/peteryuX/retinaface-tf2
