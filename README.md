# FACEMASK DETECTION

This project is all about detecting whether an image has a face wearing a mask or not.

Find the dataset from [Google Drive](https://drive.google.com/drive/folders/1FiNjWDb5kC3VbJS_ofisS4vsth4ziyuf?usp=sharing)

The following are the libraries required to run this project all available in [PyPI](https://pypi.org/):

- opencv-python
- tensorflow

In the models folder are two model files:
1. **model-v1.h5**: Version 1 of the model. It's overfitting.
2. **model-v2.h5**: Version 2 of the model. Dealt with overfitting by introducing Data Augmentation and Dropout Regularization.

### To run a live demo via your webcam or any other video source
Run the **video.py** file [Example Preview](https://youtu.be/yr5Zt3Fzsao)

### To work on images only
Run the **image.py** file

## FUTURE UPDATES
This project is a working progress and the following are the updates am looking to make:
- Find the optimal cut-off point
- Use a pretrained model (Transfer Learning)