# DEEP-CODI (Coronavirus Diagnostic)

### Brief:

The COVID-19 pandemic is severely impacting the health and wellbeing of countless people worldwide. Early detection of infected patients is a crucial first step in controlling the disease, which can be achieved through radiography, according to prior literature that shows COVID-19 causes chest abnormalities noticeable in chest x-rays.

Deep Codi learns these abnormalities and is able to accurately predict whether a patient is infected with coronavirus based on the patient’s chest x-ray. Codi is an effective diagnosis tool that has immediate downstream effects in clinical settings and in the field of radiology.


### Data:

The data folder is omitted from the git repo since it is large. For clarity and consistency, the folder structure is:

```
|code
|data
|--main_dataset
  |--test
    |--1_covid
    |--0_non
      |--Atelectasis
      |--Cardiomegaly
      |--Consolidation
      |--Edema
      |--Enlarged_Cardiomediastinum
      |--Fracture
      |--Lung_Lesion
      |--Lung_Opacity
      |--No_Finding
      |--Pleural_Other
      |--Pneumonia
      |--Pneumothorax
      |--Support_Devices
  |--train
    |--1_covid
    |--0_non
```

Where `main_dataset` has been renamed from the original folder `data_upload_v2`, as additional data sets may be added at a later point.
The `main_dataset` can be found [here](https://github.com/shervinmin/DeepCovid/tree/master/data). Additionally the covid and non folders were renamed to have their class labels with an underscore in front of their names.


### Docs:

Documents submitted for this project are conveniently linked here:
* [DevPost](https://devpost.com/software/deep-codi-coronavirus-diagnostic)
* [Outline](https://docs.google.com/document/d/1EEI7X_CQr9wfGwV87lb6Td_VjfkSVE8X5ixjkUxLoks/edit?usp=sharing)
* [Reflection](https://docs.google.com/document/d/1cysJC3PYWxQsm3N-E76wBlRUhUhj_eQgIkaq2fx0ai8/edit?usp=sharing)
* [Full Write-Up](https://docs.google.com/document/d/1CDUS4I8IUazt2MgNDGo80qzDctjvrCk8PTx2pGfq408/edit?usp=sharing)
* [Poster](https://docs.google.com/presentation/d/1QtGSQqj6rozHlAr_3FzNPmZVkF9BLhAr/edit#slide=id.p1)

### Running the Code:
All three models must be run with a python 3.6+ installation or virtual environment with all of the modules in requirements.txt installed. 

To run the training and testing for the hand written VGG-like model just run "python main_not_pretrained.py"

Stock VGG with random weights and the transfer learning VGG model are run slightly differently. 

Stock VGG is run with the "python stock.py <train/test>" for training or testing respectively.

Transfer learning VGG is run with the "python main.py <train/test>" for training or testing respectively.

### Other:

Please feel free to add / edit, or contact us for more details.
