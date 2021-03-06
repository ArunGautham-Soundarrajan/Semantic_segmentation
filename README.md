<h1 align="center">Evaluation of SOTA Semantic Segmentation Architectures</h1>


## 📝 Table of Contents
- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

# 🧐 About <a name = "about"></a>
With the advancement of AI and Computer vision, most of the things are being automated including self-driving cars, packaging, OCR etc. There are several State of the Art architecures out there, but how good are they for segmenting objects in a cluttered scene ? To find that out,we have evaluated the most popular SOTA architecture's using two Object in clutter datasets.

# 🏁 Getting Started <a name = "getting_started"></a>
These instructions will help you make you set up this project in your working environment and to reproduce the results.

* ## Prerequistes
The packages and versions used for this project can be found in *requirements.txt*. It is highly recommended to create a new virtual environment before executing the below command to avoid any independencies. 
```
pip install -r requirements.txt
```

* ## Datasets
The two datasets used in the project for training and evaluation are,
* [OCID](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/) (Object Clutter Indoor Dataset)
* [FAT](http://research.nvidia.com/publication/2018-06_Falling-Things) (Falling Things Dataset)

The dataset's can be downloaded from the given links and should be unizipped and placed in the main repository so a subset of it can be taken. 
```
📁 Semantic_segmentation/
├─📁 OCID-dataset/
│  ├─📁 OCID-dataset/
│    ├─📁 ARID10/
│    │ ├─📁 floor/
│    │ └─📁 table/
│    ├─📁 ARID20/
│    │ ├─📁 floor/
│    │ └─📁 table/
│    └─📁 YCB10/
│      ├─📁 floor/
│      ├─📄 info.txt
│      └─📁 table/
├─📁 fat/
│  ├─📁 fat/
│    ├─📁 mixed/
│    └─📁 single/
```
Make sure the unzipped data sits in the main repo with the following structure, so the subset can be taken automatically. Once the data is downloaded and placed like the above, the following `fat_data_processor.py` and `ocid_data_processor.py` can be run to process the data.

Alternatively, the processed data used in this project can be downloaded directly and placed in the main repo using the links below.
* [OCID](https://drive.google.com/drive/folders/16f2pH0Q1RXI7UkvLASOOcxdIZuVNO_P0?usp=sharing)
* [FAT](https://drive.google.com/drive/folders/10syG9kxpvBw6_k4Psx0SEu0aFH8jv42O?usp=sharing)


* ## Repo Structure
This is how the repo should be and folders for plots, metrics, models etc., will be created automatically after the first run.
This is just to make sure, the data is in the right repository
```
📁 Semantic_segmentation/
├─📁 Data_OCID/
│ ├─📁 images/
│ │ └─📄 image.png 1066 file(s)
│ └─📁 labels/
│   └─📄 mask.png 1066 file(s)
├─📁 fat_data/
│ ├─📁 images/
│ │ └─📄 image.jpg 4000 file(s)
│ └─📁 labels/
│   └─📄 mask.png 4000 file(s)
├─📁 metrics/
│ ├─📄 metrics.csv file(s)
├─📁 models/
│ ├─📄 models.pth file(s)
├─📁 plots/
│ ├─📄 loss.png file(s)
├─📁 test_plots/
│ ├─📄 inference.png file(s)
├─📄 README.md
├─📄 customDataset.py
├─📄 evaluation_metrics.py
├─📄 fat_data_processor.py
├─📄 inference.py
├─📄 main.py
├─📄 models.py
├─📄 ocid_data_processor.py
├─📄 plots.py
├─📄 prediction.py
├─📄 requirements.txt
└─📄 trainer.py
```

* ## Preparing the data
Once you unzipped both the data in the main directory as shown in the repo structure, you need to run these two scripts which would take the subset of the data to work with and create a seperate directory for them. In your terminal run these blocks to execute
```
python ocid_data_processor.py
```
This might take a while to run, as it copies each image and their label. Once its done, run the below line to take a subset for FAT dataset.
```
python fat_data_processor.py
```
* ## Pretrained Models

These are pretrained models and the folder can be placed in the main repo.
* [Pre-trained models](https://drive.google.com/drive/folders/14gaqN26VECiVGnDpsAlY-IBVWeRMkyr6?usp=sharing)

# 🎈 Usage <a name = "usage"></a>

The models can trained using the command line interface. The arguments are passed after the name of main.py file.
The arguments are passed in the order of model followed by dataset.

```
python main.py [model] [dataset]
```

To get help,

```
python main.py -h
```

Which will return a short description about the options for each arguments. The following are possible arguments that can be passed,

```
Model:
b :U-Net Model (Baseline)
d :DeepLab v3+
p :PSP Net

Data:
1 :OCID subset (ARID20)
2 :FAT subset.
```

Below the example for training a U-Net model on Falling things dataset,

```
python main.py b 2
```
  


# ✍️ Authors <a name = "authors"></a>
* [Arun Gautham Soundarrajan](https://github.com/ArunGautham-Soundarrajan)

# 🎉 Acknowledgments <a name = "acknowledgement"></a>
 
Project Supervisor : [Dr. Gerardo Aragon Camarasa](https://www.gla.ac.uk/schools/computing/staff/gerardoaragoncamarasa/) 
