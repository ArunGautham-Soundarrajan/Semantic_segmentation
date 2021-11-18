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

The dataset's can be downloaded from the given links.

* ## Repo Structure

**TBD**

* ## Preparing the data
Once you unzipped both the data in the main directory as shown in the repo structure, you need to run these two scripts which would take the subset of the data to work with and create a seperate directory for them. In your terminal run these blocks to execute
```
python ocid_data_processor.py
```
This might take a while to run, as it copies each image and their label. Once its done, run the below line to take a subset for FAT dataset.
```
python fat_data_processor.py
```


# 🎈 Usage <a name = "usage"></a>

**TBD**

# ✍️ Authors <a name = "authors"></a>
* [Arun Gautham Soundarrajan](https://github.com/ArunGautham-Soundarrajan)

# 🎉 Acknowledgments <a name = "acknowledgement"></a>

OCID Dataset
```
@inproceedings{DBLP:conf/icra/SuchiPFV19,
  author    = {Markus Suchi and
               Timothy Patten and
               David Fischinger and
               Markus Vincze},
  title     = {EasyLabel: {A} Semi-Automatic Pixel-wise Object Annotation Tool for
               Creating Robotic {RGB-D} Datasets},
  booktitle = {International Conference on Robotics and Automation, {ICRA} 2019,
               Montreal, QC, Canada, May 20-24, 2019},
  pages     = {6678--6684},
  year      = {2019},
  crossref  = {DBLP:conf/icra/2019},
  url       = {https://doi.org/10.1109/ICRA.2019.8793917},
  doi       = {10.1109/ICRA.2019.8793917},
  timestamp = {Tue, 13 Aug 2019 20:25:20 +0200},
  biburl    = {https://dblp.org/rec/bib/conf/icra/SuchiPFV19},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
FAT Dataset
```
@INPROCEEDINGS{tremblay2018arx:fat,
  AUTHOR = "Jonathan Tremblay and Thang To and Stan Birchfield",
  TITLE = "Falling Things: {A} Synthetic Dataset for {3D} Object Detection and Pose Estimation",
  BOOKTITLE = "CVPR Workshop on Real World Challenges and New Benchmarks for Deep Learning in Robotic Vision",
  MONTH = jun,
  YEAR = 2018}
```
