![UTA-DataScience-Logo](https://github.com/user-attachments/assets/36b0607e-06da-485c-97a1-34a4f0552141)

# ICU Mortality and Readmission Prediction Capstone 1

This repository contains a deep learning project that identifies Southeast Asian countries from Google Street View images, using scene-specific coordinates and queries to gather real-world visual data.

* **IMPORTANT:** This dataset was created using © Google Street View Static API. THe actual imagery is not distributed here to comply with Google's terms of service. You must obtain your own API key and download the images using the provided coordinates and pipelines

## OVERVIEW

This project

## SUMMARY OF WORK DONE

### Data

  * **Type:**
    * Input: 
    * Output: 
  * **Source:**
    * Custom-built data set using queries from Overpass Turbo and images from Google Street View Static API
  * **Size of Classes:**
    * 886 images from Indonesia, 893 from Malaysia, 896 from the Philippines, and 898 from Thailand. 3573 total images
  * **Splits:**
    * 80% training and 20% validation
   
#### Compiling Data and Image Pre-processing

* **Data Collection and Cleaning:**
    * h
    * h
* **Pre-processing:**
 
#### Data Visualization

viz here

### Problem Formulation

* **Models Used:**
  * ****
  * **M=**
  * **m** 
 
### Training

Model training was conducted locally on a custom-built **Windows PC equipped with an AMD Ryzen 7 7700X CPU, RTX 4060 GPU, and 64 GB of DDR5 RAM**, using Jupyter Notebook. The training utilized TensorFlow/Keras along with key libraries such as numpy, matplotlib, and pandas.

**Challenges:** o

### Performance Comparison

#### **b**

#### **w**

### Conclusions

**WHAT THIS MEANS:** wee

### Future Work
 
* **Increase dataset size** to support deeper and more expressive models
* **Collect more diverse scenery samples**, especially for countries with regional visual differences (e.g., urban vs. rural Philippines)
* **Improve sampling strategy** by selecting coordinates ***randomly*** across the entire country instead of sequentially, reducing spatial bias
* **Ensure broader geographic coverage** to avoid clustering in specific types of landscapes (e.g., beaches only)
* **Incorporate finer-grained features** (types of vehicles like tricycles or tuk-tuks) that may help distinguish between countries

## HOW TO REPRODUCE RESULTS

### Overview of Files in Repository

The list below follows the chronological order in which each component of the project was developed:

* **file:** weee

### Software Setup

This project was developed and executed in Google Colab Jupyter Notebook. If you don’t already have it installed, you can download it as part of the Anaconda distribution or install it via pip "pip install notebook".

* packages

### Data

* **Websites Used:**
    * **data:** https://overpass-turbo.eu/
* Dataset was built using Google Street View Static API with manually curated coordinates across Indonesia, Malaysia, the Philippines, and Thailand
* Coordinates were gathered using map queries for specific sceneries (e.g., beaches, cities), reprojected, and sorted into a GeoDataFrame. API calls checked for available imagery; first 150 valid coordinates per scenery per country were used to conserve quota
    * **For reference, see GeoJSON_to_CSV.ipynb**
 
### Training

* Install required packages in notebook
* Download and prepare the data (either from scratch or above, or use files in the Coordinates folder of this directory)
* Models were trained using TensorFlow/Keras with early stopping and validation monitoring. Images were split into training and validation sets, preprocessed (resized and batched), and fed into models like MobileNetV2 and ResNet50. Training ran on a local machine with GPU support over multiple sessions

***For reference, see g.ipynb***

#### Performance Evaluation

* After training, model performance can be evaluated using the validation set

***For reference, see m.ipynb***

## CITATIONS

[1]

[2] 
