# pyPET - A Python library for the analysis of positron emission tomography (PET) data

Author: 

Samuel Kuttner<sup>1,2</sup>, PhD, Medical Physicist

<sup>1</sup>PET Imaging Center, University Hospital of North Norway

<sup>2</sup>UiT Machine Learning Group, Department of Physics and Technology, UiT The Arctic University of Norway

samuel.kuttner@uit.no

[Google Scholar](https://scholar.google.se/citations?hl=en&user=AFVmH7AAAAAJ&view_op=list_works&sortby=pubdate)

## Background
Positron emission tomography (PET) is a medical imaging technique for sampling the three dimensional spatial and temporal distribution of an intravenously injected radioactive tracer in a patient. 
It is an excellent tool for visualizing metabolic, physiological and functional information in cardiology, oncology and neurological disorders, e.g. Alzheimer’s and Parkinson’s disease. 
The most commonly used tracer is [<sup>18</sup>F]-Fluorodeoxyglucose (FDG), a glucose analogue that is irreversibly trapped in cells with glucose metabolism. 

There are two basic ways to acquire PET data: Frozen in time (static), or as a time sequence (dynamic). Static PET images are simple to interpret and have a low noise level. However, they show merely a snapshot in time. 
In contrast, dynamic PET images, although requiring longer scans and having higher noise levels, comprise the whole radiotracer uptake pattern over time. 
Visual observation of the time frames is insufficient to fully exploit the potential with dynamic PET. Instead, tracer kinetic modeling may be used to assess biological parameters, such as the metabolic rate of glucose, drug receptor occupancy, myocardial blood flow, and oxygen consumption. 
Such measures are not possible to derive from a static PET image. 

Tracer kinetic modeling requires at least one tissue curve and and an arterial input function (AIF). 
This Python library contains scripts, data and an example notebook to perform tracer kinetic modeling for dynamic PET data.

Voxel-wise (parametric) implementation for the Patlak model using least squares estimation is included, as well as a for-loop-based implementation voxel-wise Two-tissue reversible compartment model (very slow).

## Aim
The purpose of this library is to provide scripts to process PET imaging data in Python, including tracer kinetic modeling and basic reading of PET DICOM files. 

[Tissue_compartment_modeling.py](Tissue_compartment_modeling.py) contains code for the following compartment models:
- One-tissue compartment model [1]
- Two-tissue irreversible compartment model [1]
- Two-tissue reversible compartment model (regional and voxe-wise (parametric)) [1]
- Patlak graphical analysis (regional and voxe-wise (parametric)) [2]

The one- and two-tissue compartment models now also support using a fixed value of the fractional blood volume, vB. Simply call the functions with arguments (...fit_vB=False, vB_fixed=vB), where vB is the desired fixed value of the fractional blood volume.

In addition, for visualization purposes, the commonly used Feng parametric AIF model [3] is included.

## Example files
- The Jupyter notebook [dPET_Kinetic_modeling_example.ipynb](dPET_Kinetic_modeling_example.ipynb) contains an example/exercise to show how to perform tissue compartment modeling in Python using some of the mentioned compartment models.
- [Read_PET_dicom_file.py](Read_PET_dicom_file.py) contains an example of how to read a a PET DICOM file from a Siemens mMR scanner

## Requirments
You can work either using your local standard Python installation, or by using for example Google Colabs.

## Dataset

The [Example_data](Example_data) for this repository consists of two files:
- The AIF of a mouse measured during 45 minutes with 1s sampling interval.
- Time-activity curves from four tissue regions (brain, left ventricle, liver and myocardium) measured during a 45 minute PET scan with framing: 1x30s, 24x5s, 9x20s, 8x300s.

## Citation
If you use code from this repository in your academic research or in publications, please cite the following paper:

- Kuttner, S., Luppino, L. T., Convert, L., Sarrhini, O., Lecomte, R., Kampffmeyer, M. C., Sundset, R., & Jenssen, R. (2024). Deep learning derived input function in dynamic [18F]FDG PET imaging of mice. Frontiers in Nuclear Medicine, 4. https://doi.org/10.3389/fnume.2024.1372379

## References

1. Gunn, R. N., Gunn, S. R., & Cunningham, V. J. (2001). Positron emission tomography compartmental models. Journal of Cerebral Blood Flow and Metabolism : Official Journal of the International Society of Cerebral Blood Flow and Metabolism, 21(6), 635–652. https://doi.org/10.1097/00004647-200106000-00002

2. Patlak, C. S., & Blasberg, R. G. (1985). Graphical evaluation of blood-to-brain transfer constants from multiple-time uptake data. Generalizations. Journal of Cerebral Blood Flow and Metabolism : Official Journal of the International Society of Cerebral Blood Flow and Metabolism, 5(4), 584–590. https://doi.org/10.1038/jcbfm.1985.87

3. Feng, D., Huang, S. C., & Wang, X. (1993). Models for computer simulation studies of input functions for tracer kinetic modeling with positron emission tomography. International Journal of Bio-Medical Computing, 32(2), 95–110. https://doi.org/10.1016/0020-7101(93)90049-C

## License

pyPET
Copyright (C) 2025  Samuel Kuttner 

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation version 3. 

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  

See the GNU General Public License for more details, or refer to <https://www.gnu.org/licenses/>.
