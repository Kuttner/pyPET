# pyPET - A Python library for the analysis of positron emission tomography (PET) data

Samuel Kuttner. 2024-09-25

samuel.kuttner@uit.no

## Background
Positron emission tomography (PET) is a medical imaging technique for sampling the three dimensional spatial and temporal distribution of an intravenously injected radioactive tracer in a patient. 
It is an excellent tool for visualizing metabolic, physiological and functional information in cardiology, oncology and neurological disorders, e.g. Alzheimer’s and Parkinson’s disease. 
The most commonly used tracer is [^{18}F]-Fluorodeoxyglucose (FDG), a glucose analogue that is irreversibly trapped in cells with glucose metabolism. 

There are two basic ways to acquire PET data: Frozen in time (static), or as a time sequence (dynamic). Static PET images are simple to interpret and have a low noise level. However, they show merely a snapshot in time. 
In contrast, dynamic PET images, although requiring longer scans and having higher noise levels, comprise the whole radiotracer uptake pattern over time. 
Visual observation of the time frames is insufficient to fully exploit the potential with dynamic PET. Instead, tracer kinetic modeling may be used to assess biological parameters, such as the metabolic rate of glucose, drug receptor occupancy, myocardial blood flow, and oxygen consumption. 
Such measures are not possible to derive from a static PET image. 

Tracer kinetic modeling requires at least one tissue curve and and an arterial input function (AIF). 
This Python library contains scripts, data and an example notebook to perform tracer kinetic modeling for dynamic PET data.

## Aim
The purpose of this exercise is to learn how to perform tissue compartment modeling in Python using the reversible two tissue compartment model.

## Requirments
You can work either using your local standard Python installation, or by using for example Google Colabs.

## Dataset

Example data for this package is available in this repository under './Example_data/'.
  
The data consists of two files:
- The AIF of a mouse measured during 45 minutes with 1s sampling interval.
- Time-activity curves from four tissue regions (brain, left ventricle, liver and myocardium) measured during a 45 minute PET scan with framing: 1x30s, 24x5s, 9x20s, 8x300s.

## Citation
If you use code from this repository in your academic research or in publications, please cite the following paper:

Kuttner, S., Luppino, L. T., Convert, L., Sarrhini, O., Lecomte, R., Kampffmeyer, M. C., Sundset, R., & Jenssen, R. (2024). Deep learning derived input function in dynamic [18F]FDG PET imaging of mice. Frontiers in Nuclear Medicine, 4. https://doi.org/10.3389/fnume.2024.1372379

## License

pyPET
Copyright (C) 2024  Samuel Kuttner 

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation version 3. 

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  

See the GNU General Public License for more details, or refer to <https://www.gnu.org/licenses/>.
