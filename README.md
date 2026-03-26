# Processing code for 2021 McMurdo frazil ice, currents and supercooling data

This repository contains the processing scripts and notebooks used for the analyses presented in the preprint (submitted to *Journal of Geophysical Research: Oceans*) and archived on Zenodo under DOI: **10.5281/zenodo.18687544**.

The code supports the processing of ADCP, CT, CTD, and Icefin data collected in 2021, including calibration procedures, uncertainty analysis, frazil‑ice property extraction, and supercooling calculations.

---

## ✅ Requirements

The following Python packages are required:

- `seawater`  
- `gsw`  
- Python **3.8.1**  
- `numpy`  
- `matplotlib`  
- `opencv`  
- `pyTMD`  
- `xarray`  
- `pandas`  
- `cmcrameri`  

Additional packages may be needed depending on your analysis environment.

---

## 📁 Code Description

Below is an overview of all scripts and notebooks included in this repository, grouped by subsystem.

### **ADCP Processing**
- **`ADCP/calibration.py`**  
  Reads `.mat` files, applies the *Purdie (2012)* ADCP calibration, and saves processed output in NetCDF format.

- **`ADCP/Wind_rose_plot.ipynb`**  
  Generates **Figure 10** (wind‑rose plot).

- **`ADCP/TMD_currents.py`**  
  Computes modeled tidal current speed time series using **pyTMD**.

---

### **CT (Conductivity–Temperature) 2021**
- **`CT_2021/read.py`**  
  Parses SeaBird `.cnv` CT files and outputs a NetCDF dataset.

- **`CT_2021/CT_arm_processing.ipynb`**  
  Runs the parser, computes supercooling, and stores results in NetCDF format.

---

### **CTD 2021**
- **`CTD_2021/plot_CTD.py`**  
  Visualization routines for CTD cast results.

- **`CTD_2021/error_analysis.py`**  
  Adds uncertainty estimates and error bars to CTD plots.

- **`CTD_2021/SBE_cast2_3.ipynb`**  
  Plots all high‑quality SBE CTD casts.

---

### **Icefin**
- **`Icefin/analyze_frazil_2Nov.ipynb`**  
  Extracts frames from 4k videos and plots Icefin depth during filming.

- **`Icefin/Depth_of_Field.ipynb`**  
  Computes depth of field for the SubC 4 camera system.

- **`Icefin/frazil_size_3Nov.ipynb`**  
  Computes apparent frazil‑ice diameters and concentrations from Icefin video imagery.

- **`Icefin/analyze_videos.py`**  
  Processing routines for frazil ice apparent size and concentration extraction.

- **`Icefin/read.py`**  
  Parser for Icefin pressure‑sensor data from CSV files.

---

### **Cross‑Instrument Analyses**
- **`ADCP_and_Icefin.ipynb`**  
  Compares apparent frazil‑ice concentration (Icefin) with ADCP‑derived frazil proxy.

- **`ADCP_timeseries_analysis.ipynb`**  
  Time‑series analyses used to generate **Figure 11**.

- **`all_supercooling_2021.ipynb`**  
  Full 2021 supercooling time series combining both CT and CTD datasets.

- **`FIC_SC.ipynb`**  
  Combines Icefin apparent frazil‑ice concentrations with CTD‑derived supercooling.

- **`frazil_size_and_concentration_uncertainty.ipynb`**  
  Estimates uncertainty for apparent frazil‑ice concentration retrievals.

- **`supercooling_uncertainty.ipynb`**  
  Computes uncertainty estimates for CTD supercooling analyses.

---

## 📄 Citation

If you use this repository, please cite the Zenodo record:

> **Caldarella et al. (2025)**. *Data and code for 2021 McMurdo frazil ice, currents and supercooling analyses*.  
> Zenodo. DOI: **10.5281/zenodo.18687544**

A full journal citation will be added upon publication in *JGR Oceans*.

---

## ✅ Notes

- The repository contains **research‑grade processing pipelines**, not optimized for general deployment.
- Download "DATA.zip" from Zenodo and keep it two directories down from this code
- Some notebooks require manual file‑path configuration depending on your data directory structure.  
- All analysis steps match those used in the submitted manuscript.

---
