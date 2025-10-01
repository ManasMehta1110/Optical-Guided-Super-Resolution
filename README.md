# Optical-Guided Super-Resolution for Thermal Imagery  

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)  
![Python](https://img.shields.io/badge/python-3.10%2B-blue)  
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)  
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-success)  
![Git LFS](https://img.shields.io/badge/Git%20LFS-enabled-yellow)  
[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-orange)](https://your-streamlit-app-link.com)  

---

## 🔍 Problem Statement  
Thermal infrared (TIR) remote sensing is crucial for **urban heat island analysis, wildfire monitoring, and precision agriculture**. However:  
- **Thermal sensors**: Low resolution (30m).  
- **Optical sensors**: High resolution (10m) but no thermal information.  

This mismatch reduces the effectiveness of TIR applications.  

The challenge: **Enhance coarse thermal images using high-resolution optical data while preserving thermal fidelity.**

---

## 🚀 Proposed Solution  
We developed a **fusion-based super-resolution pipeline** that leverages **deep learning (DualEDSR model)** to enhance low-resolution thermal images using spatial details from optical images.  

Key steps in our pipeline:  
1. **Data Preparation**: Extract optical + thermal bands from satellite imagery.  
2. **Resampling**: Align optical (10m) and thermal (30m) grids.  
3. **Model Training**: Train a DualEDSR super-resolution model with fusion of modalities.  
4. **Evaluation**: Metrics – **PSNR, SSIM, RMSE**.  
5. **Deployment**: An interactive **Streamlit web app** where users upload optical & thermal `.tif` files and get a **super-resolved thermal map**.  

---

## 🛠 Tech Stack  
- **Python 3.10+**  
- **PyTorch** – model training & inference  
- **Torchvision** – pretrained backbones  
- **Rasterio** – geospatial image handling  
- **NumPy, Matplotlib** – preprocessing & visualization  
- **Streamlit** – deployment & interactive UI  
- **Git LFS** – manage large model files  

---

## 📊 Results  
- **Validation after 3 epochs:**  
  - **PSNR**: 42.4 dB  
  - **SSIM**: 0.9269  
  - **RMSE**: 0.0159  

The model produces **sharper, high-resolution thermal outputs** while preserving **true temperature patterns**.  

---

## 🌐 Deployment  
We deployed a **Streamlit Cloud app** for real-world testing.  

**Workflow:**  
1. Upload **thermal (30m)** and **optical (10m)** `.tif` images.  
2. Model enhances the thermal image to **10m resolution**.  
3. View/download super-resolved thermal maps.  

---

## 📂 Repository Structure  

```
Optical-Guided-Super-Resolution/
│── data_raw/              # raw satellite data (ignored in repo)
│── data_processed/        # processed files + best_model.pth
│   └── best_model.pth     # trained model checkpoint
│── main2.ipynb            # training pipeline (model, dataset, training loop)
│── model.py               # DualEDSR model definition
│── streamlit_app.py       # Streamlit deployment script
│── requirements.txt       # dependencies
│── README.md              # project documentation
```

---

## 📥 Installation & Usage  

### 1️⃣ Clone Repository  
```bash
git clone https://github.com/ManasMehta1110/Optical-Guided-Super-Resolution.git
cd Optical-Guided-Super-Resolution
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit App  
```bash
streamlit run streamlit_app.py
```

---

## 📚 Research & References  
- Landsat-8/9 Operational Land Imager (OLI) & Thermal Infrared Sensor (TIRS) – NASA/USGS.  
- TorchGeo: https://huggingface.co/datasets/torchgeo/ssl4eo_l_benchmark  
- EDSR: Enhanced Deep Super-Resolution Network.  
- Smart India Hackathon Problem Statement (Thermal Super-Resolution).  

---

👉 This project demonstrates how **AI + remote sensing** can make thermal imaging more powerful and practical for **urban planning, agriculture, and disaster management**.  
