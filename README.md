# ☁️ CloudClear LISS-IV

**Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery**

> Built for ISRO/NRSC hackathon. Lightweight, CPU-friendly, production-ready architecture.

---

## 🎯 Problem

Persistent cloud cover over NER India destroys 70%+ of optical satellite data. Traditional masking = black holes. We reconstruct what's underneath using generative AI.

## 🏗️ Architecture

- **Generator**: U-Net with learned Cloud Attention Module
- **Discriminator**: PatchGAN (multi-scale, 70×70 patches)
- **Loss**: Charbonnier + SSIM + Gradient + Adversarial + Spectral constraints
- **Input**: 256×256 patches, 3 bands (Green, Red, NIR)
- **Output**: Cloud-free reconstruction with preserved spectral angles

## 📊 Results

| Metric | Target | Status |
|--------|--------|--------|
| Cloud detection (attention) | Pixel-level | ✅ Working |
| Cloud removal | Visual consistency | ✅ Working |
| Spectral preservation | SAM < 5° | 🔄 In progress |
| Real data validation | Bhoonidhi LISS-IV | 🔄 Pending |

**Training progression:** Epoch 1 → 16 shows convergence of attention maps and spectral preservation.

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Train on synthetic data (default)
python scripts/train.py --config configs/default.yaml

# Inference on single image
python scripts/inference.py --input cloudy.tif --output clear.tif --checkpoint checkpoints/best.pth

# Evaluate metrics
python scripts/evaluate.py --pred_dir predictions/ --gt_dir ground_truth/
```

## 📁 Structure

```
cloudclear/
├── models/          # Generator, Discriminator, Losses
├── data/            # Dataset loaders + synthetic cloud generator
├── training/        # Training loop + callbacks
├── inference/       # Inference + evaluation
└── utils/           # Metrics, visualization helpers
```

## 🛠️ Tech Stack

- PyTorch
- Rasterio / GDAL
- OpenCV
- NumPy / Scikit-image
- Matplotlib

## 📜 License

MIT — open for ISRO, academic, and commercial use.

## 🙏 Acknowledgments

Built with guidance from the open-source remote sensing community. Data sources: Bhoonidhi (NRSC), Sentinel-2 (ESA), Landsat (USGS).

Status: Architecture validated on synthetic data. Integrating real LISS-IV scenes from Bhoonidhi.

Contact: roshan114400@gmail.com | www.linkedin.com/in/roshan-mohanvel-b7894a3a1
