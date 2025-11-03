# 🌍 Environmental Monitoring with YOLO

A deep learning project that applies **YOLO (You Only Look Once)** for **environmental monitoring** using satellite and drone imagery.  
The model detects people, animals, and vehicles in real time to help track deforestation, illegal hunting, and other environmental changes.

---

## ⚙️ How It Works
- **Dataset:** COCO 2017 (Common Objects in Context)  
- **Model:** YOLOv4 fine-tuned on COCO  
- **Process:**  
  1. Download and analyze dataset  
  2. Preprocess and augment images  
  3. Train YOLOv4 with TensorFlow  
  4. Evaluate precision & recall  
  5. Detect objects in satellite images  

---

## 🧠 Tech Stack
`Python` · `TensorFlow/Keras` · `Scikit-learn` · `Matplotlib` · `Requests`

---

## 🚀 Run It Yourself
```bash
git clone https://github.com/ekincelikdemir/environmental-monitoring-yolo
cd environmental-monitoring-yolo
pip install -r requirements.txt
python environmental_monitoring_code.py
```

---

## 📊 Results
| Metric | Score |
|:--|:--|
| **Precision** | ~0.92 |
| **Recall** | ~0.90 |

YOLOv4 achieved real-time detection performance (~45 FPS) and high accuracy on satellite imagery.

---

## 🛰️ Use Case
Detecting vehicles, people, and animals from aerial images to automate large-scale **environmental surveillance**.

---

## 🔗 Learn More
Read the full project breakdown and results here:  
👉 **[Medium – Automating Environmental Monitoring with YOLO](https://medium.com/@ekincelikdemir/automating-environmental-monitoring-with-yolo)**

---

**Author:** Ekin Cem Çelikdemir · Berlin, Germany  
📧 [LinkedIn](https://linkedin.com/in/ekincelikdemir) | [Medium](https://medium.com/@ekincelikdemir)
