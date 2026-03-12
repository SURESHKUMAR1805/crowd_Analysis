---
title: CrowdShield — Crowd Threat Detection
emoji: ⚡
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: true
---

# ⚡ CrowdShield — Real-Time Crowd Threat Detection System

## Problem Statement
Detect critical incidents in crowd scenarios and generate official reports for authorities.

**Detects 5 threat classes:**
| Class | Alert Level | Action |
|---|---|---|
| 🚑 Ambulance Stuck | HIGH | Clear path immediately |
| ⚔️ Fighting | HIGH | Deploy crowd control |
| 🔥 Fire | CRITICAL | Evacuate, call fire brigade |
| 💨 Smoke | HIGH | Investigate immediately |
| 🔫 Weapon | CRITICAL | Alert security forces |

---

## Features
- **Real-time Detection** — Image, Video, Live webcam feed
- **Instant Alerts** — Color-coded CRITICAL/HIGH alert panel
- **Auto Incident Log** — All detections saved to `incident_log.docx`
- **RAG Reporting** — Query incidents using AI, zero hallucination
- **Official Reports** — Generate summaries for authorities

---

## Tech Stack
```
Detection  : YOLOv8m (Ultralytics)
Training   : Google Colab T4 GPU
Dataset    : 4547 images, 5 classes
UI         : Gradio
Embeddings : sentence-transformers/all-MiniLM-L6-v2
Vector DB  : FAISS
LLM        : google/flan-t5-base
RAG        : LangChain RetrievalQA
Log        : python-docx (.docx)
```

---

## Model Performance (V1 — 100 epochs)
```
Class        AP50    Status
ambulance    0.875   ✅ Excellent
fighting     0.983   ✅ Perfect
fire         0.274   ⚠️ Improving
smoke        0.135   ⚠️ Improving
weapon       0.749   ✅ Good
mAP50        0.603   ✅ Good
```

---

## Deployment — Hugging Face

### Files needed:
```
app.py              ← main application
requirements.txt    ← dependencies
best.pt             ← trained YOLOv8m weights (upload after training)
```

### Steps:
```
1. huggingface.co → New Space → SDK: Gradio
2. Upload: app.py + requirements.txt + best.pt
3. Space builds automatically (~3 mins)
4. Share URL with officials
```

---

## How to Use

### Image Detection
```
1. Click "IMAGE" tab
2. Upload crowd image
3. Click "ANALYZE THREAT"
4. View detections + alerts
5. Download incident_log.docx
```

### Video Detection
```
1. Click "VIDEO" tab
2. Upload crowd video
3. Click "ANALYZE THREAT"
4. View annotated video + alerts
5. Download incident_log.docx
```

### Live Feed
```
1. Click "LIVE FEED" tab
2. Allow camera access
3. Model detects in real-time
4. Alerts shown instantly
```

### RAG Incident Report
```
1. Run detections first (populates log)
2. Click "INCIDENT REPORT" tab
3. Click "LOAD INCIDENT LOG"
4. Ask questions in chat:
   - "Summarize all incidents for officials"
   - "How many CRITICAL alerts?"
   - "Generate official report"
5. Answers from log only — no hallucination
```

---

## Project Flow
```
Colab Training          Hugging Face App
──────────────          ────────────────
YOLOv8m trains    →     Image/Video/Live Detection
Saves best.pt     →     Upload to Space
                        ↓
                        Detections → Alert Panel
                        ↓
                        Auto-save → incident_log.docx
                        ↓
                        RAG queries → Official Reports
                        ↓
                        Share with Authorities
```

---

## RAG System
```
incident_log.docx
      ↓
Docx2txtLoader → load text
      ↓
RecursiveCharacterTextSplitter → 500 char chunks
      ↓
all-MiniLM-L6-v2 → embeddings
      ↓
FAISS → vector store
      ↓
Your question → similarity search → top 4 chunks
      ↓
Flan-T5-base → answer from chunks only
      ↓
Zero hallucination answer ✅
```
