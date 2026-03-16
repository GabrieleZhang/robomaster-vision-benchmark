# RoboMaster Vision Benchmark
Comparative Study of Modern Object Detection Architectures for Real-Time Armor Detection

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)](https://pytorch.org/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2024-blueviolet.svg)](https://docs.openvino.ai/)

Benchmark and analysis of modern object detection architectures under 'real robotic deployment constraints'.

</div>

---

# TL;DR

We benchmark three modern object detection models for RoboMaster armor detection:

- YOLOv11
- YOLO26
- RT-DETR

Key observations from our experiments:

• All models achieve ~98% mAP with sufficient training data  
• GPU inference: YOLOv11 provides the highest throughput (~126 FPS)  
• CPU inference: all S-sized models achieve ~13 FPS  
• Model size appears to be a dominant factor affecting CPU performance  

Next step:

Nano-sized models + quantization to achieve '200+ FPS CPU inference' for real deployment.

---

# Overview

This project investigates the performance characteristics of several modern object detection architectures for 'real-time robotic perception in RoboMaster systems'.

Unlike typical academic benchmarks focusing primarily on accuracy, this project prioritizes 'practical deployment constraints', including:

- CPU inference latency
- inference stability (latency jitter)
- model efficiency
- deployment complexity

The goal is to identify architectures that are suitable for 'real-time onboard deployment' in robotic platforms.

---

# Architecture Comparison

## YOLOv11

Type: CNN-based detector  
Characteristics:

- Anchor-free detection
- Mature ecosystem
- Highly optimized GPU kernels

Advantages:

- strong real-time performance
- stable latency

Limitations:

- requires NMS post-processing

Best suited for:

GPU-based real-time systems.

---

## YOLO26

Type: CNN-based end-to-end detector  

Characteristics:

- NMS-free architecture
- end-to-end training pipeline

Advantages:

- simplified deployment
- potentially better CPU efficiency

Limitations:

- relatively new architecture
- ecosystem still evolving

Best suited for:

edge and CPU deployment scenarios.

---

## RT-DETR

Type: Transformer-based detector

Characteristics:

- end-to-end detection
- no NMS required
- global attention modeling

Advantages:

- strong performance on small objects
- simplified post-processing

Limitations:

- higher parameter count
- slower inference

Best suited for:

server-side or GPU-heavy inference scenarios.

---

# Dataset

Dataset used in this study:

- Size: ~30,000 images
- Task: armor plate detection
- Classes: 20 armor types
- Format: COCO + YOLO annotations

Split:

- 80% training
- 20% validation

---

# Experimental Setup

## Hardware

GPU:
NVIDIA RTX 4090 (24GB)

CPU:
Intel Core Ultra 9

## Framework

Training:

PyTorch 2.1

CPU inference:

OpenVINO 2024

## Input configuration

Image size:

640 × 640

Batch size:

1

---

# Accuracy Results

All models converge to similar accuracy levels after training.

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|------|------|------|------|------|
| YOLOv11-S | 98.65% | 91.71% | 96.16% | 98.08% |
| YOLO26-S | 98.39% | 90.86% | 96.03% | 98.16% |
| RT-DETR-L | 98.37% | 91.25% | 95.93% | 98.03% |

Observation:

With sufficient training data, the accuracy difference between architectures becomes relatively small (<0.3%).

This suggests that 'efficiency and deployability may become more important than marginal accuracy gains' for this task.

---

# GPU Benchmark

Environment:

GPU: NVIDIA RTX 4090  
Framework: PyTorch + CUDA  

| Model | FPS | Latency | Latency Jitter | Parameters |
|------|------|------|------|------|
| YOLOv11-S | 126.1 | 7.93 ms | ±0.28 ms | 9.4M |
| YOLO26-S | 108.8 | 9.20 ms | ±0.69 ms | 9.5M |
| RT-DETR-L | 32.8 | 30.50 ms | ±2.60 ms | 30.1M |

Observation:

YOLOv11 shows the lowest latency jitter in our benchmark environment.

---

# CPU Benchmark

Environment:

CPU: Intel Core Ultra  
Framework: OpenVINO  
Precision: FP32

| Model | FPS | Latency | Jitter | GPU Slowdown |
|------|------|------|------|------|
| YOLOv11-S | 13.22 | 75.6 ms | ±14.8 ms | 9.5× |
| YOLO26-S | 13.19 | 75.8 ms | ±17.1 ms | 8.2× |
| RT-DETR-L | 3.17 | 315.9 ms | ±56.4 ms | 10.3× |

Observation:

CPU inference shows roughly '8–10× slowdown compared to GPU inference' for S-sized models.

---

# Key Insight

A significant performance gap was observed when comparing with our team's existing deployment baseline:

Team baseline:

YOLOX-Nano (~0.9M parameters) → 200+ FPS CPU inference

Our tested models:

YOLO-S (~9M parameters) → ~13 FPS CPU inference

This suggests that 'model size plays a major role in CPU deployment performance'.

---

# Optimization Roadmap

Based on current findings:

