# SCENE-Net: White-box Model for 3D Point Cloud Segmentation

Welcome to the SCENE-Net repository! This repository contains the implementation of SCENE-Net, a white-box model leveraging Group Equivariant Non-Expansive Operators (GENEOs) for transparent segmentation in 3D point clouds.

## Overview

SCENE-Net is a novel approach designed to address segmentation challenges within extensive 3D point clouds. This repository includes:
- `/core`: Source code for SCENE-Net's implementation.
- `/data-sample`: TS40K dataset covering rural and forest terrains for training and evaluation.
- `/experiments`: Scripts for assessing SCENE-Net's performance against the SemanticKITTI benchmark.

## Requirements

- Python 3.x
- PyTorch
- Open3D
- NumPy
- Other necessary libraries specified in `requirements.txt`

## Usage

1. **Clone the repository:**
git clone https://github.com/dlavado/scene-net.git
cd scene-net



2. **Install dependencies:**
pip install -r requirements.txt
