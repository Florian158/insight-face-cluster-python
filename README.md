# README

## General Description

This is a small example how to use [InsightFace](https://insightface.ai/) with python.
It runs a local Face Recognition Model and detects all faces on a image.
After that all faces and their embeddings are stored and clustered with the DBScan Algorithm.
The clustered images are stored in a new folder and the first image is duplicated and the clustered person is marked with a rectangle.

## How to run

### Activate virtual python env

```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

### run python script

```bash
python .\insightFace.py
```

## Prerequisites

sure not complete but some are:
C++ build tool e.g. [here](https://visualstudio.microsoft.com/de/visual-cpp-build-tools/)  
Windows 10/11 SDK

```bash
pip install insightface onnxruntime opencv-python numpy scikit-learn setuptools wheel scipy
```
