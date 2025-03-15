# QAnvas: Q & A Chatbot for Canvas

## Overview

The proliferation of online learning platforms like Canvas has revolutionized education by providing seamless access to course materials, assignments, and communication tools for students and instructors. However, the vast amount of data on these platforms—encompassing course schedules, announcements, assignments, discussions, grades, and resources—often overwhelms users, making it difficult to efficiently locate relevant information or obtain personalized support. This information overload results in wasted time, missed deadlines, and decreased productivity, presenting a critical business challenge for educational institutions and learners alike.

This project proposes the development of an advanced **LLM-powered chatbot** integrated with Canvas to address these challenges. The chatbot will harness **natural language processing (NLP)** to enable intuitive queries (e.g., _"What are my upcoming assignments?"_ or _"What did my instructor announce today?"_), delivering real-time, accurate responses. By streamlining information retrieval and enhancing user experience, the solution will improve time management and boost engagement within the Canvas ecosystem. This initiative aims to resolve the business problem of inefficiency in online learning platforms, offering a **scalable, user-friendly tool** that showcases the practical application of **language processing technologies** in education.

---

## Installation Guide

### 1. Install Tesseract OCR
Download and install **Tesseract OCR** from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki). After installation, add the installation path to your system environment variables.

#### Steps:
- Download and install Tesseract OCR.
- Add the installation path to the system environment variable.

Example path (Windows):
```bash
C:\Program Files\Tesseract-OCR
```

### 2. Set Path for Poppler
Download and install **Tesseract OCR** from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki). After installation, add the installation path to your system environment variables.

#### Steps:
- Add the bin directory to your system environment variable.

Example path (Windows):
```bash
C:\path\to\poppler-24.08.0\Library\bin
```

### 3. Setup using Docker (Recommended) or Python

#### Option 1: Using Docker
- Ensure you have **Docker** installed.
- Use the provided `Dockerfile`  to set up the environment.

#### Option 2: Using Python
- Ensure **Python version ≤ 3.11.0** (due to `chroma-hnswlib` compatibility).
- Install dependencies using pip:
```bash
pip install -r requirements.txt
```

### 4. Canvas Key Access
Ensure you have access to the required Canvas Key to authenticate and use the system.

