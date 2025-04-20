# QAnvas: Q & A Chatbot for Canvas

## Overview

The proliferation of online learning platforms like Canvas has revolutionized education by providing seamless access to course materials, assignments, and communication tools for students and instructors. However, the vast amount of data on these platforms—encompassing course schedules, announcements, assignments, discussions, grades, and resources—often overwhelms users, making it difficult to efficiently locate relevant information or obtain personalized support. This information overload results in wasted time, missed deadlines, and decreased productivity, presenting a critical business challenge for educational institutions and learners alike.

This project proposes the development of an advanced **LLM-powered chatbot** integrated with Canvas to address these challenges. The chatbot will harness **natural language processing (NLP)** to enable intuitive queries (e.g., _"What are my upcoming assignments?"_ or _"What did my instructor announce today?"_), delivering real-time, accurate responses. By streamlining information retrieval and enhancing user experience, the solution will improve time management and boost engagement within the Canvas ecosystem. This initiative aims to resolve the business problem of inefficiency in online learning platforms, offering a **scalable, user-friendly tool** that showcases the practical application of **language processing technologies** in education.

## Sample Response
```bash
You: can i check my exam schedule? i am FT 2024 AIS06 Student. Also can i check which slide i can refer for transformer?


Question: can i check my exam schedule? i am FT 2024 AIS06 Student. Also can i check which slide i can refer for transformer?
{'history': [SystemMessage(content='', additional_kwargs={}, response_metadata={})]}

Debug: Raw agent response:
 {"action": "Get Timetable Or Exam Schedule", "action_input": {"full_time": true, "intake_year": 2024, "course": "AIS06"}}
{"action": "Retrieve Lecture Slides", "action_input": {"query": "transformer", "subjects": null}}
Tool Results.

{'Get Timetable Or Exam Schedule': 'Course: Machine Reasoning\nCourse Manager: Gary\t\t\nDate\tDay\tTime\n19 Aug 2024\tMonday\t9:00am - 5:30pm\n20 Aug 2024\tTuesday\t9:00am - 5:30pm\n22 Aug 2024\tThursday\t9:00am - 5:30pm\n23 Aug 2024\tFriday\t9:00am - 5:30pm\nCourse: Reasoning Systems\nCourse Manager: Gary\t\t\nDate\tDay\tTime\n26 Aug 2024\tMonday\t9:00am - 5:30pm\n27 Aug 2024\tTuesday\t9:00am - 5:30pm\n28 Aug 2024\tWednesday\t9:00am - 5:30pm\n29 Aug 2024\tThursday\t9:00am - 5:30pm\n30 Aug 2024\tFriday\t9:00am - 5:30pm\nCourse: Cognitive Systems\nCourse Manager: Zhenzhen\t\t\nDate\tDay\tTime\n2 Sep 2024\tMonday\t9:00am - 5:30pm\n3 Sep 2024\tTuesday\t9:00am - 5:30pm\n4 Sep 2024\tWednesday\t9:00am - 5:30pm\n\nCourse: Problem Solving Using Pattern Recognition\nCourse Manager: Fangming\t\t\nDate\tDay\tTime\n9 Sep 2024\tMonday\t9:00am - 5:00pm\n10 Sep 2024\tTuesday\t9:00am - 5:00pm\n11 Sep 2024\tWednesday\t9:00am - 5:00pm\n12 Sep 2024\tThursday\t9:00am - 5:00pm\n13 Sep 2024\tFriday\t9:00am - 5:00pm\nCourse: Pattern Recognition and Machine Learning Systems\nCourse Manager: Fangming\t\t\nDate\tDay\tTime\n16 Sep 2024\tMonday\t9:00am - 5:00pm\n17 Sep 2024\tTuesday\t9:00am - 5:00pm\n18 Sep 2024\tWednesday\t9:00am - 5:00pm\n19 Sep 2024\tThursday\t9:00am - 5:00pm\n20 Sep 2024\tFriday\t9:00am - 5:00pm\nCourse: Intelligent Sensing and Sense Making\nCourse Manager: Tian Jing\t\t\nDate\tDay\tTime\n23 Sep 2024\tMonday\t9:00am - 5:00pm\n24 Sep 2024\tTuesday\t9:00am - 5:00pm\n25 Sep 2024\tWednesday\t9:00am - 5:00pm\n26 Sep 2024\tThursday\t9:00am - 5:00pm\n\nExam Date\tExam Time\tGraduate Certificate(s)\n4 Nov, Monday\t6:30pm to 10:00pm\tEBA5001 Analytics Project Management\n\t\tISY5001 Intelligent Reasoning Systems\n\t\tSWE5001 Architecting Scalable Systems\n5 Nov, Tuesday\t6:30pm to 10:00pm\tEBA5002 Business Analytics Practice\n\t\tISY5002 Pattern Recognition Systems\n\t\tSWE5003 Engineering Big Data\n6 Nov, Wednesday\t6:30pm to 10:00pm\tEBA5004 Practical Language Processing\n\t\tSWE5004 Architecting Smart Systems\n7 Nov, Thursday\t6:30pm to 10:00pm\tEBA5005 Specialized Predictive Modelling and Forecasting\n\t\tEBA5008 Intelligent Financial Risk Management\n\t\tISY5003 Intelligent Robotic Systems\n\t\tSWE5005 Securing Ubiquitous Systems\n\nCourse: Robotic Systems\nCourse Manager: Nicholas\t\t\nDate\tDay\tTime\n6 Jan 2025\tMonday\t9:00am - 5:00pm\n7 Jan 2025\tTuesday\t9:00am - 5:00pm\n8 Jan 2025\tWednesday\t9:00am - 5:00pm\n13 Jan 2025\tMonday\t9:00am - 5:00pm\n14 Jan 2025\tTuesday\t9:00am - 5:00pm\nCourse: Autonomous Robots & Vehicles\nCourse Manager: Nicholas\t\t\nDate\tDay\tTime\n16 Jan 2025\tThursday\t9:00am - 5:00pm\n17 Jan 2025\tFriday\t9:00am - 5:00pm\n20 Jan 2025\tMonday\t9:00am - 5:00pm\n21 Jan 2025\tTuesday\t9:00am - 5:00pm\n22 Jan 2025\tWednesday\t9:00am - 5:00pm\nCourse: Human-Robot System Engineering\nCourse Manager: Xavier\t\t\nDate\tDay\tTime\n3 Feb 2025\tMonday\t9:00am - 5:00pm\n4 Feb 2025\tTuesday\t9:00am - 5:00pm\n5 Feb 2025\tWednesday\t9:00am - 5:00pm\n6 Feb 2025\tThursday\t9:00am - 5:00pm\n\nCourse: Vision Systems\nCourse Manager: Tian Jing\t\t\nDate\tDay\tTime\n6 Jan 2025\tMonday\t9:00am - 5:00pm\n7 Jan 2025\tTuesday\t9:00am - 5:00pm\n8 Jan 2025\tWednesday\t9:00am - 5:00pm\n9 Jan 2025\tThursday\t9:00am - 5:00pm\n10 Jan 2025\tFriday\t9:00am - 5:00pm\nCourse: Spatial Reasoning from Sensor Data\nCourse Manager: Tian Jing\t\t\nDate\tDay\tTime\n13 Jan 2025\tMonday\t9:00am - 5:00pm\n14 Jan 2025\tTuesday\t9:00am - 5:00pm\n15 Jan 2025\tWednesday\t9:00am - 5:00pm\nCourse: Real time Audio-Visual Sensing and Sense Making\nCourse Manager: Tian Jing\t\t\nDate\tDay\tTime\n20 Jan 2025\tMonday\t9:00am - 5:00pm\n21 Jan 2025\tTuesday\t9:00am - 5:00pm\n22 Jan 2025\tWednesday\t9:00am - 5:00pm\n23 Jan 2025\tThursday\t9:00am - 5:00pm\n\nCourse: Text Analytics\nCourse Manager: Zhenzhen\t\t\nDate\tDay\tTime\n10 Feb 2025\tMonday\t9:00am - 5:00pm\n11 Feb 2025\tTuesday\t9:00am - 5:00pm\n12 Feb 2025\tWednesday\t9:00am - 5:00pm\nCourse: New Media and Sentiment Mining\nCourse Manager: Aobo\t\t\nDate\tDay\tTime\n17 Feb 2025\tMonday\t9:00am - 5:00pm\n18 Feb 2025\tTuesday\t9:00am - 5:00pm\n19 Feb 2025\tWednesday\t9:00am - 5:00pm\nCourse: Text Processing Using Machine Learning\nCourse Manager: Zhenzhen\t\t\nDate\tDay\tTime\n24 Feb 2025\tMonday\t9:00am - 5:00pm\n25 Feb 2025\tTuesday\t9:00am - 5:00pm\n26 Feb 2025\tWednesday\t9:00am - 5:00pm\n27 Feb 2025\tThursday\t9:00am - 5:00pm\nCourse: Conversational Uls\nCourse Manager: Zhenzhen\t\t\nDate\tDay\tTime\n3 Mar 2025\tMonday\t9:00am - 5:00pm\n4 Mar 2025\tTuesday\t9:00am - 5:00pm\n5 Mar 2025\tWednesday\t9:00am - 5:00pm\n6 Mar 2025\tThursday\t9:00am - 5:00pm\n\nCourse: Advanced Machine Learning for Financial Services\nCourse Manager: Dave Ho\t\t\nDate\tDay\tTime\n10 Feb 2025\tMonday\t9:00am - 5:00pm\n11 Feb 2025\tTuesday\t9:00am - 5:00pm\n12 Feb 2025\tWednesday\t9:00am - 5:00pm\n15 Feb 2025\tSaturday\t9:00am - 5:00pm\nCourse: Explainable and Responsible AI for Finance\nCourse Manager: Tian Jing\t\t\nDate\tDay\tTime\n17 Feb 2025\tMonday\t9:00am - 5:00pm\n18 Feb 2025\tTuesday\t9:00am - 5:00pm\n22 Feb 2025\tSaturday\t9:00am - 5:00pm\nCourse: Alternative Data for Fintech Innovation\nCourse Manager: Scott\t\t\nDate\tDay\tTime\n24 Feb 2025\tMonday\t9:00am - 5:00pm\n25 Feb 2025\tTuesday\t9:00am - 5:00pm\n26 Feb 2025\tWednesday\t9:00am - 5:00pm\nCourse: Credit Risk Modelling and Analytics\nCourse Manager: Brian\t\t\nDate\tDay\tTime\n3 Mar 2025\tMonday\t9:00am - 5:00pm\n4 Mar 2025\tTuesday\t9:00am - 5:00pm\n6 Mar 2025\tThursday\t9:00am - 5:00pm\n7 Mar 2025\tFriday\t9:00am - 5:00pm\n\nExam Date\tExam Time\tGraduate Certificate (s)\n13 May, Tuesday\t6:30pm to 10:00pm\tEBA5003 Customer Analytics\n\t\tEBA5006 Big Data Analytics\n\t\tISY5004 Intelligent Sensing Systems\n\t\tSWE5003 Engineering Big Data\n14 May, Wednesday\t6:30pm to 10:00pm\tEBA5002 Business Analytics Practice\n\t\tISY5001 Intelligent Reasoning Systems\n\t\tSWE5001 Architecting Scalable Systems\n15 May, Thursday\t6:30pm to 10:00pm\tEBA5004 Practical Language Processing\n\t\tEBA5005 Specialized Predictive Modelling and Forecasting\n\t\tEBA5008 Intelligent Financial Risk Management\n\t\tSWE5004 Architecting Smart Systems\n16 May, Friday\t6:30pm to 10:00pm\tEBA5001 Analytics Project Management\n\t\tISY5003 Intelligent Robotic Systems\n\t\tSWE5005 Securing Ubiquitous Systems', 'Retrieve Lecture Slides': 'The term "transformer" in the provided excerpts refers to a type of model architecture used in the field of Natural Language Processing (NLP). \n\nThe Transformer model is a novel architecture that aims to solve sequence-to-sequence tasks while handling long-range dependencies with ease. It forms the foundation of the BERT and OpenAI GPT-2/3/4 algorithms (File: files\\EBA5004 Practical Language Processing [2420]\\course files\\02_TPML\\Day 2\\TPML_day_2.v4.pdf, Page: 27, Link: https://canvas.nus.edu.sg//files/TPML_day_2.v4.pdf). \n\nThe original use of the transformer was for machine translation, translating from one language to another (File: files\\EBA5004 Practical Language Processing [2420]\\course files'}

Agent: Yes, you can check your exam schedule. Here are the details:

For the year 2024, your exams are scheduled as follows:
- 4th November: EBA5001 Analytics Project Management, ISY5001 Intelligent Reasoning Systems, SWE5001 Architecting Scalable Systems
- 5th November: EBA5002 Business Analytics Practice, ISY5002 Pattern Recognition Systems, SWE5003 Engineering Big Data
- 6th November: EBA5004 Practical Language Processing, SWE5004 Architecting Smart Systems
- 7th November: EBA5005 Specialized Predictive Modelling and Forecasting, EBA5008 Intelligent Financial Risk Management, ISY5003 Intelligent Robotic Systems, SWE5005 Securing Ubiquitous Systems

For the year 2025, your exams are scheduled as follows:
- 13th May: EBA5003 Customer Analytics, EBA5006 Big Data Analytics, ISY5004 Intelligent Sensing Systems, SWE5003 Engineering Big Data
- 14th May: EBA5002 Business Analytics Practice, ISY5001 Intelligent Reasoning Systems, SWE5001 Architecting Scalable Systems
- 15th May: EBA5004 Practical Language Processing, EBA5005 Specialized Predictive Modelling and Forecasting, EBA5008 Intelligent Financial Risk Management, SWE5004 Architecting Smart Systems
- 16th May: EBA5001 Analytics Project Management, ISY5003 Intelligent Robotic Systems, SWE5005 Securing Ubiquitous Systems

Regarding your query about the transformer, it refers to a type of model architecture used in the field of Natural Language Processing (NLP). The Transformer model is a novel architecture that aims to solve sequence-to-sequence tasks while handling long-range dependencies with ease. It forms the foundation of the BERT and OpenAI GPT-2/3/4 algorithms. You can refer to the slides in the file "TPML_day_2.v4.pdf" under the course "EBA5004 Practical Language Processing". The specific details about the transformer model are on page 27 of this file. You can access the file [here](https://canvas.nus.edu.sg//files/TPML_day_2.v4.pdf).
```

---

## Installation Guide

### Important
Make sure have Microsoft Visual C++ Redistributable being installed.
[Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)

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
Download and install **Poppler Packaged for Windows** from [Poppler Packaged for Windows](https://github.com/oschwartz10612/poppler-windows) or use the ones in this repo. After installation, add the installation path to your system environment variables.

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

## System Information
### 1. Document Extraction Pipeline
#### 1.File Discovery & Loading
- Walks the files/ directory and picks up .pdf, .pptx, .doc, and .docx.
- For PDFs: uses PyPDFLoader to pull out text; for other formats, uses UnstructuredLoader.
- Table extraction (PDF only):
- Opens with pdfplumber, iterates each page’s tables, formats rows into tab‑delimited strings, and appends to the page’s page_content.
  
#### 2 .OCR Fallback
- If a page’s extracted text is under OCR_WORD_THRESHOLD (5 words):
- Convert page to image(s) via pdf2image’s convert_from_path.
- Run pytesseract.image_to_string on each image.
- Replace the sparse text with the OCR output.

#### 3.Text Cleaning & Metadata
- Strips HTML (strip_html), collapses whitespace, and redacts emails/phone numbers.
- Attaches metadata to each Document:
```bash
{
  "file_path": "...",
  "canvas_link": "https://canvas.nus.edu.sg/files/…"
}
```

#### 4. Chunking Non‑PDF Docs
- Runs CharacterTextSplitter (chunk_size=1000, overlap=200) on non‑PDF documents to keep embedding inputs within token limits.

### 2. Vector Store Construction
- Embedding Model all-MiniLM-L6-v2 via HuggingFaceEmbeddings
- Building with Chroma

### 3. Agent 
- Intent Extraction Agent - Runs the initial prompt in CanvasAgent.ask() to turn free‑form user text into a list of tool calls (intents + inputs).
- Document‑Retrieval Agent - Inside retrieve_lecture_slides_by_topic: synthesizes retrieved slide excerpts into a coherent “mini‑lecture” answer.
- Final‑Synthesis Agent - After all tools run, re‑invokes GPT‑4 to integrate every tool’s output into one clear, well‑structured response.




