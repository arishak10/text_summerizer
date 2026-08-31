# Text Summarizer

An NLP-based Text Summarization project that generates a concise summary from a given text using a Transformer-based deep learning model.

## Project Overview

This project is built using Natural Language Processing (NLP) and Deep Learning techniques to automatically summarize text into a shorter and meaningful version.

The project uses a T5-based Transformer model to understand the input text and generate a concise summary.

## Features

- Text summarization using Deep Learning
- Transformer-based NLP model
- T5 architecture
- Trained/evaluated using the SAMSum dataset
- Generates concise summaries from input text
- Interactive Streamlit interface

## Technologies Used

- Python
- Natural Language Processing (NLP)
- Hugging Face Transformers
- T5
- PyTorch
- Pandas
- NumPy
- Streamlit
- Git & GitHub

## Project Structure

```text
text_summarizer/
│
├── app.py
├── saved_summary_model/
│   └── model.safetensors
│
├── samsum-train.csv
├── samsum-validation.csv
├── requirements.txt
└── README.md

## Installation 
1. Clone the Repository

2.git clone
 ```bash
 https://github.com/arishak10/text_summarizer.git
```
3. Navigate to the Project Directory
```bash
cd text_summarizer
  ```
5. Create a Virtual Environment
 ```bash
python -m venv venv
 ```
7. Activate the Virtual Environment
Windows
 ```bash
venv\Scripts\activate
  ```
macOS / Linux
 ```bash
source venv/bin/activate
 ```
9. Install Required Dependencies
 ```bash
pip install -r requirements.txt
 ```
10. Run the Project
```bash 
python main.py
```


## How It Works

The application follows these steps:

User enters a text or conversation.
The input text is passed to the summarization model.
The Transformer model processes the input.
The model generates a concise summary.
The generated summary is displayed in the Streamlit interface.
# ~Example:
Input
John: Hey, are you coming to the meeting today?
Sarah: Yes, I will be there at 3 PM.
John: Great! Don't forget to bring the project report.
Sarah: Sure, I'll bring it with me.
Output
John and Sarah discuss attending a meeting at 3 PM and bringing the project report.


##  Learning & Acknowledgement

This project was developed as part of my learning during a course on Natural Language Processing and Deep Learning.

I built this project with guidance from the course instructor and by referring to the concepts, examples, and implementation approaches taught during the course. The project helped me understand practical concepts such as Transformer-based models, T5, tokenization, and text summarization.

This repository represents my learning and implementation of the concepts covered during the course.

## Author
Arisha Khan

GitHub:
https://github.com/arishak10
