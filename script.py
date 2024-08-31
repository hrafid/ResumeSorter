#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import shutil
from PyPDF2 import PdfReader
import PyPDF2 
import argparse


### extracting script.py directory as _.pt file is also located there
script_dir = os.path.dirname(os.path.abspath(__file__)) 
model_path = os.path.join(script_dir, 'state_dict_model.pt')## saved _.pt file of best model

# defininglable mapping sequence (same as while trainin model)
label_map = {
    'AGRICULTURE': 0, 'SALES': 1, 'ACCOUNTANT': 2, 'AVIATION': 3, 'BANKING': 4, 'CONSULTANT': 5, 'FINANCE': 6, 'PUBLIC-RELATIONS': 7, 
    'BUSINESS-DEVELOPMENT': 8, 'CHEF': 9, 'AUTOMOBILE': 10, 'INFORMATION-TECHNOLOGY': 11, 'DIGITAL-MEDIA': 12, 'ENGINEERING': 13, 
    'ARTS': 14, 'HR': 15, 'APPAREL': 16, 'HEALTHCARE': 17, 'FITNESS': 18, 'CONSTRUCTION': 19, 'TEACHER': 20, 'ADVOCATE': 21, 'BPO': 22, 'DESIGNER': 23
}
reverse_label_map = {v: k for k, v in label_map.items()}

# 
### initializing and loading model from saved .pt file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=24)
model.to(device) 

### load model weights from file
model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
model.eval()

# this function extractss text data from pdf
def extract_text(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
        return text

### function for classifying resume category
def categorize_resume(resume_text):
    #inputs = tokenizer(resume_text, padding='max_length', truncation=True,return_tensors='pt')# appplying tokenizer to text
    #inputs = tokenizer(resume_text, return_tensors='pt', truncation=True, padding=True)
    inputs = tokenizer(resume_text, return_tensors='pt', truncation=True, padding='max_length')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()# model prediction on resume data
    category = reverse_label_map[predicted_label] # translating prediction into category 
    return category



# function for processing resumes

def process_resumes(input_directory):

    output_directory=os.path.join(input_directory, 'Categorized resume')

    categorized_resumes = []
    count=0
    for filename in os.listdir(input_directory):
        if filename.endswith(".pdf"): # check if a certain file is pdf, if  pdf than proceed
            file_path = os.path.join(input_directory, filename)
            resume_text = extract_text(file_path)# extracting resume text 
            category = categorize_resume(resume_text)# classifying resume

            category_folder = os.path.join(output_directory, category)
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)# creating  category folder if alredy not created

            shutil.move(file_path, os.path.join(category_folder, filename))
            categorized_resumes.append({"filename": filename,"category": category})
            count+=1
            
    if count!=0:
        df = pd.DataFrame(categorized_resumes)
        csv_path = os.path.join(output_directory, 'categorized_resumes.csv')
        df.to_csv(csv_path, index=False)
        print(f"Resumes are sorted at: {output_directory}")
        print(f"A CSV is generated at {output_directory}")

    else: 
        print("No pdf was found!")





if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Resume Categorization With NLP")
        parser.add_argument('--input_dir', type=str, help="Resume folder path")
        args = parser.parse_args()
        process_resumes(args.input_dir)
        
    except:
        print("Directory not found!")


# In[ ]:




