
# ResumeSorter
### NLP-Driven Resume Classification pipeline

This project applies Machine Learning and Natural Language Processing to automate the categorization of resumes. The goal is to develop a script that classifies resumes based on their content, streamlining the review process. We employ BERT, an advanced language model with strong contextual understanding, to train on resume data. This allows BERT to identify key features for various job categories, offering a more efficient and accurate method for sorting resumes and improving recruitment processes.


## Running the Script
To run the Python script follow these steps:

### Step 1: Clone the GitHub Repository 
* Clone the repository to your local machine using Git.
* Open command prompt and run the following command:

```bash
  git clone https://github.com/hrafid/ResumeSorter.git

```

### Step 2: Navigate to the Cloned Directory 
```bash
  cd ResumeSorter
```

### Step 3: Downloading Additional File
* This script requires an additional file to execute.
* Download the file from this [link](www,google.com) and place it in the same directory where the cloned files are located.
* Afterwards your directory should look something like this:

```bash
ResumeSorter/
│
├── script.py
├── requirements.txt
├── additional_file.ext   # Additional file 

```
### Step 4: Set Up a Virtual Environment (Optional)
* Creating a venv with conda and activating it
```bash
  conda create -n resumeNlp python==3.8.0
  conda activate resumeNlp
```
### Step 5: Install dependencies
* Installing the packages listed in **'requirements.txt'** file
```bash
  pip install -r requirements.txt
```

### Step 6: Run the script
*After installing all dependencies and ensuring the additional file is in place, you can execute the script by running the following command:
```bash
  python script.py --file_path <path/to/pdfs> 
```
> Replace **'<path/to/pdfs>'** with the folder directory containing the resume pdfs. 

### What to Expect After Running the Script
* The script will read through the resume PDFs and categorize them based on the content of the resumes.
* Category folders within the same directory will be created based on the domain of resume.
* Resumes will be moved into their respective catagory folders

