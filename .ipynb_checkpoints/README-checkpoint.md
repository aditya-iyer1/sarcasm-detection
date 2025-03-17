# Sarcasm Detection Project - Setup Guide

Welcome to the Sarcasm Detection Project! This guide provides setup instructions for getting started with the project repository. Please follow these steps to set up your local environment and begin working.

---


## Getting Started

### 1. Clone the Repository

If you haven’t already, clone the repository to your local machine:

```zsh
git clone <repo-url>
cd <repo-name>
```

### 2. Set up Virtual Environment

```zsh
python -m venv venv  # Create virtual environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```zsh
pip install -r requirements.txt
```

### 4. Accessing the Data

The dataset files are stored in the /data folder of the repository.

Each record consists of three attributes:

`is_sarcastic`: 1 if the record is sarcastic, otherwise 0
`headline`: The headline of the news article
`article_link`: link to the original news article. Useful in collecting supplementary data



Use the following to import the json file:

```python
import json

def parse_data(file):
    for l in open(file, 'r'):
        yield json.loads(l)

data = list(parse_data('../data/Sarcasm_Headlines_Dataset_v2.json'))
```

