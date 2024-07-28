# Text classifier

Simple russian text classifier based on Navec - library of pretrained word embeddings for Russian language

## Installation
```bash
pip install -r requirements.txt
```

## Usage
First download emdeddings and put them to folder data:
```
wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar
```

For example lets use <a href="https://github.com/yutkin/Lenta.Ru-News-Dataset">dump of lenta.ru by @yutkin</a>. Download and put to folder data:
```bash
wget https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.0/lenta-ru-news.csv.gz
```

Set up classification parameters in config.py and run the code
```python
from source.create_dataset import download_news
from source.fit_model import fit_text_classifier_model

# Generate Dataset
download_news()

# Fit model
fit_text_classifier_model()
```

Classify text
```python
from source.classify_text import classify_text

# Classify text
classify_text("В Москве прошли соревнования по шахматам")

'Спорт'
```
