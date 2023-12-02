# NER For Furniture Products Recognition

## Overview
This project is dedicated to the development of a Named Entity Recognition (NER) system tailored specifically for identifying furniture products within textual data. Named Entity Recognition, a subset of Natural Language Processing (NLP), involves the identification and classification of named entities, such as objects, locations, and quantities, in unstructured text.

## Goal
The primary objective of this project is to construct a robust NER model capable of precisely recognizing and categorizing mentions of furniture products in various types of textual data, including product descriptions and other content related to furniture.

## Strenghts
- The project structured in a modular fashion for user-friendly interaction.
- Easy to scrape the data using the WebScraping class
- Posibilities of parsing all links from a website using the derivated class GetURLS
- Were extracted 569 product names from 562 functional websites
- Was extracted meta contain only from the websites where product names were found. It is possible that the other sites were having product names and we might confuse the classificator if the products are labeled as non-products.
- When annotating the data based on the extracted product names, a similarity threshold of 0.9 was considered for labeling the content.
- Easy to create dataset from text file using createDataset module
- Posibilitie to split the data with respect to unbalanced number of products vs non-products

## Results:
- Found 569 product names from 562 functional websites.
- Data: 497 product names were found from 569 scraped data and 3587 sentences containing text withouth product names

![Data balance](https://github.com/LuizaRusnac/nerFurnitureProd/blob/master/unbalanced_classes.png)

   Mean sentences length from database: 8.2242 `+-` 15.7733
![Data distribution](https://github.com/LuizaRusnac/nerFurnitureProd/blob/master/histogram_data_distibution.png)!

- Using **bert-base-case** model, for test set after 10 iterations: 
- precision: 0.6763
- recall: 0.6167
- f1: 0.6451
- Using **distil-bert-uncase** model, for test set after 10 iterations:
- precision: 0.5469
- recall: 0.4876
- f1: 0.5156

## Future work:
- The project extract names and content only from the provided urls. The reason is that certain websites featured products other than furniture (ex: www.factorybuys.com.au). It could impact the database by incorrectly labeling products other than furniture. For future work we could have a dictionary and scrap the data over the desired categories: chairs, sofas etc.
- A more rigorous training approach involves incorporating validation tests and halting the network when the accuracy on the validation set either decreases or remains stagnant over a specified number, N, of epochs.

## Steps
1. **Web Scraping for Product Names:**
   - Parsed a CSV file containing web pages and performed scraping for product names in h1, h2, and h3 tags.
   - Results were saved in 'product_name.txt'.

2. **Preprocessing of Web Scraping Results for Product Names:**
   - Processed the extracted product names to eliminate accent characters, irrelevant characters, punctuation, extra whitespaces, and extra lines.
   - The generated text file underwent visual analysis to remove elements unrelated to products.
   - Final results were saved in 'product_name_preprocessed_final.txt'.

3. **Web Scraping for Content:**
   - Parsed websites containing product names for another round of scraping to retrieve their content.
   - Final results were saved in 'meta_content.txt'.

4. **Preprocessing of Web Scraping Results for Content:**
   - Preprocessed the extracted meta-data to eliminate accent characters, irrelevant characters, punctuation, extra whitespaces, and extra lines.
   - Final results were saved in 'meta_content_preprocessed.txt'.

5. **Labeling Data and Creating Dataset:**
   - Defined two methods for labeling the data:
     1. Using Levenshtein distance.
     2. Using NLP from spaCy.
   - Concluded that spaCy's NLP does not capture all labels as it doesn't recognize entities in product names.
   - In creating dataset the unbalanced data was considered. Data was split in train and test with 70% of product names + 70% of samples withouth products in  train and 30% in test

6. **Training the Data:**
   - Utilized the extracted dataset to train the NER model.

7. **Evaluate the model:**
    - The model was evaluated on new website

## Project Structure

The project is organized into several modules, each containing classes related to specific functionalities.

### 1. `webScraping` Module

#### Class: `ProductScraper`

Responsible for parsing web pages and extracting html content.
Includes classes **WebScraper** along with the derivated class **GetURLS**

#### **WebScraper** class
Atributes:
- url: the url to parse
- agents: agents used to be send to a web server when making a request
- html_content: the fetched content from url
- __soup: the parsed html content

Methods:
- __init__(self, url, agents = None): constructor
- fetch_html(self): fetch content from url
- _get_soup(self): parse the html content
- scrape(self, method='find_all', get = 'text', *args, **kwargs): scrame the html content with the desired metods and tags

Example:
```python
from webScraping import WebScraper

scraper = WebScraper()
product_names_h1 = scraper.scrape(method = 'select', selector = 'h1[class*="product"]')
print(product_names_h1)
```

#### **GetURLS** class
This class is tasked with retrieving all website links from a given URL. It's a derived class form WebScraper class.

Atributes:
- domain: extracted domain from webpage
- urls: extracted urls from webpage domain

Methods:
- __init__(self, url, agents = None): constructor
- preprocess_url(self, referrer, url): preprocess the url
- scrape_recursive(self, url=None): scrape the url recursive to extract all sites

Example:
```python
from webScraping import GetURLS

url = "https://example.com"
urls_test = GetURLS(url, agents)
```
### 2. `preprocessText` Module

#### Class: `PreprocessText`

Preprocess the extracted text from url to eliminate accent chars, irrelevant chars, punctuation, extra whitespaces and extra lines
The class has no atributes.

Important methods:
- preprocess(self, text): eliminate accent chars, irrelevant chars,punctuation, extra whitespaces and extra lines

Example:
```python
from preprocessText import PreprocessText

preprocessor = PreprocessText()
preprocessed_data = preprocessor.preprocess(data)
print(preprocessed_data)
```

### 3. `createDataset` Module

#### Class: `CreateDataset`
This class is used to label the data be providing the product names and to create the dataset for the ner classifier

Important atributes:
- data: the input websites content 
- labels: the labels of the words in data
- train_dataset: the created train dataset ready for the classifier
- test_dataset: the created test dataset ready for the classifier

Important methods:
- __init__(self, data, labels = None): constructor
- label_data_using_levenshtein(self, target_group, similarity_threshold=0.8): label the dataset using levenshtein similarity.
    Example: 
    ```python
        text = ['Box Bed Blush Plush Velvet is best for you']
        product_name =  ['Box Bed Blush Plush Velvet']
        dataset = CreateDataset(text)
        print(dataset.label_data_using_lavenshtein(product_name))
    ```
    Outputs: ['B-PROD', 'B-PROD', 'I-PROD', 'I-PROD', 'I-PROD', 'O', 'O', 'O', 'O']

- label_data_using_nlp(self, target_group): label the dataset using npl found entities.
- split_data_balanced(self, split_percent = 0.7): split data with balanced class. The 70% of products and 70% of non products go in train data and the rest of 30% of products and non products go to test data.The data is randomized over founded products and after the split.
- split_data_using_sklear(self, test_size=0.3, random_state=None): split data randomly withouth consideration to unbalance dataset
- create_datasets_for_trainig(self): create the dataset ready for the classifier

Example:
```python
from createDataset import CreateDataset

dataset = CreateDataset(data)
dataset.label_data_using_lavenshtein(labels_to_be_found)
dataset.split_data_balanced()
dataset.create_datasets_for_trainig()
print(dataset.train_dataset)
```

### 4. `clasifier` Module

#### Class: `NerClasifier`
This class is used to classify the data

Important methods:
- def __init__(self, path_to_train_data, path_to_test_data, label_list, 
                model= "distilbert-base-uncased", batch_size=16, padding = "longest", task = "ner", evaluaton_strategy = "epoch", learning_rate = 1e-4, learning_decay = 1e-5,
                metric = "seqeval", train_epochs=10): Constructor to initialize the classifier with specified attributes
- clasifier.tokenize_data(self): tokenize the input data
- clasifier.train(self): method to train the ner using the tokenized data
- evaluate(self): method to evaluate the test set
- save_model(self): method to save the model after training
- predict_new_text(self, text): method used to predict new inputs

Example:
```python
from clasifier import NerCasifier

clasifier = NerClassifier('./path-train-data/', './path-test-data/', label_list, train_epochs = 10)
clasifier.tokenize_data()
clasifier.train()
clasifier.evaluate()
clasifier.save_model('/path-to-save/')
```

### 5. `utils` Module
Module containing util methods for writing and reading text and csv.