from Scripts.webScraping import WebScraper, GetURLS
from Scripts.utils import write_text, read_text, write_csv, read_csv, write_to_text
from Scripts.preprocessText import PreprocessText
from Scripts.clasifier import NerClassifier
from Scripts.createDataset import CreateDataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer
import torch
import numpy as np

agents = [
    'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.83 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36'
    ]

urls_file = 'furniture stores pages.csv'
product_names_output = 'product_names.txt'
product_names_preprocessed = 'product_names_preprocessed.txt'
urls_from_extracted_product_names = 'urls_from_extracted_product_names.csv'
meta_content_output = 'meta_content.txt'
meta_content_output_preprocessed = 'meta_content_preprocessed.txt'
test_output = "text_output.txt"

label_list = ['O','B-PROD','I-PROD']

if __name__ == '__main__':
    urls = read_csv(urls_file)

    # """First step: Parsing the csv file and scraping for product names
    #  in h1, h2 and h3"""
    count = 0
    for url in urls:
        scraper = WebScraper(url, agents)
        product_names_h1 = scraper.scrape(method = 'select', selector = 'h1[class*="product"]')
        product_names_h2 = scraper.scrape(method = 'select', selector = 'h2[class*="product"]')
        product_names_h3 = scraper.scrape(method = 'select', selector = 'h3[class*="product"]')

        if scraper._soup:
            count = count + 1

        if product_names_h1 or product_names_h2 or product_names_h3:
            product_names = product_names_h1 + product_names_h2 + product_names_h3

        if product_names:
            print(len(product_names), " extracted from url:", scraper.url)
            write_text(product_names_output, product_names, 'a')
            write_csv(urls_from_extracted_product_names, scraper.url, 'a')
        else:
            print("The url not scraped: ", scraper.url)

    print("Nr of scrapped urls: ", count)

    # # """Second step: Preprocess the extracted product names to eliminate 
    # # accent chars, irrelevant chars, punctuation, extra whitespaces and extra line"""
    text = read_text(product_names_output)
    preprocessor = PreprocessText()
    preprocessed_product_names = preprocessor.preprocess(text)
    write_text(product_names_preprocessed, preprocessed_product_names)

    # # """3rd step: Parsing the csv file containing urls where products were found
    # # and scraping for meta content"""
    urls_products = read_csv(urls_from_extracted_product_names)
    count = 0
    for url in urls_products:
        scraper = WebScraper(url, agents)
        meta_content = scraper.scrape(method = 'find_all', get = 'content', tag = 'meta')

        if meta_content:
            print("Meta extracted from url:", scraper.url)
            write_text(meta_content_output, meta_content, 'a')
        else:
            print("The url not scraped: ", scraper.url)

    # # """4th step: Preprocess the extracted meta data to eliminate 
    # # accent chars, irrelevant chars, punctuation, extra whitespaces and extra line"""
    text = read_text(meta_content_output)
    preprocessor = PreprocessText()
    preprocessed_meta_data = preprocessor.preprocess(text)
    write_text(meta_content_output_preprocessed, preprocessed_meta_data)

    # """5th step: Label data and create dataset
    # In this step were defined two methods for label the data:
    #             1. By using Lavenshtein
    #             2. By using nlp from spacy
    # Was concluded that nlp from spacy don't find all the labels"""
    preprocessed_meta_data = read_text("meta_content_preprocessed.txt")
    preprocessed_product_names = read_text("product_names_preprocessed_final.txt")
    dataset = CreateDataset(preprocessed_meta_data)
    dataset.label_data_using_lavenshtein(preprocessed_product_names)
    print("Nr. of labels detected using Lavenshtein method: ", dataset.nr_of_labeled_entities(), " from ", len(preprocessed_product_names), " names")
    dataset.label_data_using_nlp(preprocessed_product_names)
    print("Nr. of labels detected using nlp from spacy method: ", dataset.nr_of_labeled_entities(), " from ", len(preprocessed_product_names), " names")
    dataset = CreateDataset(preprocessed_meta_data)
    dataset.label_data_using_lavenshtein(preprocessed_product_names)
    print("Nr. of sentences containing products: ", dataset.get_nr_of_products())
    print("Nr. of sentences withouth products: ", dataset.get_nr_of_non_products())

    sentences_length = dataset.sentences_length()
    print("Mean sentences length from database: ", np.mean(sentences_length))
    print("Std of sentences length from database: ", np.std(sentences_length))

    dataset.classes_balance_plot()
    dataset.sentence_distribution_histogram()

    dataset.split_data_balanced()
    dataset.create_datasets_for_trainig()
    write_to_text(dataset.train_dataset, "./train-data/train_dataset.txt")
    write_to_text(dataset.test_dataset, "./test-data/test_dataset.txt")

    """6th step: Train the data with the extracted dataset"""
    clasifier = NerClassifier('./train-data/', './test-data/', label_list, model='bert-base-cased', train_epochs = 10)
    clasifier.tokenize_data()
    clasifier.train()
    clasifier.evaluate()
    print(clasifier.eval_results)
    clasifier.save_model(f'train-model-bert.model')

    # """7th step: Predict new products"""
    url = "https://www.bordbar.de/en/shop/configuration/door/404?gclid=Cj0KCQiAgqGrBhDtARIsAM5s0_n5YPqa4MJVyZORIMK0rFfHeaEMW2rt1TUYZ47qZ6elmT86eSINQfsaAnxlEALw_wcB"
    urls = GetURLS(url, agents)
    urls.scrape_recursive()

    content = []
    for url in urls.urls:  
        meta_content = urls.scrape(method = 'find_all', get = 'content', tag = 'meta')
        if meta_content:
            print("Meta extracted from url:", url)
            content.extend(meta_content)
        else:
            print("The url not scraped: ", url)

    processed_content = PreprocessText()
    content = processed_content.preprocess(content)

    products_list = []
    for token in content:
        predicted_labels = clasifier.predict_new_text(token)
        products_list.append(clasifier.from_predict_to_products(predicted_labels))