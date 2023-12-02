from sklearn.model_selection import train_test_split
import Levenshtein
import spacy
import numpy as np
import random
import matplotlib.pyplot as plt

class CreateDataset:
    def __init__(self, data, labels = None):
        self.data = data
        self.labels = labels
        self.train_tokens = None
        self.test_tokens = None
        self.train_labels = None
        self.test_labels = None
        self.train_dataset = None
        self.test_dataset = None

    def __len__(self):
        return len(self.data)
    
    def train_len(self):
        return len(self.train_dataset)
    
    def test_len(self):
        return len(self.test_dataset)

    def __getitem__(self, index):
        return self.data[index]
    
    def get_nr_of_products(self):
        return  sum(token.count('B-PROD') for token in self.labels)
    
    def get_nr_of_non_products(self):
        return sum('B-PROD' not in token for token in self.labels)
    
    def sentences_length(self):
        return [len(sentence.split()) for sentence in self.data]
    
    def sentence_distribution_histogram(self):
        sentence_lengths = self.sentences_length()
        plt.hist(sentence_lengths, bins=range(min(sentence_lengths), max(sentence_lengths) + 1, 1), edgecolor='black')
        plt.xlabel('Sentence Length')
        plt.ylabel('Frequency')
        plt.title('Histogram of Sentence Lengths')

        mean_length = np.mean(sentence_lengths)
        std_length = np.std(sentence_lengths)
        annotation_text = f"Mean: {mean_length:.2f}\nStd: {std_length:.2f}"
        plt.annotate(annotation_text, xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top')

        plt.show()

    def classes_balance_plot(self):
        categories = ['Products', 'Non-Products']
        counts = [self.get_nr_of_products(), self.get_nr_of_non_products()]

        plt.bar(categories, counts, color=['blue', 'green'])
        plt.xlabel('Categories')
        plt.ylabel('Number of Samples')
        plt.title('Number of Products vs. Number of Non-Products')
        plt.show()

    
    def label_data_using_lavenshtein(self, target_group, similarity_threshold=0.8):
        self.labels = [['O' for j in range(len(self.data[i].split()))] for i in range(len(self.data))]
        for nsample, sample_data in enumerate(self.data):
            sample_data = sample_data.lower().split()
            for prod in target_group:
                prod = prod.lower().split()
                for element_idx in range(len(sample_data) - len(prod) + 1):
                    subset = sample_data[element_idx:element_idx+len(prod)]
                    subset_str = ' '.join(subset)
                    similarity = Levenshtein.ratio(subset_str, ' '.join(prod))
                    if similarity >= similarity_threshold:
                        self.labels[nsample][element_idx] = 'B-PROD'
                        for index in range(1, len(prod)):
                            if sample_data[element_idx + index] in prod:
                                self.labels[nsample][element_idx + index] = 'I-PROD'
                    element_idx = element_idx + len(prod)
        return self.data, self.labels
    
    def label_data_using_nlp(self, target_group):
        self.labels = [['O' for j in range(len(self.data[i]))] for i in range(len(self.data))]

        nlp = spacy.load("en_core_web_sm")
        for nsample, sample_data in enumerate(self.data):
            doc = nlp(sample_data)
            
            for ent in doc.ents:
                if ent.text in target_group:
                    start = ent.start
                    end = ent.end

                    self.labels[nsample][start] = "B-PROD"

                    for i in range(start + 1, end):
                        self.labels[nsample][i] = "I-PROD"

        return self.data, self.labels
    
    def nr_of_labeled_entities(self, entitie = 'B-PROD'):
        return sum(row.count(entitie) for row in self.labels)
    
    def get_indices_by_label(self, target_label = 'B-PROD'):
        return np.asarray([idx for idx, label in enumerate(self.labels) if target_label in label]).astype(int)
    
    def shuffle_data(self, l1, l2):
        combined_lists = list(zip(l1, l2))
        random.shuffle(combined_lists)
        
        shuffled_l1, shuffled_l2 = zip(*combined_lists)
        return list(shuffled_l1), list(shuffled_l2)
    
    def data_to_dataset(self, text, tags):
        dataset_final = []
        for elem in range(len(text)):
            dataset_final.extend(list(zip(text[elem], tags[elem])))
            dataset_final.append(('\n',''))
        return dataset_final[:-1]

    def split_data_balanced(self, split_percent = 0.7):
        data = [item for item in self.data]
        product_data_idx = self.get_indices_by_label()
        non_product_data_idx = np.asarray([idx for idx in range(len(self.data)) if idx not in product_data_idx]).astype(int)

        product_data = [data[product_data_idx[i]] for i in range(len(product_data_idx))]
        non_product_data = [data[non_product_data_idx[i]] for i in range(len(non_product_data_idx))]
        product_labels = [self.labels[product_data_idx[i]] for i in range(len(product_data_idx))]
        non_product_labels = [self.labels[non_product_data_idx[i]] for i in range(len(non_product_data_idx))]

        train_product_nr = round(split_percent * len(product_data))
        train_non_product_nr = round(split_percent * len(non_product_data))

        product_data, product_labels = self.shuffle_data(product_data, product_labels)
        non_product_data, non_product_labels = self.shuffle_data(non_product_data, non_product_labels)

        train_data = product_data[:train_product_nr] + non_product_data[:train_non_product_nr]
        test_data = product_data[train_product_nr:] + non_product_data[train_non_product_nr:]
        train_labels = product_labels[:train_product_nr] + non_product_labels[:train_non_product_nr]
        test_labels = product_labels[train_product_nr:] + non_product_labels[train_non_product_nr:]

        self.train_tokens, self.train_labels = self.shuffle_data(train_data, train_labels)
        self.test_tokens, self.test_labels = self.shuffle_data(test_data, test_labels)
        

    def split_data_using_sklearn(self, test_size=0.3, random_state=None):
        self.train_tokens, self.test_tokens, self.train_labels, self.test_labels = train_test_split(self.data, self.labels, test_size=test_size, random_state=random_state)

    def create_datasets_for_trainig(self):
        train_tokens = [item.split() for item in self.train_tokens]
        test_tokens = [item.split() for item in self.test_tokens]
        self.train_dataset = self.data_to_dataset(train_tokens, self.train_labels)
        self.test_dataset = self.data_to_dataset(test_tokens, self.test_labels)


