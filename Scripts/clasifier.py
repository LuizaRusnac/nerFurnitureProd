import os
import itertools
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer, RobertaTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, TrainerCallback
from transformers import DataCollatorForTokenClassification
import torch

class NerClassifier:
    def __init__(self, path_to_train_data, path_to_test_data, label_list, model = "bert-base-cased",
                  batch_size=16, padding = "longest", task = "ner",
                  evaluaton_strategy = "epoch", learning_rate = 1e-4, learning_decay = 1e-5,
                  metric = "seqeval", train_epochs=10):
        self.path_to_train_data = path_to_train_data
        self.path_to_test_data = path_to_test_data
        self.label_list = label_list
        self.model_checkpoint = model
        self.batch_size = batch_size
        self.padding = padding
        self.train_tokenized_datasets = None
        self.test_tokenized_datasets = None
        self.task = task
        self.trainer = None
        self.train_dataset = None
        self.test_dataset = None
        self.eval_results = None
        self.predicted_labels = None
        self.labels = None
        self.metric = load_metric(metric)
        self.args = TrainingArguments(
            f"test-{task}",
            evaluation_strategy = evaluaton_strategy,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=train_epochs,
            weight_decay=learning_decay,
        )
        self.label_encoding_dict = {label: index for index, label in enumerate(label_list)}
        self.model = AutoModelForTokenClassification.from_pretrained(model, num_labels=len(label_list))
        self.tokenizer = AutoTokenizer.from_pretrained(model, padding=padding)
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)

    def get_all_tokens_and_ner_tags(self, directory):
        return pd.concat([self.get_tokens_and_ner_tags(os.path.join(directory, filename)) for filename in os.listdir(directory)]).reset_index().drop('index', axis=1)
        
    def get_tokens_and_ner_tags(self, filename):
        with open(filename, 'r', encoding="utf8") as f:
            lines = f.readlines()
            split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]
            tokens = [[x.split('\t')[0] for x in y] for y in split_list]
            entities = [[x.split('\t')[1][:-1] for x in y] for y in split_list] 
        return pd.DataFrame({'tokens': tokens, 'ner_tags': entities})
  
    def get_un_token_dataset(self):
        train_df = self.get_all_tokens_and_ner_tags(self.path_to_train_data)
        test_df = self.get_all_tokens_and_ner_tags(self.path_to_test_data)
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

        return (train_dataset, test_dataset)

    def tokenize_and_align_labels(self, examples):
        label_all_tokens = True
        tokenized_inputs = self.tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"{self.task}_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif label[word_idx] == '0':
                    label_ids.append(0)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label_encoding_dict[label[word_idx]])
                else:
                    label_ids.append(self.label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
            
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def tokenize_data(self):
        self.train_dataset, self.test_dataset = self.get_un_token_dataset()
        self.train_tokenized_datasets= self.train_dataset.map(self.tokenize_and_align_labels, batched=True)
        self.test_tokenized_datasets = self.test_dataset.map(self.tokenize_and_align_labels, batched=True)

    def compute_metrics(self, pred):
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [[self.label_list[pred] for (pred, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[self.label_list[l] for (pred, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        self.eval_results = results
        self.predicted_labels = predictions
        self.labels = labels
        return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}
        
    def train(self):
        self.trainer = Trainer(
            self.model,
            self.args,
            train_dataset=self.train_tokenized_datasets,
            eval_dataset=self.test_tokenized_datasets,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            )
        self.trainer.train()

    def evaluate(self):
        self.trainer = Trainer(
            self.model,
            self.args,
            train_dataset=None,
            eval_dataset=self.test_tokenized_datasets,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            )
        self.model.eval()
        self.trainer.evaluate(eval_dataset = self.test_tokenized_datasets)

    def predict(self, text):
        self.trainer = Trainer(
            self.model,
            self.args,
            train_dataset=self.train_tokenized_datasets,
            eval_dataset=self.test_tokenized_datasets,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            )
        self.trainer.predict(text)

    def save_model(self, filepath):
        self.trainer = Trainer(
            self.model,
            self.args,
            train_dataset=self.train_tokenized_datasets,
            eval_dataset=self.test_tokenized_datasets,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            )
        self.trainer.save_model(filepath)

    def predict_new_text(self, text):
        tokens = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            raw_output = self.model(**tokens)
        predicted_labels = torch.argmax(raw_output.logits, dim=2)

        decoded_labels = [self.model.config.id2label[int(label_id)] for label_id in predicted_labels[0]]

        return list(zip(self.tokenizer.convert_ids_to_tokens(tokens['input_ids'][0]), decoded_labels))


    def save_metrics_callback(self, output_dir=".", file_prefix="metrics_epoch"):
        class SaveMetricsCallback(TrainerCallback):
            def on_epoch_end(self, args, state, control, model, tokenizer, **kwargs):
                # Save the metrics at the end of each epoch
                if state is not None and state.log_metrics is not None:
                    metrics = state.log_metrics
                    epoch_number = state.epoch
                    file_path = f"{output_dir}/{file_prefix}_{epoch_number}.csv"
                    with open(file_path, "w") as f:
                        f.write("%s\t%s" %(epoch_number, metrics))
                    print(f"Metrics saved to: {file_path}")

    def predicted_label(self, text, truncation=True, is_split_into_words=False, return_tensors='pt'):
        outputs = self.model(text["input_ids"])
        return outputs.logits.argmax(-1)
    
    def from_predict_to_products(self, predicted_labels):
        products = [(word, label) for word, label in predicted_labels if label in ('LABEL_0', 'LABEL_1')]
        list = [token for token, _ in products]
        final_list = []
        reconstructed_words = []
        for token in list:
            if token.startswith('##'):
                prev_word += token[2:]
                reconstructed_words.pop()
                reconstructed_words.append(prev_word)
            else:
                prev_word = token
                reconstructed_words.append(prev_word)
        final_list.append(" ".join(reconstructed_words))
        return final_list



    
