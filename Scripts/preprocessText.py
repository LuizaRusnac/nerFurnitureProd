import re
import unicodedata

class PreprocessText:
    def remove_url(self, text): 
        return [re.sub(r'https?://\S+|www\.\S+', '', txt) for txt in text if txt]

    def remove_accented_chars(self, text):
        return [unicodedata.normalize('NFKD', txt).encode('ascii', 'ignore').decode('utf-8', 'ignore') for txt in text]

    def remove_irr_char(self, text):
        return [re.sub(r'[^a-zA-Z1-9]', ' ', txt) for txt in text] 

    def remove_punctuation(self, text):
        return [re.sub(r'[^a-zA-Z0-9]', ' ', txt) for txt in text]

    def remove_extra_whitespaces(self, text):
        return [re.sub(r'^\s*|\s\s*', ' ', txt).strip() for txt in text]

    def word_count(self, text):
        return len(text.split())

    def remove_extra_lines(self, text):
        return [txt for txt in text if txt!='']

    def separate_text(self, text):
        return [re.sub(r'(?<!^)([A-Z])', r' \1', s) for s in text]

    def conctenate_upper_words(self, text):
        return [re.sub(r'([A-Z]+)', r' \1 ', s) for s in text]
    
    def keep_unique_strings(self, text):
        return list(dict.fromkeys(text))

    def preprocess(self, text):
        if text:
            preprocessed_text = self.remove_url(text)
            preprocessed_text = self.remove_accented_chars(preprocessed_text)
            preprocessed_text = self.remove_irr_char(preprocessed_text)
            preprocessed_text = self.remove_punctuation(preprocessed_text)
            preprocessed_text = self.keep_unique_strings(preprocessed_text)
            preprocessed_text = self.remove_extra_whitespaces(preprocessed_text)
            preprocessed_text = self.remove_extra_lines(preprocessed_text)
        else: 
            ValueError("The text content is empty")
            return None
        return preprocessed_text

