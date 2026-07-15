import re
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class TextPreprocessor:
    def clean_text(self, text):
        text = str(text).lower()
        # Keep punctuation that indicates emotion.
        # FIX: apostrophe (') is now preserved. Previously it was stripped,
        # which silently broke keyword phrases like "don't understand" and
        # "doesn't make sense" in predict.py / bert_model.py -- those
        # keywords could never match because clean_text always turned them
        # into "don t understand" first. Only the no-apostrophe variants
        # ("dont understand") were ever able to fire.
        text = re.sub(r"[^a-zA-Z\s,!']", ' ', text)
        tokens = nltk.word_tokenize(text)

        # Keep ALL meaningful words, remove only basic articles
        skip_words = {'the', 'a', 'an'}
        tokens = [t for t in tokens if t not in skip_words and len(t) > 1]

        return ' '.join(tokens) if tokens else text