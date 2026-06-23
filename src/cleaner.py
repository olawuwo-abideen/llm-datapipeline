import re


class Cleaner:

    def clean(self, text):

        text = re.sub(r"http\S+", "", text)

        text = re.sub(r"<.*?>", "", text)

        text = text.lower()

        text = text.strip()

        return text