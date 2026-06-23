import re


class PIIDetector:

    def anonymize(self, text):

        email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'

        phone_pattern = r'\d{11}'

        text = re.sub(email_pattern, "[EMAIL]", text)

        text = re.sub(phone_pattern, "[PHONE]", text)

        return text