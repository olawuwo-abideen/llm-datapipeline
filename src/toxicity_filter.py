class ToxicityFilter:

    banned_words = ["hate", "kill", "racist"]

    def filter_text(self, text):

        for word in self.banned_words:

            if word in text:

                return "[FLAGGED TOXIC CONTENT]"

        return text