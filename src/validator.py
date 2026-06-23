class Validator:

    def validate(self, df):

        if "text" not in df.columns:
            raise Exception("Text column missing")

        return True