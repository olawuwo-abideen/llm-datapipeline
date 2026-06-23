class Deduplicator:

    def remove_duplicates(self, df):

        df = df.drop_duplicates()

        return df