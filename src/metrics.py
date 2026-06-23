class Metrics:

    def generate(self, before_df, after_df, pii_found=0, toxic_flagged=0):

        metrics = {}

        # dataset size
        metrics["before_size"] = len(before_df)
        metrics["after_size"] = len(after_df)

        # duplicates removed
        metrics["duplicates_removed"] = (
            metrics["before_size"] - metrics["after_size"]
        )

        # reduction percentage
        metrics["reduction_percent"] = (
            (metrics["before_size"] - metrics["after_size"])
            / metrics["before_size"]
        ) * 100

        # pii detected
        metrics["pii_found"] = pii_found

        # toxic content flagged
        metrics["toxic_flagged"] = toxic_flagged

        print("\n=== DATASET METRICS ===")
        print("Before size:", metrics["before_size"])
        print("After size:", metrics["after_size"])
        print("Duplicates removed:", metrics["duplicates_removed"])
        print("PII found:", metrics["pii_found"])
        print("Toxic content flagged:", metrics["toxic_flagged"])
        print("Reduction %:", round(metrics["reduction_percent"], 2))

        return metrics