import json


class Exporter:

    def save(self, data):

        with open("processed.json", "w") as f:

            json.dump(data, f, indent=4)