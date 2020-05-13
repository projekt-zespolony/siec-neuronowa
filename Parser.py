import xml.etree.ElementTree as ET


class XmlTrainingDataParser:

    def __init__(self):
        self.data = [[]]
        self.labels = [[]]

    def parse(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()

        for m in root:
            for elem in m:
                if elem.tag == 'values':
                    self.data += elem.attrib.values()
                else:
                    self.labels += elem.attrib.values()

    # Call only once you called parse first
    def acquire_data(self):
        return self.data, self.labels
