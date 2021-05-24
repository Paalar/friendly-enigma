import sys
from importlib import reload
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def print_header(title):
    print(f"""

        --- {title} ---

    """)

def generate_fake_data(noise_level = 0):
    sys.argv = [sys.argv[0]]
    sys.argv.append(str(noise_level))
    fakeDataModule = sys.modules.get("data.fake.generateFakeData")
    explanationsModule = sys.modules.get("data.fake.generateOneHotExplanationsFromFake")
    reload(fakeDataModule)
    reload(explanationsModule)    
    sys.argv = [sys.argv[0]]

def get_auroc_of_model(runner):
    return runner.model.metrics["test"][0].compute()['AUROC/head-0/test']

class AUROC_Table:
    columns = ["Model", "AUROC"]

    def __init__(self, title):
        self.header = title
        self.data = []

    def append(self, key, data):
        self.data.append([key, "{:.2f}".format(data)])

    def print(self):
        print("""
        
        """)
        print("-------------------")
        print("")
        print(f"\t-- {self.header} --")
        print("")
        format_row = "{:>12}" * (len(self.columns)+1)
        print(format_row.format("", *self.columns))
        for row in self.data:
            print(format_row.format("", *row))
        print("")
        print("-------------------")
        print("""
        
        """)
