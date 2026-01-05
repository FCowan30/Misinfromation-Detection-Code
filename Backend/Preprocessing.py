import pandas as pd

def Load_training_data():
    # load dataset
    df = pd.read_csv("data/fake.csv")
    df2 = pd.read_csv("data/True.csv")
    
    print("Dataset loaded successfuylly.")
    print("Shape of Fake news dataset: ", df.shape)
    print("Shape of True news dataset: ", df2.shape)

# Testing the function directly from file.
if __name__ == "__main__":
    Load_training_data()