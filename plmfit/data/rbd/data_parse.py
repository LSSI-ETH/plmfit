import pandas as pd


if __name__=='__main__':
    
    # Imports the dataset
    data = pd.read_csv('full_data.csv')
    
    # For now these are the only columns of interest
    data = data[["aa_seq","len","mouse","random"]]
    
    # Dropping duplciates just in case there are any
    data.drop_duplicates(subset="aa_seq", keep="first", inplace=True)
    
    # Renaming the mouse column to label
    data.rename(columns = {"mouse":"label"},inplace = True)
    
    # Export to csv
    data.to_csv("rbd_data_full.csv",index = False)