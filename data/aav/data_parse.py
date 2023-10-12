import pandas as pd


if __name__=='__main__':
    data= pd.read_csv('full_data.csv' , dtype={'one_vs_many_split_validation': float}) # solves DtypeWarning: Columns have mixed types. Specify dtype option on import or set low_memory=False in Pandas
    #data= pd.DataFrame(columns = ["aa_seq","ed" ,"score" ])
    data['one_vs_many_split_validation'] = data['one_vs_many_split_validation'].astype(float)
    data.rename(columns = {"full_aa_sequence" : "aa_seq"}, inplace = True)
    #data.drop(columns=['mutation_mask' , 'mutated_region'])
    data.drop_duplicates(subset = 'aa_seq' , keep='first', inplace = True)
    data = data[~data['aa_seq'].str.contains('\*')]
    data['len'] = data['aa_seq'].apply(lambda x : len(x))
    data.to_csv("aav_data_full.csv")       
