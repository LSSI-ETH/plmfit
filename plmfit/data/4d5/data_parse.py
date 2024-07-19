# parse wildtype data from wildtype.fasta that looks like that:
# >1N8Z_1|Chain A|Herceptin Fab (antibody) - light chain|Mus musculus (10090)
# DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC
# >1N8Z_2|Chain B|Herceptin Fab (antibody) - heavy chain|Mus musculus (10090)
# EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEP
# >1N8Z_3|Chain C|Receptor protein-tyrosine kinase erbB-2|Homo sapiens (9606)
# TQVCTGTDMKLRLPASPETHLDMLRHLYQGCQVVQGNLELTYLPTNASLSFLQDIQEVQGYVLIAHNQVRQVPLQRLRIVRGTQLFEDNYALAVLDNGDPLNNTTPVTGASPGGLRELQLRSLTEILKGGVLIQRNPQLCYQDTILWKDIFHKNNQLALTLIDTNRSRACHPCSPMCKGSRCWGESSEDCQSLTRTVCAGGCARCKGPLPTDCCHEQCAAGCTGPKHSDCLACLHFNHSGICELHCPALVTYNTDTFESMPNPEGRYTFGASCVTACPYNYLSTDVGSCTLVCPLHNQEVTAEDGTQRCEKCSKPCARVCYGLGMEHLREVRAVTSANIQEFAGCKKIFGSLAFLPESFDGDPASNTAPLQPEQLQVFETLEEITGYLYISAWPDSLPDLSVFQNLQVIRGRILHNGAYSLTLQGLGISWLGLRSLRELGSGLALIHHNTHLCFVHTVPWDQLFRNPHQALLHTANRPEDECVGEGLACHQLCARGHCWGPGPTQCVNCSQFLRGQECVEECRVLQGLPREYVNARHCLPCHPECQPQNGSVTCFGPEADQCVACAHYKDPPFCVARCPSGVKPDLSYMPIWKFPDEEGACQPCPIN

import pandas as pd
import plmfit.shared_utils.utils as utils
import os
import json
import numpy as np

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in



# File paths for the dataset and FASTA file
csv_path_positive = "4d5_high_4th_sort.csv"
csv_path_negative = "4d5_neg_4th_sort.csv"
fasta_path = "wildtype.fasta"

# Parse the FASTA file to extract sequence IDs and sequences
sequences = utils.read_fasta(
    os.path.join(script_dir, fasta_path)
)

cdrl3 = sequences[list(sequences.keys())[0]]
cdrh3 = sequences[list(sequences.keys())[1]]

mutated_region_cdrl3 = "QHYTTPPT"
mutated_region_cdrh3 = "WGGDGFYAM"

# Load dataset from CSV file
pos_data = pd.read_csv(
    os.path.join(script_dir, csv_path_positive)
) 

neg_data = pd.read_csv(
    os.path.join(script_dir, csv_path_negative)
)

# combine the positive and negative data and add a column 'binary_score' to indicate the class
pos_data['binary_score'] = 1
neg_data['binary_score'] = 0

data = pd.concat([pos_data, neg_data])

# function to compare two strings and return the number of differences
def compare_strings(s1, s2):
    return sum([1 for i in range(len(s1)) if s1[i] != s2[i]])

# calculate no_mut column based on mutated region for each chain, no mask is used
data['no_mut'] = data['AASeq'].apply(lambda x: compare_strings(mutated_region_cdrh3 + mutated_region_cdrl3, x))

# each entry has 17 aminoacids, split them into 9 and 8
data['cdr_h3'] = data['AASeq'].apply(lambda x: x[:9])
data['cdr_l3'] = data['AASeq'].apply(lambda x: x[9:])



# at position 98 of the wildtype cdrh3 at each entry prepend the wildtype cdrh3 and append from position 107 onwards
data['cdr_h3'] = data.apply(lambda x: cdrh3[:98] + x['cdr_h3'] + cdrh3[107:], axis=1)
data['cdr_l3'] = data.apply(lambda x: cdrl3[:89] + x['cdr_l3'] + cdrl3[97:], axis=1)

# add a column 'aa_seq' that concatenates cdr_h3 and cdr_l3
data['aa_seq'] = data['cdr_h3'] + data['cdr_l3']
wildtype = cdrh3 + cdrl3

# Calculate and add a new column for the length of each amino acid sequence
data["len"] = data["aa_seq"].apply(len)

# Creating a new DataFrame with the specified columns
new_data = pd.DataFrame(
    {
        "aa_seq": data["aa_seq"],
        "len": data["len"],
        "no_mut": data["no_mut"],
        "binary_score": data["binary_score"],
        "cdr_h3": data["cdr_h3"],
        "cdr_l3": data["cdr_l3"]
    }
)

new_data.drop_duplicates(subset="aa_seq", keep="first", inplace=True)

# Save the new DataFrame to a CSV file
new_data.to_csv(os.path.join(script_dir, "4d5_data_full.csv"), index=False)


