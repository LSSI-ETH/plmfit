import ft_frameworks
import utils
from Bio import SeqIO
import torch
import extract_embs
from transformers import pipeline

model = utils.load_model('progen2-small')
tokenizer = utils.load_tokenizer('progen2-small')
wt = ''                
sequences = utils.read_fasta('/Users/tbikias/workspace/plmfit/data/aav/P03135.fasta')
for sequence_id, sequence_data in sequences.items():
     print(f"Sequence ID: {sequence_id}")
     wt = sequence_data
     
data = utils.load_dataset('aav' , 'splits/low_vs_high')
data['len'] = data['sequence'].apply(lambda x : len(x))
max_len = max(data['len'].values)
#eq_tokens = utils.tokenize


#target = torch.tensor(tokenizer.encode(data['sequence'].values.tolist()))
print(f'')
enc = extract_embs.extract_embeddings(model,'progen2-small', data['sequence'], 'aav', tokenizer, max_len)

pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer)                 
#d = pipe('ASDASDASDASDASDASDA')
     
input_ids = torch.tensor(tokenizer.encode(wt).ids).view([1, -1])

output = model(input_ids)

lgt = output.logits
pkeys = output.past_key_values
hs = output.hidden_states
attns = output.attentions
