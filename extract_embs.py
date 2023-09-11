import torch
import torch.utils.data as data_utils
import logger as l
import time 
import torch.cuda
import psutil
import argparse
import utils

    
parser = argparse.ArgumentParser(description='embeddings')
parser.add_argument('--model_name', type=str, default='progen2-small') ## options ['progen2-small', 'progen2-xlarge', 'progen2-oas', 'progen2-medium', 'progen2-base', 'progen2-BFD90' , 'progen2-large']
parser.add_argument('--gpus', type=int , default=0) 
parser.add_argument('--gres', type=str , default='gpumem:24g') 
parser.add_argument('--mempercpu', type=int , default= 0)
parser.add_argument('--nodes', type=int , default= 1)

parser.add_argument('--batch_size', type=int , default= 4)
parser.add_argument('--layer', type=int , default= 12)
parser.add_argument('--data_type', type=str , default= 'aav')
parser.add_argument('--reduction', type=str , default= 'mean')
parser.add_argument('--emb_dim', type=int , default= 1024)

args = parser.parse_args()
logger = l.Logger(f'logger_extract_embeddings_{args.data_type}_{args.model_name}_layer{args.layer}_{args.reduction}.txt')
device = 'cpu'
fp16 = False
if torch.cuda.is_available():
    device = "cuda:0"
    fp16 = True
    logger.log(f' Running on {torch.cuda.get_device_properties(0).name}')

else:
    logger.log(f' No gpu found rolling device back to {device}')



def extract_embeddings(model, model_name, aa_seq, data_type, tokenizer , max_len , layer = 12, reduction = 'mean'):
    
    logger.log(f' Encoding {len(aa_seq)} sequences....')
    seq_tokens = [ torch.tensor(tokenizer.encode(i).ids) for i in aa_seq ] 
    for i in range(len(seq_tokens)):
        pad = tokenizer.get_vocab()['<|pad|>'] * torch.ones(max_len - seq_tokens[i].shape[0], dtype = int)
        seq_tokens[i] = torch.cat((seq_tokens[i], pad), dim = 0)
    seq_tokens = torch.stack(seq_tokens)
    
    seq_dataset = data_utils.TensorDataset( seq_tokens)
    
    seq_loader =  data_utils.DataLoader(seq_dataset, batch_size=args.batch_size, shuffle=False)
    logger.log(f' Calculating embeddings {len(seq_dataset)} sequences....')
    embs = torch.zeros((len(seq_dataset), args.emb_dim))
    i = 0
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled= fp16):
            for batch in seq_loader:
                start = time.time()
                out = model(batch[0]).hidden_states
                memory_usage = psutil.virtual_memory()
                
                embs[i : i+args.batch_size, : ] = torch.mean(out[layer - 1] , dim = 1)
                del out
                i = i + args.batch_size
                logger.log(f' {i} / {len(seq_dataset)} | {time.time() - start:.2f}s | memory usage : {100 - memory_usage.percent:.2f}%')
       


    torch.save(embs,f'./data/{data_type}/embeddings/{data_type}_{model_name}_embs_layer{layer}_{reduction}.pt')
    t = torch.load(f'./data/{data_type}/embeddings/{data_type}_{model_name}_embs_layer{layer}_{reduction}.pt')
    logger.log(f'Saved embeddings ({t.shape[1]}-d) as "{data_type}_{model_name}_embs_layer{layer}_{reduction}.pt" ')
    return embs
 
if __name__=='__main__':



    model = utils.load_model(args.model_name)
    data = utils.load_dataset(args.data_type , 'splits/low_vs_high')
    data['len'] = data['sequence'].apply(lambda x : len(x))
    max_len = max(data['len'].values)
    tokenizer = utils.load_tokenizer(args.model_name)
    
    extract_embeddings(model, args.model_name, data['sequence'], args.data_type, tokenizer, max_len, args.layer , args.reduction)
    
