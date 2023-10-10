from models.pretrained_models import *
from models.models import LinearRegression

model = ProGenPLM()

head = LinearRegression(32 , 1)
#odel.concat_task_specific_head(head)

model.fine_tune('aav' , 'full_retrain' , epochs = 5 , lr = 0.001 , optimizer = 'adam' , batch_size = 2, train_split_name = 'two_vs_many_split', val_split = 0.2 , loss_f = 'mse' , log_interval = 1)

#model.extract_embeddings('aav' , batch_size = 1 , layer = 12 )
