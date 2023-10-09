from models.pretrained_models import *
from models.models import LinearRegression

model = ProgenSmall151M()

head = LinearRegression(1024 , 1)
#model.concat_task_specific_head(head)

model.fine_tune('aav' , 'full_retrain' , epochs = 5 , lr = 0.001 , optimizer = 'adam' , batch_size = 16 , train_split_name = 'two_vs_many_split', val_split = 0.2 , loss_f = 'mse' )
