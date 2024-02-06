from plmfit.models.pretrained_models import *
from plmfit.models.downstream_heads import LinearRegression
from plmfit.models.fine_tuning import *
import argparse
import plmfit.logger as l

parser = argparse.ArgumentParser(
    description='Embeddings extraction', fromfile_prefix_chars='@')
parser.add_argument('--layer', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args()

if __name__ == '__main__':
    logger = l.Logger("./train_logger")
    fine_tuner = FullRetrainFineTuner(
        epochs=5, lr=0.0006, batch_size=8,  val_split=0.2, log_interval=1)
    model = ProGenFamily(progen_model_name='progen2-small')
    logger.log('Model loaded')
    # head = LinearRegression(32, 1)
    # model.concat_task_specific_head(head)
    # model.fine_tune('meltome', fine_tuner, 'set', 'adam', 'mse')

    model.extract_embeddings('gb1' , batch_size = args.batch_size , layer = args.layer )
