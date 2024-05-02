from plmfit.shared_utils import utils, data_explore

def developing(args, logger):
    # Adapters development
    model = utils.init_plm(args.plm, logger)
    assert model != None, 'Model is not initialized'

    print(model)