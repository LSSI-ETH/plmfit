from plmfit.shared_utils.utils import init_plm


def extract_embeddings(args, logger):
    model = init_plm(args.plm, logger)
    assert model != None, 'Model is not initialized'

    model.extract_embeddings(data_type=args.data_type, layer=args.layer,
                            reduction=args.reduction)