import torch
import rationale_net.models.encoder as encoder
import rationale_net.models.generator as generator
import rationale_net.models.tagger as tagger
import rationale_net.models.empty as empty
import rationale_net.utils.learn as learn
import os
import pdb

def get_model(args, embeddings, train_data):
    if args.snapshot is None:
        if args.use_as_tagger == True:
            gen = empty.Empty()
            model = tagger.Tagger(embeddings, args)
        else:
            gen   = generator.Generator(embeddings, args)
            model = encoder.Encoder(embeddings, args)
    else :
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
            gen_path = learn.get_gen_path(args.snapshot)
            if args.cuda:
                if os.path.exists(gen_path):
                    gen   = torch.load(gen_path)
                model = torch.load(args.snapshot)
            else:
                if os.path.exists(gen_path):
                    gen   = torch.load(gen_path, map_location=lambda storage, loc: storage)
                    gen.args.cuda = "false"
                model = torch.load(args.snapshot, map_location=lambda storage, loc: storage)
                model.args.cuda = "false"
        except :
            print("Sorry, This snapshot doesn't exist.")

    if args.num_gpus > 1:
        model = nn.DataParallel(model,
                                    device_ids=range(args.num_gpus))

        if not gen is None:
            gen = nn.DataParallel(gen,
                                    device_ids=range(args.num_gpus))
    return gen, model
