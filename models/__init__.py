import importlib
import os
import pickle

from models.wav2vec2_modified import WAV2VEC2_MODIFIED

def get_model(args):
    # if args.trainer == "classification_audio":
    #     # if os.path.exists(os.path.join(os.getcwd(), 'wav2vec_model.pickle')):
    #     #     with open('wav2vec_model.pickle', 'rb') as file:
    #     #         model = pickle.load(file)
    #     # else:
    #     #     model = WAV2VEC2_MODIFIED.from_pretrained("kresnik/wav2vec2-large-xlsr-korean", config=args.config)
    #     #     model.args = args
    #     #     with open('wav2vec_model.pickle', 'wb') as file:
    #     #         pickle.dump(model, file, pickle.HIGHEST_PROTOCOL)

    #     model = WAV2VEC2_MODIFIED.from_pretrained("kresnik/wav2vec2-large-xlsr-korean", config=args.config)
    #     model.args = args

    #     model.freeze_feature_extractor()

    #     return model

    model_module = importlib.import_module("models." + args.model)
    model = getattr(model_module, args.model.upper())

    return model