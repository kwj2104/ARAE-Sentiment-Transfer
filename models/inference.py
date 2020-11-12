import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Normal
import argparse
import random
from models.helpers import to_gpu, Corpus, batchify, Dictionary

#import helpers
import json
import os
import numpy as np
from models.models_final import Seq2Seq2Decoder
import nltk

#parser = argparse.ArgumentParser(description='ARAE for Yelp transfer')
# Path Arguments

#parser.add_argument('--data_path', type=str, default='data/', required=False,
#                    help='location of the data corpus')
DATA_PATH = 'data/'
#parser.add_argument('--device_id', type=str, default='0')
DEVICE_ID='0'
#parser.add_argument('--seed', type=int, default=1111,
#                    help='random seed')
SEED = 1111
#parser.add_argument('--outf', type=str, default='yelp_example',
#                    help='output directory name')

OUTF='yelp_example'
#parser.add_argument('--load_vocab', type=str, default="",
#                    help='path to load vocabulary from')

LOAD_VOCAB = ""

# Data Processing Arguments
#parser.add_argument('--vocab_size', type=int, default=30000,
#                    help='cut vocabulary down to this size '
#                         '(most frequently seen words in train)')

VOCAB_SIZE = 30000

#parser.add_argument('--maxlen', type=int, default=25,
#                    help='maximum sentence length')
MAXLEN=25

#parser.add_argument('--lowercase', dest='lowercase', action='store_true',
#                    help='lowercase all text')
LOWERCASE = True
#parser.add_argument('--no-lowercase', dest='lowercase', action='store_true',
#                    help='not lowercase all text')
#parser.set_defaults(lowercase=True)
#
#parser.add_argument('--batch_size', type=int, default=64, metavar='N',
#                    help='batch size')
#
##Hyperparams for autoencoder
## Model Arguments
#parser.add_argument('--emsize', type=int, default=128,
#                    help='size of word embeddings')
#parser.add_argument('--nhidden', type=int, default=128,
#                    help='number of hidden units per layer')
#parser.add_argument('--nlayers', type=int, default=1,
#                    help='number of layers')
#parser.add_argument('--noise_r', type=float, default=0.1,
#                    help='stdev of noise for autoencoder (regularizer)')
#parser.add_argument('--hidden_init', action='store_true',
#                    help="initialize decoder hidden state with encoder's")
#parser.add_argument('--dropout', type=float, default=0.0,
#                    help='dropout applied to layers (0 = no dropout)')
#
#parser.add_argument('--cuda', dest='cuda', action='store_true',
#                    help='use CUDA')
#parser.set_defaults(cuda=False)
#
#args = parser.parse_args()

NO_LOWERCASE=False
BATCH_SIZE=64
EMSIZE=128
NHIDDEN=128
NLAYERS=1
NOISE_R=0.1
HIDDEN_INIT=False
DROPOUT=0
CUDA=False

os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID

# Set the random seed manually for reproducibility.
#random.seed(111)
#np.random.seed(args.seed)
#torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not CUDA:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")
    else:
        pass
        #torch.cuda.manual_seed(args.seed)

def sent_inference(source_sent, vocab=None, state_dict="models/trained_models/autoencoder_model_50.pt", encoder_no=1):
    #Load vocab

    if vocab == None:
        json_file = open('models/yelp_example/vocab.json')
        json_str = json_file.read()
        vocab = json.loads(json_str)

    vocabdict = Dictionary(vocab)
    ntokens = len(vocabdict.word2idx)

    #Create Model
    autoencoder = Seq2Seq2Decoder(emsize=EMSIZE,
                      nhidden=NHIDDEN,
                      ntokens=ntokens,
                      nlayers=NLAYERS,
                      noise_r=NOISE_R,
                      hidden_init=HIDDEN_INIT,
                      dropout=DROPOUT,
                      gpu=CUDA)
    autoencoder.load_state_dict(torch.load(state_dict, map_location=lambda storage, loc: storage))

    autoencoder.eval()

    #Tokenize and clean input, turn into Tensor
    sent_tok = nltk.word_tokenize(source_sent)
    sent_tok_lc = [word.lower() for word in sent_tok]
    sent_pos = nltk.pos_tag(sent_tok_lc)

    for i, word in enumerate(sent_tok_lc):
        if sent_pos[i][1] == 'CD':
            sent_tok_lc[i] = '_num_'

    source_sent = [1]
    for i, word in enumerate(sent_tok_lc):
        if word in vocabdict.word2idx:
            source_sent.append(vocabdict.word2idx[word])
        else:
            source_sent.append(3) #unk

    source_tensor = torch.Tensor(source_sent).unsqueeze(0).long()
    source_length = [len(source_sent)]


    #Inference
    encoded = autoencoder.encode(source_tensor, source_length, noise=False)
    decoded, all_vals = autoencoder.generate(encoder_no, encoded, maxlen=50)

    #Print Decoded Sentence
    cont = True
    confidence = []
    word_count = 0
    decoded_sent = []
    for i, idx in enumerate(decoded[0]):
        if cont:
            decoded_sent.append((vocabdict.idx2word[idx.item()], all_vals[i]))
            word_count += 1
        if idx.item() == 2:
            cont = False
    return decoded_sent


#json_file = open('yelp_example/vocab.json')
#json_str = json_file.read()
#vocabdict = json.loads(json_str)
#datafiles = [(os.path.join(args.data_path, "valid1.txt"), "valid1", True)]
#corpus = Corpus(datafiles,
#                maxlen=args.maxlen,
#                vocab_size=args.vocab_size,
#                lowercase=args.lowercase,
#                vocab=vocabdict)

#source = "the people who ordered off the menu did n’t seem to do much better"
#
##print("SOURCE:")
#print(source)
#
#encoder_no = 1
#
#print(inference(source, state_dict="trained_models/autoencoder_model_25.pt", encoder_no=encoder_no))
#inference(source, state_dict="trained_models/autoencoder_model_50.pt", encoder_no=encoder_no)
#inference(source, state_dict="trained_models/autoencoder_model_lambda10_50.pt", encoder_no=encoder_no)

#eval_batch_size = 1
#test1_data = batchify(corpus.data['valid2'], eval_batch_size, shuffle=False)
#print("Loaded data!")
#
#
#### Inference ----------------------------------------
#n = random.randint(0, 25000)
#source, target, lengths = test1_data[n]
