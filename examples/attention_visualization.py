from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import re

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, token_a, token_b, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.token_a = token_a
        self.token_b = token_b
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputFeature`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        token_a = []
        token_b = []
        input_type_ids = []
        token_a.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            token_a.append(token)
            input_type_ids.append(0)
        token_a.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                token_b.append(token)
                input_type_ids.append(1)
            token_b.append("[SEP]")
            input_type_ids.append(1)

        tokens = token_a + token_b
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                token_a=token_a,
                token_b=token_b,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples


def get_attentions(tokens_a, tokens_b, attn):
    """Compute representation of the attention to pass to the d3 visualization
    Args:
      tokens_a: tokens in sentence A
      tokens_b: tokens in sentence B
      attn: numpy array, attention
          [num_layers, batch_size, num_heads, seq_len, seq_len]
    Returns:
      Dictionary of attention representations with the structure:
      {
        'all': Representations for showing all attentions at the same time. (source = AB, target = AB)
        'a': Sentence A self-attention (source = A, target = A)
        'b': Sentence B self-attention (source = B, target = B)
        'ab': Sentence A -> Sentence B attention (source = A, target = B)
        'ba': Sentence B -> Sentence A attention (source = B, target = A)
      }
      and each sub-dictionary has structure:
      {
        'att': list of inter attentions matrices, one for each layer. Each is of shape [num_heads, source_seq_len, target_seq_len]
        'top_text': list of source tokens, to be displayed on the left of the vis
        'bot_text': list of target tokens, to be displayed on the right of the vis
      }
    """

    all_attns = []
    a_attns = []
    b_attns = []
    ab_attns = []
    ba_attns = []
    slice_a = slice(0, len(tokens_a)) # Positions corresponding to sentence A in input
    slice_b = slice(len(tokens_a), len(tokens_a) + len(tokens_b)) # Position corresponding to sentence B in input
    num_layers = len(attn)
    for layer in range(num_layers):
        layer_attn = attn[layer][0] # Get layer attention (assume batch size = 1), shape = [num_heads, seq_len, seq_len]
        all_attns.append([[[round(float(k), 3) for k in j] for j in i] for i in layer_attn.tolist()]) # Append AB->AB attention for layer, across all heads
        a_attns.append([[[round(float(k), 3) for k in j] for j in i] for i in layer_attn[:, slice_a, slice_a].tolist()]) # Append A->A attention for layer, across all heads
        b_attns.append([[[round(float(k), 3) for k in j] for j in i] for i in layer_attn[:, slice_b, slice_b].tolist()]) # Append B->B attention for layer, across all heads
        ab_attns.append([[[round(float(k), 3) for k in j] for j in i] for i in layer_attn[:, slice_a, slice_b].tolist()]) # Append A->B attention for layer, across all heads
        ba_attns.append([[[round(float(k), 3) for k in j] for j in i] for i in layer_attn[:, slice_b, slice_a].tolist()]) # Append B->A attention for layer, across all heads

    attentions =  {
        'all': {
            'att': all_attns,
            'top_text': tokens_a + tokens_b,
            'bot_text': tokens_a + tokens_b
        },
        'a': {
            'att': a_attns,
            'top_text': tokens_a,
            'bot_text': tokens_a
        },
        'b': {
            'att': b_attns,
            'top_text': tokens_b,
            'bot_text': tokens_b
        },
        'ab': {
            'att': ab_attns,
            'top_text': tokens_a,
            'bot_text': tokens_b
        },
        'ba': {
            'att': ba_attns,
            'top_text': tokens_b,
            'bot_text': tokens_a
        }
    }

    return attentions


def mean_attention(attn):
    mean_attn = attn[len(attn)]


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file", default="./examples/test.txt", type=str)
    parser.add_argument("--output_file", default="./examples/output.txt", type=str)

    ## Other parameters
    parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help = "local_rank for distributed training on gpus")
    parser.add_argument("--cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")

    args = parser.parse_args()

    if args.local_rank == -1 or not args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

    layer_indexes = [int(x) for x in args.layers.split(",")]

    tokenizer = BertTokenizer.from_pretrained('./multi_cased_L-12_H-768_A-12', do_lower_case=False)

    examples = read_examples(args.input_file)

    features = convert_examples_to_features(
        examples=examples, seq_length=args.max_seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model = BertModel.from_pretrained('./multi_cased_L-12_H-768_A-12')
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_type_ids = torch.tensor([f.input_type_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_token_a = [f.token_a for f in features]
    all_token_b = [f.token_b for f in features]
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_type_ids, all_input_mask, all_example_index)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    model.eval()
    i = 0
    with open(args.output_file, "w", encoding='utf-8') as writer:
        for input_ids, input_type_ids, input_mask, example_indices in eval_dataloader:
            input_ids = input_ids.to(device)
            input_type_ids = input_type_ids.to(device)
            input_mask = input_mask.to(device)

            _, _, attn = model(input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
            attn_tensor = torch.stack([attn_data['attn_probs'] for attn_data in attn])
            attn = attn_tensor.data.numpy()
            attentions = get_attentions(all_token_a[i], all_token_b[i], attn)
            i += 1


if __name__ == "__main__":
    main()
