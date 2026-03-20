"""
Geneformer embedding extractor.

Usage:
  from geneformer import EmbExtractor
  embex = EmbExtractor(model_type="CellClassifier",
                       num_classes=3,
                       emb_mode="cell",
                       cell_emb_style="mean_pool",
                       filter_data={"cell_type":["cardiomyocyte"]},
                       max_ncells=1000,
                       max_ncells_to_plot=1000,
                       emb_layer=-1,
                       emb_label=["disease","cell_type"],
                       labels_to_plot=["disease","cell_type"],
                       forward_batch_size=100,
                       nproc=16,
                       summary_stat=None)
  embs = embex.extract_embs("path/to/model",
                            "path/to/input_data",
                            "path/to/output_directory",
                            "output_prefix")
  embex.plot_embs(embs=embs,
                  plot_style="heatmap",
                  output_directory="path/to/output_directory",
                  output_prefix="output_prefix")


Edited to return gene embedings as tenosrs

"""

# imports
import logging
import pickle
import re

import torch

from .utils import pad_tensor_list

logger = logging.getLogger(__name__)


def get_gf_repo():
    # site_packages_dirs = site.getsitepackages()

    # geneformer_repo_path = None

    # for directory in site_packages_dirs:
    #     potential_path = Path(directory) / 'geneformer'
    #     if potential_path.exists():
    #         geneformer_repo_path = potential_path
    #         break

    # if geneformer_repo_path is None:
    #     raise ValueError('Geneformer not found in site-packages directories')

    geneformer_repo_path = '/nfs/team361/mm58/Geneformer'

    return geneformer_repo_path


class EmbExtractor:
    valid_option_dict = {
        'model_type': {'Pretrained', 'GeneClassifier', 'CellClassifier'},
        'num_classes': {int},
        'emb_mode': {'cell', 'gene'},
        'cell_emb_style': {'mean_pool'},
        'filter_data': {None, dict},
        'max_ncells': {None, int},
        'emb_layer': {-1, 0},
        'emb_label': {None, list},
        'labels_to_plot': {None, list},
        'nproc': {int},
        'summary_stat': {None, 'mean', 'median'},
    }

    def __init__(
        self,
        model_type='Pretrained',
        num_classes=0,
        emb_mode='gene',
        max_ncells=None,
        emb_layer=-1,
        nproc=4,
        summary_stat=None,
        token_dictionary_file=None,  # so will raise an error if not provided
    ):
        """
        Initialize embedding extractor.

        Parameters
        ----------
        model_type : {"Pretrained","GeneClassifier","CellClassifier"}
            Whether model is the pretrained Geneformer
            or a fine-tuned gene or cell classifier.
        num_classes : int
            If model is a gene or cell classifier,
            specify number of classes it was trained to classify.
            For the pretrained Geneformer model,
            number of classes is 0 as it is not a classifier.
        emb_mode : {"cell","gene"}
            Whether to output cell or gene embeddings.
        max_ncells : None, int
            Maximum number of cells to extract embeddings from.
            Default is 1000 cells randomly sampled from input data.
            If None, will extract embeddings from all cells.
        emb_layer : {-1, 0}
            Embedding layer to extract.
            The last layer is most specifically weighted to optimize
            the given learning objective.
            Generally, it is best to extract the 2nd to last layer
            to get a more general representation.
            -1: 2nd to last layer
            0: last layer
        emb_label : None, list
            List of column name(s) in .dataset to add as labels
            to embedding output.
        labels_to_plot : None, list
            Cell labels to plot.
            Shown as color bar in heatmap.
            Shown as cell color in umap.
            Plotting umap requires labels to plot.
        nproc : int
            Number of CPU processes to use.
        summary_stat : {None, "mean", "median"}
            If not None, outputs only approximated mean or
            median embedding of input data.
            Recommended if encountering memory constraints
            while generating goal embedding positions.
            Slower but more memory-efficient.
        token_dictionary_file : Path
            Path to pickle file containing token dictionary
            (Ensembl ID:token).
        """

        self.model_type = model_type
        self.num_classes = num_classes
        self.emb_mode = emb_mode
        self.cell_emb_style = 'gene'
        self.max_ncells = max_ncells
        self.emb_layer = emb_layer
        self.nproc = nproc
        self.summary_stat = summary_stat

        # load token dictionary (Ensembl IDs:token)
        with open(token_dictionary_file, 'rb') as f:
            self.gene_token_dict = pickle.load(f)

        self.pad_token_id = self.gene_token_dict.get('<pad>')

    def pad_tensor(self, tensor, pad_token_id, max_len):
        tensor = torch.nn.functional.pad(
            tensor,
            pad=(0, max_len - tensor.numel()),
            mode='constant',
            value=pad_token_id,
        )
        return tensor

    # def pad_tensor_list(
    #     self, tensor_list, dynamic_or_constant, pad_token_id, model_input_size
    # ):
    #     # Determine maximum tensor length
    #     if dynamic_or_constant == 'dynamic':
    #         max_len = max([tensor.squeeze().numel() for tensor in tensor_list])
    #     elif type(dynamic_or_constant) == int:
    #         max_len = dynamic_or_constant
    #     else:
    #         max_len = model_input_size
    #         logger.warning(
    #             'If padding style is constant, must provide integer value. '
    #             f'Setting padding to max input size {model_input_size}.'
    #         )

    #     # pad all tensors to maximum length
    #     tensor_list = [
    #         self.pad_tensor(tensor, pad_token_id, max_len) for tensor in tensor_list
    #     ]

    #     # return stacked tensors
    #     return torch.stack(tensor_list)

    def get_model_input_size(self, model):
        return int(
            re.split('\\(|,', str(model.bert.embeddings.position_embeddings))[1]  # noqa
        )

    def quant_layers(self, model):
        layer_nums = []
        for name, parameter in model.named_parameters():
            if 'layer' in name:
                layer_nums += [int(name.split('layer.')[1].split('.')[0])]
        return int(max(layer_nums)) + 1

    # def gen_attention_mask(self, minibatch_encoding, max_len=2048):
    #     if max_len is None:
    #         max_len = max(minibatch_encoding['length'])

    #     original_lens = minibatch_encoding['length']
    #     attention_mask = [
    #         [1] * original_len + [0] * (max_len - original_len)
    #         if original_len <= max_len
    #         else [1] * max_len
    #         for original_len in original_lens
    #     ]

    #     return torch.tensor(attention_mask).to(minibatch_encoding['input_ids'].device)

    def gen_attention_mask(self, minibatch_encoding):
        # Get device from the 'input_ids' tensor
        device = minibatch_encoding['input_ids'].device

        # Convert 'original_lens' to a tensor
        original_lens = minibatch_encoding['length']
        max_len = max(original_lens)
        if not isinstance(original_lens, torch.Tensor):
            original_lens = torch.tensor(original_lens, device=device)

        # Create a mask for each sequence in the batch
        # Initialize a tensor of zeros with the shape [batch_size, max_len]
        attention_mask = torch.zeros((len(original_lens), max_len), device=device)

        seq_range = torch.arange(max_len, device=device).expand(
            len(original_lens), max_len
        )
        attention_mask[seq_range < original_lens.unsqueeze(1)] = 1

        return attention_mask

    def extract_embs(self, model, input_data, inference, use_grad=False):
        """
        Extract embeddings from input data and save as results in output_directory.

        Parameters
        ----------
        model_directory : Path
            Path to directory containing model
        input_data_file : Path
            Path to directory containing .dataset inputs
        output_directory : Path
            Path to directory where embedding data will be saved as csv
        output_prefix : str
            Prefix for output file
        output_torch_embs : bool
            Whether or not to also output the embeddings as a tensor.
            Note, if true, will output embeddings as both dataframe and tensor.
        """

        layer_to_quant = self.quant_layers(model) + self.emb_layer

        model_input_size = self.get_model_input_size(model)

        input_data_minibatch = input_data['input_ids']
        input_data_minibatch = pad_tensor_list(
            input_data_minibatch, 'dynamic', self.pad_token_id, model_input_size
        )

        if inference:
            model.eval()

        if use_grad:
            outputs = model(
                input_ids=input_data_minibatch,
                attention_mask=self.gen_attention_mask(input_data),
            )
        else:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_data_minibatch,
                    attention_mask=self.gen_attention_mask(input_data),
                )

        embs = outputs.hidden_states[layer_to_quant]

        return embs
