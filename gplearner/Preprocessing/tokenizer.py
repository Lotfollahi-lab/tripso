####################################################
# Build custom tokenizer to return only GP genes
####################################################

import numpy as np
from datasets import Dataset
from geneformer.tokenizer import TranscriptomeTokenizer


class GPTokenizer(TranscriptomeTokenizer):
    def __init__(self, gp_genes, do_ensembl_conversion, **kwargs):
        super().__init__(**kwargs)
        self.gp_genes = gp_genes

        # convert gene names to token ids
        if do_ensembl_conversion:
            ensembl_id = [self.gene_mapping_dict.get(gene, None) for gene in gp_genes]
        else:
            ensembl_id = gp_genes

        gp_token_ids = [self.gene_token_dict.get(gene, None) for gene in ensembl_id]
        gp_token_ids = [token for token in gp_token_ids if token is not None]
        self.gp_tokens = set(gp_token_ids)

    def create_dataset(
        self,
        tokenized_cells,
        cell_metadata,
        use_generator=False,
        keep_uncropped_input_ids=False,
    ):
        print('Creating dataset.')
        # create dict for dataset creation

        tokenized_by_gp = []

        for i, genes in enumerate(tokenized_cells):
            # x = list(set(genes) & self.gp_tokens)
            x = [g for g in genes if g in self.gp_tokens]
            tokenized_by_gp += [x]

        dataset_dict = {'input_ids': tokenized_by_gp}

        if self.custom_attr_name_dict is not None:
            dataset_dict.update(cell_metadata)

        # create dataset
        if use_generator:

            def dict_generator():
                for i in range(len(tokenized_cells)):
                    yield {k: dataset_dict[k][i] for k in dataset_dict.keys()}

            output_dataset = Dataset.from_generator(dict_generator, num_proc=self.nproc)
        else:
            output_dataset = Dataset.from_dict(dataset_dict)

        # filter out cells with no genes
        output_dataset = output_dataset.filter(lambda x: len(x['input_ids']) > 0)

        def format_cell_features(example):
            # Store original uncropped input_ids in separate feature
            if keep_uncropped_input_ids:
                example['input_ids_uncropped'] = example['input_ids']
                example['length_uncropped'] = len(example['input_ids'])

            # Truncate/Crop input_ids to input size
            if self.special_token:
                example['input_ids'] = example['input_ids'][
                    0 : self.model_input_size - 2
                ]  # truncate to leave space for CLS and EOS token

                example['input_ids'] = np.insert(
                    example['input_ids'], 0, self.gene_token_dict.get('<cls>')
                )

                example['input_ids'] = np.insert(
                    example['input_ids'],
                    len(example['input_ids']),
                    self.gene_token_dict.get('<eos>'),
                )
            else:
                # Truncate/Crop input_ids to input size
                example['input_ids'] = example['input_ids'][0 : self.model_input_size]

            example['length'] = len(example['input_ids'])

            return example

        output_dataset_truncated = output_dataset.map(
            format_cell_features, num_proc=self.nproc
        )

        return output_dataset_truncated
