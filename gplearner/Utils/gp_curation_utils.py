from tqdm import tqdm

from .utils import remove_leading_numbers_and_underscore

####################################
# Building ontology DAG
####################################

# This section is adapted from anc2vec
# https://github.com/aedera/anc2vec/blob/main/anc2vec/train/onto.py
# accessed 15/12/2023

# root terms
BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS,
    'cellular_component': CELLULAR_COMPONENT,
    'molecular_function': MOLECULAR_FUNCTION,
    'biological_process': BIOLOGICAL_PROCESS,
}
NAMESPACES = {
    'cc': 'cellular_component',
    'mf': 'molecular_function',
    'bp': 'biological_process',
}
namespace2go = {
    'cellular_component': CELLULAR_COMPONENT,
    'molecular_function': MOLECULAR_FUNCTION,
    'biological_process': BIOLOGICAL_PROCESS,
}


class Ontology(object):
    def __init__(
        self,
        filename='pathways_db/go.obo',
        with_rels=True,
        remove_obs=True,
        include_alt_ids=False,
    ):
        """
        if with_rels=False only consider is_a as relationship
        """
        self.fname = filename
        self.remove_obs = remove_obs
        self.include_alt_ids = include_alt_ids
        self.leaves = []
        self.ont = self.load_data(filename, with_rels)

    def load_data(self, filename, with_rels):
        ont = dict()
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['has_part'] = list()
                    obj['regulates'] = list()
                    obj['negatively_regulates'] = list()
                    obj['positively_regulates'] = list()
                    obj['occurs_in'] = list()
                    obj['ends_during'] = list()
                    obj['happens_during'] = list()
                    obj['alt_ids'] = set([])
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(': ')  # noqa E741
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].add(l[1])
                        # breakpoint()
                    # avoid storing the main id as alternative id
                    # obj['alt_ids'] -= set([obj['id']])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        rel_type = it[0]
                        term_in_rel = it[1]
                        obj[rel_type].append(term_in_rel)
                        # breakpoint()
                        # add all types of relationships
                        # obj['is_a'].append(it[1])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
            if obj is not None:
                ont[obj['id']] = obj
        #
        for term_id in list(ont.keys()):
            if self.include_alt_ids:
                for t_id in ont[term_id]['alt_ids']:  # add alt_ids as ontology terms
                    ont[t_id] = ont[term_id]
            if self.remove_obs and ont[term_id]['is_obsolete']:
                del ont[term_id]

        # REMOVE PART OF ONTOLOGY
        for term_id in list(ont.keys()):
            if ont[term_id]['namespace'] != 'biological_process':
                del ont[term_id]

        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
        # generate leaves
        for term_id, val in ont.items():
            if len(val['children']) == 0:  # no children
                self.leaves.append(term_id)
        return ont

    def get_ancestors(self, term_id):
        if term_id not in self.ont:
            return set()
        #                  set(self.ont[term_id]['has_part'])
        parents = (
            set(self.ont[term_id]['is_a'])
            | set(self.ont[term_id]['part_of'])
            | set(self.ont[term_id]['regulates'])
            | set(self.ont[term_id]['negatively_regulates'])
            | set(self.ont[term_id]['positively_regulates'])
            | set(self.ont[term_id]['occurs_in'])
            | set(self.ont[term_id]['ends_during'])
            | set(self.ont[term_id]['happens_during'])
        )
        if len(parents) < 1:
            return [[term_id]]
        branches = []
        for parent_id in parents:
            branches += [b + [term_id] for b in self.get_ancestors(parent_id)]
        return branches

    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_blanket(self, term_id):
        return (
            set(self.ont[term_id]['is_a'])
            | self.ont[term_id]['children']
            | set(self.ont[term_id]['part_of'])
            | set(self.ont[term_id]['has_part'])
            | set(self.ont[term_id]['regulates'])
            | set(self.ont[term_id]['negatively_regulates'])
            | set(self.ont[term_id]['positively_regulates'])
            | set(self.ont[term_id]['occurs_in'])
            | set(self.ont[term_id]['ends_during'])
            | set(self.ont[term_id]['happens_during'])
        )

    def get_one_gen_ancestors(self, term_id):
        if term_id not in self.ont:
            return set()
        #                  set(self.ont[term_id]['has_part'])
        parents = (
            set(self.ont[term_id]['is_a'])
            | set(self.ont[term_id]['part_of'])
            | set(self.ont[term_id]['regulates'])
            | set(self.ont[term_id]['negatively_regulates'])
            | set(self.ont[term_id]['positively_regulates'])
            | set(self.ont[term_id]['occurs_in'])
            | set(self.ont[term_id]['ends_during'])
            | set(self.ont[term_id]['happens_during'])
        )
        if len(parents) < 1:
            return [[term_id]]
        return parents

    def get_go_from_name(self, term_name):
        terms = set()
        for go_id, obj in self.ont.items():
            if term_name.lower().replace('_', ' ') in obj['name'].lower():
                terms.add(go_id)
        return terms

    def add_genes(self, gobp):
        for p in tqdm(gobp.columns):
            gp = p.replace('GOBP_', '')
            gp = remove_leading_numbers_and_underscore(gp)
            x = self.get_go_from_name(gp)
            x = list(x)

            if len(x) == 1:
                self.ont[x[0]]['gene_set'] = set(gobp[p].dropna())
                self.ont[x[0]]['gobp_name'] = p
            elif len(x) > 1:
                for i in range(len(x)):
                    self.ont[x[i]]['gene_set'] = set(gobp[p].dropna())
                    self.ont[x[i]]['flag'] = 'one_to_many_gene_set_match'
                    self.ont[x[i]]['gobp_name'] = p
            else:
                continue

        for node_id in self.ont.keys():
            if 'gene_set' not in self.ont[node_id].keys():
                self.ont[node_id]['gene_set'] = set()

    def move_up_and_filter(
        self, gene_set, threshold, node_id, genes_to_keep, id_to_keep, col_to_keep
    ):
        # Check if the node exists and the gene set satisfies the threshold
        if len(self.ont[node_id]['gene_set'] & set(gene_set)) > threshold:
            if len(self.ont[node_id]['gene_set']) < 150:
                genes_to_keep[self.ont[node_id]['name']] = self.ont[node_id]['gene_set']
                id_to_keep.add(node_id)
                col_to_keep.add(self.ont[node_id]['gobp_name'])
        elif len(self.ont[node_id]['gene_set']) == 0:
            # Skip pathways without gene annotations
            return
        else:
            # If the node doesn't exist in the ontology,
            # or the threshold is not met, move up in the tree
            parents = self.get_one_gen_ancestors(node_id)

            # Recursively move up for each parent
            for parent_id in parents:
                self.move_up_and_filter(
                    gene_set,
                    threshold,
                    parent_id,
                    genes_to_keep,
                    id_to_keep,
                    col_to_keep,
                )


####################################
# Other helper functions
####################################


def filter_by_size(
    db, genes_freq, genes_rare, threshold_value, threshold_rare, max_gp_len=100
):
    """
    For an input dataframe where GP names are column names
    and gene lists are the columns

    filter out GP which do not meet the threshold criteria
    * max max_gp_len genes per GP
    * min threshold_value genes present in genes50
    (where genes50 is a list of genes which are expressed in at least 50% of the cells)

    """
    gp_to_keep = []
    gp_common = []
    gp_rare = []

    for gp in db.columns:
        # filter for max size
        # and keep if have at least threshold_value genes in expressed in 50% of cells
        # of at least threshold_rare genes in expressed in 10% of cells
        if (len(db[gp].dropna()) < max_gp_len) & (
            (len(set(db[gp].dropna()) & set(genes_freq)) > threshold_value)
            | (len(set(db[gp].dropna()) & set(genes_rare)) > threshold_rare)
        ):
            gp_to_keep.append(gp)

            if len(set(db[gp].dropna()) & set(genes_freq)) > threshold_value:
                gp_common.append(gp)

            if len(set(db[gp].dropna()) & set(genes_rare)) > threshold_rare:
                gp_rare.append(gp)

    gpdb = db[gp_to_keep]

    print('Number of GP with common genes:', len(gp_common))
    print('Number of GP with rare genes:', len(gp_rare))

    return gpdb


def rm_overlapping_gp(df, token_df, threshold=0.3):
    gp_to_drop = []

    # Calculate intersection over length of non-null elements
    for i in tqdm(df.columns):
        for j in df.columns:
            if i == j:
                continue

            intersection = len(set(df[i].dropna()) & set(df[j].dropna()))
            intersection_ratio = (
                intersection / len(df[i].dropna()) if len(df[i].dropna()) > 0 else 0
            )

            if intersection_ratio > threshold:
                # for each gene program,
                # average the proportion of cells which express their genes
                ni = token_df[token_df['gene'].isin(set(df[i].dropna()))]['prop'].mean()
                nj = token_df[token_df['gene'].isin(set(df[j].dropna()))]['prop'].mean()

                if ni > nj:
                    gp_to_drop.append(j)
                else:
                    gp_to_drop.append(i)

    print('GP to drop:', *gp_to_drop)

    df = df[[c for c in df.columns if c not in gp_to_drop]]

    return df
