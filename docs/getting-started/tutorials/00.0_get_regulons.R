## Build gp_curated.csv: CollecTRI regulons of literature-selected haematopoietic TFs.
## Output is wide: one column per GP (GP_<TF>), first row is the TF itself, followed
## by its CollecTRI targets in CollecTRI order; columns sorted by size (desc), ties
## alphabetical. Filtered downstream by 00.1_make_gpdb.ipynb -> gpdb_tf.csv.
##
## Usage: Rscript 00.0_get_regulons.R [out_file]   (default: gp_curated.csv)
##
## Note: CollecTRI is fetched live from OmniPath, so a rebuild will not be byte-identical
## to the shipped gp_curated.csv (Jan 2026): GP names, sizes and column order match, but
## a few targets were renamed since (SLC9A3R1 -> NHERF1, THEG -> SPMAP2, C4B -> C4B_2,
## H3-3A -> H3-3B, some H2A/H3/H4 histone genes), a few adjacent targets swap order, and
## the shipped file lists DDIT3 twice in GP_DDIT3 (TF + self-loop).

library(decoupleR)
library(tidyverse)

args <- commandArgs(trailingOnly = TRUE)
out_file <- if (length(args) > 0) args[1] else 'gp_curated.csv'
if (file.exists(out_file)) stop(out_file, ' exists; pass another output path or remove it first')

regulon_df <- decoupleR::get_collectri(organism='human', split_complexes=FALSE)

# TFs selected from literature on haematopoiesis / bone marrow differentiation
tfs <- c('USF1', 'NFE2L2', 'RUNX1', 'FOXO3', 'MYB', 'E2F4', 'IRF1', 'GATA1',
         'CTCF', 'MYCN', 'ATF4', 'ATF3', 'JUNB', 'JUND', 'GATA2', 'DDIT3',
         'TAL1', 'FLI1', 'ELF1', 'RUNX3', 'KLF2', 'PRDM1', 'IRF2', 'NR2C2',
         'ERG', 'IKZF1', 'SNAI2', 'NFYB', 'HOXA9', 'IRF5', 'ZBTB7A', 'KLF1',
         'LMO2', 'NFIX', 'ETV6', 'MEIS1', 'SOX6', 'NFE2'
        )

# Keep protein-coding targets only (newer CollecTRI releases also ship TF-miRNA edges)
reg_df <- regulon_df %>%
  filter(source %in% tfs, !str_detect(target, '^hsa-'))

setdiff(tfs, unique(reg_df$source))

# TF first, then its targets (self-loop not repeated)
gp_list <- reg_df %>%
  group_by(source) %>%
  summarize(genes = list(unique(c(first(source), target))), .groups = 'drop') %>%
  mutate(n_genes = lengths(genes)) %>%
  arrange(desc(n_genes), source)

print(gp_list %>% select(source, n_genes), n = nrow(gp_list))

# Wide format, shorter GPs padded with empty cells
max_len <- max(gp_list$n_genes)
gp_wide <- map(gp_list$genes, ~ c(.x, rep(NA_character_, max_len - length(.x)))) %>%
  set_names(paste0('GP_', gp_list$source)) %>%
  as_tibble()

write_csv(gp_wide, out_file, na = '')
