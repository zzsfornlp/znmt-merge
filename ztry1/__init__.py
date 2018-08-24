# the modules that is currently tried on the basic of s2s-att baseline

# todo-list (17.11.22)
# 0. n-best outputs and inputs (check 17.11.21)
# 1. for search_beam: local pruning, global pruning and combining
# 2. search_branch
# 2.5. analyse the branching states and count the prunings
# 3. rerank (& with gold & l2r+r2l)
# 4. analysis of the bleus and the searching: how many steps diverge, why beam is better than greedy?
# 5. target n-gram instead of RNN & difficult points & error states

# * how to analysis (for n-best, n=1,2,...,k and maybe for each length bin of sentences)
# 1. one file: oracle
# 2*(leave out). one file: inner bleu
# 3. one file: ranking of the worst sent-BLEU, which ones are the problems
# 4. two files: equal, difference(better or worse)
# 5. multiple files: equal, best for which sentences

# todo-list (17.12.04)
# 1. searching
# 1.1 analyse (maybe should noticing the difficult points)
# 1.2 re-rank (& with gold & l2r+r2l)
# 1.3 n-gram instead of RNN
# 1.4 extract merges & stat more for sg (analyzing sstater)
# 2. training
# 2.0 margin (with no prob)
# 2.1 error states
# 2.2 recombine
# 2.3 MED

# todo-list (18.01.18)
# extract merges/segments
# tuning the parameters of MED
# mt_select -> maybe select good pred sentences as gold (** maybe later)

# todo (18.01.27)
# merging for decoding: cov heuristics & checking
