"""
Step 0: convert the Desmedt et al. 2008 breast-cancer modules (genefu R package,
data/mod1.rda) into a plain CSV so Python can use them without R.

Input : data/gene_sets/mod1.rda   (download from https://github.com/bhklab/genefu, data/ folder)
Output: data/gene_sets/desmedt2008_modules.csv  (probe, EntrezGene.ID, coefficient, module)
"""
import os
import pandas as pd
import rdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GS = os.path.join(ROOT, "data", "gene_sets")

parsed = rdata.parser.parse_file(os.path.join(GS, "mod1.rda"))
mod = rdata.conversion.convert(parsed)["mod1"]
out = pd.concat([pd.DataFrame(v).assign(module=str(k)) for k, v in mod.items()])
out.to_csv(os.path.join(GS, "desmedt2008_modules.csv"), index=False)
print(out.groupby("module").size().to_string())
