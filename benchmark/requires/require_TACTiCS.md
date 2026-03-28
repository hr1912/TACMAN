
# create environment

```shell
conda create -n TACTiCS --file require_TACTiCS --yes
conda activate TACTiCS
```
# git clone TACTiCS

```shell
# cd benchmark/script
git clone https://github.com/kbiharie/TACTiCS.git

```

# download data
```shell
cd benchmark/script/TACTiCS
wget -b -c -o download.log https://zenodo.org/records/11191718/files/generated.zip
wget -b -c -o download.log https://zenodo.org/records/11191718/files/data.zip
mkdir tutorial
cd tutorial
unzip ../generated.zip
unzip ../data.zip
```

# localized modifications

> add file `__init__.py`

```python
from . import genes
```

> modify file `TACTiCS/tactics.py`

```python
import TACTiCS.utils as utils
```

> add file `TACTiCS_genes_path.csv`

```csv
sp,path
human,tutorial/human_names.pkl
mouse,tutorial/mouse_names.pkl
zebrafish,tutorial/zebrafish_names.pkl
chicken,tutorial/chicken_names.pkl
```

> add file `TACTiCS_protein_dist.csv`
```csv
sp_ref,sp_que,path
human,mouse,tutorial/human_mouse_dist.pkl
mouse,human,tutorial/mouse_human_dist.pkl
zebrafish,human,tutorial/zebrafish_human_dist.pkl
zebrafish,mouse,tutorial/zebrafish_mouse_dist.pkl
zebrafish,chicken,tutorial/zebrafish_chicken_dist.pkl
```

# init

```shell
cd benchmark
conda activate TACTiCS
python script/init_TACTiCS.py
# [out] .../tutorial/human_mouse_dist.pkl
# [out] .../tutorial/mouse_human_dist.pkl

```