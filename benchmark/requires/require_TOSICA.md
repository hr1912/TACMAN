
# create environment

```shell
conda create -n TOSICA python scanpy pytorch-cpu=1.10 torchvision torchaudio -c pytorch --yes
conda install -n TOSICA ipykernel ipywidgets tensorboard einops -c pytorch --yes
conda install -n TOSICA  pyyaml -y
```
# git clone TOSICA

```shell
# cd benchmark/script
git clone https://github.com/JackieHanLab/TOSICA.git

```

# localized modifications


> modify file`TOSICA/__init__.py`

```python
# __version__ = '1.0.0'
# Import locally, TOSICA does not exist in python3.9/site-packages
# version is stored in the parent directory
from pathlib import Path
__version__ = Path(__file__).absolute().parent.parent.joinpath('VERSION.txt').read_text().replace('\n','')

```

> modify file `TOSICA/train.py`

```python
def fit_model(adata, gmt_path, project = None,...):
    # ...

    # project_path = os.getcwd()+'/%s'%project
    # if os.path.exists(project_path) is False:
    #     os.makedirs(project_path)

    # Use the directly passed-in path
    from pathlib import Path
    project_path = Path(project).joinpath('TrainingModel')
    project_path.mkdir(parents=True,exist_ok=True)
    project = project_path.name
    project_path = str(project_path)

    # ...


    # else:
    #     torch.save(model.state_dict(), "/%s"%project_path+"/model-{}.pth".format(epoch))
    # not root
    else:
        torch.save(model.state_dict(), "%s"%project_path+"/model-{}.pth".format(epoch))
```

> modify file `TOSICA/pre.py`

```python

def prediect( #....
    # mask = np.load(mask_path)
    # project_path = os.getcwd()+'/%s'%project
    # pathway = pd.read_csv(project_path+'/pathway.csv', index_col=0)
    # dictionary = pd.read_table(project_path+'/label_dictionary.csv', sep=',',header=0,index_col=0)

    # Use the directly provided path and read mask, pathway, dictionary
    from pathlib import Path
    project_path = Path(project)
    mask = np.load(project_path.joinpath('TrainingModel','mask.npy'))
    pathway = pd.read_csv(project_path.joinpath('TrainingModel','pathway.csv'), index_col=0)
    dictionary = pd.read_table(project_path.joinpath('TrainingModel',
                                                     'label_dictionary.csv'),
                               sep=',',header=0,index_col=0)
    project = project_path.name
    project_path = str(project_path)

```


> add file `TOSICA_gmt_path.csv`

```csv
sp,path
human,TOSICA/resources/GO_bp.gmt
mouse,TOSICA/resources/m_GO_bp.gmt
```
