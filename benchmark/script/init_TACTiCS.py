from pathlib import Path
from run_TACTiCS import p_model_root
from run_TACTiCS import ut
from TACTiCS.genes import embed_proteins, calc_dist
from copy import deepcopy

p_benchmark = p_model_root.parent.parent
p_data = p_benchmark.joinpath("data/mtx")

# https://huggingface.co/Rostlab/prot_bert
PATH_PROT_BERT = p_model_root.joinpath("prot_bert")
assert PATH_PROT_BERT.exists()


human = dict(
    name="Human",
    sequences=p_model_root.joinpath("tutorial/human_proteins.fasta"),
    embeddings=p_model_root.joinpath("tutorial/human_proteins_embeddings.pkl"),
    #  counts=p_model_root.joinpath("tutorial/human.h5ad"),
    counts=ut.sc.load_adata(p_data.joinpath("HCL_Spleen")),
    genes=p_model_root.joinpath("tutorial/human_names.pkl"),
    #  column="Subclass",
    column="cell_type",
)

mouse = dict(
    name="Mouse",
    sequences=p_model_root.joinpath("tutorial/mouse_proteins.fasta"),
    embeddings=p_model_root.joinpath("tutorial/mouse_proteins_embeddings.pkl"),
    #  counts=p_model_root.joinpath("tutorial/mouse.h5ad"),
    counts=ut.sc.load_adata(p_data.joinpath("HCL_Spleen")),
    genes=p_model_root.joinpath("tutorial/mouse_names.pkl"),
    #  column="Subclass"
    column="cell_type",
)


def embed_proteins_and_calc_dist(sp1, sp2, p_out):
    sp1 = deepcopy(sp1)
    sp2 = deepcopy(sp2)
    print(
        "[embed_proteins_and_calc_dist] {} {}".format(
            sp1["name"],
            sp2["name"],
        )
    )
    embed_proteins(sp1, model_path=PATH_PROT_BERT)
    embed_proteins(sp2, model_path=PATH_PROT_BERT)
    calc_dist(sp1, sp2, str(p_out))
    if p_out.exists():
        print("[out] {}".format(p_out))
    else:
        print("[fail] {}".format(p_out))


if __name__ == "__main__":
    embed_proteins_and_calc_dist(
        human, mouse, p_model_root.joinpath("tutorial/human_mouse_dist.pkl")
    )
    embed_proteins_and_calc_dist(
        mouse, human, p_model_root.joinpath("tutorial/mouse_human_dist.pkl")
    )
    print("\nfinis\n".center(100, "-"))
    pass
