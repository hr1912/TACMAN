if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-species')
    # For data input
    parser.add_argument('--species_1_path',
                        default="/public/workspace/ruru_97/projects/data/ByTissue//pancreas/GSE84113/human.h5ad")
    parser.add_argument('--species_2_path',
                        default="/public/workspace/ruru_97/projects/data/ByTissue//pancreas/GSE84113/mouse.h5ad")
    parser.add_argument('--homo_method', default="biomart")
    parser.add_argument('--tessie', default="pancreas")
    parser.add_argument('--species', nargs='+', default=["human", "mouse"])

    # Process params for setting of bioinformatic
    parser.add_argument('--homo_rel', default="n2n")
    parser.add_argument('--nums_hvgs', default=2000)
    parser.add_argument('--nums_degs', default=50)

    # For environment construction
    parser.add_argument("--seeds", nargs='+', type=int, default=[123],
                        help="the seed used in the training")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action='store_true', default=True)
    parser.add_argument("--root", type=str, default='../data/')
    parser.add_argument("--stages", nargs='+', type=int, default=[10, 10, 10],
                        help="The epoch setting for each stage.")

    # For pre-processing
    parser.add_argument("--emb_path", type=str, default='../data/')
    parser.add_argument("--embed-size", type=int, default=64,
                        help="inital embedding size of nodes with no attributes")
    parser.add_argument("--num-hops", type=int, default=2,
                        help="number of hops for propagation of raw labels")

    # For network structure
    parser.add_argument("--hidden", type=int, default=64)

    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of layers of feature projection")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout on activation")
    parser.add_argument("--input-drop", type=float, default=0.1,
                        help="input dropout of input features")
    parser.add_argument("--att-drop", type=float, default=0.,
                        help="attention dropout of model")
    parser.add_argument("--residual", action='store_true', default=False,
                        help="whether to connect the input features")
    parser.add_argument("--act", type=str, default='relu',
                        help="the activation function of the model")

    # For training
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=100,
                        help="early stop of times of the experiment")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="the threshold of multi-stage learning, confident nodes "
                             + "whose score above this threshold would be added into the training set")
    parser.add_argument("--gama", type=float, default=0.5,
                        help="parameter for the KL loss")
    parser.add_argument("--start-stage", type=int, default=0)
    parser.add_argument("--reload", type=str, default='')
    args = parser.parse_args()
    main(args)
