import __init__
import os
from ogb.nodeproppred import Evaluator
import torch
from torch_geometric.utils import add_self_loops
from args import ArgsInit
from ogb.nodeproppred import PygNodePropPredDataset
from model import DeeperGCN

from multiprocessing import Pool

import json


@torch.no_grad()
def test(model, x, edge_index, y_true, split_idx, evaluator):
    # test on CPU
    model.eval()
    out = model(x, edge_index)

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return {"train_acc": train_acc, "valid_acc": valid_acc, "test_acc": test_acc}

def eval_model(params):
    model_load_path, args = params
    if os.path.isdir(args.model_load_path):
        model_load_dir = args.model_load_path
        model_load_path = os.path.join(model_load_dir, model_load_path)
    print("Starting evaluating model stored at", model_load_path)

    dataset = PygNodePropPredDataset(name=args.dataset, root=args.data_folder)
    graph = dataset[0]

    if args.self_loop:
        graph.edge_index = add_self_loops(edge_index=graph.edge_index,
                                          num_nodes=graph.num_nodes)[0]
    split_idx = dataset.get_idx_split()

    evaluator = Evaluator(args.dataset)

    args.in_channels = graph.x.size(-1)
    args.num_tasks = dataset.num_classes

    model = DeeperGCN(args)
    ckpt = torch.load(model_load_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model_state_dict'])
    test_res = test(model, graph.x, graph.edge_index, graph.y, split_idx,
                  evaluator)
    test_res["model_load_path"] = model_load_path

    return test_res

def main():
    args = ArgsInit().args

    if os.path.isdir(args.model_load_path):
        model_load_paths = os.listdir(args.model_load_path)
    else:
        model_load_paths = [args.model_load_path]

    num_workers = len(model_load_paths)
    with Pool(num_workers) as pool:
        all_res =  list(pool.imap_unordered(eval_model, [(p, args) for p in model_load_paths]))

    print(all_res)

    with open(args.eval_res_path, "w") as f:
        for res_dict in all_res:
            print(json.dumps(res_dict), file=f)


if __name__ == "__main__":
    main()
