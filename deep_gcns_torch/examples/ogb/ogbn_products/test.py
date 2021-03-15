import __init__
import os
from ogb.nodeproppred import Evaluator
import torch
import numpy as np

from multiprocessing import Pool

from torch_sparse import SparseTensor
from torch_geometric.utils import add_self_loops
from tqdm import tqdm

from args import ArgsInit
from ogb.nodeproppred import PygNodePropPredDataset
from model import DeeperGCN

from utils.data_util import intersection, random_partition_graph, generate_sub_graphs



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

def eval_with_partition(args):
    model_load_path = args.model_load_path
    print("Starting evaluating model stored at", model_load_path)

    device = torch.device("cuda")
       
    dataset = PygNodePropPredDataset(name=args.dataset, root=args.data_folder)
    graph = dataset[0]
    adj = SparseTensor(row=graph.edge_index[0],
                       col=graph.edge_index[1])
    if args.self_loop:
        adj = adj.set_diag()
        graph.edge_index = add_self_loops(edge_index=graph.edge_index,
                                          num_nodes=graph.num_nodes)[0]
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(args.dataset)

    args.in_channels = graph.x.size(-1)
    args.num_tasks = dataset.num_classes

    print('%s' % args)
    model = DeeperGCN(args).to(device)
    ckpt = torch.load(model_load_path)
    model.load_state_dict(ckpt['model_state_dict'])


    res, out = test_with_partition(
        model, graph, adj, split_idx,
        num_clusters=args.eval_cluster_number,
        num_classes = args.num_tasks, 
        partition_method=args.partition_method,
        evaluator=evaluator,
        device=device
    )
    print(res)
    torch.save(out, f'{model_load_path}/out.pt')
    return res



@torch.no_grad()
def test_with_partition(model, graph, adj, split_idx, num_clusters,
                        num_classes, partition_method, evaluator, device=None):
    x, y_true, num_nodes = graph.x, graph.y, graph.num_nodes

    if partition_method == "random":
        parts = random_partition_graph(num_nodes, num_clusters)
        data = generate_sub_graphs(adj, parts, cluster_number=num_clusters)
    else:
        raise ValueError(f"Invalid partition method {partition_method}")

    model.eval()
    if device is None:
        device = torch.device("cuda")

    sg_nodes, sg_edges = data

    y_pred = torch.zeros(num_nodes, dtype=torch.long).to(device)
    all_logits = torch.zeros((num_nodes, num_classes), dtype=torch.float)

    for idx in tqdm(range(len(sg_nodes))):
        x_ = x[sg_nodes[idx]].to(device)
        sg_edges_ = sg_edges[idx].to(device)
        logits, raw_softmax = model(x_, sg_edges_)
        pred = logits.argmax(dim=-1)
        y_pred[sg_nodes[idx]] = pred
        all_logits[sg_nodes[idx]] = raw_softmax

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

    res_dict = {"train_acc": train_acc, "valid_acc": valid_acc,
                "test_acc": test_acc}
    return res_dict, all_logits




def main():
    args = ArgsInit().args
    eval_with_partition(args)

    # if os.path.isdir(args.model_load_path):
    #     model_load_paths = os.listdir(args.model_load_path)
    # else:
    #     model_load_paths = [args.model_load_path]
    #
    # num_workers = len(model_load_paths)
    # with Pool(num_workers) as pool:
    #     all_res =  list(pool.imap_unordered(eval_model, [(p, args) for p in model_load_paths]))
    #
    # print(all_res)
    #
    # with open(args.eval_res_path, "w") as f:
    #     for res_dict in all_res:
    #         print(json.dumps(res_dict), file=f)


if __name__ == "__main__":
    main()
