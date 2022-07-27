import copy
import argparse
from tqdm import tqdm
import yaml

import torch
from torch.nn.functional import cosine_similarity
from torch.optim import AdamW
from torch_geometric.utils import subgraph
from torch_geometric.loader import NeighborSampler, NeighborLoader, GraphSAINTRandomWalkSampler
from torch_geometric.data import DataLoader
from torch_geometric.data import Data

from gnn_dataset.utils import get_pyg_dataset, output_embs, dotdict
from models.bgrl import *
from models.encoder import GraphConv_Encoder, SAGE_Encoder, GCN_Encoder, GAT_Encoder


def main(args, config=None):

    if config is None:
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    config = dotdict(config)
    if config.gpu != -1 and torch.cuda.is_available():
        config.device = 'cuda:{}'.format(config.gpu)
    else:
        config.device = 'cpu'
    if args and args.recache:
        config.use_cached = False
    else:
        config.use_cached = True
    print(config)

    dataset = get_pyg_dataset(config.dataname, save_dir=config.save_dir, use_cached=config.use_cached,
                              inductive=not config.transductive)
    print(dataset)
    data = dataset[0]

    train_edge_index, train_edge_weight = subgraph(data.train_mask, edge_index=data.edge_index,
                                                   edge_attr=data.edge_weight)
    train_graph = Data(data.x, train_edge_index, edge_weight=train_edge_weight, n_id=torch.arange(data.num_nodes))
    if config.saint_sampling:
        train_loader = GraphSAINTRandomWalkSampler(train_graph, batch_size=6000, walk_length=2,
                                             num_steps=5, sample_coverage=100,
                                             save_dir=dataset.processed_dir,
                                             num_workers=8)
    else:
        train_loader = NeighborLoader(train_graph, num_neighbors=config.num_neighbors, shuffle=True,
                                      input_nodes=None, batch_size=config.batch_size)

    # prepare transforms
    transform_1 = get_graph_drop_transform(drop_edge_p=config.drop_edge_p_1, drop_feat_p=config.drop_feat_p_1)
    transform_2 = get_graph_drop_transform(drop_edge_p=config.drop_edge_p_2, drop_feat_p=config.drop_feat_p_2)

    # build networks
    input_size, hidden_size, emb_size = data.x.size(1), config.encoder_hidden_layer, config.emb_size
    #encoder = GCN_Encoder([input_size, hidden_size, hidden_size, config.emb_size])
    #encoder = SAGE_Encoder(input_size, hidden_size, config.emb_size, num_layers=config.num_layers)
    #encoder = GraphConv_Encoder(input_size, hidden_size, config.emb_size, num_layers=config.num_layers)
    encoder = GAT_Encoder(input_size=input_size,
                          hidden_size=hidden_size, 
                          embedding_size=config.emb_size, 
                          num_layers=config.num_layers,
                          heads=config.num_heads)

    if config.verbose:
        print(encoder)
    predictor = MLP_Predictor(emb_size, emb_size, hidden_size=config.predictor_hidden_size)
    model = BGRL(encoder, predictor).to(config.device)

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # scheduler
    mm_scheduler = CosineDecayScheduler(1 - config.mm, 0, config.epochs)

    cnt_wait = 0
    best = 1e9
    best_t = 0
    for epoch in range(config.epochs):
        model.train()

        # update momentum
        mm = 1 - mm_scheduler.get(epoch)

        total_loss = 0
        for batch in train_loader:
            batch = batch.to(config.device)

            # forward
            optimizer.zero_grad()

            x1, x2 = transform_1(batch), transform_2(batch)

            q1, y2 = model(x1, x2)
            q2, y1 = model(x2, x1)

            loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
            loss.backward()

            # update online network
            optimizer.step()
            # update target network
            model.update_target_network(mm)

            total_loss += float(loss)

        loss = total_loss / data.num_nodes

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), f'saved_models/bgrl-{config.dataname}.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == config.patience:
            if config.verbose:
                print('Early stopping!')
            break

        if config.verbose:
            print('Epoch: {0}, Loss: {1:0.8f}'.format(epoch, loss*100))

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load(f'saved_models/bgrl-{config.dataname}.pkl'))

    embeds = compute_representations(model.online_encoder, data, device=config.device, batch_size=config.batch_size)
    output_embs(embeds.cpu().detach().numpy(), dataset=dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BGRL')
    parser.add_argument('--config', type=str, default='./config/tmall/bgrl.yaml', help='Config file.')
    parser.add_argument('--recache', action='store_true', help='Recache GNN dataset.')

    args = parser.parse_args()
    main(args)