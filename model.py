import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

def build_mlp(in_size, hidden_size, out_size, lay_norm=True):
    module = nn.Sequential(nn.Linear(in_size, hidden_size),
                           nn.ReLU(),
                           nn.Linear(hidden_size, hidden_size),
                           nn.ReLU(),
                           nn.Linear(hidden_size, hidden_size),
                           nn.ReLU(),
                           nn.Linear(hidden_size, out_size))
    if lay_norm:
        return nn.Sequential(module,  nn.LayerNorm(normalized_shape=out_size))
    
    return module


class GraphNetBlock(nn.Module):
    def __init__(self, args, hidden_size):
        super(GraphNetBlock, self).__init__()
        self.args = args
        
        eb_input_dim = 3 * hidden_size
        nb_input_dim = 2 * hidden_size
        self.node_mlp = build_mlp(nb_input_dim, hidden_size, hidden_size)

        adv_input_dim = hidden_size

        if self.args.dir_message == 'concat':
            adv_input_dim = 2 * hidden_size
        elif self.args.dir_message == 'concat_learnable':
            self.concat_proj = nn.Linear(2 * hidden_size, hidden_size)
        elif self.args.dir_message == 'diff_learnable':
            self.diff_proj = nn.Linear(hidden_size, hidden_size)

        self.advection_output_mlp = build_mlp(adv_input_dim, hidden_size, hidden_size)

        if self.args.dir_info == "learned_scalar":
            self.alignment_weight = nn.Linear(hidden_size, hidden_size)
    
    def node_update(self, node_features, edge_index, advection):
        # Node update
        senders_idx, receivers_idx = edge_index

        aggregated_advection = scatter_add(advection, receivers_idx, dim=0, dim_size=node_features.size(0))

        collected_nodes = torch.cat([node_features, aggregated_advection], dim=-1)
        updated_node_features = self.node_mlp(collected_nodes)

        return updated_node_features
    
    def advection_update(self, node_features, edge_index, edge_attr):
        senders_idx, receivers_idx = edge_index
        senders_attr = node_features[senders_idx]
        receivers_attr = node_features[receivers_idx]

        if self.args.dir_info == "vector":
            dir_inf = senders_attr * edge_attr
        elif self.args.dir_info == "scalar":
            dir_inf = torch.sum(senders_attr * edge_attr, dim=-1, keepdim=True)
        elif self.args.dir_info == "cosine":
            dir_inf = F.cosine_similarity(senders_attr, edge_attr, dim=-1).unsqueeze(-1)
        elif self.args.dir_info == "kernel":
            dir_inf = torch.exp(-torch.sum((senders_attr - edge_attr)**2, dim=-1, keepdim=True) / self.args.sigma)
        elif self.args.dir_info == "learned_scalar":
            dir_inf = torch.sum(self.alignment_weight(senders_attr) * edge_attr, dim=-1, keepdim=True)
        else:
            raise NotImplementedError
        
        if self.args.dir_message == 'diff':
            raw_msg = receivers_attr - senders_attr
        elif self.args.dir_message == 'receiver':
            raw_msg = receivers_attr
        elif self.args.dir_message == 'sender':
            raw_msg = senders_attr
        elif self.args.dir_message == 'concat':
            raw_msg = torch.cat([receivers_attr, senders_attr], dim=-1)
        elif self.args.dir_message == 'concat_learnable':
            combined = torch.cat([receivers_attr, senders_attr], dim=-1)
            raw_msg = self.concat_proj(combined)
        elif self.args.dir_message == 'diff_learnable':
            diff = receivers_attr - senders_attr
            raw_msg = self.diff_proj(diff)
        else:
            raise NotImplementedError

        advection = self.advection_output_mlp(dir_inf * raw_msg)

        return advection


    def forward(self, node_features, edge_index, edge_attr):
        """
        node_features (Tensor): Node feature matrix of shape [num_nodes, node_feature_dim].
        edge_index (Tensor): Edge index tensor of shape [2, num_edges].
        edge_attr (Tensor): Edge feature matrix of shape [num_edges, edge_feature_dim].
        """
        original_node_features = node_features.clone()
        original_edge_attr = edge_attr.clone()

        advection = self.advection_update(node_features, edge_index, edge_attr)

        updated_node_features = self.node_update(node_features, edge_index, advection)

        if self.args.layer_node_residual_connection:
            node_features = original_node_features + updated_node_features
        else:
            node_features = original_node_features

        if self.args.layer_edge_residual_connection:
            edge_attr = original_edge_attr + advection
        else:
            edge_attr = original_edge_attr

        return node_features, edge_index, edge_attr


class EncodeProcessDecode(nn.Module):
    def __init__(self, args):
        super(EncodeProcessDecode, self).__init__()
        self._latent_size = args.latent_size
        self._output_size = args.output_size
        self._num_layers = args.num_layers
        self._message_passing_steps = args.message_passing_steps
        self._mask_latent_size = args.mask_latent_size
        self._node_input_size = args.node_input_size
        self._edge_input_size = args.edge_input_size

        self._use_random = args.use_random
        self._ratio = args.ratio

        self.args = args

        self.node_embedding_mlp = build_mlp(self._node_input_size+self._mask_latent_size, self._latent_size, self._latent_size, lay_norm=True)
        self.edge_embedding_mlp = build_mlp(self._edge_input_size , self._latent_size, self._latent_size, lay_norm=True)
        self.to_interpolate_mask_mlp = build_mlp(1, self._mask_latent_size, self._mask_latent_size, lay_norm=False)

        processer_list = []
        for _ in range(self._message_passing_steps):
            processer_list.append(GraphNetBlock(args, self._latent_size))
        self.processer_list = nn.ModuleList(processer_list)

        self.decoder_mlp = build_mlp(self._latent_size, self._latent_size, self._output_size, lay_norm=False)

    def _encoder(self, x, edge_attr):
        node_latents = self.node_embedding_mlp(x)
        edge_latents = self.edge_embedding_mlp(edge_attr)
        
        return node_latents, edge_latents
    
    def _decoder(self, x):
        node_decoded = self.decoder_mlp(x)
        return node_decoded

    def forward(self, graph):
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr

        if self._use_random:
            node_feature, to_interpolate_mask = self.randomize_graph(graph)
        else:
            node_feature = graph.x
            to_interpolate_mask = graph.to_interpolate_mask

        original_node_feature = node_feature.clone()
        original_to_interpolate_mask = to_interpolate_mask.clone()

        to_interpolate_mask = self.to_interpolate_mask_mlp(to_interpolate_mask.float())
        node_feature = torch.cat([node_feature, to_interpolate_mask], dim = 1)

        node_features, edge_attr = self._encoder(x = node_feature.float(), edge_attr = edge_attr.float())
        original_embed_node_features = node_features.clone()

        for model in self.processer_list:
            node_features, edge_index, edge_attr = model(node_features, edge_index, edge_attr)

        if self.args.latent_residual_connection:
            node_features = original_embed_node_features + node_features

        decoded = self._decoder(node_features)

        if self.args.model_residual_connection:
            decoded = original_node_feature + decoded

        return decoded, original_to_interpolate_mask
    
    def randomize_graph(self, graph):
        device = graph.y.device
        num_nodes = graph.y.size(0)
        to_interpolate_mask = torch.zeros((num_nodes,), dtype=torch.long, device=device)
        node_feature = torch.zeros_like(graph.x, device=device)

        for graph_idx in graph.batch.unique():
            node_mask = (graph.batch == graph_idx)
            # print(f"graph_idx: {graph_idx}")
            
            num_nodes_graph = node_mask.sum().item()
            # print(f"num_nodes_graph: {num_nodes_graph}")
            target_interpolate = int((1 - self._ratio) * num_nodes_graph) # num of 1s

            random_mask = torch.bernoulli(torch.full((num_nodes_graph,), 0.5, device=device)).long()
            current_interpolate = random_mask.sum().item()

            if current_interpolate > target_interpolate:
                ones_idx = torch.where(random_mask == 1)[0]
                perm = ones_idx[torch.randperm(len(ones_idx))]
                flip = current_interpolate - target_interpolate
                random_mask[perm[:flip]] = 0

            elif current_interpolate < target_interpolate:
                zeros_idx = torch.where(random_mask == 0)[0]
                perm = zeros_idx[torch.randperm(len(zeros_idx))]
                flip = target_interpolate - current_interpolate
                random_mask[perm[:flip]] = 1
            
            assert random_mask.sum().item() == target_interpolate

            to_interpolate_mask[node_mask] = random_mask

            # node feature x
            locations = graph.pos[node_mask, :]
            frame_data = graph.y[node_mask, :]
            mask_bool = random_mask.bool()
            interpolate_indices = torch.where(mask_bool)[0]
            non_interpolate_indices = torch.where(~mask_bool)[0]

            interpolate_locations = locations[interpolate_indices]
            non_interpolate_locations = locations[non_interpolate_indices]
            distances = torch.cdist(interpolate_locations, non_interpolate_locations, p=2)
            nearest_neighbor_indices = distances.argmin(dim=1)
            nearest_indices = non_interpolate_indices[nearest_neighbor_indices]
            nearest_features = frame_data[nearest_indices]

            input_data = frame_data.clone()
            input_data[interpolate_indices] = nearest_features

            node_feature[node_mask, :] = input_data
        
        to_interpolate_mask = to_interpolate_mask.unsqueeze(-1)

        return node_feature, to_interpolate_mask
    

