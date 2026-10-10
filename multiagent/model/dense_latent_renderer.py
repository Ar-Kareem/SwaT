import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from multiagent.model.dense_latent import GroupMLP


class DenseLatentRenderer(nn.Module):
    def __init__(self, receiver_layers, receiver_kv_heads, receiver_head_dim,
                 sender_layers, sender_kv_heads, sender_head_dim, expansion=16):
        super().__init__()
        self.receiver_layers=receiver_layers
        self.receiver_kv_heads=receiver_kv_heads
        self.receiver_head_dim=receiver_head_dim
        self.sender_layers=sender_layers
        self.sender_kv_heads=sender_kv_heads
        self.sender_head_dim=sender_head_dim
        self.expansion=expansion
        self.sender_flat_dim=sender_layers*2*sender_kv_heads*sender_head_dim
        self.layer_map=[round(i*(sender_layers-1)/max(receiver_layers-1,1)) for i in range(receiver_layers)]
        self.register_buffer('head_map',torch.tensor([round(i*(sender_kv_heads-1)/max(receiver_kv_heads-1,1)) for i in range(receiver_kv_heads)]))
        self.keys=nn.ModuleList([GroupMLP(receiver_kv_heads,sender_head_dim,receiver_head_dim,expansion,torch.float32) for _ in range(receiver_layers)])
        self.values=nn.ModuleList([GroupMLP(receiver_kv_heads,sender_head_dim,receiver_head_dim,expansion,torch.float32) for _ in range(receiver_layers)])
        self.gate_logits=nn.Parameter(torch.zeros(receiver_layers,receiver_kv_heads))
        self.bound_stats=False

    def config_dict(self):
        names=('receiver_layers','receiver_kv_heads','receiver_head_dim','sender_layers','sender_kv_heads','sender_head_dim','expansion')
        return {'arch':'dense-latent',**{name:getattr(self,name) for name in names}}

    def forward(self,receiver_embeds,receiver_positions,memory=None,memory_positions=None,
                memory_mask=None,memory_key_limits=None,receiver_mask=None):
        if memory is None or memory_key_limits is None:
            raise ValueError('dense-latent requires sender cache and causal character-boundary limits')
        batch,seq=receiver_positions.shape
        if batch!=1:
            row=lambda t,i:None if t is None else t[i:i+1]
            return torch.cat([self.forward(row(receiver_embeds,i),receiver_positions[i:i+1],memory[i:i+1],
                                           row(memory_positions,i),row(memory_mask,i),memory_key_limits[i:i+1],
                                           row(receiver_mask,i)) for i in range(batch)],dim=0)
        limits=memory_key_limits[0].long().clamp(0,memory.shape[1])
        index=(limits-1).clamp_min(0)
        valid=limits>0
        if memory_mask is not None:
            valid=valid & memory_mask[0].bool().index_select(0,index)
        if receiver_mask is not None:
            valid=valid & receiver_mask[0].bool()
        source=memory.view(batch,memory.shape[1],self.sender_layers,2,self.sender_kv_heads,self.sender_head_dim)
        outputs=[]
        for layer,source_layer in enumerate(self.layer_map):
            def project(source,index,valid,layer=layer,source_layer=source_layer):
                x=source[0,:,source_layer].index_select(0,index).index_select(2,self.head_map)
                k=x[:,0].transpose(0,1).to(self.keys[layer].w1.dtype)
                v=x[:,1].transpose(0,1).to(self.values[layer].w1.dtype)
                gate=self.gate_logits[layer].sigmoid()[:,None,None]
                pk=self.keys[layer](k)*gate
                pv=self.values[layer](v)*gate
                out=torch.stack([pk,pv],dim=0).permute(2,0,1,3)
                return out.to(memory.dtype)*valid[:,None,None,None].to(memory.dtype)
            if torch.is_grad_enabled():
                out=checkpoint(project,source,index,valid,use_reentrant=False)
            else:
                out=project(source,index,valid)
            outputs.append(out)
        return torch.stack(outputs,dim=1).unsqueeze(0)
