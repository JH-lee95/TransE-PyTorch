import numpy as np
import torch
import torch.nn as nn
import sys


class TransW(nn.Module):

    def __init__(self, entity_count, relation_count, lm,tokenizer, id2entity,id2relation, device,dim=1024, norm=1, margin=1.0):
        super(TransW, self).__init__()
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.device = device
        self.norm = norm
        self.dim=dim
    
        self.entities_emb = self._init_enitity_emb()
        self.relations_emb = self._init_relation_emb()
        self.word_emb_model=lm.get_input_embeddings()
        self.tokenizer=tokenizer

        self.id2entity=id2entity
        self.id2relation=id2relation

        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')

    def _init_enitity_emb(self):
        entities_emb = nn.Embedding(num_embeddings=self.entity_count + 1,
                                    embedding_dim=self.dim,
                                    padding_idx=self.entity_count)
        uniform_range = 6 / np.sqrt(self.dim)
        entities_emb.weight.data.uniform_(-uniform_range, uniform_range)
        return entities_emb

    def _init_relation_emb(self):
        relations_emb = nn.Embedding(num_embeddings=self.relation_count + 1,
                                     embedding_dim=self.dim,
                                     padding_idx=self.relation_count)
        uniform_range = 6 / np.sqrt(self.dim)
        relations_emb.weight.data.uniform_(-uniform_range, uniform_range)
        # -1 to avoid nan for OOV vector
        relations_emb.weight.data[:-1, :].div_(relations_emb.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        return relations_emb


    def get_token_embeddings(self,batch_text):
      batch_emb=[]
      # batch_ids=self.tokenizer(batch_text,padding="max_length",max_length=16)["input_ids"]

      batch_input_ids=self.tokenizer(batch_text)["input_ids"]

      for input_id in batch_input_ids:
        token_emb=self.word_emb_model(torch.tensor(input_id).to(self.device))
        token_emb=token_emb.mean(dim=0)
        batch_emb.append(token_emb)

      return torch.stack(batch_emb)
  



    def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor):

        # -1 to avoid nan for OOV vector
        self.entities_emb.weight.data[:-1, :].div_(self.entities_emb.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))

        assert positive_triplets.size()[1] == 3
        positive_distances = self._distance(positive_triplets)

        assert negative_triplets.size()[1] == 3
        negative_distances = self._distance(negative_triplets)

        return self.loss(positive_distances, negative_distances), positive_distances, negative_distances

    def predict(self, triplets: torch.LongTensor):

        return self._distance(triplets)

    def loss(self, positive_distances, negative_distances):
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.criterion(positive_distances, negative_distances, target)

    def _distance(self, triplets):
        assert triplets.size()[1] == 3
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]

        heads_text=[]
        relations_text=[]
        tails_text=[]

        for h,r,t in zip(heads,relations,tails):
          heads_text.append(self.id2entity[int(h)])
          relations_text.append(self.id2relation[int(r)])
          tails_text.append(self.id2entity[int(t)])

        heads_text_emb=self.get_token_embeddings(heads_text)
        relations_text_emb=self.get_token_embeddings(relations_text)
        tails_text_emb=self.get_token_embeddings(tails_text)

        return (self.entities_emb(heads) * heads_text_emb + self.relations_emb(relations) * relations_text_emb - self.entities_emb(tails) * tails_text_emb).norm(p=self.norm,
                                                                                                          dim=1)
