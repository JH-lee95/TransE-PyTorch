from collections import Counter
from torch.utils import data
from typing import Dict, Tuple

Mapping = Dict[str, int]


def create_triples(datapath):
  with open(datapath,"r") as f:
    lines=f.readlines()
    triples=[]
    for line in lines:
      triple=line.replace("\n","").split("\t")
      triples.append(triple)

  return triples


class PersonaTripleDataset(data.Dataset):
  def __init__(self,ent_dict,rel_dict,triples):
    self.ent2id=ent_dict
    self.rel2id=rel_dict
    self.triples=triples


  def __len__(self):
    return len(self.triples)


  def __getitem__(self,idx):
    h,r,t=self.triples[idx]

    h_id=self.ent2id[h]
    r_id=self.rel2id[r]
    t_id=self.ent2id[t]

    return h_id,r_id,t_id