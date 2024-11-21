from rdkit.Chem import MolFromSmiles
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data, Batch
from typing import List
import torch

######################
# adapted from AttentiveFP (https://doi.org/10.1021/acs.jmedchem.9b00959) codebase
# https://github.com/OpenDrugAI/AttentiveFP/blob/master/code/AttentiveFP/Featurizer.py
# https://github.com/OpenDrugAI/AttentiveFP/blob/master/code/AttentiveFP/getFeatures.py
######################


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(
            "input {0} not in allowable set{1}:".format(x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom, explicit_H=False, use_chirality=True):
    results = (
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "B",
                "C",
                "N",
                "O",
                "F",
                "Si",
                "P",
                "S",
                "Cl",
                "As",
                "Se",
                "Br",
                "Te",
                "I",
                "At",
                "other",
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
        + one_of_k_encoding_unk(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
                "other",
            ],
        )
        + [atom.GetIsAromatic()]
    )
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + \
            one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = (
                results
                + one_of_k_encoding_unk(atom.GetProp("_CIPCode"), ["R", "S"])
                + [atom.HasProp("_ChiralityPossible")]
            )
        except:
            results = results + [False, False] + \
                [atom.HasProp("_ChiralityPossible")]

    return np.array(results)


def bond_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()), ["STEREONONE",
                                    "STEREOANY", "STEREOZ", "STEREOE"]
        )
    return np.array(bond_feats)


class MolGraph(object):
    def __init__(self):
        self.nodes = {}  # dict of lists of nodes, keyed by node type
        self.adjacency_list = []  # added by SH to store sparse adjacency list
        self.mol = None  # added by SH to store sparse adjacency list

    def new_node(self, ntype, features=None, rdkit_ix=None):
        new_node = Node(ntype, features, rdkit_ix)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node

    def add_subgraph(self, subgraph):
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))

    def feature_array(self, ntype):
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]])

    def rdkit_ix_array(self):
        return np.array([node.rdkit_ix for node in self.nodes["atom"]])

    def neighbor_list(self, self_ntype, neighbor_ntype):
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n: i for i, n in enumerate(
            self.nodes[neighbor_ntype])}
        return [
            [
                neighbor_idxs[neighbor]
                for neighbor in self_node.get_neighbors(neighbor_ntype)
            ]
            for self_node in self.nodes[self_ntype]
        ]


class Node(object):
    __slots__ = ["ntype", "features", "_neighbors", "rdkit_ix"]

    def __init__(self, ntype, features, rdkit_ix):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_ix = rdkit_ix

    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)

    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]


def graph_from_smiles(smiles):
    graph = MolGraph()
    mol = MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)
    atoms_by_rd_idx = {}
    for atom in mol.GetAtoms():
        new_atom_node = graph.new_node(
            "atom", features=atom_features(atom), rdkit_ix=atom.GetIdx()
        )
        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node

    for bond in mol.GetBonds():
        atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        new_bond_node = graph.new_node("bond", features=bond_features(bond))
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        atom1_node.add_neighbors((atom2_node,))

    mol_node = graph.new_node("molecule")
    mol_node.add_neighbors(graph.nodes["atom"])
    return graph


######################
# following code written by SH
######################


class RDKitGraphFeaturizer:
    """uses RDkit to featurize a list of SMILES strings
    based on the AttentiveFP implementation adapted above,
    and converts them to pytorch-geometric graph Data.
    construct an adjacency list to achieve this"""

    def __init__(self, bidirectional=True, self_loop=False):
        """bidirectional: adds [to, from] for bidirectional message passing
        self_loop: adds [from, from] for self-loop message passing"""

        self.graphs = []
        self.bidirectional = bidirectional
        self.self_loop = self_loop

    def make_adjacency_list(self, smi, edge_features):
        """constructs a sparse adjacency list of [from, to] bonds.
        returns: [[sender1, sender2, ..., senderN],
                  [receiver1, receiver2, ..., receiverN]]"""

        mol = MolFromSmiles(smi)
        adjacency_list = []
        natoms = mol.GetNumAtoms()
        for bond in mol.GetBonds():
            adjacency_list.append(
                [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]
            )

        if self.bidirectional:
            for bond in mol.GetBonds():
                adjacency_list.append(
                    [bond.GetEndAtom().GetIdx(), bond.GetBeginAtom().GetIdx()]
                )
            # stack edge features again
            edge_features = np.concat([edge_features, edge_features])

        if self.self_loop:
            for node in mol.GetAtoms():
                adjacency_list.append([node.GetIdx(), node.GetIdx()])

            # first adds all-zero features to edge_feats for self_loops
            self_loop_feats = np.zeros((natoms, edge_features.shape[1]))

            # then extend edge_feats by column: one-hot of self_loop
            self_loop_onehot = np.concat(
                [np.zeros(len(edge_features)), np.ones(natoms)]
            ).reshape(-1, 1)

            # then concat them together
            edge_features = np.concat([edge_features, self_loop_feats], axis=0)
            edge_features = np.concat(
                [edge_features, self_loop_onehot], axis=1)

        return np.array(adjacency_list, dtype=int).transpose(), edge_features

    def featurize_smiles(self, smi: str, temp=None):
        graph = graph_from_smiles(smi)
        node_features = graph.feature_array("atom")
        edge_features = graph.feature_array("bond")

        adjacency_list, edge_features = self.make_adjacency_list(
            smi, edge_features)

        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(adjacency_list, dtype=torch.long),
            edge_attr=torch.tensor(edge_features, dtype=torch.float),
            y=torch.tensor(temp, dtype=torch.float) if temp else None,
        )

    def __call__(self, smiles: List[str], temps: List[float] = None):
        if temps is not None:
            return [
                self.featurize_smiles(smi, t) for smi, t in list(zip(smiles, temps))
            ]
        else:
            return [self.featurize_smiles(smi) for smi in smiles]


if __name__ == "__main__":
    # verify on a couple of example surfactants
    smiles = [
        "CCCCCCCCCCCC[N+](C)(C)CC[N+](C)(C)CCCCCCCCCCCC.[Br-].[Br-]",
        "CCCCCCCCCCCC[N+](C)(C)CCCCCCCCCCCC.[Br-]",
        "CCCCCCCCCCCCCCCCOCCOCCOCCOCCOCCOCCO",
        "CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]",
    ]
    featurizer = RDKitGraphFeaturizer(bidirectional=True, self_loop=False)
    pyg_graphs = featurizer(smiles)
    [g.validate(raise_on_error=True) for g in pyg_graphs]

    featurizer = RDKitGraphFeaturizer(bidirectional=True, self_loop=True)
    pyg_graphs_self = featurizer(smiles)

    print("edge feats")
    print(pyg_graphs[0].edge_attr.shape, pyg_graphs_self[0].edge_attr.shape)
    print(pyg_graphs[1].edge_attr.shape, pyg_graphs_self[1].edge_attr.shape)
    print(pyg_graphs[2].edge_attr.shape, pyg_graphs_self[2].edge_attr.shape)

    print(pyg_graphs[0])
    print(pyg_graphs[1])
    print(pyg_graphs[2])

    print("nodes atom types")
    print(pyg_graphs[0].x[:, :6])

    print("edges")
    [
        print(t)
        for t in list(
            zip(
                pyg_graphs[0].edge_index[0],
                pyg_graphs[0].edge_index[1],
                pyg_graphs[0].edge_attr[:],
            )
        )
    ]

    feats = Batch.from_data_list(pyg_graphs)
    print(feats)
