import pandas as pd
from rdkit import Chem


def find_unlabelled_centers(smiles: str):
    """Return a list of atom indices for unlabelled stereogenic centers."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # invalid SMILES

    centers = Chem.FindMolChiralCenters(
        mol,
        force=True,
        includeUnassigned=True,
        useLegacyImplementation=False,
    )

    # keep atoms with unknown stereochemistry ("?")
    return [idx for idx, label in centers if label == "?"]


def main(save=False):
    df = pd.read_csv("data/surfpro_literature.csv")

    results = []
    for smiles in df.SMILES:
        centers = find_unlabelled_centers(smiles)

        if centers is None:
            results.append("")
        else:
            results.append(";".join(map(str, centers)))

    df["atom_idx_unlabelled_stereogenic_centers"] = results

    if save:
        df.to_csv(
            "data/surfpro_literature_stereogenic_unlabelled.csv",
            index=False,
        )


if __name__ == "__main__":
    # change save kwarg to save the resulting spreadsheet
    main(save=False)
