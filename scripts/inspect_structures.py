import io
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

rdDepictor.SetPreferCoordGen(True)

def main():
    df = pd.read_csv("data/surfpro_literature.csv")

    bookmark = 1480

    plt.ion()  # interactive mode ON
    fig, ax = plt.subplots()

    for idx, smiles in enumerate(df.SMILES, start=2):
        if idx < bookmark:
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Skipping invalid SMILES at row {idx}")
            continue

        centers = Chem.FindMolChiralCenters(
            mol, force=True, includeUnassigned=True, useLegacyImplementation=False
        )

        # compute coords once
        if not mol.GetNumConformers():
            rdDepictor.Compute2DCoords(mol)

        highlight = [c[0] for c in centers if c[1] == "?"]

        drawer = Draw.MolDraw2DCairo(600, 600)
        drawer.DrawMolecule(mol, highlightAtoms=highlight)
        drawer.FinishDrawing()

        img = drawer.GetDrawingText()

        ax.clear()
        ax.imshow(plt.imread(io.BytesIO(img), format="png"))
        ax.set_title(f"Index {idx}", fontweight="bold")
        ax.axis("off")

        plt.draw()
        plt.pause(0.001)

        input("Press ENTER to continue...")

    plt.ioff()

if __name__ == "__main__":
    main()
