"""Date demonstrative pentru etapa 1: volum CBCT sintetic + predicții hard-codate.
Se înlocuiesc în etapa 5 cu un volum real și cu ieșirea modelului antrenat.
Etichetele clinice (grupe) trebuie validate cu medicul coordonator."""
import numpy as np

N = 72

CASES = {
    "Chist dentigerous": dict(
        group="Chist odontogen", probs=[0.82, 0.08, 0.06, 0.04], theta=0.9, rad=0.19, kind="smooth"),
    "Chist radicular": dict(
        group="Chist odontogen", probs=[0.07, 0.84, 0.05, 0.04], theta=2.1, rad=0.15, kind="smooth"),
    "Keratochist odontogen": dict(
        group="Chist odontogen", probs=[0.06, 0.09, 0.78, 0.07], theta=0.4, rad=0.22, kind="smooth"),
    "Ameloblastom": dict(
        group="Tumoră benignă, agresivă local", probs=[0.03, 0.04, 0.10, 0.83], theta=1.5, rad=0.25, kind="septa"),
}
CLASSES = ["Chist dentigerous", "Chist radicular", "Keratochist odontogen", "Ameloblastom"]


def make_case(name: str, seed: int = 0):
    """Returnează (volum, mască, heatmap, centru în voxeli) pentru un caz demonstrativ."""
    c = CASES[name]
    rng = np.random.default_rng(seed)
    ax = np.linspace(-1, 1, N)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")

    vol = np.zeros((N, N, N), dtype=np.float32)
    vol[(x / 0.9) ** 2 + (y / 0.8) ** 2 + (z / 0.95) ** 2 < 1] = 0.42
    r = np.sqrt((x / 0.7) ** 2 + (y / 0.6) ** 2)
    arch = (r > 0.78) & (r < 0.95) & (y > -0.15) & (np.abs(z) < 0.45)
    vol[arch] = 0.9
    teeth = (r > 0.7) & (r < 0.8) & (y > -0.1) & (np.abs(z) < 0.5) & (np.sin(18 * np.arctan2(y, x)) > 0.2)
    vol[teeth] = 1.0

    cx, cy, cz = 0.7 * 0.865 * np.cos(c["theta"]), 0.6 * 0.865 * np.sin(c["theta"]), 0.0
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
    mask = d < c["rad"]
    inside = 0.25 + 0.05 * rng.standard_normal(vol.shape)
    if c["kind"] == "septa":
        inside += 0.25 * (np.sin(34 * x + 20 * y) * np.sin(30 * z + 12 * y) > 0.45)
    vol[mask] = inside[mask]
    vol[(d >= c["rad"]) & (d < c["rad"] + 0.035)] = 0.97
    vol += 0.03 * rng.standard_normal(vol.shape).astype(np.float32)
    vol = np.clip(vol, 0, 1)

    heat = np.exp(-(d ** 2) / (2 * (c["rad"] * 1.1) ** 2))
    center = tuple(int(round((v + 1) / 2 * (N - 1))) for v in (cx, cy, cz))
    return vol, mask, heat.astype(np.float32), center


def lesion_volume_cm3(mask: np.ndarray, voxel_mm: float = 1.5) -> float:
    """Volum în cm³; presupune voxeli izotropi (implicit 1,5 mm (volum sintetic))."""
    return float(mask.sum() * (voxel_mm ** 3) / 1000.0)