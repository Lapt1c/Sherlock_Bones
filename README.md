# AI pentru zâmbete: detecția, segmentarea și clasificarea tumorilor osoase maxilo-faciale pe CBCT

> MIRPR 2026-2027 · Universitatea Babeș-Bolyai · Proiect 6 (coordonator: dna. dr. Mihaela Hedeșiu)

## Echipa
| Nume | Specializare / grupa | GitHub | Metoda asignată |
|------|----------------------|--------|-----------------|
| ...  | ...                  | @...   | ex. 3D nnU-Net / ResUNet |
| ...  | ...                  | @...   | ex. 3D TransUNet (hibrid CNN-Transformer) |
| ...  | ...                  | @...   | ex. clasificare ViT/CNN pe ROI (dacă sunteți 3) |

## Problema abordată
Descriere scurtă (2-4 propoziții): ce se rezolvă, de ce e nevoie de AI, cine sunt utilizatorii
(stomatologi, radiologi, chirurgi OMF), ce intră (volum CBCT) și ce iese (mască de segmentare, clasa leziunii, hartă de explicabilitate).

## Schema soluției
![Schema soluției](documentation/report/Fig/graphical_abstract.png)
<!-- înlocuiți cu diagrama voastră: CBCT -> preprocesare -> segmentare -> ROI -> clasificare -> explicații -> aplicație -->

## Date
- DOLCHID (figshare, vezi `software-solution/data/README.md`)
- Licență și reguli etice/GDPR: ...

## Rezultate (se completează pe parcurs)
| Model | Dice | IoU | F1 (clasificare) | Timp inferență | Etapa |
|-------|------|-----|------------------|----------------|-------|

## Structură repo
- `software-solution/` cod, notebook-uri, aplicație, date
- `documentation/` raport LaTeX, slide-uri, link teaser

## Rulare rapidă
```bash
cd software-solution
pip install -r requirements.txt
jupyter lab notebooks/
```

## Teaser
Link YouTube: ...
