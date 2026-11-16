# saved/

Copies of training outputs that you chose to keep and share. Everything
under this folder is **tracked by git** (all other generated data is ignored).

Created by the **산출물 저장 / Save outputs** button on the training screen
(or "복사본 저장 / Save a copy" in the saved-model dialog):

| Folder | Content |
| --- | --- |
| `datasets/<name>/` | a collected training dataset (Parquet shards, `ABOUT.md`) |
| `modules/<name>/` | intent + emotion modules of one run (`intent.phase2model`, `emotion.phase2model`) |
| `main_ai/<name>.bangmodel` | a trained main AI model |
| `reports/<name>.json` | the training report of the saved run |

Every file carries its metadata (kind, creation time, initial conditions,
free-text notes). Open `ABOUT.md` or `<file>.about.md` to read it.

There is no Git LFS. GitHub rejects files above 100 MB and warns above 50 MB;
the save dialog warns before writing such a file.
