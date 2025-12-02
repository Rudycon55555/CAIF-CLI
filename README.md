# CAIF‑CLI (Code AI Fast – Command Line Interface)

CAIF‑CLI is a **dependency‑free AI runtime** written in pure Python.  
It lets you **code, train, and export working AI models** using simple `.caif` scripts — no external libraries required.

---

## ✨ Features
- **Dependency‑free**: Uses only Python’s standard library (`sys`, `os`, `socket`, `math`, `random`, `urllib`).
- **Command‑driven workflow**: Write `.caif` scripts with human‑readable commands like `SET PARAMETER`, `IMPORT DATA`, `TRAIN FOR`, `EXPORT MODEL`.
- **Real training loop**: Implements a single‑hidden‑layer neural network with forward/backward passes in pure Python.
- **Dataset parsing**: Reads numeric input/output pairs from JSONL files or URLs.
- **Export models**: Produces standalone `.py` files containing learned weights and a working `predict()` function.
- **Tool/XP connections**: Supports `CONNECT TO XP` (network sockets) and `CONNECT TO TC` (tool calls) for integration.

---

## 📦 Installation

Clone the repo and run directly with Python 3.8+:

```bash
git clone https://github.com/yourusername/CAIF-CLI.git
cd CAIF-CLI
python CAIF-CLI.py [replace these brackets and this text inside with the filepath (relative or absolute) to a CAIF File, which ends in .caif]
```
Then to use the model you just made, navigate to the directory of it (~/CAIF-CLI), then run

```bash
python [replace these brackets and this text inside with the filepath (relative or absolute) to the exported Python File]
```
