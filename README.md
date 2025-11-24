# TBA4565 Module GPS

This repo contains my solution to the two projects in TBA4565 Module GPS. 

> Project1: GPS Absolute Point Positioning with code pseudorange.

> Project2: Accurate Relative Positioning with Carrier Phases

---

## 1) Prerequisites

* **Python** 
* **Git**

---

## 2) How to run

```bash
# 1) Clone the repo
git clone https://github.com/mortened/TBA4565-GPS.git
cd TBA4565-GPS

# 2) Create & activate a virtualenv 
python3 -m venv .venv

# 3) Activate the virtual environment

# macOS/linux:
source .venv/bin/activate

# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Windows CMD:
.venv\Scripts\activate.bat

# 4) Install project dependencies
pip install -r requirements.txt

# 5) For project 1, run (in order)
python Project1/satellites.py
python Project2/receiver.py

# 5) For project 2, run 
python Project2/main.py
```

---
