# PROJECT NAME
# DATE
# PROJECT GOAL
# PROJECT FOLDER STRUCTURE
# ANY MAKEFILE INSTRUCTIONS

### Folder Structure
```
.
├ README.md                 - this readme file.
├── /data/                  - all data files put here
│   ├── proc/                   - processed files (e.g. CDO/NCO output, ROMS subsets)
│   └── raw/                    - raw data
├── /docs/                  - PDFs, manuals, etc
├── /notebooks/             - exploratory science in notebooks
│   
│   
└── /src                    - source code
    ├── analyses/               - repeatable analyses
    ├── ext/                    - source code from other people
    ├── functions/              - functions
    └── project/                - project code, final toolkit code, etc.
```
---
##### Python data science template
This `cookie cutter data science` template is built with python codes and is primarily designed for analysing big ocean data, for example from ROMS, satellite or other model/observational data.
###### Instructions:
1. `git clone` this repository with a new folder name of the new project.
2.  Update this `README.md` with a description of what this project is.
3.  Write exploratory science in the `notebooks/1.0-notebook-description.ipynb` folder.
4.  Add new functions to `src/functions/`
5.  Add new repeated analyses to `src/analyses/`
6.  Add new "final" source code (e.g. code for: making article plots, saving data, etc) to `src/final/`
7.  Push any new (and useful) functions back here `github.com/dgwyther/python-data-science-template`.
