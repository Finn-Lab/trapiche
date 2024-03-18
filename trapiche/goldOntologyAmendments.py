# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01.00.00_goldOntologyAmendments.ipynb.

# %% auto 0
__all__ = ['TAG', 'DATA_DIR', 'TMP_DIR', 'biome_herarchy_file', 'biome_herarchy_df', 'biome_herarchy_amended_file',
           'biome_herarchy_dct', 'biome_original_graph_file', 'biome_original_graph', 'biome_graph_file', 'biome_graph',
           'HOST_CATEGORIES', 'ENV_CATEGORIES', 'ENG_CATEGORIES', 'gold_categories']

# %% ../nbs/01.00.00_goldOntologyAmendments.ipynb 3
import os
from . import config # fixed structure to import thisc
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import mysql.connector
from credentials import _credentials
from .utils import jsonCompressed

# %% ../nbs/01.00.00_goldOntologyAmendments.ipynb 4
TAG = 'goldOntologyAmendments'

# %% ../nbs/01.00.00_goldOntologyAmendments.ipynb 5
DATA_DIR = f"{config.datadir}/{TAG}"
TMP_DIR = f"{DATA_DIR}/temp"
os.makedirs(TMP_DIR,exist_ok=True)

# %% ../nbs/01.00.00_goldOntologyAmendments.ipynb 8
biome_herarchy_file = f"{DATA_DIR}/biome_herarchy_df.tsv"

# %% ../nbs/01.00.00_goldOntologyAmendments.ipynb 11
biome_herarchy_df = pd.read_csv(biome_herarchy_file,sep='\t')

# %% ../nbs/01.00.00_goldOntologyAmendments.ipynb 17
biome_herarchy_amended_file = f"{DATA_DIR}/biome_herarchy_amended.json.gz"

# %% ../nbs/01.00.00_goldOntologyAmendments.ipynb 20
biome_herarchy_dct = jsonCompressed.read(biome_herarchy_amended_file)

# %% ../nbs/01.00.00_goldOntologyAmendments.ipynb 23
biome_original_graph_file = f"{DATA_DIR}/biome_original_graph.graphml"

# %% ../nbs/01.00.00_goldOntologyAmendments.ipynb 25
biome_original_graph = nx.read_graphml(biome_original_graph_file)

# %% ../nbs/01.00.00_goldOntologyAmendments.ipynb 28
biome_graph_file = f"{DATA_DIR}/biome_graph.graphml"

# %% ../nbs/01.00.00_goldOntologyAmendments.ipynb 30
biome_graph = nx.read_graphml(biome_graph_file)

# %% ../nbs/01.00.00_goldOntologyAmendments.ipynb 33
HOST_CATEGORIES = {
    "Taxonomy Clade": [
        "Algae",
        "Amphibia",
        "Animal",
        "Annelida",
        "Arthropoda",
        "Birds",
        "Cnidaria",
        "Echinodermata",
        "Fish",
        "Fungi",
        "Human",
        "Insecta",
        "Invertebrates",
        "Mammals",
        "Microbial",
        "Mollusca",
        "Plants",
        "Reptile",
        "Spiralia",
        "Tunicates",
        "Epiphytes",
        "Endosymbionts",
        "Protists",
        "Protozoa",
        "Porifera",
        "Microbial",
        "Endosymbionts",
        "Brown Algae",
        "Green algae",
        "Red algae",
        "Ectosymbionts ",
        "Bacteria",
        "Dinoflagellates",
        "Bryozoans",
        "Cnidaria",
        "Echinodermata",
        "Ascidians",
        "Endophytes",
        "Gymnolaemates",
        "Coral",
        "Sea Urchin"
        "Ectosymbionts",
        "Epibionts",
    ],
    "Functional System": [
        "Digestive system",
        "Excretory system",
        "Circulatory system",
        "Reproductive system",
        "Respiratory system",
        "Lymphatic",
        "Nervous system",
        "Fossil",
        "Rhizosphere",
        "Lympathic system"
        "Female",
    ],
    "Body Part": [
        "Oral cavity",
        "Skin",
        "Integument",
        "Oral",
        "Venom gland",
        "Phylloplane",
        "Rhizome",
        "Rhizoplane",
        "Trophosome",
        "Root",
        "Gastrointestinal tract",
        "Lymph nodes",
        "Vagina",
        "Urethra",
        "Nasopharyngeal",
        "Pulmonary system",
        "Axilla",
        "Medial distal leg",
        "Naris",
        "Umbilicus",
        "Volar forearm",
        "retroauricular crease",
        "Venom gland",
        "Lungs",
        "Brain",
        "Gills",
        "Rumen",
        "Lung",
        "Rectum",
        "Foregut",
        "anterior nares",
        "posterior fornix",
        "Periodontal pockets",
        "P1 segment",
        "Midpoint vagina",
        "Midgut",
        "Cecum",
        "hard palate",
        "Kidneys",
        "Venous leg ulcers",
        "Sigmoid colon",
        "Stomach",
        "P3 segment",
        "tongue dorsum",
        "Nasal cavity",
        "Throat",
        "Hindgut",
        "Thoracic segments",
        "Small intestine",
        "Cuticle",
        "Digestive tube",
        "Glands",
        "Shell",
        "Large intestine",
        "Ceca",
        "Gut",
        "Pharynx",
        "Endoperitrophic space",
        "Proctodeal segment",
        "Introitus",
        "Trachea",
        "Duodenal",
        "Intestine",
    ],
    "Material": [
        "Saliva",
        "Soil",
        "Fecal",
        "Bone",
        "Milk",
        "Venom",
        "Blood",
        "Slime",
        "Cerebrospinal fluid",
        "Urine",
        "Lumen",
        "Sputum",
        "Feces",
        "Buccal mucosa",
        "Palatine tonsils",
        "buccal mucosa",
        "Subgingival plaque",
        "Garden dump",
        "Attached Keratinized gingiva",
        "Intracellular",
        "Egg capsule",
    ],
    "Enrichment": [
        "Viriome",
        "Fungus gallery",
        "Bacteriomes",
        "Fungus garden",
        "Symbiotic fungal gardens and galleries",
    ],
}

# %% ../nbs/01.00.00_goldOntologyAmendments.ipynb 35
ENV_CATEGORIES = {
    "primary":[
        'root:Environmental:Air',
        'root:Environmental:Aquatic:Aquaculture',
        'root:Environmental:Aquatic:Estuary',
        'root:Environmental:Aquatic:Freshwater',
        'root:Environmental:Aquatic:Lentic',
        'root:Environmental:Aquatic:Marine',
        'root:Environmental:Aquatic:Meromictic lake',
        'root:Environmental:Aquatic:Non-marine Saline and Alkaline',
        'root:Environmental:Aquatic:Sediment',
        'root:Environmental:Aquatic:Thermal springs',
        'root:Environmental:Terrestrial:Agricultural field',
        'root:Environmental:Terrestrial:Asphalt lakes',
        'root:Environmental:Terrestrial:Deep subsurface',
        'root:Environmental:Terrestrial:Geologic',
        'root:Environmental:Terrestrial:Oil reservoir',
        'root:Environmental:Terrestrial:Rock-dwelling (subaerial biofilm)',
        'root:Environmental:Terrestrial:Soil',
        'root:Environmental:Terrestrial:Volcanic',
    ]
}


# %% ../nbs/01.00.00_goldOntologyAmendments.ipynb 36
ENG_CATEGORIES = {
    "primary":[
        'root:Engineered:Biogas plant',
        'root:Engineered:Bioreactor',
        'root:Engineered:Bioremediation',
        'root:Engineered:Biotransformation',
        'root:Engineered:Built environment',
        'root:Engineered:Food production',
        'root:Engineered:Industrial production',
        'root:Engineered:Lab Synthesis',
        'root:Engineered:Lab enrichment',
        'root:Engineered:Modeled',
        'root:Engineered:Solid waste',
        'root:Engineered:Wastewater',
    ]
}


# %% ../nbs/01.00.00_goldOntologyAmendments.ipynb 39
gold_categories = {
    "root:Engineered":ENG_CATEGORIES,
    "root:Environmental":ENV_CATEGORIES,
    "root:Host-associated":HOST_CATEGORIES,
}
