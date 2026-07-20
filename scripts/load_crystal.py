from dotenv import load_dotenv
import os
from mp_api.client import MPRester

load_dotenv()

api_key = os.getenv("MP_API_KEY")
if not api_key:
    raise ValueError("MP_API_KEY not found in environment. Set it in your .env file.")

with MPRester(api_key) as mpr:
    structure = mpr.get_structure_by_material_id("mp-149")

print(structure)
print(structure.lattice)

for site in structure:
    print(site.specie, site.frac_coords)
