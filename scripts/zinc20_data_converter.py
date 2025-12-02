"""
ZINC20 Data Converter: From Docking Output to Simulator Parameters
===================================================================

Converts your molecular docking results (PDBQT, PDB, SMILES) into 
parameters for the SCI Scar Simulator V2.

Workflow:
1. Parse PDBQT files → extract binding affinity
2. Convert kcal/mol → kJ/mol
3. Map binding affinity → k_on_enhancement (empirical scaling)
4. Optionally parse SMILES for druglikeness checks

"""

import os
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from pathlib import Path


@dataclass
class ParsedCompound:
    """
    Compound data extracted from docking outputs.
    
    Attributes
    ----------
    zinc_id : str
        ZINC database ID or compound name
    binding_affinity_kcal : float
        Binding affinity in kcal/mol (from Vina)
    binding_affinity_kj : float
        Binding affinity in kJ/mol (converted)
    smiles : str, optional
        SMILES string if available
    pdb_file : str, optional
        Path to PDB file if available
    pdbqt_file : str, optional
        Path to PDBQT file
    k_on_enhancement : float
        Estimated fold-increase in TIMP2-MMP binding rate
    production_boost : float
        Estimated TIMP2 production enhancement
    notes : str
        Any additional notes or warnings
    """
    zinc_id: str
    binding_affinity_kcal: float
    binding_affinity_kj: float = field(init=False)
    smiles: Optional[str] = None
    pdb_file: Optional[str] = None
    pdbqt_file: Optional[str] = None
    k_on_enhancement: float = field(init=False)
    production_boost: float = 0.003  # Conservative default
    notes: str = ""
    
    def __post_init__(self):
        # Convert kcal/mol to kJ/mol
        self.binding_affinity_kj = self.binding_affinity_kcal * 4.184
        
        # Estimate k_on_enhancement from binding affinity
        self.k_on_enhancement = self._estimate_k_on_enhancement()
    
    def _estimate_k_on_enhancement(self) -> float:
        """
        Empirical mapping from binding affinity to k_on enhancement.
        
        Based on:
        - Tighter binding (more negative ΔG) → stronger TIMP2-MMP stabilization
        - Linear-log relationship between K_d and kinetic rates
        
        Calibration points (adjust based on your assay data):
        - -6 kcal/mol (weak): k_on × 1.1
        - -8 kcal/mol (moderate): k_on × 1.5
        - -10 kcal/mol (strong): k_on × 2.0
        - -12 kcal/mol (very strong): k_on × 2.5
        """
        affinity = self.binding_affinity_kcal
        
        # Piecewise linear mapping
        if affinity >= -6:
            # Weak binders - minimal effect
            return 1.0 + 0.05 * abs(affinity + 6)
        elif affinity >= -8:
            # Moderate binders
            return 1.1 + 0.2 * abs(affinity + 6)
        elif affinity >= -10:
            # Good binders
            return 1.5 + 0.25 * abs(affinity + 8)
        elif affinity >= -12:
            # Strong binders
            return 2.0 + 0.25 * abs(affinity + 10)
        else:
            # Very strong binders (cap at 3.0)
            return min(3.0, 2.5 + 0.1 * abs(affinity + 12))
    
    def to_zinc20_compound(self):
        """Convert to ZINC20Compound object for simulator."""
        # Import here to avoid circular dependency
        from test_zinc20_compounds import ZINC20Compound
        
        return ZINC20Compound(
            name=self.zinc_id,
            binding_affinity=self.binding_affinity_kj,
            k_on_enhancement=self.k_on_enhancement,
            production_boost=self.production_boost
        )
    
    def __repr__(self):
        return (f"ParsedCompound('{self.zinc_id}')\n"
                f"  Affinity: {self.binding_affinity_kcal:.2f} kcal/mol "
                f"({self.binding_affinity_kj:.1f} kJ/mol)\n"
                f"  k_on enhancement: {self.k_on_enhancement:.2f}×\n"
                f"  Production boost: {self.production_boost:.4f}")


# ============================================================================
# PDBQT PARSING
# ============================================================================

def parse_pdbqt_affinity(pdbqt_path: str) -> Tuple[float, int]:
    """
    Extract binding affinity from AutoDock Vina PDBQT output.
    
    Vina outputs look like:
    ```
    REMARK VINA RESULT:    -8.5      0.000      0.000
    ```
    
    Parameters
    ----------
    pdbqt_path : str
        Path to PDBQT file
    
    Returns
    -------
    best_affinity : float
        Best binding affinity (kcal/mol, negative = better)
    n_modes : int
        Number of binding modes found
    """
    affinities = []
    
    with open(pdbqt_path, 'r') as f:
        for line in f:
            if 'VINA RESULT' in line or 'RESULT' in line:
                # Parse: REMARK VINA RESULT:    -8.5      0.000      0.000
                parts = line.split()
                for i, part in enumerate(parts):
                    try:
                        val = float(part)
                        if val < 0:  # Binding affinities are negative
                            affinities.append(val)
                            break
                    except ValueError:
                        continue
    
    if not affinities:
        raise ValueError(f"No binding affinity found in {pdbqt_path}")
    
    return min(affinities), len(affinities)


def parse_pdbqt_batch(pdbqt_dir: str, pattern: str = "*.pdbqt") -> List[ParsedCompound]:
    """
    Parse multiple PDBQT files from a directory.
    
    Parameters
    ----------
    pdbqt_dir : str
        Directory containing PDBQT files
    pattern : str
        Glob pattern for file matching
    
    Returns
    -------
    compounds : list of ParsedCompound
        Parsed compound data
    """
    from glob import glob
    
    pdbqt_files = glob(os.path.join(pdbqt_dir, pattern))
    compounds = []
    
    print(f"\nParsing {len(pdbqt_files)} PDBQT files from {pdbqt_dir}...")
    
    for pdbqt_path in pdbqt_files:
        try:
            affinity, n_modes = parse_pdbqt_affinity(pdbqt_path)
            
            # Extract ZINC ID from filename
            filename = os.path.basename(pdbqt_path)
            zinc_id = filename.replace('.pdbqt', '').replace('_out', '')
            
            compound = ParsedCompound(
                zinc_id=zinc_id,
                binding_affinity_kcal=affinity,
                pdbqt_file=pdbqt_path,
                notes=f"{n_modes} binding modes"
            )
            compounds.append(compound)
            
        except Exception as e:
            print(f"  Warning: Could not parse {pdbqt_path}: {e}")
    
    print(f"✓ Successfully parsed {len(compounds)} compounds")
    return compounds


# ============================================================================
# PDB PARSING
# ============================================================================

def parse_pdb_coordinates(pdb_path: str) -> np.ndarray:
    """
    Extract ligand heavy atom coordinates from PDB file.
    
    Useful for:
    - Calculating binding site center
    - Measuring pose RMSD
    - Visualization
    
    Parameters
    ----------
    pdb_path : str
        Path to PDB file
    
    Returns
    -------
    coords : np.ndarray
        (N, 3) array of atom coordinates
    """
    coords = []
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except ValueError:
                    continue
    
    return np.array(coords)


def get_binding_site_center(pdb_path: str) -> Tuple[float, float, float]:
    """Calculate geometric center of ligand atoms."""
    coords = parse_pdb_coordinates(pdb_path)
    if len(coords) == 0:
        raise ValueError(f"No coordinates found in {pdb_path}")
    center = coords.mean(axis=0)
    return tuple(center)


# ============================================================================
# SMILES UTILITIES
# ============================================================================

def parse_smiles_file(smiles_path: str) -> Dict[str, str]:
    """
    Parse SMILES file (tab-separated: SMILES\tID).
    
    Parameters
    ----------
    smiles_path : str
        Path to SMILES file
    
    Returns
    -------
    smiles_dict : dict
        {zinc_id: smiles_string}
    """
    smiles_dict = {}
    
    with open(smiles_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                smiles, zinc_id = parts[0], parts[1]
                smiles_dict[zinc_id] = smiles
            elif len(parts) == 1:
                # Just SMILES, no ID
                smiles_dict[f"compound_{len(smiles_dict)}"] = parts[0]
    
    return smiles_dict


def compute_druglikeness(smiles: str) -> Dict[str, float]:
    """
    Compute Lipinski's Rule of Five descriptors.
    
    Requires RDKit (optional dependency).
    
    Parameters
    ----------
    smiles : str
        SMILES string
    
    Returns
    -------
    descriptors : dict
        MW, LogP, HBD, HBA, violations
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        violations = sum([
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10
        ])
        
        return {
            "MW": mw,
            "LogP": logp,
            "HBD": hbd,
            "HBA": hba,
            "Lipinski_violations": violations
        }
        
    except ImportError:
        return {"error": "RDKit not installed"}


# ============================================================================
# MAIN CONVERSION PIPELINE
# ============================================================================

def convert_docking_results(
    pdbqt_dir: Optional[str] = None,
    pdbqt_files: Optional[List[str]] = None,
    smiles_file: Optional[str] = None,
    affinity_cutoff: float = -7.0,
    output_csv: Optional[str] = None
) -> List[ParsedCompound]:
    """
    Full pipeline: PDBQT → ParsedCompound → ready for simulator.
    
    Parameters
    ----------
    pdbqt_dir : str, optional
        Directory containing PDBQT files
    pdbqt_files : list, optional
        Explicit list of PDBQT file paths
    smiles_file : str, optional
        SMILES file to merge with docking results
    affinity_cutoff : float
        Only keep compounds better than this (kcal/mol)
    output_csv : str, optional
        Save results to CSV
    
    Returns
    -------
    compounds : list of ParsedCompound
        Filtered, converted compounds
    """
    compounds = []
    
    # Parse PDBQT files
    if pdbqt_dir:
        compounds.extend(parse_pdbqt_batch(pdbqt_dir))
    
    if pdbqt_files:
        for pdbqt_path in pdbqt_files:
            try:
                affinity, n_modes = parse_pdbqt_affinity(pdbqt_path)
                filename = os.path.basename(pdbqt_path)
                zinc_id = filename.replace('.pdbqt', '').replace('_out', '')
                
                compound = ParsedCompound(
                    zinc_id=zinc_id,
                    binding_affinity_kcal=affinity,
                    pdbqt_file=pdbqt_path,
                    notes=f"{n_modes} binding modes"
                )
                compounds.append(compound)
            except Exception as e:
                print(f"Warning: {pdbqt_path}: {e}")
    
    # Merge SMILES if available
    if smiles_file and os.path.exists(smiles_file):
        smiles_dict = parse_smiles_file(smiles_file)
        for compound in compounds:
            if compound.zinc_id in smiles_dict:
                compound.smiles = smiles_dict[compound.zinc_id]
    
    # Filter by affinity cutoff
    n_before = len(compounds)
    compounds = [c for c in compounds if c.binding_affinity_kcal <= affinity_cutoff]
    n_after = len(compounds)
    
    print(f"\nFiltering by affinity ≤ {affinity_cutoff} kcal/mol:")
    print(f"  {n_before} → {n_after} compounds")
    
    # Sort by affinity (best first)
    compounds.sort(key=lambda c: c.binding_affinity_kcal)
    
    # Save to CSV if requested
    if output_csv:
        save_compounds_csv(compounds, output_csv)
    
    return compounds


def save_compounds_csv(compounds: List[ParsedCompound], output_path: str):
    """Save parsed compounds to CSV for review."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'ZINC_ID', 
            'Affinity_kcal', 
            'Affinity_kJ', 
            'k_on_enhancement',
            'production_boost',
            'SMILES',
            'Notes'
        ])
        
        for c in compounds:
            writer.writerow([
                c.zinc_id,
                f"{c.binding_affinity_kcal:.2f}",
                f"{c.binding_affinity_kj:.1f}",
                f"{c.k_on_enhancement:.2f}",
                f"{c.production_boost:.4f}",
                c.smiles or "",
                c.notes
            ])
    
    print(f"✓ Saved to {output_path}")


# ============================================================================
# QUICK ENTRY FUNCTIONS
# ============================================================================

def from_affinity(zinc_id: str, affinity_kcal: float, 
                  production_boost: float = 0.003) -> ParsedCompound:
    """
    Quick creation from just affinity value.
    
    Use this when you have docking scores but no files.
    
    Example
    -------
    >>> compound = from_affinity("ZINC000012345", -9.2)
    >>> print(compound)
    """
    return ParsedCompound(
        zinc_id=zinc_id,
        binding_affinity_kcal=affinity_kcal,
        production_boost=production_boost
    )


def from_affinity_batch(data: List[Tuple[str, float]]) -> List[ParsedCompound]:
    """
    Quick creation from list of (zinc_id, affinity) tuples.
    
    Example
    -------
    >>> compounds = from_affinity_batch([
    ...     ("ZINC000012345", -9.2),
    ...     ("ZINC000054321", -8.7),
    ...     ("ZINC000099999", -10.1),
    ... ])
    """
    return [from_affinity(zinc_id, affinity) for zinc_id, affinity in data]


# ============================================================================
# MANUAL ENTRY (for your specific compounds)
# ============================================================================

def enter_compounds_manually():
    """
    Interactive mode for entering compound data.
    
    Run this to input your ZINC20 screening hits.
    """
    print("\n" + "="*70)
    print(" MANUAL COMPOUND ENTRY")
    print("="*70)
    print("\nEnter your compound data. Type 'done' when finished.\n")
    
    compounds = []
    
    while True:
        zinc_id = input("ZINC ID (or 'done'): ").strip()
        if zinc_id.lower() == 'done':
            break
        
        try:
            affinity = float(input("Binding affinity (kcal/mol, e.g., -8.5): "))
        except ValueError:
            print("Invalid affinity, skipping...")
            continue
        
        smiles = input("SMILES (optional, press Enter to skip): ").strip() or None
        
        compound = ParsedCompound(
            zinc_id=zinc_id,
            binding_affinity_kcal=affinity,
            smiles=smiles
        )
        
        print(f"\n{compound}\n")
        compounds.append(compound)
    
    print(f"\n✓ Entered {len(compounds)} compounds")
    return compounds


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    
    # Example 1: From affinity values directly
    print("\n" + "="*70)
    print(" EXAMPLE 1: From affinity values")
    print("="*70)
    
    example_compounds = from_affinity_batch([
        ("ZINC000012345678", -8.5),
        ("ZINC000087654321", -9.8),
        ("ZINC000011112222", -7.2),
        ("ZINC000033334444", -10.5),
    ])
    
    for c in example_compounds:
        print(f"\n{c}")
    
    # Example 2: Show k_on enhancement scaling
    print("\n" + "="*70)
    print(" K_ON ENHANCEMENT SCALING")
    print("="*70)
    print("\nBinding affinity → k_on enhancement mapping:\n")
    
    test_affinities = [-5, -6, -7, -8, -9, -10, -11, -12, -13]
    for aff in test_affinities:
        c = from_affinity("test", aff)
        print(f"  {aff:4.0f} kcal/mol → k_on × {c.k_on_enhancement:.2f}")
    
    print("\n" + "="*70)
    print(" USAGE INSTRUCTIONS")
    print("="*70)
    print("""
    For your PDBQT files:
    ---------------------
    compounds = parse_pdbqt_batch('/path/to/pdbqt/directory/')
    
    For specific files:
    -------------------
    compounds = convert_docking_results(
        pdbqt_files=['compound1.pdbqt', 'compound2.pdbqt'],
        affinity_cutoff=-7.0
    )
    
    From affinity values (quickest):
    ---------------------------------
    compounds = from_affinity_batch([
        ("ZINC_ID_1", -9.2),
        ("ZINC_ID_2", -8.5),
    ])
    
    Interactive entry:
    ------------------
    compounds = enter_compounds_manually()
    
    Then test in simulator:
    -----------------------
    from test_zinc20_compounds import test_compound
    
    for compound in compounds:
        sim = test_compound(compound.to_zinc20_compound())
    """)
