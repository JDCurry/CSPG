"""
Testing ZINC20 Small Molecules in SCI Scar Simulator V2
========================================================

This script demonstrates how to test your TIMP2-enhancing compounds
from the ZINC20 screen in the V2 simulator.

Workflow:
1. Define compound properties (from docking/binding assays)
2. Translate to model parameters
3. Run simulation
4. Compare to control
5. Identify synergies (e.g., with chABC)

"""

import numpy as np
import matplotlib.pyplot as plt
from sci_scar_simulator_v2 import SCIScarSimulatorV2


class ZINC20Compound:
    """
    Represents a small molecule from your ZINC20 screen.
    
    Attributes
    ----------
    name : str
        ZINC ID or compound name
    binding_affinity : float
        Predicted binding affinity to TIMP2 (kJ/mol)
    k_on_enhancement : float
        Fold-increase in TIMP2-MMP binding rate
    production_boost : float
        Direct TIMP2 production enhancement
    """
    
    def __init__(self, name, binding_affinity, k_on_enhancement=1.0, 
                 production_boost=0.0):
        self.name = name
        self.binding_affinity = binding_affinity
        self.k_on_enhancement = k_on_enhancement
        self.production_boost = production_boost
    
    def __repr__(self):
        return (f"ZINC20Compound('{self.name}', "
                f"ΔG={self.binding_affinity:.1f} kJ/mol, "
                f"k_on×{self.k_on_enhancement:.2f}, "
                f"prod+{self.production_boost:.3f})")


# ============================================================================
# EXAMPLE COMPOUNDS (Replace with your actual ZINC20 hits)
# ============================================================================

# Example 1: Moderate binder, primarily affects TIMP2-MMP interaction
compound_A = ZINC20Compound(
    name="ZINC000012345678",
    binding_affinity=-35.2,  # kJ/mol (from Vina or similar)
    k_on_enhancement=1.6,    # 60% increase in binding rate
    production_boost=0.003   # Modest production boost
)

# Example 2: Strong binder, dual mechanism
compound_B = ZINC20Compound(
    name="ZINC000087654321",
    binding_affinity=-42.8,  # Stronger binder
    k_on_enhancement=2.2,    # 120% increase in binding
    production_boost=0.008   # Significant production boost
)

# Example 3: Production enhancer (e.g., transcriptional upregulator)
compound_C = ZINC20Compound(
    name="ZINC000011112222",
    binding_affinity=-28.5,  # Weaker direct binding
    k_on_enhancement=1.1,    # Minimal binding enhancement
    production_boost=0.012   # Strong production effect
)


# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def test_compound(compound, injury_type='contusion', duration_days=28, 
                  dose_time_hours=0):
    """
    Test a single compound in the SCI scar model.
    
    Parameters
    ----------
    compound : ZINC20Compound
        Compound to test
    injury_type : str
        'contusion' or 'laceration'
    duration_days : int
        Simulation duration
    dose_time_hours : float
        When to administer compound (0 = immediate)
    
    Returns
    -------
    sim : SCIScarSimulatorV2
        Completed simulation
    """
    print(f"\n{'='*70}")
    print(f" Testing: {compound.name}")
    print(f"{'='*70}")
    print(f"  Binding affinity: {compound.binding_affinity:.1f} kJ/mol")
    print(f"  k_on enhancement: {compound.k_on_enhancement:.2f}×")
    print(f"  TIMP2 boost: +{compound.production_boost:.3f}")
    print(f"  Dose time: t={dose_time_hours}h")
    
    # Create simulator
    sim = SCIScarSimulatorV2(grid_size=64, dt=0.5, genotype='typical')
    sim.create_injury(center=(32, 32), radius=8, injury_type=injury_type)
    
    # Run to dose time
    if dose_time_hours > 0:
        sim.run(duration_hours=dose_time_hours, verbose=False)
    
    # Apply compound
    baseline_k_on = 0.5  # From default parameters
    sim.set_parameters(
        k_on=baseline_k_on * compound.k_on_enhancement,
        small_mol_boost=compound.production_boost
    )
    
    # Run remainder of simulation
    remaining_hours = duration_days * 24 - dose_time_hours
    sim.run(duration_hours=remaining_hours, verbose=False)
    
    # Print results
    final_S = sim.history['S_injury'][-1]
    final_M = sim.history['M_injury'][-1]
    final_T = sim.history['T_injury'][-1]
    
    print(f"\n  Results (day {duration_days}):")
    print(f"    CSPG level: {final_S:.2f}×")
    print(f"    MMP activity: {final_M:.2f}")
    print(f"    TIMP2 level: {final_T:.2f}")
    
    return sim


def compare_compounds(compounds, include_control=True):
    """
    Compare multiple compounds head-to-head.
    
    Parameters
    ----------
    compounds : list of ZINC20Compound
        Compounds to compare
    include_control : bool
        Include untreated control
    
    Returns
    -------
    results : dict
        Simulation results for each compound
    """
    print("\n" + "="*70)
    print(" COMPOUND COMPARISON")
    print("="*70)
    
    results = {}
    
    # Control
    if include_control:
        print("\nRunning control (no treatment)...")
        control_sim = SCIScarSimulatorV2(grid_size=64, dt=0.5, genotype='typical')
        control_sim.create_injury(center=(32, 32), radius=8, injury_type='contusion')
        control_sim.run(duration_hours=672, verbose=False)
        results['Control'] = control_sim
    
    # Compounds
    for compound in compounds:
        sim = test_compound(compound, duration_days=28, dose_time_hours=0)
        results[compound.name] = sim
    
    # Generate comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    colors = ['#333333', '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    
    # Panel A: CSPG time course
    ax = axes[0]
    for i, (name, sim) in enumerate(results.items()):
        t_days = np.array(sim.history['time']) / 24
        S = np.array(sim.history['S_injury'])
        lw = 2.5 if name == 'Control' else 2.0
        ls = '--' if name == 'Control' else '-'
        ax.plot(t_days, S, color=colors[i % len(colors)], lw=lw, ls=ls, label=name)
    
    ax.axhline(4.0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('CSPG (fold change)', fontsize=11)
    ax.set_title('A. Scar Formation', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 5)
    
    # Panel B: Final CSPG levels (bar chart)
    ax = axes[1]
    names = list(results.keys())
    final_S = [results[name].history['S_injury'][-1] for name in names]
    bars = ax.bar(range(len(names)), final_S, color=colors[:len(names)],
                   edgecolor='black', linewidth=1.2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Final CSPG (day 28)', fontsize=11)
    ax.set_title('B. Endpoint Comparison', fontsize=12, fontweight='bold')
    ax.axhline(4.0, color='gray', ls=':', alpha=0.5)
    ax.set_ylim(0, 5)
    
    # Add value labels
    for bar, val in zip(bars, final_S):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.1, f'{val:.2f}×',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Panel C: Scar reduction vs control
    ax = axes[2]
    if include_control:
        control_S = results['Control'].history['S_injury'][-1]
        compound_names = [n for n in names if n != 'Control']
        reductions = [(1 - results[n].history['S_injury'][-1] / control_S) * 100 
                      for n in compound_names]
        
        bars = ax.bar(range(len(compound_names)), reductions,
                      color=colors[1:len(compound_names)+1],
                      edgecolor='black', linewidth=1.2)
        ax.set_xticks(range(len(compound_names)))
        ax.set_xticklabels(compound_names, rotation=45, ha='right')
        ax.set_ylabel('Scar Reduction (%)', fontsize=11)
        ax.set_title('C. Efficacy vs Control', fontsize=12, fontweight='bold')
        ax.axhline(0, color='gray', ls='-', lw=1)
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, reductions):
            ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('ZINC20 Compound Comparison in SCI Scar Model',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('zinc20_compound_comparison.png', dpi=200, bbox_inches='tight')
    print("\n✓ Saved: zinc20_compound_comparison.png")
    plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    if include_control:
        control_S = results['Control'].history['S_injury'][-1]
        print(f"\nControl (no treatment): {control_S:.2f}×")
        print("\nCompound efficacy:")
        for name in names:
            if name != 'Control':
                final_S = results[name].history['S_injury'][-1]
                reduction = (1 - final_S / control_S) * 100
                print(f"  {name}: {final_S:.2f}× ({reduction:+.1f}%)")
    
    return results


def test_synergy_with_chABC(compound, chABC_dose=0.006):
    """
    Test compound + chABC combination therapy.
    
    Parameters
    ----------
    compound : ZINC20Compound
        Small molecule to test
    chABC_dose : float
        chABC degradation rate (gamma_therapy)
    
    Returns
    -------
    results : dict
        Results for all treatment arms
    """
    print("\n" + "="*70)
    print(f" SYNERGY TEST: {compound.name} + chABC")
    print("="*70)
    
    treatments = {
        'Control': {},
        f'{compound.name}': {
            'k_on': 0.5 * compound.k_on_enhancement,
            'small_mol_boost': compound.production_boost
        },
        'chABC': {
            'gamma_therapy': chABC_dose
        },
        'Combined': {
            'k_on': 0.5 * compound.k_on_enhancement,
            'small_mol_boost': compound.production_boost,
            'gamma_therapy': chABC_dose
        }
    }
    
    results = {}
    
    for name, params in treatments.items():
        print(f"\nRunning {name}...")
        sim = SCIScarSimulatorV2(grid_size=64, dt=0.5, genotype='typical')
        sim.set_parameters(**params)
        sim.create_injury(center=(32, 32), radius=8, injury_type='contusion')
        sim.run(duration_hours=672, verbose=False)
        results[name] = sim
        
        final_S = sim.history['S_injury'][-1]
        print(f"  Final CSPG: {final_S:.2f}×")
    
    # Plot synergy
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#333333', '#E74C3C', '#3498DB', '#2ECC71']
    
    # Time course
    ax = axes[0]
    for i, (name, sim) in enumerate(results.items()):
        t_days = np.array(sim.history['time']) / 24
        S = np.array(sim.history['S_injury'])
        lw = 2.5 if name == 'Control' else 2.0
        ls = '--' if name == 'Control' else '-'
        ax.plot(t_days, S, color=colors[i], lw=lw, ls=ls, label=name)
    
    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('CSPG (fold change)', fontsize=11)
    ax.set_title('CSPG Time Course', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 5)
    
    # Synergy analysis
    ax = axes[1]
    names = list(results.keys())
    final_S = [results[name].history['S_injury'][-1] for name in names]
    bars = ax.bar(range(len(names)), final_S, color=colors,
                   edgecolor='black', linewidth=1.2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=0)
    ax.set_ylabel('Final CSPG (day 28)', fontsize=11)
    ax.set_title('Endpoint Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 5)
    
    # Add value labels
    for bar, val in zip(bars, final_S):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.1, f'{val:.2f}×',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle(f'Synergy Analysis: {compound.name} + chABC',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('zinc20_synergy_analysis.png', dpi=200, bbox_inches='tight')
    print("\n✓ Saved: zinc20_synergy_analysis.png")
    plt.show()
    
    # Test for synergy
    control_S = results['Control'].history['S_injury'][-1]
    compound_S = results[list(results.keys())[1]].history['S_injury'][-1]
    chABC_S = results['chABC'].history['S_injury'][-1]
    combined_S = results['Combined'].history['S_injury'][-1]
    
    # Additive prediction
    compound_effect = control_S - compound_S
    chABC_effect = control_S - chABC_S
    additive_prediction = control_S - (compound_effect + chABC_effect)
    
    # Actual combined effect
    synergy = additive_prediction - combined_S
    
    print("\n" + "-"*70)
    print("Synergy Analysis:")
    print("-"*70)
    print(f"  Control: {control_S:.2f}×")
    print(f"  {compound.name} alone: {compound_S:.2f}× (Δ = {compound_effect:.2f})")
    print(f"  chABC alone: {chABC_S:.2f}× (Δ = {chABC_effect:.2f})")
    print(f"  Additive prediction: {additive_prediction:.2f}×")
    print(f"  Actual combined: {combined_S:.2f}×")
    print(f"  Synergy: {synergy:.2f}× {'(SYNERGISTIC)' if synergy > 0.1 else '(ADDITIVE)'}")
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" ZINC20 COMPOUND TESTING SUITE")
    print("="*70)
    
    # Define your compounds (replace with actual data)
    compounds = [compound_A, compound_B, compound_C]
    
    # Test 1: Individual compound performance
    print("\n" + "="*70)
    print(" TEST 1: Individual Compound Screening")
    print("="*70)
    results = compare_compounds(compounds, include_control=True)
    
    # Test 2: Synergy with chABC (use best compound)
    best_compound = compound_B  # Replace with actual best hit
    print("\n" + "="*70)
    print(" TEST 2: Synergy with chABC")
    print("="*70)
    synergy_results = test_synergy_with_chABC(best_compound, chABC_dose=0.006)
    
    # Test 3: Genotype interaction (A;A variant)
    print("\n" + "="*70)
    print(" TEST 3: Genotype × Compound Interaction")
    print("="*70)
    
    for genotype in ['typical', 'A;A']:
        print(f"\n{genotype.upper()} genotype:")
        sim = SCIScarSimulatorV2(grid_size=64, dt=0.5, genotype=genotype)
        sim.set_parameters(
            k_on=0.5 * best_compound.k_on_enhancement,
            small_mol_boost=best_compound.production_boost
        )
        sim.create_injury(center=(32, 32), radius=8, injury_type='contusion')
        sim.run(duration_hours=672, verbose=False)
        
        final_S = sim.history['S_injury'][-1]
        print(f"  Final CSPG: {final_S:.2f}×")
    
    print("\n" + "="*70)
    print(" TESTING COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - zinc20_compound_comparison.png")
    print("  - zinc20_synergy_analysis.png")
    print("\n✓ All tests complete!\n")
