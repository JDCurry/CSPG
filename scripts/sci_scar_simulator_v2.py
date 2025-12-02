"""
SCI Scar Formation Model V2: TIMP2/MMP Dynamics
================================================

Extends the original three-phase model with explicit biochemistry:
- MMP-2/MMP-9 proteolytic activity
- TIMP2 inhibition (genotype-dependent)
- TIMP2-MMP complex formation
- Therapeutic interventions (small molecules, chABC)

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class MMPValidationData:
    """Literature data for MMP-2/MMP-9 time courses in rodent SCI."""
    
    @staticmethod
    def mmp2_timecourse():
        """
        MMP-2 activity post-SCI (based on Noble et al. 2002, Wells et al. 2003).
        Peaks around day 3-7, remains elevated through day 14.
        """
        time_hours = np.array([0, 6, 12, 24, 48, 72, 168, 336, 672])
        time_days = time_hours / 24
        
        # Activity as fold-change over baseline (gelatinase assays)
        activity = np.array([1.0, 1.5, 2.2, 3.5, 5.8, 7.2, 6.0, 3.5, 2.0])
        std_dev = np.array([0.1, 0.3, 0.4, 0.6, 0.9, 1.1, 0.9, 0.6, 0.4])
        
        return time_hours, time_days, activity, std_dev
    
    @staticmethod
    def mmp9_timecourse():
        """
        MMP-9 activity post-SCI (based on Noble et al. 2002).
        Biphasic: acute spike (hours), then secondary wave (days 3-7).
        """
        time_hours = np.array([0, 4, 12, 24, 48, 72, 168, 336, 672])
        time_days = time_hours / 24
        
        # Activity as fold-change (more acute than MMP-2)
        activity = np.array([1.0, 8.5, 6.0, 4.0, 3.5, 5.5, 4.0, 2.0, 1.2])
        std_dev = np.array([0.1, 1.2, 1.0, 0.7, 0.6, 0.9, 0.7, 0.4, 0.2])
        
        return time_hours, time_days, activity, std_dev
    
    @staticmethod
    def timp2_timecourse():
        """
        TIMP2 expression post-SCI (based on Hsu et al. 2006).
        Counter-regulatory response: rises to inhibit MMPs.
        """
        time_hours = np.array([0, 24, 48, 72, 168, 336, 672])
        time_days = time_hours / 24
        
        # Expression as fold-change (mRNA/protein)
        expression = np.array([1.0, 1.8, 3.2, 4.5, 5.2, 4.0, 2.8])
        std_dev = np.array([0.1, 0.3, 0.5, 0.7, 0.8, 0.6, 0.5])
        
        return time_hours, time_days, expression, std_dev


class SCIScarSimulatorV2:
    """
    Spinal cord injury scar formation with explicit TIMP2/MMP dynamics.
    
    State Variables (2D fields)
    ---------------------------
    S : CSPG density (fold change over baseline)
    M : Active MMP pool (MMP-2 + MMP-9, normalized units)
    T : Free TIMP2 (normalized units)
    C : TIMP2-MMP complex (inactive, normalized units)
    
    Genotype Support
    ----------------
    - 'typical': Standard TIMP2 production
    - 'A;A': TIMP2 A;A variant (enhanced production, neuroplasticity)
    - 'small_mol': Exogenous TIMP2 enhancement via small molecule
    """
    
    def __init__(
        self,
        grid_size: int = 64,
        dt: float = 0.5,
        genotype: str = 'typical'
    ):
        """
        Initialize simulator.
        
        Parameters
        ----------
        grid_size : int
            Spatial grid dimension (grid_size × grid_size)
        dt : float
            Time step (hours)
        genotype : str
            'typical', 'A;A', or 'small_mol'
        """
        self.grid_size = grid_size
        self.dt = dt
        self.genotype = genotype
        
        # State variables
        self.S = np.ones((grid_size, grid_size))  # CSPG at baseline
        self.M = np.zeros((grid_size, grid_size))  # MMP initially low
        self.T = np.ones((grid_size, grid_size))   # TIMP2 at baseline
        self.C = np.zeros((grid_size, grid_size))  # No complex initially
        
        # Injury mask (will be set by create_injury)
        self.injury_mask = np.zeros((grid_size, grid_size), dtype=bool)
        self.penumbra_mask = np.zeros((grid_size, grid_size), dtype=bool)
        
        # Time tracking
        self.time = 0.0
        self.history = {
            'time': [],
            'S_injury': [],
            'S_penumbra': [],
            'M_injury': [],
            'T_injury': [],
            'C_injury': []
        }
        
        # Set default parameters
        self.set_default_parameters()
        self.set_genotype_parameters()
    
    def set_default_parameters(self):
        """Literature-based biochemical parameters."""
        
        # === CSPG Production (from V1) ===
        self.alpha_peak = 0.028      # Peak astrocytic production
        self.alpha_onset = 3         # Days to astrocyte activation
        self.alpha_width = 6         # Activation width
        self.alpha_acute = 0.012     # Acute inflammatory spike
        self.tau_acute = 24          # Acute decay (hours)
        self.eta_secondary = 0.018   # Secondary invasion wave
        self.eta_onset = 36          # Hours to invasion
        self.eta_width = 12.0        # Invasion width
        self.S_max = 5.0             # Maximum CSPG level
        
        # === MMP Dynamics ===
        # Production (injury-induced)
        self.beta_mmp2 = 0.008       # MMP-2 baseline production rate
        self.beta_mmp9_acute = 0.15  # MMP-9 acute spike
        self.beta_mmp9_chronic = 0.012  # MMP-9 chronic production
        self.tau_mmp9_acute = 12     # MMP-9 acute decay (hours)
        
        # Decay
        self.mu_mmp = 0.08           # MMP degradation rate (1/hours)
        
        # Diffusion
        self.D_mmp = 0.8             # MMP diffusion coefficient
        
        # === TIMP2 Dynamics ===
        self.T_production_basal = 0.005   # Baseline TIMP2 production
        self.T_production_injury = 0.015  # Injury-induced TIMP2
        self.T_onset = 24                 # Hours to TIMP2 upregulation
        self.T_width = 24                 # Upregulation width
        self.d_T = 0.04                   # TIMP2 degradation (1/hours)
        self.D_timp = 0.5                 # TIMP2 diffusion
        
        # === TIMP2-MMP Binding ===
        # Based on biochemical K_d ~ 0.5-5 nM for TIMP2-MMP2
        self.k_on = 0.5              # Binding rate (1/hours)
        self.k_off = 0.02            # Unbinding rate (1/hours)
        self.d_C = 0.01              # Complex degradation
        
        # === CSPG-MMP Interaction ===
        self.gamma_mmp = 0.15        # MMP-mediated CSPG degradation
        self.delta_feedback = 0.00025  # CSPG self-inhibition
        
        # === Therapeutic Interventions ===
        self.gamma_therapy = 0.0     # chABC degradation rate
        self.small_mol_boost = 0.0   # Small molecule TIMP2 enhancement
    
    def set_genotype_parameters(self):
        """Set genotype-specific TIMP2 production."""
        
        if self.genotype == 'A;A':
            # TIMP2 A;A variant: enhanced neuroplasticity
            self.T_production_injury = 0.020  # ~33% increase
            self.neuroplasticity_factor = 1.35
            print(f"🧬 Genotype: TIMP2 A;A (enhanced production)")
            
        elif self.genotype == 'small_mol':
            # Exogenous small molecule enhancement
            self.small_mol_boost = 0.008
            self.neuroplasticity_factor = 1.0
            print(f"💊 Small molecule TIMP2 enhancer active")
            
        else:  # typical
            self.neuroplasticity_factor = 1.0
            print(f"🧬 Genotype: Typical TIMP2")
    
    def set_parameters(self, **kwargs):
        """Update parameters dynamically."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown parameter '{key}'")
    
    def create_injury(
        self,
        center: Tuple[int, int],
        radius: float,
        injury_type: str = 'contusion'
    ):
        """
        Create injury site with core and penumbra.
        
        Parameters
        ----------
        center : tuple
            (x, y) coordinates of injury center
        radius : float
            Injury core radius (in grid units)
        injury_type : str
            'contusion' or 'laceration'
        """
        x, y = np.meshgrid(
            np.arange(self.grid_size),
            np.arange(self.grid_size),
            indexing='ij'
        )
        
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        # Core (direct tissue damage)
        self.injury_mask = dist <= radius
        
        # Penumbra (secondary injury zone)
        self.penumbra_mask = (dist > radius) & (dist <= radius * 1.6)
        
        # Initialize MMP burst in injury zone
        if injury_type == 'contusion':
            # Contusion: graded MMP release
            self.M[self.injury_mask] = 2.0
            self.M[self.penumbra_mask] = 0.5
        elif injury_type == 'laceration':
            # Laceration: severe MMP burst
            self.M[self.injury_mask] = 5.0
            self.M[self.penumbra_mask] = 1.5
        
        print(f"✓ Created {injury_type} injury (r={radius})")
        print(f"  Core: {np.sum(self.injury_mask)} pixels")
        print(f"  Penumbra: {np.sum(self.penumbra_mask)} pixels")
    
    def compute_alpha(self, t_hours: float) -> float:
        """Three-phase CSPG production rate (from V1)."""
        # Acute inflammatory spike
        acute = self.alpha_acute * np.exp(-t_hours / self.tau_acute)
        
        # Primary astrocytic production
        astro_activation = 1.0 / (1.0 + np.exp(-(t_hours - self.alpha_onset*24) / (self.alpha_width*24)))
        astro = self.alpha_peak * astro_activation
        
        # Secondary invasion wave
        invasion_activation = 1.0 / (1.0 + np.exp(-(t_hours - self.eta_onset) / self.eta_width))
        invasion = self.eta_secondary * invasion_activation
        
        return acute + astro + invasion
    
    def compute_beta_mmp(self, t_hours: float) -> Tuple[float, float]:
        """
        Injury-induced MMP production.
        
        Returns
        -------
        beta_2 : MMP-2 production rate (gradual rise)
        beta_9 : MMP-9 production rate (biphasic)
        """
        # MMP-2: gradual rise to peak around day 3-7
        t_days = t_hours / 24
        mmp2_activation = 1.0 / (1.0 + np.exp(-(t_days - 2.0) / 1.5))
        beta_2 = self.beta_mmp2 * mmp2_activation
        
        # MMP-9: biphasic (acute spike + chronic)
        acute_spike = self.beta_mmp9_acute * np.exp(-t_hours / self.tau_mmp9_acute)
        chronic_activation = 1.0 / (1.0 + np.exp(-(t_days - 3.0) / 2.0))
        beta_9 = acute_spike + self.beta_mmp9_chronic * chronic_activation
        
        return beta_2, beta_9
    
    def compute_T_production(self, t_hours: float) -> float:
        """
        TIMP2 production (counter-regulatory response).
        
        Genotype-dependent: A;A variant has enhanced production.
        """
        # Injury-induced upregulation
        t_signal = 1.0 / (1.0 + np.exp(-(t_hours - self.T_onset) / self.T_width))
        
        # Base production + injury response
        T_prod = (
            self.T_production_basal +
            self.T_production_injury * t_signal +
            self.small_mol_boost  # Small molecule enhancement
        )
        
        return T_prod * self.neuroplasticity_factor
    
    def step(self):
        """Advance simulation by one time step."""
        
        # Current time-dependent rates
        alpha_t = self.compute_alpha(self.time)
        beta_2, beta_9 = self.compute_beta_mmp(self.time)
        beta_mmp = beta_2 + beta_9
        T_prod = self.compute_T_production(self.time)
        
        # === Update S (CSPG) ===
        # Production - MMP degradation - self-inhibition - therapy
        dS = (
            alpha_t * (self.injury_mask | self.penumbra_mask) -
            self.gamma_mmp * self.M * self.S -
            self.delta_feedback * self.S**2 -
            self.gamma_therapy * self.S
        )
        
        self.S += self.dt * dS
        self.S = np.clip(self.S, 0.5, self.S_max)
        
        # === Update M (MMP) ===
        # Production (injury zones) - decay - TIMP2 binding + complex unbinding + diffusion
        production_mask = (self.injury_mask | self.penumbra_mask).astype(float)
        
        dM = (
            beta_mmp * production_mask -
            self.mu_mmp * self.M -
            self.k_on * self.M * self.T +
            self.k_off * self.C
        )
        
        # Add diffusion (MMPs spread from injury)
        dM += self.D_mmp * laplace(self.M)
        
        self.M += self.dt * dM
        self.M = np.maximum(self.M, 0)
        
        # === Update T (TIMP2) ===
        # Production - MMP binding + complex unbinding - degradation + diffusion
        dT = (
            T_prod * production_mask -
            self.k_on * self.M * self.T +
            self.k_off * self.C -
            self.d_T * self.T
        )
        
        # Add diffusion
        dT += self.D_timp * laplace(self.T)
        
        self.T += self.dt * dT
        self.T = np.maximum(self.T, 0)
        
        # === Update C (TIMP2-MMP Complex) ===
        # Binding - unbinding - degradation
        dC = (
            self.k_on * self.M * self.T -
            self.k_off * self.C -
            self.d_C * self.C
        )
        
        self.C += self.dt * dC
        self.C = np.maximum(self.C, 0)
        
        # Update time
        self.time += self.dt
    
    def run(self, duration_hours: float = 672, verbose: bool = True):
        """
        Run simulation for specified duration.
        
        Parameters
        ----------
        duration_hours : float
            Simulation duration (default 28 days)
        verbose : bool
            Print progress updates
        """
        n_steps = int(duration_hours / self.dt)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f" Running SCI Scar Simulation V2")
            print(f"{'='*60}")
            print(f"Duration: {duration_hours/24:.1f} days ({n_steps} steps)")
            print(f"Genotype: {self.genotype}")
            print(f"{'='*60}\n")
        
        for step in range(n_steps):
            self.step()
            
            # Record history every 12 hours
            if step % int(12 / self.dt) == 0:
                self.history['time'].append(self.time)
                self.history['S_injury'].append(np.mean(self.S[self.injury_mask]))
                self.history['S_penumbra'].append(np.mean(self.S[self.penumbra_mask]))
                self.history['M_injury'].append(np.mean(self.M[self.injury_mask]))
                self.history['T_injury'].append(np.mean(self.T[self.injury_mask]))
                self.history['C_injury'].append(np.mean(self.C[self.injury_mask]))
            
            # Progress updates
            if verbose and step % (n_steps // 10) == 0:
                t_days = self.time / 24
                S_core = np.mean(self.S[self.injury_mask])
                M_core = np.mean(self.M[self.injury_mask])
                T_core = np.mean(self.T[self.injury_mask])
                print(f"  t={t_days:5.1f}d | S={S_core:.2f}x | M={M_core:.2f} | T={T_core:.2f}")
        
        if verbose:
            print(f"\n✓ Simulation complete (t={self.time/24:.1f} days)")
        
        return self.history


def compare_genotypes(save_figure: bool = True):
    """
    Compare scar formation across genotypes.
    
    Generates a comprehensive comparison figure showing:
    - CSPG trajectories
    - MMP dynamics
    - TIMP2 responses
    """
    genotypes = ['typical', 'A;A', 'small_mol']
    colors = ['#E74C3C', '#2ECC71', '#3498DB']
    labels = ['Typical TIMP2', 'A;A variant', 'Small molecule']
    
    results = {}
    
    print("\n" + "="*70)
    print(" GENOTYPE COMPARISON SIMULATION")
    print("="*70)
    
    for genotype in genotypes:
        print(f"\nSimulating {genotype}...")
        sim = SCIScarSimulatorV2(grid_size=64, dt=0.5, genotype=genotype)
        sim.create_injury(center=(32, 32), radius=8, injury_type='contusion')
        sim.run(duration_hours=672, verbose=False)
        results[genotype] = sim
    
    # Generate comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Panel A: CSPG in injury core
    ax = axes[0, 0]
    for i, genotype in enumerate(genotypes):
        sim = results[genotype]
        t_days = np.array(sim.history['time']) / 24
        S_injury = np.array(sim.history['S_injury'])
        ax.plot(t_days, S_injury, color=colors[i], lw=2.5, label=labels[i])
    
    ax.axhline(4.0, color='gray', ls=':', alpha=0.5, label='Mature scar (~4×)')
    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('CSPG Level (fold change)', fontsize=11)
    ax.set_title('A. Scar Formation by Genotype', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 5)
    ax.grid(alpha=0.3)
    
    # Panel B: MMP activity
    ax = axes[0, 1]
    for i, genotype in enumerate(genotypes):
        sim = results[genotype]
        t_days = np.array(sim.history['time']) / 24
        M_injury = np.array(sim.history['M_injury'])
        ax.plot(t_days, M_injury, color=colors[i], lw=2.5, label=labels[i])
    
    # Add validation data
    _, t_val, mmp2_val, std_val = MMPValidationData.mmp2_timecourse()
    ax.errorbar(t_val, mmp2_val, yerr=std_val, fmt='ko', markersize=8,
                capsize=4, alpha=0.6, label='MMP-2 (lit.)', zorder=10)
    
    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('MMP Activity (normalized)', fontsize=11)
    ax.set_title('B. MMP Dynamics', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 28)
    ax.grid(alpha=0.3)
    
    # Panel C: TIMP2 response
    ax = axes[0, 2]
    for i, genotype in enumerate(genotypes):
        sim = results[genotype]
        t_days = np.array(sim.history['time']) / 24
        T_injury = np.array(sim.history['T_injury'])
        ax.plot(t_days, T_injury, color=colors[i], lw=2.5, label=labels[i])
    
    # Add validation data
    _, t_val, timp2_val, std_val = MMPValidationData.timp2_timecourse()
    ax.errorbar(t_val, timp2_val, yerr=std_val, fmt='ks', markersize=8,
                capsize=4, alpha=0.6, label='TIMP2 (lit.)', zorder=10)
    
    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('TIMP2 Level (normalized)', fontsize=11)
    ax.set_title('C. TIMP2 Counter-Regulation', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 28)
    ax.grid(alpha=0.3)
    
    # Panel D: TIMP2-MMP complex
    ax = axes[1, 0]
    for i, genotype in enumerate(genotypes):
        sim = results[genotype]
        t_days = np.array(sim.history['time']) / 24
        C_injury = np.array(sim.history['C_injury'])
        ax.plot(t_days, C_injury, color=colors[i], lw=2.5, label=labels[i])
    
    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('TIMP2-MMP Complex', fontsize=11)
    ax.set_title('D. MMP Inhibition (Complex Formation)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, 28)
    ax.grid(alpha=0.3)
    
    # Panel E: Spatial snapshots at day 7 (typical vs A;A)
    for i, (genotype, label) in enumerate(zip(['typical', 'A;A'], ['Typical', 'A;A variant'])):
        ax = axes[1, i+1]
        sim = results[genotype]
        im = ax.imshow(sim.S, cmap='Reds', vmin=0.5, vmax=5.0)
        ax.set_title(f'E{i+1}. Day 28 CSPG ({label})', fontsize=11, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(
        'SCI Scar Formation V2: TIMP2/MMP Dynamics & Genotype Effects',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    
    if save_figure:
        plt.savefig('genotype_comparison.png', dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print("\n✓ Generated: genotype_comparison.png")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*70)
    print(" GENOTYPE COMPARISON SUMMARY")
    print("="*70)
    
    for genotype, label in zip(genotypes, labels):
        sim = results[genotype]
        final_S = sim.history['S_injury'][-1]
        final_M = sim.history['M_injury'][-1]
        final_T = sim.history['T_injury'][-1]
        
        print(f"\n{label}:")
        print(f"  Final CSPG (day 28): {final_S:.2f}×")
        print(f"  Final MMP activity:  {final_M:.2f}")
        print(f"  Final TIMP2 level:   {final_T:.2f}")
        
        # Compute reduction vs typical
        if genotype != 'typical':
            typical_S = results['typical'].history['S_injury'][-1]
            reduction = (1 - final_S / typical_S) * 100
            print(f"  Scar reduction vs typical: {reduction:.1f}%")
    
    return results


if __name__ == "__main__":
    # Run genotype comparison
    results = compare_genotypes(save_figure=True)
    
    print("\n" + "="*70)
    print(" NEXT STEPS")
    print("="*70)
    print("\n1. Validate MMP time courses against literature")
    print("2. Test small molecule candidates from ZINC20 screen")
    print("3. Explore chABC + TIMP2 enhancement synergy")
    print("4. Expand to 3D with DTI-driven anisotropy")
    print("\n✓ V2 implementation complete!\n")
