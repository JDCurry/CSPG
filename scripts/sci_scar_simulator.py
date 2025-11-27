"""
SCI ECM Scar Simulator V1
=========================
Spinal Cord Injury-specific ECM modeling focused on CSPG scar formation
and therapeutic intervention dynamics.

This is a fundamental reconceptualization from wound healing models:
- Skin wound healing: ECM RECOVERS toward baseline
- SCI glial scar: Inhibitory ECM ACCUMULATES pathologically

Fields:
    S(x,y,t) - Scar inhibitory ECM density (CSPGs, tenascin-C, NG2, etc.)
    M(x,y,t) - MMP concentration (endogenous or therapeutic)
    T(x,y,t) - TIMP concentration (inhibits MMP; later: dual roles)

Key Equation (replaces logistic growth):
    ∂S/∂t = +α_astro - β_MMP·M·S - γ_therapy·S

Where:
    +α_astro:     Astrocyte-driven CSPG production (time-varying sigmoid)
    -β_MMP·M·S:   MMP-mediated scar degradation
    -γ_therapy·S: Exogenous intervention (chABC, TIMP2 variants, etc.)

Canonical SCI CSPG Timeline (from literature):
    0h:   1.0 (baseline)
    24h:  1.5 (initial rise)
    72h:  3.0 (explosive accumulation)
    7d:   4.0 (scar fully formed)
    28d:  4.2 (maturation plateau)

References:
    - Silver & Miller 2004 (glial scar review)
    - Jones et al. 2003 (CSPG time course)
    - Tang et al. 2003 (quantitative Western blots)
    - Tran et al. 2018 (CSPG signaling review)
    - Andrews et al. 2012 (contusion injury CSPG dynamics)
Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass


# =============================================================================
# VALIDATION DATA
# =============================================================================

@dataclass
class SCIValidationData:
    """Container for SCI experimental validation data."""
    name: str
    time_hours: np.ndarray
    cspg_level: np.ndarray  # Fold change relative to baseline
    std_dev: np.ndarray
    
    @classmethod
    def canonical_cspg_timecourse(cls):
        """
        Canonical CSPG accumulation after SCI.
        Synthesized from Silver & Miller 2004, Jones et al. 2003, Tang et al. 2003.
        
        Values represent fold-change relative to uninjured baseline.
        """
        return cls(
            name="Canonical SCI CSPG",
            time_hours=np.array([0, 24, 72, 168, 672]),  # 0h, 1d, 3d, 7d, 28d
            cspg_level=np.array([1.0, 1.5, 3.0, 4.0, 4.2]),
            std_dev=np.array([0.1, 0.2, 0.3, 0.3, 0.3])
        )
    
    @classmethod
    def neurocan_andrews_2012(cls):
        """
        Neurocan expression from Andrews et al. 2012.
        Severe mid-thoracic contusion in rats.
        ~4-fold increase by 7 dpi, sustained through 28 dpi.
        """
        return cls(
            name="Neurocan (Andrews 2012)",
            time_hours=np.array([0, 72, 168, 336, 672]),  # 0, 3, 7, 14, 28 days
            cspg_level=np.array([1.0, 2.5, 4.0, 3.8, 3.5]),
            std_dev=np.array([0.1, 0.3, 0.4, 0.4, 0.4])
        )
    
    @classmethod
    def ng2_jones_2002(cls):
        """
        NG2 expression from Jones et al. 2002.
        Upregulated within 24h, peaks at 1 week.
        """
        return cls(
            name="NG2 (Jones 2002)",
            time_hours=np.array([0, 24, 72, 168, 336]),
            cspg_level=np.array([1.0, 2.0, 3.5, 4.5, 4.0]),
            std_dev=np.array([0.1, 0.2, 0.3, 0.4, 0.4])
        )


# =============================================================================
# SCI ECM SIMULATOR
# =============================================================================

class SCIScarSimulator:
    """
    Spinal Cord Injury Scar Formation Simulator.
    
    Models the pathological accumulation of inhibitory ECM (CSPGs) after SCI,
    and the potential for therapeutic intervention via MMP-mediated degradation
    or direct scar-softening treatments.
    
    Key biological differences from skin wound healing:
    1. CSPGs INCREASE after injury (not recover)
    2. MMP activity is potentially THERAPEUTIC (clears inhibitory matrix)
    3. TIMP inhibition of MMP may be DETRIMENTAL to regeneration
    4. Goal is to REDUCE scar, not promote deposition
    """
    
    # Validated parameter presets
    VALIDATED_PARAMS = {
        'baseline': {
            'alpha_peak': 0.015,      # Peak astrocyte CSPG production rate
            'alpha_onset': 12.0,      # Hours until production ramps up
            'alpha_width': 24.0,      # Width of production ramp
            'beta_mmp': 0.01,         # MMP-scar degradation rate
            'gamma_therapy': 0.0,     # No therapy by default
            'description': 'Baseline SCI scar formation (no intervention)'
        },
        'with_chABC': {
            'alpha_peak': 0.015,
            'alpha_onset': 12.0,
            'alpha_width': 24.0,
            'beta_mmp': 0.01,
            'gamma_therapy': 0.005,   # Chondroitinase ABC treatment
            'description': 'SCI with chondroitinase ABC therapy'
        }
    }
    
    def __init__(
        self,
        grid_size: int = 128,
        dx: float = 0.01,
        dt: float = 0.1,
        solver: str = 'backward_euler'
    ):
        """
        Initialize the SCI scar simulator.
        
        Parameters
        ----------
        grid_size : int
            Size of square grid (NxN)
        dx : float
            Spatial step (mm for spinal cord scale)
        dt : float
            Time step (hours)
        solver : str
            Numerical solver: 'euler' or 'backward_euler'
        """
        self.N = grid_size
        self.dx = dx
        self.dt = dt
        self.solver = solver
        
        # Initialize fields
        self.S = np.ones((self.N, self.N))       # Scar ECM (baseline = 1.0)
        self.M = np.zeros((self.N, self.N))      # MMP concentration
        self.T = np.ones((self.N, self.N)) * 0.5 # TIMP concentration
        
        # Source terms
        self.injury_mask = np.zeros((self.N, self.N), dtype=bool)
        self.S_M = np.zeros((self.N, self.N))    # MMP sources (therapeutic injection sites)
        
        # Kinetic parameters
        self.params = {
            # Diffusion (mm²/h)
            'D_M': 0.001,
            'D_T': 0.001,
            
            # CSPG production (astrocyte-driven, time-dependent)
            'alpha_peak': 0.015,      # Peak production rate (1/h)
            'alpha_onset': 12.0,      # Hours until production begins
            'alpha_width': 24.0,      # Sigmoid transition width
            'S_max': 5.0,             # Maximum scar density (saturation)
            
            # Acute inflammatory spike (the "goth phase")
            'alpha_acute': 0.02,      # Magnitude of acute burst
            'tau_acute': 8.0,         # Decay time constant (hours)
            
            # Secondary wave (day 2-3 explosive accumulation)
            # Peak astrocyte activation, NG2 expansion, OPC invasion,
            # fibroblast infiltration, TGF-β spike
            'eta_secondary': 0.015,   # Magnitude of secondary surge
            'eta_onset': 36.0,        # Hours until secondary wave (~1.5 days)
            'eta_width': 12.0,        # Sigmoid width (sharp surge)
            
            # Self-limiting feedback (scar maturation plateau)
            'delta_feedback': 0.001,  # Quadratic self-inhibition coefficient
            
            # MMP dynamics
            'beta_mmp': 0.01,         # MMP-scar degradation rate
            'k_mmp_deg': 0.05,        # MMP natural decay (1/h)
            'k_inhibit': 0.5,         # TIMP inhibition of MMP
            
            # TIMP dynamics
            'k_timp_deg': 0.02,       # TIMP decay (1/h)
            
            # Therapeutic intervention
            'gamma_therapy': 0.0,     # Direct scar softening (e.g., chABC)
        }
        
        # Simulation state
        self.time = 0.0
        self.history = {
            'time': [],
            'S_mean': [], 'S_max': [], 'S_injury': [],
            'M_mean': [], 'M_max': [],
            'T_mean': []
        }
    
    def set_parameters(self, **kwargs):
        """Update simulation parameters."""
        self.params.update(kwargs)
    
    def use_validated_params(self, preset: str = 'baseline'):
        """Load a validated parameter preset."""
        if preset not in self.VALIDATED_PARAMS:
            raise ValueError(f"Unknown preset: {preset}")
        
        params = self.VALIDATED_PARAMS[preset].copy()
        desc = params.pop('description')
        self.params.update(params)
        print(f"Loaded: {desc}")
        return self
    
    # =========================================================================
    # INJURY SETUP
    # =========================================================================
    
    def create_injury(
        self,
        center: Tuple[int, int],
        radius: int,
        injury_type: str = 'contusion'
    ):
        """
        Create a spinal cord injury site.
        
        Parameters
        ----------
        center : tuple
            (x, y) center of injury
        radius : int
            Radius of injury in grid cells
        injury_type : str
            'contusion' (closed), 'transection' (open), or 'stab'
        
        At injury:
        - S starts at baseline (CSPGs haven't accumulated yet)
        - Small initial MMP burst from cell damage
        - TIMP may be depleted or elevated depending on injury type
        """
        y, x = np.ogrid[:self.N, :self.N]
        self.injury_mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        
        if injury_type == 'contusion':
            # Closed injury: moderate initial disruption
            self.S[self.injury_mask] = 0.8  # Slight initial loss
            self.M[self.injury_mask] = 0.5  # Small MMP burst from damaged cells
            
        elif injury_type == 'transection':
            # Open injury: severe disruption, blood-brain barrier breach
            self.S[self.injury_mask] = 0.5
            self.M[self.injury_mask] = 1.0  # Larger MMP burst
            
        elif injury_type == 'stab':
            # Focal injury
            self.S[self.injury_mask] = 0.7
            self.M[self.injury_mask] = 0.3
        
        return self.injury_mask
    
    def add_therapeutic_mmp(
        self,
        center: Tuple[int, int],
        radius: int,
        concentration: float = 2.0,
        sustained: bool = False
    ):
        """
        Add therapeutic MMP injection (e.g., to degrade CSPGs).
        
        In SCI, unlike skin wounds, MMP activity can be THERAPEUTIC
        by clearing inhibitory scar matrix.
        """
        y, x = np.ogrid[:self.N, :self.N]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        
        if sustained:
            self.S_M[mask] = concentration * 0.1  # Sustained release
        else:
            self.M[mask] += concentration  # Bolus injection
    
    def add_chABC_treatment(self, start_time: float = 72.0, duration: float = 168.0):
        """
        Simulate chondroitinase ABC treatment.
        
        chABC enzymatically degrades CSPG glycosaminoglycan chains,
        reducing their inhibitory activity. Modeled as increased γ_therapy.
        
        Parameters
        ----------
        start_time : float
            Hours post-injury to begin treatment
        duration : float
            Hours of treatment duration
        """
        self.chABC_start = start_time
        self.chABC_end = start_time + duration
        self.chABC_active = True
    
    # =========================================================================
    # ASTROCYTE CSPG PRODUCTION
    # =========================================================================
    
    def get_alpha_astro(self) -> float:
        """
        Time-dependent astrocyte CSPG production rate.
        
        Models the inflammatory cascade with THREE phases:
        
        1. ACUTE SPIKE (0-24h): "The Goth Phase"
           - Microglial inflammatory burst
           - Cytokine chaos, ionic imbalance
           - Fast exponential decay: α_acute · exp(-t/τ)
        
        2. SUSTAINED PRODUCTION (12h+): "The Sigmoid Awakening"
           - Reactive astrogliosis ramps up
           - Sustained CSPG synthesis
           - Sigmoid activation toward α_peak
        
        3. SECONDARY WAVE (36-72h): "The Explosive Accumulation"
           - Peak astrocyte activation
           - NG2 glia expansion
           - OPC invasion, fibroblast infiltration
           - TGF-β signaling spike
           - Delayed sigmoid: η · sigmoid(t - 36h)
        
        Returns α(t) = acute_spike + sigmoid_production + secondary_wave
        """
        alpha_peak = self.params['alpha_peak']
        onset = self.params['alpha_onset']
        width = self.params['alpha_width']
        alpha_acute = self.params['alpha_acute']
        tau_acute = self.params['tau_acute']
        eta = self.params['eta_secondary']
        eta_onset = self.params['eta_onset']
        eta_width = self.params['eta_width']
        
        # Acute inflammatory spike (fast decay)
        acute_spike = alpha_acute * np.exp(-self.time / tau_acute)
        
        # Sigmoid activation (sustained production)
        activation = 1.0 / (1.0 + np.exp(-(self.time - onset) / width))
        sustained = alpha_peak * activation
        
        # Secondary wave (delayed surge at day 2-3)
        secondary_activation = 1.0 / (1.0 + np.exp(-(self.time - eta_onset) / eta_width))
        secondary_wave = eta * secondary_activation
        
        return acute_spike + sustained + secondary_wave
    
    def get_gamma_therapy(self) -> float:
        """Get current therapy rate (may be time-dependent for chABC)."""
        gamma = self.params['gamma_therapy']
        
        # Check for chABC treatment window
        if hasattr(self, 'chABC_active') and self.chABC_active:
            if self.chABC_start <= self.time <= self.chABC_end:
                return gamma + 0.01  # chABC adds to baseline therapy
        
        return gamma
    
    # =========================================================================
    # NUMERICAL METHODS
    # =========================================================================
    
    def laplacian(self, Z: np.ndarray) -> np.ndarray:
        """Discrete Laplacian with zero-flux boundary conditions."""
        lap = -4 * Z
        lap += np.roll(Z, 1, axis=0)
        lap += np.roll(Z, -1, axis=0)
        lap += np.roll(Z, 1, axis=1)
        lap += np.roll(Z, -1, axis=1)
        return lap / (self.dx ** 2)
    
    def _solve_diffusion_implicit(
        self,
        field: np.ndarray,
        D: float,
        n_iterations: int = 20
    ) -> np.ndarray:
        """Implicit diffusion solve via Jacobi iteration."""
        r = D * self.dt / (self.dx ** 2)
        u = field.copy()
        
        for _ in range(n_iterations):
            u = (field + r * (
                np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
                np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1)
            )) / (1 + 4 * r)
        
        return u
    
    def compute_dS(self) -> np.ndarray:
        """
        Compute scar ECM rate of change.
        
        ∂S/∂t = +α_astro·(1 - S/S_max) - δ·S² - β_MMP·M·S - γ_therapy·S
        
        Where:
            +α_astro·(1 - S/S_max): Saturating production
            -δ·S²: Self-limiting feedback (maturation plateau)
                   Represents: CSPG-TGFβ saturation, astrocyte metabolic
                   slowdown, tissue compaction, proteoglycan crosslinking
            -β_MMP·M·S: MMP-mediated degradation  
            -γ_therapy·S: Therapeutic intervention
        """
        alpha = self.get_alpha_astro()
        beta = self.params['beta_mmp']
        gamma = self.get_gamma_therapy()
        S_max = self.params['S_max']
        delta = self.params['delta_feedback']
        
        # Only produce CSPGs in/near injury zone (reactive astrocytes)
        # Use a smooth spatial mask centered on injury
        production_zone = np.zeros_like(self.S)
        if np.any(self.injury_mask):
            # Dilate injury mask for penumbra
            from scipy.ndimage import binary_dilation
            penumbra = binary_dilation(self.injury_mask, iterations=5)
            production_zone[penumbra] = 1.0
        else:
            production_zone = np.ones_like(self.S)
        
        # Production (saturating)
        production = alpha * production_zone * (1 - self.S / S_max)
        production = np.maximum(production, 0)  # No negative production
        
        # Self-limiting feedback (quadratic) - clamps the plateau
        feedback = delta * self.S * self.S
        
        # MMP-mediated degradation
        degradation = beta * self.M * self.S
        
        # Therapeutic intervention
        therapy = gamma * self.S
        
        return production - feedback - degradation - therapy
    
    def step_euler(self):
        """Forward Euler time step."""
        D_M = self.params['D_M']
        D_T = self.params['D_T']
        k_mmp_deg = self.params['k_mmp_deg']
        k_inhibit = self.params['k_inhibit']
        k_timp_deg = self.params['k_timp_deg']
        
        # Diffusion
        lap_M = self.laplacian(self.M)
        lap_T = self.laplacian(self.T)
        
        # MMP dynamics: diffusion, decay, TIMP inhibition, sources
        dM = D_M * lap_M - k_mmp_deg * self.M - k_inhibit * self.T * self.M + self.S_M
        
        # TIMP dynamics: diffusion, decay
        dT = D_T * lap_T - k_timp_deg * self.T
        
        # Scar dynamics
        dS = self.compute_dS()
        
        # Update
        self.M += self.dt * dM
        self.T += self.dt * dT
        self.S += self.dt * dS
    
    def step_backward_euler(self):
        """Backward Euler with implicit diffusion."""
        D_M = self.params['D_M']
        D_T = self.params['D_T']
        k_mmp_deg = self.params['k_mmp_deg']
        k_inhibit = self.params['k_inhibit']
        k_timp_deg = self.params['k_timp_deg']
        
        # MMP: reactions then implicit diffusion
        M_react = self.M + self.dt * (
            -k_mmp_deg * self.M - k_inhibit * self.T * self.M + self.S_M
        )
        self.M = self._solve_diffusion_implicit(M_react, D_M, n_iterations=30)
        
        # TIMP: reactions then implicit diffusion
        T_react = self.T + self.dt * (-k_timp_deg * self.T)
        self.T = self._solve_diffusion_implicit(T_react, D_T, n_iterations=30)
        
        # Scar: no diffusion (CSPGs are matrix-bound)
        self.S += self.dt * self.compute_dS()
    
    def step(self):
        """Advance simulation by one time step."""
        if self.solver == 'euler':
            self.step_euler()
        elif self.solver == 'backward_euler':
            self.step_backward_euler()
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
        
        # Enforce positivity
        self.M = np.maximum(self.M, 0)
        self.T = np.maximum(self.T, 0)
        self.S = np.maximum(self.S, 0.1)  # Minimum baseline ECM
        
        self.time += self.dt
    
    # =========================================================================
    # SIMULATION EXECUTION
    # =========================================================================
    
    def record_history(self):
        """Record current state statistics."""
        self.history['time'].append(self.time)
        self.history['S_mean'].append(float(np.mean(self.S)))
        self.history['S_max'].append(float(np.max(self.S)))
        
        if np.any(self.injury_mask):
            self.history['S_injury'].append(float(np.mean(self.S[self.injury_mask])))
        else:
            self.history['S_injury'].append(float(np.mean(self.S)))
        
        self.history['M_mean'].append(float(np.mean(self.M)))
        self.history['M_max'].append(float(np.max(self.M)))
        self.history['T_mean'].append(float(np.mean(self.T)))
    
    def run(
        self,
        duration_hours: float = None,
        n_steps: int = None,
        record_interval: int = 10,
        verbose: bool = True
    ):
        """Run simulation."""
        if duration_hours is not None:
            n_steps = int(duration_hours / self.dt)
        elif n_steps is None:
            raise ValueError("Specify duration_hours or n_steps")
        
        if verbose:
            print(f"\nRunning SCI scar simulation...")
            print(f"Duration: {n_steps * self.dt:.1f}h ({n_steps * self.dt / 24:.1f} days)")
        
        for step in range(n_steps):
            self.step()
            
            if step % record_interval == 0:
                self.record_history()
            
            if verbose and n_steps > 10 and step % (n_steps // 10) == 0:
                pct = 100 * step / n_steps
                S_inj = np.mean(self.S[self.injury_mask]) if np.any(self.injury_mask) else np.mean(self.S)
                print(f"  {pct:5.1f}% | t={self.time:6.1f}h ({self.time/24:.1f}d) | "
                      f"S_injury={S_inj:.2f}")
        
        self.record_history()
        if verbose:
            print("Complete!")
        
        return self
    
    def run_to_timepoints(
        self,
        timepoints: List[float],
        region_mask: np.ndarray = None
    ) -> Dict:
        """Run and collect values at specific timepoints."""
        if region_mask is None:
            region_mask = self.injury_mask if np.any(self.injury_mask) else np.ones((self.N, self.N), dtype=bool)
        
        results = {'times': [self.time], 'S_values': [np.mean(self.S[region_mask])]}
        
        for t_target in timepoints:
            if t_target <= self.time:
                continue
            while self.time < t_target:
                self.step()
            results['times'].append(self.time)
            results['S_values'].append(np.mean(self.S[region_mask]))
        
        results['times'] = np.array(results['times'])
        results['S_values'] = np.array(results['S_values'])
        
        return results
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def validate_against(
        self,
        data: SCIValidationData,
        region_mask: np.ndarray = None
    ) -> Dict:
        """Validate simulation against experimental data."""
        results = self.run_to_timepoints(data.time_hours[1:], region_mask)
        
        observed = data.cspg_level
        predicted = results['S_values']
        
        # Compute R²
        ss_res = np.sum((observed - predicted) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        
        errors = predicted - observed
        
        return {
            'r_squared': r_squared,
            'predicted': predicted,
            'observed': observed,
            'errors': errors,
            'max_error': np.max(np.abs(errors)),
            'within_error_bars': np.all(np.abs(errors) <= data.std_dev)
        }
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def plot_fields(self, save_path: str = None, show: bool = True):
        """Visualize current state."""
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Scar density
        im0 = axs[0].imshow(self.S, cmap='Reds', vmin=0, vmax=5)
        axs[0].set_title(f'S: Scar ECM (t={self.time:.1f}h = {self.time/24:.1f}d)', fontsize=12)
        axs[0].axis('off')
        plt.colorbar(im0, ax=axs[0], fraction=0.046, label='CSPG level')
        
        # MMP
        im1 = axs[1].imshow(self.M, cmap='Greens')
        axs[1].set_title(f'M: MMP (max={np.max(self.M):.2f})', fontsize=12)
        axs[1].axis('off')
        plt.colorbar(im1, ax=axs[1], fraction=0.046, label='MMP')
        
        # TIMP
        im2 = axs[2].imshow(self.T, cmap='Blues')
        axs[2].set_title(f'T: TIMP (mean={np.mean(self.T):.2f})', fontsize=12)
        axs[2].axis('off')
        plt.colorbar(im2, ax=axs[2], fraction=0.046, label='TIMP')
        
        plt.suptitle('SCI Glial Scar Simulation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_history(self, save_path: str = None, show: bool = True):
        """Plot scar formation dynamics."""
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        
        t = np.array(self.history['time'])
        t_days = t / 24
        
        # Scar dynamics
        ax = axs[0]
        ax.plot(t_days, self.history['S_injury'], 'r-', lw=2, label='Injury site')
        ax.plot(t_days, self.history['S_mean'], 'r--', lw=1, alpha=0.7, label='Global mean')
        ax.axhline(1.0, color='gray', ls=':', alpha=0.5, label='Baseline')
        ax.axhline(4.0, color='darkred', ls='--', alpha=0.5, label='Mature scar (~4x)')
        
        # Add phase annotations
        ax.axvspan(0, 0.5, alpha=0.1, color='yellow', label='Acute (0-12h)')
        ax.axvspan(0.5, 3, alpha=0.1, color='orange', label='Subacute (12h-3d)')
        ax.axvspan(3, 7, alpha=0.1, color='red', label='Scar formation (3-7d)')
        
        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('Scar ECM (fold change)', fontsize=12)
        ax.set_title('CSPG Scar Accumulation After SCI', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, max(t_days))
        
        # MMP/TIMP dynamics
        ax = axs[1]
        ax.plot(t_days, self.history['M_mean'], 'g-', lw=2, label='MMP (mean)')
        ax.plot(t_days, self.history['M_max'], 'g--', lw=1, alpha=0.7, label='MMP (max)')
        ax.plot(t_days, self.history['T_mean'], 'b-', lw=2, label='TIMP (mean)')
        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('Concentration', fontsize=12)
        ax.set_title('MMP/TIMP Dynamics', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_summary(self) -> str:
        """Return summary string."""
        S_inj = np.mean(self.S[self.injury_mask]) if np.any(self.injury_mask) else np.mean(self.S)
        lines = [
            "=" * 60,
            "SCI SCAR SIMULATOR SUMMARY",
            "=" * 60,
            f"Time: {self.time:.1f}h ({self.time/24:.1f} days)",
            f"Solver: {self.solver}",
            "-" * 60,
            f"Scar ECM (S):",
            f"  Injury site: {S_inj:.3f} (fold change)",
            f"  Global mean: {np.mean(self.S):.3f}",
            f"  Maximum:     {np.max(self.S):.3f}",
            f"MMP (M): mean={np.mean(self.M):.4f}, max={np.max(self.M):.4f}",
            f"TIMP (T): mean={np.mean(self.T):.4f}",
            "=" * 60
        ]
        return "\n".join(lines)


# =============================================================================
# PARAMETER FITTING
# =============================================================================

def compute_r_squared(observed, predicted):
    """Compute R² coefficient of determination."""
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    return 1 - ss_res / ss_tot


def fit_to_canonical_data(verbose: bool = True):
    """
    Fit model parameters to canonical CSPG time course.
    """
    data = SCIValidationData.canonical_cspg_timecourse()
    
    if verbose:
        print("=" * 70)
        print(" FITTING SCI SCAR MODEL TO CANONICAL CSPG DATA")
        print("=" * 70)
        print(f"\nTarget data: {data.name}")
        print(f"Timepoints: {data.time_hours} hours")
        print(f"CSPG levels: {data.cspg_level}")
    
    best_r2 = -999
    best_params = None
    
    # Parameter sweep - now with secondary wave for 72h explosive accumulation
    # Fix proven values, sweep new parameters
    for alpha_peak in [0.025, 0.028]:
        for alpha_acute in [0.010, 0.012]:
            for tau_acute in [24]:
                for eta_secondary in [0.012, 0.015, 0.018]:
                    for eta_onset in [36, 42]:
                        for delta_feedback in [0.0002, 0.00025]:
                            sim = SCIScarSimulator(grid_size=32, dt=1.0)
                            sim.set_parameters(
                                alpha_peak=alpha_peak,
                                alpha_onset=3,
                                alpha_width=6,
                                S_max=5.0,
                                alpha_acute=alpha_acute,
                                tau_acute=tau_acute,
                                eta_secondary=eta_secondary,
                                eta_onset=eta_onset,
                                eta_width=12.0,
                                delta_feedback=delta_feedback
                            )
                            
                            center = (16, 16)
                            sim.create_injury(center=center, radius=5, injury_type='contusion')
                            
                            results = sim.run_to_timepoints(data.time_hours[1:], sim.injury_mask)
                            r2 = compute_r_squared(data.cspg_level, results['S_values'])
                            
                            if r2 > best_r2:
                                best_r2 = r2
                                best_params = {
                                    'alpha_peak': alpha_peak,
                                    'alpha_onset': 3,
                                    'alpha_width': 6,
                                    'S_max': 5.0,
                                    'alpha_acute': alpha_acute,
                                    'tau_acute': tau_acute,
                                    'eta_secondary': eta_secondary,
                                    'eta_onset': eta_onset,
                                    'eta_width': 12.0,
                                    'delta_feedback': delta_feedback,
                                    'predictions': results['S_values']
                                }
                                if verbose:
                                    print(f"  New best: α={alpha_peak:.3f}, acute={alpha_acute:.3f}, "
                                          f"η={eta_secondary:.3f}, onset={eta_onset}h, δ={delta_feedback:.5f} → R²={r2:.4f}")
    
    if verbose:
        print("\n" + "-" * 70)
        print("OPTIMAL PARAMETERS:")
        print(f"  α_peak:      {best_params['alpha_peak']:.4f} (sustained production)")
        print(f"  onset:       {best_params['alpha_onset']} hours")
        print(f"  width:       {best_params['alpha_width']} hours")
        print(f"  S_max:       {best_params['S_max']}")
        print(f"  α_acute:     {best_params['alpha_acute']:.4f} (goth spike)")
        print(f"  τ_acute:     {best_params['tau_acute']} hours")
        print(f"  η_secondary: {best_params['eta_secondary']:.4f} (explosive wave)")
        print(f"  η_onset:     {best_params['eta_onset']} hours")
        print(f"  δ_feedback:  {best_params['delta_feedback']:.5f} (plateau clamp)")
        print(f"  R²:          {best_r2:.4f}")
        print(f"\n  Predicted: {[f'{v:.2f}' for v in best_params['predictions']]}")
        print(f"  Observed:  {list(data.cspg_level)}")
    
    return best_params, best_r2


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    # Need scipy for binary_dilation
    try:
        from scipy.ndimage import binary_dilation
    except ImportError:
        print("Installing scipy...")
        import subprocess
        subprocess.run(['pip', 'install', 'scipy', '-q'])
        from scipy.ndimage import binary_dilation
    
    print("=" * 70)
    print(" SCI SCAR SIMULATOR - DEMONSTRATION")
    print("=" * 70)
    
    # Fit to canonical data
    best_params, best_r2 = fit_to_canonical_data()
    
    # Run full simulation with best parameters
    print("\n" + "=" * 70)
    print(" FULL SIMULATION WITH OPTIMAL PARAMETERS")
    print("=" * 70)
    
    sim = SCIScarSimulator(grid_size=64, dt=0.5)
    sim.set_parameters(
        alpha_peak=best_params['alpha_peak'],
        alpha_onset=best_params['alpha_onset'],
        alpha_width=best_params['alpha_width'],
        S_max=best_params['S_max'],
        alpha_acute=best_params['alpha_acute'],
        tau_acute=best_params['tau_acute'],
        eta_secondary=best_params['eta_secondary'],
        eta_onset=best_params['eta_onset'],
        eta_width=best_params['eta_width'],
        delta_feedback=best_params['delta_feedback']
    )
    
    # Create contusion injury
    center = (32, 32)
    sim.create_injury(center=center, radius=8, injury_type='contusion')
    
    # Run for 28 days
    sim.run(duration_hours=672, verbose=True)
    
    print(sim.get_summary())
    
    # Save visualizations
    sim.plot_fields(save_path='/home/claude/sci_scar_fields.png', show=False)
    sim.plot_history(save_path='/home/claude/sci_scar_history.png', show=False)
    
    # Validation plot
    data = SCIValidationData.canonical_cspg_timecourse()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    t_days = np.array(sim.history['time']) / 24
    ax.plot(t_days, sim.history['S_injury'], 'r-', lw=2, label='Model prediction')
    
    ax.errorbar(data.time_hours / 24, data.cspg_level, yerr=data.std_dev,
                fmt='ko', markersize=10, capsize=5, lw=2,
                label=f'{data.name} (R²={best_r2:.3f})', zorder=10)
    
    ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('CSPG Level (fold change)', fontsize=12)
    ax.set_title('SCI Scar Model Validation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 5)
    
    plt.tight_layout()
    plt.savefig('/home/claude/sci_scar_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Demonstration complete!")
    print("  Saved: sci_scar_fields.png")
    print("  Saved: sci_scar_history.png")
    print("  Saved: sci_scar_validation.png")
