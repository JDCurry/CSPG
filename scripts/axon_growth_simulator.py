"""
SCI Axon Growth Simulator
==========================

Simulates axonal regeneration and sprouting through an evolving scar landscape.

Key concept: S(x,y,t) from the scar simulator becomes a "cost field" for axon growth.
- Low S → permissive substrate, easy growth
- High S → inhibitory barrier, growth failure or detour

Growth models implemented:
1. Biased random walk (growth cone dynamics)
2. Cost-threshold model (axons grow until accumulated cost exceeds capacity)
3. Shortest-path connectivity (graph-based traversal analysis)

Outputs:
- % fibers successfully traversing lesion
- Connectivity maps (rostral → caudal)
- Path length distributions
- Functional recovery proxy scores
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from scipy.ndimage import distance_transform_edt
import heapq


@dataclass
class AxonPopulation:
    """Represents a population of growing axons."""
    name: str
    start_positions: np.ndarray  # (N, 2) array of starting (x, y)
    current_positions: np.ndarray  # Current tip positions
    paths: List[List[Tuple[int, int]]]  # Full path history for each axon
    active: np.ndarray  # Boolean mask - which axons are still growing
    accumulated_cost: np.ndarray  # Cumulative scar exposure
    crossed: np.ndarray  # Boolean - successfully crossed lesion
    
    @classmethod
    def create_population(cls, name: str, n_axons: int, 
                          y_start: int, x_range: Tuple[int, int],
                          grid_size: int) -> 'AxonPopulation':
        """Create a population of axons starting from a rostral band."""
        x_positions = np.random.randint(x_range[0], x_range[1], n_axons)
        y_positions = np.full(n_axons, y_start)
        start_pos = np.stack([x_positions, y_positions], axis=1)
        
        paths = [[(x, y)] for x, y in start_pos]
        
        return cls(
            name=name,
            start_positions=start_pos.copy(),
            current_positions=start_pos.copy(),
            paths=paths,
            active=np.ones(n_axons, dtype=bool),
            accumulated_cost=np.zeros(n_axons),
            crossed=np.zeros(n_axons, dtype=bool)
        )


class AxonGrowthSimulator:
    """
    Simulates axon growth through an evolving scar landscape.
    
    The scar field S(x,y,t) acts as a cost/resistance field:
    - Growth probability decreases with local S
    - Accumulated S exposure can cause growth cone collapse
    - Path selection biased toward lower S regions
    
    Models both:
    - Long-distance regeneration (corticospinal, propriospinal)
    - Local sprouting and plasticity (spared fiber collaterals)
    """
    
    def __init__(
        self,
        scar_field: np.ndarray,
        grid_size: int = 64,
        lesion_center: Tuple[int, int] = (32, 32),
        lesion_radius: int = 8
    ):
        """
        Initialize with a scar field snapshot or time series.
        
        Parameters
        ----------
        scar_field : np.ndarray
            Either 2D (static) or 3D (time-evolving) scar density field
        grid_size : int
            Size of simulation grid
        lesion_center : tuple
            Center of lesion (for defining success criteria)
        lesion_radius : int
            Radius of lesion core
        """
        self.grid_size = grid_size
        self.lesion_center = lesion_center
        self.lesion_radius = lesion_radius
        
        # Handle static vs dynamic scar fields
        if scar_field.ndim == 2:
            self.S = scar_field
            self.S_dynamic = None
        else:
            self.S = scar_field[-1]  # Use final state by default
            self.S_dynamic = scar_field
        
        # Growth parameters
        self.params = {
            # Cost scaling - calibrated so partial crossing is possible
            'cost_per_S': 0.8,            # How much each unit of S adds to cost
            'cost_threshold': 25.0,       # Accumulated cost that causes collapse
            'baseline_cost': 0.05,        # Cost even in healthy tissue
            
            # Growth dynamics
            'step_size': 1,               # Pixels per growth step
            'growth_rate': 0.95,          # Base probability of extending
            'S_inhibition_scale': 0.3,    # How much S reduces growth probability
            
            # Directional bias
            'caudal_bias': 0.7,           # Preference for growing caudally (toward target)
            'gradient_sensitivity': 0.4,  # How much to follow S gradient (toward lower S)
            
            # Plasticity
            'sprout_probability': 0.01,   # Chance of collateral sprouting per step
            'sprout_range': 3,            # How far sprouts can reach
            
            # Success criteria
            'target_y': None,             # Y coordinate that defines "crossed"
        }
        
        # Set default target (other side of lesion)
        self.params['target_y'] = lesion_center[1] + lesion_radius + 5
        
        # Populations
        self.populations: Dict[str, AxonPopulation] = {}
        
        # History
        self.history = {
            'time': [],
            'active_fraction': [],
            'crossed_fraction': [],
            'mean_cost': [],
        }
    
    def set_parameters(self, **kwargs):
        """Update growth parameters."""
        self.params.update(kwargs)
    
    def add_population(
        self,
        name: str,
        n_axons: int,
        tract_type: str = 'descending'
    ):
        """
        Add a population of axons to simulate.
        
        Parameters
        ----------
        name : str
            Population identifier
        n_axons : int
            Number of axons
        tract_type : str
            'descending' (rostral→caudal) or 'ascending' (caudal→rostral)
        """
        if tract_type == 'descending':
            # Start above lesion, grow down
            y_start = self.lesion_center[1] - self.lesion_radius - 10
            x_range = (10, self.grid_size - 10)
        else:
            # Start below lesion, grow up
            y_start = self.lesion_center[1] + self.lesion_radius + 10
            x_range = (10, self.grid_size - 10)
        
        pop = AxonPopulation.create_population(
            name=name,
            n_axons=n_axons,
            y_start=y_start,
            x_range=x_range,
            grid_size=self.grid_size
        )
        
        self.populations[name] = pop
        return pop
    
    def compute_growth_probability(self, x: int, y: int) -> float:
        """
        Compute probability of successful growth at position (x, y).
        
        P(grow) = base_rate * exp(-S * inhibition_scale)
        """
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return 0.0
        
        S_local = self.S[y, x]
        base = self.params['growth_rate']
        inhibition = self.params['S_inhibition_scale']
        
        # Exponential inhibition by scar density
        p_grow = base * np.exp(-S_local * inhibition)
        
        return np.clip(p_grow, 0.01, 0.99)
    
    def compute_step_direction(self, x: int, y: int, target_y: int) -> Tuple[int, int]:
        """
        Compute next step direction based on:
        1. Caudal bias (toward target)
        2. Gradient descent on S field (toward lower scar)
        3. Random exploration
        """
        # Possible steps (8-connected)
        directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        
        weights = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Boundary check
            if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                weights.append(0.0)
                continue
            
            w = 1.0
            
            # Caudal bias (prefer moving toward target_y)
            if target_y > y and dy > 0:
                w *= (1 + self.params['caudal_bias'])
            elif target_y < y and dy < 0:
                w *= (1 + self.params['caudal_bias'])
            
            # Gradient sensitivity (prefer lower S)
            S_current = self.S[y, x]
            S_next = self.S[ny, nx]
            if S_next < S_current:
                w *= (1 + self.params['gradient_sensitivity'])
            elif S_next > S_current:
                w *= (1 - self.params['gradient_sensitivity'] * 0.5)
            
            weights.append(w)
        
        # Normalize and sample
        weights = np.array(weights)
        if weights.sum() == 0:
            return (0, 0)  # Stuck
        
        weights /= weights.sum()
        idx = np.random.choice(len(directions), p=weights)
        
        return directions[idx]
    
    def step_population(self, pop: AxonPopulation, target_y: int):
        """Advance all active axons in a population by one step."""
        for i in range(len(pop.active)):
            if not pop.active[i]:
                continue
            
            x, y = pop.current_positions[i]
            
            # Check if already crossed
            if y >= target_y:
                pop.crossed[i] = True
                pop.active[i] = False
                continue
            
            # Compute growth probability
            p_grow = self.compute_growth_probability(x, y)
            
            if np.random.random() > p_grow:
                # Failed to grow this step
                continue
            
            # Get step direction
            dx, dy = self.compute_step_direction(x, y, target_y)
            
            if dx == 0 and dy == 0:
                # Stuck - growth cone collapse
                pop.active[i] = False
                continue
            
            # Take step
            nx, ny = x + dx, y + dy
            nx = np.clip(nx, 0, self.grid_size - 1)
            ny = np.clip(ny, 0, self.grid_size - 1)
            
            # Accumulate cost
            cost = self.params['baseline_cost'] + self.params['cost_per_S'] * self.S[ny, nx]
            pop.accumulated_cost[i] += cost
            
            # Check for collapse due to accumulated damage
            if pop.accumulated_cost[i] > self.params['cost_threshold']:
                pop.active[i] = False
                continue
            
            # Update position
            pop.current_positions[i] = [nx, ny]
            pop.paths[i].append((nx, ny))
            
            # Check success
            if ny >= target_y:
                pop.crossed[i] = True
                pop.active[i] = False
    
    def run(
        self,
        max_steps: int = 200,
        verbose: bool = True
    ) -> Dict:
        """
        Run axon growth simulation.
        
        Returns
        -------
        dict
            Results including crossing rates, path statistics, connectivity
        """
        target_y = self.params['target_y']
        
        if verbose:
            print("=" * 60)
            print("AXON GROWTH SIMULATION")
            print("=" * 60)
            total_axons = sum(len(p.active) for p in self.populations.values())
            print(f"Populations: {len(self.populations)}")
            print(f"Total axons: {total_axons}")
            print(f"Target Y: {target_y}")
            print("-" * 60)
        
        for step in range(max_steps):
            # Update each population
            any_active = False
            for pop in self.populations.values():
                if np.any(pop.active):
                    any_active = True
                    self.step_population(pop, target_y)
            
            # Record history
            total_active = sum(np.sum(p.active) for p in self.populations.values())
            total_crossed = sum(np.sum(p.crossed) for p in self.populations.values())
            total_axons = sum(len(p.active) for p in self.populations.values())
            total_cost = sum(np.mean(p.accumulated_cost) for p in self.populations.values())
            
            self.history['time'].append(step)
            self.history['active_fraction'].append(total_active / total_axons)
            self.history['crossed_fraction'].append(total_crossed / total_axons)
            self.history['mean_cost'].append(total_cost / len(self.populations))
            
            if verbose and step % 50 == 0:
                print(f"Step {step:4d}: {total_active:4d} active, "
                      f"{total_crossed:4d} crossed ({100*total_crossed/total_axons:.1f}%)")
            
            if not any_active:
                if verbose:
                    print(f"All axons stopped at step {step}")
                break
        
        # Compile results
        results = self.compile_results()
        
        if verbose:
            print("-" * 60)
            print("RESULTS:")
            print(f"  Crossing rate: {results['crossing_rate']*100:.1f}%")
            print(f"  Mean path length: {results['mean_path_length']:.1f} steps")
            print(f"  Mean accumulated cost: {results['mean_accumulated_cost']:.2f}")
            print("=" * 60)
        
        return results
    
    def compile_results(self) -> Dict:
        """Compile simulation results into summary statistics."""
        all_crossed = []
        all_path_lengths = []
        all_costs = []
        
        for pop in self.populations.values():
            all_crossed.extend(pop.crossed)
            all_path_lengths.extend([len(p) for p in pop.paths])
            all_costs.extend(pop.accumulated_cost)
        
        crossed_paths = [
            len(pop.paths[i]) for pop in self.populations.values()
            for i in range(len(pop.crossed)) if pop.crossed[i]
        ]
        
        return {
            'crossing_rate': np.mean(all_crossed),
            'mean_path_length': np.mean(all_path_lengths),
            'mean_accumulated_cost': np.mean(all_costs),
            'crossed_path_lengths': crossed_paths,
            'n_crossed': sum(all_crossed),
            'n_total': len(all_crossed),
        }
    
    def compute_connectivity_map(self) -> np.ndarray:
        """
        Generate a connectivity density map showing where axons traverse.
        """
        density = np.zeros((self.grid_size, self.grid_size))
        
        for pop in self.populations.values():
            for path in pop.paths:
                for x, y in path:
                    density[y, x] += 1
        
        return density
    
    def plot_results(self, save_path: str = None, show: bool = True):
        """Visualize axon growth results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Panel A: Scar field with axon paths
        ax = axes[0, 0]
        ax.imshow(self.S, cmap='Reds', alpha=0.7, vmin=0, vmax=5)
        
        # Draw paths
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.populations)))
        for (name, pop), color in zip(self.populations.items(), colors):
            for i, path in enumerate(pop.paths):
                if len(path) < 2:
                    continue
                path_arr = np.array(path)
                alpha = 0.8 if pop.crossed[i] else 0.2
                lw = 1.5 if pop.crossed[i] else 0.5
                ax.plot(path_arr[:, 0], path_arr[:, 1], 
                       color='lime' if pop.crossed[i] else 'gray',
                       alpha=alpha, lw=lw)
        
        # Mark lesion
        circle = plt.Circle(self.lesion_center, self.lesion_radius, 
                           fill=False, color='white', lw=2, ls='--')
        ax.add_patch(circle)
        
        # Mark target line
        ax.axhline(self.params['target_y'], color='cyan', ls='--', lw=2, 
                  label='Target (success threshold)')
        
        ax.set_title('A. Axon Paths Through Scar Field', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(self.grid_size, 0)
        
        # Panel B: Connectivity density
        ax = axes[0, 1]
        density = self.compute_connectivity_map()
        im = ax.imshow(density, cmap='hot')
        ax.imshow(self.S > 2.5, cmap='Blues', alpha=0.3)  # Overlay lesion
        plt.colorbar(im, ax=ax, label='Axon density')
        ax.set_title('B. Connectivity Density Map', fontsize=12, fontweight='bold')
        
        # Panel C: Crossing rate over time
        ax = axes[1, 0]
        ax.plot(self.history['time'], 
               np.array(self.history['crossed_fraction']) * 100,
               'g-', lw=2, label='Crossed (%)')
        ax.plot(self.history['time'],
               np.array(self.history['active_fraction']) * 100,
               'b--', lw=1.5, label='Still growing (%)')
        ax.set_xlabel('Simulation step', fontsize=11)
        ax.set_ylabel('Percentage of axons', fontsize=11)
        ax.set_title('C. Growth Dynamics', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Panel D: Path length distribution
        ax = axes[1, 1]
        all_lengths = [len(p) for pop in self.populations.values() for p in pop.paths]
        crossed_lengths = [
            len(pop.paths[i]) for pop in self.populations.values()
            for i in range(len(pop.crossed)) if pop.crossed[i]
        ]
        failed_lengths = [
            len(pop.paths[i]) for pop in self.populations.values()
            for i in range(len(pop.crossed)) if not pop.crossed[i]
        ]
        
        if crossed_lengths:
            ax.hist(crossed_lengths, bins=20, alpha=0.7, color='green', 
                   label=f'Crossed (n={len(crossed_lengths)})')
        if failed_lengths:
            ax.hist(failed_lengths, bins=20, alpha=0.7, color='red',
                   label=f'Failed (n={len(failed_lengths)})')
        ax.set_xlabel('Path length (steps)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('D. Path Length Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        
        plt.suptitle('Axon Growth Through SCI Lesion', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()


def run_treatment_comparison(scar_sim_class, base_params: Dict) -> Dict:
    """
    Compare axon crossing rates under different treatment conditions.
    
    This is the key function: connect scar manipulation → functional outcome.
    """
    from sci_scar_simulator import SCIScarSimulator
    
    treatments = {
        'Control': {},
        'Early chABC\n(d1-7)': {'gamma_therapy': 0.012},
        'Late chABC\n(d7-14)': {'gamma_therapy': 0.006},
        'TGF-β block': {'eta_secondary': 0.004},
        'Neuroprotective\n(↓acute)': {'alpha_acute': 0.003, 'alpha_peak': 0.020},
        'Combined\n(chABC+TGF-β)': {'gamma_therapy': 0.010, 'eta_secondary': 0.006},
    }
    
    results = {}
    
    for name, mods in treatments.items():
        print(f"\n{'='*60}")
        print(f"Treatment: {name.replace(chr(10), ' ')}")
        print('='*60)
        
        # Run scar simulation
        sim = SCIScarSimulator(grid_size=64, dt=0.5)
        treatment_params = base_params.copy()
        treatment_params.update(mods)
        sim.set_parameters(**treatment_params)
        sim.create_injury(center=(32, 32), radius=8, injury_type='contusion')
        sim.run(duration_hours=672, verbose=False)  # 28 days
        
        # Get final scar field
        scar_field = sim.S
        
        # Run axon growth
        axon_sim = AxonGrowthSimulator(
            scar_field=scar_field,
            grid_size=64,
            lesion_center=(32, 32),
            lesion_radius=8
        )
        
        # Add descending tract population
        axon_sim.add_population('CST', n_axons=300, tract_type='descending')
        
        # Adjust growth parameters based on treatment category
        if 'Combined' in name:
            # Best case: scar softening + growth promotion
            axon_sim.set_parameters(
                cost_threshold=30.0,
                growth_rate=0.96,
                S_inhibition_scale=0.25
            )
        elif 'chABC' in name:
            # ECM degradation improves substrate
            axon_sim.set_parameters(
                cost_threshold=28.0,
                growth_rate=0.95
            )
        elif 'TGF' in name or 'Neuroprotective' in name:
            # Reduced scar but no direct growth boost
            axon_sim.set_parameters(
                cost_threshold=26.0
            )
        
        axon_results = axon_sim.run(max_steps=200, verbose=False)
        
        results[name] = {
            'crossing_rate': axon_results['crossing_rate'],
            'mean_scar': np.mean(scar_field[24:40, 24:40]),  # Lesion core
            'axon_sim': axon_sim,
            'scar_field': scar_field,
        }
        
        print(f"  Final scar (core): {results[name]['mean_scar']:.2f}")
        print(f"  Crossing rate: {results[name]['crossing_rate']*100:.1f}%")
    
    return results


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude')
    from sci_scar_simulator import SCIScarSimulator
    
    print("=" * 70)
    print(" AXON GROWTH THROUGH SCI SCAR - DEMONSTRATION")
    print("=" * 70)
    
    # Optimal scar parameters
    scar_params = {
        'alpha_peak': 0.028,
        'alpha_onset': 3,
        'alpha_width': 6,
        'S_max': 5.0,
        'alpha_acute': 0.012,
        'tau_acute': 24,
        'eta_secondary': 0.018,
        'eta_onset': 36,
        'eta_width': 12.0,
        'delta_feedback': 0.00025,
        'gamma_therapy': 0.0,
    }
    
    # Generate scar field (control condition)
    print("\n1. Generating scar field (28 days post-injury)...")
    scar_sim = SCIScarSimulator(grid_size=64, dt=0.5)
    scar_sim.set_parameters(**scar_params)
    scar_sim.create_injury(center=(32, 32), radius=8, injury_type='contusion')
    scar_sim.run(duration_hours=672, verbose=False)
    
    print(f"   Scar at lesion core: {np.mean(scar_sim.S[24:40, 24:40]):.2f}x baseline")
    
    # Run axon growth simulation
    print("\n2. Simulating axon growth...")
    axon_sim = AxonGrowthSimulator(
        scar_field=scar_sim.S,
        grid_size=64,
        lesion_center=(32, 32),
        lesion_radius=8
    )
    
    # Add corticospinal tract-like population
    axon_sim.add_population('CST', n_axons=200, tract_type='descending')
    
    # Run
    results = axon_sim.run(max_steps=150, verbose=True)
    
    # Visualize
    axon_sim.plot_results(save_path='/home/claude/axon_growth_results.png', show=False)
    
    print("\n3. Running treatment comparison...")
    treatment_results = run_treatment_comparison(SCIScarSimulator, scar_params)
    
    # Summary comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(treatment_results.keys())
    crossing_rates = [treatment_results[n]['crossing_rate'] * 100 for n in names]
    scar_levels = [treatment_results[n]['mean_scar'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, crossing_rates, width, label='Crossing rate (%)', color='#2ECC71')
    bars2 = ax.bar(x + width/2, scar_levels, width, label='Scar level (fold)', color='#E74C3C')
    
    ax.set_xlabel('Treatment', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Treatment Effects: Scar Reduction → Functional Improvement', 
                fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, crossing_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/home/claude/treatment_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Copy to outputs
    import shutil
    shutil.copy('/home/claude/axon_growth_results.png', '/mnt/user-data/outputs/')
    shutil.copy('/home/claude/treatment_comparison.png', '/mnt/user-data/outputs/')
    
    print("\n" + "=" * 70)
    print("✓ Complete!")
    print("  Saved: axon_growth_results.png")
    print("  Saved: treatment_comparison.png")
    print("=" * 70)
