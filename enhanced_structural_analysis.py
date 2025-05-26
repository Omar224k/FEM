"""
Comprehensive Structural Analysis Program using Stiffness Matrix Method
under supervision of Prof. Mostafa Shawky 
Done by Students Omar Anwer - Mahmoud Khaled - Zakaria khedr
This program performs structural analysis of determinate/indeterminate frames and trusses
using the Direct Stiffness Method. It can handle:
- 2D Frame elements (with bending)
- 2D Truss elements (axial only)
- Multiple elements and nodes
- Various support conditions
- Point loads and distributed loads
- Calculation of displacements, reactions, and internal forces

Author: Enhanced from existing structural_stiffness.py
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json

@dataclass
class Node:
    """Represents a structural node with coordinates and DOF information"""
    id: int
    x: float
    y: float
    restraints: List[bool] = None  # [u_x, u_y, theta] restraints
    
    def __post_init__(self):
        if self.restraints is None:
            self.restraints = [False, False, False]  # Free by default

@dataclass
class Element:
    """Represents a structural element (beam/truss)"""
    id: int
    node_i: int
    node_j: int
    E: float  # Young's modulus
    A: float  # Cross-sectional area
    I: float = 0.0  # Moment of inertia (0 for truss elements)
    element_type: str = 'frame'  # 'frame' or 'truss'
    
@dataclass
class Load:
    """Represents a load applied to the structure"""
    node_id: int
    dof: int  # 0=u_x, 1=u_y, 2=theta
    magnitude: float
    load_type: str = 'point'  # 'point' or 'distributed'

class StructuralAnalyzer:
    """Main class for structural analysis using stiffness matrix method"""
    
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.elements: Dict[int, Element] = {}
        self.loads: List[Load] = []
        self.global_K = None
        self.displacements = None
        self.reactions = None
        self.element_forces = {}
        
    def add_node(self, node_id: int, x: float, y: float, restraints: List[bool] = None):
        """Add a node to the structure"""
        self.nodes[node_id] = Node(node_id, x, y, restraints)
        
    def add_element(self, elem_id: int, node_i: int, node_j: int, E: float, A: float, I: float = 0.0, element_type: str = 'frame'):
        """Add an element to the structure"""
        self.elements[elem_id] = Element(elem_id, node_i, node_j, E, A, I, element_type)
        
    def add_load(self, node_id: int, dof: int, magnitude: float, load_type: str = 'point'):
        """Add a load to the structure"""
        self.loads.append(Load(node_id, dof, magnitude, load_type))
        
    def get_element_length_angle(self, element: Element) -> Tuple[float, float]:
        """Calculate element length and angle"""
        node_i = self.nodes[element.node_i]
        node_j = self.nodes[element.node_j]
        
        dx = node_j.x - node_i.x
        dy = node_j.y - node_i.y
        length = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        
        return length, angle
        
    def frame_element_stiffness(self, element: Element) -> np.ndarray:
        """Calculate stiffness matrix for frame element (6x6)"""
        L, angle = self.get_element_length_angle(element)
        c = np.cos(angle)
        s = np.sin(angle)
        
        E, A, I = element.E, element.A, element.I
        
        # Local stiffness matrix components
        EA_L = E * A / L
        EI_L = E * I / L
        EI_L2 = EI_L / L
        EI_L3 = EI_L2 / L
        
        # Local stiffness matrix (6x6)
        k_local = np.array([
            [ EA_L,       0,        0,    -EA_L,        0,        0],
            [    0,  12*EI_L3,  6*EI_L2,       0,  -12*EI_L3,  6*EI_L2],
            [    0,   6*EI_L2,   4*EI_L,       0,   -6*EI_L2,   2*EI_L],
            [-EA_L,       0,        0,     EA_L,        0,        0],
            [    0, -12*EI_L3, -6*EI_L2,       0,   12*EI_L3, -6*EI_L2],
            [    0,   6*EI_L2,   2*EI_L,       0,   -6*EI_L2,   4*EI_L]
        ])
        
        # Transformation matrix (6x6)
        T = np.zeros((6, 6))
        T_sub = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        T[:3, :3] = T_sub
        T[3:, 3:] = T_sub
        
        # Global stiffness matrix
        return T.T @ k_local @ T
        
    def truss_element_stiffness(self, element: Element) -> np.ndarray:
        """Calculate stiffness matrix for truss element (4x4 -> 6x6 with zeros for rotation)"""
        L, angle = self.get_element_length_angle(element)
        c = np.cos(angle)
        s = np.sin(angle)
        
        EA_L = element.E * element.A / L
        
        # Local stiffness matrix for truss (2x2 per node, no rotation)
        k_local_2x2 = EA_L * np.array([
            [ 1, -1],
            [-1,  1]
        ])
        
        # Expand to 6x6 with rotation DOFs as zeros
        k_local = np.zeros((6, 6))
        # Axial terms
        k_local[0, 0] = k_local[0, 3] = k_local[3, 0] = k_local[3, 3] = EA_L
        k_local[0, 3] = k_local[3, 0] = -EA_L
        
        # Transformation matrix
        T = np.zeros((6, 6))
        T_sub = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        T[:3, :3] = T_sub
        T[3:, 3:] = T_sub
        
        return T.T @ k_local @ T
        
    def assemble_global_stiffness(self):
        """Assemble the global stiffness matrix"""
        n_nodes = len(self.nodes)
        n_dofs = 3 * n_nodes  # 3 DOFs per node (u_x, u_y, theta)
        
        self.global_K = np.zeros((n_dofs, n_dofs))
        
        for element in self.elements.values():
            if element.element_type == 'frame':
                k_elem = self.frame_element_stiffness(element)
            else:  # truss
                k_elem = self.truss_element_stiffness(element)
                
            # Get global DOF indices
            node_i_dofs = [3 * element.node_i, 3 * element.node_i + 1, 3 * element.node_i + 2]
            node_j_dofs = [3 * element.node_j, 3 * element.node_j + 1, 3 * element.node_j + 2]
            global_dofs = node_i_dofs + node_j_dofs
            
            # Assemble into global matrix
            for i, gi in enumerate(global_dofs):
                for j, gj in enumerate(global_dofs):
                    self.global_K[gi, gj] += k_elem[i, j]
                    
    def apply_loads(self) -> np.ndarray:
        """Create the global load vector"""
        n_dofs = 3 * len(self.nodes)
        F = np.zeros(n_dofs)
        
        for load in self.loads:
            global_dof = 3 * load.node_id + load.dof
            F[global_dof] += load.magnitude
            
        return F
        
    def get_boundary_conditions(self) -> Tuple[List[int], List[int]]:
        """Get lists of fixed and free DOFs"""
        fixed_dofs = []
        free_dofs = []
        
        for node_id, node in self.nodes.items():
            for i, is_fixed in enumerate(node.restraints):
                global_dof = 3 * node_id + i
                if is_fixed:
                    fixed_dofs.append(global_dof)
                else:
                    free_dofs.append(global_dof)
                    
        return fixed_dofs, free_dofs
        
    def solve(self):
        """Solve the structural system"""
        # Assemble global stiffness matrix
        self.assemble_global_stiffness()
        
        # Apply loads
        F = self.apply_loads()
        
        # Get boundary conditions
        fixed_dofs, free_dofs = self.get_boundary_conditions()
        
        # Partition system
        Kff = self.global_K[np.ix_(free_dofs, free_dofs)]
        Ff = F[free_dofs]
        
        # Solve for free displacements
        try:
            Uf = np.linalg.solve(Kff, Ff)
        except np.linalg.LinAlgError:
            raise ValueError("Singular stiffness matrix - structure is unstable or improperly constrained")
            
        # Assemble full displacement vector
        n_dofs = 3 * len(self.nodes)
        self.displacements = np.zeros(n_dofs)
        self.displacements[free_dofs] = Uf
        
        # Calculate reactions
        self.reactions = self.global_K @ self.displacements - F
        
        # Calculate element forces
        self.calculate_element_forces()
        
    def calculate_element_forces(self):
        """Calculate internal forces in each element"""
        self.element_forces = {}
        
        for elem_id, element in self.elements.items():
            # Get element displacements
            node_i_dofs = [3 * element.node_i, 3 * element.node_i + 1, 3 * element.node_i + 2]
            node_j_dofs = [3 * element.node_j, 3 * element.node_j + 1, 3 * element.node_j + 2]
            elem_dofs = node_i_dofs + node_j_dofs
            elem_displacements = self.displacements[elem_dofs]
            
            # Get element stiffness matrix
            if element.element_type == 'frame':
                k_elem = self.frame_element_stiffness(element)
            else:
                k_elem = self.truss_element_stiffness(element)
                
            # Calculate element forces
            elem_forces = k_elem @ elem_displacements
            
            # Store forces with labels
            self.element_forces[elem_id] = {
                'node_i': {
                    'Fx': elem_forces[0],
                    'Fy': elem_forces[1], 
                    'Mz': elem_forces[2]
                },
                'node_j': {
                    'Fx': elem_forces[3],
                    'Fy': elem_forces[4],
                    'Mz': elem_forces[5]
                }
            }
            
    def get_results_summary(self) -> Dict:
        """Get a summary of analysis results"""
        if self.displacements is None:
            return {"error": "Analysis not performed yet. Call solve() first."}
            
        results = {
            'nodes': {},
            'elements': {},
            'max_displacement': np.max(np.abs(self.displacements)),
            'max_reaction': np.max(np.abs(self.reactions))
        }
        
        # Node results
        for node_id in self.nodes.keys():
            base_dof = 3 * node_id
            results['nodes'][node_id] = {
                'displacements': {
                    'u_x': self.displacements[base_dof],
                    'u_y': self.displacements[base_dof + 1],
                    'theta': self.displacements[base_dof + 2]
                },
                'reactions': {
                    'R_x': self.reactions[base_dof],
                    'R_y': self.reactions[base_dof + 1],
                    'M_z': self.reactions[base_dof + 2]
                }
            }
            
        # Element results
        results['elements'] = self.element_forces
        
        return results
        
    def plot_structure(self, show_deformed=True, scale_factor=100):
        """Plot the structure with optional deformed shape"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original structure
        ax1.set_title('Original Structure')
        self._plot_structure_on_axis(ax1, deformed=False)
        
        # Deformed structure
        if show_deformed and self.displacements is not None:
            ax2.set_title(f'Deformed Structure (Scale: {scale_factor}x)')
            self._plot_structure_on_axis(ax2, deformed=True, scale_factor=scale_factor)
        else:
            ax2.set_title('Loads and Supports')
            self._plot_structure_on_axis(ax2, deformed=False, show_loads=True)
            
        plt.tight_layout()
        return fig
        
    def _plot_structure_on_axis(self, ax, deformed=False, scale_factor=100, show_loads=False):
        """Helper method to plot structure on given axis"""
        # Plot elements
        for element in self.elements.values():
            node_i = self.nodes[element.node_i]
            node_j = self.nodes[element.node_j]
            
            if deformed and self.displacements is not None:
                # Add displacements
                xi = node_i.x + scale_factor * self.displacements[3 * element.node_i]
                yi = node_i.y + scale_factor * self.displacements[3 * element.node_i + 1]
                xj = node_j.x + scale_factor * self.displacements[3 * element.node_j]
                yj = node_j.y + scale_factor * self.displacements[3 * element.node_j + 1]
            else:
                xi, yi = node_i.x, node_i.y
                xj, yj = node_j.x, node_j.y
                
            # Different colors for different element types
            color = 'blue' if element.element_type == 'frame' else 'red'
            linewidth = 3 if element.element_type == 'frame' else 2
            
            ax.plot([xi, xj], [yi, yj], color=color, linewidth=linewidth, 
                   label=f'Element {element.id}' if element.id == list(self.elements.keys())[0] else "")
                   
        # Plot nodes
        for node in self.nodes.values():
            if deformed and self.displacements is not None:
                x = node.x + scale_factor * self.displacements[3 * node.id]
                y = node.y + scale_factor * self.displacements[3 * node.id + 1]
            else:
                x, y = node.x, node.y
                
            ax.plot(x, y, 'ko', markersize=8)
            ax.text(x + 0.1, y + 0.1, f'N{node.id}', fontsize=10)
            
            # Show supports
            if any(node.restraints):
                if node.restraints[0] and node.restraints[1]:  # Fixed support
                    triangle = patches.RegularPolygon((x, y-0.2), 3, 0.1, 
                                                     orientation=0, facecolor='black')
                    ax.add_patch(triangle)
                elif node.restraints[1]:  # Roller support
                    circle = patches.Circle((x, y-0.15), 0.05, facecolor='white', 
                                          edgecolor='black')
                    ax.add_patch(circle)
                    
        # Show loads
        if show_loads:
            for load in self.loads:
                node = self.nodes[load.node_id]
                if load.dof == 0:  # Horizontal force
                    ax.arrow(node.x, node.y, 0.3 * np.sign(load.magnitude), 0, 
                            head_width=0.1, head_length=0.05, fc='green', ec='green')
                elif load.dof == 1:  # Vertical force
                    ax.arrow(node.x, node.y, 0, 0.3 * np.sign(load.magnitude), 
                            head_width=0.1, head_length=0.05, fc='green', ec='green')
                            
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
    def export_results(self, filename: str):
        """Export results to JSON file"""
        results = self.get_results_summary()
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
            
    def validate_with_analytical(self, analytical_results: Dict) -> Dict:
        """Compare results with analytical solutions"""
        if self.displacements is None:
            return {"error": "Analysis not performed yet"}
            
        comparison = {}
        
        # Compare displacements
        for node_id in analytical_results.get('displacements', {}):
            if node_id in self.nodes:
                numerical = self.get_results_summary()['nodes'][node_id]['displacements']
                analytical = analytical_results['displacements'][node_id]
                
                comparison[f'node_{node_id}_displacement'] = {
                    'numerical': numerical,
                    'analytical': analytical,
                    'error_percent': {}
                }
                
                for dof in ['u_x', 'u_y', 'theta']:
                    if analytical[dof] != 0:
                        error = abs((numerical[dof] - analytical[dof]) / analytical[dof]) * 100
                        comparison[f'node_{node_id}_displacement']['error_percent'][dof] = error
                        
        return comparison

# Example usage and validation functions
def example_cantilever_beam():
    """Example: Cantilever beam with tip load"""
    analyzer = StructuralAnalyzer()
    
    # Add nodes
    analyzer.add_node(0, 0, 0, [True, True, True])  # Fixed support
    analyzer.add_node(1, 4, 0, [False, False, False])  # Free end
    
    # Add frame element
    E = 200e6  # kN/m²
    A = 0.01   # m²
    I = 8.33e-6  # m⁴
    analyzer.add_element(0, 0, 1, E, A, I, 'frame')
    
    # Add load (10 kN downward at tip)
    analyzer.add_load(1, 1, -10)  # Node 1, DOF 1 (y), -10 kN
    
    # Solve
    analyzer.solve()
    
    # Analytical solution for comparison
    L = 4.0
    P = 10.0
    analytical_tip_deflection = -P * L**3 / (3 * E * I)  # Negative (downward)
    analytical_tip_rotation = -P * L**2 / (2 * E * I)    # Negative (clockwise)
    
    print("CANTILEVER BEAM ANALYSIS")
    print("=" * 50)
    print(f"Numerical tip deflection: {analyzer.displacements[4]:.6f} m")
    print(f"Analytical tip deflection: {analytical_tip_deflection:.6f} m")
    print(f"Error: {abs(analyzer.displacements[4] - analytical_tip_deflection)/abs(analytical_tip_deflection)*100:.2f}%")
    print()
    print(f"Numerical tip rotation: {analyzer.displacements[5]:.6f} rad")
    print(f"Analytical tip rotation: {analytical_tip_rotation:.6f} rad")
    print(f"Error: {abs(analyzer.displacements[5] - analytical_tip_rotation)/abs(analytical_tip_rotation)*100:.2f}%")
    
    return analyzer

def example_simple_truss():
    """Example: Simple 2D truss"""
    analyzer = StructuralAnalyzer()
    
    # Add nodes
    analyzer.add_node(0, 0, 0, [True, True, False])    # Pin support
    analyzer.add_node(1, 4, 0, [False, True, False])   # Roller support
    analyzer.add_node(2, 2, 3, [False, False, False])  # Free node
    
    # Add truss elements
    E = 200e6  # kN/m²
    A = 0.01   # m²
    analyzer.add_element(0, 0, 2, E, A, 0, 'truss')  # Bottom left to top
    analyzer.add_element(1, 1, 2, E, A, 0, 'truss')  # Bottom right to top
    analyzer.add_element(2, 0, 1, E, A, 0, 'truss')  # Bottom chord
    
    # Add load (20 kN downward at top)
    analyzer.add_load(2, 1, -20)  # Node 2, DOF 1 (y), -20 kN
    
    # Solve
    analyzer.solve()
    
    print("\nSIMPLE TRUSS ANALYSIS")
    print("=" * 50)
    results = analyzer.get_results_summary()
    
    # Print node displacements
    for node_id, node_results in results['nodes'].items():
        print(f"Node {node_id}:")
        print(f"  Displacement: u_x = {node_results['displacements']['u_x']:.6f} m")
        print(f"                u_y = {node_results['displacements']['u_y']:.6f} m")
        if any(analyzer.nodes[node_id].restraints):
            print(f"  Reactions: R_x = {node_results['reactions']['R_x']:.2f} kN")
            print(f"             R_y = {node_results['reactions']['R_y']:.2f} kN")
        print()
    
    return analyzer

def example_indeterminate_frame():
    """Example: Indeterminate frame"""
    analyzer = StructuralAnalyzer()
    
    # Add nodes for a portal frame
    analyzer.add_node(0, 0, 0, [True, True, True])    # Fixed support left
    analyzer.add_node(1, 6, 0, [True, True, True])    # Fixed support right
    analyzer.add_node(2, 0, 4, [False, False, False]) # Top left
    analyzer.add_node(3, 6, 4, [False, False, False]) # Top right
    
    # Add frame elements
    E = 200e6  # kN/m²
    A = 0.02   # m²
    I = 1.67e-4  # m⁴
    
    analyzer.add_element(0, 0, 2, E, A, I, 'frame')  # Left column
    analyzer.add_element(1, 1, 3, E, A, I, 'frame')  # Right column
    analyzer.add_element(2, 2, 3, E, A, I, 'frame')  # Beam
    
    # Add loads
    analyzer.add_load(2, 0, 15)   # 15 kN horizontal at top left
    analyzer.add_load(3, 1, -25) # 25 kN vertical at top right
    
    # Solve
    analyzer.solve()
    
    print("\nINDETERMINATE FRAME ANALYSIS")
    print("=" * 50)
    results = analyzer.get_results_summary()
    
    # Print results
    for node_id, node_results in results['nodes'].items():
        print(f"Node {node_id}:")
        print(f"  Displacement: u_x = {node_results['displacements']['u_x']:.6f} m")
        print(f"                u_y = {node_results['displacements']['u_y']:.6f} m")
        print(f"                θ = {node_results['displacements']['theta']:.6f} rad")
        if any(analyzer.nodes[node_id].restraints):
            print(f"  Reactions: R_x = {node_results['reactions']['R_x']:.2f} kN")
            print(f"             R_y = {node_results['reactions']['R_y']:.2f} kN")
            print(f"             M_z = {node_results['reactions']['M_z']:.2f} kN·m")
        print()
    
    return analyzer

if __name__ == "__main__":
    # Run examples
    print("COMPREHENSIVE STRUCTURAL ANALYSIS EXAMPLES")
    print("=" * 60)
    
    # Example 1: Cantilever beam
    cantilever = example_cantilever_beam()
    
    # Example 2: Simple truss
    truss = example_simple_truss()
    
    # Example 3: Indeterminate frame
    frame = example_indeterminate_frame()
    
    # Plot structures
    fig1 = cantilever.plot_structure()
    fig1.suptitle('Cantilever Beam Analysis')
    
    fig2 = truss.plot_structure()
    fig2.suptitle('Simple Truss Analysis')
    
    fig3 = frame.plot_structure()
    fig3.suptitle('Indeterminate Frame Analysis')
    
    plt.show()
