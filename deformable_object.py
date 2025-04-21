from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.sph.integrator import EPECIntegrator, PECIntegrator
from pysph.sph.integrator_step import WCSPHStep
from pysph.sph.equation import Group
from pysph.sph.basic_equations import (
    ContinuityEquation, XSPHCorrection
)
from pysph.sph.wc.basic import (
    TaitEOS, MomentumEquation
)
from pysph.sph.solid_mech.basic import (
    HookesDeviatoricStressRate,
    MomentumEquationWithStress, MonaghanArtificialStress
)
import numpy as np

class DeformableObjectSim(Application):
    def create_particles(self):
        # Define a deformable cube as SPH particles
        dx = 0.05
        x, y, z = np.mgrid[-0.5:0.5:dx, -0.5:0.5:dx, -0.5:0.5:dx]
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()
        
        # Initialize stress tensor components (for 3D)
        n = len(x)
        zeros = np.zeros_like(x)
        
        return self.get_particles(
            arrays={
                'x': x, 'y': y, 'z': z,
                'u': zeros,  # velocity x
                'v': zeros,  # velocity y
                'w': zeros,  # velocity z
                'rho': np.ones_like(x) * 1000,  # density
                'm': np.ones_like(x) * dx**3 * 1000,  # mass
                'h': np.ones_like(x) * dx * 1.2,  # smoothing length
                'p': zeros,  # pressure
                # Stress tensor components
                's00': zeros, 's01': zeros, 's02': zeros,
                's10': zeros, 's11': zeros, 's12': zeros,
                's20': zeros, 's21': zeros, 's22': zeros,
                # Elastic properties
                'e': np.ones_like(x) * 1e6,  # Young's modulus (Pa)
                'nu': np.ones_like(x) * 0.45,  # Poisson's ratio
                'cs': np.ones_like(x) * 50.0,  # Speed of sound
            },
            name='object'
        )

    def create_equations(self):
        equations = [
            Group(equations=[
                # Continuity equation
                ContinuityEquation(dest='object', sources=['object']),
                
                # Equation of state (Tait equation for weakly compressible)
                TaitEOS(
                    dest='object', sources=None, 
                    rho0=1000.0, c0=50.0, gamma=7.0
                ),
                
                # Stress rate equations (Hooke's law)
                HookesDeviatoricStressRate(
                    dest='object', sources=None
                ),
                
                # Momentum equation with stress terms
                MomentumEquationWithStress(
                    dest='object', sources=['object'],
                    alpha=0.1, beta=0.1
                ),
                
                # Artificial stress to prevent tensile instability
                MonaghanArtificialStress(
                    dest='object', sources=['object'],
                    eps=0.3
                ),
                
                # XSPH for smoother motion
                XSPHCorrection(
                    dest='object', sources=['object'],
                    eps=0.5
                )
            ], real=True)
        ]
        return equations

    def create_solver(self):
        kernel = CubicSpline(dim=3)
        integrator = EPECIntegrator(elastic=WCSPHStep())
        solver = Solver(dim=3, integrator=integrator, kernel=kernel,
                n_damp=50, tf=1.0, dt=1e-3, adaptive_timestep=True,
                pfreq=100, cfl=0.5, output_at_times=[1e-1, 1.0])
        
        return solver