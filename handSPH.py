#!/usr/bin/env python3
import numpy as np

from pysph.base.utils            import get_particle_array
from pysph.solver.application    import Application
from pysph.sph.scheme            import SchemeChooser
from pysph.sph.solid_mech.basic  import (
    ElasticSolidsScheme,
    get_particle_array_elastic_dynamics
)
from pysph.sph.equation          import Group
from pysph.sph.basic_equations   import BodyForce

class GraspDeformableBlock(Application):
    def initialize(self):
        # Simulation parameters
        self.dim    = 3
        self.dx     = 0.01
        self.hdx    = 1.3
        self.rho0   = 1000.0

        # Material: rubber‐like block
        self.E_block = 1e7    # 10 MPa
        self.nu      = 0.49
        self.c0      = 50.0

        # Geometry (m)
        self.block_size    = (0.3, 0.2, 0.1)
        self.platform_size = (1.0, 0.6, 0.04)   # 4 layers @ dx=0.01
        self.gripper_size  = (0.05, 0.1, 0.2)

    def create_box(self, center, size):
        nx = max(3, int(round(size[0]/self.dx)))
        ny = max(3, int(round(size[1]/self.dx)))
        nz = max(3, int(round(size[2]/self.dx)))
        xs = np.linspace(center[0] - size[0]/2 + self.dx/2,
                         center[0] + size[0]/2 - self.dx/2, nx)
        ys = np.linspace(center[1] - size[1]/2 + self.dx/2,
                         center[1] + size[1]/2 - self.dx/2, ny)
        zs = np.linspace(center[2] - size[2]/2 + self.dx/2,
                         center[2] + size[2]/2 - self.dx/2, nz)
        x,y,z = np.meshgrid(xs, ys, zs, indexing='ij')
        return x.ravel(), y.ravel(), z.ravel()

    def create_particles(self):
        # 1) Deformable block
        bx, by, bz = self.create_box(
            center=(0, 0, self.platform_size[2] + 0.5*self.block_size[2]),
            size=self.block_size
        )
        block = get_particle_array_elastic_dynamics(
            name='block',
            x=bx, y=by, z=bz,
            h=self.hdx*self.dx,
            m=self.dx**3 * self.rho0,
            rho=self.rho0,
            constants={
                'E':       self.E_block,
                'nu':      self.nu,
                'rho_ref': self.rho0,
                'c0_ref':  self.c0
            }
        )

        # 2) Platform & gripper jaws as stiff, heavy elastic solids
        def make_solid(name, center, size, E, mass_factor):
            x,y,z = self.create_box(center, size)
            return get_particle_array_elastic_dynamics(
                name=name,
                x=x, y=y, z=z,
                h=self.hdx*self.dx,
                m=self.dx**3 * self.rho0 * mass_factor,
                rho=self.rho0,
                constants={
                    'E':       E,
                    'nu':      self.nu,
                    'rho_ref': self.rho0,
                    'c0_ref':  self.c0
                }
            )

        # Make table & jaws 100× stiffer and 1000× heavier per particle
        E_rigid    = 1e9
        mass_factor= 1e3

        platform = make_solid(
            'platform',
            (0, 0, self.platform_size[2]/2),
            self.platform_size,
            E_rigid, mass_factor
        )
        gripper1 = make_solid(
            'gripper1',
            (-0.4, 0,
             self.platform_size[2] + 0.5*self.gripper_size[2]),
            self.gripper_size,
            E_rigid, mass_factor
        )
        gripper2 = make_solid(
            'gripper2',
            ( 0.4, 0,
             self.platform_size[2] + 0.5*self.gripper_size[2]),
            self.gripper_size,
            E_rigid, mass_factor
        )

        # Keep handles for post_step
        self.gr1, self.gr2 = gripper1, gripper2

        return [block, platform, gripper1, gripper2]

    def create_scheme(self):
        # Solve all four bodies in one elastic solids SPH solve:
        elastic = ElasticSolidsScheme(
            elastic_solids=['block', 'platform', 'gripper1', 'gripper2'],
            solids=[],
            dim=self.dim,
            artificial_stress_eps=0.5,
            xsph_eps=0.5
        )
        return SchemeChooser(default='elastic', elastic=elastic)

    def configure_scheme(self):
        # A reasonable timestep and output frequency
        self.scheme.configure_solver(dt=1e-4, tf=2.0, pfreq=100)

    def create_equations(self):
        eqns = self.scheme.get_equations()
        # Gravity only on the block:
        eqns.append(Group(equations=[
            BodyForce(dest='block', sources=None, fx=0, fy=0, fz=-9.81)
        ], real=False))
        return eqns

    def post_step(self, solver):
        # Only assign jaw velocities; DO NOT manually move x,y,z here!
        g1, g2 = self.gr1, self.gr2
        dt = solver.dt

        # Compute target eye‐to‐block overlap of 1 cm
        half_b = 0.5*self.block_size[0]
        half_g = 0.5*self.gripper_size[0]
        target = -half_b - half_g + 0.01

        # If not yet at target, close in slowly so block deforms
        if g1.x[0] < target:
            g1.u[:] =  0.02  # 2 cm/s pinch
            g2.u[:] = -0.02
            g1.w[:] = g2.w[:] = 0.0
        else:
            # After pinch, lift at 30 cm/s
            g1.u[:] = g2.u[:] = 0.0
            g1.w[:] = g2.w[:] = 0.3

if __name__ == '__main__':
    app = GraspDeformableBlock()
    app.run()
