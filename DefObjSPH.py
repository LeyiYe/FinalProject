#!/usr/bin/env python3
import numpy as np

from pysph.base.utils              import get_particle_array
from pysph.solver.application      import Application
from pysph.sph.scheme              import SchemeChooser
from pysph.sph.solid_mech.basic    import (
    ElasticSolidsScheme,
    get_particle_array_elastic_dynamics
)
from pysph.sph.equation            import Group
from pysph.sph.basic_equations     import BodyForce

class GraspDeformableBlock(Application):
    def initialize(self):
        # Simulation / material
        self.dim    = 3
        self.dx     = 0.01
        self.hdx    = 1.3
        self.rho0   = 1000.0

        # Block: rubber‐like
        self.E_block = 1e7     # 10 MPa
        self.nu      = 0.49
        self.c0      = 50.0

        # Geometry
        self.block_size    = (0.3, 0.2, 0.1)
        # make platform thicker for solid contact
        self.platform_size = (1.0, 0.6, 0.04)
        self.gripper_size  = (0.05, 0.1, 0.2)

        # Lift‐phase gravity toggle
        self.gravity_on = False

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
        # --- deformable rubber block ---
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
                'E':      self.E_block,
                'nu':     self.nu,
                'rho_ref':self.rho0,
                'c0_ref': self.c0
            }
        )

        # --- platform & gripper jaws as very stiff solids ---
        def make_solid(name, center, size, E, mass_factor):
            x,y,z = self.create_box(center, size)
            return get_particle_array_elastic_dynamics(
                name=name,
                x=x, y=y, z=z,
                h=self.hdx*self.dx,
                m=self.dx**3 * self.rho0 * mass_factor,
                rho=self.rho0,
                constants={
                    'E':      E,
                    'nu':     self.nu,
                    'rho_ref':self.rho0,
                    'c0_ref': self.c0
                }
            )

        # stiff + heavy
        E_rigid    = 1e8    # 100 MPa
        mass_factor= 1e3    # 1000× heavier

        platform = make_solid(
            'platform',
            (0, 0, self.platform_size[2]/2),
            self.platform_size,
            E_rigid, mass_factor
        )
        g1 = make_solid(
            'gripper1',
            (-0.4, 0, self.platform_size[2] + 0.5*self.gripper_size[2]),
            self.gripper_size,
            E_rigid, mass_factor
        )
        g2 = make_solid(
            'gripper2',
            ( 0.4, 0, self.platform_size[2] + 0.5*self.gripper_size[2]),
            self.gripper_size,
            E_rigid, mass_factor
        )
        self.gr1, self.gr2 = g1, g2

        return [block, platform, g1, g2]

    def create_scheme(self):
        elastic = ElasticSolidsScheme(
            elastic_solids=['block','platform','gripper1','gripper2'],
            solids=[],
            dim=self.dim,
            artificial_stress_eps=0.5,
            xsph_eps=0.5
        )
        return SchemeChooser(default='elastic', elastic=elastic)

    def configure_scheme(self):
        # bigger dt for speed
        self.scheme.configure_solver(dt=1e-4, tf=2.0, pfreq=100)

    def create_equations(self):
        eqns = self.scheme.get_equations()
        # gravity is applied *manually* in post_step after lift
        return eqns

    def post_step(self, solver):
        dt = solver.dt
        block = self.particles[0]

        # 1) Move the jaws *slowly* so the block deforms under pinch
        target = -0.5*self.block_size[0] - 0.5*self.gripper_size[0] + 0.01
        g1, g2 = self.gr1, self.gr2

        if g1.x[0] < target:
            # slow approach: 0.02 m/s
            g1.u[:] =  0.02
            g2.u[:] = -0.02
            g1.w[:] = g2.w[:] = 0.0
        else:
            # after pinch, lift at 0.3 m/s
            g1.u[:] = g2.u[:] = 0.0
            g1.w[:] = g2.w[:] = 0.3
            # once block cleared table, enable gravity
            com_z = np.mean(block.z)
            lift_z = self.platform_size[2] + 0.5*self.block_size[2] + 0.01
            if not self.gravity_on and com_z > lift_z:
                self.gravity_on = True

        # nudge jaws so contact is maintained
        for g in (g1, g2):
            g.x += g.u * dt
            g.y += g.v * dt
            g.z += g.w * dt

        # 2) If gravity_on, deform under own weight
        if self.gravity_on:
            # manually apply gravity acceleration to block
            block.w[:] += -9.81 * dt

if __name__ == '__main__':
    app = GraspDeformableBlock()
    app.run()
