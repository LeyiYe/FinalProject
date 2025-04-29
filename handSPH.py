#!/usr/bin/env python3
import numpy as np

from pysph.base.utils             import get_particle_array
from pysph.solver.application     import Application
from pysph.sph.scheme             import SchemeChooser
from pysph.sph.solid_mech.basic   import (
    ElasticSolidsScheme,
    get_particle_array_elastic_dynamics
)
from pysph.sph.equation           import Group
from pysph.sph.basic_equations    import BodyForce

class GraspDeformableBlock(Application):
    def initialize(self):
        # simulation & material params
        self.dim     = 3
        self.dx      = 0.01
        self.hdx     = 1.3
        self.rho0    = 1000.0
        self.E_block = 1e7      # rubber block
        self.nu      = 0.49
        self.c0      = 50.0

        # geometry dims (m)
        self.block_size    = (0.3, 0.2, 0.1)
        self.platform_size = (1.0, 0.6, 0.02)
        self.gripper_size  = (0.05, 0.1, 0.2)

    def create_block(self, center, size):
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
        # 1) deformable block
        bx, by, bz = self.create_block(
            center=(0, 0, self.platform_size[2] + self.block_size[2]/2),
            size=self.block_size
        )
        block = get_particle_array_elastic_dynamics(
            name='block',
            x=bx, y=by, z=bz,
            h=self.hdx*self.dx,
            m=self.dx**3 * self.rho0,
            rho=self.rho0,
            constants={
                'E'      : self.E_block,
                'nu'     : self.nu,
                'rho_ref': self.rho0,
                'c0_ref' : self.c0
            }
        )

        # 2) platform & jaws as “stiff but stable” solids
        def make_solid(name, center, size, E_solid, mass_factor):
            x,y,z = self.create_block(center, size)
            return get_particle_array_elastic_dynamics(
                name=name,
                x=x, y=y, z=z,
                h=self.hdx*self.dx,
                m=self.dx**3 * self.rho0 * mass_factor,
                rho=self.rho0,
                constants={
                    'E'      : E_solid,
                    'nu'     : self.nu,
                    'rho_ref': self.rho0,
                    'c0_ref' : self.c0
                }
            )

        # choose moderate stiffness & mass so dt=5e-5 remains stable
        E_rigid    = 1e8   # 10× block stiffness
        mass_factor= 1e3   # 1000× per-particle mass

        platform = make_solid(
            'platform',
            (0, 0, self.platform_size[2]/2),
            self.platform_size,
            E_rigid, mass_factor
        )
        gripper1 = make_solid(
            'gripper1',
            (-0.4, 0, self.platform_size[2] + self.gripper_size[2]/2),
            self.gripper_size,
            E_rigid, mass_factor
        )
        gripper2 = make_solid(
            'gripper2',
            ( 0.4, 0, self.platform_size[2] + self.gripper_size[2]/2),
            self.gripper_size,
            E_rigid, mass_factor
        )

        return [block, platform, gripper1, gripper2]

    def create_scheme(self):
        # include ALL four bodies so they’re integrated
        elastic = ElasticSolidsScheme(
            elastic_solids=['block','platform','gripper1','gripper2'],
            solids=[],
            dim=self.dim,
            artificial_stress_eps=0.5,
            xsph_eps=0.5
        )
        return SchemeChooser(default='elastic', elastic=elastic)

    def configure_scheme(self):
        # keep your original timestep
        self.scheme.configure_solver(dt=5e-5, tf=2.0, pfreq=50)

    def create_equations(self):
        eqns = self.scheme.get_equations()
        # gravity only on the block
        eqns.append(Group(equations=[
            BodyForce(dest='block', sources=None, fx=0, fy=0, fz=-9.81)
        ], real=False))
        return eqns

    def post_step(self, solver):
        # jaw open/close stays exactly as before
        g1, g2 = self.particles[2], self.particles[3]
        dt = solver.dt
        half_b = 0.5*self.block_size[0]
        half_g = 0.5*self.gripper_size[0]
        target = -half_b - half_g + 0.02

        if g1.x[0] < target:
            g1.u[:] =  0.2; g2.u[:] = -0.2
            g1.w[:] = g2.w[:] = 0.0
        else:
            g1.u[:] = g2.u[:] = 0.0
            g1.w[:] = g2.w[:] = 0.3

        # these velocities now get integrated
        for gr in (g1, g2):
            gr.x  += gr.u * dt
            gr.y  += gr.v * dt
            gr.z  += gr.w * dt
            gr.z0 += gr.w * dt

if __name__ == '__main__':
    app = GraspDeformableBlock()
    app.run()
