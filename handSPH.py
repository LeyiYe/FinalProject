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
        # core sim parameters
        self.dim    = 3
        self.dx     = 0.01
        self.hdx    = 1.3
        self.rho0   = 1000.0
        self.E      = 1e7      # block stiffness
        self.nu     = 0.49
        self.c0     = 50.0

        # geometry (m)
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
        # --- deformable block ---
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
                'E'      : self.E,
                'nu'     : self.nu,
                'rho_ref': self.rho0,
                'c0_ref' : self.c0
            }
        )

        # helper to build a “rigid” platform/gripper as a super-stiff solid
        def make_rigid(name, center, size):
            x,y,z = self.create_block(center, size)
            # boost stiffness & mass so it barely moves
            stiff_E = 1e9
            mass_factor = 1e4
            return get_particle_array_elastic_dynamics(
                name=name,
                x=x, y=y, z=z,
                h=self.hdx*self.dx,
                m=self.dx**3 * self.rho0 * mass_factor,
                rho=self.rho0,
                constants={
                    'E'      : stiff_E,
                    'nu'     : self.nu,
                    'rho_ref': self.rho0,
                    'c0_ref' : self.c0
                }
            )

        platform = make_rigid(
            'platform',
            (0, 0, self.platform_size[2]/2),
            self.platform_size
        )
        gripper1 = make_rigid(
            'gripper1',
            (-0.4, 0, self.platform_size[2] + self.gripper_size[2]/2),
            self.gripper_size
        )
        gripper2 = make_rigid(
            'gripper2',
            ( 0.4, 0, self.platform_size[2] + self.gripper_size[2]/2),
            self.gripper_size
        )

        return [block, platform, gripper1, gripper2]

    def create_scheme(self):
        elastic = ElasticSolidsScheme(
            elastic_solids=['block', 'platform', 'gripper1', 'gripper2'],
            solids=[],
            dim=self.dim,
            artificial_stress_eps=0.5,
            xsph_eps=0.5
        )
        return SchemeChooser(default='elastic', elastic=elastic)

    def configure_scheme(self):
        self.scheme.configure_solver(dt=5e-5, tf=2.0, pfreq=50)

    def create_equations(self):
        eqns = self.scheme.get_equations()
        # only apply gravity to the deformable block
        eqns.append(Group(equations=[
            BodyForce(dest='block', sources=None,
                      fx=0, fy=0, fz=-9.81)
        ], real=False))
        return eqns

    def post_step(self, solver):
        # your gripper open/close logic
        g1, g2 = self.particles[2], self.particles[3]
        dt = solver.dt
        half_block = 0.5*self.block_size[0]
        half_grip  = 0.5*self.gripper_size[0]
        target = -half_block - half_grip + 0.02

        if g1.x[0] < target:
            # closing
            g1.u[:] =  0.2; g2.u[:] = -0.2
            g1.w[:] = g2.w[:] = 0.0
        else:
            # lifting
            g1.u[:] = g2.u[:] = 0.0
            g1.w[:] = g2.w[:] = 0.3

        for gr in (g1, g2):
            gr.x  += gr.u * dt
            gr.y  += gr.v * dt
            gr.z  += gr.w * dt
            gr.z0 += gr.w * dt

if __name__ == '__main__':
    app = GraspDeformableBlock()
    app.run()
