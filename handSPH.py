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
        # dimensions & resolution
        self.dim    = 3
        self.dx     = 0.01
        self.hdx    = 1.3
        self.rho0   = 1000.0

        # rubber‐like block material
        self.Eb = 1e7
        self.nu = 0.49
        self.c0 = 50.0

        # geometry (m)
        self.block_size    = (0.3, 0.2, 0.1)
        self.platform_size = (1.0, 0.6, 0.04)   # 4 layers @ dx=0.01
        self.gripper_size  = (0.05, 0.1, 0.2)

    def create_box(self, center, size):
        nx = max(3, int(round(size[0] / self.dx)))
        ny = max(3, int(round(size[1] / self.dx)))
        nz = max(3, int(round(size[2] / self.dx)))
        xs = np.linspace(center[0] - size[0]/2 + self.dx/2,
                         center[0] + size[0]/2 - self.dx/2, nx)
        ys = np.linspace(center[1] - size[1]/2 + self.dx/2,
                         center[1] + size[1]/2 - self.dx/2, ny)
        zs = np.linspace(center[2] - size[2]/2 + self.dx/2,
                         center[2] + size[2]/2 - self.dx/2, nz)
        x, y, z = np.meshgrid(xs, ys, zs, indexing='ij')
        return x.ravel(), y.ravel(), z.ravel()

    def create_particles(self):
        # 1) Deformable block
        bx, by, bz = self.create_box(
            (0, 0, self.platform_size[2] + 0.5*self.block_size[2]),
            self.block_size
        )
        block = get_particle_array_elastic_dynamics(
            name='block',
            x=bx, y=by, z=bz,
            h=self.hdx*self.dx,
            m=self.dx**3 * self.rho0,
            rho=self.rho0,
            constants={
                'E':       self.Eb,
                'nu':      self.nu,
                'rho_ref': self.rho0,
                'c0_ref':  self.c0
            }
        )

        # 2) Platform and grippers as "rigid" via the solids list
        def make_rigid(name, center, size):
            x, y, z = self.create_box(center, size)
            pa = get_particle_array_elastic_dynamics(
                name=name,
                x=x, y=y, z=z,
                h=self.hdx*self.dx,
                m=self.dx**3 * self.rho0 * 1e3,   # heavy
                rho=self.rho0,
                constants={
                    'E':       1e9,    # stiff (1 GPa)
                    'nu':      self.nu,
                    'rho_ref': self.rho0,
                    'c0_ref':  self.c0
                }
            )
            # mark as rigid & boundary
            pa.is_boundary[:] = 1
            pa.is_rigid[:]   = 1
            return pa

        platform = make_rigid(
            'platform',
            (0, 0, self.platform_size[2]/2),
            self.platform_size
        )
        gr1 = make_rigid(
            'gripper1',
            (-0.4, 0,
             self.platform_size[2] + 0.5*self.gripper_size[2]),
            self.gripper_size
        )
        gr2 = make_rigid(
            'gripper2',
            ( 0.4, 0,
             self.platform_size[2] + 0.5*self.gripper_size[2]),
            self.gripper_size
        )
        # keep handles for post_step
        self.gr1, self.gr2 = gr1, gr2

        return [block, platform, gr1, gr2]

    def create_scheme(self):
        # block is elastic, platform & jaws are rigid solids
        scheme = ElasticSolidsScheme(
            elastic_solids=['block'],
            solids=['platform','gripper1','gripper2'],
            dim=self.dim,
            artificial_stress_eps=0.5,
            xsph_eps=0.5
        )
        return SchemeChooser(default='elastic', elastic=scheme)

    def configure_scheme(self):
        # reasonable dt for stability + fewer outputs
        self.scheme.configure_solver(dt=1e-4, tf=2.0, pfreq=100)

    def create_equations(self):
        eqns = self.scheme.get_equations()
        # gravity on block
        eqns.append(Group(equations=[
            BodyForce(dest='block', sources=None,
                      fx=0, fy=0, fz=-9.81)
        ], real=False))
        return eqns

    def post_step(self, solver):
        # only assign jaw velocities; do NOT move x/y/z by hand
        g1, g2 = self.gr1, self.gr2
        dt      = solver.dt

        half_b = 0.5*self.block_size[0]
        half_g = 0.5*self.gripper_size[0]
        # pinch 1 cm, then lift
        target = -half_b - half_g + 0.01

        if g1.x[0] < target:
            # slow pinch (0.02 m/s) so you see deformation
            g1.u[:] =  0.02
            g2.u[:] = -0.02
            g1.w[:] = g2.w[:] = 0.0
        else:
            # lift at 0.3 m/s
            g1.u[:] = g2.u[:] = 0.0
            g1.w[:] = g2.w[:] = 0.3

if __name__ == '__main__':
    app = GraspDeformableBlock()
    app.run()
