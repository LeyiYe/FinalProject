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
        self.dim  = 3
        self.dx   = 0.01
        self.hdx  = 1.3
        self.rho0 = 1000.0

        # rubber block
        self.Eb = 1e7
        self.nu = 0.49
        self.c0 = 50.0

        # geometry
        self.block_size    = (0.3, 0.2, 0.1)
        self.platform_size = (1.0, 0.6, 0.04)  # 4 layers at dx=0.01
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
        # 1) rubber block
        bx,by,bz = self.create_box(
            (0,0,self.platform_size[2] + 0.5*self.block_size[2]),
            self.block_size
        )
        block = get_particle_array_elastic_dynamics(
            name='block', x=bx, y=by, z=bz,
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

        # 2) platform and grippers AS elastic solids (very stiff & heavy)
        def make_elastic_rigid(name, center, size, E, massf):
            x,y,z = self.create_box(center, size)
            return get_particle_array_elastic_dynamics(
                name=name,
                x=x, y=y, z=z,
                h=self.hdx*self.dx,
                m=self.dx**3 * self.rho0 * massf,
                rho=self.rho0,
                constants={
                    'E':       E,
                    'nu':      self.nu,
                    'rho_ref': self.rho0,
                    'c0_ref':  self.c0
                }
            )

        E_rigid    = 1e9   # 1 GPa
        mass_factor= 1e3   # 1000× heavier

        platform = make_elastic_rigid(
            'platform',
            (0, 0, self.platform_size[2]/2),
            self.platform_size,
            E_rigid, mass_factor
        )
        gr1 = make_elastic_rigid(
            'gripper1',
            (-0.4, 0,
             self.platform_size[2]+0.5*self.gripper_size[2]),
            self.gripper_size,
            E_rigid, mass_factor
        )
        gr2 = make_elastic_rigid(
            'gripper2',
            ( 0.4, 0,
             self.platform_size[2]+0.5*self.gripper_size[2]),
            self.gripper_size,
            E_rigid, mass_factor
        )

        # keep handles for jaw motion
        self.gr1, self.gr2 = gr1, gr2

        return [block, platform, gr1, gr2]

    def create_scheme(self):
        # Solve all four bodies together
        scheme = ElasticSolidsScheme(
            elastic_solids=['block','platform','gripper1','gripper2'],
            solids=[],
            dim=self.dim,
            artificial_stress_eps=0.5,
            xsph_eps=0.5
        )
        return SchemeChooser(default='elastic', elastic=scheme)

    def configure_scheme(self):
        self.scheme.configure_solver(dt=1e-4, tf=2.0, pfreq=100)

    def create_equations(self):
        eqns = self.scheme.get_equations()
        # gravity on the block
        eqns.append(Group(equations=[
            BodyForce(dest='block', sources=None,
                      fx=0, fy=0, fz=-9.81)
        ], real=False))
        return eqns

    def post_step(self, solver):
        # only set velocities on the jaws; integrator moves them
        g1, g2 = self.gr1, self.gr2
        dt = solver.dt

        half_b = 0.5*self.block_size[0]
        half_g = 0.5*self.gripper_size[0]
        target = -half_b - half_g + 0.01  # 1 cm overlap

        if g1.x[0] < target:
            # slow pinch so block deforms
            g1.u[:] =  0.02
            g2.u[:] = -0.02
            g1.w[:] = g2.w[:] = 0.0
        else:
            # lift
            g1.u[:] = g2.u[:] = 0.0
            g1.w[:] = g2.w[:] = 0.3

if __name__ == '__main__':
    app = GraspDeformableBlock()
    app.run()
