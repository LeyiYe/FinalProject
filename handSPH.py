#!/usr/bin/env python3
import numpy as np

from pysph.base.utils           import get_particle_array
from pysph.solver.application   import Application
from pysph.sph.scheme           import SchemeChooser
from pysph.sph.solid_mech.basic import (
    ElasticSolidsScheme,
    get_particle_array_elastic_dynamics
)
from pysph.sph.equation         import Group
from pysph.sph.basic_equations  import BodyForce

class GraspDeformableBlock(Application):
    def initialize(self):
        # Basic sim parameters
        self.dim    = 3
        self.dx     = 0.01
        self.hdx    = 1.3
        self.rho0   = 1000.0

        # Material properties
        self.E_block = 1e6    # 1 MPa for a softer rubber
        self.nu      = 0.49
        self.c0      = 50.0

        # Geometry (m)
        self.block_size    = (0.3, 0.2, 0.1)
        # PLATFORM THICKNESS DOUBLED TO 0.04 m → 4 LAYERS @ dx=0.01
        self.platform_size = (1.0, 0.6, 0.04)
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
                'E'      : self.E_block,
                'nu'     : self.nu,
                'rho_ref': self.rho0,
                'c0_ref' : self.c0
            }
        )

        # 2) Platform & gripper jaws as stiff elastic solids
        def make_rigid(name, center, size, E_rigid, mass_factor):
            x,y,z = self.create_box(center, size)
            return get_particle_array_elastic_dynamics(
                name=name,
                x=x, y=y, z=z,
                h=self.hdx*self.dx,
                m=self.dx**3 * self.rho0 * mass_factor,
                rho=self.rho0,
                constants={
                    'E'      : E_rigid,
                    'nu'     : self.nu,
                    'rho_ref': self.rho0,
                    'c0_ref' : self.c0
                }
            )

        # RIGID SOLID PARAMETERS
        E_rigid    = 1e8    # 100 MPa
        mass_factor= 1e3    # 1,000× heavier per particle

        platform = make_rigid(
            'platform',
            (0, 0, self.platform_size[2]/2),
            self.platform_size,
            E_rigid, mass_factor
        )
        gripper1 = make_rigid(
            'gripper1',
            (-0.4, 0,
             self.platform_size[2] + 0.5*self.gripper_size[2]),
            self.gripper_size,
            E_rigid, mass_factor
        )
        gripper2 = make_rigid(
            'gripper2',
            ( 0.4, 0,
             self.platform_size[2] + 0.5*self.gripper_size[2]),
            self.gripper_size,
            E_rigid, mass_factor
        )

        return [block, platform, gripper1, gripper2]

    def create_scheme(self):
        # All bodies in one elastic‐solid solve
        scheme = ElasticSolidsScheme(
            elastic_solids=['block','platform','gripper1','gripper2'],
            solids=[],
            dim=self.dim,
            artificial_stress_eps=1.0,   # stronger artificial‐stress damping
            xsph_eps=1.0                 # stronger XSPH smoothing
        )
        return SchemeChooser(default='elastic', elastic=scheme)

    def configure_scheme(self):
        # dt small enough for stability, but not tiny
        self.scheme.configure_solver(dt=5e-5, tf=2.0, pfreq=50)

    def create_equations(self):
        eqns = self.scheme.get_equations()
        # gravity only on the deformable block
        eqns.append(Group(equations=[
            BodyForce(dest='block', sources=None,
                      fx=0, fy=0, fz=-9.81)
        ], real=False))
        return eqns

    def post_step(self, solver):
        # move jaws manually (incorporating the integrator step too)
        g1, g2 = self.particles[2], self.particles[3]
        dt     = solver.dt

        half_b = 0.5*self.block_size[0]
        half_g = 0.5*self.gripper_size[0]
        target = -half_b - half_g + 0.005  # 5 mm overlap

        if g1.x[0] < target:
            # closing
            g1.u[:] =  0.2; g2.u[:] = -0.2
            g1.w[:] = g2.w[:] = 0.0
        else:
            # lifting
            g1.u[:] = g2.u[:] = 0.0
            g1.w[:] = g2.w[:] = 0.3

        # nudge them so any minor slip is corrected
        for g in (g1, g2):
            g.x += g.u * dt
            g.y += g.v * dt
            g.z += g.w * dt

if __name__ == '__main__':
    app = GraspDeformableBlock()
    app.run()
