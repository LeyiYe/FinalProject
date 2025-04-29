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
from pysph.sph.rigid_body       import RigidBodyWallCollision

class GraspDeformableBlock(Application):
    def initialize(self):
        # core sim params
        self.dim     = 3
        self.dx      = 0.01
        self.hdx     = 1.3
        self.rho0    = 1000.0
        self.E_block = 1e7       # rubber‐like block
        self.nu      = 0.49
        self.c0      = 50.0

        # geometry (m)
        self.block_size    = (0.3, 0.2, 0.1)
        self.platform_size = (1.0, 0.6, 0.02)
        self.gripper_size  = (0.05, 0.1, 0.2)

    def create_box(self, center, size):
        """Uniform grid inside a box."""
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
        # ----------------------------------------------------------------------------
        # 1) deformable block
        # ----------------------------------------------------------------------------
        bx, by, bz = self.create_box(
            center=(0, 0,
                    self.platform_size[2] + 0.5*self.block_size[2]),
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
        # DEM–contact extras
        block.add_property('rad_s', default=self.dx*0.5)
        for n in (
            'tang_disp_x','tang_disp_y','tang_disp_z',
            'tang_velocity_x','tang_velocity_y','tang_velocity_z',
            'fx','fy','fz','total_mass'
        ):
            block.add_property(n, type='double', default=0.0)
        block.total_mass[:] = np.sum(block.m)

        # ----------------------------------------------------------------------------
        # 2) rigid walls: platform + two gripper jaws (infinite mass)
        # ----------------------------------------------------------------------------
        def make_wall(name, center, size, normal):
            x,y,z = self.create_box(center, size)
            pa = get_particle_array(
                name=name,
                x=x, y=y, z=z,
                h=self.hdx*self.dx,
                m=1e12,             # effectively infinite mass
                rho=self.rho0,
                is_boundary=1,
                is_rigid=1
            )
            pa.add_property('nx', type='double', default=normal[0])
            pa.add_property('ny', type='double', default=normal[1])
            pa.add_property('nz', type='double', default=normal[2])
            return pa

        platform = make_wall(
            'platform',
            (0, 0, self.platform_size[2]/2),
            self.platform_size,
            normal=(0,0,1)
        )
        gr1 = make_wall(
            'gripper1',
            (-0.4, 0,
             self.platform_size[2] + 0.5*self.gripper_size[2]),
            self.gripper_size,
            normal=( 1,0,0)
        )
        gr2 = make_wall(
            'gripper2',
            ( 0.4, 0,
             self.platform_size[2] + 0.5*self.gripper_size[2]),
            self.gripper_size,
            normal=(-1,0,0)
        )
        self.gr1, self.gr2 = gr1, gr2

        return [block, platform, gr1, gr2]

    def create_scheme(self):
        # only the block is solved elastically
        elastic = ElasticSolidsScheme(
            elastic_solids=['block'],
            solids=[],
            dim=self.dim,
            artificial_stress_eps=0.5,
            xsph_eps=0.5
        )
        return SchemeChooser(default='elastic', elastic=elastic)

    def configure_scheme(self):
        # larger dt + moderate pfreq
        self.scheme.configure_solver(dt=2e-4, tf=2.0, pfreq=50)

    def create_equations(self):
        eqns = self.scheme.get_equations()

        # gravity on block
        eqns.append(Group(equations=[
            BodyForce(dest='block', sources=None,
                      fx=0, fy=0, fz=-9.81)
        ], real=False))

        # DEM wall collisions (soft springs kn=1e3)
        eqns.append(Group(equations=[
            RigidBodyWallCollision('block', ['platform'],   kn=1e3, mu=0.2, en=0.8),
            RigidBodyWallCollision('block', ['gripper1'],  kn=1e3, mu=0.2, en=0.8),
            RigidBodyWallCollision('block', ['gripper2'],  kn=1e3, mu=0.2, en=0.8),
        ], real=False))

        return eqns

    def post_step(self, solver):
        # --------------------------------------------------------------------
        # a) hard‐clamp any block particles below the table
        # --------------------------------------------------------------------
        block = self.particles[0]
        floor_z = self.platform_size[2]
        below = block.z < floor_z
        if np.any(below):
            block.z[below] = floor_z
            # zero any downward velocity
            block.w[below] = np.maximum(block.w[below], 0.0)

        # --------------------------------------------------------------------
        # b) manually move the jaws: overshoot 5 mm, then lift
        # --------------------------------------------------------------------
        g1, g2 = self.gr1, self.gr2
        dt     = solver.dt

        half_b = 0.5*self.block_size[0]
        half_g = 0.5*self.gripper_size[0]
        target = -half_b - half_g + 0.005

        if g1.x[0] < target:
            g1.u[:] =  0.2
            g2.u[:] = -0.2
            g1.w[:] = g2.w[:] = 0.0
        else:
            g1.u[:] = g2.u[:] = 0.0
            g1.w[:] = g2.w[:] = 0.3

        for g in (g1, g2):
            g.x += g.u * dt
            g.y += g.v * dt
            g.z += g.w * dt

if __name__ == '__main__':
    app = GraspDeformableBlock()
    app.run()
