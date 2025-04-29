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
from pysph.sph.rigid_body        import RigidBodyWallCollision

class GraspDeformableBlock(Application):
    def initialize(self):
        # --- sim & material params ---
        self.dim     = 3
        self.dx      = 0.01
        self.hdx     = 1.3
        self.rho0    = 1000.0
        self.E_block = 1e7     # rubber‐like block
        self.nu      = 0.49
        self.c0      = 50.0

        # geometry (m)
        self.block_size    = (0.3, 0.2, 0.1)
        self.platform_size = (1.0, 0.6, 0.02)
        self.gripper_size  = (0.05, 0.1, 0.2)

    def create_box(self, center, size):
        """Return (x,y,z) in a regular grid inside the box."""
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
        # 1) Deformable block
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
        # DEM‐contact extras for RigidBodyWallCollision:
        block.add_property('rad_s', default=self.dx*0.5)
        for pname in (
            'tang_disp_x','tang_disp_y','tang_disp_z',
            'tang_velocity_x','tang_velocity_y','tang_velocity_z',
            'fx','fy','fz','total_mass'
        ):
            block.add_property(pname, type='double', default=0.0)
        block.total_mass[:] = np.sum(block.m)

        # ----------------------------------------------------------------------------
        # 2) Rigid “walls”: platform and two gripper jaws
        #    these never move (infinite mass) but need a plane normal (nx,ny,nz)
        # ----------------------------------------------------------------------------
        def make_wall(name, center, size, normal):
            x,y,z = self.create_box(center, size)
            pa = get_particle_array(
                name=name,
                x=x, y=y, z=z,
                h=self.hdx*self.dx,
                m=1e12,               # effectively infinite mass
                rho=self.rho0,
                is_boundary=1,
                is_rigid=1
            )
            # plane normal on each wall‐particle
            pa.add_property('nx', type='double', default=normal[0])
            pa.add_property('ny', type='double', default=normal[1])
            pa.add_property('nz', type='double', default=normal[2])
            return pa

        platform = make_wall(
            'platform',
            (0, 0, self.platform_size[2]/2),
            self.platform_size,
            normal=(0,0,1)    # upward‐facing floor
        )
        gripper1 = make_wall(
            'gripper1',
            (-0.4, 0,
             self.platform_size[2]+0.5*self.gripper_size[2]),
            self.gripper_size,
            normal=(1,0,0)    # left jaw faces +x
        )
        gripper2 = make_wall(
            'gripper2',
            ( 0.4, 0,
             self.platform_size[2]+0.5*self.gripper_size[2]),
            self.gripper_size,
            normal=(-1,0,0)   # right jaw faces –x
        )

        return [block, platform, gripper1, gripper2]

    def create_scheme(self):
        # Only the block participates in ElasticSolidsScheme:
        elastic = ElasticSolidsScheme(
            elastic_solids=['block'],
            solids=[],
            dim=self.dim,
            artificial_stress_eps=0.5,
            xsph_eps=0.5
        )
        return SchemeChooser(default='elastic', elastic=elastic)

    def configure_scheme(self):
        # Slightly smaller dt for DEM‐contact stability
        self.scheme.configure_solver(dt=2e-5, tf=2.0, pfreq=50)

    def create_equations(self):
        eqns = self.scheme.get_equations()
        # gravity on the block only
        eqns.append(Group(equations=[
            BodyForce(dest='block', sources=None,
                      fx=0, fy=0, fz=-9.81)
        ], real=False))
        # DEM‐style wall collisions for block vs each rigid wall
        eqns.append(Group(equations=[
            RigidBodyWallCollision('block', ['platform'], kn=1e4, mu=0.2, en=0.8),
            RigidBodyWallCollision('block', ['gripper1'], kn=1e4, mu=0.2, en=0.8),
            RigidBodyWallCollision('block', ['gripper2'], kn=1e4, mu=0.2, en=0.8),
        ], real=False))
        return eqns

    def post_step(self, solver):
        # open/close logic — only *set* velocities, never manually move x/y/z.
        g1, g2 = self.particles[2], self.particles[3]
        half_b = 0.5*self.block_size[0]
        half_g = 0.5*self.gripper_size[0]
        target = -half_b - half_g + 0.02

        if g1.x[0] < target:
            # closing in
            g1.u[:] =  0.2
            g2.u[:] = -0.2
            g1.w[:] = g2.w[:] = 0.0
        else:
            # lifting up
            g1.u[:] = g2.u[:] = 0.0
            g1.w[:] = g2.w[:] = 0.3

if __name__ == '__main__':
    app = GraspDeformableBlock()
    app.run()
