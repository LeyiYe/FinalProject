#!/usr/bin/env python3
import numpy as np

from pysph.base.utils           import get_particle_array
from pysph.solver.application   import Application
from pysph.sph.scheme           import SchemeChooser
from pysph.sph.solid_mech.basic import get_particle_array_elastic_dynamics
from pysph.sph.equation         import Equation, Group
from pysph.sph.basic_equations import BodyForce
from pysph.sph.rigid_body       import RigidBodyWallCollision
# import BodyForce  <-- removed
# no custom gravity needed; use built-in BodyForce       import RigidBodyWallCollision

class FloorRepulsion(Equation):
    def __init__(self, dest, floor_z, k_pen, c_pen):
        super().__init__(dest, [])
        self.floor_z = floor_z
        self.k_pen   = k_pen
        self.c_pen   = c_pen

    def loop(self, d_idx, d_z, d_w, d_aw):
        # Spring + dashpot directly into vertical acceleration
        pen = self.floor_z - d_z[d_idx]
        if pen > 0.0:
            d_aw[d_idx] += self.k_pen * pen - self.c_pen * d_w[d_idx]

class ForceToAcceleration(Equation):
    def __init__(self, dest):
        super().__init__(dest, [])

    def loop_all(self, d_idx, d_fx, d_fy, d_fz, d_m, d_au, d_av, d_aw):
        # Convert contact forces into accelerations via F=ma
        d_au[d_idx] += d_fx[d_idx] / d_m[d_idx]
        d_av[d_idx] += d_fy[d_idx] / d_m[d_idx]
        d_aw[d_idx] += d_fz[d_idx] / d_m[d_idx]

class GraspDeformableBlock(Application):
    def initialize(self):
        # Simulation parameters
        self.dim     = 3
        self.dx      = 0.01
        self.hdx     = 1.3
        self.rho0    = 1000.0

        # Material properties (rubber‐like)
        self.E_block = 1e7
        self.nu      = 0.49
        self.c0      = 50.0

        # Geometry
        self.block_size    = (0.3, 0.2, 0.1)
        self.platform_size = (1.0, 0.6, 0.04)
        self.gripper_size  = (0.05, 0.1, 0.2)

        # Enable gravity to settle block on platform
        self.gravity_enabled = True

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
        block.add_property('rad_s', default=self.dx * 0.5)
        for prop in ('tang_disp_x','tang_disp_y','tang_disp_z',
                     'tang_velocity_x','tang_velocity_y','tang_velocity_z',
                     'fx','fy','fz','total_mass'):
            block.add_property(prop, type='double', default=0.0)
        block.total_mass[:] = np.sum(block.m)

        # 2) Rigid bodies: platform + two grippers
        def make_wall(name, center, size, normal):
            x,y,z = self.create_box(center, size)
            pa = get_particle_array(
                name=name,
                x=x, y=y, z=z,
                h=self.hdx*self.dx,
                m=1e12,
                rho=self.rho0,
                is_boundary=1,
                is_rigid=1
            )
            pa.add_property('nx', type='double', default=normal[0])
            pa.add_property('ny', type='double', default=normal[1])
            pa.add_property('nz', type='double', default=normal[2])
            return pa

        plat = make_wall('platform',
                         (0, 0, self.platform_size[2]/2),
                         self.platform_size, normal=(0,0,1))
        g1 = make_wall('gripper1',
                       (-0.4, 0, self.platform_size[2] + 0.5*self.gripper_size[2]),
                       self.gripper_size, normal=(1,0,0))
        g2 = make_wall('gripper2',
                       (0.4, 0, self.platform_size[2] + 0.5*self.gripper_size[2]),
                       self.gripper_size, normal=(-1,0,0))
        self.gr1, self.gr2 = g1, g2

        return [block, plat, g1, g2]

    def create_scheme(self):
        from pysph.sph.solid_mech.basic import ElasticSolidsScheme
        # include both deformable block and rigid bodies for neighbor search
        scheme = ElasticSolidsScheme(
            elastic_solids=['block'],
            solids=['platform','gripper1','gripper2'],
            dim=self.dim,
            artificial_stress_eps=0.5,
            xsph_eps=0.5
        )
        return SchemeChooser(default='elastic', elastic=scheme)

    def configure_scheme(self):
        self.scheme.configure_solver(dt=1e-4, tf=2.0, pfreq=200)

    def create_equations(self):
        eqns = self.scheme.get_equations()

        # optional gravity to settle block first
        if self.gravity_enabled:
            eqns.append(Group(
                equations=[BodyForce(dest='block', sources=[],
                                     fx=0.0, fy=0.0, fz=-9.81)],
                real=False
            ))

        # floor spring → vertical accel
        eqns.append(Group(
            equations=[FloorRepulsion(dest='block', floor_z=self.platform_size[2],
                                       k_pen=5e5, c_pen=50.0)],
            real=False
        ))

        # DEM collisions + refresh neighbor lists
        eqns.append(Group(
            equations=[
                RigidBodyWallCollision('block', ['platform'], kn=1e3, mu=0.2, en=0.8),
                RigidBodyWallCollision('block', ['gripper1'], kn=1e3, mu=0.2, en=0.8),
                RigidBodyWallCollision('block', ['gripper2'], kn=1e3, mu=0.2, en=0.8),
            ],
            real=False,
            update_nnps=True
        ))

        # convert contact forces → accelerations
        eqns.append(Group(
            equations=[ForceToAcceleration(dest='block')],
            real=False
        ))

        return eqns

    def post_step(self, solver):
        dt = solver.dt
        block = self.particles[0]

        # clamp block under floor
        floor_z = self.platform_size[2]
        below = block.z < floor_z
        if np.any(below):
            block.z[below] = floor_z
            block.w[below] = np.maximum(block.w[below], 0.0)

        # move the jaws
        half_b = 0.5 * self.block_size[0]
        half_g = 0.5 * self.gripper_size[0]
        target = -half_b - half_g + 0.01
        g1, g2 = self.gr1, self.gr2
        if g1.x[0] < target:
            g1.u[:] =  0.2; g2.u[:] = -0.2
            g1.w[:] = g2.w[:] = 0.0
        else:
            g1.u[:] = g2.u[:] = 0.0
            g1.w[:] = g2.w[:] = 0.3

        # manual position update for contact
        for g in (g1, g2):
            g.x += g.u * dt
            g.y += g.v * dt
            g.z += g.w * dt

if __name__ == '__main__':
    app = GraspDeformableBlock()
    app.run()
