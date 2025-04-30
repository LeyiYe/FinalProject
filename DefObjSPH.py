#!/usr/bin/env python3
import numpy as np

from pysph.base.utils           import get_particle_array
from pysph.solver.application   import Application
from pysph.sph.scheme           import SchemeChooser
from pysph.sph.solid_mech.basic import get_particle_array_elastic_dynamics
from pysph.sph.equation         import Equation, Group
from pysph.sph.basic_equations  import BodyForce
from pysph.sph.rigid_body       import RigidBodyWallCollision

class FloorRepulsion(Equation):
    def __init__(self, dest, floor_z, k_pen, c_pen):
        super().__init__(dest, [])
        self.floor_z = floor_z
        self.k_pen   = k_pen
        self.c_pen   = c_pen

    def loop(self, d_idx, d_z, d_w, d_aw):
        pen = self.floor_z - d_z[d_idx]
        if pen > 0.0:
            d_aw[d_idx] += self.k_pen * pen - self.c_pen * d_w[d_idx]

class ForceToAcceleration(Equation):
    def __init__(self, dest):
        super().__init__(dest, [])

    def loop_all(self, d_idx, d_fx, d_fy, d_fz, d_m, d_au, d_av, d_aw):
        # apply Newton's 2nd law on each body
        d_au[d_idx] += d_fx[d_idx] / d_m[d_idx]
        d_av[d_idx] += d_fy[d_idx] / d_m[d_idx]
        d_aw[d_idx] += d_fz[d_idx] / d_m[d_idx]

class GraspDeformableBlock(Application):
    def initialize(self):
        self.dim     = 3
        self.dx      = 0.01
        self.hdx     = 1.3
        self.rho0    = 1000.0

        self.E_block = 1e7
        self.nu      = 0.49
        self.c0      = 50.0

        self.block_size    = (0.3, 0.2, 0.1)
        self.platform_size = (1.0, 0.6, 0.04)
        self.gripper_size  = (0.05, 0.1, 0.2)

        self.gravity_enabled = True

    def create_box(self, center, size):
        nx = max(3, int(round(size[0]/self.dx)))
        ny = max(3, int(round(size[1]/self.dx)))
        nz = max(3, int(round(size[2]/self.dx)))
        xs = np.linspace(center[0]-size[0]/2+self.dx/2,
                         center[0]+size[0]/2-self.dx/2, nx)
        ys = np.linspace(center[1]-size[1]/2+self.dx/2,
                         center[1]+size[1]/2-self.dx/2, ny)
        zs = np.linspace(center[2]-size[2]/2+self.dx/2,
                         center[2]+size[2]/2-self.dx/2, nz)
        x,y,z = np.meshgrid(xs, ys, zs, indexing='ij')
        return x.ravel(), y.ravel(), z.ravel()

    def create_particles(self):
        bx,by,bz = self.create_box(
            center=(0,0,self.platform_size[2]+0.5*self.block_size[2]),
            size=self.block_size)
        block = get_particle_array_elastic_dynamics(
            name='block', x=bx, y=by, z=bz,
            h=self.hdx*self.dx, m=self.dx**3*self.rho0,
            rho=self.rho0,
            constants={'E':self.E_block,'nu':self.nu,
                       'rho_ref':self.rho0,'c0_ref':self.c0}
        )
        # add collision force arrays
        block.add_property('rad_s', default=self.dx*0.5)
        for p in ['tang_disp_x','tang_disp_y','tang_disp_z',
                  'tang_velocity_x','tang_velocity_y','tang_velocity_z',
                  'fx','fy','fz','total_mass']:
            block.add_property(p, type='double', default=0.0)
        block.total_mass[:] = np.sum(block.m)

        def make_wall(name, center, size, normal):
            x,y,z = self.create_box(center, size)
            pa = get_particle_array(name=name, x=x,y=y,z=z,
                                     h=self.hdx*self.dx,
                                     m=1e12, rho=self.rho0,
                                     is_boundary=1, is_rigid=1)
            pa.add_property('nx', type='double', default=normal[0])
            pa.add_property('ny', type='double', default=normal[1])
            pa.add_property('nz', type='double', default=normal[2])
            # allocate force arrays for reaction
            for f in ['fx','fy','fz']:
                pa.add_property(f, type='double', default=0.0)
            pa.add_property('total_mass', type='double', default=np.sum(pa.m))
            return pa

        plat = make_wall('platform', (0,0,self.platform_size[2]/2),
                         self.platform_size, normal=(0,0,1))
        g1 = make_wall('gripper1',
                       (-0.4,0,self.platform_size[2]+0.5*self.gripper_size[2]),
                       self.gripper_size, normal=(1,0,0))
        g2 = make_wall('gripper2',
                       ( 0.4,0,self.platform_size[2]+0.5*self.gripper_size[2]),
                       self.gripper_size, normal=(-1,0,0))
        self.gr1,self.gr2 = g1,g2
        return [block, plat, g1, g2]

    def create_scheme(self):
        from pysph.sph.solid_mech.basic import ElasticSolidsScheme
        scheme = ElasticSolidsScheme(
            elastic_solids=['block'], solids=[], dim=self.dim,
            artificial_stress_eps=0.5, xsph_eps=0.5,
            visco_alpha=0.05, visco_beta=0.0)
        return SchemeChooser(default='elastic', elastic=scheme)

    def configure_scheme(self):
        self.scheme.configure_solver(dt=1e-4, tf=2.0, pfreq=200)

    def create_equations(self):
        eqns = self.scheme.get_equations()
        if self.gravity_enabled:
            eqns.append(Group(equations=[BodyForce(dest='block', sources=[],
                                                    fx=0, fy=0, fz=-9.81)],
                               real=False))
        # DEM collisions
        eqns.append(Group(equations=[
            RigidBodyWallCollision('block',['platform'],kn=1e5,mu=0.3,en=0.2),
            RigidBodyWallCollision('block',['gripper1'],kn=1e5,mu=0.3,en=0.2),
            RigidBodyWallCollision('block',['gripper2'],kn=1e5,mu=0.3,en=0.2),
        ], real=False, update_nnps=True))
                # convert forces into accelerations for the block only
        eqns.append(
            Group(
                equations=[ForceToAcceleration(dest='block')],
                real=False
            )
        )
        return eqns

    def post_step(self, solver):
        dt = solver.dt
        # move jaws and pin their velocities to zero
        half_b, half_g = 0.5*self.block_size[0], 0.5*self.gripper_size[0]
        target = -half_b - half_g + 0.01
        g1, g2 = self.gr1, self.gr2
        if g1.x[0] < target:
            g1.u[:]=0.2; g2.u[:]=-0.2; g1.w[:]=g2.w[:]=0.0
        else:
            g1.u[:]=g2.u[:]=0.0; g1.w[:]=0.3
        for g in (g1,g2):
            g.x += g.u*dt; g.y += g.v*dt; g.z += g.w*dt
            g.u[:] = g.v[:] = g.w[:] = 0.0

if __name__ == '__main__':
    app = GraspDeformableBlock()
    app.run()
