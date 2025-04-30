#!/usr/bin/env python3
import numpy as np

from pysph.base.utils            import get_particle_array
from pysph.solver.application    import Application
from pysph.sph.scheme            import SchemeChooser
from pysph.sph.solid_mech.basic  import (
    ElasticSolidsScheme,
    get_particle_array_elastic_dynamics
)
from pysph.sph.equation          import Group, Equation
from pysph.sph.basic_equations   import BodyForce

# A quick spring–dashpot to stop the block falling through the table.
class FloorRepulsion(Equation):
    def __init__(self, dest, floor_z, k, c):
        super().__init__(dest, [])
        self.floor_z = floor_z
        self.k = k
        self.c = c

    def loop(self, d_idx, d_z, d_w, d_fz):
        pen = self.floor_z - d_z[d_idx]
        if pen > 0:
            d_fz[d_idx] += self.k * pen - self.c * d_w[d_idx]

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
        self.platform_size = (1.0, 0.6, 0.04)  # 4 layers @ dx
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
        # 1) block
        bx,by,bz = self.create_box(
            (0,0,self.platform_size[2]+0.5*self.block_size[2]),
            self.block_size
        )
        block = get_particle_array_elastic_dynamics(
            name='block', x=bx, y=by, z=bz,
            h=self.hdx*self.dx,
            m=self.dx**3*self.rho0,
            rho=self.rho0,
            constants={'E':self.Eb,'nu':self.nu,
                       'rho_ref':self.rho0,'c0_ref':self.c0}
        )
        # 2) platform & grippers as stiff elastic solids
        def make_solid(name, center, size, E, massf):
            x,y,z = self.create_box(center,size)
            return get_particle_array_elastic_dynamics(
                name=name, x=x, y=y, z=z,
                h=self.hdx*self.dx,
                m=self.dx**3*self.rho0*massf,
                rho=self.rho0,
                constants={'E':E,'nu':self.nu,
                           'rho_ref':self.rho0,'c0_ref':self.c0}
            )

        E_r = 1e9
        mf  = 1e3
        platform = make_solid(
            'platform',
            (0,0,self.platform_size[2]/2),
            self.platform_size,
            E_r, mf
        )
        g1 = make_solid(
            'gripper1',
            (-0.4,0,
             self.platform_size[2]+0.5*self.gripper_size[2]),
            self.gripper_size,
            E_r, mf
        )
        g2 = make_solid(
            'gripper2',
            ( 0.4,0,
             self.platform_size[2]+0.5*self.gripper_size[2]),
            self.gripper_size,
            E_r, mf
        )
        self.gr1, self.gr2 = g1, g2

        return [block, platform, g1, g2]

    def create_scheme(self):
        scheme = ElasticSolidsScheme(
            elastic_solids=['block','platform','gripper1','gripper2'],
            solids=[],
            dim=self.dim,
            artificial_stress_eps=0.5,
            xsph_eps=0.5
        )
        return SchemeChooser(default='elastic', elastic=scheme)

    def configure_scheme(self):
        self.scheme.configure_solver(dt=5e-5, tf=2.0, pfreq=50)

    def create_equations(self):
        eqns = self.scheme.get_equations()
        # gravity on block only
        eqns.append(Group(equations=[
            BodyForce(dest='block', sources=None,
                      fx=0, fy=0, fz=-9.81)
        ], real=False))
        # floor penalty so block never tunnels
        eqns.append(Group(equations=[
            FloorRepulsion(dest='block',
                           floor_z=self.platform_size[2],
                           k=5e5, c=50.0)
        ], real=False))
        return eqns

    def post_step(self, solver):
        dt = solver.dt
        block = self.particles[0]

        # clamp block below floor
        floor = self.platform_size[2]
        mask  = block.z < floor
        if np.any(mask):
            block.z[mask] = floor
            block.w[mask] = np.maximum(block.w[mask], 0.0)

        # set jaw velocities only
        half_b = 0.5*self.block_size[0]
        half_g = 0.5*self.gripper_size[0]
        target = -half_b - half_g + 0.01
        g1, g2 = self.gr1, self.gr2

        if g1.x[0] < target:
            g1.u[:] =  0.02
            g2.u[:] = -0.02
            g1.w[:] = g2.w[:] = 0.0
        else:
            g1.u[:] = g2.u[:] = 0.0
            g1.w[:] = g2.w[:] = 0.3

        # **no** manual g1.x/g2.x updates here—integrator moves them

if __name__ == '__main__':
    app = GraspDeformableBlock()
    app.run()
