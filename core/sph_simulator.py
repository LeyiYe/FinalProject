from pysph.solver.solver import Solver
from pysph.sph.scheme import SchemeChooser

class SPHSimulator:
    def __init__(self, config):
        self.config = config
        self.scheme = None
        self.solver = None
        self.particles = {}
        
    def setup_simulation(self, objects):
        """Initialize SPH simulation with given objects"""
        self._create_scheme()
        self._create_solver()
        self._add_objects(objects)
        
    def _create_scheme(self):
        """Create SPH scheme based on config"""
        # Implement your scheme selection logic here
        raise NotImplementedError("You need to implement _create_scheme()")
        
    def _create_solver(self):
        """Create the SPH solver"""
        if self.scheme is None:
            raise RuntimeError("Scheme must be created before solver")
            
        self.solver = Solver(
            dim=3,  # 3D simulation
            integrator=self.config.get('integrator', 'euler'),
            dt=self.config.get('dt', 0.001),
            tf=self.config.get('tf', 1.0),
            adaptive_timestep=self.config.get('adaptive_timestep', False)
        )
        
    def _add_objects(self, objects):
        """Add objects to the simulation"""
        for obj in objects:
            self.particles[obj.name] = obj.get_particle_array()
            
    def step(self, dt):
        """Advance simulation by one time step"""
        if self.solver is None:
            raise RuntimeError("Solver not initialized")
        self.solver.step(dt)
        
    def get_particle_state(self):
        """Return particle positions for visualization"""
        return {name: arr.get('x', 'y', 'z') 
                for name, arr in self.particles.items()}