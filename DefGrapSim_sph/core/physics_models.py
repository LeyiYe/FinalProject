from pysph.sph.equation import Group
from pysph.sph.solid_mech.basic import (
    ElasticForce,
    HookesDeviatoricForce,
    MomentumEquationWithStress,
    MonaghanArtificialStress
)

def get_elastic_scheme():
    """Returns equations for elastic deformation"""
    equations = [
        Group(equations=[
            # Stress-strain relationship
            ElasticForce(
                dest='sphere', sources=None,
                shear_modulus_from_youngpoisson=True
            ),
            
            # Artificial stress prevents particle clumping
            MonaghanArtificialStress(dest='sphere', sources=None),
            
            # Momentum equation with stress
            MomentumEquationWithStress(dest='sphere', sources=None)
        ])
    ]
    return equations