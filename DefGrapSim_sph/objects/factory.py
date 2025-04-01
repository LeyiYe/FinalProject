from .sphere import SphSphere
#from .cube import SphCube

class ObjectFactory:
    @staticmethod
    def create_object(obj_type, config):
        """Create SPH object based on type and config"""
        creators = {
            'sphere': SphSphere,
            #'cube': SphCube
        }
        
        if obj_type not in creators:
            raise ValueError(f"Unknown object type: {obj_type}")
            
        return creators[obj_type](config)