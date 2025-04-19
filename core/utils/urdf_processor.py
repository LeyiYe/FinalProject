import xml.etree.ElementTree as ET
import numpy as np
from transforms3d.euler import euler2mat

class URDFProcessor:
    def __init__(self, urdf_file):
        self.tree = ET.parse(urdf_file)
        self.root = self.tree.getroot()
        self.links = self._parse_links()
        self.joints = self._parse_joints()
        
    def _parse_links(self):
        links = {}
        for link in self.tree.findall('link'):
            link_name = link.get('name')
            collisions = []
            
            for collision in link.findall('collision'):
                geo = collision.find('geometry')
                origin = collision.find('origin')
                pos = [0,0,0]
                rot = np.eye(3)
                
                if origin is not None:
                    xyz = origin.get('xyz', '0 0 0').split()
                    pos = [float(x) for x in xyz]
                    rpy = origin.get('rpy', '0 0 0').split()
                    rot = euler2mat(*[float(x) for x in rpy])
                
                if geo is not None:
                    geo_type = None
                    geo_params = {}
                    for child in geo:
                        geo_type = child.tag
                        geo_params = child.attrib
                        break
                    
                    if geo_type is not None:
                        if geo_type == 'mesh' and 'filename' in geo_params:
                            geo_params['filename'] = geo_params['filename'].replace(
                                'package://franka_description/', '')
                        
                        collisions.append({
                            'type': geo_type,
                            'params': geo_params,
                            'position': pos,
                            'rotation': rot
                        })
            
            links[link_name] = {'collisions': collisions}
        return links
    
    def _parse_joints(self):
        joints = {}
        for joint in self.tree.findall('joint'):
            joint_name = joint.get('name')
            joint_type = joint.get('type')
            
            origin = joint.find('origin')
            pos = [0, 0, 0]
            rot = np.eye(3)
            
            if origin is not None:
                xyz = origin.get('xyz', '0 0 0').split()
                pos = [float(x) for x in xyz]
                rpy = origin.get('rpy', '0 0 0').split()
                rot = euler2mat(*[float(x) for x in rpy])
            
            axis = joint.find('axis')
            joint_axis = [0, 0, 0]
            if axis is not None:
                joint_axis = [float(x) for x in axis.get('xyz', '0 0 0').split()]
            
            joints[joint_name] = {
                'type': joint_type,
                'parent': joint.find('parent').get('link'),
                'child': joint.find('child').get('link'),
                'position': pos,
                'rotation': rot,
                'axis': joint_axis
            }
        return joints
    
    def get_collision_shapes(self):
        """Returns simplified collision representation"""
        return self.links