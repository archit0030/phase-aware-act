"""Contact-specific diagnostics requiring explicit task geometry and reference."""
import math


def position_telemetry(obs, task):
    """World positions for offline sub-goals; no inferred or default fake poses."""
    def read(key):
        if key is None or key not in obs:return None
        value=[float(v) for v in obs[key]]
        if len(value)!=3 or not all(math.isfinite(v) for v in value):
            raise ValueError('Position telemetry requires finite 3D coordinates')
        return value
    target=task.get('target_position_key')
    placement=task.get('placement_position_key','plate_1_pos')
    return dict(eef_position_m=read(task.get('eef_position_key','robot0_eef_pos')),
                target_object_position_m=read(target),placement_target_position_m=read(placement),
                target_object_position_key=target,placement_position_key=placement)


class ContactDiagnostics:
    def __init__(self,env,task):
        self.env,self.task,self.seen=env,task,False
        self.spec=task.get('contact')
        self.gripper=self.objects=None
        if self.spec:
            self.gripper={env.sim.model.geom_name2id(name) for name in self.spec['gripper_geom_names']}
            self.objects={env.sim.model.geom_name2id(name) for name in self.spec['object_geom_names']}
            if not self.gripper or not self.objects:
                raise ValueError('Contact geometry groups must not be empty')
            if self.spec.get('grasp_reference_site'):
                self.reference_site=env.sim.model.site_name2id(self.spec['grasp_reference_site'])
            else:
                self.reference_site=None

    def observe(self,obs):
        result=dict(contact=None,first_contact=False,contact_eef_reference_error_mm=None)
        if not self.spec:
            return result
        data=self.env.sim.data
        contact=any((int(c.geom1) in self.gripper and int(c.geom2) in self.objects)
                    or (int(c.geom2) in self.gripper and int(c.geom1) in self.objects)
                    for c in (data.contact[index] for index in range(data.ncon)))
        result.update(contact=contact,first_contact=contact and not self.seen)
        self.seen=self.seen or contact
        if contact and self.reference_site is not None:
            eef=obs[self.task.get('eef_position_key','robot0_eef_pos')]
            reference=data.site_xpos[self.reference_site]
            result['contact_eef_reference_error_mm']=1000.*math.sqrt(sum(
                (float(eef[d])-float(reference[d]))**2 for d in range(3)))
        return result
