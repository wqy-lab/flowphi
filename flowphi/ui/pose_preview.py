"""Document adapter for the shared, headless hand solver."""
import copy
from flowphi.hand_solver import HandMotionSolver, chain_to, idle_chain
from flowphi.pose import PoseSampler


class PosePreview(HandMotionSolver):
    def __init__(self):
        super().__init__()
        self.document = self.chart = None
        self.motion_revision = self.pose_revision = -1
        self.override = None

    def set_document(self, document, chart=None):
        if document is not self.document or chart is not self.chart:
            self.motion_revision = self.pose_revision = -1
        self.document, self.chart = document, chart
        self.ready = document is not None
        self.override = None
        self.pose_revision = -1

    def set_pose(self, pose):
        self._reset_limits()
        self.override = pose
        self.sampler = PoseSampler(pose if pose is not None else self.document.pose if self.document else None)
        self.pose_revision = self.document.pose_revision if self.document else -1

    def refresh(self):
        doc = self.document
        if doc is None:
            return
        if self.override is None and self.pose_revision != doc.pose_revision:
            self._reset_limits()
            self.sampler = PoseSampler(doc.pose)
            self.pose_revision = doc.pose_revision
        if self.motion_revision == doc.motion_revision:
            return
        from flowphi.ui.contact_state import resolve_contacts
        self.warning = ''
        try:
            contacts, _, problems = resolve_contacts(doc, self.chart)
            if problems:
                self.warning = '触控动作有冲突，接触预览以当前分配估算'
        except (ValueError, KeyError, IndexError):
            contacts = [copy.deepcopy(doc.contact(i)) for i in range(len(doc))]
            self.warning = '触控动作有冲突，接触预览以当前分配估算'
        self.set_motion(contacts, doc.baseline['profile'], doc.baseline['physical_screen'],
                        settings=doc.baseline.get('settings'), pose=self.override if self.override is not None else doc.pose,
                        guides=doc.baseline.get('pose_guides', ()), pose_style=doc.baseline.get('pose_style'))
        self.motion_revision = doc.motion_revision
