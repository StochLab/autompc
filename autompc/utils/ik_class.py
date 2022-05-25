# Mujoco Half-Cheetah IK Solver

import collections
import logging
from dm_control import mujoco
import mujoco_py
from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils import inverse_kinematics as ik
import numpy as np
mjlib = mjbindings.mjlib

from typing import List

class HalfCheetahKinematics:
    def __init__(self, path='assets/half_cheetah.xml'):
        self.model_path = path
        self.model = mujoco_py.load_model_from_path(self.model_path)

        self.torso_name = 'torso'
        self.torso_id = self.model.body_name2id(self.torso_name)

        self.bthigh_name = 'bthigh'
        self.bthigh_id = self.model.body_name2id(self.bthigh_name)
        self.bshin_name = 'bshin'
        self.bshin_id = self.model.body_name2id(self.bshin_name)
        self.bfoot_name = 'bfoot'
        self.bfoot_id = self.model.body_name2id(self.bfoot_name)

        self.fthigh_name = 'fthigh'
        self.fthigh_id = self.model.body_name2id(self.fthigh_name)
        self.fshin_name = 'fshin'
        self.fshin_id = self.model.body_name2id(self.fshin_name)
        self.ffoot_name = 'ffoot'
        self.ffoot_id = self.model.body_name2id(self.ffoot_name)

        self.joint_rootx_name = 'rootx'
        self.joint_rootx_id = self.model.joint_name2id(self.joint_rootx_name)
        self.joint_rooty_name = 'rooty'
        self.joint_rooty_id = self.model.joint_name2id(self.joint_rooty_name)
        self.joint_rootz_name = 'rootz'
        self.joint_rootz_id = self.model.joint_name2id(self.joint_rootz_name)

        self.joint_bthigh_name = 'bthigh'
        self.joint_bthigh_id = self.model.joint_name2id(self.joint_bthigh_name)
        self.joint_bshin_name = 'bshin'
        self.joint_bshin_id = self.model.joint_name2id(self.joint_bshin_name)
        self.joint_bfoot_name = 'bfoot'
        self.joint_bfoot_id = self.model.joint_name2id(self.joint_bfoot_name)
        self.bjoints = [self.joint_bthigh_name, self.joint_bshin_name, self.joint_bfoot_name]
        self.bjoints_id = [self.joint_bthigh_id, self.joint_bshin_id, self.joint_bfoot_id]

        self.joint_fthigh_name = 'fthigh'
        self.joint_fthigh_id = self.model.joint_name2id(self.joint_fthigh_name)
        self.joint_fshin_name = 'fshin'
        self.joint_fshin_id = self.model.joint_name2id(self.joint_fshin_name)
        self.joint_ffoot_name = 'ffoot'
        self.joint_ffoot_id = self.model.joint_name2id(self.joint_ffoot_name)
        self.fjoints = [self.joint_fthigh_name, self.joint_fshin_name, self.joint_ffoot_name]
        self.fjoints_id = [self.joint_fthigh_id, self.joint_fshin_id, self.joint_ffoot_id]

    def inverseKinematicsLeg(self, physics: mujoco.Physics, ee_target: List, ee_site: str, joint_names):
        """
        Inverse Kinematics for a single leg
        :param sim: MjSim object
        :param ee_target: End-effector target
        :param ee_site: End-effector site
        :param joint_names: List of joint names
        :return: List of joint angles
        """
        ee_target = np.array(ee_target)
        ee_target = ee_target.reshape(3, 1)
        q_ik = self.inverseKinematics(physics, ee_target, ee_site, joint_names)
        return q_ik

    def inverseKinematics(self, physics: mujoco.Physics, ee_target: List, ee_target_id: str, joint_names: List[str]):
        dtype = physics.data.qpos.dtype

        jac = np.empty((3, physics.model.nv), dtype=dtype)
        err = np.empty(3, dtype=dtype)
        jac_pos = jac
        err_pos = err

        update_nv = np.zeros(physics.model.nv, dtype=dtype)

        physics = physics.copy(share_model=True)

        # Ensure that the Cartesian position of the site is up to date.
        mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)

        # Convert site name to index.
        site_id = physics.model.name2id(ee_target_id, 'site')

        # These are views onto the underlying MuJoCo buffers. mj_fwdPosition will
        # update them in place, so we can avoid indexing overhead in the main loop.
        print(site_id)
        print(physics.data.site_xpos)
        site_xpos = physics.named.data.site_xpos[ee_target_id]
        print(site_xpos)

        # This is an index into the rows of `update` and the columns of `jac`
        # that selects DOFs associated with joints that we are allowed to manipulate.
        # Find the indices of the DOFs belonging to each named joint. Note that
        # these are not necessarily the same as the joint IDs, since a single joint
        # may have >1 DOF (e.g. ball joints).
        indexer = physics.named.model.dof_jntid.axes.row
        # `dof_jntid` is an `(nv,)` array indexed by joint name. We use its row
        # indexer to map each joint name to the indices of its corresponding DOFs.
        dof_indices = indexer.convert_key_item(joint_names)

        steps = 0
        success = False
        tol = 1e-3
        regularization_threshold=0.1,
        regularization_strength=3e-2,
        max_update_norm=2.0,
        progress_thresh=20.0,
        max_steps=100

        for steps in range(max_steps):

            err_norm = 0.0

            # Translational error.
            err_pos[:] = ee_target - site_xpos
            err_norm += np.linalg.norm(err_pos)

            if err_norm < tol:
                logging.debug('Converged after %i steps: err_norm=%3g', steps, err_norm)
                success = True
                break
            else:
                # TODO(b/112141670): Generalize this to other entities besides sites.
                mjlib.mj_jacSite(
                    physics.model.ptr, physics.data.ptr, jac_pos, None, site_id)
                jac_joints = jac[:, dof_indices]

                # TODO(b/112141592): This does not take joint limits into consideration.
                reg_strength = (
                    regularization_strength if err_norm > regularization_threshold
                    else 0.0)
                update_joints = self.nullspace_method(
                    jac_joints, err, regularization_strength=reg_strength)

                update_norm = np.linalg.norm(update_joints)

                # Check whether we are still making enough progress, and halt if not.
                progress_criterion = err_norm / update_norm
                if progress_criterion > progress_thresh:
                    logging.debug('Step %2i: err_norm / update_norm (%3g) > '
                                'tolerance (%3g). Halting due to insufficient progress',
                                steps, progress_criterion, progress_thresh)
                    break

                if update_norm > max_update_norm:
                    update_joints *= max_update_norm / update_norm

                # Write the entries for the specified joints into the full `update_nv`
                # vector.
                update_nv[dof_indices] = update_joints

                # Update `physics.qpos`, taking quaternions into account.
                mjlib.mj_integratePos(physics.model.ptr, physics.data.qpos, update_nv, 1)

                # Compute the new Cartesian position of the site.
                mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)

                logging.debug('Step %2i: err_norm=%-10.3g update_norm=%-10.3g',
                                steps, err_norm, update_norm)

        if not success and steps == max_steps - 1:
            logging.warning('Failed to converge after %i steps: err_norm=%3g',
                            steps, err_norm)

        # Our temporary copy of physics.data is about to go out of scope, and when
        # it does the underlying mjData pointer will be freed and physics.data.qpos
        # will be a view onto a block of deallocated memory. We therefore need to
        # make a copy of physics.data.qpos while physics.data is still alive.
        qpos = physics.data.qpos.copy()

        IKResult = collections.namedtuple(
            'IKResult', ['qpos', 'err_norm', 'steps', 'success'])

        return IKResult(qpos=qpos, err_norm=err_norm, steps=steps, success=success)

    def nullspace_method(self, jac_joints, delta, regularization_strength=0.0):
        """Calculates the joint velocities to achieve a specified end effector delta.
        Args:
            jac_joints: The Jacobian of the end effector with respect to the joints. A
            numpy array of shape `(ndelta, nv)`, where `ndelta` is the size of `delta`
            and `nv` is the number of degrees of freedom.
            delta: The desired end-effector delta. A numpy array of shape `(3,)` or
            `(6,)` containing either position deltas, rotation deltas, or both.
            regularization_strength: (optional) Coefficient of the quadratic penalty
            on joint movements. Default is zero, i.e. no regularization.
        Returns:
            An `(nv,)` numpy array of joint velocities.
        Reference:
            Buss, S. R. S. (2004). Introduction to inverse kinematics with jacobian
            transpose, pseudoinverse and damped least squares methods.
            https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
        """
        hess_approx = jac_joints.T.dot(jac_joints)
        joint_delta = jac_joints.T.dot(delta)
        if regularization_strength > 0:
            # L2 regularization
            hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
            return np.linalg.solve(hess_approx, joint_delta)
        else:
            return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]

if __name__ == '__main__':
    kin = HalfCheetahKinematics()
    sim = mujoco.Physics.from_xml_path(kin.model_path)
    sim.forward()
    sim_copy = sim.copy(share_model=True)
    sim_copy.forward()
    # btarget_pos = np.copy(sim_copy.data.xpos[kin.bfoot_id])

    bJac_ee_p = np.zeros((3, sim_copy.model.nv))
    bJac_ee_r = np.zeros((3, sim_copy.model.nv))
    site_id = sim_copy.model.name2id('bfootsite', 'site')
    mjlib.mj_jacSite(sim_copy.model.ptr, sim_copy.data.ptr, bJac_ee_p, bJac_ee_r, site_id)
    bcontrol_joints_id = kin.bjoints_id
    print(bcontrol_joints_id)
    bJ = np.concatenate((bJac_ee_p, bJac_ee_r), axis=0)[:, bcontrol_joints_id]
    print(bJ.shape)
    print(bJ)

    fJac_ee_p = np.zeros((3, sim_copy.model.nv))
    fJac_ee_r = np.zeros((3, sim_copy.model.nv))
    site_id = sim_copy.model.name2id('ffootsite', 'site')
    mjlib.mj_jacSite(sim_copy.model.ptr, sim_copy.data.ptr, fJac_ee_p, fJac_ee_r, site_id)
    fcontrol_joints_id = kin.fjoints_id
    print(fcontrol_joints_id)
    fJ = np.concatenate((fJac_ee_p, fJac_ee_r), axis=0)[:, fcontrol_joints_id]
    print(fJ.shape)
    print(fJ)

    # print(sim_copy.data.xpos)
    # print(btarget_pos)
    # print(sim_copy.data.qpos)
    # print(kin.inverseKinematicsLeg(sim, btarget_pos, 'bfootsite', kin.bjoints))

    # import gym
    # env = gym.make('HalfCheetah-v2')
    # env.reset()
    # masses = env.model.body_mass
    # print(masses)
    # masses[2:] = 0
    # print(masses)
    # for _ in range(1000):
    #     # env.step(env.action_space.sample())
    #     env.step(np.zeros(env.action_space.sample().shape))
    #     env.render()

