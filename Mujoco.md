---
layout: page
title: Python
---
Use Hopper as an example

xml file for hopper
```
<worldbody>
  <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
  <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 .125" type="plane" material="MatPlane"/>
  <body name="torso" pos="0 0 1.25">
    <camera name="track" mode="trackcom" pos="0 -3 1" xyaxes="1 0 0 0 0 1"/>
    <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
    <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" ref="1.25" stiffness="0" type="slide"/>
    <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 1.25" stiffness="0" type="hinge"/>
    <geom friction="0.9" fromto="0 0 1.45 0 0 1.05" name="torso_geom" size="0.05" type="capsule"/>
    <body name="thigh" pos="0 0 1.05">
      <joint axis="0 -1 0" name="thigh_joint" pos="0 0 1.05" range="-150 0" type="hinge"/>
      <geom friction="0.9" fromto="0 0 1.05 0 0 0.6" name="thigh_geom" size="0.05" type="capsule"/>
      <body name="leg" pos="0 0 0.35">
        <joint axis="0 -1 0" name="leg_joint" pos="0 0 0.6" range="-150 0" type="hinge"/>
        <geom friction="0.9" fromto="0 0 0.6 0 0 0.1" name="leg_geom" size="0.04" type="capsule"/>
        <body name="foot" pos="0.13/2 0 0.1">
          <joint axis="0 -1 0" name="foot_joint" pos="0 0 0.1" range="-45 45" type="hinge"/>
          <geom friction="2.0" fromto="-0.13 0 0.1 0.26 0 0.1" name="foot_geom" size="0.06" type="capsule"/>
        </body>
      </body>
    </body>
  </body>
</worldbody>
```

gym file
```
class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='hopper.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=1e-3,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_state_range=(-100.0, 100.0),
                 healthy_z_range=(0.7, float('inf')),
                 healthy_angle_range=(-0.2, 0.2),
                 reset_noise_scale=5e-3,
                 exclude_current_positions_from_observation=True):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 4)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.sim.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(
            np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = np.clip(
            self.sim.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
        }

        return observation, reward, done, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
```

How to Query into observation and action space
```
env.observation_space.sample()
env.action_space.sample()

env.sim.data.qpos #print the position of all joints
env.sim.data.qvel #print the velocity of all joints

env.unwrapped.sim.model.get_joint_qpos_addr('joint name specified in xml file')
env.unwrapped.sim.model.get_joint_qvel_addr('joint name specified in xml file')

env.sim.data.get_joint_qpos('joint name')
env.sim.data.get_joint_qvel('joint_name')
```

specifically, if you query into Hopper environment:
```
>>> env.sim.data.qpos
array([ 2.47727731e-04, 1.24890847e+00,  2.80610397e-03,  2.78544215e-04,-4.40898036e-03, -4.39037374e-03])
>>> env.sim.data.qvel
array([-0.00033653, -0.00471339,  0.00331068,  0.004636  , -0.00074496,-0.00011751])
>>> env.unwrapped.sim.model.get_joint_qpos_addr('rootx')
0
>>> env.unwrapped.sim.model.get_joint_qpos_addr('rootz')
1
>>> env.unwrapped.sim.model.get_joint_qpos_addr('rooty')
2
>>> env.unwrapped.sim.model.get_joint_qpos_addr('thigh_joint')
3
>>> env.unwrapped.sim.model.get_joint_qpos_addr('leg_joint')
4
>>> env.unwrapped.sim.model.get_joint_qpos_addr('foot_joint')
5
>>> env.unwrapped.sim.model.get_joint_qvel_addr('rootx')
0
>>> env.unwrapped.sim.model.get_joint_qvel_addr('rootz')
1
>>> env.unwrapped.sim.model.get_joint_qvel_addr('rooty')
2
>>> env.unwrapped.sim.model.get_joint_qvel_addr('thigh_joint')3
>>> env.unwrapped.sim.model.get_joint_qvel_addr('leg_joint')
4
>>> env.unwrapped.sim.model.get_joint_qvel_addr('foot_joint')
5
```

Together with the gym hopper file, we know that the x position is not available for observation. Therefore, there are total of 11 data point in the observation state_vector
```
Observation:

    Num    Observation                                 Min            Max
           x_position(exclude shown up in info instead) Not Limited
    0      rootz                                        Not Limited
    1      rooty                                        Not Limited
    2      thigh joint                                  -150           0
    3      leg joint                                    -150           0
    4      foot joint                                   -45           45    
    5      velocity of rootx                           -10            10
    6      velocity of rootz                           -10            10
    7      velocity of rooty                           -10            10
    8      angular velocity of thigh joint             -10            10
    9      angular velocity of leg joint               -10            10
    10     angular velocity of foot joint              -10            10

Actions:
    0     Thigh Joint Motor                             -1             1
    1     Leg Joint Motor                               -1             1
    2     Foot Joint Motor                              -1             1

    >>> env=gym.make('Hopper-v3')
    >>> env.reset()
    array([ 1.24885056e+00, -3.67318929e-03, -3.10737816e-03,  2.74530568e-04,
           -3.72422016e-03,  3.96886960e-03, -3.75303191e-03, -2.53948964e-03,
           -8.94237721e-04,  1.59178106e-03,  1.33892738e-03])
    >>> env.step([0,0,-1])
    (array([ 1.24862799e+00, -3.79981943e-03, -3.03018520e-03,  2.86991224e-04,
           -9.51065486e-03, -2.37253238e-02, -5.19349612e-02, -2.92790524e-02,
            2.01785114e-02,  1.48786922e-03, -1.44621836e+00]), 0.9891249579219452, False, {'x_position': -0.0010432361970847508, 'x_velocity': -0.009875042078054808})


```
