import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from typing import List, Dict, Tuple, Optional
from IPython.display import HTML
import time
import logging

# class Agent: pass
class MultiHumanTracker: pass

from leader_follower import LeaderFollowerPlanner
from simulator import CrowdSimulator, Obstacle, Agent, SocialForceModel
class Leader_Follower_SFM:
    """
    Implementation of the Social Force Model.
    """

    # this is the default constructor, so everything is automatically put in
    def __init__(self, v0: float = 1.0, tau: float = 0.2, A: float = 1000.0, B: float = 0.15,
                max_speed: float = 2.0, radius: float = 0.3,
                interaction_range: float = 3.0, time_step: float = 0.25,
                delta_t: float = 0.2, field_of_view_angle_deg: float = 200.0,
                c_perception: float = 0.5, sim=None):
        self.v0 = v0                   
        self.tau = tau                 
        self.A = A                    
        self.B = B                     
        self.radius = radius          
        self.max_speed = max_speed
        self.interaction_range = interaction_range
        self.time_step = time_step

        self.delta_t = delta_t        
        self.fov_cos_angle = np.cos(np.deg2rad(field_of_view_angle_deg) / 2.0)
        self.c_perception = c_perception 
        self.last_predicted_w = np.array([0.0, 0.0])
        self.sim=sim

    def _compute_desired_acceleration(self, position, goal, current_velocity, desired_speed):
        direction_to_goal = np.array(goal) - np.array(position)
        distance = np.linalg.norm(direction_to_goal)

        if distance < 1e-6:
            return np.array([0.0, 0.0])

        e_i = direction_to_goal / distance
        desired_velocity = desired_speed * e_i
        desired_accleration = (desired_velocity - np.array(current_velocity)) / self.tau
        return desired_accleration
    
    def _compute_social_acceleration(self, agent_pos, agent_desired_dir, other_agents: List['Agent']):
        social_acceleration = np.array([0.0, 0.0])
        for beta in other_agents:
            pos_beta = beta.get_position()
            if np.allclose(agent_pos, pos_beta):
                continue
            
            r_alpha_beta = agent_pos - pos_beta
            d_alpha_beta = np.linalg.norm(r_alpha_beta)
            if d_alpha_beta > self.interaction_range:
                continue

            vel_beta = beta.get_velocity()
            v_beta_norm = np.linalg.norm(vel_beta)
            e_beta = vel_beta / v_beta_norm if v_beta_norm > 1e-4 else np.array([0., 0.])
            
            r_prime = r_alpha_beta - v_beta_norm * self.delta_t * e_beta 
            b = 0.5 * np.sqrt(max(0, (d_alpha_beta + np.linalg.norm(r_prime))**2 - (v_beta_norm * self.delta_t)**2))

            force_mag = self.A * np.exp(-b / self.B)
            n_alpha_beta = r_alpha_beta / d_alpha_beta
            f_alpha_beta = force_mag * n_alpha_beta

            cos_phi = np.dot(agent_desired_dir, -n_alpha_beta)
            if cos_phi < self.fov_cos_angle:
                f_alpha_beta *= self.c_perception

            social_acceleration += f_alpha_beta
            
        return social_acceleration
    
    def _compute_obstacle_acceleration(self, position, agent_desired_dir, obstacles: List['Obstacle']):
        """
        Computes repulsive 'forces' from obstacles.
        """
        obstacle_acceleration = np.array([0.0, 0.0])
        for wall in obstacles:
            p1, p2 = wall.p1, wall.p2
            line_vec = p2 - p1
            line_len_sq = np.dot(line_vec, line_vec)
            
            if line_len_sq < 1e-6: continue
            
            t = np.dot(position - p1, line_vec) / line_len_sq
            t = np.clip(t, 0, 1)
            closest_point = p1 + t * line_vec

            d_iB_vec = position - closest_point
            d_iB = np.linalg.norm(d_iB_vec)

            if d_iB > self.interaction_range or d_iB < 1e-6:
                continue
            
            n_iB = d_iB_vec / d_iB
            force_mag = self.A * np.exp(-d_iB / self.B)
            f_iB = force_mag * n_iB
            if np.dot(agent_desired_dir, -n_iB) < self.fov_cos_angle:
                f_iB *= self.c_perception

            obstacle_acceleration += f_iB
        return obstacle_acceleration
    
    def _compute_group_coherence_force(self, agent: Agent, all_agents: List[Agent], groups: List[List[int]]):
        total_force = np.array([0.0, 0.0])
        return total_force * 3.0  
    
    def _compute_group_repulsive_force(self, agent: Agent, all_agents: List[Agent], groups: List[List[int]], threshold=0.55):

        total_force = np.array([0.0, 0.0])
        return total_force * 2
    
    def get_last_predicted_preferred_velocity(self) -> np.ndarray:
        return self.last_predicted_w

    def predict(self, agent: 'Agent', other_agents: List['Agent'], obstacles: List['Obstacle'] = [], groups: List[List[int]] = []) -> Tuple[float, float]:
        # print("THIS IS OTHER AGENTS: ", len(other_agents))
        position = agent.get_position()
        goal = agent.get_goal() #### THIS IS WHERE WE'D INTEGRATE LEADER FOLLOWER
        actual_velocity = agent.get_velocity()
        preferred_velocity = agent.get_preferred_velocity() 
        desired_speed = self.v0 ## this is also where we'd integrate leader follower (??????)

        # call Leader Follower Algorithm -------------------------
        leaderClass = LeaderFollowerPlanner(goal)
        state = [agent.px, agent.py, agent.vx, agent.vy]
        human_step = [] # 2D array which is a list of all humans, each human has an [ID, px, py, vx, vy]; n*5; ID order doesn't matter
        for human in other_agents[1:]:
            human_step.append([
                human.id,
                human.px,
                human.py,
                human.vx,
                human.vy
            ])
        
        gx, gy, ideal_vel = leaderClass.getNewSubGoal(state, human_step)
        print("In LF_SFM, leader is: ", leaderClass.leader)
        self.sim.leader_ID.append(leaderClass.leader)
        print(f"{self.sim.leader_ID=}")
        goal = [gx, gy]
        desired_speed = ideal_vel
        print("goal:", goal)
        # ----------------------------------------------------

        direction_to_goal = goal - position
        dist_to_goal = np.linalg.norm(direction_to_goal)
        agent_desired_dir = direction_to_goal / dist_to_goal if dist_to_goal > 1e-6 else np.array([0.,0.])

        desired_acceleration = self._compute_desired_acceleration(position, goal, actual_velocity, desired_speed)
        social_acceleration = self._compute_social_acceleration(position, agent_desired_dir, other_agents)
        obstacle_acceleration = self._compute_obstacle_acceleration(position, agent_desired_dir, obstacles)

        group_coherence = self._compute_group_coherence_force(agent, other_agents, groups)
        group_repulsive = self._compute_group_repulsive_force(agent, other_agents, groups)
        
        total_acceleration = desired_acceleration + social_acceleration + obstacle_acceleration + group_repulsive +  group_coherence 
        noise_magnitude = 0.01 
        noise = noise_magnitude * self.A * (np.random.rand(2) - 0.5)
        total_acceleration += noise

        new_preferred_velocity = preferred_velocity + total_acceleration * self.time_step
        self.last_predicted_w = new_preferred_velocity

        speed_w = np.linalg.norm(new_preferred_velocity)
        if speed_w > self.max_speed:
            new_actual_velocity = (new_preferred_velocity / speed_w) * self.max_speed
        else:
            new_actual_velocity = new_preferred_velocity

        return new_actual_velocity[0], new_actual_velocity[1]
