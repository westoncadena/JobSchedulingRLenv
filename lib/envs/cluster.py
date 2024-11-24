import gym
from gym import spaces
import numpy as np

class ClusterEnv(gym.Env):
    """
    Custom environment for cluster job scheduling.
    """

    name = "Gridworld"

    def __init__(self, num_nodes, max_jobs, max_resources):
        super(ClusterEnv, self).__init__()

        # Environment parameters
        self.num_nodes = num_nodes         # Number of nodes in the cluster
        self.max_jobs = max_jobs           # Maximum jobs in the queue
        self.max_resources = max_resources # Maximum resources per node (e.g., CPU, Memory)

        # State space: Encodes cluster state and job queue
        self.observation_space = spaces.Dict({
            "node_resources": spaces.Box(low=0, high=self.max_resources, shape=(self.num_nodes, 2), dtype=np.float32),
            "job_queue": spaces.Box(low=0, high=1, shape=(self.max_jobs, 4), dtype=np.float32)
        })

        # Action space: Assign a job to a node, reorder jobs, or delay/reject
        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(3),  # 0: Assign, 1: Reorder, 2: Delay/Reject
            "job_index": spaces.Discrete(self.max_jobs),  # Index of the job in the queue
            "node_index": spaces.Discrete(self.num_nodes),  # Target node for assignment (if applicable)
            "delay_steps": spaces.Discrete(10)  # Number of time steps to delay a job (if applicable)
        })

        # Initialize environment state
        self.reset()

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        # Node resources: Each node starts fully available (CPU, Memory)
        self.node_resources = np.full((self.num_nodes, 2), self.max_resources, dtype=np.float32)

        # Job queue: Randomly generate jobs with resource requirements and priorities
        self.job_queue = np.random.uniform(0, 1, size=(self.max_jobs, 4)).astype(np.float32)
        # Columns: [CPU Requirement, Memory Requirement, Priority, Arrival Time]

        # Other environment variables
        self.time_step = 0
        self.done = False

        return (self._get_obs(), {"prob": 1})

    def _get_obs(self):
        """
        Generate the current observation.
        """
        return {
            "node_resources": self.node_resources,
            "job_queue": self.job_queue
        }

    def step(self, action):
        """
        Apply an action and return the next state, reward, done, and info.
        """
        action_type = action["action_type"]
        job_index = action["job_index"]
        node_index = action.get("node_index", None)
        delay_steps = action.get("delay_steps", 0)

        reward = 0

        if action_type == 0:  # Assign job to a node
            reward = self._assign_job(job_index, node_index)
        elif action_type == 1:  # Reorder the job queue
            reward = self._reorder_jobs(job_index)
        elif action_type == 2:  # Delay or reject a job
            reward = self._delay_or_reject_job(job_index, delay_steps)

        # Advance time step
        self.time_step += 1

        # Check if all jobs are processed or other termination conditions
        self.done = self._check_done()

        truncated = False

        return self._get_obs(), reward, self.done, truncated, {}

    def _assign_job(self, job_index, node_index):
        """
        Assign a job to a node if resources are available.
        """
        job = self.job_queue[job_index]
        cpu_req, mem_req = job[0], job[1]

        if (self.node_resources[node_index, 0] >= cpu_req and 
            self.node_resources[node_index, 1] >= mem_req):
            # Deduct resources
            self.node_resources[node_index, 0] -= cpu_req
            self.node_resources[node_index, 1] -= mem_req
            # Mark job as completed (e.g., set its requirements to zero)
            self.job_queue[job_index, :] = 0
            return 10  # Positive reward for successful assignment
        else:
            return -1  # Negative reward for invalid assignment

    def _reorder_jobs(self, job_index):
        """
        Move the specified job to the front of the queue.
        """
        if job_index < len(self.job_queue):
            job = self.job_queue[job_index]
            self.job_queue = np.delete(self.job_queue, job_index, axis=0)
            self.job_queue = np.vstack(([job], self.job_queue))
            return 1  # Small reward for reordering
        else:
            return -1  # Penalty for invalid job index

    def _delay_or_reject_job(self, job_index, delay_steps):
        """
        Delay or reject a job based on its index and delay time.
        """
        if delay_steps > 0:
            # Implement delay logic (e.g., reschedule job for a future time step)
            return -delay_steps  # Penalty proportional to delay
        else:
            # Reject the job
            self.job_queue[job_index, :] = 0  # Mark job as rejected
            return -5  # Penalty for rejection

    def _check_done(self):
        """
        Check if the environment should terminate.
        """
        # Terminate if all jobs are completed or max time steps reached
        return np.all(self.job_queue[:, 0:2] == 0) or self.time_step >= 100

    def render(self, mode="human"):
        """
        Render the current state (e.g., print the state or visualize it).
        """
        print(f"Time Step: {self.time_step}")
        print("Node Resources:")
        print(self.node_resources)
        print("Job Queue:")
        print(self.job_queue)

