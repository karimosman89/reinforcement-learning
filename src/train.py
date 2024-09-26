import gym
from dqn_agent import DQNAgent

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Training loop
for e in range(1000):
    state = env.reset()
    state = state.reshape(1, state_size)
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = next_state.reshape(1, state_size)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"Episode: {e}/{1000}, score: {time}")
            break
    if len(agent.memory) > 32:
        agent.replay(32)
