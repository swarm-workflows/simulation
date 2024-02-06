from agents.common.gpu_action import GpuAction

if __name__ == '__main__':
    action = GpuAction()
    action.perform(batch_size=16)
